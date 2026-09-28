// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:resource_scope/resource_scope.dart';
import 'package:ndarray/ndarray.dart' as nd;

import 'dtype.dart';
export 'dtype.dart'
    show DTypeTag, DTypeSpec, AnySpec, Boolean, NDArrayBaseElements;
import 'buffer.dart';
import 'device.dart';
import 'exceptions.dart';
import 'slice.dart';
import 'operations/manipulation.dart' as manip;
import 'backend/compute_engine.dart';
import 'backend/kernels.dart';
import 'autograd/autograd.dart';
import 'serialization/webgpu_pipeline.dart';

/// An N-dimensional array living on a GPU device.
///
/// Implements [ScopedResource] for automatic memory management within [ResourceScope.scope].
final class GpuArray<T extends DTypeTag>
    implements ffi.Finalizable, ScopedResource {
  /// The underlying GPU buffer holding tensor data.
  final GpuBuffer buffer;

  /// The dimensions of the tensor.
  final List<int> shape;

  /// The memory stride (in elements) for each dimension.
  final List<int> strides;

  /// The data type of elements in this tensor.
  final DType<T> dtype;

  /// The GPU device hosting this tensor.
  final GpuDevice device;

  /// Offset in elements from the start of [buffer].
  final int offsetElements;

  /// Whether elements are contiguous in C-order in memory.
  final bool isContiguous;

  /// The parent array if this tensor is a view, preventing early garbage collection.
  final GpuArray? _parent;

  /// Whether this tensor tracks gradients for automatic differentiation.
  bool requiresGrad;

  /// Accumulated gradient tensor on device.
  GpuArray? grad;

  /// The backward computation node that produced this tensor.
  GradFn? gradFn;

  bool _isDisposed = false;

  GpuArray._(
    this.buffer, {
    required this.shape,
    required this.strides,
    required this.dtype,
    required this.device,
    this.offsetElements = 0,
    bool? isContiguous,
    GpuArray? parent,
    this.requiresGrad = false,
    this.grad,
    this.gradFn,
  }) : isContiguous = isContiguous ?? ShapeUtils.isContiguous(shape, strides),
       _parent = parent {
    ResourceScope.track(this);
    if (parent != null) {
      buffer.retain();
    }
  }

  /// Creates a [GpuArray] initialized from a flat or nested Dart list of values.
  factory GpuArray.fromList(
    List<dynamic> values,
    List<int> shape,
    DType<T> dtype, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final totalSize = ShapeUtils.computeSize(shape);
    final flatList = _flattenList(values);

    if (flatList.length != totalSize) {
      throw ArgumentError(
        'List length (${flatList.length}) does not match shape total size ($totalSize)',
      );
    }

    final byteSize = totalSize * dtype.byteWidth;
    final gpuBuffer = dev.createBuffer(
      sizeInBytes: byteSize,
      usage:
          GpuBufferUsage.storage |
          GpuBufferUsage.copyDst |
          GpuBufferUsage.copySrc,
    );
    gpuBuffer.detachFromScope();

    final strides = ShapeUtils.computeCStrides(shape);
    final array = GpuArray<T>._(
      gpuBuffer,
      shape: List.unmodifiable(shape),
      strides: List.unmodifiable(strides),
      dtype: dtype,
      device: dev,
      requiresGrad: requiresGrad,
    );

    for (var i = 0; i < totalSize; i++) {
      ComputeEngine.writeAny(gpuBuffer, dtype, i, flatList[i]);
    }

    return array;
  }

  /// Creates an uninitialized [GpuArray] of the specified [shape] and [dtype].
  factory GpuArray.empty(
    List<int> shape,
    DType<T> dtype, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final totalSize = ShapeUtils.computeSize(shape);
    final byteSize = totalSize * dtype.byteWidth;
    final gpuBuffer = dev.createBuffer(
      sizeInBytes: byteSize,
      usage:
          GpuBufferUsage.storage |
          GpuBufferUsage.copyDst |
          GpuBufferUsage.copySrc,
    );
    gpuBuffer.detachFromScope();
    final strides = ShapeUtils.computeCStrides(shape);

    return GpuArray<T>._(
      gpuBuffer,
      shape: List.unmodifiable(shape),
      strides: List.unmodifiable(strides),
      dtype: dtype,
      device: dev,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a [GpuArray] wrapping an existing [buffer] with explicit [shape] and [strides].
  factory GpuArray.fromBuffer({
    required GpuBuffer buffer,
    required List<int> shape,
    required List<int> strides,
    required DType<T> dtype,
    required GpuDevice device,
    int offsetElements = 0,
    bool? isContiguous,
    GpuArray? parent,
    bool requiresGrad = false,
  }) {
    if (parent == null) {
      buffer.retain();
    }
    return GpuArray<T>._(
      buffer,
      shape: List.unmodifiable(shape),
      strides: List.unmodifiable(strides),
      dtype: dtype,
      device: device,
      offsetElements: offsetElements,
      isContiguous: isContiguous,
      parent: parent,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a [GpuArray] of the specified [shape] filled with zeros.
  factory GpuArray.zeros(
    List<int> shape,
    DType<T> dtype, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    final array = GpuArray<T>.empty(
      shape,
      dtype,
      device: device,
      requiresGrad: requiresGrad,
    );
    final totalSize = array.size;
    for (var i = 0; i < totalSize; i++) {
      ComputeEngine.writeValue(array.buffer, dtype, i, 0.0);
    }
    return array;
  }

  /// Creates a [GpuArray] of the specified [shape] filled with ones.
  factory GpuArray.ones(
    List<int> shape,
    DType<T> dtype, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    final Object oneVal = dtype == DType.boolean
        ? true
        : (dtype.isFloating ? 1.0 : 1);
    return GpuArray<T>.filled(
      shape,
      oneVal,
      dtype,
      device: device,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a [GpuArray] of the specified [shape] filled with [value].
  factory GpuArray.filled(
    List<int> shape,
    Object value,
    DType<T> dtype, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    final array = GpuArray<T>.empty(
      shape,
      dtype,
      device: device,
      requiresGrad: requiresGrad,
    );
    final totalSize = array.size;
    for (var i = 0; i < totalSize; i++) {
      ComputeEngine.writeAny(array.buffer, dtype, i, value);
    }
    return array;
  }

  /// Creates a [GpuArray] by copying data from an existing host [NDArray].
  factory GpuArray.fromNDArray(
    nd.NDArray<T> ndarray, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final contiguousND = ndarray.isContiguous ? ndarray : ndarray.copy();
    final byteSize = contiguousND.size * contiguousND.dtype.byteWidth;

    final gpuBuffer = dev.createBufferWithData(
      contiguousND.pointer,
      byteSize,
      GpuBufferUsage.storage | GpuBufferUsage.copyDst | GpuBufferUsage.copySrc,
    );
    gpuBuffer.detachFromScope();

    if (!identical(contiguousND, ndarray)) {
      contiguousND.dispose();
    }

    return GpuArray<T>._(
      gpuBuffer,
      shape: List.unmodifiable(ndarray.shape),
      strides: List.unmodifiable(ShapeUtils.computeCStrides(ndarray.shape)),
      dtype: ndarray.dtype,
      device: dev,
      requiresGrad: requiresGrad,
    );
  }

  /// Total number of elements in this tensor.
  int get size => ShapeUtils.computeSize(shape);

  /// Number of dimensions (rank) of this tensor.
  int get rank => shape.length;

  /// Total size in bytes of the allocated tensor elements.
  int get byteSize => size * dtype.byteWidth;

  /// Whether this tensor is a 2D square matrix.
  bool get isSquare => rank == 2 && shape[0] == shape[1];

  /// Whether this tensor is a leaf node in the autograd computation graph.
  bool get isLeaf => requiresGrad && gradFn == null;

  /// Runs backward automatic differentiation starting from this tensor.
  void backward([GpuArray? gradient, bool retainGraph = false]) =>
      runBackward(this, gradient, retainGraph);

  /// Resets the accumulated gradient on this tensor.
  void zeroGrad() {
    grad = null;
  }

  /// Returns a new tensor sharing the same buffer, detached from the current autograd graph.
  GpuArray<T> detach() => GpuArray<T>._(
    buffer,
    shape: shape,
    strides: strides,
    dtype: dtype,
    device: device,
    offsetElements: offsetElements,
    isContiguous: isContiguous,
    parent: _parent ?? this,
    requiresGrad: false,
  );

  /// Returns the scalar value if this array has exactly one element.
  dynamic get scalar {
    if (size != 1) {
      throw StateError('Cannot retrieve scalar from tensor with size $size');
    }
    final raw = ComputeEngine.readAny(
      buffer,
      dtype,
      0,
      offsetElements: offsetElements,
    );
    if (dtype == DType.boolean) {
      return raw == true || (raw is num && raw != 0);
    }
    if (raw is num && (dtype == DType.float64 || dtype == DType.float32)) {
      return raw.toDouble();
    }
    return raw;
  }

  @override
  bool get isDisposed => _isDisposed;

  // --- Elementwise Arithmetic & Operations ---

  /// Elementwise addition (`this + other`). Supports broadcasting and scalars.
  GpuArray operator +(dynamic other) => add(other);

  /// Elementwise subtraction (`this - other`). Supports broadcasting and scalars.
  GpuArray operator -(dynamic other) => subtract(other);

  /// Elementwise multiplication (`this * other`). Supports broadcasting and scalars.
  GpuArray operator *(dynamic other) => multiply(other);

  /// Elementwise division (`this / other`). Supports broadcasting and scalars.
  GpuArray operator /(dynamic other) => divide(other);

  /// Elementwise negation (`-this`).
  GpuArray<T> operator -() => negate();

  /// Elementwise addition with another [GpuArray] or scalar.
  GpuArray add(dynamic other) => _dispatchBinary(BinaryOp.add, other);

  /// Elementwise subtraction with another [GpuArray] or scalar.
  GpuArray subtract(dynamic other) => _dispatchBinary(BinaryOp.subtract, other);

  /// Elementwise multiplication with another [GpuArray] or scalar.
  GpuArray multiply(dynamic other) => _dispatchBinary(BinaryOp.multiply, other);

  /// Elementwise division with another [GpuArray] or scalar.
  GpuArray divide(dynamic other) => _dispatchBinary(BinaryOp.divide, other);

  /// Elementwise power with another [GpuArray] or scalar.
  GpuArray pow(dynamic other) => _dispatchBinary(BinaryOp.power, other);

  /// Elementwise remainder with another [GpuArray] or scalar.
  GpuArray remainder(dynamic other) =>
      _dispatchBinary(BinaryOp.remainder, other);

  /// Elementwise maximum with another [GpuArray] or scalar.
  GpuArray maximum(dynamic other) => _dispatchBinary(BinaryOp.maximum, other);

  /// Elementwise minimum with another [GpuArray] or scalar.
  GpuArray minimum(dynamic other) => _dispatchBinary(BinaryOp.minimum, other);

  /// Elementwise equality comparison (`==`). Returns a boolean [GpuArray].
  GpuArray<Boolean> equal(dynamic other) =>
      _dispatchComparison(BinaryOp.equal, other);

  /// Elementwise inequality comparison (`!=`). Returns a boolean [GpuArray].
  GpuArray<Boolean> notEqual(dynamic other) =>
      _dispatchComparison(BinaryOp.notEqual, other);

  /// Elementwise greater than comparison (`>`). Returns a boolean [GpuArray].
  GpuArray<Boolean> greater(dynamic other) =>
      _dispatchComparison(BinaryOp.greater, other);

  /// Elementwise greater than or equal comparison (`>=`). Returns a boolean [GpuArray].
  GpuArray<Boolean> greaterEqual(dynamic other) =>
      _dispatchComparison(BinaryOp.greaterEqual, other);

  /// Elementwise less than comparison (`<`). Returns a boolean [GpuArray].
  GpuArray<Boolean> less(dynamic other) =>
      _dispatchComparison(BinaryOp.less, other);

  /// Elementwise less than alias (`<`).
  GpuArray<Boolean> lessThan(dynamic other) => less(other);

  /// Elementwise less than or equal comparison (`<=`). Returns a boolean [GpuArray].
  GpuArray<Boolean> lessEqual(dynamic other) =>
      _dispatchComparison(BinaryOp.lessEqual, other);

  /// Elementwise less than or equal alias (`<=`).
  GpuArray<Boolean> lessThanOrEqual(dynamic other) => lessEqual(other);

  /// Elementwise greater than alias (`>`).
  GpuArray<Boolean> greaterThan(dynamic other) => greater(other);

  /// Elementwise greater than or equal alias (`>=`).
  GpuArray<Boolean> greaterThanOrEqual(dynamic other) => greaterEqual(other);

  // --- Unary Math Operations ---

  /// Computes elementwise negation.
  GpuArray<T> negate() => _dispatchUnary(UnaryOp.negate);

  /// Computes elementwise absolute value.
  GpuArray<T> abs() => _dispatchUnary(UnaryOp.abs);

  /// Computes elementwise square root.
  GpuArray<T> sqrt() => _dispatchUnary(UnaryOp.sqrt);

  /// Computes elementwise exponential ($e^x$).
  GpuArray<T> exp() => _dispatchUnary(UnaryOp.exp);

  /// Computes elementwise natural logarithm ($\ln x$).
  GpuArray<T> log() => _dispatchUnary(UnaryOp.log);

  /// Computes elementwise sine ($\sin x$).
  GpuArray<T> sin() => _dispatchUnary(UnaryOp.sin);

  /// Computes elementwise cosine ($\cos x$).
  GpuArray<T> cos() => _dispatchUnary(UnaryOp.cos);

  /// Computes elementwise tangent ($\tan x$).
  GpuArray<T> tan() => _dispatchUnary(UnaryOp.tan);

  /// Computes elementwise arcsine ($\arcsin x$).
  GpuArray<T> asin() => _dispatchUnary(UnaryOp.asin);

  /// Computes elementwise arccosine ($\arccos x$).
  GpuArray<T> acos() => _dispatchUnary(UnaryOp.acos);

  /// Computes elementwise arctangent ($\arctan x$).
  GpuArray<T> atan() => _dispatchUnary(UnaryOp.atan);

  /// Computes elementwise hyperbolic sine ($\sinh x$).
  GpuArray<T> sinh() => _dispatchUnary(UnaryOp.sinh);

  /// Computes elementwise hyperbolic cosine ($\cosh x$).
  GpuArray<T> cosh() => _dispatchUnary(UnaryOp.cosh);

  /// Computes elementwise hyperbolic tangent ($\tanh x$).
  GpuArray<T> tanh() => _dispatchUnary(UnaryOp.tanh);

  /// Computes elementwise floor.
  GpuArray<T> floor() => _dispatchUnary(UnaryOp.floor);

  /// Computes elementwise ceiling.
  GpuArray<T> ceil() => _dispatchUnary(UnaryOp.ceil);

  /// Computes elementwise round.
  GpuArray<T> round() => _dispatchUnary(UnaryOp.round);

  // --- Reductions ---

  /// Computes the sum of elements over the entire tensor or along [axis].
  GpuArray sum({int? axis, bool keepDims = false}) =>
      _dispatchReduction('sum', axis: axis, keepDims: keepDims);

  /// Computes the arithmetic mean of elements over the entire tensor or along [axis].
  GpuArray mean({int? axis, bool keepDims = false}) =>
      _dispatchReduction('mean', axis: axis, keepDims: keepDims);

  /// Computes the product of elements over the entire tensor or along [axis].
  GpuArray prod({int? axis, bool keepDims = false}) =>
      _dispatchReduction('prod', axis: axis, keepDims: keepDims);

  /// Computes the minimum value over the entire tensor or along [axis].
  GpuArray min({int? axis, bool keepDims = false}) =>
      _dispatchReduction('min', axis: axis, keepDims: keepDims);

  /// Computes the maximum value over the entire tensor or along [axis].
  GpuArray max({int? axis, bool keepDims = false}) =>
      _dispatchReduction('max', axis: axis, keepDims: keepDims);

  // --- Linear Algebra ---

  /// Matrix multiplication of two 2D or batched N-D tensors.
  GpuArray<R> matmul<R extends DTypeTag>(GpuArray other) {
    if (rank < 1 || other.rank < 1) {
      throw GpuShapeMismatchException('matmul', shape, other.shape);
    }

    if (rank == 1 && other.rank == 1) {
      if (shape[0] != other.shape[0]) {
        throw GpuShapeMismatchException('matmul', shape, other.shape);
      }
      final outDtype = _promotedDType(dtype, other.dtype);
      final dst = GpuArray<R>.empty([], outDtype as DType<R>, device: device);
      GpuKernels.executeMatmul(
        srcA: buffer,
        shapeA: shape,
        stridesA: strides,
        offsetA: offsetElements,
        dtypeA: dtype,
        srcB: other.buffer,
        shapeB: other.shape,
        stridesB: other.strides,
        offsetB: other.offsetElements,
        dtypeB: other.dtype,
        dst: dst.buffer,
        outShape: [],
        outStrides: [],
        offsetDst: dst.offsetElements,
        dtypeDst: outDtype,
      );
      if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
        dst.requiresGrad = true;
        dst.gradFn = MatmulBackward(this, other);
      }
      return dst;
    }

    if (rank == 2 && other.rank == 2) {
      if (shape[1] != other.shape[0]) {
        throw GpuShapeMismatchException('matmul', shape, other.shape);
      }
      final outShape = [shape[0], other.shape[1]];
      final outDtype = _promotedDType(dtype, other.dtype);
      final dst = GpuArray<R>.empty(
        outShape,
        outDtype as DType<R>,
        device: device,
      );
      GpuKernels.executeMatmul(
        srcA: buffer,
        shapeA: shape,
        stridesA: strides,
        offsetA: offsetElements,
        dtypeA: dtype,
        srcB: other.buffer,
        shapeB: other.shape,
        stridesB: other.strides,
        offsetB: other.offsetElements,
        dtypeB: other.dtype,
        dst: dst.buffer,
        outShape: outShape,
        outStrides: dst.strides,
        offsetDst: dst.offsetElements,
        dtypeDst: outDtype,
      );
      if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
        dst.requiresGrad = true;
        dst.gradFn = MatmulBackward(this, other);
      }
      return dst;
    }

    // Batched N-D matmul
    final m = shape[rank - 2];
    final k1 = shape[rank - 1];
    final k2 = other.shape[other.rank - 2];
    final n = other.shape[other.rank - 1];

    if (k1 != k2) {
      throw GpuShapeMismatchException('matmul', shape, other.shape);
    }

    final batchA = shape.sublist(0, rank - 2);
    final batchB = other.shape.sublist(0, other.rank - 2);
    final batchOut = ShapeUtils.broadcastShapes(batchA, batchB);
    final outShape = [...batchOut, m, n];

    final outDtype = _promotedDType(dtype, other.dtype);
    final dst = GpuArray<R>.empty(
      outShape,
      outDtype as DType<R>,
      device: device,
    );
    GpuKernels.executeMatmul(
      srcA: buffer,
      shapeA: shape,
      stridesA: strides,
      offsetA: offsetElements,
      dtypeA: dtype,
      srcB: other.buffer,
      shapeB: other.shape,
      stridesB: other.strides,
      offsetB: other.offsetElements,
      dtypeB: other.dtype,
      dst: dst.buffer,
      outShape: outShape,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      dtypeDst: outDtype,
    );
    if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
      dst.requiresGrad = true;
      dst.gradFn = MatmulBackward(this, other);
    }
    return dst;
  }

  /// Dot product or matrix multiplication.
  GpuArray<R> dot<R extends DTypeTag>(GpuArray other) => matmul<R>(other);

  // --- Tensor Views & Transformations ---

  /// Returns a new view of this tensor with [newShape].
  GpuArray<T> reshape(List<int> newShape) {
    final totalSize = ShapeUtils.computeSize(newShape);
    if (totalSize != size) {
      throw ArgumentError(
        'Cannot reshape tensor of size $size to shape $newShape (size $totalSize)',
      );
    }

    GpuArray<T> res;
    if (isContiguous) {
      res = GpuArray<T>._(
        buffer,
        shape: List.unmodifiable(newShape),
        strides: List.unmodifiable(ShapeUtils.computeCStrides(newShape)),
        dtype: dtype,
        device: device,
        offsetElements: offsetElements,
        parent: this,
      );
    } else {
      final contiguousCopy = copy();
      res = contiguousCopy.reshape(newShape);
    }

    if (isGradEnabled && requiresGrad) {
      res.requiresGrad = true;
      res.gradFn = ReshapeBackward(this, shape);
    }

    return res;
  }

  /// Permutes the axes of this tensor.
  GpuArray<T> transpose([List<int>? axes]) {
    final perm = axes ?? List.generate(rank, (i) => rank - 1 - i);
    if (perm.length != rank) {
      throw ArgumentError(
        'Permutation axes length must match tensor rank ($rank)',
      );
    }

    final newShape = List<int>.generate(rank, (i) => shape[perm[i]]);
    final newStrides = List<int>.generate(rank, (i) => strides[perm[i]]);

    final res = GpuArray<T>._(
      buffer,
      shape: List.unmodifiable(newShape),
      strides: List.unmodifiable(newStrides),
      dtype: dtype,
      device: device,
      offsetElements: offsetElements,
      parent: this,
    );

    if (isGradEnabled && requiresGrad) {
      res.requiresGrad = true;
      res.gradFn = TransposeBackward(this, perm);
    }

    return res;
  }

  /// Returns a 1D flattened view or contiguous copy of this tensor.
  GpuArray<T> flatten() => reshape([size]);

  /// Removes dimensions of size 1 at [axis], or all size-1 dimensions if [axis] is omitted.
  GpuArray<T> squeeze({int? axis}) {
    final newShape = <int>[];
    if (axis != null) {
      final normAxis = axis < 0 ? axis + rank : axis;
      for (var i = 0; i < rank; i++) {
        if (i == normAxis) {
          if (shape[i] != 1) {
            throw ArgumentError(
              'Cannot squeeze axis $axis with size ${shape[i]}',
            );
          }
        } else {
          newShape.add(shape[i]);
        }
      }
    } else {
      for (final dim in shape) {
        if (dim != 1) newShape.add(dim);
      }
      if (newShape.isEmpty && shape.isNotEmpty) {
        newShape.add(1);
      }
    }
    final res = reshape(newShape);
    if (isGradEnabled && requiresGrad) {
      res.requiresGrad = true;
      res.gradFn = SqueezeBackward(this, shape);
    }
    return res;
  }

  /// Inserts a new dimension of size 1 at position [axis].
  GpuArray<T> unsqueeze(int axis) {
    final normAxis = axis < 0 ? axis + rank + 1 : axis;
    if (normAxis < 0 || normAxis > rank) {
      throw RangeError.range(normAxis, 0, rank, 'axis');
    }
    final newShape = List<int>.from(shape)..insert(normAxis, 1);
    final res = reshape(newShape);
    if (isGradEnabled && requiresGrad) {
      res.requiresGrad = true;
      res.gradFn = UnsqueezeBackward(this, shape);
    }
    return res;
  }

  /// Creates a new contiguous copy of this tensor in device memory.
  GpuArray<T> copy() {
    final dst = GpuArray<T>.empty(shape, dtype, device: device);
    GpuKernels.copyStrided(
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtypeSrc: dtype,
      dst: dst.buffer,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      dtypeDst: dtype,
    );
    return dst;
  }

  /// Casts this tensor to a different [targetDType].
  GpuArray<R> astype<R extends DTypeTag>(DType<R> targetDType) {
    if (dtype == targetDType) return this as GpuArray<R>;
    final dst = GpuArray<R>.empty(shape, targetDType, device: device);
    GpuKernels.copyStrided(
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtypeSrc: dtype,
      dst: dst.buffer,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      dtypeDst: targetDType,
    );
    return dst;
  }

  /// Returns a strided subview of this array according to [specs].
  GpuArray<T> slice(List<dynamic> specs) {
    final view = computeSliceView(
      shape: shape,
      strides: strides,
      offsetElements: offsetElements,
      specs: specs,
    );

    final res = GpuArray<T>._(
      buffer,
      shape: view.shape,
      strides: view.strides,
      dtype: dtype,
      device: device,
      offsetElements: view.offsetElements,
      parent: this,
    );

    if (isGradEnabled && requiresGrad) {
      res.requiresGrad = true;
      res.gradFn = SliceBackward(this, specs);
    }

    return res;
  }

  /// Slices this array using [index] (can be a [Slice], [Index], [All], [NewAxis], [Ellipsis], or a [List] of them).
  dynamic operator [](dynamic index) {
    if (index is List) {
      return slice(index);
    }
    return slice([index]);
  }

  /// Interchange two axes of this array.
  GpuArray<T> swapaxes(int axis1, int axis2) =>
      manip.swapaxes(this, axis1, axis2);

  /// Move axes of this array to new positions.
  GpuArray<T> moveaxis(dynamic source, dynamic destination) =>
      manip.moveaxis(this, source, destination);

  /// Repeats elements of this array [repeats] times along [axis].
  GpuArray<T> repeat(int repeats, {int? axis}) =>
      manip.repeat(this, repeats, axis: axis);

  /// Extracts diagonal or constructs diagonal array.
  GpuArray<T> diag({int k = 0}) => manip.diag(this, k: k);

  /// Returns specified diagonals of this array.
  GpuArray<T> diagonal({int offset = 0, int axis1 = 0, int axis2 = 1}) =>
      manip.diagonal(this, offset: offset, axis1: axis1, axis2: axis2);

  /// Returns the sum along diagonals of this array.
  dynamic trace({int offset = 0, int axis1 = 0, int axis2 = 1}) =>
      manip.trace(this, offset: offset, axis1: axis1, axis2: axis2);

  /// Returns upper triangular portion of this array.
  GpuArray<T> triu({int k = 0}) => manip.triu(this, k: k);

  /// Returns lower triangular portion of this array.
  GpuArray<T> tril({int k = 0}) => manip.tril(this, k: k);

  /// Reverses the order of elements along the given [axis].
  GpuArray<T> flip({dynamic axis}) => manip.flip(this, axis: axis);

  /// Roll array elements along a given [axis].
  GpuArray<T> roll(dynamic shift, {dynamic axis}) =>
      manip.roll(this, shift, axis: axis);

  /// Rotates an array by 90 degrees in the plane specified by [axes].
  GpuArray<T> rot90({int k = 1, List<int> axes = const [0, 1]}) =>
      manip.rot90(this, k: k, axes: axes);

  /// Pads this array with [padWidth].
  GpuArray<T> pad(
    List<List<int>> padWidth, {
    String mode = 'constant',
    dynamic constantValues = 0,
  }) => manip.pad(this, padWidth, mode: mode, constantValues: constantValues);

  /// Promotes two [DType]s following NumPy's type promotion hierarchy.
  static DType promoteDTypes(DType a, DType b) => _promotedDType(a, b);

  // --- Conversions & Host Interop ---

  /// Downloads this GPU tensor into host memory as a standard [NDArray].
  nd.NDArray<T> toNDArray() {
    final contiguousArray = isContiguous ? this : copy();
    final ndarray = nd.NDArray<T>.create(shape, dtype);

    contiguousArray.buffer.copyToHost(
      ndarray.pointer,
      byteSize,
      offset: contiguousArray.offsetElements * dtype.byteWidth,
    );

    if (!identical(contiguousArray, this)) {
      contiguousArray.dispose();
    }

    return ndarray;
  }

  /// Returns a flat Dart list containing a copy of the elements in this tensor.
  List<dynamic> toList() {
    final hostND = toNDArray();
    final list = hostND.toList();
    hostND.dispose();
    return list;
  }

  /// Returns the scalar value of a 0D or 1-element tensor.
  dynamic item() {
    if (size != 1) {
      throw ArgumentError(
        'item() only works on tensors with exactly 1 element (has $size).',
      );
    }
    return ComputeEngine.readAny(
      buffer,
      dtype,
      0,
      offsetElements: offsetElements,
    );
  }

  /// Returns a nested Dart list representation matching the tensor's multidimensional shape.
  List<dynamic> toNestedList() {
    final flat = toList();
    if (rank <= 1) return flat;

    dynamic build(int dim, int offset) {
      if (dim == rank - 1) {
        return flat.sublist(offset, offset + shape[dim]);
      }
      var subStride = 1;
      for (var d = dim + 1; d < rank; d++) {
        subStride *= shape[d];
      }
      final list = <dynamic>[];
      for (var i = 0; i < shape[dim]; i++) {
        list.add(build(dim + 1, offset + i * subStride));
      }
      return list;
    }

    return build(0, 0) as List<dynamic>;
  }

  @override
  String toString() {
    if (_isDisposed) {
      return 'GpuArray(disposed)';
    }
    return 'GpuArray<$T>(shape: $shape, dtype: ${dtype.name}, device: ${device.name})';
  }

  // --- Internal Helpers ---

  GpuArray _dispatchBinary(BinaryOp op, dynamic other) {
    if (other is GpuArray) {
      final outShape = ShapeUtils.broadcastShapes(shape, other.shape);
      final outDtype = _promotedDType(dtype, other.dtype);
      final dst = GpuArray.empty(outShape, outDtype, device: device);

      GpuKernels.executeBinaryOp(
        op: op,
        srcA: buffer,
        shapeA: shape,
        stridesA: strides,
        offsetA: offsetElements,
        dtypeA: dtype,
        srcB: other.buffer,
        shapeB: other.shape,
        stridesB: other.strides,
        offsetB: other.offsetElements,
        dtypeB: other.dtype,
        dst: dst.buffer,
        outShape: outShape,
        outStrides: dst.strides,
        offsetDst: dst.offsetElements,
        dtypeDst: outDtype,
      );

      if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
        dst.requiresGrad = true;
        switch (op) {
          case BinaryOp.add:
            dst.gradFn = AddBackward(this, other);
            break;
          case BinaryOp.subtract:
            dst.gradFn = SubBackward(this, other);
            break;
          case BinaryOp.multiply:
            dst.gradFn = MulBackward(this, other);
            break;
          case BinaryOp.divide:
            dst.gradFn = DivBackward(this, other);
            break;
          case BinaryOp.power:
            dst.gradFn = PowBackward(this, other);
            break;
          default:
            break;
        }
      }

      return dst;
    } else if (other is num || other is bool) {
      final scalarArray = GpuArray.filled(
        [],
        other as Object,
        dtype,
        device: device,
      );
      final res = _dispatchBinary(op, scalarArray);
      if (!res.requiresGrad) {
        scalarArray.dispose();
      }
      return res;
    } else {
      throw ArgumentError(
        'Unsupported operand type for binary operation: ${other.runtimeType}',
      );
    }
  }

  GpuArray<Boolean> _dispatchComparison(BinaryOp op, dynamic other) {
    if (other is GpuArray) {
      final outShape = ShapeUtils.broadcastShapes(shape, other.shape);
      final dst = GpuArray<Boolean>.empty(
        outShape,
        DType.boolean,
        device: device,
      );

      GpuKernels.executeBinaryOp(
        op: op,
        srcA: buffer,
        shapeA: shape,
        stridesA: strides,
        offsetA: offsetElements,
        dtypeA: dtype,
        srcB: other.buffer,
        shapeB: other.shape,
        stridesB: other.strides,
        offsetB: other.offsetElements,
        dtypeB: other.dtype,
        dst: dst.buffer,
        outShape: outShape,
        outStrides: dst.strides,
        offsetDst: dst.offsetElements,
        dtypeDst: DType.boolean,
      );

      return dst;
    } else if (other is num || other is bool) {
      final scalarArray = GpuArray.filled(
        [],
        other as Object,
        dtype,
        device: device,
      );
      final res = _dispatchComparison(op, scalarArray);
      scalarArray.dispose();
      return res;
    } else {
      throw ArgumentError(
        'Unsupported operand type for comparison: ${other.runtimeType}',
      );
    }
  }

  GpuArray<T> _dispatchUnary(UnaryOp op) {
    final dst = GpuArray<T>.empty(shape, dtype, device: device);
    GpuKernels.executeUnaryOp(
      op: op,
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtypeSrc: dtype,
      dst: dst.buffer,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      dtypeDst: dtype,
    );

    if (isGradEnabled && requiresGrad) {
      dst.requiresGrad = true;
      switch (op) {
        case UnaryOp.negate:
          dst.gradFn = NegBackward(this);
          break;
        case UnaryOp.sqrt:
          dst.gradFn = SqrtBackward(this, dst);
          break;
        case UnaryOp.exp:
          dst.gradFn = ExpBackward(this, dst);
          break;
        case UnaryOp.log:
          dst.gradFn = LogBackward(this);
          break;
        case UnaryOp.tanh:
          dst.gradFn = TanhBackward(this, dst);
          break;
        default:
          break;
      }
    }

    return dst;
  }

  GpuArray _dispatchReduction(String op, {int? axis, bool keepDims = false}) {
    List<int> outShape;
    if (axis == null) {
      outShape = keepDims ? List.filled(rank, 1) : [];
    } else {
      final normAxis = axis < 0 ? axis + rank : axis;
      if (normAxis < 0 || normAxis >= rank) {
        throw RangeError.range(normAxis, 0, rank - 1, 'axis');
      }
      outShape = [];
      for (var i = 0; i < rank; i++) {
        if (i == normAxis) {
          if (keepDims) outShape.add(1);
        } else {
          outShape.add(shape[i]);
        }
      }
    }

    final isComplex = dtype == DType.complex64 || dtype == DType.complex128;
    final GpuArray dst = (op == 'mean' && !isComplex)
        ? GpuArray<Float64>.empty(outShape, DType.float64, device: device)
        : GpuArray<T>.empty(outShape, dtype, device: device);
    final outDtype = (op == 'mean' && !isComplex) ? DType.float64 : dtype;

    GpuKernels.executeReduction(
      op: op,
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtypeSrc: dtype,
      dst: dst.buffer,
      outShape: outShape,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      dtypeDst: outDtype,
      axis: axis,
    );

    if (isGradEnabled && requiresGrad) {
      dst.requiresGrad = true;
      if (op == 'sum') {
        dst.gradFn = SumBackward(this, axis: axis, keepDims: keepDims);
      } else if (op == 'mean') {
        dst.gradFn = MeanBackward(this, axis: axis, keepDims: keepDims);
      }
    }

    return dst;
  }

  static DType _promotedDType(DType a, DType b) {
    if (a == DType.boolean && b == DType.boolean) return DType.boolean;
    if (a == b) return a;
    if (a == DType.boolean) return b;
    if (b == DType.boolean) return a;

    // Complex promotion
    if (a == DType.complex128 || b == DType.complex128) return DType.complex128;
    if (a == DType.complex64 || b == DType.complex64) {
      final other = (a == DType.complex64) ? b : a;
      if (other == DType.float64 ||
          other == DType.int64 ||
          other == DType.uint64 ||
          other == DType.complex128) {
        return DType.complex128;
      }
      return DType.complex64;
    }

    // Floating point promotion
    if (a == DType.float64 || b == DType.float64) return DType.float64;
    if (a == DType.float32 || b == DType.float32) {
      final other = (a == DType.float32) ? b : a;
      if (other == DType.int64 || other == DType.uint64) return DType.float64;
      return DType.float32;
    }
    if ((a == DType.float16 && b == DType.bfloat16) ||
        (a == DType.bfloat16 && b == DType.float16)) {
      return DType.float32;
    }
    if (a == DType.float16 || b == DType.float16) {
      final other = (a == DType.float16) ? b : a;
      if (other == DType.int64 || other == DType.uint64) return DType.float64;
      if (other == DType.int32 ||
          other == DType.uint32 ||
          other == DType.int16 ||
          other == DType.uint16) {
        return DType.float32;
      }
      return DType.float16;
    }
    if (a == DType.bfloat16 || b == DType.bfloat16) {
      final other = (a == DType.bfloat16) ? b : a;
      if (other == DType.int64 || other == DType.uint64) return DType.float64;
      if (other == DType.int32 ||
          other == DType.uint32 ||
          other == DType.int16 ||
          other == DType.uint16) {
        return DType.float32;
      }
      return DType.bfloat16;
    }

    // Integer promotions
    final isASigned =
        a == DType.int64 ||
        a == DType.int32 ||
        a == DType.int16 ||
        a == DType.int8;
    final isBSigned =
        b == DType.int64 ||
        b == DType.int32 ||
        b == DType.int16 ||
        b == DType.int8;

    if (isASigned && isBSigned) {
      final maxBytes = math.max(a.byteWidth, b.byteWidth);
      if (maxBytes >= 8) return DType.int64;
      if (maxBytes >= 4) return DType.int32;
      if (maxBytes >= 2) return DType.int16;
      return DType.int8;
    }

    if (!isASigned && !isBSigned) {
      final maxBytes = math.max(a.byteWidth, b.byteWidth);
      if (maxBytes >= 8) return DType.uint64;
      if (maxBytes >= 4) return DType.uint32;
      if (maxBytes >= 2) return DType.uint16;
      return DType.uint8;
    }

    // Mixed signed and unsigned
    final signed = isASigned ? a : b;
    final unsigned = isASigned ? b : a;

    if (signed.byteWidth > unsigned.byteWidth) {
      return signed;
    }
    if (unsigned.byteWidth == 1) return DType.int16;
    if (unsigned.byteWidth == 2) return DType.int32;
    if (unsigned.byteWidth == 4) return DType.int64;
    return DType.float64;
  }

  static List<dynamic> _flattenList(List<dynamic> list) {
    final result = <dynamic>[];
    for (final item in list) {
      if (item is List) {
        result.addAll(_flattenList(item));
      } else {
        result.add(item);
      }
    }
    return result;
  }

  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    ResourceScope.untrack(this);
    buffer.release();
  }

  @override
  ScopedResource detachFromScope() {
    ResourceScope.untrack(this);
    buffer.detachFromScope();
    return this;
  }

  @override
  ScopedResource detachToParentScope() {
    ResourceScope.promoteToParent(this);
    buffer.detachToParentScope();
    return this;
  }

  /// Packages this [GpuArray]'s underlying compute pipeline and data into an interactive client-side [WebGpuWidget].
  WebGpuWidget toWebGpuWidget({
    String? title,
    bool renderToCanvas = false,
    List<dynamic> sliders = const [],
  }) {
    final parsedSliders = sliders.map((e) => e as WebGpuSlider).toList();
    final rawND = toNDArray();
    final rawList = rawND.toList();
    final f32List = Float32List.fromList(
      rawList.map((e) => (e as num).toDouble()).toList(),
    );
    final base64Payload = base64Encode(f32List.buffer.asUint8List());
    rawND.dispose();

    const wgsl = '''
struct Uniforms {
  total_elements: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  var idx = global_id.x;
  let stride = num_workgroups.x * 256u;
  while (idx < uniforms.total_elements) {
    dst[idx] = src[idx];
    idx += stride;
  }
}
''';

    final pkg = GpuComputePipelinePackage(
      name: title ?? 'GpuArray_${dtype.name}',
      wgslCode: wgsl,
      inputs: [
        GpuBufferPayload(
          bindingIndex: 0,
          name: 'src',
          dtype: dtype,
          shape: shape,
          base64Data: base64Payload,
          sizeInBytes: buffer.sizeInBytes,
        ),
      ],
      output: GpuBufferPayload(
        bindingIndex: 1,
        name: 'dst',
        dtype: dtype,
        shape: shape,
        isOutput: true,
        sizeInBytes: buffer.sizeInBytes,
      ),
      uniforms: [ShapeUtils.computeSize(shape), 0, 0, 0],
      sliders: parsedSliders,
      renderToCanvas: renderToCanvas,
    );

    return WebGpuWidget(pkg, title: title ?? 'GpuArray Interactive Inspector');
  }
}
