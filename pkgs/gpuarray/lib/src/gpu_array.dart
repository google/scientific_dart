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

import 'dart:ffi' as ffi;
import 'dart:math' as math;

import 'package:ndarray/ndarray.dart' as nd;
import 'package:resource_scope/resource_scope.dart';

import 'autograd/autograd.dart';
import 'backend/compute_engine.dart';
import 'backend/kernels.dart';
import 'buffer.dart';
import 'device.dart';
import 'dtype.dart';
import 'exceptions.dart';
import 'operations/manipulation.dart' as manip;
import 'slice.dart';

export 'package:resource_scope/resource_scope.dart'
    show ResourceScope, ScopedResource;

export 'dtype.dart'
    show AnySpec, Bool, Boolean, DTypeSpec, DTypeTag, NDArrayBaseElements;

/// An N-dimensional array living on a GPU device.
///
/// Implements [ScopedResource] for automatic memory management within
/// [ResourceScope.scope].
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

  /// The parent array if this tensor is a view, preventing early garbage
  /// collection.
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
  }) : isContiguous = isContiguous ?? isContiguousLayout(shape, strides),
       _parent = parent {
    ResourceScope.track(this);
    if (parent != null) {
      buffer.retain();
    }
  }

  static GpuArray<T> _create<T extends DTypeTag>(
    GpuBuffer buffer, {
    required List<int> shape,
    required List<int> strides,
    required DType<T> dtype,
    required GpuDevice device,
    int offsetElements = 0,
    bool? isContiguous,
    GpuArray? parent,
    bool requiresGrad = false,
    GpuArray? grad,
    GradFn? gradFn,
  }) {
    final GpuArray<DTypeTag> instance = switch (dtype) {
      DType.float64 => GpuArray<Float64>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.float64,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.float32 => GpuArray<Float32>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.float32,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.float16 => GpuArray<Float16>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.float16,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.bfloat16 => GpuArray<BFloat16>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.bfloat16,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.int64 => GpuArray<Int64>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.int64,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.int32 => GpuArray<Int32>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.int32,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.int16 => GpuArray<Int16>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.int16,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.int8 => GpuArray<Int8>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.int8,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.uint64 => GpuArray<Uint64>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.uint64,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.uint32 => GpuArray<Uint32>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.uint32,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.uint16 => GpuArray<Uint16>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.uint16,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.uint8 => GpuArray<Uint8>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.uint8,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.boolean => GpuArray<Boolean>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.boolean,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.complex64 => GpuArray<Complex64>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.complex64,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
      DType.complex128 => GpuArray<Complex128>._(
        buffer,
        shape: shape,
        strides: strides,
        dtype: DType.complex128,
        device: device,
        offsetElements: offsetElements,
        isContiguous: isContiguous,
        parent: parent,
        requiresGrad: requiresGrad,
        grad: grad,
        gradFn: gradFn,
      ),
    };
    return instance as GpuArray<T>;
  }

  /// Creates a [GpuArray] initialized from a flat or nested Dart list of
  /// [values].
  factory GpuArray.fromList(
    List<dynamic> values,
    List<int> shape,
    DType<T> dtype, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final totalSize = computeSize(shape);
    final flatList = _flattenList(values);

    if (flatList.length != totalSize) {
      throw ArgumentError.value(
        values.length,
        'values',
        'Must have flattened length ($totalSize) matching shape $shape.',
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

    final strides = computeCStrides(shape);
    final array = GpuArray._create<T>(
      gpuBuffer,
      shape: List.unmodifiable(shape),
      strides: List.unmodifiable(strides),
      dtype: dtype,
      device: dev,
      requiresGrad: requiresGrad,
    );

    for (var i = 0; i < totalSize; i++) {
      writeBufferAny(gpuBuffer, dtype, i, flatList[i]);
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
    for (final dim in shape) {
      if (dim < 0) {
        throw ArgumentError.value(
          shape,
          'shape',
          'Must not contain negative dimensions.',
        );
      }
    }
    final dev = device ?? GpuDevice.defaultDevice;
    final totalSize = computeSize(shape);
    final byteSize = totalSize * dtype.byteWidth;
    final gpuBuffer = dev.createBuffer(
      sizeInBytes: byteSize,
      usage:
          GpuBufferUsage.storage |
          GpuBufferUsage.copyDst |
          GpuBufferUsage.copySrc,
    );
    gpuBuffer.detachFromScope();
    final strides = computeCStrides(shape);

    return GpuArray._create<T>(
      gpuBuffer,
      shape: List.unmodifiable(shape),
      strides: List.unmodifiable(strides),
      dtype: dtype,
      device: dev,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a [GpuArray] wrapping an existing [buffer] with explicit [shape]
  /// and [strides].
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
    if (buffer.isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot create GpuArray from a disposed GpuBuffer.',
      );
    }
    if (parent == null) {
      buffer.retain();
    }
    return GpuArray._create<T>(
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
      writeBufferValue(array.buffer, dtype, i, 0.0);
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
      writeBufferAny(array.buffer, dtype, i, value);
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

    return GpuArray._create<T>(
      gpuBuffer,
      shape: List.unmodifiable(ndarray.shape),
      strides: List.unmodifiable(computeCStrides(ndarray.shape)),
      dtype: ndarray.dtype,
      device: dev,
      requiresGrad: requiresGrad,
    );
  }

  void _checkNotDisposed() {
    if (_isDisposed || buffer.isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot operate on a disposed GpuArray.',
      );
    }
  }

  /// Executes [callback] with a zero-copy, unmanaged [nd.NDArray] view backed
  /// directly by this array's synchronized host mirror pointer.
  ///
  /// The [shape] must not contain zero-length dimensions (`!shape.contains(0)`).
  R withTemporaryNDArrayView<R>(R Function(nd.NDArray<T> view) callback) {
    _checkNotDisposed();
    if (shape.contains(0)) {
      throw StateError(
        'Cannot create an NDArray.fromPointer view for an empty shape $shape.',
      );
    }
    buffer.ensureHostSynced();

    var minRelativeOffset = 0;
    for (var d = 0; d < shape.length; d++) {
      if (strides[d] < 0 && shape[d] > 0) {
        minRelativeOffset += (shape[d] - 1) * strides[d];
      }
    }
    final byteOffset = (offsetElements + minRelativeOffset) * dtype.byteWidth;
    final dataPtr = (buffer.address + byteOffset).cast<ffi.Void>();

    final view = nd.NDArray.unmanaged(
      () => nd.NDArray<T>.fromPointer(dataPtr, shape, dtype, strides: strides),
    );
    return callback(view);
  }

  /// Total number of elements in this tensor.
  int get size => computeSize(shape);

  /// Number of dimensions (rank) of this tensor.
  int get rank => shape.length;

  /// Number of dimensions (rank) of this tensor (alias for [rank]).
  int get ndim => shape.length;

  /// Total size in bytes of the allocated tensor elements.
  int get byteSize => size * dtype.byteWidth;

  /// Whether this tensor is a 2D square matrix.
  bool get isSquare => rank == 2 && shape[0] == shape[1];

  /// Whether this tensor is a leaf node in the autograd computation graph.
  bool get isLeaf => requiresGrad && gradFn == null;

  /// Runs backward automatic differentiation starting from this tensor.
  void backward({GpuArray? gradient, bool retainGraph = false}) {
    _checkNotDisposed();
    runBackward(this, gradient: gradient, retainGraph: retainGraph);
  }

  /// Resets the accumulated gradient on this tensor.
  void zeroGrad() {
    grad = null;
  }

  /// Creates a new tensor view sharing the same buffer, detached from the
  /// current autograd graph.
  GpuArray<T> detach() {
    _checkNotDisposed();
    return GpuArray._create<T>(
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
  }

  /// The scalar value of this array when it has exactly one element.
  ///
  /// This array must not be disposed and must have [size] equal to `1`.
  dynamic get scalar {
    _checkNotDisposed();
    if (size != 1) {
      throw StateError('Cannot retrieve scalar from tensor with size $size.');
    }
    final raw = readBufferAny(buffer, dtype, 0, offsetElements: offsetElements);
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
  GpuArray operator +(Object? other) => add(other);

  /// Elementwise subtraction (`this - other`). Supports broadcasting and scalars.
  GpuArray operator -(Object? other) => subtract(other);

  /// Elementwise multiplication (`this * other`). Supports broadcasting and scalars.
  GpuArray operator *(Object? other) => multiply(other);

  /// Elementwise division (`this / other`). Supports broadcasting and scalars.
  GpuArray operator /(Object? other) => divide(other);

  /// Elementwise negation (`-this`).
  GpuArray<T> operator -() => negate();

  /// Elementwise addition with another [GpuArray] or scalar.
  GpuArray add(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.add, other, out: out);

  /// Elementwise subtraction with another [GpuArray] or scalar.
  GpuArray subtract(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.subtract, other, out: out);

  /// Elementwise multiplication with another [GpuArray] or scalar.
  GpuArray multiply(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.multiply, other, out: out);

  /// Elementwise division with another [GpuArray] or scalar.
  GpuArray divide(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.divide, other, out: out);

  /// Elementwise power with another [GpuArray] or scalar.
  GpuArray pow(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.power, other, out: out);

  /// Elementwise remainder with another [GpuArray] or scalar.
  GpuArray remainder(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.remainder, other, out: out);

  /// Elementwise maximum with another [GpuArray] or scalar.
  GpuArray maximum(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.maximum, other, out: out);

  /// Elementwise minimum with another [GpuArray] or scalar.
  GpuArray minimum(Object? other, {GpuArray? out}) =>
      _dispatchBinary(BinaryOp.minimum, other, out: out);

  /// Elementwise equality comparison (`==`). Returns a boolean [GpuArray].
  GpuArray<Boolean> equal(Object? other, {GpuArray<Boolean>? out}) =>
      _dispatchComparison(BinaryOp.equal, other, out: out);

  /// Elementwise inequality comparison (`!=`). Returns a boolean [GpuArray].
  GpuArray<Boolean> notEqual(Object? other, {GpuArray<Boolean>? out}) =>
      _dispatchComparison(BinaryOp.notEqual, other, out: out);

  /// Elementwise greater than comparison (`>`). Returns a boolean [GpuArray].
  GpuArray<Boolean> greater(Object? other, {GpuArray<Boolean>? out}) =>
      _dispatchComparison(BinaryOp.greater, other, out: out);

  /// Elementwise greater than or equal comparison (`>=`).
  GpuArray<Boolean> greaterEqual(Object? other, {GpuArray<Boolean>? out}) =>
      _dispatchComparison(BinaryOp.greaterEqual, other, out: out);

  /// Elementwise less than comparison (`<`). Returns a boolean [GpuArray].
  GpuArray<Boolean> less(Object? other, {GpuArray<Boolean>? out}) =>
      _dispatchComparison(BinaryOp.less, other, out: out);

  /// Elementwise less than alias (`<`).
  GpuArray<Boolean> lessThan(Object? other, {GpuArray<Boolean>? out}) =>
      less(other, out: out);

  /// Elementwise less than or equal comparison (`<=`).
  GpuArray<Boolean> lessEqual(Object? other, {GpuArray<Boolean>? out}) =>
      _dispatchComparison(BinaryOp.lessEqual, other, out: out);

  /// Elementwise less than or equal alias (`<=`).
  GpuArray<Boolean> lessThanOrEqual(Object? other, {GpuArray<Boolean>? out}) =>
      lessEqual(other, out: out);

  /// Elementwise greater than alias (`>`).
  GpuArray<Boolean> greaterThan(Object? other, {GpuArray<Boolean>? out}) =>
      greater(other, out: out);

  /// Elementwise greater than or equal alias (`>=`).
  GpuArray<Boolean> greaterThanOrEqual(
    Object? other, {
    GpuArray<Boolean>? out,
  }) => greaterEqual(other, out: out);

  // --- Unary Math Operations ---

  /// Computes elementwise negation.
  GpuArray<T> negate({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.negate, out: out);

  /// Computes elementwise absolute value.
  GpuArray<T> abs({GpuArray<T>? out}) => _dispatchUnary(UnaryOp.abs, out: out);

  /// Computes elementwise square root.
  GpuArray<T> sqrt({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.sqrt, out: out);

  /// Computes elementwise exponential ($e^x$).
  GpuArray<T> exp({GpuArray<T>? out}) => _dispatchUnary(UnaryOp.exp, out: out);

  /// Computes elementwise natural logarithm ($\ln x$).
  GpuArray<T> log({GpuArray<T>? out}) => _dispatchUnary(UnaryOp.log, out: out);

  /// Computes elementwise sine ($\sin x$).
  GpuArray<T> sin({GpuArray<T>? out}) => _dispatchUnary(UnaryOp.sin, out: out);

  /// Computes elementwise cosine ($\cos x$).
  GpuArray<T> cos({GpuArray<T>? out}) => _dispatchUnary(UnaryOp.cos, out: out);

  /// Computes elementwise tangent ($\tan x$).
  GpuArray<T> tan({GpuArray<T>? out}) => _dispatchUnary(UnaryOp.tan, out: out);

  /// Computes elementwise arcsine ($\arcsin x$).
  GpuArray<T> asin({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.asin, out: out);

  /// Computes elementwise arccosine ($\arccos x$).
  GpuArray<T> acos({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.acos, out: out);

  /// Computes elementwise arctangent ($\arctan x$).
  GpuArray<T> atan({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.atan, out: out);

  /// Computes elementwise hyperbolic sine ($\sinh x$).
  GpuArray<T> sinh({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.sinh, out: out);

  /// Computes elementwise hyperbolic cosine ($\cosh x$).
  GpuArray<T> cosh({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.cosh, out: out);

  /// Computes elementwise hyperbolic tangent ($\tanh x$).
  GpuArray<T> tanh({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.tanh, out: out);

  /// Computes elementwise floor.
  GpuArray<T> floor({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.floor, out: out);

  /// Computes elementwise ceiling.
  GpuArray<T> ceil({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.ceil, out: out);

  /// Computes elementwise round.
  GpuArray<T> round({GpuArray<T>? out}) =>
      _dispatchUnary(UnaryOp.round, out: out);

  // --- Reductions ---

  /// Computes the sum of elements over the entire tensor or along [axis].
  GpuArray<T> sum({int? axis, bool keepDims = false, GpuArray<T>? out}) =>
      _dispatchReduction('sum', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the arithmetic mean of elements over the entire tensor or along
  /// [axis].
  GpuArray mean({int? axis, bool keepDims = false, GpuArray? out}) =>
      _dispatchReduction('mean', axis: axis, keepDims: keepDims, out: out);

  /// Computes the product of elements over the entire tensor or along [axis].
  GpuArray<T> prod({int? axis, bool keepDims = false, GpuArray<T>? out}) =>
      _dispatchReduction('prod', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the minimum value over the entire tensor or along [axis].
  GpuArray<T> min({int? axis, bool keepDims = false, GpuArray<T>? out}) =>
      _dispatchReduction('min', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the maximum value over the entire tensor or along [axis].
  GpuArray<T> max({int? axis, bool keepDims = false, GpuArray<T>? out}) =>
      _dispatchReduction('max', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  // --- Linear Algebra ---

  /// Matrix multiplication of two 1D, 2D, or batched N-D tensors.
  GpuArray<R> matmul<R extends DTypeTag>(GpuArray other, {GpuArray<R>? out}) {
    _checkNotDisposed();
    other._checkNotDisposed();

    if (rank < 1 || other.rank < 1) {
      throw GpuShapeMismatchException('matmul', shape, other.shape);
    }

    final outDtype = _promotedDType(dtype, other.dtype) as DType<R>;

    if (rank == 1 && other.rank == 1) {
      if (shape[0] != other.shape[0]) {
        throw GpuShapeMismatchException('matmul', shape, other.shape);
      }
      final dst = _prepareOut<R>('matmul', const [], outDtype, out);
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
        outShape: const [],
        outStrides: const [],
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
      final dst = _prepareOut<R>('matmul', outShape, outDtype, out);

      var executedViaNDArray = false;
      if (device.backend.isSimulated &&
          dtype == other.dtype &&
          dtype == outDtype &&
          (dtype == DType.float32 || dtype == DType.float64) &&
          !shape.contains(0) &&
          !other.shape.contains(0) &&
          dst.buffer.address != buffer.address &&
          dst.buffer.address != other.buffer.address) {
        withTemporaryNDArrayView((viewA) {
          other.withTemporaryNDArrayView((viewB) {
            dst.withTemporaryNDArrayView((viewDst) {
              nd.matmul(viewA, viewB, out: viewDst);
            });
          });
        });
        dst.buffer.markHostModified();
        executedViaNDArray = true;
      }

      if (!executedViaNDArray) {
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
      }
      if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
        dst.requiresGrad = true;
        dst.gradFn = MatmulBackward(this, other);
      }
      return dst;
    }

    // Batched N-D matmul
    if (rank < 2 || other.rank < 2) {
      throw GpuShapeMismatchException('matmul', shape, other.shape);
    }
    final m = shape[rank - 2];
    final k1 = shape[rank - 1];
    final k2 = other.shape[other.rank - 2];
    final n = other.shape[other.rank - 1];

    if (k1 != k2) {
      throw GpuShapeMismatchException('matmul', shape, other.shape);
    }

    final batchA = shape.sublist(0, rank - 2);
    final batchB = other.shape.sublist(0, other.rank - 2);
    final batchOut = broadcastShapes(batchA, batchB);
    final outShape = [...batchOut, m, n];

    final dst = _prepareOut<R>('matmul', outShape, outDtype, out);
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
  GpuArray<R> dot<R extends DTypeTag>(GpuArray other, {GpuArray<R>? out}) =>
      matmul<R>(other, out: out);

  // --- Tensor Views & Transformations ---

  /// Reshapes this tensor to [newShape], returning a view when contiguous or a
  /// contiguous copy otherwise.
  GpuArray<T> reshape(List<int> newShape) {
    _checkNotDisposed();
    final resolvedShape = List<int>.of(newShape);
    var inferIndex = -1;
    var knownProduct = 1;
    for (var i = 0; i < resolvedShape.length; i++) {
      final dim = resolvedShape[i];
      if (dim == -1) {
        if (inferIndex != -1) {
          throw ArgumentError.value(
            newShape,
            'newShape',
            'Must contain at most one -1 dimension.',
          );
        }
        inferIndex = i;
      } else if (dim < 0) {
        throw ArgumentError.value(
          newShape,
          'newShape',
          'Dimensions must be non-negative or -1.',
        );
      } else {
        knownProduct *= dim;
      }
    }
    if (inferIndex != -1) {
      if (knownProduct == 0 || size % knownProduct != 0) {
        throw ArgumentError.value(
          newShape,
          'newShape',
          'Cannot infer dimension for tensor of size $size.',
        );
      }
      resolvedShape[inferIndex] = size ~/ knownProduct;
    }

    final totalSize = computeSize(resolvedShape);
    if (totalSize != size) {
      throw ArgumentError.value(
        newShape,
        'newShape',
        'Must have total size matching $size (got $totalSize).',
      );
    }

    final GpuArray<T> res;
    if (isContiguous) {
      res = GpuArray._create<T>(
        buffer,
        shape: List.unmodifiable(resolvedShape),
        strides: List.unmodifiable(computeCStrides(resolvedShape)),
        dtype: dtype,
        device: device,
        offsetElements: offsetElements,
        parent: this,
      );
    } else {
      res = GpuArray<T>.empty(resolvedShape, dtype, device: device);
      GpuKernels.copyStrided(
        src: buffer,
        shape: shape,
        strides: strides,
        offsetSrc: offsetElements,
        dtypeSrc: dtype,
        dst: res.buffer,
        outStrides: computeCStrides(shape),
        offsetDst: 0,
        dtypeDst: dtype,
      );
    }

    if (isGradEnabled && requiresGrad) {
      res.requiresGrad = true;
      res.gradFn = ReshapeBackward(this, shape);
    }

    return res;
  }

  /// Permutes the axes of this tensor.
  GpuArray<T> transpose([List<int>? axes]) {
    _checkNotDisposed();
    final perm = axes ?? List.generate(rank, (i) => rank - 1 - i);
    if (perm.length != rank) {
      throw ArgumentError.value(
        axes,
        'axes',
        'Must have length matching tensor rank ($rank).',
      );
    }

    final seen = <int>{};
    final normPerm = <int>[];
    for (final ax in perm) {
      final norm = ax < 0 ? ax + rank : ax;
      if (norm < 0 || norm >= rank) {
        throw GpuAxisOutOfBoundsException(ax, rank);
      }
      if (!seen.add(norm)) {
        throw ArgumentError.value(
          axes,
          'axes',
          'Must not contain duplicate axes.',
        );
      }
      normPerm.add(norm);
    }

    final newShape = List<int>.generate(rank, (i) => shape[normPerm[i]]);
    final newStrides = List<int>.generate(rank, (i) => strides[normPerm[i]]);

    final res = GpuArray._create<T>(
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
      res.gradFn = TransposeBackward(this, normPerm);
    }

    return res;
  }

  /// Flattens this tensor into a 1D view or contiguous copy.
  GpuArray<T> flatten() => reshape([size]);

  /// Removes dimensions of size 1 at [axis], or all size-1 dimensions if
  /// [axis] is omitted.
  GpuArray<T> squeeze({int? axis}) {
    _checkNotDisposed();
    final newShape = <int>[];
    if (axis != null) {
      final normAxis = axis < 0 ? axis + rank : axis;
      if (normAxis < 0 || normAxis >= rank) {
        throw GpuAxisOutOfBoundsException(axis, rank);
      }
      for (var i = 0; i < rank; i++) {
        if (i == normAxis) {
          if (shape[i] != 1) {
            throw ArgumentError.value(
              axis,
              'axis',
              'Must select an axis of size 1 (got size ${shape[i]}).',
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
    _checkNotDisposed();
    final normAxis = axis < 0 ? axis + rank + 1 : axis;
    if (normAxis < 0 || normAxis > rank) {
      throw RangeError.range(normAxis, 0, rank, 'axis');
    }
    final newShape = List<int>.of(shape)..insert(normAxis, 1);
    final res = reshape(newShape);
    if (isGradEnabled && requiresGrad) {
      res.requiresGrad = true;
      res.gradFn = UnsqueezeBackward(this, shape);
    }
    return res;
  }

  /// Creates a contiguous copy of this tensor in device memory (or writes into
  /// [out] if provided).
  GpuArray<T> copy({GpuArray<T>? out}) {
    _checkNotDisposed();
    final dst = _prepareOut<T>('copy', shape, dtype, out);
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
  GpuArray<R> astype<R extends DTypeTag>(
    DType<R> targetDType, {
    GpuArray<R>? out,
  }) {
    _checkNotDisposed();
    if (dtype == targetDType && out == null) return this as GpuArray<R>;
    final dst = _prepareOut<R>('astype', shape, targetDType, out);
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

  /// Creates a strided subview of this array according to [specs].
  GpuArray<T> slice(List<Object?> specs) {
    _checkNotDisposed();
    final view = computeSliceView(
      shape: shape,
      strides: strides,
      offsetElements: offsetElements,
      specs: specs,
    );

    final res = GpuArray._create<T>(
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

  /// Slices this array using [index] (a [SliceSpec], `int`, or `List` of specs).
  GpuArray<T> operator [](Object index) {
    if (index is List) {
      return slice(index);
    }
    return slice([index]);
  }

  /// Interchanges two axes of this array.
  GpuArray<T> swapaxes(int axis1, int axis2) =>
      manip.swapaxes(this, axis1, axis2);

  /// Moves axes of this array to new positions.
  GpuArray<T> moveaxis(Object source, Object destination) =>
      manip.moveaxis(this, source, destination);

  /// Repeats elements of this array [repeats] times along [axis].
  GpuArray<T> repeat(int repeats, {int? axis, GpuArray<T>? out}) =>
      manip.repeat(this, repeats, axis: axis, out: out);

  /// Extracts a diagonal or constructs a diagonal array.
  GpuArray<T> diag({int k = 0, GpuArray<T>? out}) =>
      manip.diag(this, k: k, out: out);

  /// Extracts specified diagonals of this array.
  GpuArray<T> diagonal({
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    GpuArray<T>? out,
  }) => manip.diagonal(
    this,
    offset: offset,
    axis1: axis1,
    axis2: axis2,
    out: out,
  );

  /// Computes the sum along diagonals of this array as a [GpuArray].
  GpuArray<T> trace({
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    GpuArray<T>? out,
  }) => manip.trace(this, offset: offset, axis1: axis1, axis2: axis2, out: out);

  /// Extracts the upper triangular portion of this array.
  GpuArray<T> triu({int k = 0, GpuArray<T>? out}) =>
      manip.triu(this, k: k, out: out);

  /// Extracts the lower triangular portion of this array.
  GpuArray<T> tril({int k = 0, GpuArray<T>? out}) =>
      manip.tril(this, k: k, out: out);

  /// Reverses the order of elements along the given [axis].
  GpuArray<T> flip({Object? axis}) => manip.flip(this, axis: axis);

  /// Rolls array elements along a given [axis].
  GpuArray<T> roll(Object shift, {Object? axis, GpuArray<T>? out}) =>
      manip.roll(this, shift, axis: axis, out: out);

  /// Rotates an array by 90 degrees in the plane specified by [axes].
  GpuArray<T> rot90({int k = 1, List<int> axes = const [0, 1]}) =>
      manip.rot90(this, k: k, axes: axes);

  /// Pads this array with [padWidth] using [mode].
  GpuArray<T> pad(
    List<List<int>> padWidth, {
    manip.PadMode mode = manip.PadMode.constant,
    Object constantValues = 0,
    GpuArray<T>? out,
  }) => manip.pad(
    this,
    padWidth,
    mode: mode,
    constantValues: constantValues,
    out: out,
  );

  /// Broadcasts this array to [targetShape].
  GpuArray<T> broadcastTo(List<int> targetShape) =>
      manip.broadcastTo(this, targetShape);

  /// Promotes two [DType]s following NumPy's type promotion hierarchy.
  static DType promoteDTypes(DType a, DType b) => _promotedDType(a, b);

  // --- Conversions & Host Interop ---

  /// Downloads this GPU tensor into host memory as a standard [nd.NDArray].
  nd.NDArray<T> toNDArray() {
    _checkNotDisposed();
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

  /// Copies the elements of this tensor into a flat Dart list.
  List<dynamic> toList() {
    _checkNotDisposed();
    final total = size;
    final result = <dynamic>[];
    if (total == 0) return result;
    final coords = List<int>.filled(rank, 0);
    for (var i = 0; i < total; i++) {
      var elemOffset = 0;
      for (var d = 0; d < rank; d++) {
        elemOffset += coords[d] * strides[d];
      }
      final raw = readBufferAny(
        buffer,
        dtype,
        elemOffset,
        offsetElements: offsetElements,
      );
      if (dtype == DType.boolean) {
        result.add(raw == true || (raw is num && raw != 0));
      } else {
        result.add(raw);
      }
      for (var d = rank - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < shape[d]) break;
        coords[d] = 0;
      }
    }
    return result;
  }

  /// Copies the elements of this tensor into a nested Dart list matching
  /// [shape].
  List<dynamic> toNestedList() {
    _checkNotDisposed();
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

  GpuArray<R> _prepareOut<R extends DTypeTag>(
    String opName,
    List<int> outShape,
    DType<R> outDType,
    GpuArray? out,
  ) {
    if (out != null) {
      out._checkNotDisposed();
      if (out.size > 1 && out.strides.contains(0)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must be writeable and not a broadcasted view.',
        );
      }
      if (!areShapesEqual(out.shape, outShape)) {
        throw GpuShapeMismatchException('$opName(out)', out.shape, outShape);
      }
      if (out.dtype != outDType) {
        throw ArgumentError.value(
          out.dtype,
          'out.dtype',
          'Must match output dtype $outDType.',
        );
      }
      return out as GpuArray<R>;
    }
    return GpuArray<R>.empty(outShape, outDType, device: device);
  }

  bool _tryNDArrayBinary(BinaryOp op, GpuArray other, GpuArray dst) {
    if (!device.backend.isSimulated ||
        dtype != other.dtype ||
        dtype != dst.dtype ||
        shape.contains(0) ||
        other.shape.contains(0) ||
        dst.shape.contains(0) ||
        dst.buffer.address == buffer.address ||
        dst.buffer.address == other.buffer.address) {
      return false;
    }

    final isFloat = dtype == DType.float32 || dtype == DType.float64;
    final isComplex = dtype == DType.complex64 || dtype == DType.complex128;
    final isStdInt =
        dtype == DType.int8 ||
        dtype == DType.int16 ||
        dtype == DType.int32 ||
        dtype == DType.int64 ||
        dtype == DType.uint8 ||
        dtype == DType.uint16 ||
        dtype == DType.uint32 ||
        dtype == DType.uint64;

    switch (op) {
      case BinaryOp.add:
      case BinaryOp.subtract:
      case BinaryOp.multiply:
        if (!isFloat && !isComplex && !isStdInt) return false;
      case BinaryOp.divide:
        if (!isFloat && !isComplex) return false;
      case BinaryOp.power:
      case BinaryOp.remainder:
      case BinaryOp.maximum:
      case BinaryOp.minimum:
        if (!isFloat) return false;
      default:
        return false;
    }

    withTemporaryNDArrayView((viewA) {
      other.withTemporaryNDArrayView((viewB) {
        dst.withTemporaryNDArrayView((viewDst) {
          switch (op) {
            case BinaryOp.add:
              nd.add(viewA, viewB, out: viewDst);
            case BinaryOp.subtract:
              nd.subtract(viewA, viewB, out: viewDst);
            case BinaryOp.multiply:
              nd.multiply(viewA, viewB, out: viewDst);
            case BinaryOp.divide:
              nd.divide(viewA, viewB, out: viewDst);
            case BinaryOp.power:
              nd.power(viewA, viewB, out: viewDst);
            case BinaryOp.remainder:
              nd.remainder(viewA, viewB, out: viewDst);
            case BinaryOp.maximum:
              nd.binaryUfunc(
                viewA,
                viewB,
                op: nd.BinaryOp.maximum,
                out: viewDst,
              );
            case BinaryOp.minimum:
              nd.binaryUfunc(
                viewA,
                viewB,
                op: nd.BinaryOp.minimum,
                out: viewDst,
              );
            default:
              break;
          }
        });
      });
    });
    dst.buffer.markHostModified();
    return true;
  }

  bool _tryNDArrayUnary(UnaryOp op, GpuArray<T> dst) {
    if (!device.backend.isSimulated ||
        (dtype != DType.float32 && dtype != DType.float64) ||
        shape.contains(0) ||
        dst.buffer.address == buffer.address) {
      return false;
    }

    withTemporaryNDArrayView((viewSrc) {
      dst.withTemporaryNDArrayView((viewDst) {
        final ndOp = switch (op) {
          UnaryOp.negate => nd.UnaryOp.negative,
          UnaryOp.abs => nd.UnaryOp.abs,
          UnaryOp.sqrt => nd.UnaryOp.sqrt,
          UnaryOp.exp => nd.UnaryOp.exp,
          UnaryOp.log => nd.UnaryOp.log,
          UnaryOp.sin => nd.UnaryOp.sin,
          UnaryOp.cos => nd.UnaryOp.cos,
          UnaryOp.tan => nd.UnaryOp.tan,
          UnaryOp.asin => nd.UnaryOp.arcsin,
          UnaryOp.acos => nd.UnaryOp.arccos,
          UnaryOp.atan => nd.UnaryOp.arctan,
          UnaryOp.sinh => nd.UnaryOp.sinh,
          UnaryOp.cosh => nd.UnaryOp.cosh,
          UnaryOp.tanh => nd.UnaryOp.tanh,
          UnaryOp.floor => nd.UnaryOp.floor,
          UnaryOp.ceil => nd.UnaryOp.ceil,
          UnaryOp.round => nd.UnaryOp.rint,
        };
        nd.unaryUfunc(viewSrc, op: ndOp, out: viewDst);
      });
    });
    dst.buffer.markHostModified();
    return true;
  }

  GpuArray _dispatchBinary(BinaryOp op, Object? other, {GpuArray? out}) {
    _checkNotDisposed();
    if (other is GpuArray) {
      other._checkNotDisposed();
      final outShape = broadcastShapes(shape, other.shape);
      final outDtype = _promotedDType(dtype, other.dtype);
      final dst = _prepareOut(op.name, outShape, outDtype, out);

      if (!_tryNDArrayBinary(op, other, dst)) {
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
      }

      if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
        dst.requiresGrad = true;
        switch (op) {
          case BinaryOp.add:
            dst.gradFn = AddBackward(this, other);
          case BinaryOp.subtract:
            dst.gradFn = SubBackward(this, other);
          case BinaryOp.multiply:
            dst.gradFn = MulBackward(this, other);
          case BinaryOp.divide:
            dst.gradFn = DivBackward(this, other);
          case BinaryOp.power:
            dst.gradFn = PowBackward(this, other);
          default:
            break;
        }
      }

      return dst;
    } else if (other is num || other is bool || other is Complex) {
      final scalarArray = GpuArray.filled(
        const [],
        other!,
        dtype,
        device: device,
      );
      final res = _dispatchBinary(op, scalarArray, out: out);
      if (!res.requiresGrad) {
        scalarArray.dispose();
      }
      return res;
    } else {
      throw ArgumentError.value(
        other,
        'other',
        'Must be a GpuArray, num, bool, or Complex.',
      );
    }
  }

  GpuArray<Boolean> _dispatchComparison(
    BinaryOp op,
    Object? other, {
    GpuArray<Boolean>? out,
  }) {
    _checkNotDisposed();
    if (other is GpuArray) {
      other._checkNotDisposed();
      final outShape = broadcastShapes(shape, other.shape);
      final dst = _prepareOut<Boolean>(op.name, outShape, DType.boolean, out);

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
    } else if (other is num || other is bool || other is Complex) {
      final scalarArray = GpuArray.filled(
        const [],
        other!,
        dtype,
        device: device,
      );
      try {
        return _dispatchComparison(op, scalarArray, out: out);
      } finally {
        scalarArray.dispose();
      }
    } else {
      throw ArgumentError.value(
        other,
        'other',
        'Must be a GpuArray, num, bool, or Complex.',
      );
    }
  }

  GpuArray<T> _dispatchUnary(UnaryOp op, {GpuArray<T>? out}) {
    _checkNotDisposed();
    final dst = _prepareOut<T>(op.name, shape, dtype, out);
    if (!_tryNDArrayUnary(op, dst)) {
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
    }

    if (isGradEnabled && requiresGrad) {
      dst.requiresGrad = true;
      switch (op) {
        case UnaryOp.negate:
          dst.gradFn = NegBackward(this);
        case UnaryOp.sqrt:
          dst.gradFn = SqrtBackward(this, dst);
        case UnaryOp.exp:
          dst.gradFn = ExpBackward(this, dst);
        case UnaryOp.log:
          dst.gradFn = LogBackward(this);
        case UnaryOp.tanh:
          dst.gradFn = TanhBackward(this, dst);
        default:
          break;
      }
    }

    return dst;
  }

  GpuArray _dispatchReduction(
    String op, {
    int? axis,
    bool keepDims = false,
    GpuArray? out,
  }) {
    _checkNotDisposed();
    List<int> outShape;
    if (axis == null) {
      outShape = keepDims ? List.filled(rank, 1) : const [];
    } else {
      final normAxis = axis < 0 ? axis + rank : axis;
      if (normAxis < 0 || normAxis >= rank) {
        throw GpuAxisOutOfBoundsException(axis, rank);
      }
      outShape = <int>[];
      for (var i = 0; i < rank; i++) {
        if (i == normAxis) {
          if (keepDims) outShape.add(1);
        } else {
          outShape.add(shape[i]);
        }
      }
    }

    final isComplex = dtype == DType.complex64 || dtype == DType.complex128;
    final GpuArray dst;
    if (out != null) {
      dst = _prepareOut(op, outShape, out.dtype, out);
    } else if (op == 'mean' && !isComplex) {
      dst = GpuArray<Float64>.empty(outShape, DType.float64, device: device);
    } else {
      dst = GpuArray<T>.empty(outShape, dtype, device: device);
    }

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
      dtypeDst: dst.dtype,
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
    return this;
  }
}
