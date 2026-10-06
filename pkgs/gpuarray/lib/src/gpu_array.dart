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

import 'dart:core' as core;
import 'dart:core';
import 'dart:ffi' as ffi;
import 'dart:math' as math;

import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:resource_scope/resource_scope.dart';

import 'autograd/autograd.dart';
import 'backend/compute_engine.dart';
import 'backend/kernels.dart';
import 'buffer.dart';
import 'device.dart';
import 'dtype.dart';
import 'exceptions.dart';
import 'operations/indexing.dart' as indexing;
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
  GpuBuffer _buffer;

  /// The underlying GPU buffer holding tensor data.
  GpuBuffer get buffer => _buffer;

  /// The dimensions of the tensor.
  final List<int> shape;

  /// The memory stride (in elements) for each dimension.
  final List<int> strides;

  /// The data type of elements in this tensor.
  final DType<T> dtype;

  GpuDevice _device;

  /// The GPU device hosting this tensor.
  GpuDevice get device => _device;

  /// Offset in elements from the start of [buffer].
  final int offsetElements;

  /// Whether elements are contiguous in C-order in memory.
  final bool isContiguous;

  /// The parent array if this tensor is a view, preventing early garbage
  /// collection.
  final GpuArray<DTypeTag>? _parent;

  bool _requiresGrad;

  /// Whether this tensor tracks gradients for automatic differentiation.
  bool get requiresGrad => _requiresGrad;

  /// Updates whether this tensor tracks gradients for automatic differentiation.
  ///
  /// The [dtype] must be a floating-point or complex type when [value] is
  /// `true`.
  set requiresGrad(core.bool value) {
    _validateRequiresGrad(dtype, value);
    _requiresGrad = value;
  }

  static void _validateRequiresGrad(DType<DTypeTag> dtype, bool requiresGrad) {
    if (requiresGrad && !dtype.isFloating && !dtype.isComplex) {
      throw ArgumentError.value(
        dtype,
        'dtype',
        'Must be a floating-point or complex DType when requiresGrad is true.',
      );
    }
  }

  /// Accumulated gradient tensor on device.
  GpuArray<DTypeTag>? grad;

  /// The backward computation node that produced this tensor.
  GradFn? gradFn;

  bool _isDisposed = false;

  GpuArray._(
    GpuBuffer buffer, {
    required List<int> shape,
    required List<int> strides,
    required this.dtype,
    required GpuDevice device,
    this.offsetElements = 0,
    bool? isContiguous,
    GpuArray<DTypeTag>? parent,
    bool requiresGrad = false,
    this.grad,
    this.gradFn,
  }) : _buffer = buffer,
       _device = device,
       shape = List<int>.unmodifiable(shape),
       strides = List<int>.unmodifiable(strides),
       _requiresGrad = requiresGrad,
       isContiguous = isContiguous ?? isContiguousLayout(shape, strides),
       _parent = parent {
    _validateRequiresGrad(dtype, requiresGrad);
    ResourceScope.track(this);
    if (parent != null) {
      _buffer.retain();
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
    GpuArray<DTypeTag>? parent,
    bool requiresGrad = false,
    GpuArray<DTypeTag>? grad,
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
    List<Object?> values,
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
    _validateRequiresGrad(dtype, requiresGrad);
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
    for (var i = 0; i < totalSize; i++) {
      final item = flatList[i];
      if (item is! num &&
          item is! bool &&
          item is! BigInt &&
          item is! Complex) {
        throw ArgumentError.value(
          item,
          'values',
          'Must contain only numeric, boolean, BigInt, or Complex elements.',
        );
      }
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

    if (totalSize > 0) {
      using((arena) {
        final staging = arena<ffi.Uint8>(byteSize);
        for (var i = 0; i < totalSize; i++) {
          writePointerAny(staging, dtype, i, flatList[i]);
        }
        gpuBuffer.copyFromHost(staging.cast<ffi.Void>(), byteSize);
      });
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
    _validateRequiresGrad(dtype, requiresGrad);
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
    GpuArray<DTypeTag>? parent,
    bool requiresGrad = false,
  }) {
    if (buffer.isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot create GpuArray from a disposed GpuBuffer.',
      );
    }
    _validateRequiresGrad(dtype, requiresGrad);
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
    if (array.byteSize > 0) {
      array.buffer.clear();
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
    if (value is! num &&
        value is! bool &&
        value is! BigInt &&
        value is! Complex) {
      throw ArgumentError.value(
        value,
        'value',
        'Must be a numeric, boolean, BigInt, or Complex value.',
      );
    }
    final array = GpuArray<T>.empty(
      shape,
      dtype,
      device: device,
      requiresGrad: requiresGrad,
    );
    if (array.size > 0) {
      GpuKernels.executeFill(
        dst: array.buffer,
        outShape: array.shape,
        outStrides: array.strides,
        offsetDst: 0,
        dtypeDst: dtype,
        value: value,
      );
    }
    return array;
  }

  /// Creates a [GpuArray] of the specified [shape] filled with [value] (alias for [GpuArray.filled]).
  factory GpuArray.full(
    List<int> shape,
    Object value,
    DType<T> dtype, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) => GpuArray<T>.filled(
    shape,
    value,
    dtype,
    device: device,
    requiresGrad: requiresGrad,
  );

  /// Creates an uninitialized [GpuArray] with the same shape and device as [prototype].
  static GpuArray<R> emptyLike<R extends DTypeTag>(
    GpuArray<DTypeTag> prototype, {
    DType<R>? dtype,
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    prototype._checkNotDisposed();
    final targetDType = dtype ?? (prototype.dtype as DType<R>);
    return GpuArray<R>.empty(
      prototype.shape,
      targetDType,
      device: device ?? prototype.device,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a zero-filled [GpuArray] with the same shape and device as [prototype].
  static GpuArray<R> zerosLike<R extends DTypeTag>(
    GpuArray<DTypeTag> prototype, {
    DType<R>? dtype,
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    prototype._checkNotDisposed();
    final targetDType = dtype ?? (prototype.dtype as DType<R>);
    return GpuArray<R>.zeros(
      prototype.shape,
      targetDType,
      device: device ?? prototype.device,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a ones-filled [GpuArray] with the same shape and device as [prototype].
  static GpuArray<R> onesLike<R extends DTypeTag>(
    GpuArray<DTypeTag> prototype, {
    DType<R>? dtype,
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    prototype._checkNotDisposed();
    final targetDType = dtype ?? (prototype.dtype as DType<R>);
    return GpuArray<R>.ones(
      prototype.shape,
      targetDType,
      device: device ?? prototype.device,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a [GpuArray] filled with [fillValue] matching the shape and device of [prototype].
  static GpuArray<R> fullLike<R extends DTypeTag>(
    GpuArray<DTypeTag> prototype,
    Object fillValue, {
    DType<R>? dtype,
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    prototype._checkNotDisposed();
    final targetDType = dtype ?? (prototype.dtype as DType<R>);
    return GpuArray<R>.filled(
      prototype.shape,
      fillValue,
      targetDType,
      device: device ?? prototype.device,
      requiresGrad: requiresGrad,
    );
  }

  /// Creates a 1D [GpuArray] with evenly spaced values within `[start, stop)`.
  static GpuArray<R> arange<R extends DTypeTag>(
    num startOrStop, [
    num? stop,
    num step = 1,
    DType<R>? dtype,
    GpuDevice? device,
  ]) {
    if (step == 0) {
      throw ArgumentError.value(step, 'step', 'Must not be zero.');
    }
    final actualStart = stop == null ? 0.0 : startOrStop.toDouble();
    final actualStop = stop == null ? startOrStop.toDouble() : stop.toDouble();
    final stepDouble = step.toDouble();
    final rawCount = ((actualStop - actualStart) / stepDouble).ceil();
    final count = math.max(0, rawCount);
    final targetDType =
        dtype ??
        ((startOrStop is double || stop is double || step is double)
            ? DType.float32 as DType<R>
            : DType.int32 as DType<R>);
    final array = GpuArray<R>.empty([count], targetDType, device: device);
    if (count > 0) {
      GpuKernels.executeGenerator(
        mode: 'arange',
        dst: array.buffer,
        outShape: array.shape,
        offsetDst: 0,
        dtypeDst: targetDType,
        start: actualStart,
        step: stepDouble,
      );
    }
    return array;
  }

  /// Creates a 1D [GpuArray] of [num] evenly spaced samples over `[start, stop]`.
  static GpuArray<R> linspace<R extends DTypeTag>(
    num start,
    num stop, {
    int num = 50,
    bool endpoint = true,
    DType<R>? dtype,
    GpuDevice? device,
  }) {
    RangeError.checkNotNegative(num, 'num');
    final targetDType = dtype ?? (DType.float32 as DType<R>);
    final array = GpuArray<R>.empty([num], targetDType, device: device);
    if (num == 0) return array;
    final startDouble = start.toDouble();
    final stopDouble = stop.toDouble();
    final div = endpoint ? (num - 1) : num;
    final stepDouble = div > 0 ? (stopDouble - startDouble) / div : 0.0;
    GpuKernels.executeGenerator(
      mode: 'linspace',
      dst: array.buffer,
      outShape: array.shape,
      offsetDst: 0,
      dtypeDst: targetDType,
      start: startDouble,
      step: stepDouble,
    );
    if (endpoint && num > 1) {
      GpuKernels.executeFill(
        dst: array.buffer,
        outShape: const [1],
        outStrides: const [1],
        offsetDst: num - 1,
        dtypeDst: targetDType,
        value: stopDouble,
      );
    }
    return array;
  }

  /// Creates a 1D [GpuArray] of [num] numbers spaced evenly on a log scale (`base^start` to `base^stop`).
  static GpuArray<R> logspace<R extends DTypeTag>(
    num start,
    num stop, {
    int num = 50,
    bool endpoint = true,
    num base = 10.0,
    DType<R>? dtype,
    GpuDevice? device,
  }) {
    RangeError.checkNotNegative(num, 'num');
    final targetDType = dtype ?? (DType.float32 as DType<R>);
    final array = GpuArray<R>.empty([num], targetDType, device: device);
    if (num == 0) return array;
    final startDouble = start.toDouble();
    final stopDouble = stop.toDouble();
    final baseDouble = base.toDouble();
    final div = endpoint ? (num - 1) : num;
    final stepDouble = div > 0 ? (stopDouble - startDouble) / div : 0.0;
    GpuKernels.executeGenerator(
      mode: 'logspace',
      dst: array.buffer,
      outShape: array.shape,
      offsetDst: 0,
      dtypeDst: targetDType,
      start: startDouble,
      step: stepDouble,
      base: baseDouble,
    );
    return array;
  }

  /// Creates a 1D [GpuArray] of [num] numbers spaced evenly on a geometric progression from [start] to [stop].
  static GpuArray<R> geomspace<R extends DTypeTag>(
    num start,
    num stop, {
    int num = 50,
    bool endpoint = true,
    DType<R>? dtype,
    GpuDevice? device,
  }) {
    if (start == 0) {
      throw ArgumentError.value(start, 'start', 'Must not be zero.');
    }
    if (stop == 0) {
      throw ArgumentError.value(stop, 'stop', 'Must not be zero.');
    }
    if ((start < 0) != (stop < 0)) {
      throw ArgumentError.value(
        stop,
        'stop',
        'Must have the same sign as start ($start).',
      );
    }
    final sign = start < 0 ? -1.0 : 1.0;
    final logStart = math.log(start.abs()) / math.ln10;
    final logStop = math.log(stop.abs()) / math.ln10;
    final positive = logspace<R>(
      logStart,
      logStop,
      num: num,
      endpoint: endpoint,
      base: 10.0,
      dtype: dtype,
      device: device,
    );
    if (sign < 0) {
      final negated = positive.negate();
      positive.dispose();
      return negated;
    }
    return positive;
  }

  /// Creates a 2D [GpuArray] with ones on the [k]-th diagonal and zeros elsewhere.
  static GpuArray<R> eye<R extends DTypeTag>(
    int rows, {
    int? cols,
    int k = 0,
    DType<R>? dtype,
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    RangeError.checkNotNegative(rows, 'rows');
    final actualCols = cols ?? rows;
    RangeError.checkNotNegative(actualCols, 'cols');
    final targetDType = dtype ?? (DType.float32 as DType<R>);
    final array = GpuArray<R>.empty(
      [rows, actualCols],
      targetDType,
      device: device,
      requiresGrad: requiresGrad,
    );
    if (array.size > 0) {
      GpuKernels.executeGenerator(
        mode: 'eye',
        dst: array.buffer,
        outShape: array.shape,
        offsetDst: 0,
        dtypeDst: targetDType,
        cols: actualCols,
        k: k,
      );
    }
    return array;
  }

  /// Creates a 2D [GpuArray] with ones at and below the [k]-th diagonal and zeros elsewhere.
  static GpuArray<R> tri<R extends DTypeTag>(
    int rows, {
    int? cols,
    int k = 0,
    DType<R>? dtype,
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    RangeError.checkNotNegative(rows, 'rows');
    final actualCols = cols ?? rows;
    RangeError.checkNotNegative(actualCols, 'cols');
    final targetDType = dtype ?? (DType.float32 as DType<R>);
    final array = GpuArray<R>.empty(
      [rows, actualCols],
      targetDType,
      device: device,
      requiresGrad: requiresGrad,
    );
    if (array.size > 0) {
      GpuKernels.executeGenerator(
        mode: 'tri',
        dst: array.buffer,
        outShape: array.shape,
        offsetDst: 0,
        dtypeDst: targetDType,
        cols: actualCols,
        k: k,
      );
    }
    return array;
  }

  /// Creates a [GpuArray] by copying data from an existing host [NDArray].
  factory GpuArray.fromNDArray(
    nd.NDArray<T> ndarray, {
    GpuDevice? device,
    bool requiresGrad = false,
  }) {
    _validateRequiresGrad(ndarray.dtype, requiresGrad);
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
  void backward({GpuArray<DTypeTag>? gradient, bool retainGraph = false}) {
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
  GpuArray<T> operator +(Object? other) => add(other);

  /// Elementwise subtraction (`this - other`). Supports broadcasting and scalars.
  GpuArray<T> operator -(Object? other) => subtract(other);

  /// Elementwise multiplication (`this * other`). Supports broadcasting and scalars.
  GpuArray<T> operator *(Object? other) => multiply(other);

  /// Elementwise division (`this / other`). Supports broadcasting and scalars.
  GpuArray<T> operator /(Object? other) => divide(other);

  /// Elementwise modulo/remainder (`this % other`). Supports broadcasting and scalars.
  GpuArray<T> operator %(Object? other) => remainder(other);

  /// Elementwise floor division (`this ~/ other`). Supports broadcasting and scalars.
  GpuArray<T> operator ~/(Object? other) => floorDivide(other);

  /// Elementwise negation (`-this`).
  GpuArray<T> operator -() => negate();

  /// Elementwise bitwise AND (`this & other`).
  GpuArray<T> operator &(Object? other) => bitwiseAnd(other);

  /// Elementwise bitwise OR (`this | other`).
  GpuArray<T> operator |(Object? other) => bitwiseOr(other);

  /// Elementwise bitwise XOR (`this ^ other`).
  GpuArray<T> operator ^(Object? other) => bitwiseXor(other);

  /// Elementwise bitwise NOT / inversion (`~this`).
  GpuArray<T> operator ~() => bitwiseNot();

  /// Elementwise bitwise left shift (`this << other`).
  GpuArray<T> operator <<(Object? other) => leftShift(other);

  /// Elementwise bitwise right shift (`this >> other`).
  GpuArray<T> operator >>(Object? other) => rightShift(other);

  GpuArray<T> _asT(GpuArray<DTypeTag> res) {
    if (res is GpuArray<T>) return res;
    final casted = res.astype<T>(dtype);
    if (res.requiresGrad) {
      casted.requiresGrad = true;
      casted.gradFn = res.gradFn;
      res.gradFn = null;
    }
    res.dispose();
    return casted;
  }

  /// Elementwise addition with another [GpuArray] or scalar.
  GpuArray<T> add(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.add, other, out: out));

  /// Elementwise subtraction with another [GpuArray] or scalar.
  GpuArray<T> subtract(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.subtract, other, out: out));

  /// Elementwise multiplication with another [GpuArray] or scalar.
  GpuArray<T> multiply(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.multiply, other, out: out));

  /// Elementwise division with another [GpuArray] or scalar.
  GpuArray<T> divide(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.divide, other, out: out));

  /// Elementwise floor division with another [GpuArray] or scalar.
  GpuArray<T> floorDivide(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.floorDivide, other, out: out));

  /// Elementwise power with another [GpuArray] or scalar.
  GpuArray<T> pow(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.power, other, out: out));

  /// Elementwise remainder with another [GpuArray] or scalar.
  GpuArray<T> remainder(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.remainder, other, out: out));

  /// Elementwise C-style fmod remainder with another [GpuArray] or scalar.
  GpuArray<T> fmod(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.fmod, other, out: out));

  /// Elementwise maximum with another [GpuArray] or scalar.
  GpuArray<T> maximum(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.maximum, other, out: out));

  /// Elementwise minimum with another [GpuArray] or scalar.
  GpuArray<T> minimum(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.minimum, other, out: out));

  /// Elementwise two-argument arctangent (`atan2(this, other)`).
  GpuArray<T> atan2(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.atan2, other, out: out));

  /// Elementwise hypotenuse (`sqrt(this^2 + other^2)`).
  GpuArray<T> hypot(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.hypot, other, out: out));

  /// Elementwise copy sign of [other] to magnitude of `this`.
  GpuArray<T> copysign(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.copysign, other, out: out));

  /// Elementwise `this * 2^other`.
  GpuArray<T> ldexp(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.ldexp, other, out: out));

  /// Elementwise greatest common divisor.
  GpuArray<T> gcd(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.gcd, other, out: out));

  /// Elementwise least common multiple.
  GpuArray<T> lcm(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.lcm, other, out: out));

  /// Elementwise bitwise AND with [other].
  GpuArray<T> bitwiseAnd(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.bitwiseAnd, other, out: out));

  /// Elementwise bitwise OR with [other].
  GpuArray<T> bitwiseOr(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.bitwiseOr, other, out: out));

  /// Elementwise bitwise XOR with [other].
  GpuArray<T> bitwiseXor(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.bitwiseXor, other, out: out));

  /// Elementwise bitwise left shift by [other].
  GpuArray<T> leftShift(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.leftShift, other, out: out));

  /// Elementwise bitwise right shift by [other].
  GpuArray<T> rightShift(Object? other, {GpuArray<DTypeTag>? out}) =>
      _asT(_dispatchBinary(BinaryOp.rightShift, other, out: out));

  /// Elementwise equality comparison (`==`). Returns a boolean [GpuArray].
  GpuArray<Boolean> equal(Object? other, {GpuArray<DTypeTag>? out}) =>
      _dispatchComparison(BinaryOp.equal, other, out: out);

  /// Elementwise inequality comparison (`!=`). Returns a boolean [GpuArray].
  GpuArray<Boolean> notEqual(Object? other, {GpuArray<DTypeTag>? out}) =>
      _dispatchComparison(BinaryOp.notEqual, other, out: out);

  /// Elementwise greater than comparison (`>`). Returns a boolean [GpuArray].
  GpuArray<Boolean> greater(Object? other, {GpuArray<DTypeTag>? out}) =>
      _dispatchComparison(BinaryOp.greater, other, out: out);

  /// Elementwise greater than or equal comparison (`>=`).
  GpuArray<Boolean> greaterEqual(Object? other, {GpuArray<DTypeTag>? out}) =>
      _dispatchComparison(BinaryOp.greaterEqual, other, out: out);

  /// Elementwise less than comparison (`<`). Returns a boolean [GpuArray].
  GpuArray<Boolean> less(Object? other, {GpuArray<DTypeTag>? out}) =>
      _dispatchComparison(BinaryOp.less, other, out: out);

  /// Elementwise less than alias (`<`).
  GpuArray<Boolean> lessThan(Object? other, {GpuArray<DTypeTag>? out}) =>
      less(other, out: out);

  /// Elementwise less than or equal comparison (`<=`).
  GpuArray<Boolean> lessEqual(Object? other, {GpuArray<DTypeTag>? out}) =>
      _dispatchComparison(BinaryOp.lessEqual, other, out: out);

  /// Elementwise less than or equal alias (`<=`).
  GpuArray<Boolean> lessThanOrEqual(Object? other, {GpuArray<DTypeTag>? out}) =>
      lessEqual(other, out: out);

  /// Elementwise greater than alias (`>`).
  GpuArray<Boolean> greaterThan(Object? other, {GpuArray<DTypeTag>? out}) =>
      greater(other, out: out);

  /// Elementwise greater than or equal alias (`>=`).
  GpuArray<Boolean> greaterThanOrEqual(
    Object? other, {
    GpuArray<DTypeTag>? out,
  }) => greaterEqual(other, out: out);

  // --- Unary Math Operations ---

  /// Computes elementwise negation.
  GpuArray<T> negate({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.negate, out: out);

  /// Computes elementwise absolute value.
  GpuArray<T> abs({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.abs, out: out);

  /// Computes elementwise square root.
  GpuArray<T> sqrt({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.sqrt, out: out);

  /// Computes elementwise exponential ($e^x$).
  GpuArray<T> exp({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.exp, out: out);

  /// Computes elementwise natural logarithm ($\ln x$).
  GpuArray<T> log({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.log, out: out);

  /// Computes elementwise sine ($\sin x$).
  GpuArray<T> sin({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.sin, out: out);

  /// Computes elementwise cosine ($\cos x$).
  GpuArray<T> cos({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.cos, out: out);

  /// Computes elementwise tangent ($\tan x$).
  GpuArray<T> tan({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.tan, out: out);

  /// Computes elementwise arcsine ($\arcsin x$).
  GpuArray<T> asin({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.asin, out: out);

  /// Computes elementwise arccosine ($\arccos x$).
  GpuArray<T> acos({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.acos, out: out);

  /// Computes elementwise arctangent ($\arctan x$).
  GpuArray<T> atan({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.atan, out: out);

  /// Computes elementwise hyperbolic sine ($\sinh x$).
  GpuArray<T> sinh({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.sinh, out: out);

  /// Computes elementwise hyperbolic cosine ($\cosh x$).
  GpuArray<T> cosh({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.cosh, out: out);

  /// Computes elementwise hyperbolic tangent ($\tanh x$).
  GpuArray<T> tanh({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.tanh, out: out);

  /// Computes elementwise floor.
  GpuArray<T> floor({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.floor, out: out);

  /// Computes elementwise ceiling.
  GpuArray<T> ceil({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.ceil, out: out);

  /// Computes elementwise round.
  GpuArray<T> round({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.round, out: out);

  /// Computes elementwise round to nearest integer (`rint`).
  GpuArray<T> rint({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.rint, out: out);

  /// Computes elementwise truncation toward zero (`trunc`).
  GpuArray<T> trunc({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.trunc, out: out);

  /// Computes elementwise truncation toward zero (alias for [trunc]).
  GpuArray<T> fix({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.trunc, out: out);

  /// Computes elementwise sign indication (`-1`, `0`, `1`).
  GpuArray<T> sign({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.sign, out: out);

  /// Computes elementwise bitwise NOT / inversion (`~this`).
  GpuArray<T> bitwiseNot({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.bitwiseNot, out: out);

  /// Computes elementwise bitwise inversion (alias for [bitwiseNot]).
  GpuArray<T> invert({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.bitwiseNot, out: out);

  /// Computes elementwise complex conjugate.
  GpuArray<T> conj({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.conj, out: out);

  /// Computes elementwise complex conjugate (alias for [conj]).
  GpuArray<T> conjugate({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.conj, out: out);

  /// Computes elementwise cube root.
  GpuArray<T> cbrt({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.cbrt, out: out);

  /// Computes elementwise reciprocal (`1 / this`).
  GpuArray<T> reciprocal({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.reciprocal, out: out);

  /// Computes elementwise square (`this * this`).
  GpuArray<T> square({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.square, out: out);

  /// Computes elementwise reciprocal square root (`1 / sqrt(this)`).
  GpuArray<T> rsqrt({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.rsqrt, out: out);

  /// Computes elementwise `exp(this) - 1`.
  GpuArray<T> expm1({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.expm1, out: out);

  /// Computes elementwise `2^this`.
  GpuArray<T> exp2({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.exp2, out: out);

  /// Computes elementwise base-2 logarithm.
  GpuArray<T> log2({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.log2, out: out);

  /// Computes elementwise base-10 logarithm.
  GpuArray<T> log10({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.log10, out: out);

  /// Computes elementwise `log(1 + this)`.
  GpuArray<T> log1p({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.log1p, out: out);

  /// Computes elementwise inverse hyperbolic sine.
  GpuArray<T> asinh({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.asinh, out: out);

  /// Computes elementwise inverse hyperbolic cosine.
  GpuArray<T> acosh({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.acosh, out: out);

  /// Computes elementwise inverse hyperbolic tangent.
  GpuArray<T> atanh({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.atanh, out: out);

  /// Converts angles from degrees to radians.
  GpuArray<T> deg2rad({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.deg2rad, out: out);

  /// Converts angles from degrees to radians (alias for [deg2rad]).
  GpuArray<T> radians({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.deg2rad, out: out);

  /// Converts angles from radians to degrees.
  GpuArray<T> rad2deg({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.rad2deg, out: out);

  /// Converts angles from radians to degrees (alias for [rad2deg]).
  GpuArray<T> degrees({GpuArray<DTypeTag>? out}) =>
      _dispatchUnary(UnaryOp.rad2deg, out: out);

  /// Tests elementwise for `NaN` values.
  GpuArray<Boolean> isnan({GpuArray<DTypeTag>? out}) =>
      _dispatchUnaryPredicate('isnan', out: out);

  /// Tests elementwise for positive or negative infinity.
  GpuArray<Boolean> isinf({GpuArray<DTypeTag>? out}) =>
      _dispatchUnaryPredicate('isinf', out: out);

  /// Tests elementwise for finiteness (not infinity and not `NaN`).
  GpuArray<Boolean> isfinite({GpuArray<DTypeTag>? out}) =>
      _dispatchUnaryPredicate('isfinite', out: out);

  /// Tests elementwise whether the sign bit is set.
  GpuArray<Boolean> signbit({GpuArray<DTypeTag>? out}) =>
      _dispatchUnaryPredicate('signbit', out: out);

  /// Replaces `NaN` with [nan], positive infinity with [posinf], and negative
  /// infinity with [neginf].
  GpuArray<T> nanToNum({
    double nan = 0.0,
    double? posinf,
    double? neginf,
    GpuArray<DTypeTag>? out,
  }) => _dispatchNanToNum(nan: nan, posinf: posinf, neginf: neginf, out: out);

  /// Clips (limits) the values in this array to `[aMin, aMax]`.
  GpuArray<T> clip(Object? aMin, Object? aMax, {GpuArray<DTypeTag>? out}) {
    _checkNotDisposed();
    if (aMin == null && aMax == null) {
      throw ArgumentError.value(
        null,
        'aMin',
        'Must specify at least one of aMin or aMax.',
      );
    }
    if (aMin != null && aMax != null) {
      final lower = maximum(aMin);
      try {
        return lower.minimum(aMax, out: out);
      } finally {
        if (!identical(lower, out) && !lower.requiresGrad) {
          lower.dispose();
        }
      }
    }
    if (aMin != null) {
      return maximum(aMin, out: out);
    }
    return minimum(aMax, out: out);
  }

  /// Evaluates elementwise whether elements are equal to [other] within tolerance.
  GpuArray<Boolean> isClose(
    Object? other, {
    double rtol = 1e-05,
    double atol = 1e-08,
    bool equalNan = false,
    GpuArray<DTypeTag>? out,
  }) => _dispatchIsClose(
    other,
    rtol: rtol,
    atol: atol,
    equalNan: equalNan,
    out: out,
  );

  /// Evaluates elementwise whether elements are equal to [other] within tolerance.
  GpuArray<Boolean> isclose(
    Object? other, {
    double rtol = 1e-05,
    double atol = 1e-08,
    bool equalNan = false,
    GpuArray<DTypeTag>? out,
  }) => isClose(other, rtol: rtol, atol: atol, equalNan: equalNan, out: out);

  /// Returns `true` if all elements are equal to [other] within tolerance.
  bool allClose(
    Object? other, {
    double rtol = 1e-05,
    double atol = 1e-08,
    bool equalNan = false,
  }) {
    final mask = isClose(other, rtol: rtol, atol: atol, equalNan: equalNan);
    try {
      final reduced = mask.all();
      try {
        return reduced.scalar as bool;
      } finally {
        reduced.dispose();
      }
    } finally {
      mask.dispose();
    }
  }

  /// Returns `true` if all elements are equal to [other] within tolerance.
  bool allclose(
    Object? other, {
    double rtol = 1e-05,
    double atol = 1e-08,
    bool equalNan = false,
  }) => allClose(other, rtol: rtol, atol: atol, equalNan: equalNan);

  // --- Reductions ---

  /// Computes the sum of elements over the entire tensor or along [axis].
  GpuArray<T> sum({
    int? axis,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction(
            'sum',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<T>;

  /// Computes the sum of elements treating `NaN`s as zero.
  GpuArray<T> nansum({
    int? axis,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction(
            'nansum',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<T>;

  /// Computes the product of elements over the entire tensor or along [axis].
  GpuArray<T> prod({
    int? axis,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction(
            'prod',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<T>;

  /// Computes the minimum value over the entire tensor or along [axis].
  GpuArray<T> min({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('min', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the minimum value ignoring any `NaN`s.
  GpuArray<T> nanmin({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('nanmin', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the maximum value over the entire tensor or along [axis].
  GpuArray<T> max({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('max', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the maximum value ignoring any `NaN`s.
  GpuArray<T> nanmax({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('nanmax', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the peak-to-peak range (`maximum - minimum`) along [axis].
  GpuArray<T> ptp({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('ptp', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<T>;

  /// Computes the indices of minimum values over the entire tensor or along [axis].
  GpuArray<Int64> argmin({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('argmin', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<Int64>;

  /// Computes the indices of maximum values over the entire tensor or along [axis].
  GpuArray<Int64> argmax({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('argmax', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<Int64>;

  /// Counts the number of non-zero values in the array or along [axis].
  GpuArray<Int64> countNonzero({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction(
            'count_nonzero',
            axis: axis,
            keepDims: keepDims,
            out: out,
          )
          as GpuArray<Int64>;

  /// Tests whether all array elements along [axis] (or the entire array) evaluate to true.
  GpuArray<Boolean> all({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('all', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<Boolean>;

  /// Tests whether any array element along [axis] (or the entire array) evaluates to true.
  GpuArray<Boolean> any({
    int? axis,
    bool keepDims = false,
    GpuArray<DTypeTag>? out,
  }) =>
      _dispatchReduction('any', axis: axis, keepDims: keepDims, out: out)
          as GpuArray<Boolean>;

  // --- Linear Algebra ---

  /// Matrix multiplication of two 1D, 2D, or batched N-D tensors.
  GpuArray<T> matmul(GpuArray<DTypeTag> other, {GpuArray<DTypeTag>? out}) {
    _checkNotDisposed();
    other._checkNotDisposed();
    if (out != null) {
      out._checkNotDisposed();
    }
    if (other.device != device) {
      throw ArgumentError.value(
        other.device,
        'other.device',
        'Must reside on the same GpuDevice ($device) as this tensor.',
      );
    }

    if (rank < 1 || other.rank < 1) {
      throw GpuShapeMismatchException('matmul', shape, other.shape);
    }

    final outDtype = _promotedDType(dtype, other.dtype);

    if (rank == 1 && other.rank == 1) {
      if (shape[0] != other.shape[0]) {
        throw GpuShapeMismatchException('matmul', shape, other.shape);
      }
      final dst = _prepareOut('matmul', const [], outDtype, out);
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
      return _asT(dst);
    }

    // 2D (or N-D) x 1D: [..., M, K] @ [K] -> [..., M]
    if (rank >= 2 && other.rank == 1) {
      final k1 = shape[rank - 1];
      final k2 = other.shape[0];
      if (k1 != k2) {
        throw GpuShapeMismatchException('matmul', shape, other.shape);
      }
      final outShape = shape.sublist(0, rank - 1);
      final dst = _prepareOut('matmul', outShape, outDtype, out);
      GpuKernels.executeMatmul(
        srcA: buffer,
        shapeA: shape,
        stridesA: strides,
        offsetA: offsetElements,
        dtypeA: dtype,
        srcB: other.buffer,
        shapeB: [...List<int>.filled(rank - 2, 1), k2, 1],
        stridesB: [...List<int>.filled(rank - 2, 0), other.strides[0], 0],
        offsetB: other.offsetElements,
        dtypeB: other.dtype,
        dst: dst.buffer,
        outShape: [...outShape, 1],
        outStrides: [...dst.strides, 0],
        offsetDst: dst.offsetElements,
        dtypeDst: outDtype,
      );
      if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
        dst.requiresGrad = true;
        dst.gradFn = MatmulBackward(this, other);
      }
      return _asT(dst);
    }

    // 1D x 2D (or N-D): [K] @ [..., K, N] -> [..., N]
    if (rank == 1 && other.rank >= 2) {
      final k1 = shape[0];
      final k2 = other.shape[other.rank - 2];
      final n = other.shape[other.rank - 1];
      if (k1 != k2) {
        throw GpuShapeMismatchException('matmul', shape, other.shape);
      }
      final batchB = other.shape.sublist(0, other.rank - 2);
      final outShape = [...batchB, n];
      final dst = _prepareOut('matmul', outShape, outDtype, out);
      final batchDstStrides = dst.strides.sublist(0, dst.strides.length - 1);
      final nStrideDst = dst.strides.last;
      GpuKernels.executeMatmul(
        srcA: buffer,
        shapeA: [...List<int>.filled(other.rank - 2, 1), 1, k1],
        stridesA: [...List<int>.filled(other.rank - 2, 0), 0, strides[0]],
        offsetA: offsetElements,
        dtypeA: dtype,
        srcB: other.buffer,
        shapeB: other.shape,
        stridesB: other.strides,
        offsetB: other.offsetElements,
        dtypeB: other.dtype,
        dst: dst.buffer,
        outShape: [...batchB, 1, n],
        outStrides: [...batchDstStrides, 0, nStrideDst],
        offsetDst: dst.offsetElements,
        dtypeDst: outDtype,
      );
      if (isGradEnabled && (requiresGrad || other.requiresGrad)) {
        dst.requiresGrad = true;
        dst.gradFn = MatmulBackward(this, other);
      }
      return _asT(dst);
    }

    if (rank == 2 && other.rank == 2) {
      if (shape[1] != other.shape[0]) {
        throw GpuShapeMismatchException('matmul', shape, other.shape);
      }
      final outShape = [shape[0], other.shape[1]];
      final dst = _prepareOut('matmul', outShape, outDtype, out);

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
      return _asT(dst);
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
    final batchOut = broadcastShapes(batchA, batchB);
    final outShape = [...batchOut, m, n];

    final dst = _prepareOut('matmul', outShape, outDtype, out);
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
    return _asT(dst);
  }

  /// Dot product or matrix multiplication.
  GpuArray<T> dot(GpuArray<DTypeTag> other, {GpuArray<DTypeTag>? out}) =>
      matmul(other, out: out);

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
  GpuArray<T> copy({GpuArray<DTypeTag>? out}) {
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

  /// Sorts elements of this array in ascending order along [axis].
  GpuArray<T> sort({
    int? axis = -1,
    SortKind kind = SortKind.quicksort,
    GpuArray<T>? out,
  }) => indexing.sort<T>(this, axis: axis, kind: kind, out: out);

  /// Computes the 64-bit integer indices that would sort this array along [axis].
  GpuArray<Int64> argsort({
    int? axis = -1,
    SortKind kind = SortKind.quicksort,
    GpuArray<Int64>? out,
  }) => indexing.argsort<T>(this, axis: axis, kind: kind, out: out);

  /// Finds the [k] largest or smallest elements and their 64-bit indices
  /// along [axis].
  ({GpuArray<T> values, GpuArray<Int64> indices}) topk(
    int k, {
    int axis = -1,
    bool largest = true,
    bool sorted = true,
    GpuArray<T>? outValues,
    GpuArray<Int64>? outIndices,
  }) => indexing.topk<T>(
    this,
    k,
    axis: axis,
    largest: largest,
    sorted: sorted,
    outValues: outValues,
    outIndices: outIndices,
  );

  /// Rearranges elements along [axis] so that the [kth] element is in its
  /// final sorted position.
  GpuArray<T> partition(Object kth, {int? axis = -1, GpuArray<T>? out}) =>
      indexing.partition<T>(this, kth, axis: axis, out: out);

  /// Computes the 64-bit integer indices that would partition this array at
  /// [kth] along [axis].
  GpuArray<Int64> argpartition(
    Object kth, {
    int? axis = -1,
    GpuArray<Int64>? out,
  }) => indexing.argpartition<T>(this, kth, axis: axis, out: out);

  /// Finds 64-bit indices where elements of [v] should be inserted into this
  /// sorted 1-D array to maintain order.
  GpuArray<Int64> searchsorted(
    Object v, {
    SearchSide side = SearchSide.left,
    GpuArray<DTypeTag>? sorter,
    GpuArray<Int64>? out,
  }) => indexing.searchsorted<T>(this, v, side: side, sorter: sorter, out: out);

  /// Finds the sorted unique elements of this array.
  GpuArray<T> unique({int? axis}) => indexing.unique<T>(this, axis: axis);

  /// Finds the sorted unique elements and their 64-bit first-occurrence indices.
  ({GpuArray<T> values, GpuArray<Int64> indices}) uniqueWithIndex({
    int? axis,
  }) => indexing.uniqueWithIndex<T>(this, axis: axis);

  /// Finds the sorted unique elements and the 64-bit inverse reconstruction
  /// indices.
  ({GpuArray<T> values, GpuArray<Int64> inverse}) uniqueWithInverse({
    int? axis,
  }) => indexing.uniqueWithInverse<T>(this, axis: axis);

  /// Finds the sorted unique elements and their 64-bit occurrence counts.
  ({GpuArray<T> values, GpuArray<Int64> counts}) uniqueWithCounts({
    int? axis,
  }) => indexing.uniqueWithCounts<T>(this, axis: axis);

  /// Finds the sorted unique elements along with first-occurrence `indices`,
  /// `inverse` indices, and `counts`.
  ({
    GpuArray<T> values,
    GpuArray<Int64> indices,
    GpuArray<Int64> inverse,
    GpuArray<Int64> counts,
  })
  uniqueAll({int? axis}) => indexing.uniqueAll<T>(this, axis: axis);

  /// Counts occurrences of each value in this 1-D non-negative integer array.
  GpuArray<DTypeTag> bincount({
    GpuArray<DTypeTag>? weights,
    int minlength = 0,
    GpuArray<DTypeTag>? out,
  }) =>
      indexing.bincount(this, weights: weights, minlength: minlength, out: out);

  /// Computes the cumulative sum of elements along [axis].
  GpuArray<DTypeTag> cumsum({
    int? axis,
    DType? dtype,
    GpuArray<DTypeTag>? out,
  }) => indexing.cumsum(this, axis: axis, dtype: dtype, out: out);

  /// Computes the cumulative product of elements along [axis].
  GpuArray<DTypeTag> cumprod({
    int? axis,
    DType? dtype,
    GpuArray<DTypeTag>? out,
  }) => indexing.cumprod(this, axis: axis, dtype: dtype, out: out);

  /// Computes the [n]-th discrete difference along [axis].
  GpuArray<T> diff({
    int n = 1,
    int axis = -1,
    Object? prepend,
    Object? append,
    GpuArray<T>? out,
  }) => indexing.diff<T>(
    this,
    n: n,
    axis: axis,
    prepend: prepend,
    append: append,
    out: out,
  );

  /// Finds the indices of non-zero elements per dimension as
  /// `List<GpuArray<Int64>>`.
  List<GpuArray<Int64>> nonzero() => indexing.nonzero(this);

  /// Finds indices that are non-zero in the flattened version of this array.
  GpuArray<Int64> flatnonzero() => indexing.flatnonzero(this);

  /// Finds the indices of non-zero elements as a 2-D [Int64] array.
  GpuArray<Int64> argwhere() => indexing.argwhere(this);

  /// Takes elements from this array along [axis] at [indices].
  GpuArray<T> take(GpuArray<DTypeTag> indices, {int? axis, GpuArray<T>? out}) =>
      indexing.take<T>(this, indices, axis: axis, out: out);

  /// Replaces specified elements of this array with [values] at flat [indices].
  void put(GpuArray<DTypeTag> indices, GpuArray<T> values) =>
      indexing.put<T>(this, indices, values);

  /// Takes values from this array by matching 1-D index and data slices along
  /// [axis].
  GpuArray<T> takeAlongAxis(
    GpuArray<DTypeTag> indices,
    int axis, {
    GpuArray<T>? out,
  }) => indexing.takeAlongAxis<T>(this, indices, axis, out: out);

  /// Puts [values] into this array by matching 1-D index and data slices along
  /// [axis].
  void putAlongAxis(GpuArray<DTypeTag> indices, GpuArray<T> values, int axis) =>
      indexing.putAlongAxis<T>(this, indices, values, axis);

  /// Promotes two [DType]s following NumPy's type promotion hierarchy.
  static DType<DTypeTag> promoteDTypes(DType<DTypeTag> a, DType<DTypeTag> b) =>
      _promotedDType(a, b);

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
    final contiguousArray = isContiguous ? this : copy();
    try {
      return using((arena) {
        final staging = arena<ffi.Uint8>(byteSize);
        contiguousArray.buffer.copyToHost(
          staging.cast<ffi.Void>(),
          byteSize,
          offset: contiguousArray.offsetElements * dtype.byteWidth,
        );
        for (var i = 0; i < total; i++) {
          final raw = readPointerAny(staging, dtype, i);
          if (dtype == DType.boolean) {
            result.add(raw == true || (raw is num && raw != 0));
          } else {
            result.add(raw);
          }
        }
        return result;
      });
    } finally {
      if (!identical(contiguousArray, this)) {
        contiguousArray.dispose();
      }
    }
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
    GpuArray<DTypeTag>? out,
  ) {
    if (out != null) {
      out._checkNotDisposed();
      if (out.device != device) {
        throw ArgumentError.value(
          out.device,
          'out.device',
          'Must reside on the same GpuDevice ($device) as this tensor.',
        );
      }
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
    if (outDType == dtype && this is GpuArray<R>) {
      return GpuArray<T>.empty(outShape, dtype, device: device) as GpuArray<R>;
    }
    return GpuArray<R>.empty(outShape, outDType, device: device);
  }

  GpuArray<DTypeTag> _dispatchBinary(
    BinaryOp op,
    Object? other, {
    GpuArray<DTypeTag>? out,
  }) {
    _checkNotDisposed();
    if (out != null) {
      out._checkNotDisposed();
    }
    if (other is GpuArray<DTypeTag>) {
      other._checkNotDisposed();
      if (other.device != device) {
        throw ArgumentError.value(
          other.device,
          'other.device',
          'Must reside on the same GpuDevice ($device) as this tensor.',
        );
      }
      final outShape = broadcastShapes(shape, other.shape);
      final outDtype = _promotedDType(dtype, other.dtype);
      final dst = _prepareOut(op.name, outShape, outDtype, out);

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
    GpuArray<DTypeTag>? out,
  }) {
    _checkNotDisposed();
    if (out != null) {
      out._checkNotDisposed();
    }
    final isOrdering =
        op == BinaryOp.greater ||
        op == BinaryOp.greaterEqual ||
        op == BinaryOp.less ||
        op == BinaryOp.lessEqual;
    if (isOrdering &&
        (dtype.isComplex ||
            (other is GpuArray<DTypeTag> && other.dtype.isComplex) ||
            other is Complex)) {
      throw UnsupportedError(
        'Ordering comparison (${op.name}) is not supported for complex numbers.',
      );
    }
    if (other is GpuArray<DTypeTag>) {
      other._checkNotDisposed();
      if (other.device != device) {
        throw ArgumentError.value(
          other.device,
          'other.device',
          'Must reside on the same GpuDevice ($device) as this tensor.',
        );
      }
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

  GpuArray<T> _dispatchUnary(UnaryOp op, {GpuArray<DTypeTag>? out}) {
    _checkNotDisposed();
    final dst = _prepareOut<T>(op.name, shape, dtype, out);
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
        case UnaryOp.abs:
          dst.gradFn = AbsBackward(this);
        case UnaryOp.sqrt:
          dst.gradFn = SqrtBackward(this, dst);
        case UnaryOp.exp:
          dst.gradFn = ExpBackward(this, dst);
        case UnaryOp.log:
          dst.gradFn = LogBackward(this);
        case UnaryOp.sin:
          dst.gradFn = SinBackward(this);
        case UnaryOp.cos:
          dst.gradFn = CosBackward(this);
        case UnaryOp.tanh:
          dst.gradFn = TanhBackward(this, dst);
        default:
          break;
      }
    }

    return dst;
  }

  GpuArray<Boolean> _dispatchUnaryPredicate(
    String op, {
    GpuArray<DTypeTag>? out,
  }) {
    _checkNotDisposed();
    final dst = _prepareOut<Boolean>(op, shape, DType.boolean, out);
    GpuKernels.executeUnaryPredicate(
      op: op,
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtypeSrc: dtype,
      dst: dst.buffer,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
    );
    return dst;
  }

  GpuArray<R> _dispatchComplexComponent<R extends DTypeTag>(
    String op,
    DType<R> expectedOutDType, {
    double scale = 1.0,
    GpuArray<DTypeTag>? out,
  }) {
    _checkNotDisposed();
    final dst = _prepareOut<R>(op, shape, expectedOutDType, out);
    GpuKernels.executeComplexComponent(
      op: op,
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtypeSrc: dtype,
      dst: dst.buffer,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      dtypeDst: expectedOutDType,
      scale: scale,
    );
    return dst;
  }

  GpuArray<T> _dispatchNanToNum({
    double nan = 0.0,
    double? posinf,
    double? neginf,
    GpuArray<DTypeTag>? out,
  }) {
    _checkNotDisposed();
    final dst = _prepareOut<T>('nanToNum', shape, dtype, out);
    final double defaultPositiveInfinity = switch (dtype) {
      DType.float16 => 65504.0,
      DType.bfloat16 => 3.3895313892515355e38,
      DType.float32 || DType.complex64 => 3.4028234663852886e38,
      _ => double.maxFinite,
    };
    GpuKernels.executeNanToNum(
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtype: dtype,
      dst: dst.buffer,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      nan: nan,
      posinf: posinf ?? defaultPositiveInfinity,
      neginf: neginf ?? -defaultPositiveInfinity,
    );
    return dst;
  }

  GpuArray<Boolean> _dispatchIsClose(
    Object? other, {
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equalNan = false,
    GpuArray<DTypeTag>? out,
  }) {
    _checkNotDisposed();
    if (out != null) {
      out._checkNotDisposed();
    }
    if (other is GpuArray<DTypeTag>) {
      other._checkNotDisposed();
      if (other.device != device) {
        throw ArgumentError.value(
          other.device,
          'other.device',
          'Must reside on the same GpuDevice ($device) as this tensor.',
        );
      }
      final outShape = broadcastShapes(shape, other.shape);
      final dst = _prepareOut<Boolean>('isClose', outShape, DType.boolean, out);
      final promoted = _promotedDType(dtype, other.dtype);
      final opDType = (promoted.isFloating || promoted.isComplex)
          ? promoted
          : DType.float64;
      GpuKernels.executeIsClose(
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
        opDType: opDType,
        rtol: rtol,
        atol: atol,
        equalNan: equalNan,
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
        return _dispatchIsClose(
          scalarArray,
          rtol: rtol,
          atol: atol,
          equalNan: equalNan,
          out: out,
        );
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

  GpuArray<DTypeTag> _dispatchReduction(
    String op, {
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) {
    _checkNotDisposed();
    if (out != null) {
      out._checkNotDisposed();
    }
    final requiresNonEmpty =
        op == 'min' ||
        op == 'max' ||
        op == 'nanmin' ||
        op == 'nanmax' ||
        op == 'ptp' ||
        op == 'argmin' ||
        op == 'argmax';
    List<int> outShape;
    if (axis == null) {
      if (requiresNonEmpty && size == 0) {
        throw StateError('Cannot compute $op of an empty array.');
      }
      outShape = keepDims ? List.filled(rank, 1) : const [];
    } else {
      final normAxis = axis < 0 ? axis + rank : axis;
      if (normAxis < 0 || normAxis >= rank) {
        throw GpuAxisOutOfBoundsException(axis, rank);
      }
      if (requiresNonEmpty && shape[normAxis] == 0) {
        throw StateError('Cannot compute $op along an empty axis (size 0).');
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

    final DType<DTypeTag> expectedOutDType;
    if (dtype != null) {
      expectedOutDType = dtype;
    } else if (op == 'argmin' || op == 'argmax' || op == 'count_nonzero') {
      expectedOutDType = DType.int64;
    } else if (op == 'all' || op == 'any') {
      expectedOutDType = DType.boolean;
    } else if (op == 'mean' || op == 'nanmean') {
      expectedOutDType = (this.dtype.isFloating || this.dtype.isComplex)
          ? this.dtype
          : DType.float64;
    } else if (op == 'variance' || op == 'std') {
      expectedOutDType = _floatComputationDType;
    } else {
      expectedOutDType = this.dtype;
    }

    final dst = _prepareOut(op, outShape, expectedOutDType, out);

    GpuKernels.executeReduction(
      op: op,
      src: buffer,
      shape: shape,
      strides: strides,
      offsetSrc: offsetElements,
      dtypeSrc: this.dtype,
      dst: dst.buffer,
      outShape: outShape,
      outStrides: dst.strides,
      offsetDst: dst.offsetElements,
      dtypeDst: dst.dtype,
      axis: axis,
      ddof: ddof,
    );

    if (isGradEnabled && requiresGrad) {
      if (op == 'sum') {
        dst.requiresGrad = true;
        dst.gradFn = SumBackward(this, axis: axis, keepDims: keepDims);
      } else if (op == 'mean') {
        dst.requiresGrad = true;
        dst.gradFn = MeanBackward(this, axis: axis, keepDims: keepDims);
      }
    }

    return dst;
  }

  static DType<DTypeTag> _promotedDType(DType<DTypeTag> a, DType<DTypeTag> b) {
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

  DType<DTypeTag> get _realComponentDType => switch (dtype) {
    DType.complex64 => DType.float32,
    DType.complex128 => DType.float64,
    _ => dtype,
  };

  DType<DTypeTag> get _floatComputationDType => switch (dtype) {
    DType.float32 || DType.complex64 => DType.float32,
    DType.float16 => DType.float16,
    DType.bfloat16 => DType.bfloat16,
    _ => DType.float64,
  };

  /// Migrates this tensor's backing GPU buffer to [targetDevice] in place.
  void moveToDevice(GpuDevice targetDevice) {
    _checkNotDisposed();
    if (identical(_device, targetDevice)) return;
    final totalBytes = _buffer.sizeInBytes;
    final newBuffer = targetDevice.createBuffer(
      sizeInBytes: totalBytes,
      usage: _buffer.usage,
    );
    newBuffer.detachFromScope();
    if (totalBytes > 0) {
      using((arena) {
        final staging = arena<ffi.Uint8>(totalBytes);
        _buffer.copyToHost(staging.cast<ffi.Void>(), totalBytes);
        newBuffer.copyFromHost(staging.cast<ffi.Void>(), totalBytes);
      });
    }
    final oldBuffer = _buffer;
    _buffer = newBuffer;
    _device = targetDevice;
    oldBuffer.release();
    grad?.moveToDevice(targetDevice);
  }

  /// Copies this tensor onto [targetDevice] as a new contiguous [GpuArray].
  GpuArray<T> toDevice(GpuDevice targetDevice) {
    _checkNotDisposed();
    if (identical(_device, targetDevice)) {
      return copy();
    }
    final contiguousArray = isContiguous ? this : copy();
    final newBuffer = targetDevice.createBuffer(
      sizeInBytes: byteSize,
      usage:
          GpuBufferUsage.storage |
          GpuBufferUsage.copyDst |
          GpuBufferUsage.copySrc,
    );
    newBuffer.detachFromScope();
    if (byteSize > 0) {
      using((arena) {
        final staging = arena<ffi.Uint8>(byteSize);
        contiguousArray.buffer.copyToHost(
          staging.cast<ffi.Void>(),
          byteSize,
          offset: contiguousArray.offsetElements * dtype.byteWidth,
        );
        newBuffer.copyFromHost(staging.cast<ffi.Void>(), byteSize);
      });
    }
    if (!identical(contiguousArray, this)) {
      contiguousArray.dispose();
    }
    return GpuArray._create<T>(
      newBuffer,
      shape: shape,
      strides: computeCStrides(shape),
      dtype: dtype,
      device: targetDevice,
      requiresGrad: requiresGrad,
    );
  }

  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    ResourceScope.untrack(this);
    _buffer.release();
  }

  @override
  ScopedResource detachFromScope() {
    _checkNotDisposed();
    ResourceScope.untrack(this);
    buffer.detachFromScope();
    return this;
  }

  @override
  ScopedResource detachToParentScope() {
    _checkNotDisposed();
    ResourceScope.promoteToParent(this);
    return this;
  }
}

/// Statistical reduction methods on [GpuArray<Float32>] preserving [Float32].
extension GpuArrayFloat32ReductionExtension on GpuArray<Float32> {
  /// Computes the arithmetic mean of tensor elements, preserving [Float32].
  GpuArray<Float32> mean({
    int? axis,
    bool keepDims = false,
    DType<Float32>? dtype,
    GpuArray<Float32>? out,
  }) =>
      _dispatchReduction(
            'mean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float32>;

  /// Computes the arithmetic mean of tensor elements ignoring `NaN`s, preserving [Float32].
  GpuArray<Float32> nanmean({
    int? axis,
    bool keepDims = false,
    DType<Float32>? dtype,
    GpuArray<Float32>? out,
  }) =>
      _dispatchReduction(
            'nanmean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float32>;

  /// Computes the variance of tensor elements, preserving [Float32].
  GpuArray<Float32> variance({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float32>? dtype,
    GpuArray<Float32>? out,
  }) =>
      _dispatchReduction(
            'variance',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float32>;

  /// Computes the standard deviation of tensor elements, preserving [Float32].
  GpuArray<Float32> std({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float32>? dtype,
    GpuArray<Float32>? out,
  }) =>
      _dispatchReduction(
            'std',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float32>;
}

/// Statistical reduction methods on [GpuArray<Float16>] preserving [Float16].
extension GpuArrayFloat16ReductionExtension on GpuArray<Float16> {
  /// Computes the arithmetic mean of tensor elements, preserving [Float16].
  GpuArray<Float16> mean({
    int? axis,
    bool keepDims = false,
    DType<Float16>? dtype,
    GpuArray<Float16>? out,
  }) =>
      _dispatchReduction(
            'mean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float16>;

  /// Computes the arithmetic mean of tensor elements ignoring `NaN`s, preserving [Float16].
  GpuArray<Float16> nanmean({
    int? axis,
    bool keepDims = false,
    DType<Float16>? dtype,
    GpuArray<Float16>? out,
  }) =>
      _dispatchReduction(
            'nanmean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float16>;

  /// Computes the variance of tensor elements, preserving [Float16].
  GpuArray<Float16> variance({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float16>? dtype,
    GpuArray<Float16>? out,
  }) =>
      _dispatchReduction(
            'variance',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float16>;

  /// Computes the standard deviation of tensor elements, preserving [Float16].
  GpuArray<Float16> std({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float16>? dtype,
    GpuArray<Float16>? out,
  }) =>
      _dispatchReduction(
            'std',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float16>;
}

/// Statistical reduction methods on [GpuArray<BFloat16>] preserving [BFloat16].
extension GpuArrayBFloat16ReductionExtension on GpuArray<BFloat16> {
  /// Computes the arithmetic mean of tensor elements, preserving [BFloat16].
  GpuArray<BFloat16> mean({
    int? axis,
    bool keepDims = false,
    DType<BFloat16>? dtype,
    GpuArray<BFloat16>? out,
  }) =>
      _dispatchReduction(
            'mean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<BFloat16>;

  /// Computes the arithmetic mean of tensor elements ignoring `NaN`s, preserving [BFloat16].
  GpuArray<BFloat16> nanmean({
    int? axis,
    bool keepDims = false,
    DType<BFloat16>? dtype,
    GpuArray<BFloat16>? out,
  }) =>
      _dispatchReduction(
            'nanmean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<BFloat16>;

  /// Computes the variance of tensor elements, preserving [BFloat16].
  GpuArray<BFloat16> variance({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<BFloat16>? dtype,
    GpuArray<BFloat16>? out,
  }) =>
      _dispatchReduction(
            'variance',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<BFloat16>;

  /// Computes the standard deviation of tensor elements, preserving [BFloat16].
  GpuArray<BFloat16> std({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<BFloat16>? dtype,
    GpuArray<BFloat16>? out,
  }) =>
      _dispatchReduction(
            'std',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<BFloat16>;
}

/// Statistical reduction methods on [GpuArray<Float64>] preserving [Float64].
extension GpuArrayFloat64ReductionExtension on GpuArray<Float64> {
  /// Computes the arithmetic mean of tensor elements, preserving [Float64].
  GpuArray<Float64> mean({
    int? axis,
    bool keepDims = false,
    DType<Float64>? dtype,
    GpuArray<Float64>? out,
  }) =>
      _dispatchReduction(
            'mean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float64>;

  /// Computes the arithmetic mean of tensor elements ignoring `NaN`s, preserving [Float64].
  GpuArray<Float64> nanmean({
    int? axis,
    bool keepDims = false,
    DType<Float64>? dtype,
    GpuArray<Float64>? out,
  }) =>
      _dispatchReduction(
            'nanmean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float64>;

  /// Computes the variance of tensor elements, preserving [Float64].
  GpuArray<Float64> variance({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float64>? dtype,
    GpuArray<Float64>? out,
  }) =>
      _dispatchReduction(
            'variance',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float64>;

  /// Computes the standard deviation of tensor elements, preserving [Float64].
  GpuArray<Float64> std({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float64>? dtype,
    GpuArray<Float64>? out,
  }) =>
      _dispatchReduction(
            'std',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float64>;
}

/// Statistical reduction methods on [GpuArray<Complex64>].
extension GpuArrayComplex64ReductionExtension on GpuArray<Complex64> {
  /// Computes the arithmetic mean of complex tensor elements, preserving [Complex64].
  GpuArray<Complex64> mean({
    int? axis,
    bool keepDims = false,
    DType<Complex64>? dtype,
    GpuArray<Complex64>? out,
  }) =>
      _dispatchReduction(
            'mean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Complex64>;

  /// Computes the arithmetic mean of complex tensor elements ignoring `NaN`s, preserving [Complex64].
  GpuArray<Complex64> nanmean({
    int? axis,
    bool keepDims = false,
    DType<Complex64>? dtype,
    GpuArray<Complex64>? out,
  }) =>
      _dispatchReduction(
            'nanmean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Complex64>;

  /// Computes the real-valued variance of [Complex64] tensor elements in [Float32].
  GpuArray<Float32> variance({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float32>? dtype,
    GpuArray<Float32>? out,
  }) =>
      _dispatchReduction(
            'variance',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float32>;

  /// Computes the real-valued standard deviation of [Complex64] tensor elements in [Float32].
  GpuArray<Float32> std({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float32>? dtype,
    GpuArray<Float32>? out,
  }) =>
      _dispatchReduction(
            'std',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float32>;
}

/// Statistical reduction methods on [GpuArray<Complex128>].
extension GpuArrayComplex128ReductionExtension on GpuArray<Complex128> {
  /// Computes the arithmetic mean of complex tensor elements, preserving [Complex128].
  GpuArray<Complex128> mean({
    int? axis,
    bool keepDims = false,
    DType<Complex128>? dtype,
    GpuArray<Complex128>? out,
  }) =>
      _dispatchReduction(
            'mean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Complex128>;

  /// Computes the arithmetic mean of complex tensor elements ignoring `NaN`s, preserving [Complex128].
  GpuArray<Complex128> nanmean({
    int? axis,
    bool keepDims = false,
    DType<Complex128>? dtype,
    GpuArray<Complex128>? out,
  }) =>
      _dispatchReduction(
            'nanmean',
            axis: axis,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Complex128>;

  /// Computes the real-valued variance of [Complex128] tensor elements in [Float64].
  GpuArray<Float64> variance({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float64>? dtype,
    GpuArray<Float64>? out,
  }) =>
      _dispatchReduction(
            'variance',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float64>;

  /// Computes the real-valued standard deviation of [Complex128] tensor elements in [Float64].
  GpuArray<Float64> std({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<Float64>? dtype,
    GpuArray<Float64>? out,
  }) =>
      _dispatchReduction(
            'std',
            axis: axis,
            ddof: ddof,
            keepDims: keepDims,
            dtype: dtype,
            out: out,
          )
          as GpuArray<Float64>;
}

/// Default statistical reduction methods on integer, boolean, and dynamically typed [GpuArray]s.
extension GpuArrayDefaultReductionExtension on GpuArray<DTypeTag> {
  /// Computes the arithmetic mean of tensor elements.
  ///
  /// Floating-point and complex inputs preserve their dtype; integer and boolean
  /// inputs promote to [Float64] unless [dtype] is specified.
  GpuArray<DTypeTag> mean({
    int? axis,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) => _dispatchReduction(
    'mean',
    axis: axis,
    keepDims: keepDims,
    dtype: dtype,
    out: out,
  );

  /// Computes the arithmetic mean of tensor elements ignoring `NaN`s.
  GpuArray<DTypeTag> nanmean({
    int? axis,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) => _dispatchReduction(
    'nanmean',
    axis: axis,
    keepDims: keepDims,
    dtype: dtype,
    out: out,
  );

  /// Computes the variance of tensor elements along [axis] or over the entire tensor.
  GpuArray<DTypeTag> variance({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) => _dispatchReduction(
    'variance',
    axis: axis,
    ddof: ddof,
    keepDims: keepDims,
    dtype: dtype,
    out: out,
  );

  /// Computes the standard deviation of tensor elements along [axis] or over the entire tensor.
  GpuArray<DTypeTag> std({
    int? axis,
    int ddof = 0,
    bool keepDims = false,
    DType<DTypeTag>? dtype,
    GpuArray<DTypeTag>? out,
  }) => _dispatchReduction(
    'std',
    axis: axis,
    ddof: ddof,
    keepDims: keepDims,
    dtype: dtype,
    out: out,
  );
}

/// Strongly-typed complex component and phase extraction methods on [GpuArray].
extension GpuArraySpecComponentExtension<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>
    on GpuArray<DTypeSpec<R, E, F, C, M, S, D>> {
  /// Extracts the real part of each element (`Complex64 -> Float32`, `Complex128 -> Float64`).
  GpuArray<R> real({GpuArray<R>? out}) => _dispatchComplexComponent<R>(
    'real',
    _realComponentDType as DType<R>,
    out: out,
  );

  /// Extracts the imaginary part of each element (`Complex64 -> Float32`, `Complex128 -> Float64`, real -> zeros).
  GpuArray<R> imag({GpuArray<R>? out}) => _dispatchComplexComponent<R>(
    'imag',
    _realComponentDType as DType<R>,
    out: out,
  );

  /// Computes the phase angle of each element in radians (or degrees if [deg] is `true`).
  GpuArray<F> angle({bool deg = false, GpuArray<F>? out}) =>
      _dispatchComplexComponent<F>(
        'angle',
        _floatComputationDType as DType<F>,
        scale: deg ? (180.0 / math.pi) : 1.0,
        out: out,
      );
}

/// Fallback complex component and phase extraction methods for dynamically typed [GpuArray]s.
extension GpuArrayDefaultComponentExtension on GpuArray<DTypeTag> {
  /// Extracts the real part of each element.
  GpuArray<DTypeTag> real({GpuArray<DTypeTag>? out}) =>
      _dispatchComplexComponent('real', _realComponentDType, out: out);

  /// Extracts the imaginary part of each element.
  GpuArray<DTypeTag> imag({GpuArray<DTypeTag>? out}) =>
      _dispatchComplexComponent('imag', _realComponentDType, out: out);

  /// Computes the phase angle of each element in radians (or degrees if [deg] is `true`).
  GpuArray<DTypeTag> angle({bool deg = false, GpuArray<DTypeTag>? out}) =>
      _dispatchComplexComponent(
        'angle',
        _floatComputationDType,
        scale: deg ? (180.0 / math.pi) : 1.0,
        out: out,
      );
}

// --- Top-Level Elementwise Binary & Bitwise Functions ---

/// Elementwise addition of [a] and [b].
GpuArray<T> add<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.add(b, out: out);

/// Elementwise subtraction of [b] from [a].
GpuArray<T> subtract<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.subtract(b, out: out);

/// Elementwise multiplication of [a] and [b].
GpuArray<T> multiply<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.multiply(b, out: out);

/// Elementwise division of [a] by [b].
GpuArray<T> divide<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.divide(b, out: out);

/// Elementwise floor division of [a] by [b].
GpuArray<T> floorDivide<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.floorDivide(b, out: out);

/// Elementwise exponentiation ($a^b$).
GpuArray<T> pow<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.pow(b, out: out);

/// Elementwise exponentiation ($a^b$, alias for [pow]).
GpuArray<T> power<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.pow(b, out: out);

/// Elementwise floor remainder ($a \bmod b$).
GpuArray<T> remainder<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.remainder(b, out: out);

/// Elementwise floor remainder ($a \bmod b$, alias for [remainder]).
GpuArray<T> mod<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.remainder(b, out: out);

/// Elementwise C-style truncated remainder (`fmod`).
GpuArray<T> fmod<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.fmod(b, out: out);

/// Elementwise maximum of [a] and [b].
GpuArray<T> maximum<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.maximum(b, out: out);

/// Elementwise minimum of [a] and [b].
GpuArray<T> minimum<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.minimum(b, out: out);

/// Elementwise four-quadrant inverse tangent $\text{atan2}(a, b)$.
GpuArray<T> atan2<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.atan2(b, out: out);

/// Elementwise Euclidean hypotenuse $\sqrt{a^2 + b^2}$.
GpuArray<T> hypot<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.hypot(b, out: out);

/// Elementwise copysign (magnitude of [a] with sign of [b]).
GpuArray<T> copysign<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.copysign(b, out: out);

/// Elementwise $a \cdot 2^b$.
GpuArray<T> ldexp<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.ldexp(b, out: out);

/// Elementwise greatest common divisor of integer tensors [a] and [b].
GpuArray<T> gcd<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.gcd(b, out: out);

/// Elementwise least common multiple of integer tensors [a] and [b].
GpuArray<T> lcm<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.lcm(b, out: out);

/// Elementwise bitwise AND (`&`).
GpuArray<T> bitwiseAnd<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.bitwiseAnd(b, out: out);

/// Elementwise bitwise OR (`|`).
GpuArray<T> bitwiseOr<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.bitwiseOr(b, out: out);

/// Elementwise bitwise XOR (`^`).
GpuArray<T> bitwiseXor<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.bitwiseXor(b, out: out);

/// Elementwise bitwise NOT (`~`).
GpuArray<T> bitwiseNot<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.bitwiseNot(out: out);

/// Elementwise bitwise inversion (`~`, alias for [bitwiseNot]).
GpuArray<T> invert<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.invert(out: out);

/// Elementwise bitwise left shift (`<<`).
GpuArray<T> leftShift<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.leftShift(b, out: out);

/// Elementwise bitwise right shift (`>>`).
GpuArray<T> rightShift<T extends DTypeTag>(
  GpuArray<T> a,
  Object? b, {
  GpuArray<T>? out,
}) => a.rightShift(b, out: out);

// --- Top-Level Elementwise Unary, Predicate & Complex Functions ---

/// Elementwise negation (`-a`).
GpuArray<T> negate<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.negate(out: out);

/// Elementwise negation (`-a`, alias for [negate]).
GpuArray<T> negative<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.negate(out: out);

/// Elementwise absolute value ($|a|$).
GpuArray<T> abs<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.abs(out: out);

/// Elementwise square root ($\sqrt{a}$).
GpuArray<T> sqrt<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.sqrt(out: out);

/// Elementwise cube root ($\sqrt[3]{a}$).
GpuArray<T> cbrt<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.cbrt(out: out);

/// Elementwise reciprocal ($1 / a$).
GpuArray<T> reciprocal<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.reciprocal(out: out);

/// Elementwise square ($a^2$).
GpuArray<T> square<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.square(out: out);

/// Elementwise reciprocal square root ($1 / \sqrt{a}$).
GpuArray<T> rsqrt<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.rsqrt(out: out);

/// Elementwise exponential ($e^a$).
GpuArray<T> exp<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.exp(out: out);

/// Elementwise $e^a - 1$.
GpuArray<T> expm1<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.expm1(out: out);

/// Elementwise $2^a$.
GpuArray<T> exp2<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.exp2(out: out);

/// Elementwise natural logarithm ($\ln(a)$).
GpuArray<T> log<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.log(out: out);

/// Elementwise base-2 logarithm ($\log_2(a)$).
GpuArray<T> log2<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.log2(out: out);

/// Elementwise base-10 logarithm ($\log_{10}(a)$).
GpuArray<T> log10<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.log10(out: out);

/// Elementwise $\ln(1 + a)$.
GpuArray<T> log1p<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.log1p(out: out);

/// Elementwise sine ($\sin(a)$).
GpuArray<T> sin<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.sin(out: out);

/// Elementwise cosine ($\cos(a)$).
GpuArray<T> cos<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.cos(out: out);

/// Elementwise tangent ($\tan(a)$).
GpuArray<T> tan<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.tan(out: out);

/// Elementwise inverse sine ($\arcsin(a)$).
GpuArray<T> asin<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.asin(out: out);

/// Elementwise inverse cosine ($\arccos(a)$).
GpuArray<T> acos<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.acos(out: out);

/// Elementwise inverse tangent ($\arctan(a)$).
GpuArray<T> atan<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.atan(out: out);

/// Elementwise hyperbolic sine ($\sinh(a)$).
GpuArray<T> sinh<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.sinh(out: out);

/// Elementwise hyperbolic cosine ($\cosh(a)$).
GpuArray<T> cosh<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.cosh(out: out);

/// Elementwise hyperbolic tangent ($\tanh(a)$).
GpuArray<T> tanh<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.tanh(out: out);

/// Elementwise inverse hyperbolic sine ($\text{asinh}(a)$).
GpuArray<T> asinh<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.asinh(out: out);

/// Elementwise inverse hyperbolic cosine ($\text{acosh}(a)$).
GpuArray<T> acosh<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.acosh(out: out);

/// Elementwise inverse hyperbolic tangent ($\text{atanh}(a)$).
GpuArray<T> atanh<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.atanh(out: out);

/// Elementwise floor ($\lfloor a \rfloor$).
GpuArray<T> floor<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.floor(out: out);

/// Elementwise ceiling ($\lceil a \rceil$).
GpuArray<T> ceil<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.ceil(out: out);

/// Elementwise round to nearest integer.
GpuArray<T> round<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.round(out: out);

/// Elementwise round to nearest even integer.
GpuArray<T> rint<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.rint(out: out);

/// Elementwise truncation toward zero.
GpuArray<T> trunc<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.trunc(out: out);

/// Elementwise truncation toward zero (alias for [trunc]).
GpuArray<T> fix<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.fix(out: out);

/// Elementwise signum ($-1$, $0$, or $+1$).
GpuArray<T> sign<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.sign(out: out);

/// Elementwise conversion from degrees to radians.
GpuArray<T> deg2rad<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.deg2rad(out: out);

/// Elementwise conversion from degrees to radians (alias for [deg2rad]).
GpuArray<T> radians<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.radians(out: out);

/// Elementwise conversion from radians to degrees.
GpuArray<T> rad2deg<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.rad2deg(out: out);

/// Elementwise conversion from radians to degrees (alias for [rad2deg]).
GpuArray<T> degrees<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.rad2deg(out: out);

/// Elementwise test for `NaN`, returning a [GpuArray<Boolean>].
GpuArray<Boolean> isnan(GpuArray<DTypeTag> a, {GpuArray<Boolean>? out}) =>
    a.isnan(out: out);

/// Elementwise test for positive or negative infinity, returning a [GpuArray<Boolean>].
GpuArray<Boolean> isinf(GpuArray<DTypeTag> a, {GpuArray<Boolean>? out}) =>
    a.isinf(out: out);

/// Elementwise test for finiteness, returning a [GpuArray<Boolean>].
GpuArray<Boolean> isfinite(GpuArray<DTypeTag> a, {GpuArray<Boolean>? out}) =>
    a.isfinite(out: out);

/// Elementwise test whether the sign bit is set, returning a [GpuArray<Boolean>].
GpuArray<Boolean> signbit(GpuArray<DTypeTag> a, {GpuArray<Boolean>? out}) =>
    a.signbit(out: out);

/// Replaces `NaN`, positive infinity, and negative infinity values in [a].
GpuArray<T> nanToNum<T extends DTypeTag>(
  GpuArray<T> a, {
  double nan = 0.0,
  double? posinf,
  double? neginf,
  GpuArray<DTypeTag>? out,
}) => a.nanToNum(nan: nan, posinf: posinf, neginf: neginf, out: out);

/// Clips (limits) the values in [a] to `[aMin, aMax]`.
GpuArray<T> clip<T extends DTypeTag>(
  GpuArray<T> a,
  Object? aMin,
  Object? aMax, {
  GpuArray<DTypeTag>? out,
}) => a.clip(aMin, aMax, out: out);

/// Evaluates elementwise whether [a] and [b] are equal within tolerance.
GpuArray<Boolean> isClose(
  GpuArray<DTypeTag> a,
  Object? b, {
  double rtol = 1e-5,
  double atol = 1e-8,
  bool equalNan = false,
  GpuArray<Boolean>? out,
}) => a.isClose(b, rtol: rtol, atol: atol, equalNan: equalNan, out: out);

/// Evaluates elementwise whether [a] and [b] are equal within tolerance (alias for [isClose]).
GpuArray<Boolean> isclose(
  GpuArray<DTypeTag> a,
  Object? b, {
  double rtol = 1e-5,
  double atol = 1e-8,
  bool equalNan = false,
  GpuArray<Boolean>? out,
}) => a.isclose(b, rtol: rtol, atol: atol, equalNan: equalNan, out: out);

/// Returns `true` if all elements of [a] and [b] are equal within tolerance.
bool allClose(
  GpuArray<DTypeTag> a,
  Object? b, {
  double rtol = 1e-5,
  double atol = 1e-8,
  bool equalNan = false,
}) => a.allClose(b, rtol: rtol, atol: atol, equalNan: equalNan);

/// Returns `true` if all elements of [a] and [b] are equal within tolerance (alias for [allClose]).
bool allclose(
  GpuArray<DTypeTag> a,
  Object? b, {
  double rtol = 1e-5,
  double atol = 1e-8,
  bool equalNan = false,
}) => a.allclose(b, rtol: rtol, atol: atol, equalNan: equalNan);

/// Extracts the real part of [a].
GpuArray<DTypeTag> real(GpuArray<DTypeTag> a, {GpuArray<DTypeTag>? out}) =>
    a._dispatchComplexComponent('real', a._realComponentDType, out: out);

/// Extracts the imaginary part of [a].
GpuArray<DTypeTag> imag(GpuArray<DTypeTag> a, {GpuArray<DTypeTag>? out}) =>
    a._dispatchComplexComponent('imag', a._realComponentDType, out: out);

/// Elementwise complex conjugate of [a].
GpuArray<T> conj<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.conj(out: out);

/// Elementwise complex conjugate of [a] (alias for [conj]).
GpuArray<T> conjugate<T extends DTypeTag>(GpuArray<T> a, {GpuArray<T>? out}) =>
    a.conjugate(out: out);

/// Computes the phase angle of each element of [a] in radians (or degrees if [deg] is `true`).
GpuArray<DTypeTag> angle(
  GpuArray<DTypeTag> a, {
  bool deg = false,
  GpuArray<DTypeTag>? out,
}) => a._dispatchComplexComponent(
  'angle',
  a._floatComputationDType,
  scale: deg ? (180.0 / math.pi) : 1.0,
  out: out,
);

// --- Top-Level Reduction & Comparison Functions ---

/// Computes the sum of elements of [a] along [axis] or over the entire tensor.
GpuArray<T> sum<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  DType<DTypeTag>? dtype,
  GpuArray<DTypeTag>? out,
}) => a.sum(axis: axis, keepDims: keepDims, dtype: dtype, out: out);

/// Computes the sum of elements of [a] treating `NaN`s as zero.
GpuArray<T> nansum<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  DType<DTypeTag>? dtype,
  GpuArray<DTypeTag>? out,
}) => a.nansum(axis: axis, keepDims: keepDims, dtype: dtype, out: out);

/// Computes the arithmetic mean of elements of [a].
GpuArray<DTypeTag> mean(
  GpuArray<DTypeTag> a, {
  int? axis,
  bool keepDims = false,
  DType<DTypeTag>? dtype,
  GpuArray<DTypeTag>? out,
}) => a._dispatchReduction(
  'mean',
  axis: axis,
  keepDims: keepDims,
  dtype: dtype,
  out: out,
);

/// Computes the arithmetic mean of elements of [a] ignoring `NaN`s.
GpuArray<DTypeTag> nanmean(
  GpuArray<DTypeTag> a, {
  int? axis,
  bool keepDims = false,
  DType<DTypeTag>? dtype,
  GpuArray<DTypeTag>? out,
}) => a._dispatchReduction(
  'nanmean',
  axis: axis,
  keepDims: keepDims,
  dtype: dtype,
  out: out,
);

/// Computes the product of elements of [a] along [axis] or over the entire tensor.
GpuArray<T> prod<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  DType<DTypeTag>? dtype,
  GpuArray<DTypeTag>? out,
}) => a.prod(axis: axis, keepDims: keepDims, dtype: dtype, out: out);

/// Computes the minimum of elements of [a] along [axis] or over the entire tensor.
GpuArray<T> min<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<T>? out,
}) => a.min(axis: axis, keepDims: keepDims, out: out);

/// Computes the minimum of elements of [a] ignoring `NaN`s.
GpuArray<T> nanmin<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<T>? out,
}) => a.nanmin(axis: axis, keepDims: keepDims, out: out);

/// Computes the maximum of elements of [a] along [axis] or over the entire tensor.
GpuArray<T> max<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<T>? out,
}) => a.max(axis: axis, keepDims: keepDims, out: out);

/// Computes the maximum of elements of [a] ignoring `NaN`s.
GpuArray<T> nanmax<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<T>? out,
}) => a.nanmax(axis: axis, keepDims: keepDims, out: out);

/// Computes the peak-to-peak range ($\max - \min$) of elements of [a].
GpuArray<T> ptp<T extends DTypeTag>(
  GpuArray<T> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<T>? out,
}) => a.ptp(axis: axis, keepDims: keepDims, out: out);

/// Computes the variance of elements of [a].
GpuArray<DTypeTag> variance(
  GpuArray<DTypeTag> a, {
  int? axis,
  int ddof = 0,
  bool keepDims = false,
  DType<DTypeTag>? dtype,
  GpuArray<DTypeTag>? out,
}) => a._dispatchReduction(
  'variance',
  axis: axis,
  ddof: ddof,
  keepDims: keepDims,
  dtype: dtype,
  out: out,
);

/// Computes the standard deviation of elements of [a].
GpuArray<DTypeTag> std(
  GpuArray<DTypeTag> a, {
  int? axis,
  int ddof = 0,
  bool keepDims = false,
  DType<DTypeTag>? dtype,
  GpuArray<DTypeTag>? out,
}) => a._dispatchReduction(
  'std',
  axis: axis,
  ddof: ddof,
  keepDims: keepDims,
  dtype: dtype,
  out: out,
);

/// Computes the indices of the minimum values of [a] as [GpuArray<Int64>].
GpuArray<Int64> argmin(
  GpuArray<DTypeTag> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<Int64>? out,
}) => a.argmin(axis: axis, keepDims: keepDims, out: out);

/// Computes the indices of the maximum values of [a] as [GpuArray<Int64>].
GpuArray<Int64> argmax(
  GpuArray<DTypeTag> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<Int64>? out,
}) => a.argmax(axis: axis, keepDims: keepDims, out: out);

/// Counts the number of non-zero elements in [a] as [GpuArray<Int64>].
GpuArray<Int64> countNonzero(
  GpuArray<DTypeTag> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<Int64>? out,
}) => a.countNonzero(axis: axis, keepDims: keepDims, out: out);

/// Tests whether all elements of [a] evaluate to `true`.
GpuArray<Boolean> all(
  GpuArray<DTypeTag> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<Boolean>? out,
}) => a.all(axis: axis, keepDims: keepDims, out: out);

/// Tests whether any element of [a] evaluates to `true`.
GpuArray<Boolean> any(
  GpuArray<DTypeTag> a, {
  int? axis,
  bool keepDims = false,
  GpuArray<Boolean>? out,
}) => a.any(axis: axis, keepDims: keepDims, out: out);

/// Elementwise equality comparison ($a == b$).
GpuArray<Boolean> equal(
  GpuArray<DTypeTag> a,
  Object? b, {
  GpuArray<Boolean>? out,
}) => a.equal(b, out: out);

/// Elementwise inequality comparison ($a \neq b$).
GpuArray<Boolean> notEqual(
  GpuArray<DTypeTag> a,
  Object? b, {
  GpuArray<Boolean>? out,
}) => a.notEqual(b, out: out);

/// Elementwise greater-than comparison ($a > b$).
GpuArray<Boolean> greater(
  GpuArray<DTypeTag> a,
  Object? b, {
  GpuArray<Boolean>? out,
}) => a.greater(b, out: out);

/// Elementwise greater-than-or-equal comparison ($a \ge b$).
GpuArray<Boolean> greaterEqual(
  GpuArray<DTypeTag> a,
  Object? b, {
  GpuArray<Boolean>? out,
}) => a.greaterEqual(b, out: out);

/// Elementwise less-than comparison ($a < b$).
GpuArray<Boolean> less(
  GpuArray<DTypeTag> a,
  Object? b, {
  GpuArray<Boolean>? out,
}) => a.less(b, out: out);

/// Elementwise less-than-or-equal comparison ($a \le b$).
GpuArray<Boolean> lessEqual(
  GpuArray<DTypeTag> a,
  Object? b, {
  GpuArray<Boolean>? out,
}) => a.lessEqual(b, out: out);
