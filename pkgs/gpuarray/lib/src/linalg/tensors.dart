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

import 'package:ndarray/ndarray.dart' as nd;
import 'package:ndarray/ndarray.dart' show DType, DTypeTag, NDArray;

import '../device.dart';
import '../gpu_array.dart';

bool _shapesEqual(List<int> first, List<int> second) {
  if (first.length != second.length) return false;
  for (var i = 0; i < first.length; i++) {
    if (first[i] != second[i]) return false;
  }
  return true;
}

void _validateOutBuffer<R extends DTypeTag>(
  GpuArray<R> out,
  List<int> expectedShape,
  DType<R> expectedDType, {
  String name = 'out',
}) {
  if (out.size > 1 && out.strides.contains(0)) {
    throw ArgumentError.value(
      out,
      name,
      'Must be writeable and not a broadcasted view.',
    );
  }
  if (!_shapesEqual(out.shape, expectedShape) || out.dtype != expectedDType) {
    throw ArgumentError.value(
      out,
      name,
      'Must be an array with shape $expectedShape and dtype $expectedDType, '
      'got shape ${out.shape} and dtype ${out.dtype}.',
    );
  }
}

GpuArray<R> _writeOrWrapResult<R extends DTypeTag>(
  NDArray<R> hostResult,
  GpuDevice device,
  GpuArray<R>? out, {
  String outName = 'out',
}) {
  if (out != null) {
    _validateOutBuffer(out, hostResult.shape, hostResult.dtype, name: outName);
    if (out.isContiguous) {
      final contiguous = hostResult.isContiguous
          ? hostResult
          : hostResult.copy();
      if (out.byteSize > 0) {
        out.buffer.copyFromHost(
          contiguous.pointer.cast<ffi.Void>(),
          out.byteSize,
          offset: out.offsetElements * out.dtype.byteWidth,
        );
      }
    } else if (out.byteSize > 0) {
      out.buffer.copyToHost(
        out.buffer.address.cast<ffi.Void>(),
        out.buffer.sizeInBytes,
      );
      final totalBufferElements = out.buffer.sizeInBytes ~/ out.dtype.byteWidth;
      final rootBufferView = NDArray<R>.fromPointer(
        out.buffer.address.cast<ffi.Void>(),
        <int>[totalBufferElements],
        out.dtype,
      );
      final outView = NDArray<R>.view(
        rootBufferView,
        shape: out.shape,
        strides: out.strides,
        offsetElements: out.offsetElements,
      );
      hostResult.copy(out: outView);
      out.buffer.copyFromHost(
        out.buffer.address.cast<ffi.Void>(),
        out.buffer.sizeInBytes,
      );
    }
    return out;
  }
  final gpuResult = GpuArray<R>.fromNDArray(hostResult, device: device);
  gpuResult.detachToParentScope();
  return gpuResult;
}

/// Evaluates the Einstein summation convention [subscripts] on [operands].
///
/// Supports both explicit mode (with `'->'`) and implicit mode (summing over
/// repeated indices and ordering unique indices alphabetically).
///
/// The [operands] list must not be empty, and neither any operand nor [out]
/// may be disposed.
GpuArray<T> einsum<T extends DTypeTag>(
  String subscripts,
  List<GpuArray<T>> operands, {
  GpuArray<T>? out,
}) {
  for (final operand in operands) {
    if (operand.isDisposed) {
      throw StateError('Cannot execute einsum on a disposed GpuArray.');
    }
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write einsum result to a disposed output GpuArray.',
    );
  }
  if (operands.isEmpty) {
    throw ArgumentError.value(operands, 'operands', 'Must not be empty.');
  }

  return NDArray.scope(() {
    final hostOperands = <NDArray<T>>[
      for (final operand in operands) operand.toNDArray(),
    ];
    final hostResult = nd.einsum<T>(
      nd.EinsumSubscripts.parse(subscripts),
      hostOperands,
    );
    return _writeOrWrapResult(hostResult, operands.first.device, out);
  });
}

/// Computes the tensor dot product of [a] and [b] along the specified [axes].
///
/// The [axes] argument may be:
/// - An `int` $k$: sums over the last $k$ axes of [a] and the first $k$ axes
///   of [b] (default `2`).
/// - A `List<int>` `[axisA, axisB]`: contracts single axis `axisA` of [a] with
///   `axisB` of [b].
/// - A `List<List<int>>` `[axesA, axesB]`: contracts the listed axes of [a]
///   with the corresponding axes of [b].
///
/// None of [a], [b], or [out] may be disposed.
GpuArray<T> tensordot<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  Object axes = 2,
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute tensordot on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write tensordot result to a disposed output GpuArray.',
    );
  }

  return NDArray.scope(() {
    final hostA = a.toNDArray();
    final hostB = b.toNDArray();
    final hostResult = nd.tensordot<T>(hostA, hostB, axes: axes);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the Kronecker product $A \otimes B$ of two arrays [a] and [b].
///
/// None of [a], [b], or [out] may be disposed.
GpuArray<T> kron<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute kron on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write kron result to a disposed output GpuArray.');
  }

  return NDArray.scope(() {
    final hostA = a.toNDArray();
    final hostB = b.toNDArray();
    final hostResult = nd.kron<T>(hostA, hostB);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the inner product of two arrays [a] and [b] over their last axes.
///
/// For 1D vectors, this is the ordinary dot product. For higher-dimensional
/// arrays, it contracts the last axis of [a] with the last axis of [b].
/// None of [a], [b], or [out] may be disposed.
GpuArray<T> inner<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute inner on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write inner result to a disposed output GpuArray.',
    );
  }

  return NDArray.scope(() {
    final hostA = a.toNDArray();
    final hostB = b.toNDArray();
    final hostResult = nd.inner<T>(hostA, hostB);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the outer product $a \otimes b$ of two arrays [a] and [b]
/// (flattened to 1D if multidimensional).
///
/// None of [a], [b], or [out] may be disposed.
GpuArray<T> outer<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute outer on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write outer result to a disposed output GpuArray.',
    );
  }

  return NDArray.scope(() {
    final hostA = a.toNDArray();
    final hostB = b.toNDArray();
    final hostResult = nd.outer<T>(hostA, hostB);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the vector cross product of [a] and [b] along [axis] (or
/// [axisA], [axisB], [axisC]).
///
/// The cross-product axis of both [a] and [b] must have length 2 or 3.
/// None of [a], [b], or [out] may be disposed.
GpuArray<T> cross<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  int axisA = -1,
  int axisB = -1,
  int axisC = -1,
  int? axis,
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute cross on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write cross result to a disposed output GpuArray.',
    );
  }

  return NDArray.scope(() {
    final hostA = a.toNDArray();
    final hostB = b.toNDArray();
    final hostResult = nd.cross<T>(
      hostA,
      hostB,
      axisa: axisA,
      axisb: axisB,
      axisc: axisC,
      axis: axis,
    );
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}
