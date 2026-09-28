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
import 'package:ndarray/ndarray.dart' show DTypeTag, NDArray;

import '../device.dart';
import '../exceptions.dart';
import '../gpu_array.dart';

export '../operations/manipulation.dart' show diagonal, trace;
export 'decompositions.dart';
export 'solvers.dart';
export 'tensors.dart';

bool _shapesEqual(List<int> first, List<int> second) {
  if (first.length != second.length) return false;
  for (var i = 0; i < first.length; i++) {
    if (first[i] != second[i]) return false;
  }
  return true;
}

List<int> _broadcastBatchShapes(
  String operation,
  List<int> shapeA,
  List<int> shapeB,
) {
  final maxRank = math.max(shapeA.length, shapeB.length);
  final outShape = List<int>.filled(maxRank, 0);
  for (var i = 0; i < maxRank; i++) {
    final indexA = shapeA.length - 1 - i;
    final indexB = shapeB.length - 1 - i;
    final extentA = indexA >= 0 ? shapeA[indexA] : 1;
    final extentB = indexB >= 0 ? shapeB[indexB] : 1;
    if (extentA != extentB && extentA != 1 && extentB != 1) {
      throw GpuShapeMismatchException(operation, shapeA, shapeB);
    }
    outShape[maxRank - 1 - i] = math.max(extentA, extentB);
  }
  return outShape;
}

GpuArray<R> _writeOrWrapResult<R extends DTypeTag>(
  NDArray<R> hostResult,
  GpuDevice device,
  GpuArray<R>? out,
) {
  if (out != null) {
    if (!_shapesEqual(out.shape, hostResult.shape) ||
        out.dtype != hostResult.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be an array with shape ${hostResult.shape} and dtype '
            '${hostResult.dtype}, got shape ${out.shape} and dtype ${out.dtype}.',
      );
    }
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

void _copyGpuToOut<R extends DTypeTag>(GpuArray<R> source, GpuArray<R> out) {
  NDArray.scope(() {
    final hostSource = source.toNDArray();
    _writeOrWrapResult(hostSource, source.device, out);
  });
}

/// Computes the matrix product of two [GpuArray]s [a] and [b].
///
/// Supports 1D dot products, 2D matrix multiplication, and higher-rank batched
/// matrix multiplication with leading dimension broadcasting.
/// None of [a], [b], or [out] may be disposed, and neither [a] nor [b] may be
/// 0-dimensional.
GpuArray<T> matmul<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute matmul on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write matmul result to a disposed output GpuArray.',
    );
  }
  if (a.rank == 0 || b.rank == 0) {
    throw GpuShapeMismatchException('matmul', a.shape, b.shape);
  }

  final is1DA = a.rank == 1;
  final is1DB = b.rank == 1;

  if (is1DA && is1DB) {
    if (a.shape[0] != b.shape[0]) {
      throw GpuShapeMismatchException('matmul', a.shape, b.shape);
    }
    if (out == null) {
      return a.matmul<T>(b);
    }
    return ResourceScope.scope(() {
      final result = a.matmul<T>(b);
      _copyGpuToOut(result, out);
      return out;
    });
  }

  final shapeA = is1DA ? <int>[1, a.shape[0]] : a.shape;
  final shapeB = is1DB ? <int>[b.shape[0], 1] : b.shape;

  final m = shapeA[shapeA.length - 2];
  final kA = shapeA[shapeA.length - 1];
  final kB = shapeB[shapeB.length - 2];
  final n = shapeB[shapeB.length - 1];

  if (kA != kB) {
    throw GpuShapeMismatchException('matmul', a.shape, b.shape);
  }

  if (out == null && !is1DA && !is1DB) {
    return a.matmul<T>(b);
  }

  final batchA = shapeA.sublist(0, shapeA.length - 2);
  final batchB = shapeB.sublist(0, shapeB.length - 2);
  final batchShape = _broadcastBatchShapes('matmul', batchA, batchB);
  final outShape = <int>[...batchShape, if (!is1DA) m, if (!is1DB) n];

  if (out != null &&
      (!_shapesEqual(out.shape, outShape) || out.dtype != a.dtype)) {
    throw ArgumentError.value(
      out,
      'out',
      'Must be an array with shape $outShape and dtype ${a.dtype}, '
          'got shape ${out.shape} and dtype ${out.dtype}.',
    );
  }

  return NDArray.scope(() {
    final hostA = a.toNDArray();
    final hostB = b.toNDArray();
    final hostResult = nd.matmul<T>(hostA, hostB);
    return _writeOrWrapResult(hostResult, a.device, out);
  });
}

/// Computes the dot product of two [GpuArray]s [a] and [b].
///
/// For 0D scalars, performs multiplication. For 1D vectors or 2D matrices,
/// performs inner/matrix multiplication via [matmul].
/// None of [a], [b], or [out] may be disposed.
GpuArray<T> dot<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute dot on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write dot result to a disposed output GpuArray.');
  }
  if (a.rank == 0 || b.rank == 0) {
    return ResourceScope.scope(() {
      final product = a.multiply(b) as GpuArray<T>;
      if (out != null) {
        _copyGpuToOut(product, out);
        return out;
      }
      product.detachToParentScope();
      return product;
    });
  }
  return matmul(a, b, out: out);
}

/// Computes the flattened vector dot product of [a] and [b].
///
/// Both [a] and [b] must have the same total number of elements, and none of
/// [a], [b], or [out] may be disposed.
GpuArray<T> vdot<T extends DTypeTag>(
  GpuArray<T> a,
  GpuArray<T> b, {
  GpuArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute vdot on a disposed GpuArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write vdot result to a disposed output GpuArray.');
  }
  if (a.size != b.size) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must be of the same total size as a (${a.size}), got ${b.size}.',
    );
  }
  return ResourceScope.scope(() {
    final flatA = a.reshape(<int>[a.size]);
    final flatB = b.reshape(<int>[b.size]);
    final product = flatA.multiply(flatB) as GpuArray<T>;
    final summed = product.sum();
    if (out != null) {
      _copyGpuToOut(summed, out);
      return out;
    }
    summed.detachToParentScope();
    return summed;
  });
}

/// Linear algebra convenience methods on [GpuArray].
extension GpuArrayLinalgExtension<T extends DTypeTag> on GpuArray<T> {
  /// Flattened vector dot product of this array and [other].
  GpuArray<T> vdot(GpuArray<T> other, {GpuArray<T>? out}) =>
      _linalgVdot(this, other, out: out);
}

const _linalgVdot = vdot;
