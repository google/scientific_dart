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
import '../ndarray.dart';
import '../ndarray_bindings.dart' as bindings;
import '../scratch_arena.dart';
import 'helpers.dart'
    show checkNativeOom, encodeDType, sharesMemory, validateOutBuffer;
import 'math.dart';
import 'stats.dart';
import 'linalg.dart';

/// Supported distance metrics for pairwise distance computations.
enum DistanceMetric {
  /// Euclidean Distance:
  ///
  /// Math:
  /// \[
  /// d(u, v) = \sqrt{\sum_{i=1}^N (u_i - v_i)^2}
  /// \]
  euclidean,

  /// Cosine Distance:
  ///
  /// Math:
  /// \[
  /// d(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}
  /// \]
  ///
  /// If either vector has zero norm, the distance is NaN.
  cosine,

  /// Hamming Distance:
  ///
  /// Math:
  /// \[
  /// d(u, v) = \frac{\#\{i : u_i \neq v_i\}}{N}
  /// \]
  ///
  /// For 0-dimensional space (N=0), the distance is NaN.
  hamming,

  /// Chebyshev Distance ($L_\infty$ distance):
  ///
  /// Math:
  /// \[
  /// d(u, v) = \max_{i=1}^N |u_i - v_i|
  /// \]
  chebyshev,
}

/// Helper to promote an NDArray to Float64 if it is not already.
NDArray<Float64> _promoteToFloat64(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  if (a.dtype == DType.float64) {
    if (a is NDArray<Float64>) return a;
    return NDArray<Float64>.view(a, shape: a.shape, strides: a.strides);
  }
  final res = NDArray<Float64>.create(a.shape, DType.float64);
  final ndim = a.shape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesSrc = cBuffer + ndim;

    for (var i = 0; i < ndim; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    bindings.s_cast_generic(
      a.pointer.cast(),
      cStridesSrc,
      encodeDType(a.dtype),
      res.pointer.cast(),
      encodeDType(DType.float64),
      cShape,
      ndim,
    );
    checkNativeOom();
  } finally {
    ScratchArena.reset(marker);
  }
  return res;
}

/// Computes the pairwise distances between observations in n-dimensional space.
///
/// Returns a condensed distance matrix [Y]. For each [i] and [j] (with [0 <= i < j < M]),
/// the metric dist(u=X[i], v=X[j]) is computed and stored in entry [ij].
///
/// **Preconditions:**
/// - [x] must not be disposed.
/// - [x] must be a 2D array of shape `[M, N]`.
/// - [x] must not have a complex data type.
/// - If [out] is provided, it must not be disposed.
/// - If [out] is provided, it must have shape `[M * (M - 1) / 2]` and type [Float64].
/// - It is an error if [x] or [out] is disposed.
/// - It is an error if [x] is not a 2D array.
/// - It is an error if [x] has a complex data type.
/// - It is an error if [out] shape does not match `[M * (M - 1) / 2]`.
///
/// **Performance considerations:**
/// - Time complexity is $O(M^2 N)$ where $M$ is the number of observations and $N$ is the number of features.
/// - Space complexity is $O(M^2)$ for the output array.
///
/// {@example /example/distance_example.dart lang=dart}
NDArray<Float64> pdist<T extends DTypeTag>(
  NDArray<T> x, {
  DistanceMetric metric = DistanceMetric.euclidean,
  NDArray<Float64>? out,
}) {
  if (x.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  if (x.shape.length != 2) {
    throw ArgumentError.value(
      x.shape,
      'x',
      'Must be a 2D array (got shape ${x.shape})',
    );
  }
  if (x.dtype.isComplex) {
    throw ArgumentError.value(
      x.dtype,
      'x',
      'Must not be a complex dtype (complex dtypes are not supported for distance metrics)',
    );
  }

  final m = x.shape[0];
  final n = x.shape[1];
  final outSize = m * (m - 1) ~/ 2;

  if (out != null) {
    validateOutBuffer(out);
  }

  if (m < 2) {
    if (out != null) {
      if (out.shape.length != 1 || out.shape[0] != 0) {
        throw ArgumentError.value(
          out.shape,
          'out',
          'Must have shape [0] (got ${out.shape})',
        );
      }
      return out;
    }
    return NDArray<Float64>.create([0], DType.float64);
  }

  if (out != null) {
    if (out.shape.length != 1 || out.shape[0] != outSize) {
      throw ArgumentError.value(
        out.shape,
        'out',
        'Must have shape [$outSize] (got ${out.shape})',
      );
    }
  }

  final bool useTempOut =
      out != null && (!out.isContiguous || sharesMemory(x, out));

  return NDArray.scope(() {
    final target = (out != null && !useTempOut)
        ? out
        : NDArray<Float64>.create([outSize], DType.float64);

    if (metric == DistanceMetric.cosine) {
      _pdistCosine(x, out: target);
      if (useTempOut) {
        target.copy(out: out);
        return out;
      }
      if (out != null) {
        return target;
      } else {
        return target.detachToParentScope();
      }
    }

    final metricVal = metric.index;

    bindings.ndarray_pdist(
      encodeDType(x.dtype),
      x.pointer.cast(),
      m,
      n,
      x.strides[0],
      x.strides[1],
      metricVal,
      target.pointer.cast(),
      target.strides[0],
    );

    if (useTempOut) {
      target.copy(out: out);
      return out;
    }
    if (out != null) {
      return target;
    } else {
      return target.detachToParentScope();
    }
  });
}

/// Computes distance between each pair of the two collections of inputs.
///
/// Returns a matrix [Y] of shape `[M, K]` where [Y[i, j]] is the distance
/// between [xa[i]] and [xb[j]].
///
/// **Preconditions:**
/// - [xa] and [xb] must not be disposed.
/// - [xa] (shape `[M, N]`) and [xb] (shape `[K, N]`) must be 2D arrays with the same column dimension [N].
/// - [xa] and [xb] must not have complex data types.
/// - If [out] is provided, it must not be disposed.
/// - If [out] is provided, it must have shape `[M, K]` and type [Float64].
/// - It is an error if [xa], [xb], or [out] is disposed.
/// - It is an error if [xa] or [xb] is not a 2D array.
/// - It is an error if [xa] and [xb] have different column dimensions.
/// - It is an error if [xa] or [xb] has a complex data type.
/// - It is an error if [out] shape does not match `[M, K]`.
///
/// **Performance considerations:**
/// - Time complexity is $O(M K N)$.
/// - Space complexity is $O(M K)$ for the output array.
///
/// {@example /example/distance_example.dart lang=dart}
NDArray<Float64> cdist<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> xa,
  NDArray<Tb> xb, {
  DistanceMetric metric = DistanceMetric.euclidean,
  NDArray<Float64>? out,
}) {
  if (xa.isDisposed || xb.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot perform operation on a disposed array.');
  }
  if (xa.shape.length != 2 || xb.shape.length != 2) {
    throw ArgumentError.value(
      [xa.shape, xb.shape],
      'xa, xb',
      'Must be 2-dimensional (got xa: ${xa.shape}, xb: ${xb.shape})',
    );
  }
  if (xa.shape[1] != xb.shape[1]) {
    throw ArgumentError.value(
      xb.shape[1],
      'xb',
      'Must have the same number of columns (features) as xa (got xa: ${xa.shape[1]}, xb: ${xb.shape[1]})',
    );
  }
  if (xa.dtype.isComplex || xb.dtype.isComplex) {
    throw ArgumentError.value(
      [xa.dtype, xb.dtype],
      'xa, xb',
      'Must not be complex dtypes (complex dtypes are not supported for distance metrics)',
    );
  }

  final m = xa.shape[0];
  final k = xb.shape[0];
  final n = xa.shape[1];
  final outShape = [m, k];

  if (out != null) {
    validateOutBuffer(out);
    if (out.shape.length != 2 || out.shape[0] != m || out.shape[1] != k) {
      throw ArgumentError.value(
        out.shape,
        'out',
        'Must have shape $outShape (got ${out.shape})',
      );
    }
  }

  final bool useTempOut =
      out != null &&
      (!out.isContiguous || sharesMemory(xa, out) || sharesMemory(xb, out));

  return NDArray.scope(() {
    final target = (out != null && !useTempOut)
        ? out
        : NDArray<Float64>.create(outShape, DType.float64);

    if (m == 0 || k == 0) {
      if (out != null) {
        return out;
      } else {
        return target.detachToParentScope();
      }
    }

    if (metric == DistanceMetric.cosine) {
      _cdistCosine(xa, xb, out: target);
      if (useTempOut) {
        target.copy(out: out);
        return out;
      }
      if (out != null) {
        return target;
      } else {
        return target.detachToParentScope();
      }
    }

    NDArray<DTypeTag> xaReal = xa;
    NDArray<DTypeTag> xbReal = xb;
    if (xa.dtype != xb.dtype) {
      xaReal = _promoteToFloat64(xa);
      xbReal = _promoteToFloat64(xb);
    }

    final metricVal = metric.index;

    bindings.ndarray_cdist(
      encodeDType(xaReal.dtype),
      xaReal.pointer.cast(),
      xbReal.pointer.cast(),
      m,
      k,
      n,
      xaReal.strides[0],
      xaReal.strides[1],
      xbReal.strides[0],
      xbReal.strides[1],
      metricVal,
      target.pointer.cast(),
      target.strides[0],
      target.strides[1],
    );

    if (useTempOut) {
      target.copy(out: out);
      return out;
    }
    if (out != null) {
      return target;
    } else {
      return target.detachToParentScope();
    }
  });
}

/// Helper for optimized Cosine pdist implementation in Dart.
/// Cosine pdist is implemented using ndarray operations, not in a single intrinsic.
NDArray<Float64> _pdistCosine<T extends DTypeTag>(
  NDArray<T> x, {
  NDArray<Float64>? out,
}) {
  final m = x.shape[0];
  final outSize = m * (m - 1) ~/ 2;
  final bool useTempOut =
      out != null && (!out.isContiguous || sharesMemory(x, out));

  return NDArray.scope(() {
    final target = (out != null && !useTempOut)
        ? out
        : NDArray<Float64>.create([outSize], DType.float64);

    final xDouble = _promoteToFloat64(x);

    final NDArray<Float64> xSq = square(xDouble);
    final NDArray<Float64> xSum = sum(xSq, axis: 1);
    final NDArray<Float64> normX = sqrt(xSum);

    final NDArray<Float64> dot = matmul<Float64>(xDouble, xDouble.transposed);

    final NDArray<Float64> normX2D = normX.reshape([m, 1]);
    final NDArray<Float64> normXT2D = normX.reshape([1, m]);
    final NDArray<Float64> denom = matmul<Float64>(normX2D, normXT2D);

    final NDArray<Float64> div = divide(dot, denom);
    final one = NDArray<Float64>.fromList([1.0], [1], DType.float64);
    final NDArray<Float64> cosDistMatrix = subtract(one, div);

    final flatPtr = cosDistMatrix.pointer.cast<ffi.Double>();
    final resPtr = target.pointer.cast<ffi.Double>();
    var idx = 0;
    for (var i = 0; i < m; i++) {
      final rowOffset = i * m;
      for (var j = i + 1; j < m; j++) {
        final v = flatPtr[rowOffset + j];
        resPtr[idx++] = v.isNaN ? v : v.clamp(0.0, 2.0);
      }
    }

    if (useTempOut) {
      target.copy(out: out);
      return out;
    }
    if (out != null) {
      return target;
    } else {
      return target.detachToParentScope();
    }
  });
}

/// Helper for optimized Cosine cdist implementation in Dart.
/// Cosine cdist is implemented using ndarray operations, not in a single intrinsic.
NDArray<Float64> _cdistCosine<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> xa,
  NDArray<Tb> xb, {
  NDArray<Float64>? out,
}) {
  final m = xa.shape[0];
  final k = xb.shape[0];
  final outShape = [m, k];

  return NDArray.scope(() {
    final result = out ?? NDArray<Float64>.create(outShape, DType.float64);

    final xaDouble = _promoteToFloat64(xa);
    final xbDouble = _promoteToFloat64(xb);

    final NDArray<Float64> xaSq = square(xaDouble);
    final NDArray<Float64> xaSum = sum(xaSq, axis: 1);
    final NDArray<Float64> normXa = sqrt(xaSum);

    final NDArray<Float64> xbSq = square(xbDouble);
    final NDArray<Float64> xbSum = sum(xbSq, axis: 1);
    final NDArray<Float64> normXb = sqrt(xbSum);

    final NDArray<Float64> dot = matmul<Float64>(xaDouble, xbDouble.transposed);

    final NDArray<Float64> normXa2D = normXa.reshape([m, 1]);
    final NDArray<Float64> normXb2D = normXb.reshape([1, k]);
    final NDArray<Float64> denom = matmul<Float64>(normXa2D, normXb2D);

    final NDArray<Float64> div = divide(dot, denom);
    final one = NDArray<Float64>.fromList([1.0], [1], DType.float64);
    subtract(one, div, out: result);
    if (result.isContiguous) {
      final resPtr = result.pointer.cast<ffi.Double>();
      final total = m * k;
      for (var i = 0; i < total; i++) {
        final v = resPtr[i];
        if (!v.isNaN) {
          resPtr[i] = v.clamp(0.0, 2.0);
        }
      }
    } else {
      for (var i = 0; i < m * k; i++) {
        final v = result.getCellFlat(i);
        if (!v.isNaN) {
          result.setCellFlat(i, v.clamp(0.0, 2.0));
        }
      }
    }

    if (out != null) {
      return result;
    } else {
      return result.detachToParentScope();
    }
  });
}
