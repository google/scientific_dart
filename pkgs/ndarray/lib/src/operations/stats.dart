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

// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;

import '../ndarray.dart';
import '../nditer.dart';

import 'dart:ffi' as ffi;

import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';
import 'helpers.dart';
import 'broadcasting.dart';
import 'linalg.dart';
import 'manipulation.dart';

List<int> _reductionTargetShape(List<int> shape, int? axis, bool keepdims) {
  if (axis == null) {
    return keepdims ? List<int>.filled(shape.length, 1) : <int>[];
  }
  final normAxis = axis < 0 ? shape.length + axis : axis;
  if (keepdims) {
    return List<int>.from(shape)..[normAxis] = 1;
  } else {
    return List<int>.from(shape)..removeAt(normAxis);
  }
}

double _r_stat_scalar_double_fallback<T>(
  NDArray<T> arr,
  int size,
  double Function(ffi.Pointer<ffi.Double>, int) rDoubleFunc,
) {
  final d = castNDArray(arr, DType.float64);
  try {
    return rDoubleFunc(d.pointer.cast(), size);
  } finally {
    d.dispose();
  }
}

dynamic _r_stat_scalar_fallback<T>(
  NDArray<T> arr,
  int size,
  double Function(ffi.Pointer<ffi.Double>, int) rDoubleFunc,
) {
  final d = castNDArray(arr, DType.float64);
  try {
    final res = rDoubleFunc(d.pointer.cast(), size);
    return normalizeScalar(res, arr.dtype);
  } finally {
    d.dispose();
  }
}

void _s_stat_strided_fallback<T>(
  NDArray<T> a,
  NDArray<T> result,
  int rank,
  int normAxis,
  List<int> squeezedDestStrides,
  void Function(
    ffi.Pointer<ffi.Double> src,
    ffi.Pointer<ffi.Int> srcStrides,
    ffi.Pointer<ffi.Double> dest,
    ffi.Pointer<ffi.Int> destStrides,
    ffi.Pointer<ffi.Int> shape,
    int rank,
    int axis,
  )
  sDoubleFunc,
) {
  final doubleA = castNDArray(a, DType.float64);
  final doubleRes = NDArray<Float64>.zeros(result.shape, DType.float64);
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = doubleA.shape[i];
      cStridesA[i] = doubleA.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }
    sDoubleFunc(
      doubleA.pointer.cast(),
      cStridesA,
      doubleRes.pointer.cast(),
      cStridesRes,
      cShape,
      rank,
      normAxis,
    );
    final casted = castNDArray(doubleRes, result.dtype);
    casted.copy(out: result);
    casted.dispose();
  } finally {
    ScratchArena.reset(marker);
    doubleA.dispose();
    doubleRes.dispose();
  }
}

/// Methods for estimating quantiles/percentiles.
///
/// The descriptions below refer to the taxonomy established by
/// Hyndman and Fan (1996), "Sample Quantiles in Statistical Packages".
///
/// Most methods interpolate between two adjacent order statistics
/// \(x_{(j)}\) and \(x_{(j+1)}\) using:
/// \[Q(p) = (1 - g) \cdot x_{(j)} + g \cdot x_{(j+1)}\]
/// where \(j\) is the floor of the virtual index, and \(g\) is the fractional part.
enum QuantileMethod {
  /// **Type 1**: Inverse of empirical cumulative distribution function.
  /// Discontinuous.
  ///
  /// \(g = 0\) if the virtual index is integer, otherwise \(1\).
  invertedCdf,

  /// **Type 2**: Similar to [invertedCdf] but with averaging at discontinuities.
  /// Discontinuous.
  ///
  /// \(g = 0.5\) if the virtual index is integer, otherwise \(1\).
  averagedInvertedCdf,

  /// **Type 3**: Nearest observation.
  /// Discontinuous.
  ///
  /// Rounds the virtual index to the nearest integer. If the fractional part
  /// is exactly 0.5, rounds to the nearest even index (1-based).
  closestObservation,

  /// **Type 4**: Linear interpolation of the empirical CDF.
  /// Continuous.
  ///
  /// \(p_k = k / N\). Virtual index is \(p \cdot N - 1\) (0-based).
  interpolatedInvertedCdf,

  /// **Type 5**: Hazen's piecewise linear function.
  /// Continuous.
  ///
  /// \(p_k = (k - 0.5) / N\). Virtual index is \(p \cdot N - 0.5\) (0-based).
  hazen,

  /// **Type 6**: Weibull-style interpolation.
  /// Continuous.
  ///
  /// \(p_k = k / (N + 1)\). Used by Minitab and SPSS.
  /// Virtual index is \(p \cdot (N + 1) - 1\) (0-based).
  weibull,

  /// **Type 7**: Linear interpolation (default).
  /// Continuous.
  ///
  /// \(p_k = (k - 1) / (N - 1)\). Used by S and Excel.
  /// Virtual index is \(p \cdot (N - 1)\) (0-based).
  linear,

  /// **Type 8**: Median-unbiased.
  /// Continuous.
  ///
  /// \(p_k = (k - 1/3) / (N + 1/3)\). Approximately median-unbiased
  /// regardless of the distribution. Recommended by Hyndman and Fan.
  medianUnbiased,

  /// **Type 9**: Normal-unbiased.
  /// Continuous.
  ///
  /// \(p_k = (k - 3/8) / (N + 1/4)\). Approximately unbiased if the
  /// underlying distribution is normal.
  normalUnbiased,

  /// **NumPy Compatibility**: Lower.
  /// Discontinuous.
  ///
  /// Always uses the lower of the two nearest observations (\(g = 0\)).
  lower,

  /// **NumPy Compatibility**: Higher.
  /// Discontinuous.
  ///
  /// Always uses the higher of the two nearest observations (\(g = 1\)).
  higher,

  /// **NumPy Compatibility**: Midpoint.
  /// Discontinuous.
  ///
  /// Always uses the average of the two nearest observations (\(g = 0.5\)).
  midpoint,

  /// **NumPy Compatibility**: Nearest.
  /// Discontinuous.
  ///
  /// Uses the nearest observation. Rounds half-integers to the nearest even integer.
  nearest,
}

/// Computes the sum of elements in the array.
///
/// If [axis] is provided, sums along that axis and returns a new array.
/// Otherwise, sums all elements and returns a 0-D array containing the sum.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final s0 = sum(a, axis: 0); // Sum along rows
/// print(s0.toList()); // [4.0, 6.0]
/// ```
NDArray<T> sum<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute sum of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write sum to a disposed output array.');
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    if (size == 0) {
      if (a.dtype.isComplex) {
        result.setCellFlat(0, Complex(0.0, 0.0) as T);
      } else if (a.dtype.isFloating) {
        result.setCellFlat(0, 0.0 as T);
      } else if (a.dtype == DType.boolean) {
        result.setCellFlat(0, false as T);
      } else {
        result.setCellFlat(0, 0 as T);
      }
      return result;
    }

    final ptr = a.isContiguous ? a.pointer : null;
    if (ptr != null) {
      dynamic acc;
      switch (a.dtype) {
        case DType.float64:
          acc = r_sum_double(ptr.cast(), size);
        case DType.float32:
          acc = r_sum_float(ptr.cast(), size);
        case DType.int64:
          acc = r_sum_int64(ptr.cast(), size);
        case DType.int32:
          acc = r_sum_int32(ptr.cast(), size);
        case DType.uint8:
          acc = r_sum_uint8(ptr.cast(), size);
        case DType.int16:
          acc = r_sum_int16(ptr.cast(), size);
        case DType.complex128:
          final c = r_sum_complex128(ptr.cast(), size);
          acc = Complex(c.r, c.i);
        case DType.complex64:
          final c = r_sum_complex64(ptr.cast(), size);
          acc = Complex(c.r, c.i);
        case DType.boolean:
          acc = r_sum_uint8(ptr.cast(), size) != 0;
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          acc = _r_stat_scalar_fallback(a, size, r_sum_double);
      }
      result.setCellFlat(0, acc as T);
      return result;
    }

    final copyA = a.copy();
    dynamic acc;
    switch (copyA.dtype) {
      case DType.float64:
        acc = r_sum_double(copyA.pointer.cast(), size);
      case DType.float32:
        acc = r_sum_float(copyA.pointer.cast(), size);
      case DType.int64:
        acc = r_sum_int64(copyA.pointer.cast(), size);
      case DType.int32:
        acc = r_sum_int32(copyA.pointer.cast(), size);
      case DType.uint8:
        acc = r_sum_uint8(copyA.pointer.cast(), size);
      case DType.int16:
        acc = r_sum_int16(copyA.pointer.cast(), size);
      case DType.complex128:
        final c = r_sum_complex128(copyA.pointer.cast(), size);
        acc = Complex(c.r, c.i);
      case DType.complex64:
        final c = r_sum_complex64(copyA.pointer.cast(), size);
        acc = Complex(c.r, c.i);
      case DType.boolean:
        acc = r_sum_uint8(copyA.pointer.cast(), size) != 0;
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        acc = _r_stat_scalar_fallback(copyA, size, r_sum_double);
    }
    copyA.dispose();
    result.setCellFlat(0, acc as T);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final result = out ?? NDArray<T>.zeros(targetShape, a.dtype);
  if (out != null) {
    result.fill(normalizeScalar(0, a.dtype) as T);
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_sum_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_sum_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int64:
        s_sum_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int32:
        s_sum_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.uint8:
        s_sum_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int16:
        s_sum_int16(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.complex128:
        s_sum_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.complex64:
        s_sum_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.boolean:
        s_sum_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          s_sum_double,
        );
    }
    return result;
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the product of elements in the array.
///
/// If [axis] is provided, multiplies along that axis and returns a new array.
/// Otherwise, multiplies all elements and returns a 0-D array containing the product.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final p0 = prod(a, axis: 0); // Product along rows
/// print(p0.toList()); // [3.0, 8.0]
/// ```
NDArray<T> prod<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate product of disposed array');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write product result to disposed out array');
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
  if (axis == null) {
    final result = out ?? NDArray<T>.zeros(targetShape, a.dtype);
    if (size == 0) {
      if (a.dtype.isComplex) {
        result.setCellFlat(0, Complex(1.0, 0.0) as T);
      } else if (a.dtype.isFloating) {
        result.setCellFlat(0, 1.0 as T);
      } else if (a.dtype == DType.boolean) {
        result.setCellFlat(0, true as T);
      } else {
        result.setCellFlat(0, 1 as T);
      }
      return result;
    }

    final ptr = a.isContiguous ? a.pointer : null;
    if (ptr != null) {
      dynamic acc;
      switch (a.dtype) {
        case DType.float64:
          acc = r_prod_double(ptr.cast(), size);
        case DType.float32:
          acc = r_prod_float(ptr.cast(), size);
        case DType.int64:
          acc = r_prod_int64(ptr.cast(), size);
        case DType.int32:
          acc = r_prod_int32(ptr.cast(), size);
        case DType.uint8:
          acc = r_prod_uint8(ptr.cast(), size);
        case DType.int16:
          acc = r_prod_int16(ptr.cast(), size);
        case DType.complex128:
          final c = r_prod_complex128(ptr.cast(), size);
          acc = Complex(c.r, c.i);
        case DType.complex64:
          final c = r_prod_complex64(ptr.cast(), size);
          acc = Complex(c.r, c.i);
        case DType.boolean:
          acc = r_prod_uint8(ptr.cast(), size) != 0;
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          acc = _r_stat_scalar_fallback(a, size, r_prod_double);
      }
      result.setCellFlat(0, acc as T);
      return result;
    }

    final copyA = a.copy();
    dynamic acc;
    switch (copyA.dtype) {
      case DType.float64:
        acc = r_prod_double(copyA.pointer.cast(), size);
      case DType.float32:
        acc = r_prod_float(copyA.pointer.cast(), size);
      case DType.int64:
        acc = r_prod_int64(copyA.pointer.cast(), size);
      case DType.int32:
        acc = r_prod_int32(copyA.pointer.cast(), size);
      case DType.uint8:
        acc = r_prod_uint8(copyA.pointer.cast(), size);
      case DType.int16:
        acc = r_prod_int16(copyA.pointer.cast(), size);
      case DType.complex128:
        final c = r_prod_complex128(copyA.pointer.cast(), size);
        acc = Complex(c.r, c.i);
      case DType.complex64:
        final c = r_prod_complex64(copyA.pointer.cast(), size);
        acc = Complex(c.r, c.i);
      case DType.boolean:
        acc = r_prod_uint8(copyA.pointer.cast(), size) != 0;
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        acc = _r_stat_scalar_fallback(copyA, size, r_prod_double);
    }
    copyA.dispose();
    result.setCellFlat(0, acc as T);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final result = out ?? NDArray<T>.ones(targetShape, a.dtype);
  if (out != null) {
    result.fill(normalizeScalar(1, a.dtype) as T);
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_prod_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_prod_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int64:
        s_prod_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int32:
        s_prod_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.uint8:
        s_prod_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int16:
        s_prod_int16(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.complex128:
        s_prod_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.complex64:
        s_prod_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.boolean:
        s_prod_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          s_prod_double,
        );
    }
    return result;
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Returns true if all elements along a given [axis] evaluate to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, true, false], [3], DType.boolean);
/// final res = all(a); // false
/// ```
NDArray<bool> all<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<bool>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute all() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write all() result to a disposed output array.');
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    var allTrue = true;
    final iter = NDIter(a);
    while (iter.moveNext()) {
      if (!isTrueHelper(a.getCellFlat(iter.index))) {
        allTrue = false;
        break;
      }
    }
    final result = out ?? NDArray<bool>.create(targetShape, DType.boolean);
    result.setCellFlat(0, allTrue);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<bool>.create(targetShape, DType.boolean);
  result.fill(true); // Initialize to true everywhere

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  reduceRecursive<T, bool>(
    a,
    result,
    List<int>.filled(rank, 0),
    List<int>.filled(rank - 1, 0),
    normAxis,
    0,
    (current, val) => current && isTrueHelper(val),
    destStrides: squeezedDestStrides,
  );

  return result;
}

/// Returns true if any element along a given [axis] evaluates to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, false, false], [3], DType.boolean);
/// final res = any(a); // true
/// ```
NDArray<bool> any<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<bool>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute any() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write any() result to a disposed output array.');
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    var anyTrue = false;
    final iter = NDIter(a);
    while (iter.moveNext()) {
      if (isTrueHelper(a.getCellFlat(iter.index))) {
        anyTrue = true;
        break;
      }
    }
    final result = out ?? NDArray<bool>.create(targetShape, DType.boolean);
    result.setCellFlat(0, anyTrue);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result =
      out ??
      NDArray<bool>.zeros(
        targetShape,
        DType.boolean,
      ); // Pre-initialized to false
  if (out != null) {
    result.fill(false);
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  reduceRecursive<T, bool>(
    a,
    result,
    List<int>.filled(rank, 0),
    List<int>.filled(rank - 1, 0),
    normAxis,
    0,
    (current, val) => current || isTrueHelper(val),
    destStrides: squeezedDestStrides,
  );

  return result;
}

/// Computes the arithmetic mean of array elements along a specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num` or Complex).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final m = mean(a); // returns 0-D array containing 2.5
/// final m0 = mean(a, axis: 0); // returns NDArray [2.0, 3.0]
/// ```
///
/// Reference: [Arithmetic Mean](https://en.wikipedia.org/wiki/Arithmetic_mean)
NDArray<R> mean<R, T>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute mean of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write mean to a disposed output array.');
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  final expectedDType = a.dtype.isComplex ? DType.complex128 : DType.float64;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final DType<R> targetDType = expectedDType as DType<R>;

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result =
        out ??
        (targetDType.isComplex
            ? NDArray<Complex>.create(targetShape, DType.complex128)
                  as NDArray<R>
            : NDArray<Float64>.create(targetShape, DType.float64)
                  as NDArray<R>);
    if (size == 0) {
      if (targetDType.isComplex) {
        result.setCellFlat(0, Complex(double.nan, double.nan) as R);
      } else {
        result.setCellFlat(0, double.nan as R);
      }
      return result;
    }

    final ptr = a.isContiguous ? a.pointer : null;
    if (ptr != null) {
      dynamic acc;
      switch (a.dtype) {
        case DType.float64:
          acc = r_mean_double(ptr.cast(), size);
        case DType.float32:
          acc = r_mean_float_to_double(ptr.cast(), size);
        case DType.int64:
          acc = r_mean_int64_to_double(ptr.cast(), size);
        case DType.int32:
          acc = r_mean_int32_to_double(ptr.cast(), size);
        case DType.uint8:
          acc = r_mean_uint8_to_double(ptr.cast(), size);
        case DType.int16:
          acc = r_mean_int16_to_double(ptr.cast(), size);
        case DType.complex128:
          final c = r_mean_complex128(ptr.cast(), size);
          acc = Complex(c.r, c.i);
        case DType.complex64:
          final c = r_mean_complex64_to_complex128(ptr.cast(), size);
          acc = Complex(c.r, c.i);
        case DType.boolean:
          acc = r_mean_uint8_to_double(ptr.cast(), size);
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          acc = _r_stat_scalar_double_fallback(a, size, r_mean_double);
      }
      result.setCellFlat(0, acc as R);
      return result;
    }

    final copyA = a.copy();
    dynamic acc;
    switch (copyA.dtype) {
      case DType.float64:
        acc = r_mean_double(copyA.pointer.cast(), size);
      case DType.float32:
        acc = r_mean_float_to_double(copyA.pointer.cast(), size);
      case DType.int64:
        acc = r_mean_int64_to_double(copyA.pointer.cast(), size);
      case DType.int32:
        acc = r_mean_int32_to_double(copyA.pointer.cast(), size);
      case DType.uint8:
        acc = r_mean_uint8_to_double(copyA.pointer.cast(), size);
      case DType.int16:
        acc = r_mean_int16_to_double(copyA.pointer.cast(), size);
      case DType.complex128:
        final c = r_mean_complex128(copyA.pointer.cast(), size);
        acc = Complex(c.r, c.i);
      case DType.complex64:
        final c = r_mean_complex64_to_complex128(copyA.pointer.cast(), size);
        acc = Complex(c.r, c.i);
      case DType.boolean:
        acc = r_mean_uint8_to_double(copyA.pointer.cast(), size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        acc = _r_stat_scalar_double_fallback(copyA, size, r_mean_double);
    }
    copyA.dispose();
    result.setCellFlat(0, acc as R);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final result =
      out ??
      (targetDType.isComplex
          ? NDArray<Complex>.create(targetShape, DType.complex128) as NDArray<R>
          : NDArray<Float64>.create(targetShape, DType.float64) as NDArray<R>);

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_mean_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_mean_float_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int64:
        s_mean_int64_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int32:
        s_mean_int32_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.uint8:
        s_mean_uint8_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int16:
        s_mean_int16_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.complex128:
        s_mean_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.complex64:
        s_mean_complex64_to_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.boolean:
        s_mean_uint8_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result as NDArray<dynamic>,
          rank,
          normAxis,
          squeezedDestStrides,
          s_mean_double,
        );
    }
    return result;
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the standard deviation of array elements along a specified axis.
///
/// Standard deviation is a measure of the spread of a distribution. The standard deviation
/// is computed for the flattened array by default, otherwise over the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final s = std(a); // returns 0-D array containing sqrt(1.25)
/// ```
///
/// Reference: [Standard Deviation](https://en.wikipedia.org/wiki/Standard_deviation)
NDArray<Float64> std<T extends num>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  int ddof = 0,
  NDArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute standard deviation of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write standard deviation to a disposed output array.',
    );
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
    if (size <= ddof || size == 0) {
      result.setCellFlat(0, Float64(double.nan));
      return result;
    }

    final ptr = a.isContiguous ? a.pointer : null;
    if (ptr != null) {
      double acc = double.nan;
      switch (a.dtype) {
        case DType.float64:
          acc = r_std_double(ptr.cast(), size, ddof);
        case DType.float32:
          acc = r_std_float_to_double(ptr.cast(), size, ddof);
        case DType.int64:
          acc = r_std_int64_to_double(ptr.cast(), size, ddof);
        case DType.int32:
          acc = r_std_int32_to_double(ptr.cast(), size, ddof);
        case DType.uint8:
          acc = r_std_uint8_to_double(ptr.cast(), size, ddof);
        case DType.int16:
          acc = r_std_int16_to_double(ptr.cast(), size, ddof);
        case DType.boolean:
          acc = r_std_uint8_to_double(ptr.cast(), size, ddof);
        case DType.complex128:
        case DType.complex64:
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          acc = _r_stat_scalar_double_fallback(
            a,
            size,
            (p, s) => r_std_double(p, s, ddof),
          );
      }
      result.setCellFlat(0, Float64(acc));
      return result;
    }

    final copyA = a.copy();
    double acc = double.nan;
    switch (copyA.dtype) {
      case DType.float64:
        acc = r_std_double(copyA.pointer.cast(), size, ddof);
      case DType.float32:
        acc = r_std_float_to_double(copyA.pointer.cast(), size, ddof);
      case DType.int64:
        acc = r_std_int64_to_double(copyA.pointer.cast(), size, ddof);
      case DType.int32:
        acc = r_std_int32_to_double(copyA.pointer.cast(), size, ddof);
      case DType.uint8:
        acc = r_std_uint8_to_double(copyA.pointer.cast(), size, ddof);
      case DType.int16:
        acc = r_std_int16_to_double(copyA.pointer.cast(), size, ddof);
      case DType.boolean:
        acc = r_std_uint8_to_double(copyA.pointer.cast(), size, ddof);
      case DType.complex128:
      case DType.complex64:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        acc = _r_stat_scalar_double_fallback(
          copyA,
          size,
          (p, s) => r_std_double(p, s, ddof),
        );
    }
    copyA.dispose();
    result.setCellFlat(0, Float64(acc));
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_std_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.float32:
        s_std_float_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.int64:
        s_std_int64_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.int32:
        s_std_int32_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.uint8:
        s_std_uint8_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.int16:
        s_std_int16_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.boolean:
        s_std_uint8_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.complex128:
      case DType.complex64:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          (s, ss, d, ds, sh, r, ax) =>
              s_std_double(s, ss, d, ds, sh, r, ax, ddof),
        );
    }
    return result;
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the variance along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final v = nanvar(a); // returns 0-D array containing 0.6666666666666666
/// ```
NDArray<Float64> nanvar<T extends num>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanvar of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanvar to a disposed output array.');
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final m = nanmean(a, axis: axis, keepdims: true);

  if (axis == null) {
    var sumSqDiff = 0.0;
    final meanVal = m.getCell(List.filled(m.rank, 0)) as num;
    m.dispose();
    if (meanVal.toDouble().isNaN) {
      final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
      result.setCell(List.filled(targetShape.length, 0), Float64(double.nan));
      return result;
    }

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    var count = 0;
    for (var i = 0; i < size; i++) {
      final val = (a.getCellFlat(i) as num).toDouble();
      if (val.isNaN) continue;
      final diff = val - meanVal.toDouble();
      sumSqDiff += diff * diff;
      count++;
    }
    final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
    if (count == 0) {
      result.setCell(List.filled(targetShape.length, 0), Float64(double.nan));
    } else {
      result.setCell(
        List.filled(targetShape.length, 0),
        Float64(sumSqDiff / count),
      );
    }
    return result;
  } else {
    final diff = subtract(a, m);
    final sqDiff = multiply(diff, diff);

    m.dispose();
    diff.dispose();

    final res = nanmean<Float64>(
      sqDiff,
      axis: axis,
      keepdims: keepdims,
      out: out,
    );
    sqDiff.dispose();
    if (out != null) {
      return out;
    }
    final resultVal = NDArray<Float64>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    return resultVal;
  }
}

/// Computes the standard deviation along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final s = nanstd(a); // returns 0-D array containing sqrt(0.6666666666666666)
/// ```
NDArray<Float64> nanstd<T extends num>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanstd of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanstd to a disposed output array.');
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final v = nanvar(a, axis: axis, keepdims: keepdims);
  if (axis == null) {
    final stdVal = math.sqrt((v.scalar as num).toDouble());
    final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
    result.setCell(List.filled(targetShape.length, 0), Float64(stdVal));
    v.dispose();
    return result;
  } else {
    final res = sqrt(v, out: out);
    if (out != null) {
      v.dispose();
      return out;
    }
    final resultVal = NDArray<Float64>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    v.dispose();
    return resultVal;
  }
}

/// Computes the minimum of elements in the array.
///
/// **Edge cases:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> min<T extends num>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute min of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write min to a disposed output array.');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute min of an empty array.');
  }
  if (axis != null) {
    final normAxis = axis < 0 ? a.shape.length + axis : axis;
    if (normAxis < 0 || normAxis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[normAxis] == 0) {
      throw ArgumentError('Cannot compute min along axis $axis of size 0.');
    }
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic minVal;
    switch (temp.dtype) {
      case DType.float64:
        minVal = r_min_double(ptr.cast(), size);
      case DType.float32:
        minVal = r_min_float(ptr.cast(), size);
      case DType.int64:
        minVal = r_min_int64_t(ptr.cast(), size);
      case DType.int32:
        minVal = r_min_int32_t(ptr.cast(), size);
      case DType.uint8:
        minVal = r_min_uint8_t(ptr.cast(), size);
      case DType.int16:
        minVal = r_min_int16_t(ptr.cast(), size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        minVal = _r_stat_scalar_fallback(temp, size, r_min_double);
      case DType.complex128:
      case DType.complex64:
      case DType.boolean:
        break;
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCell(List.filled(targetShape.length, 0), minVal as T);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < squeezedDestStrides.length; i++) {
    cStridesRes[i] = squeezedDestStrides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_min_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_min_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int64:
        s_min_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int32:
        s_min_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.uint8:
        s_min_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int16:
        s_min_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          s_min_double,
        );
      case DType.complex128:
      case DType.complex64:
      case DType.boolean:
        break;
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the minimum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmin` function.
///
/// Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
///
/// **Preconditions:**
/// - [axis], if provided, must be a valid axis index within `[0, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - [UnsupportedError] if the array contains Complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// print(nanmin(a).scalar); // 1.0 (0-D array)
/// ```
NDArray<T> nanmin<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanmin of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanmin to a disposed output array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmin');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute nanmin of an empty array.');
  }
  if (axis != null) {
    final normAxis = axis < 0 ? a.shape.length + axis : axis;
    if (normAxis < 0 || normAxis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[normAxis] == 0) {
      throw ArgumentError('Cannot compute nanmin along axis $axis of size 0.');
    }
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic minVal;
    switch (temp.dtype) {
      case DType.float64:
        minVal = r_nanmin_double(ptr.cast(), size);
      case DType.float32:
        minVal = r_nanmin_float(ptr.cast(), size);
      case DType.int64:
        minVal = r_min_int64_t(ptr.cast(), size);
      case DType.int32:
        minVal = r_min_int32_t(ptr.cast(), size);
      case DType.uint8:
        minVal = r_min_uint8_t(ptr.cast(), size);
      case DType.int16:
        minVal = r_min_int16_t(ptr.cast(), size);
      case DType.boolean:
        minVal = r_min_uint8_t(ptr.cast(), size) != 0;
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        minVal = _r_stat_scalar_fallback(temp, size, r_nanmin_double);
      case DType.complex128:
      case DType.complex64:
        throw UnsupportedError('Unsupported dtype for nanmin: ${temp.dtype}');
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCell(List.filled(targetShape.length, 0), minVal as T);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < squeezedDestStrides.length; i++) {
    cStridesRes[i] = squeezedDestStrides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_nanmin_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_nanmin_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int64:
        s_min_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int32:
        s_min_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.uint8:
        s_min_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int16:
        s_min_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.boolean:
        s_min_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          s_nanmin_double,
        );
      case DType.complex128:
      case DType.complex64:
        throw UnsupportedError('Unsupported dtype for nanmin: ${a.dtype}');
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the maximum of elements in the array.
///
/// **Edge cases:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> max<T extends num>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute max of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write max to a disposed output array.');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute max of an empty array.');
  }
  if (axis != null) {
    final normAxis = axis < 0 ? a.shape.length + axis : axis;
    if (normAxis < 0 || normAxis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[normAxis] == 0) {
      throw ArgumentError('Cannot compute max along axis $axis of size 0.');
    }
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic maxVal;
    switch (temp.dtype) {
      case DType.float64:
        maxVal = r_max_double(ptr.cast(), size);
      case DType.float32:
        maxVal = r_max_float(ptr.cast(), size);
      case DType.int64:
        maxVal = r_max_int64_t(ptr.cast(), size);
      case DType.int32:
        maxVal = r_max_int32_t(ptr.cast(), size);
      case DType.uint8:
        maxVal = r_max_uint8_t(ptr.cast(), size);
      case DType.int16:
        maxVal = r_max_int16_t(ptr.cast(), size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        maxVal = _r_stat_scalar_fallback(temp, size, r_max_double);
      case DType.complex128:
      case DType.complex64:
      case DType.boolean:
        break;
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCell(List.filled(targetShape.length, 0), maxVal as T);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < squeezedDestStrides.length; i++) {
    cStridesRes[i] = squeezedDestStrides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_max_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_max_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int64:
        s_max_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int32:
        s_max_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.uint8:
        s_max_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int16:
        s_max_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          s_max_double,
        );
      case DType.complex128:
      case DType.complex64:
      case DType.boolean:
        break;
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the maximum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmax` function.
///
/// Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
///
/// **Preconditions:**
/// - [axis], if provided, must be a valid axis index within `[0, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - [UnsupportedError] if the array contains Complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// print(nanmax(a).scalar); // 3.0 (0-D array)
/// ```
NDArray<T> nanmax<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanmax of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanmax to a disposed output array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmax');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute nanmax of an empty array.');
  }
  if (axis != null) {
    final normAxis = axis < 0 ? a.shape.length + axis : axis;
    if (normAxis < 0 || normAxis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[normAxis] == 0) {
      throw ArgumentError('Cannot compute nanmax along axis $axis of size 0.');
    }
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic maxVal;
    switch (temp.dtype) {
      case DType.float64:
        maxVal = r_nanmax_double(ptr.cast(), size);
      case DType.float32:
        maxVal = r_nanmax_float(ptr.cast(), size);
      case DType.int64:
        maxVal = r_max_int64_t(ptr.cast(), size);
      case DType.int32:
        maxVal = r_max_int32_t(ptr.cast(), size);
      case DType.uint8:
        maxVal = r_max_uint8_t(ptr.cast(), size);
      case DType.int16:
        maxVal = r_max_int16_t(ptr.cast(), size);
      case DType.boolean:
        maxVal = r_max_uint8_t(ptr.cast(), size) != 0;
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        maxVal = _r_stat_scalar_fallback(temp, size, r_nanmax_double);
      case DType.complex128:
      case DType.complex64:
        throw UnsupportedError('Unsupported dtype for nanmax: ${temp.dtype}');
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCell(List.filled(targetShape.length, 0), maxVal as T);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < squeezedDestStrides.length; i++) {
    cStridesRes[i] = squeezedDestStrides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_nanmax_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_nanmax_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int64:
        s_max_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int32:
        s_max_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.uint8:
        s_max_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.int16:
        s_max_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.boolean:
        s_max_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          s_nanmax_double,
        );
      case DType.complex128:
      case DType.complex64:
        throw UnsupportedError('Unsupported dtype for nanmax: ${a.dtype}');
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the cumulative sum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<R> cumsum<T, R>(NDArray<T> a, {int? axis, NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumsum() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cumsum result to a disposed output array.');
  }

  final DType<dynamic> targetDType = a.dtype == DType.boolean
      ? DType.int32
      : a.dtype;
  final NDArray<R> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<R>.create([size], targetDType as DType<R>);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final en = NDEnumerate<T>(a);
    dynamic acc;
    var i = 0;
    while (en.moveNext()) {
      final val = en.value;
      final numVal = (val is bool) ? (val ? 1 : 0) : val;
      if (i == 0) {
        acc = numVal;
      } else {
        acc = (acc as dynamic) + numVal;
      }
      result.setCell([i++], acc as R);
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<R>.create(a.shape, targetDType as DType<R>);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.sum);
}

/// Computes the cumulative product of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<R> cumprod<T, R>(NDArray<T> a, {int? axis, NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumprod() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cumprod result to a disposed output array.');
  }

  final DType<dynamic> targetDType = a.dtype == DType.boolean
      ? DType.int32
      : a.dtype;
  final NDArray<R> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<R>.create([size], targetDType as DType<R>);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final en = NDEnumerate<T>(a);
    dynamic acc;
    var i = 0;
    while (en.moveNext()) {
      final val = en.value;
      final numVal = (val is bool) ? (val ? 1 : 0) : val;
      if (i == 0) {
        acc = numVal;
      } else {
        acc = (acc as dynamic) * numVal;
      }
      result.setCell([i++], acc as R);
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<R>.create(a.shape, targetDType as DType<R>);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.prod);
}

/// Computes the cumulative minimum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<T> cummin<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cummin() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cummin result to a disposed output array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final flatA = a.reshape([size]);
    cumOpFFI(flatA, 0, result, CumOpType.min);
    flatA.dispose();
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.min);
}

/// Computes the cumulative maximum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<T> cummax<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cummax() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cummax result to a disposed output array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final flatA = a.reshape([size]);
    cumOpFFI(flatA, 0, result, CumOpType.max);
    flatA.dispose();
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.max);
}

/// Computes the variance of array elements along a specified axis.
///
/// Variance is a measure of the spread of a distribution. The variance is computed for
/// the flattened array by default, otherwise over the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final v = variance(a); // returns 0-D array containing 1.25
/// ```
///
/// Reference: [Variance](https://en.wikipedia.org/wiki/Variance)
NDArray<Float64> variance<T extends num>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  int ddof = 0,
  NDArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute variance of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write variance to a disposed output array.');
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
    if (size <= ddof || size == 0) {
      result.setCellFlat(0, Float64(double.nan));
      return result;
    }

    final ptr = a.isContiguous ? a.pointer : null;
    if (ptr != null) {
      double acc = double.nan;
      switch (a.dtype) {
        case DType.float64:
          acc = r_var_double(ptr.cast(), size, ddof);
        case DType.float32:
          acc = r_var_float_to_double(ptr.cast(), size, ddof);
        case DType.int64:
          acc = r_var_int64_to_double(ptr.cast(), size, ddof);
        case DType.int32:
          acc = r_var_int32_to_double(ptr.cast(), size, ddof);
        case DType.uint8:
          acc = r_var_uint8_to_double(ptr.cast(), size, ddof);
        case DType.int16:
          acc = r_var_int16_to_double(ptr.cast(), size, ddof);
        case DType.boolean:
          acc = r_var_uint8_to_double(ptr.cast(), size, ddof);
        case DType.complex128:
        case DType.complex64:
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          acc = _r_stat_scalar_double_fallback(
            a,
            size,
            (p, s) => r_var_double(p, s, ddof),
          );
      }
      result.setCellFlat(0, Float64(acc));
      return result;
    }

    final copyA = a.copy();
    double acc = double.nan;
    switch (copyA.dtype) {
      case DType.float64:
        acc = r_var_double(copyA.pointer.cast(), size, ddof);
      case DType.float32:
        acc = r_var_float_to_double(copyA.pointer.cast(), size, ddof);
      case DType.int64:
        acc = r_var_int64_to_double(copyA.pointer.cast(), size, ddof);
      case DType.int32:
        acc = r_var_int32_to_double(copyA.pointer.cast(), size, ddof);
      case DType.uint8:
        acc = r_var_uint8_to_double(copyA.pointer.cast(), size, ddof);
      case DType.int16:
        acc = r_var_int16_to_double(copyA.pointer.cast(), size, ddof);
      case DType.boolean:
        acc = r_var_uint8_to_double(copyA.pointer.cast(), size, ddof);
      case DType.complex128:
      case DType.complex64:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        acc = _r_stat_scalar_double_fallback(
          copyA,
          size,
          (p, s) => r_var_double(p, s, ddof),
        );
    }
    copyA.dispose();
    result.setCellFlat(0, Float64(acc));
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_var_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.float32:
        s_var_float_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.int64:
        s_var_int64_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.int32:
        s_var_int32_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.uint8:
        s_var_uint8_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.int16:
        s_var_int16_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.boolean:
        s_var_uint8_to_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          ddof,
        );
      case DType.complex128:
      case DType.complex64:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          (s, ss, d, ds, sh, r, ax) =>
              s_var_double(s, ss, d, ds, sh, r, ax, ddof),
        );
    }
    return result;
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the variance of array elements along a specified axis. Alias for [variance].
NDArray<Float64> var_<T extends num>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  int ddof = 0,
  NDArray<Float64>? out,
}) => variance<T>(a, axis: axis, keepdims: keepdims, ddof: ddof, out: out);

/// Computes the arithmetic mean along a specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count, walking
///   coordinate strides and tracking counts dynamically.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0, 4.0], [2, 2], DType.float64);
/// final m = nanmean(a); // returns 0-D array containing 2.6666666666666665
/// ```
NDArray<R> nanmean<R extends Object>(
  NDArray a, {
  int? axis,
  bool keepdims = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanmean of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanmean to a disposed output array.');
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  final expectedDType = a.dtype.isComplex ? DType.complex128 : DType.float64;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final DType<R> targetDType = expectedDType as DType<R>;

  if (axis == null) {
    NDArray promotedA;
    if (a.dtype.isComplex || a.dtype.isFloating) {
      promotedA = a;
    } else {
      promotedA = promoteToDouble(a);
    }

    final iter = NDIter(promotedA);
    var sumVal = (targetDType.isComplex ? Complex(0, 0) : 0.0) as dynamic;
    var count = 0;
    while (iter.moveNext()) {
      final val = promotedA.getCellRaw(iter.index);
      if (val is double && val.isNaN) continue;
      if (val is Complex && (val.real.isNaN || val.imag.isNaN)) continue;
      sumVal += val;
      count++;
    }
    if (promotedA != a) {
      promotedA.dispose();
    }
    final NDArray<R> result;
    if (out != null) {
      result = out;
    } else {
      if (targetDType.isComplex) {
        result = NDArray<Complex>.create(
          targetShape,
          DType.complex128,
        ) as NDArray<R>;
      } else {
        result =
            NDArray<Float64>.create(targetShape, DType.float64) as NDArray<R>;
      }
    }

    if (count == 0) {
      result.setCellFlat(
        0,
        (targetDType.isComplex ? Complex(double.nan, double.nan) : double.nan)
            as R,
      );
    } else {
      result.setCellFlat(0, (sumVal / count) as R);
    }
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final NDArray<R> result;
  if (out != null) {
    result = out;
    result.fill(normalizeScalar(0, targetDType) as R);
  } else {
    if (targetDType.isComplex) {
      result =
          NDArray<Complex>.zeros(targetShape, DType.complex128) as NDArray<R>;
    } else {
      result = NDArray<Float64>.zeros(targetShape, DType.float64) as NDArray<R>;
    }
  }
  final counts = NDArray<int>.zeros(targetShape, DType.int32);

  if (targetDType.isComplex) {
    final promotedA = a.dtype.isComplex ? a : promoteToComplex(a);
    nanReduceRecursive<dynamic>(
      promotedA,
      result,
      counts,
      List<int>.filled(promotedA.shape.length, 0),
      List<int>.filled(targetShape.length, 0),
      normAxis,
      0,
      keepdims: keepdims,
    );
    if (promotedA != a) promotedA.dispose();
  } else {
    final promotedA = a.dtype.isFloating ? a : promoteToDouble(a);
    nanReduceRecursive<dynamic>(
      promotedA,
      result,
      counts,
      List<int>.filled(promotedA.shape.length, 0),
      List<int>.filled(targetShape.length, 0),
      normAxis,
      0,
      keepdims: keepdims,
    );
    if (promotedA != a) promotedA.dispose();
  }

  final iter = NDIter.broadcast2(result, counts);
  while (iter.moveNext()) {
    final coords = iter.coords;
    final c = counts.getCell(coords);
    if (c == 0) {
      result.setCell(
        coords,
        (targetDType.isComplex ? Complex(double.nan, double.nan) : double.nan)
            as R,
      );
    } else {
      result.setCell(coords, ((result.getCell(coords) as dynamic) / c) as R);
    }
  }
  counts.dispose();
  return result;
}

/// Computes the q-th quantile along the specified axis.
///
/// The quantile is a value between 0 and 1.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - [q] must be within `[0.0, 1.0]`.
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [q] is out of bounds or [axis] is out of bounds.
NDArray<Float64> quantile<T extends Object>(
  NDArray<T> a,
  double q, {
  int? axis,
  QuantileMethod method = QuantileMethod.linear,
  bool keepdims = false,
  NDArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute quantile of a disposed array.');
  }
  if (a.size == 0) {
    throw ArgumentError('Cannot compute quantile of an empty array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write quantile to a disposed output array.');
  }
  if (q < 0.0 || q > 1.0) {
    throw ArgumentError('Quantile q must be between 0.0 and 1.0. Got $q');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = _reductionTargetShape(a.shape, targetAxis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<Float64>.create([], DType.float64);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          result.setCellFlat(
            0,
            Float64(r_quantile_double(a.pointer.cast(), size, q, method.index)),
          );
          return result;
        case DType.float32:
          result.setCellFlat(
            0,
            Float64(r_quantile_float(a.pointer.cast(), size, q, method.index)),
          );
          return result;
        case DType.int64:
          result.setCellFlat(
            0,
            Float64(r_quantile_int64(a.pointer.cast(), size, q, method.index)),
          );
          return result;
        case DType.int32:
          result.setCellFlat(
            0,
            Float64(r_quantile_int32(a.pointer.cast(), size, q, method.index)),
          );
          return result;
        case DType.uint8:
          result.setCellFlat(
            0,
            Float64(r_quantile_uint8(a.pointer.cast(), size, q, method.index)),
          );
          return result;
        case DType.int16:
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.complex128:
        case DType.complex64:
        case DType.boolean:
          final flat = a.flatten();
          final resVal = r_quantile_helper(flat, flat.size, q, method.index);
          flat.dispose();
          result.setCellFlat(0, Float64(resVal));
          return result;
      }
    } else {
      final flat = a.flatten();
      final resVal = r_quantile_helper(flat, flat.size, q, method.index);
      flat.dispose();
      result.setCellFlat(0, Float64(resVal));
      return result;
    }
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<Float64>.zeros(targetShape, DType.float64);

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    final squeezedDestStrides = keepdims
        ? (List<int>.from(result.strides)..removeAt(targetAxis))
        : result.strides;
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_quantile_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
          q,
          method.index,
        );
      case DType.float32:
        s_quantile_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
          q,
          method.index,
        );
      case DType.int64:
        s_quantile_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
          q,
          method.index,
        );
      case DType.int32:
        s_quantile_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
          q,
          method.index,
        );
      case DType.uint8:
        s_quantile_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
          q,
          method.index,
        );
      case DType.int16:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.complex128:
      case DType.complex64:
      case DType.boolean:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          targetAxis,
          squeezedDestStrides,
          (s, ss, d, ds, sh, r, ax) =>
              s_quantile_double(s, ss, d, ds, sh, r, ax, q, method.index),
        );
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

double r_quantile_helper(NDArray a, int size, double q, int method) {
  switch (a.dtype) {
    case DType.float64:
      return r_quantile_double(a.pointer.cast(), size, q, method);
    case DType.float32:
      return r_quantile_float(a.pointer.cast(), size, q, method);
    case DType.int64:
      return r_quantile_int64(a.pointer.cast(), size, q, method);
    case DType.int32:
      return r_quantile_int32(a.pointer.cast(), size, q, method);
    case DType.uint8:
      return r_quantile_uint8(a.pointer.cast(), size, q, method);
    case DType.int16:
    case DType.float16:
    case DType.bfloat16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.complex128:
    case DType.complex64:
    case DType.boolean:
      final d = castNDArray(a, DType.float64);
      try {
        return r_quantile_double(d.pointer.cast(), size, q, method);
      } finally {
        d.dispose();
      }
  }
}

/// Computes the q-th percentile of the data along the specified axis.
///
/// The percentile is a value between 0 and 100.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - [q] must be within `[0.0, 100.0]`.
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [q] is out of bounds or [axis] is out of bounds.
NDArray<Float64> percentile<T extends Object>(
  NDArray<T> a,
  double q, {
  int? axis,
  QuantileMethod method = QuantileMethod.linear,
  bool keepdims = false,
  NDArray<Float64>? out,
}) {
  if (q < 0.0 || q > 100.0) {
    throw ArgumentError('Percentile q must be between 0.0 and 100.0. Got $q');
  }
  return quantile(
    a,
    q / 100.0,
    axis: axis,
    method: method,
    keepdims: keepdims,
    out: out,
  );
}

/// Computes the median along the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num` or Complex).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
NDArray<T> median<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute median of a disposed array.');
  }
  if (a.size == 0) {
    throw ArgumentError('Cannot compute median of an empty array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write median to a disposed output array.');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = _reductionTargetShape(a.shape, targetAxis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<T>.create([], a.dtype);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          result.setCellFlat(0, r_median_double(a.pointer.cast(), size) as T);
          return result;
        case DType.float32:
          result.setCellFlat(0, r_median_float(a.pointer.cast(), size) as T);
          return result;
        case DType.int64:
          result.setCellFlat(0, r_median_int64(a.pointer.cast(), size) as T);
          return result;
        case DType.int32:
          result.setCellFlat(0, r_median_int32(a.pointer.cast(), size) as T);
          return result;
        case DType.uint8:
          result.setCellFlat(0, r_median_uint8(a.pointer.cast(), size) as T);
          return result;
        case DType.complex128:
          final res = r_median_complex128(a.pointer.cast(), size);
          result.setCellFlat(0, Complex(res.r, res.i) as T);
          return result;
        case DType.complex64:
          final res = r_median_complex64(a.pointer.cast(), size);
          result.setCellFlat(0, Complex(res.r, res.i) as T);
          return result;
        case DType.int16:
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.boolean:
          final flat = a.flatten();
          final resVal = r_median_helper(flat, flat.size);
          flat.dispose();
          result.setCellFlat(0, resVal as T);
          return result;
      }
    } else {
      final flat = a.flatten();
      final resVal = r_median_helper(flat, flat.size);
      flat.dispose();
      result.setCellFlat(0, resVal as T);
      return result;
    }
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<T>.zeros(targetShape, a.dtype);

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    final squeezedDestStrides = keepdims
        ? (List<int>.from(result.strides)..removeAt(targetAxis))
        : result.strides;
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
    }
    for (var i = 0; i < squeezedDestStrides.length; i++) {
      cStridesRes[i] = squeezedDestStrides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_median_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.float32:
        s_median_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int64:
        s_median_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int32:
        s_median_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.uint8:
        s_median_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex128:
        s_median_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex64:
        s_median_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int16:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.boolean:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          targetAxis,
          squeezedDestStrides,
          s_median_double,
        );
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

Object r_median_helper(NDArray a, int size) {
  switch (a.dtype) {
    case DType.float64:
      return r_median_double(a.pointer.cast(), size);
    case DType.float32:
      return r_median_float(a.pointer.cast(), size);
    case DType.int64:
      return r_median_int64(a.pointer.cast(), size);
    case DType.int32:
      return r_median_int32(a.pointer.cast(), size);
    case DType.uint8:
      return r_median_uint8(a.pointer.cast(), size);
    case DType.complex128:
      final res = r_median_complex128(a.pointer.cast(), size);
      return Complex(res.r, res.i);
    case DType.complex64:
      final res = r_median_complex64(a.pointer.cast(), size);
      return Complex(res.r, res.i);
    case DType.int16:
    case DType.float16:
    case DType.bfloat16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.boolean:
      final d = castNDArray(a, DType.float64);
      try {
        final res = r_median_double(d.pointer.cast(), size);
        return normalizeScalar(res, a.dtype);
      } finally {
        d.dispose();
      }
  }
}

/// Computes the range of values (maximum - minimum) along the specified axis.
///
/// If [axis] is null, it computes the range over the entire array and returns a 0-D array.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If [out] is provided, it must not be disposed, and it must have the correct shape and dtype.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 6.0, 4.0], [2, 2], DType.float64);
/// final p = ptp(a); // returns 0-D array containing 5.0
/// final p0 = ptp(a, axis: 0); // returns NDArray [5.0, 2.0]
/// ```
NDArray<T> ptp<T extends num>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute ptp of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write ptp to a disposed output array.');
  }
  if (a.size == 0) {
    throw ArgumentError('Cannot compute ptp of an empty array.');
  }

  final resolvedAxis = axis != null && axis < 0 ? a.rank + axis : axis;
  if (resolvedAxis != null && (resolvedAxis < 0 || resolvedAxis >= a.rank)) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final targetShape = resolvedAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(resolvedAxis));

  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  return NDArray.scope(() {
    final mx = max(a, axis: resolvedAxis);
    final mn = min(a, axis: resolvedAxis);
    final res = subtract<T, T, T>(mx, mn, out: out);
    if (out == null) {
      res.detachToParentScope();
    }
    return res;
  });
}

/// Helper to cast an NDArray to a target DType using s_cast_generic.
NDArray<R> _castTo<R>(NDArray a, DType<R> targetDType) {
  if (a.isDisposed) {
    throw StateError('Cannot execute _castTo on a disposed array.');
  }
  if (a.dtype == targetDType) {
    return a.copy() as NDArray<R>;
  }

  final res = NDArray<R>.create(a.shape, targetDType);
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

    s_cast_generic(
      a.pointer.cast(),
      cStridesSrc,
      encodeDType(a.dtype),
      res.pointer.cast(),
      encodeDType(targetDType),
      cShape,
      ndim,
    );
  } finally {
    ScratchArena.reset(marker);
  }
  return res;
}

/// Computes the weighted average along the specified axis.
///
/// If [weights] is null, it is equivalent to [mean].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If [weights] is provided, it must not be disposed.
/// - If [weights] is 1-D, its length must match the shape of [a] along [axis].
///   - If [axis] is null, [weights] can only be 1-D if [a] is also 1-D.
/// - If [weights] is not 1-D, it must have the same shape as [a].
/// - If [out] is provided, it must not be disposed and must have correct shape and dtype.
///
/// **Returns:**
/// A record containing:
/// - `average`: The computed weighted average.
/// - `sumOfWeights`: The sum of weights along the axis, promoted to the result type [R],
///   if [returned] is true. Otherwise null.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final res = average(a, weights: w, returned: true);
/// print(res.average.scalar); // 3.0
/// print(res.sumOfWeights?.scalar); // 10.0
/// ```
({NDArray<R> average, NDArray<R>? sumOfWeights})
average<T extends num, W extends num, R extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<W>? weights,
  bool returned = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute average of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write average to a disposed output array.');
  }

  final resolvedAxis = axis != null && axis < 0 ? a.rank + axis : axis;
  if (resolvedAxis != null && (resolvedAxis < 0 || resolvedAxis >= a.rank)) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final targetShape = resolvedAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(resolvedAxis));

  if (out != null) {
    final DType expectedDType;
    if (weights == null) {
      expectedDType = a.dtype.isComplex ? DType.complex128 : DType.float64;
    } else {
      var resolved = resolveDType(a.dtype, weights.dtype);
      if (resolved.isInteger) {
        resolved = DType.float64;
      }
      expectedDType = resolved;
    }
    if (!listEquals(out.shape, targetShape) || out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (weights == null) {
    final avg = mean<R, T>(a, axis: resolvedAxis, out: out);
    if (!returned) {
      return (average: avg, sumOfWeights: null);
    }
    return NDArray.scope(() {
      final scale = resolvedAxis == null ? a.size : a.shape[resolvedAxis];
      final scaleScalar = NDArray.fromList([scale], [], DType.int64);
      final promoted = _castTo<R>(scaleScalar, avg.dtype);
      final scaleArray = broadcastTo<R>(promoted, avg.shape);
      scaleArray.detachToParentScope();
      return (average: avg, sumOfWeights: scaleArray);
    });
  }

  if (weights.isDisposed) {
    throw StateError('Cannot compute average with disposed weights.');
  }

  // Validate shapes
  if (weights.shape.length == 1) {
    if (resolvedAxis == null) {
      if (a.shape.length != 1) {
        throw ArgumentError(
          'If axis is null and weights is 1-D, input array must also be 1-D.',
        );
      }
      if (weights.size != a.size) {
        throw ArgumentError(
          'weights length (${weights.size}) must match a length (${a.size}).',
        );
      }
    } else {
      if (weights.shape[0] != a.shape[resolvedAxis]) {
        throw ArgumentError(
          'Length of 1-D weights (${weights.shape[0]}) must match shape of input along axis $resolvedAxis (${a.shape[resolvedAxis]}).',
        );
      }
    }
  } else {
    if (!listEquals(weights.shape, a.shape)) {
      throw ArgumentError(
        'Shape of weights ${weights.shape} must match shape of input ${a.shape} if weights is not 1-D.',
      );
    }
  }

  return NDArray.scope(() {
    NDArray<W> broadcastedWeights = weights;

    if (weights.shape.length == 1 && a.shape.length > 1) {
      final targetAxis = resolvedAxis!;
      final reshapedShape = List<int>.filled(a.shape.length, 1);
      reshapedShape[targetAxis] = weights.shape[0];
      broadcastedWeights = weights.reshape(reshapedShape);
    }

    final weighted_a = multiply<T, W, num>(a, broadcastedWeights);
    final weighted_sum = sum<num>(weighted_a, axis: resolvedAxis);
    final sum_of_weights = sum<num>(broadcastedWeights, axis: resolvedAxis);
    final avg = divide<num, num, R>(weighted_sum, sum_of_weights, out: out);

    NDArray<R>? sumOfWeightsResult;
    if (returned) {
      final promoted = _castTo<R>(sum_of_weights, avg.dtype);
      sumOfWeightsResult = broadcastTo<R>(promoted, avg.shape);
    }

    if (out == null) {
      avg.detachToParentScope();
    }
    sumOfWeightsResult?.detachToParentScope();

    return (average: avg, sumOfWeights: sumOfWeightsResult);
  });
}

/// Estimate a covariance matrix, given data and weights.
///
/// If [out] is provided, writes the resulting covariance matrix into it.
NDArray<Float64> cov<T extends num>(
  NDArray<T> m, {
  NDArray<T>? y,
  bool rowvar = true,
  bool bias = false,
  int? ddof,
  NDArray<int>? fweights,
  NDArray<num>? aweights,
  NDArray<Float64>? out,
}) {
  if (m.isDisposed) {
    throw StateError('Cannot compute covariance of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write covariance to a disposed output array.');
  }
  if (y != null && y.isDisposed) {
    throw StateError('Cannot compute covariance with a disposed array y.');
  }
  if (fweights != null && fweights.isDisposed) {
    throw StateError('fweights is disposed.');
  }
  if (aweights != null && aweights.isDisposed) {
    throw StateError('aweights is disposed.');
  }
  if (m.size == 0) {
    throw ArgumentError('m must not be empty.');
  }
  if (y != null && y.size == 0) {
    throw ArgumentError('y must not be empty.');
  }

  return NDArray.scope(() {
    NDArray<Float64> X;
    final mDouble = m.dtype == DType.float64
        ? m as NDArray<Float64>
        : promoteToDouble(m);

    if (mDouble.shape.isEmpty || mDouble.shape.length > 2) {
      throw ArgumentError('m must be 1D or 2D.');
    }

    final bool mIs1D = m.shape.length == 1;
    NDArray<Float64> prepM = mDouble;
    if (mIs1D) {
      prepM = mDouble.reshape([1, mDouble.size]);
    }

    NDArray<Float64>? prepY;
    bool yIs1D = false;
    if (y != null) {
      final yDouble = y.dtype == DType.float64
          ? y as NDArray<Float64>
          : promoteToDouble(y);
      if (yDouble.shape.isEmpty || yDouble.shape.length > 2) {
        throw ArgumentError('y must be 1D or 2D.');
      }
      yIs1D = y.shape.length == 1;
      prepY = yDouble;
      if (yIs1D) {
        prepY = yDouble.reshape([1, yDouble.size]);
      }
    }

    if (!rowvar) {
      if (!mIs1D) {
        prepM = prepM.transpose();
      }
      if (prepY != null) {
        if (prepY.shape[0] != 1) {
          prepY = prepY.transpose();
        }
      }
    }

    if (prepY != null) {
      X = concatenate([prepM, prepY], axis: 0);
    } else {
      X = prepM;
    }

    final N = X.shape[1];
    final fweightsLocal = fweights;
    final aweightsLocal = aweights;

    if (fweightsLocal != null) {
      if (fweightsLocal.shape.length != 1 || fweightsLocal.size != N) {
        throw ArgumentError(
          'fweights must be 1D and have size equal to number of observations ($N).',
        );
      }
      final minF = min(fweightsLocal).scalar;
      if (minF < 0) {
        throw ArgumentError('fweights must be non-negative.');
      }
    }
    if (aweightsLocal != null) {
      if (aweightsLocal.shape.length != 1 || aweightsLocal.size != N) {
        throw ArgumentError(
          'aweights must be 1D and have size equal to number of observations ($N).',
        );
      }
      final minA = min(aweightsLocal).scalar;
      if (minA < 0) {
        throw ArgumentError('aweights must be non-negative.');
      }
    }

    NDArray<Float64> w;
    NDArray<Float64> a;

    if (fweightsLocal == null && aweightsLocal == null) {
      w = NDArray<Float64>.ones([N], DType.float64);
      a = NDArray<Float64>.ones([N], DType.float64);
    } else {
      final fDouble = fweightsLocal != null
          ? promoteToDouble(fweightsLocal)
          : NDArray<Float64>.ones([N], DType.float64);
      final aDouble = aweightsLocal != null
          ? (aweightsLocal.dtype == DType.float64
                ? aweightsLocal as NDArray<Float64>
                : promoteToDouble(aweightsLocal))
          : NDArray<Float64>.ones([N], DType.float64);

      w = multiply<Float64, Float64, Float64>(fDouble, aDouble);
      a = aDouble;
    }

    final v1 = sum(w).scalar;
    final wTimesA = multiply<Float64, Float64, Float64>(w, a);
    final v2 = sum(wTimesA).scalar;

    final wReshaped = w.reshape([1, N]);
    final XTimesW = multiply<Float64, Float64, Float64>(X, wReshaped);
    final sumXW = sum(XTimesW, axis: 1);
    final meanVal = divide<Float64, Float64, Float64>(
      sumXW,
      NDArray<Float64>.scalar(Float64(v1), dtype: DType.float64),
    );

    final meanReshaped = meanVal.reshape([X.shape[0], 1]);
    final X_centered = subtract<Float64, Float64, Float64>(X, meanReshaped);

    final X_centered_weighted = multiply<Float64, Float64, Float64>(
      X_centered,
      wReshaped,
    );
    final X_centered_T = X_centered.transpose();
    final dotVal = matmul<Float64, Float64, Float64>(
      X_centered_weighted,
      X_centered_T,
    );

    final int resolvedDdof = ddof ?? (bias ? 0 : 1);
    final denominator = v1 * v1 - resolvedDdof * v2;
    final double fact;
    if (denominator == 0) {
      fact = double.nan;
    } else {
      fact = v1 / denominator;
    }

    final factArr = NDArray<Float64>.scalar(
      Float64(fact),
      dtype: DType.float64,
    );
    final result = multiply<Float64, Float64, Float64>(dotVal, factArr);

    final squeezed = result.squeeze();
    if (out != null) {
      return squeezed.copy(out: out);
    }
    return squeezed.detachToParentScope();
  });
}

/// Compute Pearson product-moment correlation coefficients.
///
/// If [out] is provided, writes the resulting correlation matrix into it.
NDArray<Float64> corrcoef<T extends num>(
  NDArray<T> m, {
  NDArray<T>? y,
  bool rowvar = true,
  NDArray<int>? fweights,
  NDArray<num>? aweights,
  NDArray<Float64>? out,
}) {
  if (m.isDisposed) {
    throw StateError(
      'Cannot compute correlation coefficient of a disposed array.',
    );
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write correlation coefficient to a disposed output array.',
    );
  }
  if (y != null && y.isDisposed) {
    throw StateError(
      'Cannot compute correlation coefficient with a disposed array y.',
    );
  }

  return NDArray.scope(() {
    final C = cov(
      m,
      y: y,
      rowvar: rowvar,
      fweights: fweights,
      aweights: aweights,
    );

    if (C.shape.isEmpty) {
      final val = C.scalar;
      final resVal = val == 0.0 ? double.nan : 1.0;
      if (out != null) {
        if (!listEquals(out.shape, []) || out.dtype != DType.float64) {
          throw ArgumentError('Incompatible out buffer shape or dtype.');
        }
        out.setCell([], Float64(resVal));
        return out;
      }
      return NDArray<Float64>.scalar(
        Float64(resVal),
        dtype: DType.float64,
      ).detachToParentScope();
    }

    final K = C.shape[0];
    final std = NDArray<Float64>.create([K], DType.float64);
    for (var i = 0; i < K; i++) {
      final variance = C.getCell([i, i]);
      std.setCellFlat(i, Float64(math.sqrt(variance)));
    }

    final stdCol = std.reshape([K, 1]);
    final stdRow = std.reshape([1, K]);
    final stdOuter = multiply<Float64, Float64, Float64>(stdCol, stdRow);

    final R = divide<Float64, Float64, Float64>(C, stdOuter, out: out);
    for (var i = 0; i < K; i++) {
      if (std.getCellFlat(i) == 0.0) {
        for (var j = 0; j < K; j++) {
          R.setCell([i, j], Float64(double.nan));
          R.setCell([j, i], Float64(double.nan));
        }
      }
    }

    if (out != null) {
      return out;
    }
    return R.detachToParentScope();
  });
}

/// Computes the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
///
/// Returns a new array with the results.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0, double.nan], [2, 2], DType.float64);
/// final s = nansum(a); // returns 4.0
/// ```
NDArray<T> nansum<T extends Object>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute nansum() on a disposed array.');
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    T acc;
    switch (a.dtype) {
      case DType.int32:
      case DType.int64:
        var sumVal = 0;
        final en = NDEnumerate<T>(a);
        while (en.moveNext()) {
          sumVal += en.value as int;
        }
        acc = sumVal as T;
      case DType.complex64:
      case DType.complex128:
        var sumVal = Complex(0.0, 0.0);
        final en = NDEnumerate<T>(a);
        while (en.moveNext()) {
          final val = en.value as Complex;
          if (val.real.isNaN || val.imag.isNaN) continue;
          sumVal += val;
        }
        acc = sumVal as T;
      default:
        var sumVal = 0.0;
        final en = NDEnumerate<T>(a);
        while (en.moveNext()) {
          final val = en.value as double;
          if (val.isNaN) continue;
          sumVal += val;
        }
        acc = sumVal as T;
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCell(List.filled(targetShape.length, 0), acc);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<T>.zeros(targetShape, a.dtype);
  if (out != null) {
    result.fill(normalizeScalar(0, a.dtype) as T);
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  reduceRecursive<T, T>(
    a,
    result,
    List<int>.filled(rank, 0),
    List<int>.filled(rank - 1, 0),
    normAxis,
    0,
    (current, val) {
      if (val is double && val.isNaN) return current;
      if (val is Complex && (val.real.isNaN || val.imag.isNaN)) return current;
      return ((current as dynamic) + val) as T;
    },
    destStrides: squeezedDestStrides,
  );
  return result;
}
