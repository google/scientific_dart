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
import '../ndarray_extensions_bindings.dart';
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

double _fastContiguousMinDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_min_double(ptr, size);

double _fastContiguousMaxDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_max_double(ptr, size);

double _fastContiguousPtpDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_max_double(ptr, size) - r_min_double(ptr, size);

double _fastContiguousNanminDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_nanmin_double(ptr, size);

double _fastContiguousNanmaxDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_nanmax_double(ptr, size);

double _fastContiguousNansumDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_nansum_double(ptr, size);

double _fastContiguousNansumFloat(ffi.Pointer<ffi.Float> ptr, int size) =>
    r_nansum_float(ptr, size);

double _fastContiguousNanmeanDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_nanmean_double(ptr, size);

double _fastContiguousNanmeanFloat(ffi.Pointer<ffi.Float> ptr, int size) =>
    r_nanmean_float(ptr, size);

double _fastContiguousNanvarDouble(ffi.Pointer<ffi.Double> ptr, int size) =>
    r_nanvar_double(ptr, size);

double _fastContiguousNanvarFloat(ffi.Pointer<ffi.Float> ptr, int size) =>
    r_nanvar_float(ptr, size);

double _r_stat_scalar_double_fallback<T extends DTypeTag>(
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

dynamic _r_stat_scalar_fallback<T extends DTypeTag>(
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

void _s_stat_strided_fallback<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> result,
  int rank,
  int normAxis,
  List<int> _,
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
    final resSqueezedStrides = (doubleRes.shape.length == rank)
        ? (List<int>.from(doubleRes.strides)..removeAt(normAxis))
        : doubleRes.strides;
    for (var i = 0; i < resSqueezedStrides.length; i++) {
      cStridesRes[i] = resSqueezedStrides[i];
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
    try {
      casted.copy(out: result);
    } finally {
      if (!identical(casted, doubleRes)) {
        casted.dispose();
      }
    }
  } finally {
    ScratchArena.reset(marker);
    if (!identical(doubleA, a)) {
      doubleA.dispose();
    }
    doubleRes.dispose();
  }
}

int _r_uint64_min(NDArray arr, int size) {
  final ptr = arr.pointer.cast<ffi.Uint64>();
  var minVal = ptr[0];
  for (var i = 1; i < size; i++) {
    final v = ptr[i];
    if (uint64Compare(v, minVal) < 0) {
      minVal = v;
    }
  }
  return minVal;
}

int _r_uint64_max(NDArray arr, int size) {
  final ptr = arr.pointer.cast<ffi.Uint64>();
  var maxVal = ptr[0];
  for (var i = 1; i < size; i++) {
    final v = ptr[i];
    if (uint64Compare(v, maxVal) > 0) {
      maxVal = v;
    }
  }
  return maxVal;
}

void _s_uint64_reduce(
  NDArray a,
  NDArray result,
  int targetAxis,
  List<int> squeezedDestStrides,
  int? initialValue,
  int Function(int acc, int val) op,
) {
  _s_generic_reduce<int>(
    a,
    result,
    targetAxis,
    squeezedDestStrides,
    initialValue,
    op,
  );
}

void _s_generic_reduce<T>(
  NDArray a,
  NDArray result,
  int targetAxis,
  List<int> squeezedDestStrides,
  T? initialValue,
  T Function(T acc, T val) op,
) {
  final rank = a.shape.length;
  final axisLen = a.shape[targetAxis];
  final outSize = result.size;
  final outShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final outRank = outShape.length;

  if (outRank == 0) {
    if (axisLen == 0) {
      result.setCellFlat(0, initialValue);
      return;
    }
    var acc = initialValue ?? (a.getCellFlat(0) as T);
    final startIdx = initialValue == null ? 1 : 0;
    for (var i = startIdx; i < axisLen; i++) {
      acc = op(acc, a.getCellFlat(i) as T);
    }
    result.setCellFlat(0, acc);
    return;
  }

  final outCoords = List<int>.filled(outRank, 0);
  final aCoords = List<int>.filled(rank, 0);

  for (var outIdx = 0; outIdx < outSize; outIdx++) {
    var c = 0;
    for (var d = 0; d < rank; d++) {
      if (d == targetAxis) continue;
      aCoords[d] = outCoords[c++];
    }

    T acc;
    if (axisLen == 0) {
      acc = initialValue as T;
    } else {
      aCoords[targetAxis] = 0;
      acc = initialValue ?? (a.getCell(aCoords) as T);
      final startIdx = initialValue == null ? 1 : 0;
      for (var i = startIdx; i < axisLen; i++) {
        aCoords[targetAxis] = i;
        acc = op(acc, a.getCell(aCoords) as T);
      }
    }

    var destOffset = result.offsetElements;
    for (var d = 0; d < outRank; d++) {
      destOffset += outCoords[d] * squeezedDestStrides[d];
    }
    result.setCellRaw(destOffset, acc);

    for (var d = outRank - 1; d >= 0; d--) {
      outCoords[d]++;
      if (outCoords[d] < outShape[d]) break;
      outCoords[d] = 0;
    }
  }
}

Complex _complexMin(Complex a, Complex b) {
  if (a.real.isNaN || a.imag.isNaN) return a;
  if (b.real.isNaN || b.imag.isNaN) return b;
  if (a.real < b.real) return a;
  if (a.real > b.real) return b;
  return a.imag <= b.imag ? a : b;
}

Complex _complexMax(Complex a, Complex b) {
  if (a.real.isNaN || a.imag.isNaN) return a;
  if (b.real.isNaN || b.imag.isNaN) return b;
  if (a.real > b.real) return a;
  if (a.real < b.real) return b;
  return a.imag >= b.imag ? a : b;
}

Complex _r_complex_min(NDArray a, int size) {
  var acc = a.getCellFlat(0) as Complex;
  for (var i = 1; i < size; i++) {
    acc = _complexMin(acc, a.getCellFlat(i) as Complex);
  }
  return acc;
}

Complex _r_complex_max(NDArray a, int size) {
  var acc = a.getCellFlat(0) as Complex;
  for (var i = 1; i < size; i++) {
    acc = _complexMax(acc, a.getCellFlat(i) as Complex);
  }
  return acc;
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

DType<R> _defaultAccumDType<R extends DTypeTag>(DType dtype) =>
    (switch (dtype) {
          DType.int64 ||
          DType.int32 ||
          DType.int16 ||
          DType.int8 ||
          DType.boolean => DType.int64,
          DType.uint64 ||
          DType.uint32 ||
          DType.uint16 ||
          DType.uint8 => DType.uint64,
          _ => dtype,
        })
        as DType<R>;

/// Computes the sum of elements in the array.
///
/// If [axis] is provided, sums along that axis and returns a new array.
/// Otherwise, sums all elements and returns a 0-D array containing the sum.
/// Signed integer and boolean inputs accumulate into [DType.int64] by default,
/// and unsigned integer inputs accumulate into [DType.uint64] by default
/// (matching NumPy). Use [sumAs] to specify a different accumulation dtype.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final s0 = sum(a, axis: 0); // Sum along rows
/// print(s0.toList()); // [4.0, 6.0]
/// ```
NDArray<R> sum<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, DTypeTag, R, DTypeTag>
  >
  a, {
  int? axis,
  bool keepdims = false,
  NDArray<R>? out,
}) => sumAs<DTypeTag, R>(
  a,
  _defaultAccumDType<R>(a.dtype),
  axis: axis,
  keepdims: keepdims,
  out: out,
);

/// Computes the sum of array elements over a given [axis], accumulating and
/// returning the result in the specified target [dtype].
///
/// Refer to [sum] for full details.
NDArray<R> sumAs<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  DType<R> dtype, {
  int? axis,
  bool keepdims = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute sum of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write sum to a disposed output array.');
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  final DType<R> effectiveDType = dtype;
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != effectiveDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        sumAs<T, R>(a, dtype, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  final NDArray workA;
  final bool needsDispose;
  if (a.dtype != effectiveDType) {
    workA = castNDArray(a, effectiveDType);
    needsDispose = true;
  } else {
    workA = a;
    needsDispose = false;
  }

  try {
    if (axis == null) {
      final size = workA.shape.isEmpty
          ? 1
          : workA.shape.reduce((x, y) => x * y);
      final result = out ?? NDArray<R>.create(targetShape, effectiveDType);
      if (size == 0) {
        if (effectiveDType.isComplex) {
          result.setCellFlat(0, Complex(0.0, 0.0));
        } else if (effectiveDType.isFloating) {
          result.setCellFlat(0, 0.0);
        } else if (effectiveDType == DType.boolean) {
          result.setCellFlat(0, false);
        } else {
          result.setCellFlat(0, 0);
        }
        return result;
      }

      final ptr = workA.isContiguous ? workA.pointer : null;
      if (ptr != null) {
        dynamic acc;
        switch (workA.dtype) {
          case DType.float64:
            acc = r_sum_double(ptr.cast(), size);
          case DType.float32:
            acc = r_sum_float(ptr.cast(), size);
          case DType.int64:
          case DType.uint64:
            acc = r_sum_int64(ptr.cast(), size);
          case DType.int32:
          case DType.uint32:
            acc = r_sum_int32(ptr.cast(), size);
          case DType.uint8:
          case DType.int8:
            acc = r_sum_uint8(ptr.cast(), size);
          case DType.int16:
          case DType.uint16:
            acc = r_sum_int16(ptr.cast(), size);
          case DType.complex128:
            final c = r_sum_complex128(ptr.cast(), size);
            acc = Complex(c.r, c.i);
          case DType.complex64:
            final c = r_sum_complex64(ptr.cast(), size);
            acc = Complex(c.r, c.i);
          case DType.boolean:
            acc = r_max_uint8_t(ptr.cast(), size) != 0;
          case DType.float16:
          case DType.bfloat16:
            acc = _r_stat_scalar_fallback(workA, size, r_sum_double);
        }
        result.setCellFlat(0, acc);
        return result;
      }

      final copyA = workA.copy();
      dynamic acc;
      try {
        switch (copyA.dtype) {
          case DType.float64:
            acc = r_sum_double(copyA.pointer.cast(), size);
          case DType.float32:
            acc = r_sum_float(copyA.pointer.cast(), size);
          case DType.int64:
          case DType.uint64:
            acc = r_sum_int64(copyA.pointer.cast(), size);
          case DType.int32:
          case DType.uint32:
            acc = r_sum_int32(copyA.pointer.cast(), size);
          case DType.uint8:
          case DType.int8:
            acc = r_sum_uint8(copyA.pointer.cast(), size);
          case DType.int16:
          case DType.uint16:
            acc = r_sum_int16(copyA.pointer.cast(), size);
          case DType.complex128:
            final c = r_sum_complex128(copyA.pointer.cast(), size);
            acc = Complex(c.r, c.i);
          case DType.complex64:
            final c = r_sum_complex64(copyA.pointer.cast(), size);
            acc = Complex(c.r, c.i);
          case DType.boolean:
            acc = r_max_uint8_t(copyA.pointer.cast(), size) != 0;
          case DType.float16:
          case DType.bfloat16:
            acc = _r_stat_scalar_fallback(copyA, size, r_sum_double);
        }
      } finally {
        copyA.dispose();
      }
      result.setCellFlat(0, acc);
      return result;
    }

    final rank = workA.shape.length;
    final normAxis = axis < 0 ? rank + axis : axis;
    if (normAxis < 0 || normAxis >= rank) {
      throw RangeError.range(normAxis, 0, rank - 1, 'axis');
    }

    final result = out ?? NDArray<R>.zeros(targetShape, effectiveDType);
    if (out != null) {
      result.fill(normalizeScalar(0, effectiveDType));
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
        cShape[i] = workA.shape[i];
        cStridesA[i] = workA.strides[i];
      }
      for (var i = 0; i < squeezedDestStrides.length; i++) {
        cStridesRes[i] = squeezedDestStrides[i];
      }

      switch (workA.dtype) {
        case DType.float64:
          s_sum_double(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.float32:
          s_sum_float(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.int64:
        case DType.uint64:
          s_sum_int64(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.int32:
        case DType.uint32:
          s_sum_int32(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.uint8:
        case DType.int8:
          s_sum_uint8(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.int16:
        case DType.uint16:
          s_sum_int16(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.complex128:
          s_sum_complex128(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.complex64:
          s_sum_complex64(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.boolean:
          s_max_uint8_t(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.float16:
        case DType.bfloat16:
          _s_stat_strided_fallback(
            workA,
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
  } finally {
    if (needsDispose) {
      workA.dispose();
    }
  }
}

/// Computes the product of array elements over a given [axis].
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If [axis] is provided, it must be within `[-a.shape.length, a.shape.length - 1]`.
///
/// **Throws:**
/// - [StateError] if [a] or [out] is disposed.
/// - [RangeError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] shape or dtype does not match the expected reduction shape/dtype.
///
/// **Performance considerations:**
/// - Uses direct C FFI reductions (`r_prod_*` for 1D/contiguous, `s_prod_*` for strided) for $O(N)$ time complexity.
/// - **Overflow Warning:** Signed integer and boolean inputs accumulate into [DType.int64] by default, and unsigned integer inputs accumulate into [DType.uint64] by default (matching NumPy). Use [prodAs] to specify a different accumulation dtype.
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
NDArray<R> prod<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, DTypeTag, R, DTypeTag>
  >
  a, {
  int? axis,
  bool keepdims = false,
  NDArray<R>? out,
}) => prodAs<DTypeTag, R>(
  a,
  _defaultAccumDType<R>(a.dtype),
  axis: axis,
  keepdims: keepdims,
  out: out,
);

/// Computes the product of array elements over a given [axis], accumulating
/// and returning the result in the specified target [dtype].
///
/// Refer to [prod] for full details.
NDArray<R> prodAs<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  DType<R> dtype, {
  int? axis,
  bool keepdims = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate product of disposed array');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write product result to disposed out array');
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  final DType<R> effectiveDType = dtype;
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != effectiveDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        prodAs<T, R>(a, dtype, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  final NDArray workA;
  final bool needsDispose;
  if (a.dtype != effectiveDType) {
    workA = castNDArray(a, effectiveDType);
    needsDispose = true;
  } else {
    workA = a;
    needsDispose = false;
  }

  try {
    final size = workA.shape.isEmpty ? 1 : workA.shape.reduce((x, y) => x * y);
    if (axis == null) {
      final result = out ?? NDArray<R>.zeros(targetShape, effectiveDType);
      if (size == 0) {
        if (effectiveDType.isComplex) {
          result.setCellFlat(0, Complex(1.0, 0.0));
        } else if (effectiveDType.isFloating) {
          result.setCellFlat(0, 1.0);
        } else if (effectiveDType == DType.boolean) {
          result.setCellFlat(0, true);
        } else {
          result.setCellFlat(0, 1);
        }
        return result;
      }

      final ptr = workA.isContiguous ? workA.pointer : null;
      if (ptr != null) {
        dynamic acc;
        switch (workA.dtype) {
          case DType.float64:
            acc = r_prod_double(ptr.cast(), size);
          case DType.float32:
            acc = r_prod_float(ptr.cast(), size);
          case DType.int64:
          case DType.uint64:
            acc = r_prod_int64(ptr.cast(), size);
          case DType.int32:
          case DType.uint32:
            acc = r_prod_int32(ptr.cast(), size);
          case DType.uint8:
          case DType.int8:
            acc = r_prod_uint8(ptr.cast(), size);
          case DType.int16:
          case DType.uint16:
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
            acc = _r_stat_scalar_fallback(workA, size, r_prod_double);
        }
        result.setCellFlat(0, acc);
        return result;
      }

      final copyA = workA.copy();
      dynamic acc;
      try {
        switch (copyA.dtype) {
          case DType.float64:
            acc = r_prod_double(copyA.pointer.cast(), size);
          case DType.float32:
            acc = r_prod_float(copyA.pointer.cast(), size);
          case DType.int64:
          case DType.uint64:
            acc = r_prod_int64(copyA.pointer.cast(), size);
          case DType.int32:
          case DType.uint32:
            acc = r_prod_int32(copyA.pointer.cast(), size);
          case DType.uint8:
          case DType.int8:
            acc = r_prod_uint8(copyA.pointer.cast(), size);
          case DType.int16:
          case DType.uint16:
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
            acc = _r_stat_scalar_fallback(copyA, size, r_prod_double);
        }
      } finally {
        copyA.dispose();
      }
      result.setCellFlat(0, acc);
      return result;
    }

    final rank = workA.shape.length;
    final normAxis = axis < 0 ? rank + axis : axis;
    if (normAxis < 0 || normAxis >= rank) {
      throw RangeError.range(normAxis, 0, rank - 1, 'axis');
    }

    final result = out ?? NDArray<R>.ones(targetShape, effectiveDType);
    if (out != null) {
      result.fill(normalizeScalar(1, effectiveDType));
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
        cShape[i] = workA.shape[i];
        cStridesA[i] = workA.strides[i];
      }
      for (var i = 0; i < squeezedDestStrides.length; i++) {
        cStridesRes[i] = squeezedDestStrides[i];
      }

      switch (workA.dtype) {
        case DType.float64:
          s_prod_double(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.float32:
          s_prod_float(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.int64:
        case DType.uint64:
          s_prod_int64(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.int32:
        case DType.uint32:
          s_prod_int32(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.uint8:
        case DType.int8:
          s_prod_uint8(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.int16:
        case DType.uint16:
          s_prod_int16(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.complex128:
          s_prod_complex128(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.complex64:
          s_prod_complex64(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.boolean:
          s_prod_uint8(
            workA.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.float16:
        case DType.bfloat16:
          _s_stat_strided_fallback(
            workA,
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
  } finally {
    if (needsDispose) {
      workA.dispose();
    }
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
NDArray<Boolean> all<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute all() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write all() result to a disposed output array.');
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Boolean>.create(out.shape, out.dtype);
        all<T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (axis == null) {
    var allTrue = true;
    final iter = NDIter(a);
    while (iter.moveNext()) {
      if (!isTrueHelper(a.getCellRaw(iter.index))) {
        allTrue = false;
        break;
      }
    }
    final result = out ?? NDArray<Boolean>.create(targetShape, DType.boolean);
    result.setCellFlat(0, allTrue);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<Boolean>.create(targetShape, DType.boolean);
  result.fill(true); // Initialize to true everywhere

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  reduceRecursive<T, Boolean>(
    a,
    result,
    List<int>.filled(rank, 0),
    List<int>.filled(rank - 1, 0),
    normAxis,
    0,
    (current, val) => (current as bool) && isTrueHelper(val),
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
NDArray<Boolean> any<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute any() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write any() result to a disposed output array.');
  }

  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Boolean>.create(out.shape, out.dtype);
        any<T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (axis == null) {
    var anyTrue = false;
    final iter = NDIter(a);
    while (iter.moveNext()) {
      if (isTrueHelper(a.getCellRaw(iter.index))) {
        anyTrue = true;
        break;
      }
    }
    final result = out ?? NDArray<Boolean>.create(targetShape, DType.boolean);
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
      NDArray<Boolean>.zeros(
        targetShape,
        DType.boolean,
      ); // Pre-initialized to false
  if (out != null) {
    result.fill(false);
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  reduceRecursive<T, Boolean>(
    a,
    result,
    List<int>.filled(rank, 0),
    List<int>.filled(rank - 1, 0),
    normAxis,
    0,
    (current, val) => (current as bool) || isTrueHelper(val),
    destStrides: squeezedDestStrides,
  );

  return result;
}

/// Computes the arithmetic mean of array elements along a specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag` or Complex).
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
NDArray<R> mean<R extends DTypeTag, T extends DTypeTag>(
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        mean<R, T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }
  final DType<R> targetDType = expectedDType as DType<R>;

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result =
        out ??
        (targetDType.isComplex
            ? NDArray<DTypeTag>.full(
                    targetShape,
                    Complex(double.nan, double.nan),
                    dtype: DType.complex128,
                  )
                  as NDArray<R>
            : NDArray<Float64>.full(
                    targetShape,
                    double.nan,
                    dtype: DType.float64,
                  )
                  as NDArray<R>);
    if (size == 0) {
      if (out != null) {
        if (targetDType.isComplex) {
          result.setCellFlat(0, Complex(double.nan, double.nan));
        } else {
          result.setCellFlat(0, double.nan);
        }
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
      result.setCellFlat(0, acc);
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
    result.setCellFlat(0, acc);
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
          ? NDArray<DTypeTag>.full(
                  targetShape,
                  Complex(double.nan, double.nan),
                  dtype: DType.complex128,
                )
                as NDArray<R>
          : NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64)
                as NDArray<R>);

  if (a.shape[normAxis] == 0) {
    if (out != null) {
      result.fill(
        (targetDType.isComplex ? Complex(double.nan, double.nan) : double.nan),
      );
    }
    return result;
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
          result as NDArray<DTypeTag>,
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
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
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
NDArray<Float64> std<T extends DTypeTag>(
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Float64>.create(out.shape, out.dtype);
        std<T>(a, axis: axis, keepdims: keepdims, ddof: ddof, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (a.dtype.isComplex) {
    return NDArray.scope(() {
      final v = variance<T>(
        a,
        axis: axis,
        keepdims: keepdims,
        ddof: ddof,
        out: out,
      );
      final res = sqrt(v, out: v);
      if (out == null) {
        res.detachToParentScope();
      }
      return res;
    });
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result =
        out ??
        NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);
    if (size == 0) {
      if (out != null) {
        result.setCellFlat(0, double.nan);
      }
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
      result.setCellFlat(0, acc);
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
    result.setCellFlat(0, acc);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final result =
      out ??
      NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);

  if (a.shape[normAxis] == 0) {
    if (out != null) {
      result.fill(double.nan);
    }
    return result;
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

NDArray<Float64> _computeNanvarGeneral<T extends DTypeTag>(
  NDArray<T> a,
  List<int> targetShape,
  int? axis,
  bool keepdims,
  int ddof,
  NDArray<Float64>? out,
) {
  final isComplex = a.dtype.isComplex;
  if (axis == null) {
    final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
    final size = a.size;
    if (size == 0 || size <= ddof) {
      result.setCellFlat(0, double.nan);
      return result;
    }
    if (isComplex) {
      var sumR = 0.0;
      var sumI = 0.0;
      var count = 0;
      final iter1 = NDIter(a);
      while (iter1.moveNext()) {
        final c = a.getCellRaw(iter1.index) as Complex;
        if (c.real.isNaN || c.imag.isNaN) continue;
        sumR += c.real;
        sumI += c.imag;
        count++;
      }
      if (count == 0 || count <= ddof) {
        result.setCellFlat(0, double.nan);
        return result;
      }
      final meanR = sumR / count;
      final meanI = sumI / count;
      var ssd = 0.0;
      final iter2 = NDIter(a);
      while (iter2.moveNext()) {
        final c = a.getCellRaw(iter2.index) as Complex;
        if (c.real.isNaN || c.imag.isNaN) continue;
        final dr = c.real - meanR;
        final di = c.imag - meanI;
        ssd += dr * dr + di * di;
      }
      result.setCellFlat(0, ssd / (count - ddof));
      return result;
    } else {
      var sumV = 0.0;
      var count = 0;
      final iter1 = NDIter(a);
      while (iter1.moveNext()) {
        final v = (a.getCellRaw(iter1.index) as num).toDouble();
        if (v.isNaN) continue;
        sumV += v;
        count++;
      }
      if (count == 0 || count <= ddof) {
        result.setCellFlat(0, double.nan);
        return result;
      }
      final meanV = sumV / count;
      var ssd = 0.0;
      final iter2 = NDIter(a);
      while (iter2.moveNext()) {
        final v = (a.getCellRaw(iter2.index) as num).toDouble();
        if (v.isNaN) continue;
        final dv = v - meanV;
        ssd += dv * dv;
      }
      result.setCellFlat(0, ssd / (count - ddof));
      return result;
    }
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result =
      out ??
      NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);
  final axisLen = a.shape[normAxis];
  if (axisLen == 0 || axisLen <= ddof) {
    if (out != null) {
      result.fill(double.nan);
    }
    return result;
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;
  final outSize = result.size;
  final outShape = List<int>.from(a.shape)..removeAt(normAxis);
  final outRank = outShape.length;
  final outCoords = List<int>.filled(outRank, 0);
  final aCoords = List<int>.filled(rank, 0);

  for (var outIdx = 0; outIdx < outSize; outIdx++) {
    var cIdx = 0;
    for (var d = 0; d < rank; d++) {
      if (d == normAxis) continue;
      aCoords[d] = outCoords[cIdx++];
    }

    double sliceVar;
    if (isComplex) {
      var sumR = 0.0;
      var sumI = 0.0;
      var count = 0;
      for (var i = 0; i < axisLen; i++) {
        aCoords[normAxis] = i;
        final c = a.getCell(aCoords) as Complex;
        if (c.real.isNaN || c.imag.isNaN) continue;
        sumR += c.real;
        sumI += c.imag;
        count++;
      }
      if (count == 0 || count <= ddof) {
        sliceVar = double.nan;
      } else {
        final meanR = sumR / count;
        final meanI = sumI / count;
        var ssd = 0.0;
        for (var i = 0; i < axisLen; i++) {
          aCoords[normAxis] = i;
          final c = a.getCell(aCoords) as Complex;
          if (c.real.isNaN || c.imag.isNaN) continue;
          final dr = c.real - meanR;
          final di = c.imag - meanI;
          ssd += dr * dr + di * di;
        }
        sliceVar = ssd / (count - ddof);
      }
    } else {
      var sumV = 0.0;
      var count = 0;
      for (var i = 0; i < axisLen; i++) {
        aCoords[normAxis] = i;
        final v = (a.getCell(aCoords) as num).toDouble();
        if (v.isNaN) continue;
        sumV += v;
        count++;
      }
      if (count == 0 || count <= ddof) {
        sliceVar = double.nan;
      } else {
        final meanV = sumV / count;
        var ssd = 0.0;
        for (var i = 0; i < axisLen; i++) {
          aCoords[normAxis] = i;
          final v = (a.getCell(aCoords) as num).toDouble();
          if (v.isNaN) continue;
          final dv = v - meanV;
          ssd += dv * dv;
        }
        sliceVar = ssd / (count - ddof);
      }
    }

    var destOffset = result.offsetElements;
    for (var d = 0; d < outRank; d++) {
      destOffset += outCoords[d] * squeezedDestStrides[d];
    }
    result.setCellRaw(destOffset, sliceVar);

    for (var d = outRank - 1; d >= 0; d--) {
      outCoords[d]++;
      if (outCoords[d] < outShape[d]) break;
      outCoords[d] = 0;
    }
  }

  return result;
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
NDArray<Float64> nanvar<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  int ddof = 0,
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Float64>.create(out.shape, out.dtype);
        nanvar<T>(a, axis: axis, keepdims: keepdims, ddof: ddof, out: temp);
        return temp.copy(out: out);
      });
    }
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    if (axis == null) {
      if (a.size == 0 || a.size <= ddof) {
        final result =
            out ??
            NDArray<Float64>.full(
              targetShape,
              double.nan,
              dtype: DType.float64,
            );
        if (out != null) {
          result.setCellFlat(0, double.nan);
        }
        return result;
      }
    } else {
      final rank = a.shape.length;
      final normAxis = axis < 0 ? rank + axis : axis;
      if (normAxis < 0 || normAxis >= rank) {
        throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
      }
      if (a.shape[normAxis] == 0 || a.shape[normAxis] <= ddof) {
        final result =
            out ??
            NDArray<Float64>.full(
              targetShape,
              double.nan,
              dtype: DType.float64,
            );
        if (out != null) {
          result.fill(double.nan);
        }
        return result;
      }
    }
    return variance<T>(a, axis: axis, keepdims: keepdims, ddof: ddof, out: out);
  }

  if (a.dtype.isComplex || ddof != 0) {
    return _computeNanvarGeneral(a, targetShape, axis, keepdims, ddof, out);
  }

  if (axis == null) {
    final size = a.size;
    final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
    if (size == 0) {
      result.setCellFlat(0, double.nan);
      return result;
    }
    final temp = a.isContiguous ? a : a.copy();
    double varVal;
    switch (temp.dtype) {
      case DType.float64:
        varVal = _fastContiguousNanvarDouble(temp.pointer.cast(), size);
      case DType.float32:
        varVal = _fastContiguousNanvarFloat(temp.pointer.cast(), size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        final d = castNDArray(temp, DType.float64);
        varVal = _fastContiguousNanvarDouble(d.pointer.cast(), size);
        d.dispose();
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    result.setCellFlat(0, varVal);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result =
      out ??
      NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);

  if (a.shape[normAxis] == 0) {
    if (out != null) {
      result.fill(double.nan);
    }
    return result;
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
        s_nanvar_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
        );
      case DType.float32:
        s_nanvar_float(
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
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        _s_stat_strided_fallback(
          a,
          result,
          rank,
          normAxis,
          squeezedDestStrides,
          s_nanvar_double,
        );
    }
    return result;
  } finally {
    ScratchArena.reset(marker);
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
NDArray<Float64> nanstd<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  int ddof = 0,
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Float64>.create(out.shape, out.dtype);
        nanstd<T>(a, axis: axis, keepdims: keepdims, ddof: ddof, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  final v = nanvar(a, axis: axis, keepdims: keepdims, ddof: ddof, out: out);
  if (axis == null) {
    v.setCellFlat(0, math.sqrt(v.getCellFlat(0)));
    return v;
  } else {
    final res = sqrt(v, out: v);
    return res;
  }
}

/// Computes the minimum of elements in the array.
///
/// **Edge cases:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> min<T extends DTypeTag>(
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        min<T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic minVal;
    switch (temp.dtype) {
      case DType.float64:
        minVal = _fastContiguousMinDouble(ptr.cast(), size);
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
      case DType.uint64:
        minVal = _r_uint64_min(temp, size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint32:
      case DType.uint16:
        minVal = _r_stat_scalar_fallback(temp, size, _fastContiguousMinDouble);
      case DType.complex128:
      case DType.complex64:
        minVal = _r_complex_min(temp, size);
      case DType.boolean:
        minVal = r_min_uint8_t(ptr.cast(), size) != 0;
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCellFlat(0, minVal);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

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
      case DType.uint64:
        _s_uint64_reduce(
          a,
          result,
          normAxis,
          squeezedDestStrides,
          null,
          (acc, val) => uint64Compare(val, acc) < 0 ? val : acc,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
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
        _s_generic_reduce<Complex>(
          a,
          result,
          normAxis,
          squeezedDestStrides,
          null,
          _complexMin,
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
NDArray<T> nanmin<T extends DTypeTag>(
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
  if (a.dtype == DType.boolean) {
    return min<T>(a, axis: axis, keepdims: keepdims, out: out);
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        nanmin<T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic minVal;
    switch (temp.dtype) {
      case DType.float64:
        minVal = _fastContiguousNanminDouble(ptr.cast(), size);
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
      case DType.uint64:
        minVal = _r_uint64_min(temp, size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint32:
      case DType.uint16:
        minVal = _r_stat_scalar_fallback(
          temp,
          size,
          _fastContiguousNanminDouble,
        );
      case DType.complex128:
      case DType.complex64:
        throw UnsupportedError('Unsupported dtype for nanmin: ${temp.dtype}');
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCellFlat(0, minVal);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

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
      case DType.uint64:
        _s_uint64_reduce(
          a,
          result,
          normAxis,
          squeezedDestStrides,
          null,
          (acc, val) => uint64Compare(val, acc) < 0 ? val : acc,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
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
        throw UnsupportedError("Complex numbers are not supported for nanmin.");
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
NDArray<T> max<T extends DTypeTag>(
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        max<T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic maxVal;
    switch (temp.dtype) {
      case DType.float64:
        maxVal = _fastContiguousMaxDouble(ptr.cast(), size);
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
      case DType.uint64:
        maxVal = _r_uint64_max(temp, size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint32:
      case DType.uint16:
        maxVal = _r_stat_scalar_fallback(temp, size, _fastContiguousMaxDouble);
      case DType.complex128:
      case DType.complex64:
        maxVal = _r_complex_max(temp, size);
      case DType.boolean:
        maxVal = r_max_uint8_t(ptr.cast(), size) != 0;
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCellFlat(0, maxVal);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

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
      case DType.uint64:
        _s_uint64_reduce(
          a,
          result,
          normAxis,
          squeezedDestStrides,
          null,
          (acc, val) => uint64Compare(val, acc) > 0 ? val : acc,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
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
        _s_generic_reduce<Complex>(
          a,
          result,
          normAxis,
          squeezedDestStrides,
          null,
          _complexMax,
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
NDArray<T> nanmax<T extends DTypeTag>(
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
  if (a.dtype == DType.boolean) {
    return max<T>(a, axis: axis, keepdims: keepdims, out: out);
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        nanmax<T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic maxVal;
    switch (temp.dtype) {
      case DType.float64:
        maxVal = _fastContiguousNanmaxDouble(ptr.cast(), size);
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
      case DType.uint64:
        maxVal = _r_uint64_max(temp, size);
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint32:
      case DType.uint16:
        maxVal = _r_stat_scalar_fallback(
          temp,
          size,
          _fastContiguousNanmaxDouble,
        );
      case DType.complex128:
      case DType.complex64:
        throw UnsupportedError('Unsupported dtype for nanmax: ${temp.dtype}');
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCellFlat(0, maxVal);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  final result = out ?? NDArray<T>.create(targetShape, a.dtype);

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
      case DType.uint64:
        _s_uint64_reduce(
          a,
          result,
          normAxis,
          squeezedDestStrides,
          null,
          (acc, val) => uint64Compare(val, acc) > 0 ? val : acc,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
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
        throw UnsupportedError("Complex numbers are not supported for nanmax.");
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
NDArray<R> cumsum<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, DTypeTag, DTypeTag, R>
  >
  a, {
  int? axis,
  NDArray<R>? out,
}) => cumsumAs<DTypeTag, R>(
  a,
  _defaultAccumDType<R>(a.dtype),
  axis: axis,
  out: out,
);

/// Computes the cumulative sum of array elements along [axis], accumulating
/// and returning the result in the specified target [dtype].
///
/// Refer to [cumsum] for full details.
NDArray<R> cumsumAs<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  DType<R> dtype, {
  int? axis,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumsum() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cumsum result to a disposed output array.');
  }

  final DType<R> targetDType = dtype;
  final NDArray<R> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<R>.create([size], targetDType);
    if (out != null) {
      if (!out.isWriteable ||
          !listEquals(out.shape, [size]) ||
          out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      if (sharesMemory(a, out)) {
        return NDArray.scope(() {
          final temp = NDArray<R>.create(out.shape, out.dtype);
          cumsumAs<T, R>(a, dtype, axis: axis, out: temp);
          return temp.copy(out: out);
        });
      }
    }

    final flatA = a.shape.length == 1 ? a : a.reshape([size]);
    cumOpFFI(flatA, 0, result, CumOpType.sum);
    if (!identical(flatA, a)) {
      flatA.dispose();
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

  result = out ?? NDArray<R>.create(a.shape, targetDType);
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        cumsumAs<T, R>(a, dtype, axis: axis, out: temp);
        return temp.copy(out: out);
      });
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
NDArray<R> cumprod<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, DTypeTag, DTypeTag, R>
  >
  a, {
  int? axis,
  NDArray<R>? out,
}) => cumprodAs<DTypeTag, R>(
  a,
  _defaultAccumDType<R>(a.dtype),
  axis: axis,
  out: out,
);

/// Computes the cumulative product of array elements along [axis],
/// accumulating and returning the result in the specified target [dtype].
///
/// Refer to [cumprod] for full details.
NDArray<R> cumprodAs<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  DType<R> dtype, {
  int? axis,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumprod() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cumprod result to a disposed output array.');
  }

  final DType<R> targetDType = dtype;
  final NDArray<R> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<R>.create([size], targetDType);
    if (out != null) {
      if (!out.isWriteable ||
          !listEquals(out.shape, [size]) ||
          out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      if (sharesMemory(a, out)) {
        return NDArray.scope(() {
          final temp = NDArray<R>.create(out.shape, out.dtype);
          cumprodAs<T, R>(a, dtype, axis: axis, out: temp);
          return temp.copy(out: out);
        });
      }
    }

    final flatA = a.shape.length == 1 ? a : a.reshape([size]);
    cumOpFFI(flatA, 0, result, CumOpType.prod);
    if (!identical(flatA, a)) {
      flatA.dispose();
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

  result = out ?? NDArray<R>.create(a.shape, targetDType);
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        cumprodAs<T, R>(a, dtype, axis: axis, out: temp);
        return temp.copy(out: out);
      });
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
NDArray<T> cummin<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
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
      if (!out.isWriteable ||
          !listEquals(out.shape, [size]) ||
          out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      if (sharesMemory(a, out)) {
        return NDArray.scope(() {
          final temp = NDArray<T>.create(out.shape, out.dtype);
          cummin<T>(a, axis: axis, out: temp);
          return temp.copy(out: out);
        });
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        cummin<T>(a, axis: axis, out: temp);
        return temp.copy(out: out);
      });
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
NDArray<T> cummax<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
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
      if (!out.isWriteable ||
          !listEquals(out.shape, [size]) ||
          out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      if (sharesMemory(a, out)) {
        return NDArray.scope(() {
          final temp = NDArray<T>.create(out.shape, out.dtype);
          cummax<T>(a, axis: axis, out: temp);
          return temp.copy(out: out);
        });
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        cummax<T>(a, axis: axis, out: temp);
        return temp.copy(out: out);
      });
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
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
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
NDArray<Float64> variance<T extends DTypeTag>(
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Float64>.create(out.shape, out.dtype);
        variance<T>(a, axis: axis, keepdims: keepdims, ddof: ddof, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (a.dtype.isComplex) {
    if (axis == null) {
      final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
      final result =
          out ??
          NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);
      if (size == 0) {
        if (out != null) {
          result.setCellFlat(0, double.nan);
        }
        return result;
      }
      var sumR = 0.0;
      var sumI = 0.0;
      final iter1 = NDIter(a);
      while (iter1.moveNext()) {
        final c = a.getCellRaw(iter1.index) as Complex;
        sumR += c.real;
        sumI += c.imag;
      }
      final meanR = sumR / size;
      final meanI = sumI / size;
      var ssd = 0.0;
      final iter2 = NDIter(a);
      while (iter2.moveNext()) {
        final c = a.getCellRaw(iter2.index) as Complex;
        final dr = c.real - meanR;
        final di = c.imag - meanI;
        ssd += dr * dr + di * di;
      }
      final val = size <= ddof
          ? ((ssd == 0.0 || ssd.isNaN) ? double.nan : double.infinity)
          : ssd / (size - ddof);
      result.setCellFlat(0, val);
      return result;
    }

    final rank = a.shape.length;
    final normAxis = axis < 0 ? rank + axis : axis;
    if (normAxis < 0 || normAxis >= rank) {
      throw RangeError.range(normAxis, 0, rank - 1, 'axis');
    }
    final result =
        out ??
        NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);
    final axisLen = a.shape[normAxis];
    if (axisLen == 0) {
      if (out != null) {
        result.fill(double.nan);
      }
      return result;
    }
    final squeezedDestStrides = keepdims
        ? (List<int>.from(result.strides)..removeAt(normAxis))
        : result.strides;
    final outSize = result.size;
    final outShape = List<int>.from(a.shape)..removeAt(normAxis);
    final outRank = outShape.length;
    final outCoords = List<int>.filled(outRank, 0);
    final aCoords = List<int>.filled(rank, 0);

    for (var outIdx = 0; outIdx < outSize; outIdx++) {
      var cIdx = 0;
      for (var d = 0; d < rank; d++) {
        if (d == normAxis) continue;
        aCoords[d] = outCoords[cIdx++];
      }
      var sumR = 0.0;
      var sumI = 0.0;
      for (var i = 0; i < axisLen; i++) {
        aCoords[normAxis] = i;
        final c = a.getCell(aCoords) as Complex;
        sumR += c.real;
        sumI += c.imag;
      }
      final meanR = sumR / axisLen;
      final meanI = sumI / axisLen;
      var ssd = 0.0;
      for (var i = 0; i < axisLen; i++) {
        aCoords[normAxis] = i;
        final c = a.getCell(aCoords) as Complex;
        final dr = c.real - meanR;
        final di = c.imag - meanI;
        ssd += dr * dr + di * di;
      }
      final sliceVar = axisLen <= ddof
          ? ((ssd == 0.0 || ssd.isNaN) ? double.nan : double.infinity)
          : ssd / (axisLen - ddof);
      var destOffset = result.offsetElements;
      for (var d = 0; d < outRank; d++) {
        destOffset += outCoords[d] * squeezedDestStrides[d];
      }
      result.setCellRaw(destOffset, sliceVar);
      for (var d = outRank - 1; d >= 0; d--) {
        outCoords[d]++;
        if (outCoords[d] < outShape[d]) break;
        outCoords[d] = 0;
      }
    }
    return result;
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result =
        out ??
        NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);
    if (size == 0) {
      if (out != null) {
        result.setCellFlat(0, double.nan);
      }
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
      result.setCellFlat(0, acc);
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
    result.setCellFlat(0, acc);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final result =
      out ??
      NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64);

  if (a.shape[normAxis] == 0) {
    if (out != null) {
      result.fill(double.nan);
    }
    return result;
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
NDArray<Float64> var_<T extends DTypeTag>(
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
NDArray<R> nanmean<R extends DTypeTag>(
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        nanmean<R>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }
  final DType<R> targetDType = expectedDType as DType<R>;

  if (axis == null) {
    final size = a.size;
    if (!targetDType.isComplex && size > 0) {
      final temp = a.isContiguous ? a : a.copy();
      double meanVal;
      switch (temp.dtype) {
        case DType.float64:
          meanVal = _fastContiguousNanmeanDouble(temp.pointer.cast(), size);
        case DType.float32:
          meanVal = _fastContiguousNanmeanFloat(temp.pointer.cast(), size);
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          final d = castNDArray(temp, DType.float64);
          meanVal = _fastContiguousNanmeanDouble(d.pointer.cast(), size);
          d.dispose();
      }
      if (!identical(temp, a)) {
        temp.dispose();
      }
      final NDArray<R> result =
          out ??
          (NDArray<Float64>.create(targetShape, DType.float64) as NDArray<R>);
      result.setCellFlat(0, meanVal);
      return result;
    }

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
        result =
            NDArray<DTypeTag>.create(targetShape, DType.complex128)
                as NDArray<R>;
      } else {
        result =
            NDArray<Float64>.create(targetShape, DType.float64) as NDArray<R>;
      }
    }

    if (count == 0) {
      result.setCellFlat(
        0,
        (targetDType.isComplex ? Complex(double.nan, double.nan) : double.nan),
      );
    } else {
      result.setCellFlat(0, (sumVal / count));
    }
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  if (!targetDType.isComplex) {
    final result =
        out ??
        (NDArray<Float64>.full(targetShape, double.nan, dtype: DType.float64)
            as NDArray<R>);
    if (a.shape[normAxis] == 0) {
      if (out != null) {
        result.fill(double.nan);
      }
      return result;
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
          s_nanmean_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.float32:
          s_nanmean_float(
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
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          _s_stat_strided_fallback(
            a,
            result as NDArray<DTypeTag>,
            rank,
            normAxis,
            squeezedDestStrides,
            s_nanmean_double,
          );
      }
      return result;
    } finally {
      ScratchArena.reset(marker);
    }
  }

  final NDArray<R> result;
  if (out != null) {
    result = out;
    result.fill(normalizeScalar(0, targetDType));
  } else {
    result =
        NDArray<DTypeTag>.zeros(targetShape, DType.complex128) as NDArray<R>;
  }
  final counts = NDArray<DTypeTag>.zeros(targetShape, DType.int32);

  final promotedA = a.dtype.isComplex ? a : promoteToComplex(a);
  nanReduceRecursive<DTypeTag>(
    promotedA,
    result,
    counts,
    List<int>.filled(promotedA.shape.length, 0),
    List<int>.filled(targetShape.length, 0),
    normAxis,
    0,
    keepdims: keepdims,
  );
  if (!identical(promotedA, a)) promotedA.dispose();

  final iter = NDIter.broadcast2(result, counts);
  while (iter.moveNext()) {
    final coords = iter.coords;
    final c = counts.getCell(coords);
    if (c == 0) {
      result.setCell(coords, Complex(double.nan, double.nan));
    } else {
      result.setCell(coords, ((result.getCell(coords) as dynamic) / c));
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
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - [q] must be within `[0.0, 1.0]`.
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [q] is out of bounds or [axis] is out of bounds.
NDArray<Float64> quantile<T extends DTypeTag>(
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
  if (a.dtype.isComplex) {
    throw UnsupportedError('Quantiles are not supported for complex arrays.');
  }
  if (a.size == 0) {
    throw ArgumentError('Cannot compute quantile of an empty array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write quantile to a disposed output array.');
  }
  if (q.isNaN || q < 0.0 || q > 1.0) {
    throw ArgumentError('Quantile q must be between 0.0 and 1.0. Got $q');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = _reductionTargetShape(a.shape, targetAxis, keepdims);
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Float64>.create(out.shape, out.dtype);
        quantile<T>(
          a,
          q,
          axis: axis,
          method: method,
          keepdims: keepdims,
          out: temp,
        );
        return temp.copy(out: out);
      });
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<Float64>.create(targetShape, DType.float64);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          result.setCellFlat(
            0,
            r_quantile_double(a.pointer.cast(), size, q, method.index),
          );
          return result;
        case DType.float32:
          result.setCellFlat(
            0,
            r_quantile_float(a.pointer.cast(), size, q, method.index),
          );
          return result;
        case DType.int64:
          result.setCellFlat(
            0,
            r_quantile_int64(a.pointer.cast(), size, q, method.index),
          );
          return result;
        case DType.int32:
          result.setCellFlat(
            0,
            r_quantile_int32(a.pointer.cast(), size, q, method.index),
          );
          return result;
        case DType.uint8:
          result.setCellFlat(
            0,
            r_quantile_uint8(a.pointer.cast(), size, q, method.index),
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
          result.setCellFlat(0, resVal);
          return result;
      }
    } else {
      final flat = a.flatten();
      final resVal = r_quantile_helper(flat, flat.size, q, method.index);
      flat.dispose();
      result.setCellFlat(0, resVal);
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
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - [q] must be within `[0.0, 100.0]`.
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [q] is out of bounds or [axis] is out of bounds.
NDArray<Float64> percentile<T extends DTypeTag>(
  NDArray<T> a,
  double q, {
  int? axis,
  QuantileMethod method = QuantileMethod.linear,
  bool keepdims = false,
  NDArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute percentile of a disposed array.');
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError('Percentiles are not supported for complex arrays.');
  }
  if (q.isNaN || q < 0.0 || q > 100.0) {
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
/// - Input array [a] elements must be numeric (`T extends DTypeTag` or Complex).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
DType<R> _medianDType<R extends DTypeTag>(DType inputDType) {
  final DType resolved = switch (inputDType) {
    DType.float32 => DType.float32,
    DType.complex128 => DType.complex128,
    DType.complex64 => DType.complex64,
    DType.float64 ||
    DType.float16 ||
    DType.bfloat16 ||
    DType.int64 ||
    DType.int32 ||
    DType.int16 ||
    DType.int8 ||
    DType.uint64 ||
    DType.uint32 ||
    DType.uint16 ||
    DType.uint8 ||
    DType.boolean => DType.float64,
  };
  return resolved as DType<R>;
}

/// Computes the median along the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag` or Complex).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
NDArray<R> median<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>
  >
  a, {
  int? axis,
  bool keepdims = false,
  NDArray<R>? out,
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
  if (targetAxis != null && (targetAxis < 0 || targetAxis >= a.shape.length)) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final targetDType = _medianDType<R>(a.dtype);
  final targetShape = _reductionTargetShape(a.shape, targetAxis, keepdims);
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != targetDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        median<R>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (a.dtype != targetDType) {
    final promoted = castNDArray<Float64>(a, DType.float64);
    try {
      return median<Float64>(
            promoted,
            axis: axis,
            keepdims: keepdims,
            out: out as NDArray<Float64>?,
          )
          as NDArray<R>;
    } finally {
      promoted.dispose();
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<R>.create(targetShape, targetDType);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          result.setCellFlat(0, r_median_double(a.pointer.cast(), size));
          return result;
        case DType.float32:
          result.setCellFlat(0, r_median_float(a.pointer.cast(), size));
          return result;
        case DType.complex128:
          final res = r_median_complex128(a.pointer.cast(), size);
          result.setCellFlat(0, Complex(res.r, res.i));
          return result;
        case DType.complex64:
          final res = r_median_complex64(a.pointer.cast(), size);
          result.setCellFlat(0, Complex(res.r, res.i));
          return result;
        case DType.int64:
        case DType.int32:
        case DType.uint8:
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
          result.setCellFlat(0, resVal);
          return result;
      }
    } else {
      final flat = a.flatten();
      final resVal = r_median_helper(flat, flat.size);
      flat.dispose();
      result.setCellFlat(0, resVal);
      return result;
    }
  }

  final result = out ?? NDArray<R>.zeros(targetShape, targetDType);

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
      case DType.int64:
      case DType.int32:
      case DType.uint8:
      case DType.uint64:
      case DType.int16:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
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
    case DType.complex128:
      final res = r_median_complex128(a.pointer.cast(), size);
      return Complex(res.r, res.i);
    case DType.complex64:
      final res = r_median_complex64(a.pointer.cast(), size);
      return Complex(res.r, res.i);
    case DType.int64:
    case DType.int32:
    case DType.uint8:
    case DType.uint64:
    case DType.int16:
    case DType.float16:
    case DType.bfloat16:
    case DType.int8:
    case DType.uint32:
    case DType.uint16:
    case DType.boolean:
      final d = castNDArray(a, DType.float64);
      try {
        return r_median_double(d.pointer.cast(), size);
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
NDArray<T> ptp<T extends DTypeTag>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        ptp<T>(a, axis: axis, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (resolvedAxis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic ptpVal;
    switch (temp.dtype) {
      case DType.float64:
        ptpVal = _fastContiguousPtpDouble(ptr.cast(), size);
      case DType.float32:
        final mn = r_min_float(ptr.cast(), size);
        final mx = r_max_float(ptr.cast(), size);
        ptpVal = mx - mn;
      case DType.int64:
        final mn = r_min_int64_t(ptr.cast(), size);
        final mx = r_max_int64_t(ptr.cast(), size);
        ptpVal = mx - mn;
      case DType.int32:
        final mn = r_min_int32_t(ptr.cast(), size);
        final mx = r_max_int32_t(ptr.cast(), size);
        ptpVal = mx - mn;
      case DType.uint8:
        final mn = r_min_uint8_t(ptr.cast(), size);
        final mx = r_max_uint8_t(ptr.cast(), size);
        ptpVal = mx - mn;
      case DType.int16:
        final mn = r_min_int16_t(ptr.cast(), size);
        final mx = r_max_int16_t(ptr.cast(), size);
        ptpVal = mx - mn;
      case DType.uint64:
        final mn = _r_uint64_min(temp, size);
        final mx = _r_uint64_max(temp, size);
        ptpVal = mx - mn;
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint32:
      case DType.uint16:
        ptpVal = _r_stat_scalar_fallback(temp, size, _fastContiguousPtpDouble);
      case DType.complex128:
      case DType.complex64:
        throw ArgumentError('ptp is not supported for complex dtypes.');
      case DType.boolean:
        final mn = r_min_uint8_t(ptr.cast(), size);
        final mx = r_max_uint8_t(ptr.cast(), size);
        ptpVal = mx - mn;
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    result.setCellFlat(0, ptpVal);
    return result;
  }

  return NDArray.scope(() {
    final mx = max(a, axis: resolvedAxis);
    final mn = min(a, axis: resolvedAxis);
    final res = subtract<T>(mx, mn, out: out);
    if (out == null) {
      res.detachToParentScope();
    }
    return res;
  });
}

/// Helper to cast an NDArray to a target DType using s_cast_generic.
NDArray<R> _castTo<R extends DTypeTag>(NDArray a, DType<R> targetDType) {
  if (a.isDisposed) {
    throw StateError('Cannot execute _castTo on a disposed array.');
  }
  if (a.dtype == targetDType) {
    final res = NDArray<R>.create(a.shape, targetDType);
    a.copy(out: res);
    return res;
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
average<T extends DTypeTag, W extends DTypeTag, R extends DTypeTag>(
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
  if (weights != null && weights.isDisposed) {
    throw StateError('Cannot compute average with disposed weights.');
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
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out) ||
        (weights != null && sharesMemory(weights, out))) {
      return NDArray.scope(() {
        final temp = NDArray<R>.create(out.shape, out.dtype);
        final res = average<T, W, R>(
          a,
          axis: axis,
          weights: weights,
          returned: returned,
          out: temp,
        );
        temp.copy(out: out);
        res.sumOfWeights?.detachToParentScope();
        return (average: out, sumOfWeights: res.sumOfWeights);
      });
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
      final scaleArray = broadcastTo<R>(promoted, avg.shape).copy();
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

    var accumDType = resolveDType(a.dtype, weights.dtype);
    if (accumDType.isInteger || accumDType == DType.boolean) {
      accumDType = DType.float64;
    }
    final aCast = a.dtype == accumDType ? a : castNDArray(a, accumDType);
    final wCast = broadcastedWeights.dtype == accumDType
        ? broadcastedWeights
        : castNDArray(broadcastedWeights, accumDType);
    final weighted_a = multiply<DTypeTag>(aCast, wCast);
    final weighted_sum = sumAs<DTypeTag, DTypeTag>(
      weighted_a,
      accumDType,
      axis: resolvedAxis,
    );
    final sum_of_weights = sumAs<DTypeTag, DTypeTag>(
      wCast,
      accumDType,
      axis: resolvedAxis,
    );
    var divDType = resolveDType(weighted_sum.dtype, sum_of_weights.dtype);
    if (divDType.isInteger || divDType == DType.boolean) {
      divDType = DType.float64;
    }
    final avg = divideAs<DTypeTag, DTypeTag, R>(
      weighted_sum,
      sum_of_weights,
      divDType as DType<R>,
      out: out,
    );

    NDArray<R>? sumOfWeightsResult;
    if (returned) {
      final promoted = _castTo<R>(sum_of_weights, avg.dtype);
      sumOfWeightsResult = listEquals(promoted.shape, avg.shape)
          ? promoted
          : broadcastTo<R>(promoted, avg.shape).copy();
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
NDArray<Float64> cov<T extends DTypeTag>(
  NDArray<T> m, {
  NDArray<T>? y,
  bool rowvar = true,
  bool bias = false,
  int? ddof,
  NDArray<DTypeTag>? fweights,
  NDArray<DTypeTag>? aweights,
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
      final minF = min(fweightsLocal).scalar as num;
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
      final minA = min(aweightsLocal).scalar as num;
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

      w = multiply<Float64>(fDouble, aDouble);
      a = aDouble;
    }

    final v1 = sum(w).scalar;
    final wTimesA = multiply<Float64>(w, a);
    final v2 = sum(wTimesA).scalar;

    final wReshaped = w.reshape([1, N]);
    final XTimesW = multiply<Float64>(X, wReshaped);
    final sumXW = sum<Float64>(XTimesW, axis: 1);
    final meanVal = divide<Float64, Float64, Float64>(
      sumXW,
      NDArray<Float64>.scalar(v1, dtype: DType.float64),
    );

    final meanReshaped = meanVal.reshape([X.shape[0], 1]);
    final X_centered = subtract<Float64>(X, meanReshaped);

    final X_centered_weighted = multiply<Float64>(X_centered, wReshaped);
    final X_centered_T = X_centered.transpose();
    final dotVal = matmul<Float64>(X_centered_weighted, X_centered_T);

    final int resolvedDdof = ddof ?? (bias ? 0 : 1);
    final v1Num = v1 as num;
    final v2Num = v2 as num;
    final denominator = v1Num * v1Num - resolvedDdof * v2Num;
    final double fact;
    if (denominator == 0) {
      fact = double.nan;
    } else {
      fact = v1Num / denominator;
    }

    final factArr = NDArray<Float64>.scalar(fact, dtype: DType.float64);
    final result = multiply<Float64>(dotVal, factArr);

    final squeezed = result.squeeze();
    if (out != null) {
      return squeezed.copy(out: out);
    }
    return squeezed.copy().detachToParentScope();
  });
}

/// Compute Pearson product-moment correlation coefficients.
///
/// If [out] is provided, writes the resulting correlation matrix into it.
NDArray<Float64> corrcoef<T extends DTypeTag>(
  NDArray<T> m, {
  NDArray<T>? y,
  bool rowvar = true,
  NDArray<DTypeTag>? fweights,
  NDArray<DTypeTag>? aweights,
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
        if (!out.isWriteable ||
            !listEquals(out.shape, []) ||
            out.dtype != DType.float64) {
          throw ArgumentError('Incompatible out buffer shape or dtype.');
        }
        out.setCell([], resVal);
        return out;
      }
      return NDArray<Float64>.scalar(
        resVal,
        dtype: DType.float64,
      ).detachToParentScope();
    }

    final K = C.shape[0];
    final std = NDArray<Float64>.create([K], DType.float64);
    for (var i = 0; i < K; i++) {
      final variance = C.getCell([i, i]);
      std.setCellFlat(i, math.sqrt(variance));
    }

    final stdCol = std.reshape([K, 1]);
    final stdRow = std.reshape([1, K]);
    final stdOuter = multiply<Float64>(stdCol, stdRow);

    final R = divide<Float64, Float64, Float64>(C, stdOuter, out: out);
    for (var i = 0; i < K; i++) {
      if (std.getCellFlat(i) == 0.0) {
        for (var j = 0; j < K; j++) {
          R.setCell([i, j], double.nan);
          R.setCell([j, i], double.nan);
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
NDArray<T> nansum<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute nansum() on a disposed array.');
  }
  if (a.dtype.isInteger || a.dtype == DType.boolean) {
    return sumAs<T, T>(a, a.dtype, axis: axis, keepdims: keepdims, out: out);
  }
  final targetShape = _reductionTargetShape(a.shape, axis, keepdims);
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<T>.create(out.shape, out.dtype);
        nansum<T>(a, axis: axis, keepdims: keepdims, out: temp);
        return temp.copy(out: out);
      });
    }
  }

  if (axis == null) {
    final size = a.size;
    final result = out ?? NDArray<T>.create(targetShape, a.dtype);
    if (size == 0) {
      result.setCellFlat(0, normalizeScalar(0, a.dtype));
      return result;
    }
    Object? acc;
    switch (a.dtype) {
      case DType.float64:
        final temp = a.isContiguous ? a : a.copy();
        final s = _fastContiguousNansumDouble(temp.pointer.cast(), size);
        if (!identical(temp, a)) temp.dispose();
        acc = s;
      case DType.float32:
        final temp = a.isContiguous ? a : a.copy();
        final s = _fastContiguousNansumFloat(temp.pointer.cast(), size);
        if (!identical(temp, a)) temp.dispose();
        acc = s;
      case DType.int32:
      case DType.int64:
        var sumVal = 0;
        final en = NDEnumerate<T>(a);
        while (en.moveNext()) {
          sumVal += en.value as int;
        }
        acc = sumVal;
      case DType.complex64:
      case DType.complex128:
        var sumVal = Complex(0.0, 0.0);
        final en = NDEnumerate<T>(a);
        while (en.moveNext()) {
          final val = en.value as Complex;
          if (val.real.isNaN || val.imag.isNaN) continue;
          sumVal += val;
        }
        acc = sumVal;
      case DType.float16:
      case DType.bfloat16:
      case DType.int16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
      case DType.boolean:
        var sumVal = 0.0;
        final en = NDEnumerate<T>(a);
        while (en.moveNext()) {
          final val = en.value as double;
          if (val.isNaN) continue;
          sumVal += val;
        }
        acc = sumVal;
    }
    result.setCellFlat(0, acc);
    return result;
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<T>.zeros(targetShape, a.dtype);
  if (out != null) {
    result.fill(normalizeScalar(0, a.dtype));
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;

  if (!a.dtype.isComplex) {
    if (a.shape[normAxis] == 0) {
      return result;
    }
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
          s_nansum_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
        case DType.float32:
          s_nansum_float(
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
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          _s_stat_strided_fallback(
            a,
            result,
            rank,
            normAxis,
            squeezedDestStrides,
            s_nansum_double,
          );
      }
      return result;
    } finally {
      ScratchArena.reset(marker);
    }
  }

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
      return ((current as dynamic) + val);
    },
    destStrides: squeezedDestStrides,
  );
  return result;
}
