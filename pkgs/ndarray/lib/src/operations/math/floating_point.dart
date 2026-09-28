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
import 'dart:ffi' as ffi;

import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../helpers.dart';
import '../broadcasting.dart';
import '../../nditer.dart';

/// Returns an element-wise boolean mask indicating which elements of the array are NaN.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - If provided, the [out] recycler array must match the shape and have boolean dtype.
///
/// It is an error if the array has been disposed (throws [StateError]), or if [out] has incompatible shape or dtype (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// final mask = isnan(a); // [false, true, false]
/// ```
NDArray<bool> isnan<T>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<bool>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isnan() on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for isnan.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }
  final maskHolder = prepareMask(where, result.shape);

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isnan_double(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.float32:
        v_isnan_float(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.complex128:
        v_isnan_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.complex64:
        v_isnan_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.int32:
      case DType.int64:
      case DType.int16:
      case DType.uint8:
        final maskPtr = maskHolder.pointer;
        for (var i = 0; i < result.size; i++) {
          if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
            result.setCellFlat(i, false);
          }
        }
        return result;
      default:
        break;
    }
  } else {
    final rank = a.rank;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }
      switch (a.dtype) {
        case DType.float64:
          s_isnan_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          s_isnan_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          s_isnan_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          s_isnan_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
        case DType.int64:
        case DType.int16:
        case DType.uint8:
          final maskPtr = maskHolder.pointer;
          for (var i = 0; i < result.size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCellFlat(i, false);
            }
          }
          return result;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are positive or negative infinity.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// It is an error if the array has been disposed (throws [StateError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.infinity, 3.0], [3], DType.float64);
/// final mask = isinf(a); // [false, true, false]
/// ```
NDArray<bool> isinf<T>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<bool>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isinf() on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for isinf.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }
  final maskHolder = prepareMask(where, result.shape);

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isinf_double(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.float32:
        v_isinf_float(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.complex128:
        v_isinf_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.complex64:
        v_isinf_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.int32:
      case DType.int64:
      case DType.int16:
      case DType.uint8:
        final maskPtr = maskHolder.pointer;
        for (var i = 0; i < result.size; i++) {
          if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
            result.setCellFlat(i, false);
          }
        }
        return result;
      default:
        break;
    }
  } else {
    final rank = a.rank;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }
      switch (a.dtype) {
        case DType.float64:
          s_isinf_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          s_isinf_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          s_isinf_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          s_isinf_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
        case DType.int64:
        case DType.int16:
        case DType.uint8:
          final maskPtr = maskHolder.pointer;
          for (var i = 0; i < result.size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCellFlat(i, false);
            }
          }
          return result;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are finite (neither NaN nor infinite).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// It is an error if the array has been disposed (throws [StateError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, double.infinity], [3], DType.float64);
/// final mask = isfinite(a); // [true, false, false]
/// ```
NDArray<bool> isfinite<T extends Object>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<bool>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isfinite() on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for isfinite.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }
  final maskHolder = prepareMask(where, result.shape);

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isfinite_double(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.float32:
        v_isfinite_float(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.complex128:
        v_isfinite_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.complex64:
        v_isfinite_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case DType.int32:
      case DType.int64:
      case DType.int16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
      case DType.boolean:
        final maskPtr = maskHolder.pointer;
        for (var i = 0; i < result.size; i++) {
          if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
            result.setCellFlat(i, true);
          }
        }
        return result;
      case DType.float16:
      case DType.bfloat16:
        final doubleA = castNDArray(a, DType.float64);
        final doubleRes = isfinite(doubleA, where: where);
        doubleRes.copy(out: result);
        doubleA.dispose();
        doubleRes.dispose();
        return result;
    }
  } else {
    final rank = a.rank;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }
      switch (a.dtype) {
        case DType.float64:
          s_isfinite_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          s_isfinite_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          s_isfinite_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          s_isfinite_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
        case DType.int64:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
          final maskPtr = maskHolder.pointer;
          for (var i = 0; i < result.size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCellFlat(i, true);
            }
          }
          return result;
        case DType.float16:
        case DType.bfloat16:
          final doubleA = castNDArray(a, DType.float64);
          final doubleRes = isfinite(doubleA, where: where);
          doubleRes.copy(out: result);
          doubleA.dispose();
          doubleRes.dispose();
          return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }
}

/// Returns first element-wise argument with the sign of the second element-wise argument.
///
/// It is an error if either array has been disposed (throws [StateError]), or if either array is complex (throws [UnsupportedError]).
///
/// **Example:**
/// ```dart
/// final res = copysign(x1, x2);
/// ```
NDArray<T> copysign<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute copysign() on a disposed array.');
  }
  if (x1.dtype.isComplex || x2.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for copysign');
  }

  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = x1.dtype;

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for copysign.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(shape, targetDType);
  }
  final maskHolder = prepareMask(where, result.shape);

  try {
    if (x1.dtype == targetDType &&
        x2.dtype == targetDType &&
        x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          v_copysign_double(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_copysign_float(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    } else if (x1.dtype == targetDType &&
        x2.dtype == targetDType &&
        shape.length <= 8) {
      final rank = shape.length;
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(shape);
        final cStridesA = ScratchArena.copyInts(stridesA);
        final cStridesB = ScratchArena.copyInts(stridesB);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        switch (targetDType) {
          case DType.float64:
            s_copysign_double(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.float32:
            s_copysign_float(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    double copysignOp(double a, double b) {
      if (b == 0.0) {
        return b.isNegative ? -a.abs() : a.abs();
      }
      return b < 0.0 ? -a.abs() : a.abs();
    }

    if (targetDType.isFloating) {
      elementWiseOp<dynamic, dynamic, dynamic>(
        result,
        x1,
        x2,
        shape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) => castValue(
          copysignOp((x as num).toDouble(), (y as num).toDouble()),
          targetDType,
        ),
        maskHolder.pointer,
      );
    } else {
      elementWiseOp<dynamic, dynamic, dynamic>(
        result,
        x1,
        x2,
        shape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) => castValue(
          copysignOp((x as num).toDouble(), (y as num).toDouble()).toInt(),
          targetDType,
        ),
        maskHolder.pointer,
      );
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Returns a boolean [NDArray] where two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
NDArray<bool> isClose<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
  NDArray<dynamic>? where,
  NDArray<bool>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isClose() on a disposed array.');
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  final result = out ?? NDArray<bool>.zeros(commonShape, DType.boolean);
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for isClose.',
      );
    }
  }

  bool isNan(Object? v) =>
      (v is num && v.isNaN) || (v is Complex && (v.real.isNaN || v.imag.isNaN));
  bool isInf(Object? v) =>
      (v is num && v.isInfinite) ||
      (v is Complex && (v.real.isInfinite || v.imag.isInfinite));
  double abs(Object? v) =>
      v is num ? v.abs().toDouble() : (v is Complex ? v.abs : 0.0);
  double diff(Object? v1, Object? v2) {
    if (v1 is num && v2 is num) return (v1 - v2).abs().toDouble();
    if (v1 is Complex && v2 is Complex) return (v1 - v2).abs;
    if (v1 is num && v2 is Complex) {
      return (Complex(v1.toDouble(), 0.0) - v2).abs;
    }
    if (v1 is Complex && v2 is num) {
      return (v1 - Complex(v2.toDouble(), 0.0)).abs;
    }
    return 0.0;
  }

  final maskHolder = prepareMask(where, commonShape);
  try {
    final iter = NDIter.broadcast3(result, a, b);
    final maskPtr = maskHolder.pointer;
    var flatIdx = 0;
    while (iter.moveNext()) {
      if (maskPtr == ffi.nullptr || maskPtr[flatIdx] != 0) {
        final idxRes = iter.getIndex(0);
        final idxA = iter.getIndex(1);
        final idxB = iter.getIndex(2);
        final valA = a.getCellFlat(idxA);
        final valB = b.getCellFlat(idxB);

        var match = false;
        if (equalNan && isNan(valA) && isNan(valB)) {
          match = true;
        } else if (isInf(valA) || isInf(valB)) {
          match = valA == valB;
        } else {
          final d = diff(valA, valB);
          final limit = atol + rtol * abs(valB);
          match = d <= limit;
        }

        result.setCellFlat(idxRes, match);
      }
      flatIdx++;
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Returns true if two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)
bool allClose<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final closeMask = isClose(a, b, rtol: rtol, atol: atol, equalNan: equalNan);
  try {
    for (var i = 0; i < closeMask.size; i++) {
      if (!closeMask.getCellFlat(i)) return false;
    }
    return true;
  } finally {
    closeMask.dispose();
  }
}
