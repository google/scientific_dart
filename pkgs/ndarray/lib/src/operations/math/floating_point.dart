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
import 'dart:typed_data';
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
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<Boolean> isnan<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isnan() on a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for isnan',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<Boolean>.create(a.shape, DType.boolean);
        isnan<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<Boolean> result =
        out ??
        NDArray<Boolean>.create(
          a.shape,
          DType.boolean,
          zeroInit: where != null,
        );
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
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
          final maskPtr = maskHolder.pointer;
          for (var i = 0; i < result.size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCellFlat(i, false);
            }
          }
          return result;
        case DType.float16:
        case DType.bfloat16:
          final doubleA = castNDArray(a, DType.float64);
          try {
            return isnan(doubleA, where: where, out: result);
          } finally {
            if (!identical(doubleA, a)) doubleA.dispose();
          }
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
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            final maskPtr = maskHolder.pointer;
            for (var i = 0; i < result.size; i++) {
              if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
                result.setCellFlat(i, false);
              }
            }
            return result;
          case DType.float16:
          case DType.bfloat16:
            final doubleA = castNDArray(a, DType.float64);
            try {
              return isnan(doubleA, where: where, out: result);
            } finally {
              if (!identical(doubleA, a)) doubleA.dispose();
            }
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Returns an element-wise boolean mask indicating which elements of the array are positive or negative infinity.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// It is an error if the array has been disposed (throws [StateError]).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<Boolean> isinf<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isinf() on a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for isinf',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<Boolean>.create(a.shape, DType.boolean);
        isinf<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<Boolean> result =
        out ??
        NDArray<Boolean>.create(
          a.shape,
          DType.boolean,
          zeroInit: where != null,
        );
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
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
          final maskPtr = maskHolder.pointer;
          for (var i = 0; i < result.size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCellFlat(i, false);
            }
          }
          return result;
        case DType.float16:
        case DType.bfloat16:
          final doubleA = castNDArray(a, DType.float64);
          try {
            return isinf(doubleA, where: where, out: result);
          } finally {
            if (!identical(doubleA, a)) doubleA.dispose();
          }
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
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            final maskPtr = maskHolder.pointer;
            for (var i = 0; i < result.size; i++) {
              if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
                result.setCellFlat(i, false);
              }
            }
            return result;
          case DType.float16:
          case DType.bfloat16:
            final doubleA = castNDArray(a, DType.float64);
            try {
              return isinf(doubleA, where: where, out: result);
            } finally {
              if (!identical(doubleA, a)) doubleA.dispose();
            }
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Returns an element-wise boolean mask indicating which elements of the array are finite (neither NaN nor infinity).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// It is an error if the array has been disposed (throws [StateError]).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<Boolean> isfinite<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isfinite() on a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for isfinite',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<Boolean>.create(a.shape, DType.boolean);
        isfinite<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<Boolean> result =
        out ??
        NDArray<Boolean>.create(
          a.shape,
          DType.boolean,
          zeroInit: where != null,
        );
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
          try {
            return isfinite(doubleA, where: where, out: result);
          } finally {
            if (!identical(doubleA, a)) doubleA.dispose();
          }
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
            try {
              return isfinite(doubleA, where: where, out: result);
            } finally {
              if (!identical(doubleA, a)) doubleA.dispose();
            }
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Returns first element-wise argument with the sign of the second element-wise argument.
///
/// It is an error if either array has been disposed (throws [StateError]), or if either array is complex (throws [UnsupportedError]).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<T> copysign<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
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
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }

  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = x1.dtype;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for copysign',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(shape, targetDType);
        copysign<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, shape);

  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(shape, targetDType, zeroInit: where != null);
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

    double toDblA(Object? v) => (x1.dtype as DType<DTypeTag>) == DType.uint64
        ? uint64ToDouble(v as int)
        : (v is bool ? (v ? 1.0 : 0.0) : (v as num).toDouble());
    double toDblB(Object? v) => x2.dtype == DType.uint64
        ? uint64ToDouble(v as int)
        : (v is bool ? (v ? 1.0 : 0.0) : (v as num).toDouble());

    if (targetDType.isFloating) {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
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
        (x, y) => castValue(copysignOp(toDblA(x), toDblB(y)), targetDType),
        maskHolder.pointer,
      );
    } else if (targetDType == DType.uint64 ||
        targetDType == DType.uint32 ||
        targetDType == DType.uint16 ||
        targetDType == DType.uint8 ||
        targetDType == DType.boolean) {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
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
        (x, _) => x,
        maskHolder.pointer,
      );
    } else {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
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
        (x, y) {
          final xi = x as int;
          final yi = y as int;
          final absX = xi < 0 ? -xi : xi;
          return castValue(yi < 0 ? -absX : absX, targetDType);
        },
        maskHolder.pointer,
      );
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Changes the sign of [x1] to that of [x2] element-wise, computed into the specified target [dtype].
///
/// Casts [x1] and [x2] to [dtype] and computes [copysign], returning an
/// [NDArray<R>] whose static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [x1], [x2], [where], or [out] is disposed.
/// - Complex numbers are not supported for [x1], [x2], or [dtype].
/// - If [out] is provided, it must be writeable, have the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy copysign](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html)
NDArray<R>
copysignAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> x1,
  NDArray<Tb> x2,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute copysignAs() on a disposed array.');
  }
  if (x1.dtype.isComplex || x2.dtype.isComplex || dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for copysign');
  }
  if ((x1.dtype as DType<DTypeTag>) == dtype &&
      (x2.dtype as DType<DTypeTag>) == dtype) {
    return copysign<DTypeTag>(x1, x2, where: where, out: out) as NDArray<R>;
  }
  return NDArray.scope(() {
    final x1Cast = castNDArray<R>(x1, dtype);
    final x2Cast = castNDArray<R>(x2, dtype);
    final res = copysign<R>(x1Cast, x2Cast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
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
NDArray<Boolean> isClose<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute isClose() on a disposed array.');
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for isClose',
      );
    }
  }

  bool isNan(Object? v) =>
      (v is num && v.isNaN) || (v is Complex && (v.real.isNaN || v.imag.isNaN));
  bool isInf(Object? v) =>
      (v is num && v.isInfinite) ||
      (v is Complex && (v.real.isInfinite || v.imag.isInfinite));
  double absVal(Object? v, DType dtype) {
    if (dtype == DType.uint64) {
      return _numToUnsignedDouble(v, dtype);
    }
    return v is num ? v.toDouble().abs() : (v is Complex ? v.abs : 0.0);
  }

  double diff(Object? v1, DType dtype1, Object? v2, DType dtype2) {
    if (dtype1 == DType.uint64 && dtype2 == DType.uint64) {
      final u1 = v1 as int;
      final u2 = v2 as int;
      if (u1 == u2) return 0.0;
      final cmp = uint64Compare(u1, u2);
      final diffBits = cmp >= 0 ? (u1 - u2) : (u2 - u1);
      return _numToUnsignedDouble(diffBits, DType.uint64);
    }
    if (dtype1 == DType.uint64 || dtype2 == DType.uint64) {
      final d1 = v1 is Complex ? v1 : _numToUnsignedDouble(v1, dtype1);
      final d2 = v2 is Complex ? v2 : _numToUnsignedDouble(v2, dtype2);
      if (d1 is num && d2 is num) return (d1 - d2).abs().toDouble();
      if (d1 is num && d2 is Complex) {
        return (Complex(d1.toDouble(), 0.0) - d2).abs;
      }
      if (d1 is Complex && d2 is num) {
        return (d1 - Complex(d2.toDouble(), 0.0)).abs;
      }
      return 0.0;
    }
    if (v1 is int && v2 is int) {
      return ((v1 ^ v2) >= 0)
          ? (v1 - v2).abs().toDouble()
          : v1.toDouble().abs() + v2.toDouble().abs();
    }
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
    final bool useTempOut =
        out != null &&
        (!out.isContiguous ||
            sharesMemory(a, out) ||
            sharesMemory(b, out) ||
            (where != null && sharesMemory(where, out)));
    final result = useTempOut
        ? (where != null
              ? out.copy()
              : NDArray<Boolean>.zeros(commonShape, DType.boolean))
        : (out ?? NDArray<Boolean>.zeros(commonShape, DType.boolean));

    if (listEquals(a.shape, b.shape) &&
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous) {
      final size = a.size;
      final resPtr = result.pointer.cast<ffi.Uint8>();
      final maskPtr = maskHolder.pointer;
      var handled = true;

      switch ((a.dtype, b.dtype)) {
        case (DType.float64, DType.float64):
          final ptrA = a.pointer.cast<ffi.Double>();
          final ptrB = b.pointer.cast<ffi.Double>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                  ? 1
                  : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] =
                    _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                    ? 1
                    : 0;
              }
            }
          }
        case (DType.float32, DType.float32):
          final ptrA = a.pointer.cast<ffi.Float>();
          final ptrB = b.pointer.cast<ffi.Float>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                  ? 1
                  : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] =
                    _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                    ? 1
                    : 0;
              }
            }
          }
        case (DType.float64, DType.float32):
          final ptrA = a.pointer.cast<ffi.Double>();
          final ptrB = b.pointer.cast<ffi.Float>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                  ? 1
                  : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] =
                    _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                    ? 1
                    : 0;
              }
            }
          }
        case (DType.float32, DType.float64):
          final ptrA = a.pointer.cast<ffi.Float>();
          final ptrB = b.pointer.cast<ffi.Double>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                  ? 1
                  : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] =
                    _isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)
                    ? 1
                    : 0;
              }
            }
          }
        case (DType.int64, DType.int64):
          final ptrA = a.pointer.cast<ffi.Int64>();
          final ptrB = b.pointer.cast<ffi.Int64>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
              }
            }
          }
        case (DType.int32, DType.int32):
          final ptrA = a.pointer.cast<ffi.Int32>();
          final ptrB = b.pointer.cast<ffi.Int32>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
              }
            }
          }
        case (DType.int16, DType.int16):
          final ptrA = a.pointer.cast<ffi.Int16>();
          final ptrB = b.pointer.cast<ffi.Int16>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
              }
            }
          }
        case (DType.int8, DType.int8):
          final ptrA = a.pointer.cast<ffi.Int8>();
          final ptrB = b.pointer.cast<ffi.Int8>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
              }
            }
          }
        case (DType.uint32, DType.uint32):
          final ptrA = a.pointer.cast<ffi.Uint32>();
          final ptrB = b.pointer.cast<ffi.Uint32>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
              }
            }
          }
        case (DType.uint16, DType.uint16):
          final ptrA = a.pointer.cast<ffi.Uint16>();
          final ptrB = b.pointer.cast<ffi.Uint16>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
              }
            }
          }
        case (DType.uint8, DType.uint8):
          final ptrA = a.pointer.cast<ffi.Uint8>();
          final ptrB = b.pointer.cast<ffi.Uint8>();
          if (maskPtr == ffi.nullptr) {
            for (var i = 0; i < size; i++) {
              resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
            }
          } else {
            for (var i = 0; i < size; i++) {
              if (maskPtr[i] != 0) {
                resPtr[i] = _isCloseInt(ptrA[i], ptrB[i], rtol, atol) ? 1 : 0;
              }
            }
          }
        default:
          handled = false;
      }

      if (handled) {
        if (useTempOut) {
          result.copy(out: out);
          result.dispose();
          return out;
        }
        return result;
      }
    }

    final iter = NDIter.broadcast3(result, a, b);
    final maskPtr = maskHolder.pointer;
    var flatIdx = 0;
    while (iter.moveNext()) {
      if (maskPtr == ffi.nullptr || maskPtr[flatIdx] != 0) {
        final idxRes = iter.getIndex(0);
        final idxA = iter.getIndex(1);
        final idxB = iter.getIndex(2);
        final valA = a.getCellRaw(idxA);
        final valB = b.getCellRaw(idxB);

        var match = false;
        if (equalNan && isNan(valA) && isNan(valB)) {
          match = true;
        } else if (isInf(valA) || isInf(valB)) {
          match = valA == valB;
        } else {
          final d = diff(valA, a.dtype, valB, b.dtype);
          final limit = atol + rtol * absVal(valB, b.dtype);
          match = d <= limit;
        }

        result.setCellRaw(idxRes, match);
      }
      flatIdx++;
    }

    if (useTempOut) {
      result.copy(out: out);
      result.dispose();
      return out;
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

@pragma('vm:prefer-inline')
double _numToUnsignedDouble(Object? v, DType dtype) {
  if (dtype == DType.uint64) {
    final u = v as int;
    return u >= 0
        ? u.toDouble()
        : (u & 0x7fffffffffffffff).toDouble() + 9223372036854775808.0;
  }
  return (v as num).toDouble();
}

@pragma('vm:prefer-inline')
bool _isCloseDouble(
  double aVal,
  double bVal,
  double rtol,
  double atol,
  bool equalNan,
) {
  final bAbs = bVal.abs();
  if (bAbs < double.infinity &&
      atol < double.infinity &&
      rtol < double.infinity) {
    return (aVal - bVal).abs() <= atol + rtol * bAbs;
  }
  if (aVal.isNaN || bVal.isNaN) {
    return equalNan && aVal.isNaN && bVal.isNaN;
  }
  if (aVal.isInfinite || bVal.isInfinite) {
    return aVal == bVal;
  }
  return (aVal - bVal).abs() <= atol + rtol * bAbs;
}

@pragma('vm:prefer-inline')
bool _isCloseInt(int aVal, int bVal, double rtol, double atol) {
  final d = ((aVal ^ bVal) >= 0)
      ? (aVal - bVal).abs().toDouble()
      : aVal.toDouble().abs() + bVal.toDouble().abs();
  final limit = atol + rtol * bVal.toDouble().abs();
  return d <= limit;
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
bool allClose<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute allClose() on a disposed array.');
  }

  if (listEquals(a.shape, b.shape) && a.isContiguous && b.isContiguous) {
    final size = a.size;
    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64):
        final ptrA = a.pointer.cast<ffi.Double>();
        final ptrB = b.pointer.cast<ffi.Double>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)) {
            return false;
          }
        }
        return true;
      case (DType.float32, DType.float32):
        final ptrA = a.pointer.cast<ffi.Float>();
        final ptrB = b.pointer.cast<ffi.Float>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)) {
            return false;
          }
        }
        return true;
      case (DType.float64, DType.float32):
        final ptrA = a.pointer.cast<ffi.Double>();
        final ptrB = b.pointer.cast<ffi.Float>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)) {
            return false;
          }
        }
        return true;
      case (DType.float32, DType.float64):
        final ptrA = a.pointer.cast<ffi.Float>();
        final ptrB = b.pointer.cast<ffi.Double>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseDouble(ptrA[i], ptrB[i], rtol, atol, equalNan)) {
            return false;
          }
        }
        return true;
      case (DType.int64, DType.int64):
        final ptrA = a.pointer.cast<ffi.Int64>();
        final ptrB = b.pointer.cast<ffi.Int64>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseInt(ptrA[i], ptrB[i], rtol, atol)) return false;
        }
        return true;
      case (DType.int32, DType.int32):
        final ptrA = a.pointer.cast<ffi.Int32>();
        final ptrB = b.pointer.cast<ffi.Int32>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseInt(ptrA[i], ptrB[i], rtol, atol)) return false;
        }
        return true;
      case (DType.int16, DType.int16):
        final ptrA = a.pointer.cast<ffi.Int16>();
        final ptrB = b.pointer.cast<ffi.Int16>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseInt(ptrA[i], ptrB[i], rtol, atol)) return false;
        }
        return true;
      case (DType.int8, DType.int8):
        final ptrA = a.pointer.cast<ffi.Int8>();
        final ptrB = b.pointer.cast<ffi.Int8>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseInt(ptrA[i], ptrB[i], rtol, atol)) return false;
        }
        return true;
      case (DType.uint32, DType.uint32):
        final ptrA = a.pointer.cast<ffi.Uint32>();
        final ptrB = b.pointer.cast<ffi.Uint32>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseInt(ptrA[i], ptrB[i], rtol, atol)) return false;
        }
        return true;
      case (DType.uint16, DType.uint16):
        final ptrA = a.pointer.cast<ffi.Uint16>();
        final ptrB = b.pointer.cast<ffi.Uint16>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseInt(ptrA[i], ptrB[i], rtol, atol)) return false;
        }
        return true;
      case (DType.uint8, DType.uint8):
        final ptrA = a.pointer.cast<ffi.Uint8>();
        final ptrB = b.pointer.cast<ffi.Uint8>();
        for (var i = 0; i < size; i++) {
          if (!_isCloseInt(ptrA[i], ptrB[i], rtol, atol)) return false;
        }
        return true;
      default:
        return _allCloseFallback(
          a,
          b,
          rtol: rtol,
          atol: atol,
          equalNan: equalNan,
        );
    }
  }

  return _allCloseFallback(a, b, rtol: rtol, atol: atol, equalNan: equalNan);
}

bool _allCloseFallback<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  required double rtol,
  required double atol,
  required bool equalNan,
}) {
  final closeMask = isClose(a, b, rtol: rtol, atol: atol, equalNan: equalNan);
  try {
    if (closeMask.isContiguous) {
      final ptr = closeMask.pointer.cast<ffi.Uint8>();
      final n = closeMask.size;
      for (var i = 0; i < n; i++) {
        if (ptr[i] == 0) return false;
      }
      return true;
    }
    for (var i = 0; i < closeMask.size; i++) {
      if (!closeMask.getCellFlat(i)) return false;
    }
    return true;
  } finally {
    closeMask.dispose();
  }
}

/// Extension providing positional accessors and disposal for [modf] results.
extension ModfRecordExtension<R extends DTypeTag>
    on ({NDArray<R> fractional, NDArray<R> integral}) {
  /// The fractional part of the input array.
  NDArray<R> get $1 => fractional;

  /// The integral part of the input array.
  NDArray<R> get $2 => integral;

  /// Disposes both returned arrays.
  void dispose() {
    fractional.dispose();
    integral.dispose();
  }
}

/// Extension providing positional accessors and disposal for [frexp] results.
extension FrexpRecordExtension<R extends DTypeTag>
    on ({NDArray<R> mantissa, NDArray<Int32> exponent}) {
  /// The mantissa array in the interval $[0.5, 1)$ (or $(-1, -0.5]$).
  NDArray<R> get $1 => mantissa;

  /// The base-2 integer exponent array.
  NDArray<Int32> get $2 => exponent;

  /// Disposes both returned arrays.
  void dispose() {
    mantissa.dispose();
    exponent.dispose();
  }
}

/// Return the fractional and integral parts of an array, element-wise.
///
/// The fractional and integral parts are negative if the given number is negative.
///
/// **Preconditions:**
/// - Input [x] must be a real-valued array and not disposed.
/// - If provided, [out1] and [out2] must match [x]'s shape and resolved floating dtype.
///
/// It is an error if [x], [out1], [out2], or [where] is disposed (throws [StateError]),
/// if [x] is complex (throws [UnsupportedError]), or if [out1]/[out2] have incompatible
/// shapes/dtypes or alias each other (throws [ArgumentError]).
///
/// Reference: [NumPy modf](https://numpy.org/doc/stable/reference/generated/numpy.modf.html)
({NDArray<R> fractional, NDArray<R> integral}) modf<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> x, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out1,
  NDArray<R>? out2,
}) {
  if (x.isDisposed ||
      (out1 != null && out1.isDisposed) ||
      (out2 != null && out2.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute modf() on a disposed array.');
  }
  if (x.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for modf.');
  }
  final DType<DTypeTag> defaultDType =
      (x.dtype as DType<DTypeTag>) == DType.float32
      ? DType.float32
      : DType.float64;
  final DType<DTypeTag> resolvedDType =
      out1?.dtype ?? out2?.dtype ?? defaultDType;
  if (!resolvedDType.isFloating ||
      (resolvedDType != defaultDType && resolvedDType != x.dtype)) {
    throw ArgumentError.value(
      out1 ?? out2 ?? resolvedDType,
      out1 != null ? 'out1' : (out2 != null ? 'out2' : 'targetDType'),
      'Must have compatible shape and dtype for modf',
    );
  }
  if (out1 != null) {
    validateOutBuffer(out1, 'out1');
    if (!listEquals(out1.shape, x.shape) || out1.dtype != resolvedDType) {
      throw ArgumentError.value(
        out1,
        'out1',
        'Must have compatible shape and dtype for modf',
      );
    }
  }
  if (out2 != null) {
    validateOutBuffer(out2, 'out2');
    if (!listEquals(out2.shape, x.shape) || out2.dtype != resolvedDType) {
      throw ArgumentError.value(
        out2,
        'out2',
        'Must have compatible shape and dtype for modf',
      );
    }
  }
  if (out1 != null && out2 != null && sharesMemory(out1, out2)) {
    throw ArgumentError.value(
      out2,
      'out2',
      'Must not share memory with out1 in modf',
    );
  }
  final DType<R> targetDType = resolvedDType as DType<R>;

  final maskHolder = prepareMask(where, x.shape);
  try {
    final bool useTemp1 =
        out1 != null &&
        (sharesMemory(x, out1) || (where != null && sharesMemory(where, out1)));
    final bool useTemp2 =
        out2 != null &&
        (sharesMemory(x, out2) || (where != null && sharesMemory(where, out2)));

    final NDArray<R> res1 = useTemp1
        ? (where != null
              ? out1.copy()
              : NDArray<R>.create(x.shape, targetDType))
        : (out1 ??
              NDArray<R>.create(x.shape, targetDType, zeroInit: where != null));
    final NDArray<R> res2 = useTemp2
        ? (where != null
              ? out2.copy()
              : NDArray<R>.create(x.shape, targetDType))
        : (out2 ??
              NDArray<R>.create(x.shape, targetDType, zeroInit: where != null));

    try {
      double toDoubleVal(Object? val) {
        if ((x.dtype as DType<DTypeTag>) == DType.uint64 && val is int) {
          return BigInt.from(val).toUnsigned(64).toDouble();
        }
        if (val is bool) return val ? 1.0 : 0.0;
        return (val as num).toDouble();
      }

      unaryOp<DTypeTag, R>(
        res1,
        x,
        x.shape,
        x.strides,
        res1.strides,
        0,
        x.offsetElements,
        res1.offsetElements,
        (v) {
          final dv = toDoubleVal(v);
          if (dv.isNaN) return castValue(double.nan, targetDType);
          if (dv.isInfinite) {
            return castValue(dv.isNegative ? -0.0 : 0.0, targetDType);
          }
          final iPart = dv.truncateToDouble();
          final fPart = dv - iPart == 0.0
              ? (dv.isNegative ? -0.0 : 0.0)
              : dv - iPart;
          return castValue(fPart, targetDType);
        },
        maskHolder.pointer,
      );

      unaryOp<DTypeTag, R>(
        res2,
        x,
        x.shape,
        x.strides,
        res2.strides,
        0,
        x.offsetElements,
        res2.offsetElements,
        (v) {
          final dv = toDoubleVal(v);
          if (dv.isNaN || dv.isInfinite) {
            return castValue(dv, targetDType);
          }
          final iPart = dv.truncateToDouble();
          return castValue(iPart, targetDType);
        },
        maskHolder.pointer,
      );

      if (useTemp1) {
        res1.copy(out: out1);
      }
      if (useTemp2) {
        res2.copy(out: out2);
      }

      return (fractional: out1 ?? res1, integral: out2 ?? res2);
    } finally {
      if (useTemp1) res1.dispose();
      if (useTemp2) res2.dispose();
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Decompose the elements of [x] into mantissa and twos exponent.
///
/// Returns `(mantissa, exponent)`, where $x = \text{mantissa} \times 2^{\text{exponent}}$,
/// with the mantissa in the open interval $(-1, -0.5]$ or $[0.5, 1)$ (or $0$ when $x = 0$).
///
/// Reference: [NumPy frexp](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html)
({NDArray<R> mantissa, NDArray<Int32> exponent}) frexp<R extends DTypeTag>(
  NDArray<RealFloatOf<R>> x, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out1,
  NDArray<Int32>? out2,
}) {
  if (x.isDisposed ||
      (out1 != null && out1.isDisposed) ||
      (out2 != null && out2.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute frexp() on a disposed array.');
  }
  if (x.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for frexp.');
  }
  final DType<DTypeTag> defaultDType =
      (x.dtype as DType<DTypeTag>) == DType.float32
      ? DType.float32
      : DType.float64;
  final DType<DTypeTag> resolvedDType = out1?.dtype ?? defaultDType;
  if (!resolvedDType.isFloating ||
      (resolvedDType != defaultDType && resolvedDType != x.dtype)) {
    throw ArgumentError.value(
      out1 ?? resolvedDType,
      out1 != null ? 'out1' : 'targetDType',
      'Must have compatible shape and dtype for frexp',
    );
  }
  if (out1 != null) {
    validateOutBuffer(out1, 'out1');
    if (!listEquals(out1.shape, x.shape) || out1.dtype != resolvedDType) {
      throw ArgumentError.value(
        out1,
        'out1',
        'Must have compatible shape and dtype for frexp',
      );
    }
  }
  if (out2 != null) {
    validateOutBuffer(out2, 'out2');
    if (!listEquals(out2.shape, x.shape) || out2.dtype != DType.int32) {
      throw ArgumentError.value(
        out2,
        'out2',
        'Must have compatible shape and dtype for frexp',
      );
    }
  }
  if (out1 != null && out2 != null && sharesMemory(out1, out2)) {
    throw ArgumentError.value(
      out2,
      'out2',
      'Must not share memory with out1 in frexp',
    );
  }
  final DType<R> targetDType = resolvedDType as DType<R>;

  final maskHolder = prepareMask(where, x.shape);
  try {
    final bool useTemp1 =
        out1 != null &&
        (sharesMemory(x, out1) || (where != null && sharesMemory(where, out1)));
    final bool useTemp2 =
        out2 != null &&
        (sharesMemory(x, out2) || (where != null && sharesMemory(where, out2)));

    final NDArray<R> res1 = useTemp1
        ? (where != null
              ? out1.copy()
              : NDArray<R>.create(x.shape, targetDType))
        : (out1 ??
              NDArray<R>.create(x.shape, targetDType, zeroInit: where != null));
    final NDArray<Int32> res2 = useTemp2
        ? (where != null
              ? out2.copy()
              : NDArray<Int32>.create(x.shape, DType.int32))
        : (out2 ??
              NDArray<Int32>.create(
                x.shape,
                DType.int32,
                zeroInit: where != null,
              ));

    final f64Scratch = Float64List(1);
    final u64Scratch = f64Scratch.buffer.asUint64List();

    (double, int) decomposeFrexp(double dv) {
      if (dv == 0.0 || dv.isNaN || dv.isInfinite) {
        return (dv, 0);
      }
      var expAdjust = 0;
      var work = dv;
      f64Scratch[0] = work;
      var bits = u64Scratch[0];
      var biasedExp = (bits >>> 52) & 0x7FF;
      if (biasedExp == 0) {
        work *= 18014398509481984.0; // 2^54
        expAdjust = -54;
        f64Scratch[0] = work;
        bits = u64Scratch[0];
        biasedExp = (bits >>> 52) & 0x7FF;
      }
      final exp = biasedExp - 1022 + expAdjust;
      u64Scratch[0] = (bits & 0x800FFFFFFFFFFFFF) | (0x3FE << 52);
      return (f64Scratch[0], exp);
    }

    double toDoubleVal(Object? val) {
      if ((x.dtype as DType<DTypeTag>) == DType.uint64 && val is int) {
        return BigInt.from(val).toUnsigned(64).toDouble();
      }
      if (val is bool) return val ? 1.0 : 0.0;
      return (val as num).toDouble();
    }

    try {
      unaryOp<DTypeTag, R>(
        res1,
        x,
        x.shape,
        x.strides,
        res1.strides,
        0,
        x.offsetElements,
        res1.offsetElements,
        (v) {
          final (m, _) = decomposeFrexp(toDoubleVal(v));
          return castValue(m, targetDType);
        },
        maskHolder.pointer,
      );

      unaryOp<DTypeTag, Int32>(
        res2,
        x,
        x.shape,
        x.strides,
        res2.strides,
        0,
        x.offsetElements,
        res2.offsetElements,
        (v) {
          final (_, e) = decomposeFrexp(toDoubleVal(v));
          return e;
        },
        maskHolder.pointer,
      );

      if (useTemp1) {
        res1.copy(out: out1);
      }
      if (useTemp2) {
        res2.copy(out: out2);
      }

      return (mantissa: out1 ?? res1, exponent: out2 ?? res2);
    } finally {
      if (useTemp1) res1.dispose();
      if (useTemp2) res2.dispose();
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Returns `true` if two arrays have the same shape and elements, `false` otherwise.
///
/// Reference: [numpy.array_equal](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html)
bool arrayEqual<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  bool equalNan = false,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute arrayEqual() on a disposed array.');
  }
  if (!equalNan && a.dtype == b.dtype) {
    return a.equals(b);
  }
  if (!listEquals(a.shape, b.shape)) return false;
  if (a.size == 0) return true;
  final iter = NDIter.broadcast2(a, b);
  while (iter.moveNext()) {
    final va = a.getCellRaw(iter.getIndex(0));
    final vb = b.getCellRaw(iter.getIndex(1));
    if (a.dtype == DType.uint64 && b.dtype != DType.uint64) {
      final ua = va as int;
      if (b.dtype.isInteger) {
        final ib = vb as int;
        if (ua < 0 || ib < 0 || ua != ib) return false;
        continue;
      }
    } else if (b.dtype == DType.uint64 && a.dtype != DType.uint64) {
      final ub = vb as int;
      if (a.dtype.isInteger) {
        final ia = va as int;
        if (ub < 0 || ia < 0 || ia != ub) return false;
        continue;
      }
    } else if (va == vb) {
      continue;
    }
    if (va is Complex && vb is Complex) {
      final rEq =
          va.real == vb.real || (equalNan && va.real.isNaN && vb.real.isNaN);
      final iEq =
          va.imag == vb.imag || (equalNan && va.imag.isNaN && vb.imag.isNaN);
      if (rEq && iEq) continue;
      return false;
    }
    if (va is num && vb is num) {
      final da = _numToUnsignedDouble(va, a.dtype);
      final db = _numToUnsignedDouble(vb, b.dtype);
      if (da == db || (equalNan && da.isNaN && db.isNaN)) continue;
      return false;
    }
    return false;
  }
  return true;
}
