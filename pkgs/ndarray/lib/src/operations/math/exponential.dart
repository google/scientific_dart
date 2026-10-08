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

import 'dart:math' as math;
import 'dart:ffi' as ffi;
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../helpers.dart';

/// Computes the element-wise exponential of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// It is an error if the provided [out] buffer has an incompatible shape (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_exp_double`/`v_exp_float`).
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Exponential Function](https://en.wikipedia.org/wiki/Exponential_function)
NDArray<R> exp<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      DTypeTag,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute exp() on a disposed array.');
  }
  final DType<R> targetDType;
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    targetDType = a.dtype as DType<R>;
  } else {
    targetDType =
        ((a.dtype as DType<DTypeTag>) == DType.float32
                ? DType.float32
                : DType.float64)
            as DType<R>;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for exp',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        exp<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ?? NDArray.create(a.shape, targetDType, zeroInit: where != null);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_exp_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_exp_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_exp_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_exp_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
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
          break;
      }
    } else {
      final rank = a.shape.length;
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
            s_exp_double(
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
            s_exp_float(
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
            s_exp_complex128(
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
            s_exp_complex64(
              a.pointer.cast(),
              cStridesA,
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
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    unaryOp<DTypeTag, DTypeTag>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        if (x is Complex) {
          final expR = math.exp(x.real);
          return Complex(expR * math.cos(x.imag), expR * math.sin(x.imag));
        }
        final dx = (a.dtype as DType<DTypeTag>) == DType.uint64
            ? uint64ToDouble(x as int)
            : (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
        return castValue(math.exp(dx), result.dtype);
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise natural logarithm of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and the resolved floating-point dtype (Float32 if [a] is Float32, Float64 otherwise).
///
/// It is an error if the provided [out] buffer has an incompatible shape (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_log_double`/`v_log_float`).
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Natural Logarithm](https://en.wikipedia.org/wiki/Natural_logarithm)
NDArray<R> log<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      DTypeTag,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute log() on a disposed array.');
  }
  final DType<DTypeTag> targetDType;
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
        ? DType.float32
        : DType.float64;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for log',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        log<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        NDArray.create(
          a.shape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_log_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_log_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_log_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_log_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
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
          break;
      }
    } else {
      final rank = a.shape.length;
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
            s_log_double(
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
            s_log_float(
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
            s_log_complex128(
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
            s_log_complex64(
              a.pointer.cast(),
              cStridesA,
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
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    unaryOp<DTypeTag, DTypeTag>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        if (x is Complex) return x.log();
        final dx = (a.dtype as DType<DTypeTag>) == DType.uint64
            ? uint64ToDouble(x as int)
            : (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
        return castValue(math.log(dx), result.dtype);
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the base-2 logarithm of each element in [a].
///
/// **Supported Types:**
/// - `float32`, `float64`, `complex64`, `complex128`.
/// - Integer arrays will be upcasted to `float64`.
///
/// For complex inputs, computes `log(z) / log(2)`.
///
/// **Example:**
/// {@example /example/easy_ufuncs_example.dart lang=dart}
NDArray<R> log2<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      DTypeTag,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute log2() on a disposed array.');
  }
  final DType<DTypeTag> targetDType;
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
        ? DType.float32
        : DType.float64;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for log2',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        log2<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        NDArray.create(
          a.shape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_log2_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_log2_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_log2_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_log2_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
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
          break;
      }
    } else {
      final rank = a.shape.length;
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
            s_log2_double(
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
            s_log2_float(
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
            s_log2_complex128(
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
            s_log2_complex64(
              a.pointer.cast(),
              cStridesA,
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
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    unaryOp<DTypeTag, DTypeTag>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        if (x is Complex) return x.log() / math.log(2.0);
        final dx = (a.dtype as DType<DTypeTag>) == DType.uint64
            ? uint64ToDouble(x as int)
            : (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
        return castValue(math.log(dx) / math.log(2.0), result.dtype);
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the base-10 logarithm of each element in [a].
///
/// **Supported Types:**
/// - `float32`, `float64`, `complex64`, `complex128`.
/// - Integer arrays will be upcasted to `float64`.
///
/// For complex inputs, computes `log(z) / log(10)`.
///
/// **Example:**
/// {@example /example/easy_ufuncs_example.dart lang=dart}
NDArray<R> log10<R extends DTypeTag>(
  NDArray<
    DTypeSpec<
      DTypeTag,
      Object?,
      DTypeTag,
      DTypeTag,
      R,
      DTypeTag,
      DTypeTag,
      DTypeTag
    >
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute log10() on a disposed array.');
  }
  final DType<DTypeTag> targetDType;
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
        ? DType.float32
        : DType.float64;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for log10',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        log10<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        NDArray.create(
          a.shape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_log10_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_log10_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_log10_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_log10_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
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
          break;
      }
    } else {
      final rank = a.shape.length;
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
            s_log10_double(
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
            s_log10_float(
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
            s_log10_complex128(
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
            s_log10_complex64(
              a.pointer.cast(),
              cStridesA,
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
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    unaryOp<DTypeTag, DTypeTag>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        if (x is Complex) return x.log() / math.log(10.0);
        final dx = (a.dtype as DType<DTypeTag>) == DType.uint64
            ? uint64ToDouble(x as int)
            : (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
        return castValue(math.log(dx) / math.log(10.0), result.dtype);
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}
