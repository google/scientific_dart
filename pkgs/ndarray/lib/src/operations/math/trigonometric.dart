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
import '../broadcasting.dart';
import 'arithmetic.dart';

/// Computes the element-wise sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_sin_double`/`v_sin_float`).
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Sine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<R> sin<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute sin() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for sin',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        sin<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res =
          sin<Float64>(promoted, where: where, out: out as NDArray<Float64>?)
              as NDArray<R>;
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_sin_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_sin_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_sin_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_sin_complex64(
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
            s_sin_double(
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
            s_sin_float(
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
            s_sin_complex128(
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
            s_sin_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.sin(x as num),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise sinc of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`) or [Complex].
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_sinc_double`/`v_sinc_float` etc).
NDArray<R> sinc<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute sinc() on a disposed array.');
  }

  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for sinc',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        sinc<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res =
          sinc<Float64>(promoted, where: where, out: out as NDArray<Float64>?)
              as NDArray<R>;
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        NDArray<R>.create(
          a.shape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );

    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_sinc_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_sinc_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_sinc_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_sinc_complex64(
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
            s_sinc_double(
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
            s_sinc_float(
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
            s_sinc_complex128(
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
            s_sinc_complex64(
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

    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be a supported DType for sinc (got ${a.dtype})',
    );
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_cos_double`/`v_cos_float`).
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Cosine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<R> cos<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute cos() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for cos',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        cos<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res =
          cos<Float64>(promoted, where: where, out: out as NDArray<Float64>?)
              as NDArray<R>;
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_cos_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_cos_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_cos_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_cos_complex64(
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
            s_cos_double(
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
            s_cos_float(
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
            s_cos_complex128(
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
            s_cos_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.cos(x as num),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise tangent of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> tan<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute tan() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for tan',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        tan<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res =
          tan<Float64>(promoted, where: where, out: out as NDArray<Float64>?)
              as NDArray<R>;
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_tan_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_tan_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_tan_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_tan_complex64(
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
            s_tan_double(
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
            s_tan_float(
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
            s_tan_complex128(
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
            s_tan_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.tan(x as num),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise arc sine (inverse sine) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> asin<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute asin() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for asin',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        asin<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = asin<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_asin_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_asin_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_asin_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_asin_complex64(
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
            s_asin_double(
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
            s_asin_float(
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
            s_asin_complex128(
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
            s_asin_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.asin(x as num),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise arc cosine (inverse cosine) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> acos<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute acos() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for acos',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        acos<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = acos<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_acos_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_acos_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_acos_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_acos_complex64(
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
            s_acos_double(
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
            s_acos_float(
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
            s_acos_complex128(
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
            s_acos_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.acos(x as num),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise arc tangent (inverse tangent) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> atan<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute atan() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for atan',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        atan<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = atan<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_atan_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_atan_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_atan_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_atan_complex64(
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
            s_atan_double(
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
            s_atan_float(
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
            s_atan_complex128(
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
            s_atan_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.atan(x as num),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise hyperbolic sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<R> sinh<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute sinh() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for sinh',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        sinh<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = sinh<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_sinh_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_sinh_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_sinh_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_sinh_complex64(
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
            s_sinh_double(
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
            s_sinh_float(
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
            s_sinh_complex128(
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
            s_sinh_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        final val = (x as num).toDouble();
        return (math.exp(val) - math.exp(-val)) / 2.0;
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise hyperbolic cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<R> cosh<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute cosh() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for cosh',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        cosh<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = cosh<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_cosh_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_cosh_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_cosh_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_cosh_complex64(
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
            s_cosh_double(
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
            s_cosh_float(
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
            s_cosh_complex128(
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
            s_cosh_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        final val = (x as num).toDouble();
        return (math.exp(val) + math.exp(-val)) / 2.0;
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise hyperbolic tangent of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<R> tanh<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute tanh() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for tanh',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        tanh<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = tanh<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_tanh_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_tanh_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_tanh_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_tanh_complex64(
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
            s_tanh_double(
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
            s_tanh_float(
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
            s_tanh_complex128(
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
            s_tanh_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        final val = (x as num).toDouble();
        final exp2val = math.exp(2.0 * val);
        return (exp2val - 1.0) / (exp2val + 1.0);
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise inverse hyperbolic sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<R> asinh<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute asinh() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for asinh',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        asinh<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = asinh<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_asinh_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_asinh_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_asinh_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_asinh_complex64(
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
            s_asinh_double(
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
            s_asinh_float(
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
            s_asinh_complex128(
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
            s_asinh_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        final val = (x as num).toDouble();
        return math.log(val + math.sqrt(val * val + 1.0));
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise inverse hyperbolic cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<R> acosh<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute acosh() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for acosh',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        acosh<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = acosh<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_acosh_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_acosh_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_acosh_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_acosh_complex64(
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
            s_acosh_double(
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
            s_acosh_float(
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
            s_acosh_complex128(
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
            s_acosh_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        final val = (x as num).toDouble();
        return math.log(val + math.sqrt(val * val - 1.0));
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise inverse hyperbolic tangent of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends DTypeTag`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<R> atanh<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute atanh() on a disposed array.');
  }
  final DType<DTypeTag> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype,
    DType.float32 => DType.float32,
    _ => DType.float64,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for atanh',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        atanh<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype.isInteger ||
      (a.dtype as DType<DTypeTag>) == DType.boolean ||
      (a.dtype as DType<DTypeTag>) == DType.float16 ||
      (a.dtype as DType<DTypeTag>) == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      final res = atanh<R>(
        promoted as NDArray<InexactOf<R>>,
        where: where,
        out: out,
      );
      return res;
    } finally {
      promoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, a.shape);

  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_atanh_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_atanh_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_atanh_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_atanh_complex64(
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
            s_atanh_double(
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
            s_atanh_float(
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
            s_atanh_complex128(
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
            s_atanh_complex64(
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

    unaryOp<DTypeTag, R>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        final val = (x as num).toDouble();
        return 0.5 * math.log((1.0 + val) / (1.0 - val));
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise arc tangent of [y] / [x] with full broadcasting support.
///
/// [y] and [x] share the type parameter `T`, so both must have the same dtype;
/// the result dtype [R] is the inexact promotion of that dtype (`Float32` is
/// preserved, every other real dtype computes in `Float64`). Passing arrays of
/// different concrete dtypes is a compile-time error; when the dtype equality
/// cannot be checked statically (for example with `NDArray<AnySpec>`
/// arguments), it is validated at runtime. Use [atan2As] for mixed dtypes.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> atan2<T extends InexactOf<R>, R extends DTypeTag>(
  NDArray<T> y,
  NDArray<T> x, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (y.isDisposed ||
      x.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute atan2() on a disposed array.');
  }
  final DType<DTypeTag> yDType = y.dtype;
  final DType<DTypeTag> xDType = x.dtype;
  if (yDType != xDType) {
    throw ArgumentError.value(
      x.dtype,
      'x',
      'Must have the same dtype as y (${y.dtype})',
    );
  }
  if (yDType == DType.complex128 ||
      yDType == DType.complex64 ||
      xDType == DType.complex128 ||
      xDType == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for atan2');
  }
  final broadcastResult = broadcast(y, x);
  final shape = broadcastResult.shape;
  final DType<R> targetDType =
      ((yDType == DType.float32 && xDType == DType.float32)
              ? DType.float32
              : DType.float64)
          as DType<R>;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for atan2',
      );
    }
    if (sharesMemory(y, out) ||
        sharesMemory(x, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        atan2<T, R>(y, x, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (yDType.isInteger ||
      yDType == DType.boolean ||
      yDType == DType.float16 ||
      yDType == DType.bfloat16 ||
      xDType.isInteger ||
      xDType == DType.boolean ||
      xDType == DType.float16 ||
      xDType == DType.bfloat16) {
    final yPromoted = promoteToDouble(y);
    final xPromoted = promoteToDouble(x);
    try {
      // Integer, boolean, and half-precision inputs always compute in
      // float64, so `R` is `Float64` on this path (see `targetDType` above).
      final res = atan2<Float64, Float64>(
        yPromoted,
        xPromoted,
        where: where,
        out: out as NDArray<Float64>?,
      );
      return res as NDArray<R>;
    } finally {
      if (!identical(yPromoted, y)) yPromoted.dispose();
      if (!identical(xPromoted, x)) xPromoted.dispose();
    }
  }
  final maskHolder = prepareMask(where, shape);

  try {
    final NDArray<R> result =
        out ?? NDArray<R>.create(shape, targetDType, zeroInit: where != null);
    // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
    if (y.isContiguous &&
        x.isContiguous &&
        result.isContiguous &&
        listEquals(y.shape, x.shape)) {
      switch (targetDType) {
        case DType.float64:
          if (yDType == DType.float64 && xDType == DType.float64) {
            v_atan2_double(
              y.pointer.cast(),
              x.pointer.cast(),
              result.pointer.cast(),
              y.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (yDType == DType.float32 && xDType == DType.float32) {
            v_atan2_float(
              y.pointer.cast(),
              x.pointer.cast(),
              result.pointer.cast(),
              y.size,
              maskHolder.pointer,
            );
            return result;
          }
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
    }
    final stridesY = broadcastResult.stridesA;
    final stridesX = broadcastResult.stridesB;

    // 0C. General Multidimensional Strided Broadcasting Engine in C (Rank <= 8)
    if (shape.length <= 8) {
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(shape);
        final cStridesY = ScratchArena.copyInts(stridesY);
        final cStridesX = ScratchArena.copyInts(stridesX);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        switch (targetDType) {
          case DType.float64:
            if (yDType == DType.float64 && xDType == DType.float64) {
              s_atan2_double(
                y.pointer.cast(),
                cStridesY,
                x.pointer.cast(),
                cStridesX,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                shape.length,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float32:
            if (yDType == DType.float32 && xDType == DType.float32) {
              s_atan2_float(
                y.pointer.cast(),
                cStridesY,
                x.pointer.cast(),
                cStridesX,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                shape.length,
                maskHolder.pointer,
              );
              return result;
            }
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

    elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
      result,
      y,
      x,
      shape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      result.strides,
      0,
      y.offsetElements,
      x.offsetElements,
      result.offsetElements,
      (a, b) => math.atan2(
        (a is bool ? (a ? 1.0 : 0.0) : (a as num).toDouble()),
        (b is bool ? (b ? 1.0 : 0.0) : (b as num).toDouble()),
      ),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise four-quadrant inverse tangent of [y] / [x] into the specified target [dtype].
///
/// Casts operands or result to [dtype] and returns an [NDArray<R>] whose static
/// type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [y], [x], [where], or [out] is disposed.
/// - Complex numbers are not supported for [y], [x], or [dtype].
/// - If [out] is provided, it must be writeable, have the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy arctan2](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html)
NDArray<R>
atan2As<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> y,
  NDArray<Tb> x,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (y.isDisposed ||
      x.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute atan2As() on a disposed array.');
  }
  if (y.dtype.isComplex || x.dtype.isComplex || dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for atan2');
  }
  return NDArray.scope(() {
    if (dtype == DType.float32) {
      final yCast = castNDArray<Float32>(y, DType.float32);
      final xCast = castNDArray<Float32>(x, DType.float32);
      final res = atan2<Float32, Float32>(
        yCast,
        xCast,
        where: where,
        out: out as NDArray<Float32>?,
      );
      return (out ?? res.detachToParentScope()) as NDArray<R>;
    }
    if (dtype == DType.float64) {
      final yCast = castNDArray<Float64>(y, DType.float64);
      final xCast = castNDArray<Float64>(x, DType.float64);
      final res = atan2<Float64, Float64>(
        yCast,
        xCast,
        where: where,
        out: out as NDArray<Float64>?,
      );
      return (out ?? res.detachToParentScope()) as NDArray<R>;
    }
    final shape = broadcast(y, x).shape;
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, shape) || out.dtype != dtype) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape and dtype for atan2As',
        );
      }
    }
    final yCast = castNDArray<Float64>(y, DType.float64);
    final xCast = castNDArray<Float64>(x, DType.float64);
    final f64Res = atan2<Float64, Float64>(yCast, xCast, where: where);
    final casted = castNDArray<R>(f64Res, dtype);
    if (out != null) {
      final maskHolder = prepareMask(where, out.shape);
      try {
        unaryOp<DTypeTag, DTypeTag>(
          out,
          casted,
          out.shape,
          casted.strides,
          out.strides,
          0,
          casted.offsetElements,
          out.offsetElements,
          (v) => v,
          maskHolder.pointer,
        );
      } finally {
        maskHolder.dispose();
      }
      return out;
    }
    return casted.detachToParentScope();
  });
}

/// Computes the element-wise hypotenuse `sqrt(x1**2 + x2**2)` with broadcasting support.
///
/// [a] and [b] share the type parameter `T`, so both must have the same dtype;
/// the result dtype [R] is the inexact promotion of that dtype (`Float32` is
/// preserved, every other real dtype computes in `Float64`). Passing arrays of
/// different concrete dtypes is a compile-time error; when the dtype equality
/// cannot be checked statically (for example with `NDArray<AnySpec>`
/// arguments), it is validated at runtime. Use [hypotAs] for mixed dtypes.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> hypot<T extends InexactOf<R>, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute hypot() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for hypot');
  }
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final resType = resolveDType(a.dtype, b.dtype);
  final DType<R> targetDType =
      (resType == DType.float32 ? DType.float32 : DType.float64) as DType<R>;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for hypot',
      );
    }
    if (sharesMemory(a, out) ||
        sharesMemory(b, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        hypot<T, R>(a, b, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, shape);
  try {
    final NDArray<R> result =
        out ?? NDArray<R>.create(shape, targetDType, zeroInit: where != null);

    double hypotOp(double x, double y) {
      if (x.isInfinite || y.isInfinite) return double.infinity;
      x = x.abs();
      y = y.abs();
      if (x < y) {
        final temp = x;
        x = y;
        y = temp;
      }
      if (x == 0) return 0.0;
      final t = y / x;
      return x * math.sqrt(1.0 + t * t);
    }

    elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
      result,
      a,
      b,
      shape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      result.strides,
      0,
      a.offsetElements,
      b.offsetElements,
      result.offsetElements,
      (valA, valB) {
        final da = (a.dtype as DType<DTypeTag>) == DType.uint64
            ? uint64ToDouble(valA as int)
            : (valA is bool ? (valA ? 1.0 : 0.0) : (valA as num).toDouble());
        final db = (b.dtype as DType<DTypeTag>) == DType.uint64
            ? uint64ToDouble(valB as int)
            : (valB is bool ? (valB ? 1.0 : 0.0) : (valB as num).toDouble());
        return castValue(hypotOp(da, db), result.dtype);
      },
      maskHolder.pointer,
    );

    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise hypotenuse `sqrt(a**2 + b**2)` into the specified target [dtype].
///
/// Casts operands or result to [dtype] and returns an [NDArray<R>] whose static
/// type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - Complex numbers are not supported for [a], [b], or [dtype].
/// - If [out] is provided, it must be writeable, have the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy hypot](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html)
NDArray<R>
hypotAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute hypotAs() on a disposed array.');
  }
  if (a.dtype.isComplex || b.dtype.isComplex || dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for hypot');
  }
  return NDArray.scope(() {
    if (dtype == DType.float32) {
      final aCast = castNDArray<Float32>(a, DType.float32);
      final bCast = castNDArray<Float32>(b, DType.float32);
      final res = hypot<Float32, Float32>(
        aCast,
        bCast,
        where: where,
        out: out as NDArray<Float32>?,
      );
      return (out ?? res.detachToParentScope()) as NDArray<R>;
    }
    if (dtype == DType.float64) {
      final aCast = castNDArray<Float64>(a, DType.float64);
      final bCast = castNDArray<Float64>(b, DType.float64);
      final res = hypot<Float64, Float64>(
        aCast,
        bCast,
        where: where,
        out: out as NDArray<Float64>?,
      );
      return (out ?? res.detachToParentScope()) as NDArray<R>;
    }
    final shape = broadcast(a, b).shape;
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, shape) || out.dtype != dtype) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape and dtype for hypotAs',
        );
      }
    }
    final aCast = castNDArray<Float64>(a, DType.float64);
    final bCast = castNDArray<Float64>(b, DType.float64);
    final f64Res = hypot<Float64, Float64>(aCast, bCast, where: where);
    final casted = castNDArray<R>(f64Res, dtype);
    if (out != null) {
      final maskHolder = prepareMask(where, out.shape);
      try {
        unaryOp<DTypeTag, DTypeTag>(
          out,
          casted,
          out.shape,
          casted.strides,
          out.strides,
          0,
          casted.offsetElements,
          out.offsetElements,
          (v) => v,
          maskHolder.pointer,
        );
      } finally {
        maskHolder.dispose();
      }
      return out;
    }
    return casted.detachToParentScope();
  });
}

/// Converts angles from degrees to radians element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> deg2rad<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute deg2rad() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for deg2rad');
  }

  final targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
      ? DType.float32
      : DType.float64;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for deg2rad',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        deg2rad<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final aPromoted = (a.dtype as DType<DTypeTag>) == targetDType
      ? a
      : castNDArray(a, targetDType);
  final factor = NDArray.fromList([0.017453292519943295], [], targetDType);
  try {
    return multiply<DTypeTag>(aPromoted, factor, where: where, out: out)
        as NDArray<R>;
  } finally {
    factor.dispose();
    if (!identical(aPromoted, a)) aPromoted.dispose();
  }
}

/// Converts angles from radians to degrees element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> rad2deg<R extends DTypeTag>(
  NDArray<InexactOf<R>> a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute rad2deg() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for rad2deg');
  }

  final targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
      ? DType.float32
      : DType.float64;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for rad2deg',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        rad2deg<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final aPromoted = (a.dtype as DType<DTypeTag>) == targetDType
      ? a
      : castNDArray(a, targetDType);
  final factor = NDArray.fromList([57.29577951308232], [], targetDType);
  try {
    return multiply<DTypeTag>(aPromoted, factor, where: where, out: out)
        as NDArray<R>;
  } finally {
    factor.dispose();
    if (!identical(aPromoted, a)) aPromoted.dispose();
  }
}
