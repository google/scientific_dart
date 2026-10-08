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
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../broadcasting.dart';
import '../helpers.dart';

/// Computes the bitwise AND of two arrays, element-wise.
///
/// Calculates the bitwise AND of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// It is an error if:
/// - [a] or [b] is disposed (throws [StateError]).
/// - [a] or [b] is not integer-typed (throws [ArgumentError]).
/// - shapes are incompatible for broadcasting (throws [ArgumentError]).
/// - [out] shape or dtype is incompatible (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_and](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html)
NDArray<T> bitwiseAnd<T extends BitwiseDType>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute bitwiseAnd() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<T>(
    a,
    b,
    where,
    out,
    'bitwiseAnd',
    allowBool: true,
  );
  final maskHolder = prep.maskHolder;
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
        case DType.uint32:
          v_bitwise_and_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int64:
        case DType.uint64:
          v_bitwise_and_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.uint8:
        case DType.int8:
          v_bitwise_and_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int16:
        case DType.uint16:
          v_bitwise_and_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.boolean:
          v_logical_and(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.complex128:
        case DType.complex64:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);

        for (var i = 0; i < rank; i++) {
          cShape[i] = prep.commonShape[i];
          cStridesA[i] = prep.stridesA[i];
          cStridesB[i] = prep.stridesB[i];
          cStridesRes[i] = prep.result.strides[i];
        }

        switch (result.dtype) {
          case DType.int32:
          case DType.uint32:
            s_bitwise_and_int32(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int64:
          case DType.uint64:
            s_bitwise_and_int64(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.uint8:
          case DType.int8:
            s_bitwise_and_uint8(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int16:
          case DType.uint16:
            s_bitwise_and_int16(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.boolean:
            s_logical_and(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.complex128:
          case DType.complex64:
            throw UnsupportedError(
              'Unsupported integer DType: ${result.dtype}',
            );
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
    if (out != null && !identical(result, out)) {
      result.copy(out: out);
    }
  } finally {
    prep.maskHolder.dispose();
    if (out != null && !identical(result, out)) {
      result.dispose();
    }
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return out ?? result;
}

/// Computes the bitwise OR of two arrays, element-wise.
///
/// Calculates the bitwise OR of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// It is an error if:
/// - [a] or [b] is disposed (throws [StateError]).
/// - [a] or [b] is not integer-typed (throws [ArgumentError]).
/// - shapes are incompatible for broadcasting (throws [ArgumentError]).
/// - [out] shape or dtype is incompatible (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_or](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html)
NDArray<T> bitwiseOr<T extends BitwiseDType>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute bitwiseOr() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<T>(
    a,
    b,
    where,
    out,
    'bitwiseOr',
    allowBool: true,
  );
  final maskHolder = prep.maskHolder;
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
        case DType.uint32:
          v_bitwise_or_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int64:
        case DType.uint64:
          v_bitwise_or_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.uint8:
        case DType.int8:
          v_bitwise_or_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int16:
        case DType.uint16:
          v_bitwise_or_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.boolean:
          v_logical_or(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.complex128:
        case DType.complex64:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);

        for (var i = 0; i < rank; i++) {
          cShape[i] = prep.commonShape[i];
          cStridesA[i] = prep.stridesA[i];
          cStridesB[i] = prep.stridesB[i];
          cStridesRes[i] = prep.result.strides[i];
        }

        switch (result.dtype) {
          case DType.int32:
          case DType.uint32:
            s_bitwise_or_int32(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int64:
          case DType.uint64:
            s_bitwise_or_int64(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.uint8:
          case DType.int8:
            s_bitwise_or_uint8(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int16:
          case DType.uint16:
            s_bitwise_or_int16(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.boolean:
            s_logical_or(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.complex128:
          case DType.complex64:
            throw UnsupportedError(
              'Unsupported integer DType: ${result.dtype}',
            );
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
    if (out != null && !identical(result, out)) {
      result.copy(out: out);
    }
  } finally {
    prep.maskHolder.dispose();
    if (out != null && !identical(result, out)) {
      result.dispose();
    }
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return out ?? result;
}

/// Computes the bitwise XOR of two arrays, element-wise.
///
/// Calculates the bitwise XOR of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// It is an error if:
/// - [a] or [b] is disposed (throws [StateError]).
/// - [a] or [b] is not integer-typed (throws [ArgumentError]).
/// - shapes are incompatible for broadcasting (throws [ArgumentError]).
/// - [out] shape or dtype is incompatible (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_xor](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html)
NDArray<T> bitwiseXor<T extends BitwiseDType>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute bitwiseXor() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<T>(
    a,
    b,
    where,
    out,
    'bitwiseXor',
    allowBool: true,
  );
  final maskHolder = prep.maskHolder;
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
        case DType.uint32:
          v_bitwise_xor_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int64:
        case DType.uint64:
          v_bitwise_xor_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.uint8:
        case DType.int8:
          v_bitwise_xor_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int16:
        case DType.uint16:
          v_bitwise_xor_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.boolean:
          v_logical_xor(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.complex128:
        case DType.complex64:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);

        for (var i = 0; i < rank; i++) {
          cShape[i] = prep.commonShape[i];
          cStridesA[i] = prep.stridesA[i];
          cStridesB[i] = prep.stridesB[i];
          cStridesRes[i] = prep.result.strides[i];
        }

        switch (result.dtype) {
          case DType.int32:
          case DType.uint32:
            s_bitwise_xor_int32(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int64:
          case DType.uint64:
            s_bitwise_xor_int64(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.uint8:
          case DType.int8:
            s_bitwise_xor_uint8(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int16:
          case DType.uint16:
            s_bitwise_xor_int16(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.boolean:
            s_logical_xor(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.complex128:
          case DType.complex64:
            throw UnsupportedError(
              'Unsupported integer DType: ${result.dtype}',
            );
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
    if (out != null && !identical(result, out)) {
      result.copy(out: out);
    }
  } finally {
    prep.maskHolder.dispose();
    if (out != null && !identical(result, out)) {
      result.dispose();
    }
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return out ?? result;
}

/// Shift the bits of an integer to the left, element-wise.
///
/// Bits are shifted to the left by appending 0s at the right.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// It is an error if:
/// - [a] or [b] is disposed (throws [StateError]).
/// - [a] or [b] is not integer-typed (throws [ArgumentError]).
/// - shapes are incompatible for broadcasting (throws [ArgumentError]).
/// - [out] shape or dtype is incompatible (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy left_shift](https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html)
NDArray<T> leftShift<T extends IntegerDType>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute leftShift() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<T>(a, b, where, out, 'leftShift');
  final maskHolder = prep.maskHolder;
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
        case DType.uint32:
          v_left_shift_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int64:
        case DType.uint64:
          v_left_shift_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.uint8:
        case DType.int8:
          v_left_shift_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int16:
        case DType.uint16:
          v_left_shift_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);

        for (var i = 0; i < rank; i++) {
          cShape[i] = prep.commonShape[i];
          cStridesA[i] = prep.stridesA[i];
          cStridesB[i] = prep.stridesB[i];
          cStridesRes[i] = prep.result.strides[i];
        }

        switch (result.dtype) {
          case DType.int32:
          case DType.uint32:
            s_left_shift_int32(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int64:
          case DType.uint64:
            s_left_shift_int64(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.uint8:
          case DType.int8:
            s_left_shift_uint8(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int16:
          case DType.uint16:
            s_left_shift_int16(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.boolean:
          case DType.complex128:
          case DType.complex64:
            throw UnsupportedError(
              'Unsupported integer DType: ${result.dtype}',
            );
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
    if (out != null && !identical(result, out)) {
      result.copy(out: out);
    }
  } finally {
    prep.maskHolder.dispose();
    if (out != null && !identical(result, out)) {
      result.dispose();
    }
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return out ?? result;
}

int _rightShiftScalar(int a, int b, DType dtype) {
  switch (dtype) {
    case DType.int8:
      if (b < 0) return 0;
      if (b >= 8) return a < 0 ? -1 : 0;
      return a >> b;
    case DType.uint16:
      if (b < 0 || b >= 16) return 0;
      return (a & 0xFFFF) >>> b;
    case DType.uint32:
      if (b < 0 || b >= 32) return 0;
      return (a & 0xFFFFFFFF) >>> b;
    case DType.uint64:
      if (b < 0 || b >= 64) return 0;
      return a >>> b;
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.uint8:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      throw UnsupportedError('Unsupported integer DType: $dtype');
  }
}

/// Shift the bits of an integer to the right, element-wise.
///
/// Bits are shifted to the right.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// It is an error if:
/// - [a] or [b] is disposed (throws [StateError]).
/// - [a] or [b] is not integer-typed (throws [ArgumentError]).
/// - shapes are incompatible for broadcasting (throws [ArgumentError]).
/// - [out] shape or dtype is incompatible (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy right_shift](https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html)
NDArray<T> rightShift<T extends IntegerDType>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute rightShift() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<T>(a, b, where, out, 'rightShift');
  final maskHolder = prep.maskHolder;
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_right_shift_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int64:
          v_right_shift_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.uint8:
          v_right_shift_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int16:
          v_right_shift_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int8:
        case DType.uint16:
        case DType.uint32:
        case DType.uint64:
          elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
            result,
            aCast,
            bCast,
            prep.commonShape,
            prep.stridesA,
            prep.stridesB,
            result.strides,
            0,
            aCast.offsetElements,
            bCast.offsetElements,
            result.offsetElements,
            (va, vb) => _rightShiftScalar(va as int, vb as int, result.dtype),
            maskHolder.pointer,
          );
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);

        for (var i = 0; i < rank; i++) {
          cShape[i] = prep.commonShape[i];
          cStridesA[i] = prep.stridesA[i];
          cStridesB[i] = prep.stridesB[i];
          cStridesRes[i] = prep.result.strides[i];
        }

        switch (result.dtype) {
          case DType.int32:
            s_right_shift_int32(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int64:
            s_right_shift_int64(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.uint8:
            s_right_shift_uint8(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int16:
            s_right_shift_int16(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int8:
          case DType.uint16:
          case DType.uint32:
          case DType.uint64:
            elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
              result,
              aCast,
              bCast,
              prep.commonShape,
              prep.stridesA,
              prep.stridesB,
              result.strides,
              0,
              aCast.offsetElements,
              bCast.offsetElements,
              result.offsetElements,
              (va, vb) => _rightShiftScalar(va as int, vb as int, result.dtype),
              maskHolder.pointer,
            );
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.boolean:
          case DType.complex128:
          case DType.complex64:
            throw UnsupportedError(
              'Unsupported integer DType: ${result.dtype}',
            );
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
    if (out != null && !identical(result, out)) {
      result.copy(out: out);
    }
  } finally {
    prep.maskHolder.dispose();
    if (out != null && !identical(result, out)) {
      result.dispose();
    }
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return out ?? result;
}

/// Computes bitwise inversion, or bitwise NOT, element-wise.
///
/// Calculates the bitwise NOT of an integer or boolean array, element-wise.
/// For boolean arrays, computes the logical NOT.
///
/// **Preconditions:**
/// - [a] must be an integer-typed or boolean array.
/// - [a] must not be disposed.
/// - If provided, [out] must match the shape and dtype of [a].
///
/// It is an error if:
/// - [a] is disposed (throws [StateError]).
/// - [a] is not integer-typed or boolean (throws [ArgumentError]).
/// - [out] shape or dtype is incompatible (throws [ArgumentError]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy invert](https://numpy.org/doc/stable/reference/generated/numpy.invert.html)
NDArray<T> invert<T extends BitwiseDType>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute invert() on a disposed array.');
  }

  if (!a.dtype.isInteger && a.dtype != DType.boolean) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      'Must be integer or boolean data type for bitwise operations',
    );
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for invert',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        invert<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  final NDArray<T> result;
  try {
    result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    if (a.isContiguous && result.isContiguous) {
      final size = a.size;
      switch (a.dtype) {
        case DType.int32:
        case DType.uint32:
          v_invert_int32(
            a.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int64:
        case DType.uint64:
          v_invert_int64(
            a.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.uint8:
        case DType.int8:
          v_invert_uint8(
            a.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.int16:
        case DType.uint16:
          v_invert_int16(
            a.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.boolean:
          v_logical_not(
            a.pointer.cast(),
            result.pointer.cast(),
            size,
            maskHolder.pointer,
          );
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.complex128:
        case DType.complex64:
          throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
      }
    } else {
      final rank = a.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesSrc = cBuffer + rank;
        final cStridesRes = cBuffer + (rank * 2);

        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesSrc[i] = a.strides[i];
          cStridesRes[i] = result.strides[i];
        }

        switch (a.dtype) {
          case DType.int32:
          case DType.uint32:
            s_invert_int32(
              a.pointer.cast(),
              cStridesSrc,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int64:
          case DType.uint64:
            s_invert_int64(
              a.pointer.cast(),
              cStridesSrc,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.uint8:
          case DType.int8:
            s_invert_uint8(
              a.pointer.cast(),
              cStridesSrc,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int16:
          case DType.uint16:
            s_invert_int16(
              a.pointer.cast(),
              cStridesSrc,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.boolean:
            s_logical_not(
              a.pointer.cast(),
              cStridesSrc,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.complex128:
          case DType.complex64:
            throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
  } finally {
    maskHolder.dispose();
  }

  return result;
}

/// Computes the element-wise bitwise AND of [a] and [b] into the specified target integer [dtype].
///
/// Casts integer operands [a] and [b] to [dtype] and computes [bitwiseAnd].
///
/// **Preconditions:**
/// - [a], [b], and [dtype] must be integer data types.
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - If [out] is provided, it must be writeable, match the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_and](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html)
NDArray<R>
bitwiseAndAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
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
    throw StateError('Cannot execute bitwiseAndAs() on a disposed array.');
  }
  _validateBitwiseAsDTypes(a.dtype, b.dtype, dtype, allowBool: true);
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return bitwiseAnd<BitwiseDType>(
          a.asBitwiseDType,
          b.asBitwiseDType,
          where: where,
          out: out?.asBitwiseDType,
        )
        as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = bitwiseAnd<BitwiseDType>(
      aCast.asBitwiseDType,
      bCast.asBitwiseDType,
      where: where,
      out: out?.asBitwiseDType,
    );
    return out ?? (res.detachToParentScope() as NDArray<R>);
  });
}

/// Computes the element-wise bitwise OR of [a] and [b] into the specified target integer [dtype].
///
/// Casts integer operands [a] and [b] to [dtype] and computes [bitwiseOr].
///
/// **Preconditions:**
/// - [a], [b], and [dtype] must be integer data types.
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - If [out] is provided, it must be writeable, match the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_or](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html)
NDArray<R>
bitwiseOrAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
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
    throw StateError('Cannot execute bitwiseOrAs() on a disposed array.');
  }
  _validateBitwiseAsDTypes(a.dtype, b.dtype, dtype, allowBool: true);
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return bitwiseOr<BitwiseDType>(
          a.asBitwiseDType,
          b.asBitwiseDType,
          where: where,
          out: out?.asBitwiseDType,
        )
        as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = bitwiseOr<BitwiseDType>(
      aCast.asBitwiseDType,
      bCast.asBitwiseDType,
      where: where,
      out: out?.asBitwiseDType,
    );
    return out ?? (res.detachToParentScope() as NDArray<R>);
  });
}

/// Computes the element-wise bitwise XOR of [a] and [b] into the specified target integer [dtype].
///
/// Casts integer operands [a] and [b] to [dtype] and computes [bitwiseXor].
///
/// **Preconditions:**
/// - [a], [b], and [dtype] must be integer data types.
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - If [out] is provided, it must be writeable, match the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_xor](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html)
NDArray<R>
bitwiseXorAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
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
    throw StateError('Cannot execute bitwiseXorAs() on a disposed array.');
  }
  _validateBitwiseAsDTypes(a.dtype, b.dtype, dtype, allowBool: true);
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return bitwiseXor<BitwiseDType>(
          a.asBitwiseDType,
          b.asBitwiseDType,
          where: where,
          out: out?.asBitwiseDType,
        )
        as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = bitwiseXor<BitwiseDType>(
      aCast.asBitwiseDType,
      bCast.asBitwiseDType,
      where: where,
      out: out?.asBitwiseDType,
    );
    return out ?? (res.detachToParentScope() as NDArray<R>);
  });
}

/// Shifts the bits of [a] to the left by [b] element-wise, computed into the specified target integer [dtype].
///
/// Casts integer operands [a] and [b] to [dtype] and computes [leftShift].
///
/// **Preconditions:**
/// - [a], [b], and [dtype] must be integer data types.
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - If [out] is provided, it must be writeable, match the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy left_shift](https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html)
NDArray<R>
leftShiftAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
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
    throw StateError('Cannot execute leftShiftAs() on a disposed array.');
  }
  _validateBitwiseAsDTypes(a.dtype, b.dtype, dtype);
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return leftShift<IntegerDType>(
          a.asIntegerDType,
          b.asIntegerDType,
          where: where,
          out: out?.asIntegerDType,
        )
        as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = leftShift<IntegerDType>(
      aCast.asIntegerDType,
      bCast.asIntegerDType,
      where: where,
      out: out?.asIntegerDType,
    );
    return out ?? (res.detachToParentScope() as NDArray<R>);
  });
}

/// Shifts the bits of [a] to the right by [b] element-wise, computed into the specified target integer [dtype].
///
/// Casts integer operands [a] and [b] to [dtype] and computes [rightShift].
///
/// **Preconditions:**
/// - [a], [b], and [dtype] must be integer data types.
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - If [out] is provided, it must be writeable, match the broadcasted shape, and have dtype [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy right_shift](https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html)
NDArray<R>
rightShiftAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
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
    throw StateError('Cannot execute rightShiftAs() on a disposed array.');
  }
  _validateBitwiseAsDTypes(a.dtype, b.dtype, dtype);
  if ((a.dtype as DType<DTypeTag>) == dtype &&
      (b.dtype as DType<DTypeTag>) == dtype) {
    return rightShift<IntegerDType>(
          a.asIntegerDType,
          b.asIntegerDType,
          where: where,
          out: out?.asIntegerDType,
        )
        as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = rightShift<IntegerDType>(
      aCast.asIntegerDType,
      bCast.asIntegerDType,
      where: where,
      out: out?.asIntegerDType,
    );
    return out ?? (res.detachToParentScope() as NDArray<R>);
  });
}

void _validateBitwiseAsDTypes(
  DType<DTypeTag> aDType,
  DType<DTypeTag> bDType,
  DType<DTypeTag> targetDType, {
  bool allowBool = false,
}) {
  bool isValid(DType<DTypeTag> dt) =>
      allowBool ? (dt.isInteger || dt == DType.boolean) : dt.isInteger;
  if (!isValid(aDType) || !isValid(bDType) || !isValid(targetDType)) {
    final bad = !isValid(aDType)
        ? aDType
        : (!isValid(bDType) ? bDType : targetDType);
    final name = !isValid(aDType)
        ? 'a.dtype'
        : (!isValid(bDType) ? 'b.dtype' : 'dtype');
    throw ArgumentError.value(
      bad,
      name,
      allowBool
          ? 'Must be integer or boolean data type for bitwise operations'
          : 'Must be integer data type for bitwise operations',
    );
  }
}

({
  NDArray aCast,
  NDArray bCast,
  NDArray<T> result,
  List<int> commonShape,
  List<int> stridesA,
  List<int> stridesB,
  bool isContig,
  MaskHolder maskHolder,
})
_prepareBinaryBitwise<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b,
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
  String opName, {
  bool allowBool = false,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot perform $opName on disposed arrays.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }

  final isValid = allowBool
      ? (a.dtype.isInteger || a.dtype == DType.boolean)
      : a.dtype.isInteger;
  if (!isValid) {
    throw ArgumentError.value(
      a.dtype,
      'a.dtype',
      allowBool
          ? 'Must be integer or boolean data type for bitwise operations'
          : 'Must be integer data type for bitwise operations',
    );
  }

  final DType<T> targetDType = a.dtype;
  final preBroadcast = broadcast(a, b);
  final commonShape = preBroadcast.shape;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for $opName',
      );
    }
  }
  final maskHolder = prepareMask(where, commonShape);

  try {
    // Upcast inputs if they do not match the resolved target integer type
    final NDArray aCast = a.dtype != targetDType
        ? castNDArray(a, targetDType)
        : a;
    final NDArray bCast = b.dtype != targetDType
        ? castNDArray(b, targetDType)
        : b;

    final broadcastResult = broadcast(aCast, bCast);
    final stridesA = broadcastResult.stridesA;
    final stridesB = broadcastResult.stridesB;

    final bool needsTempOut =
        out != null &&
        (sharesMemory(aCast, out) ||
            sharesMemory(bCast, out) ||
            (where != null && sharesMemory(where, out)));

    final NDArray<T> result = needsTempOut
        ? (where != null
              ? out.copy()
              : NDArray<T>.create(commonShape, targetDType))
        : (out ??
              NDArray<T>.create(
                commonShape,
                targetDType,
                zeroInit: where != null,
              ));

    final isContig =
        aCast.isContiguous &&
        bCast.isContiguous &&
        result.isContiguous &&
        listEquals(aCast.shape, bCast.shape);

    return (
      aCast: aCast,
      bCast: bCast,
      result: result,
      commonShape: commonShape,
      stridesA: stridesA,
      stridesB: stridesB,
      isContig: isContig,
      maskHolder: maskHolder,
    );
  } catch (_) {
    maskHolder.dispose();
    rethrow;
  }
}
