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

/// Returns a boolean array of the same shape as [a] containing the element-wise logical negation.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If provided, the [out] recycler array must exactly match [a]'s shape and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if the input array [a] has been disposed.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_not`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_not](https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html)
NDArray<Boolean> logicalNot<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logicalNot() on a disposed array.');
  }
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for logicalNot.',
      );
    }
  }
  final marker = ScratchArena.marker;
  try {
    final maskHolder = prepareMask(where, a.shape);
    try {
      final NDArray<Boolean> result =
          out ??
          NDArray<Boolean>.create(
            a.shape,
            DType.boolean,
            zeroInit: where != null,
          );
      final ffi.Pointer<ffi.Uint8> aBoolPtr;
      final List<int> aBoolStrides;
      if (a.dtype == DType.boolean) {
        aBoolPtr = a.pointer.cast();
        aBoolStrides = a.strides;
      } else {
        aBoolPtr = ScratchArena.allocate<ffi.Uint8>(a.size);
        aBoolStrides = NDArray.computeCStrides(a.shape);
        if (a.isContiguous) {
          switch (a.dtype) {
            case DType.float64:
              v_to_bool_double(a.pointer.cast(), aBoolPtr, a.size, ffi.nullptr);
            case DType.float32:
              v_to_bool_float(a.pointer.cast(), aBoolPtr, a.size, ffi.nullptr);
            case DType.int64:
            case DType.uint64:
              v_to_bool_int64(a.pointer.cast(), aBoolPtr, a.size, ffi.nullptr);
            case DType.int32:
            case DType.uint32:
              v_to_bool_int32(a.pointer.cast(), aBoolPtr, a.size, ffi.nullptr);
            case DType.uint8:
            case DType.int8:
              v_to_bool_uint8(a.pointer.cast(), aBoolPtr, a.size, ffi.nullptr);
            case DType.int16:
            case DType.uint16:
              v_to_bool_int16(a.pointer.cast(), aBoolPtr, a.size, ffi.nullptr);
            case DType.float16:
            case DType.bfloat16:
              final doubleA = castNDArray(a, DType.float64);
              try {
                v_to_bool_double(
                  doubleA.pointer.cast(),
                  aBoolPtr,
                  a.size,
                  ffi.nullptr,
                );
              } finally {
                doubleA.dispose();
              }
            case DType.complex128:
              v_to_bool_complex128(
                a.pointer.cast(),
                aBoolPtr,
                a.size,
                ffi.nullptr,
              );
            case DType.complex64:
              v_to_bool_complex64(
                a.pointer.cast(),
                aBoolPtr,
                a.size,
                ffi.nullptr,
              );
            case DType.boolean:
              break;
          }
        } else {
          final ndim = a.shape.length;
          final cBuffer = ScratchArena.getStridedBuffer(ndim * 3);
          final cShape = cBuffer;
          final cStridesA = cBuffer + ndim;
          final cStridesTemp = cBuffer + (ndim * 2);
          for (var i = 0; i < ndim; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesTemp[i] = aBoolStrides[i];
          }
          switch (a.dtype) {
            case DType.float64:
              s_to_bool_double(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.float32:
              s_to_bool_float(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.int64:
            case DType.uint64:
              s_to_bool_int64(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.int32:
            case DType.uint32:
              s_to_bool_int32(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.uint8:
            case DType.int8:
              s_to_bool_uint8(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.int16:
            case DType.uint16:
              s_to_bool_int16(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.float16:
            case DType.bfloat16:
              final doubleA = castNDArray(a, DType.float64);
              try {
                final doubleStridesA = ScratchArena.copyInts(doubleA.strides);
                s_to_bool_double(
                  doubleA.pointer.cast(),
                  doubleStridesA,
                  aBoolPtr,
                  cStridesTemp,
                  cShape,
                  ndim,
                  ffi.nullptr,
                );
              } finally {
                doubleA.dispose();
              }
            case DType.complex128:
              s_to_bool_complex128(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.complex64:
              s_to_bool_complex64(
                a.pointer.cast(),
                cStridesA,
                aBoolPtr,
                cStridesTemp,
                cShape,
                ndim,
                ffi.nullptr,
              );
            case DType.boolean:
              break;
          }
        }
      }

      if (a.isContiguous && result.isContiguous) {
        v_logical_not(
          aBoolPtr,
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
      } else {
        final ndim = a.shape.length;
        final cBuffer = ScratchArena.getStridedBuffer(ndim);
        final cShape = cBuffer;
        final cStridesA = cBuffer + ndim;
        final cStridesRes = cBuffer + (ndim * 2);

        for (var i = 0; i < ndim; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = aBoolStrides[i];
          cStridesRes[i] = result.strides[i];
        }

        s_logical_not(
          aBoolPtr,
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          ndim,
          maskHolder.pointer,
        );
      }

      return result;
    } finally {
      maskHolder.dispose();
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the element-wise equality of [a] == [b] with broadcasting support.
///
/// **Preconditions:**
/// - It is an error if [a] or [b] has been disposed.
/// - It is an error if [out] (if provided) is disposed, has incompatible shape, or dtype is not [DType.boolean].
/// - It is an error if shapes of [a] and [b] cannot be broadcast together.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
/// final b = NDArray.fromList([1, 4, 3], [3], DType.int32);
/// final eq = equal(a, b); // [true, false, true]
/// ```
///
/// Reference: [NumPy equal](https://numpy.org/doc/stable/reference/generated/numpy.equal.html)
NDArray<Boolean> equal<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute equal() on a disposed array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  return _compareHelper(
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_EQ,
    where: where,
    out: out,
  );
}

/// Computes the element-wise inequality of [a] != [b] with broadcasting support.
///
/// **Preconditions:**
/// - It is an error if [a] or [b] has been disposed.
/// - It is an error if [out] (if provided) is disposed, has incompatible shape, or dtype is not [DType.boolean].
/// - It is an error if shapes of [a] and [b] cannot be broadcast together.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
/// final b = NDArray.fromList([1, 4, 3], [3], DType.int32);
/// final neq = notEqual(a, b); // [false, true, false]
/// ```
///
/// Reference: [NumPy not_equal](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html)
NDArray<Boolean> notEqual<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute notEqual() on a disposed array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  return _compareHelper(
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_NE,
    where: where,
    out: out,
  );
}

/// Computes the element-wise comparison of [a] > [b] with broadcasting support.
///
/// **Preconditions:**
/// - It is an error if [a] or [b] has been disposed.
/// - It is an error if [out] (if provided) is disposed, has incompatible shape, or dtype is not [DType.boolean].
/// - It is an error if shapes of [a] and [b] cannot be broadcast together.
///
/// Throws [UnsupportedError] if either array contains complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([5, 2, 8], [3], DType.int32);
/// final b = NDArray.fromList([3, 4, 1], [3], DType.int32);
/// final gt = greater(a, b); // [true, false, true]
/// ```
///
/// Reference: [NumPy greater](https://numpy.org/doc/stable/reference/generated/numpy.greater.html)
NDArray<Boolean> greater<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute greater() on a disposed array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  return _compareHelper(
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_GT,
    where: where,
    out: out,
  );
}

/// Computes the element-wise comparison of [a] >= [b] with broadcasting support.
///
/// **Preconditions:**
/// - It is an error if [a] or [b] has been disposed.
/// - It is an error if [out] (if provided) is disposed, has incompatible shape, or dtype is not [DType.boolean].
/// - It is an error if shapes of [a] and [b] cannot be broadcast together.
///
/// Throws [UnsupportedError] if either array contains complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([5, 2, 8], [3], DType.int32);
/// final b = NDArray.fromList([3, 2, 9], [3], DType.int32);
/// final ge = greaterEqual(a, b); // [true, true, false]
/// ```
///
/// Reference: [NumPy greater_equal](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html)
NDArray<Boolean> greaterEqual<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute greaterEqual() on a disposed array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  return _compareHelper(
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_GE,
    where: where,
    out: out,
  );
}

/// Computes the element-wise comparison of [a] < [b] with broadcasting support.
///
/// **Preconditions:**
/// - It is an error if [a] or [b] has been disposed.
/// - It is an error if [out] (if provided) is disposed, has incompatible shape, or dtype is not [DType.boolean].
/// - It is an error if shapes of [a] and [b] cannot be broadcast together.
///
/// Throws [UnsupportedError] if either array contains complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 5, 3], [3], DType.int32);
/// final b = NDArray.fromList([3, 2, 4], [3], DType.int32);
/// final lt = less(a, b); // [true, false, true]
/// ```
///
/// Reference: [NumPy less](https://numpy.org/doc/stable/reference/generated/numpy.less.html)
NDArray<Boolean> less<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute less() on a disposed array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  return _compareHelper(
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_LT,
    where: where,
    out: out,
  );
}

/// Computes the element-wise comparison of [a] <= [b] with broadcasting support.
///
/// **Preconditions:**
/// - It is an error if [a] or [b] has been disposed.
/// - It is an error if [out] (if provided) is disposed, has incompatible shape, or dtype is not [DType.boolean].
/// - It is an error if shapes of [a] and [b] cannot be broadcast together.
///
/// Throws [UnsupportedError] if either array contains complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 5], [3], DType.int32);
/// final b = NDArray.fromList([3, 2, 4], [3], DType.int32);
/// final le = lessEqual(a, b); // [true, true, false]
/// ```
///
/// Reference: [NumPy less_equal](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html)
NDArray<Boolean> lessEqual<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute lessEqual() on a disposed array.");
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  return _compareHelper(
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_LE,
    where: where,
    out: out,
  );
}

NDArray<Boolean> _compareHelper(
  NDArray a,
  NDArray b,
  List<int> commonShape,
  List<int> stridesA,
  List<int> stridesB,
  int op, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  final rank = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    final maskHolder = where != null ? prepareMask(where, commonShape) : null;
    try {
      final result =
          out ??
          NDArray<Boolean>.create(
            commonShape,
            DType.boolean,
            zeroInit: where != null,
          );
      final cShape = commonShape.isEmpty
          ? ffi.nullptr
          : ScratchArena.copyInts(commonShape);
      final cStridesA = stridesA.isEmpty
          ? ffi.nullptr
          : ScratchArena.copyInts(stridesA);
      final cStridesB = stridesB.isEmpty
          ? ffi.nullptr
          : ScratchArena.copyInts(stridesB);
      final cStridesRes = result.strides.isEmpty
          ? ffi.nullptr
          : ScratchArena.copyInts(result.strides);

      if (maskHolder == null) {
        if (out != null &&
            (sharesMemory(a, result) || sharesMemory(b, result))) {
          final tempRes = NDArray<Boolean>.create(commonShape, DType.boolean);
          try {
            ndarray_compare(
              op,
              a.dtype.index,
              b.dtype.index,
              a.pointer.cast(),
              cStridesA,
              b.pointer.cast(),
              cStridesB,
              tempRes.pointer.cast(),
              tempRes.strides.isEmpty
                  ? ffi.nullptr
                  : ScratchArena.copyInts(tempRes.strides),
              cShape,
              rank,
            );
            tempRes.copy(out: result);
          } finally {
            tempRes.dispose();
          }
        } else {
          ndarray_compare(
            op,
            a.dtype.index,
            b.dtype.index,
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        }
      } else {
        final tempRes = NDArray<Boolean>.create(commonShape, DType.boolean);
        try {
          ndarray_compare(
            op,
            a.dtype.index,
            b.dtype.index,
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            tempRes.pointer.cast(),
            tempRes.strides.isEmpty
                ? ffi.nullptr
                : ScratchArena.copyInts(tempRes.strides),
            cShape,
            rank,
          );
          unaryOp<Boolean, Boolean>(
            result,
            tempRes,
            commonShape,
            tempRes.strides,
            result.strides,
            0,
            tempRes.offsetElements,
            result.offsetElements,
            (x) => x,
            maskHolder.pointer,
          );
        } finally {
          tempRes.dispose();
        }
      }
      return result;
    } finally {
      maskHolder?.dispose();
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the element-wise truth value of [a] AND [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical AND results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_and`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_and](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html)
NDArray<Boolean> logicalAnd<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logicalAnd() on a disposed array.');
  }
  return _runBinaryLogical(
    a,
    b,
    where,
    out,
    v_logical_and,
    s_logical_and,
    'logicalAnd',
  );
}

/// Computes the element-wise truth value of [a] OR [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical OR results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_or`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_or](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html)
NDArray<Boolean> logicalOr<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logicalOr() on a disposed array.');
  }
  return _runBinaryLogical(
    a,
    b,
    where,
    out,
    v_logical_or,
    s_logical_or,
    'logicalOr',
  );
}

/// Computes the element-wise truth value of [a] XOR [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical XOR results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_xor`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_xor](https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html)
NDArray<Boolean> logicalXor<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logicalXor() on a disposed array.');
  }
  return _runBinaryLogical(
    a,
    b,
    where,
    out,
    v_logical_xor,
    s_logical_xor,
    'logicalXor',
  );
}

BroadcastResult _broadcastBinaryStrides(
  List<int> shapeA,
  List<int> stridesA,
  List<int> shapeB,
  List<int> stridesB,
) {
  if (listEquals(shapeA, shapeB)) {
    return BroadcastResult(shapeA, stridesA, stridesB);
  }
  final maxLen = shapeA.length > shapeB.length ? shapeA.length : shapeB.length;
  final commonShape = List<int>.filled(maxLen, 1);
  final newStridesA = List<int>.filled(maxLen, 0);
  final newStridesB = List<int>.filled(maxLen, 0);

  for (var i = 0; i < maxLen; i++) {
    final dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
    final dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;

    if (dimA == dimB) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimA == 1) {
      commonShape[maxLen - 1 - i] = dimB;
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimB == 1) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
    } else {
      throw ArgumentError(
        'Shapes $shapeA and $shapeB are not compatible for broadcasting',
      );
    }
  }
  return BroadcastResult(commonShape, newStridesA, newStridesB);
}

ffi.Pointer<ffi.Uint8> _castToBoolean(
  NDArray x,
  ffi.Pointer<ffi.Uint8> destPtr,
  List<int> destStrides,
) {
  if (x.isContiguous) {
    switch (x.dtype) {
      case DType.float64:
        v_to_bool_double(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.float32:
        v_to_bool_float(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.int64:
      case DType.uint64:
        v_to_bool_int64(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.int32:
      case DType.uint32:
        v_to_bool_int32(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.uint8:
      case DType.int8:
        v_to_bool_uint8(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.int16:
      case DType.uint16:
        v_to_bool_int16(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.float16:
      case DType.bfloat16:
        final doubleX = castNDArray(x, DType.float64);
        try {
          v_to_bool_double(
            doubleX.pointer.cast(),
            destPtr,
            x.size,
            ffi.nullptr,
          );
        } finally {
          doubleX.dispose();
        }
      case DType.complex128:
        v_to_bool_complex128(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.complex64:
        v_to_bool_complex64(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
      case DType.boolean:
        v_to_bool_uint8(x.pointer.cast(), destPtr, x.size, ffi.nullptr);
    }
  } else {
    final ndim = x.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesX = cBuffer + ndim;
    final cStridesTemp = cBuffer + (ndim * 2);
    for (var i = 0; i < ndim; i++) {
      cShape[i] = x.shape[i];
      cStridesX[i] = x.strides[i];
      cStridesTemp[i] = destStrides[i];
    }
    switch (x.dtype) {
      case DType.float64:
        s_to_bool_double(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.float32:
        s_to_bool_float(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.int64:
      case DType.uint64:
        s_to_bool_int64(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.int32:
      case DType.uint32:
        s_to_bool_int32(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.uint8:
      case DType.int8:
        s_to_bool_uint8(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.int16:
      case DType.uint16:
        s_to_bool_int16(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.float16:
      case DType.bfloat16:
        final doubleX = castNDArray(x, DType.float64);
        try {
          final doubleStridesX = ScratchArena.copyInts(doubleX.strides);
          s_to_bool_double(
            doubleX.pointer.cast(),
            doubleStridesX,
            destPtr,
            cStridesTemp,
            cShape,
            ndim,
            ffi.nullptr,
          );
        } finally {
          doubleX.dispose();
        }
      case DType.complex128:
        s_to_bool_complex128(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.complex64:
        s_to_bool_complex64(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
      case DType.boolean:
        s_to_bool_uint8(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
          ffi.nullptr,
        );
    }
  }
  return destPtr;
}

NDArray<Boolean> _runBinaryLogical<Ta extends DTypeTag, Tb extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  NDArray<DTypeTag>? where,
  NDArray<Boolean>? out,
  void Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Uint8>,
    int,
    ffi.Pointer<ffi.Uint8>,
  )
  contiguousFn,
  void Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    int,
    ffi.Pointer<ffi.Uint8>,
  )
  stridedFn,
  String opName,
) {
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

  final marker = ScratchArena.marker;
  try {
    final ffi.Pointer<ffi.Uint8> aBoolPtr;
    final List<int> aBoolStrides;
    if (a.dtype == DType.boolean) {
      aBoolPtr = a.pointer.cast();
      aBoolStrides = a.strides;
    } else {
      aBoolPtr = ScratchArena.allocate<ffi.Uint8>(a.size);
      aBoolStrides = NDArray.computeCStrides(a.shape);
      _castToBoolean(a, aBoolPtr, aBoolStrides);
    }

    final ffi.Pointer<ffi.Uint8> bBoolPtr;
    final List<int> bBoolStrides;
    if (b.dtype == DType.boolean) {
      bBoolPtr = b.pointer.cast();
      bBoolStrides = b.strides;
    } else {
      bBoolPtr = ScratchArena.allocate<ffi.Uint8>(b.size);
      bBoolStrides = NDArray.computeCStrides(b.shape);
      _castToBoolean(b, bBoolPtr, bBoolStrides);
    }

    final broadcastResult = _broadcastBinaryStrides(
      a.shape,
      aBoolStrides,
      b.shape,
      bBoolStrides,
    );
    final commonShape = broadcastResult.shape;
    final stridesA = broadcastResult.stridesA;
    final stridesB = broadcastResult.stridesB;

    if (out != null) {
      if (!out.isWriteable ||
          !listEquals(out.shape, commonShape) ||
          out.dtype != DType.boolean) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype for $opName.',
        );
      }
    }

    final maskHolder = prepareMask(where, commonShape);
    try {
      final NDArray<Boolean> result =
          out ??
          NDArray<Boolean>.create(
            commonShape,
            DType.boolean,
            zeroInit: where != null,
          );
      final isContig =
          a.isContiguous &&
          b.isContiguous &&
          result.isContiguous &&
          listEquals(a.shape, b.shape);

      if (isContig) {
        contiguousFn(
          aBoolPtr,
          bBoolPtr,
          result.pointer.cast(),
          result.size,
          maskHolder.pointer,
        );
      } else {
        final ndim = commonShape.length;
        final cBuffer = ScratchArena.getStridedBuffer(ndim);
        final cShape = cBuffer;
        final cStridesA = cBuffer + ndim;
        final cStridesB = cBuffer + (ndim * 2);
        final cStridesRes = cBuffer + (ndim * 3);

        for (var i = 0; i < ndim; i++) {
          cShape[i] = commonShape[i];
          cStridesA[i] = stridesA[i];
          cStridesB[i] = stridesB[i];
          cStridesRes[i] = result.strides[i];
        }

        stridedFn(
          aBoolPtr,
          cStridesA,
          bBoolPtr,
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          ndim,
          maskHolder.pointer,
        );
      }

      return result;
    } finally {
      maskHolder.dispose();
    }
  } finally {
    ScratchArena.reset(marker);
  }
}
