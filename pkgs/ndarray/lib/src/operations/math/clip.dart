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

/// Clip (limit) the values in an array using scalar bounds.
///
/// Given an interval `[min, max]`, values outside the interval are clipped
/// to the interval edges.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - [min] must be less than or equal to [max].
/// - If provided, [out] must have the exact shape and matching [DType] of [a].
///
/// **Throws:**
/// - It is an error if [a], [out], or [where] is disposed.
/// - It is an error if [a] has a complex [DType] (complex values cannot be ordered).
/// - It is an error if [out] has an incompatible shape or [DType].
///
/// **Performance considerations:**
/// - Time complexity is $O(N)$ where $N$ is the total number of elements in [a].
/// - For contiguous arrays, uses C kernels, executing in $O(N)$ time with $O(1)$ extra memory.
/// - Otherwise, performs element-wise strided iteration in Dart.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)
NDArray<T> clip<T>(
  NDArray<T> a, {
  num? min,
  num? max,
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute clip() on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for clip');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for clip.',
      );
    }
    if (out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible DType for clip.',
      );
    }
  }
  final maskHolder = prepareMask(where, result.shape);

  try {
    final resolvedMin = min ?? _getMinLimit(a.dtype);
    final resolvedMax = max ?? _getMaxLimit(a.dtype);

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);

    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_clip_double(
            a.pointer.cast(),
            result.pointer.cast(),
            resolvedMin.toDouble(),
            resolvedMax.toDouble(),
            size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_clip_float(
            a.pointer.cast(),
            result.pointer.cast(),
            resolvedMin.toDouble(),
            resolvedMax.toDouble(),
            size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    }

    if (a.dtype.isInteger) {
      final mn = resolvedMin.toInt();
      final mx = resolvedMax.toInt();
      unaryOp<dynamic, dynamic>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => castValue((x as num).toInt().clamp(mn, mx), a.dtype),
        maskHolder.pointer,
      );
    } else {
      final mn = resolvedMin.toDouble();
      final mx = resolvedMax.toDouble();
      unaryOp<dynamic, dynamic>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => castValue((x as num).toDouble().clamp(mn, mx), a.dtype),
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Clip (limit) the values in an array using array bounds that broadcast natively against the input array.
///
/// Given array bounds [min] and [max], values outside the interval are clipped
/// to the interval edges.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - [min] and [max] must not be disposed.
/// - [min] and [max] must be of real/integer numeric types (complex/boolean bounds are not supported).
/// - The shapes of [a], [min], and [max] must be compatible for broadcasting.
/// - If provided, [out] must have the exact broadcasted shape and matching [DType] of [a].
///
/// **Throws:**
/// - It is an error if [a], [min], [max], [where], or [out] is disposed.
/// - It is an error if [a] has a complex [DType] (complex values cannot be ordered).
/// - It is an error if [min] or [max] has a complex or boolean DType.
/// - It is an error if shapes are incompatible for broadcasting, or if [out] has an incompatible shape or [DType].
///
/// **Performance considerations:**
/// - Time complexity is $O(N)$ where $N$ is the total number of elements in the broadcasted shape.
/// - Performs element-wise strided iteration in Dart using a ternary walker, requiring zero heap allocations for view creation.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 5.0, 10.0], [3], DType.float64);
/// final minBounds = NDArray.fromList([2.0, 2.0, 2.0], [3], DType.float64);
/// final maxBounds = NDArray.fromList([8.0, 8.0, 8.0], [3], DType.float64);
/// final clipped = clipArray(a, min: minBounds, max: maxBounds); // [2.0, 5.0, 8.0]
/// ```
///
/// Reference: [NumPy clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)
NDArray<T> clipArray<T>(
  NDArray<T> a, {
  NDArray<T>? min,
  NDArray<T>? max,
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (min != null && min.isDisposed) ||
      (max != null && max.isDisposed) ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute clipArray() on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for clipArray');
  }
  if (min != null && (min.dtype.isComplex || min.dtype == DType.boolean)) {
    throw ArgumentError(
      'Complex/Boolean bounds are not supported for clipArray',
    );
  }
  if (max != null && (max.dtype.isComplex || max.dtype == DType.boolean)) {
    throw ArgumentError(
      'Complex/Boolean bounds are not supported for clipArray',
    );
  }

  final bool ownsMin = min == null;
  final minArr =
      min ??
      (NDArray<T>.create([], a.dtype)
        ..setCellRaw(0, _getMinLimit(a.dtype) as T));

  final bool ownsMax = max == null;
  final maxArr =
      max ??
      (NDArray<T>.create([], a.dtype)
        ..setCellRaw(0, _getMaxLimit(a.dtype) as T));

  try {
    NDArray? dummy;
    List<int> commonShape;
    try {
      final b1 = broadcast(a, minArr);
      dummy = NDArray.create(b1.shape, a.dtype);
      final b2 = broadcast(dummy, maxArr);
      commonShape = b2.shape;
    } finally {
      dummy?.dispose();
    }

    final result = out ?? NDArray<T>.create(commonShape, a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, commonShape)) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape for clipArray.',
        );
      }
      if (out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible DType for clipArray.',
        );
      }
    }
    final maskHolder = prepareMask(where, result.shape);
    try {
      final broadcastA = broadcastTo(a, commonShape);
      final broadcastMin = broadcastTo(minArr, commonShape);
      final broadcastMax = broadcastTo(maxArr, commonShape);

      final marker = ScratchArena.marker;
      try {
        final ndim = commonShape.length;
        final cBuffer = ScratchArena.getStridedBuffer(ndim, 5);
        final cShape = cBuffer;
        final cStridesA = cBuffer + ndim;
        final cStridesMin = cBuffer + (ndim * 2);
        final cStridesMax = cBuffer + (ndim * 3);
        final cStridesRes = cBuffer + (ndim * 4);

        for (var i = 0; i < ndim; i++) {
          cShape[i] = commonShape[i];
          cStridesA[i] = broadcastA.strides[i];
          cStridesMin[i] = broadcastMin.strides[i];
          cStridesMax[i] = broadcastMax.strides[i];
          cStridesRes[i] = result.strides[i];
        }

        switch (a.dtype) {
          case DType.float64:
            s_clip_double(
              (broadcastA.pointer.cast<ffi.Double>() +
                      broadcastA.offsetElements)
                  .cast(),
              cStridesA,
              (broadcastMin.pointer.cast<ffi.Double>() +
                      broadcastMin.offsetElements)
                  .cast(),
              cStridesMin,
              (broadcastMax.pointer.cast<ffi.Double>() +
                      broadcastMax.offsetElements)
                  .cast(),
              cStridesMax,
              (result.pointer.cast<ffi.Double>() + result.offsetElements)
                  .cast(),
              cStridesRes,
              cShape,
              ndim,
              maskHolder.pointer,
            );
            return result;
          case DType.float32:
            s_clip_float(
              (broadcastA.pointer.cast<ffi.Float>() + broadcastA.offsetElements)
                  .cast(),
              cStridesA,
              (broadcastMin.pointer.cast<ffi.Float>() +
                      broadcastMin.offsetElements)
                  .cast(),
              cStridesMin,
              (broadcastMax.pointer.cast<ffi.Float>() +
                      broadcastMax.offsetElements)
                  .cast(),
              cStridesMax,
              (result.pointer.cast<ffi.Float>() + result.offsetElements).cast(),
              cStridesRes,
              cShape,
              ndim,
              maskHolder.pointer,
            );
            return result;
          case DType.int64:
            s_clip_int64(
              (broadcastA.pointer.cast<ffi.Int64>() + broadcastA.offsetElements)
                  .cast(),
              cStridesA,
              (broadcastMin.pointer.cast<ffi.Int64>() +
                      broadcastMin.offsetElements)
                  .cast(),
              cStridesMin,
              (broadcastMax.pointer.cast<ffi.Int64>() +
                      broadcastMax.offsetElements)
                  .cast(),
              cStridesMax,
              (result.pointer.cast<ffi.Int64>() + result.offsetElements).cast(),
              cStridesRes,
              cShape,
              ndim,
              maskHolder.pointer,
            );
            return result;
          case DType.int32:
            s_clip_int32(
              (broadcastA.pointer.cast<ffi.Int32>() + broadcastA.offsetElements)
                  .cast(),
              cStridesA,
              (broadcastMin.pointer.cast<ffi.Int32>() +
                      broadcastMin.offsetElements)
                  .cast(),
              cStridesMin,
              (broadcastMax.pointer.cast<ffi.Int32>() +
                      broadcastMax.offsetElements)
                  .cast(),
              cStridesMax,
              (result.pointer.cast<ffi.Int32>() + result.offsetElements).cast(),
              cStridesRes,
              cShape,
              ndim,
              maskHolder.pointer,
            );
            return result;
          case DType.uint8:
            s_clip_uint8(
              (broadcastA.pointer.cast<ffi.Uint8>() + broadcastA.offsetElements)
                  .cast(),
              cStridesA,
              (broadcastMin.pointer.cast<ffi.Uint8>() +
                      broadcastMin.offsetElements)
                  .cast(),
              cStridesMin,
              (broadcastMax.pointer.cast<ffi.Uint8>() +
                      broadcastMax.offsetElements)
                  .cast(),
              cStridesMax,
              (result.pointer.cast<ffi.Uint8>() + result.offsetElements).cast(),
              cStridesRes,
              cShape,
              ndim,
              maskHolder.pointer,
            );
            return result;
          case DType.int16:
            s_clip_int16(
              (broadcastA.pointer.cast<ffi.Int16>() + broadcastA.offsetElements)
                  .cast(),
              cStridesA,
              (broadcastMin.pointer.cast<ffi.Int16>() +
                      broadcastMin.offsetElements)
                  .cast(),
              cStridesMin,
              (broadcastMax.pointer.cast<ffi.Int16>() +
                      broadcastMax.offsetElements)
                  .cast(),
              cStridesMax,
              (result.pointer.cast<ffi.Int16>() + result.offsetElements).cast(),
              cStridesRes,
              cShape,
              ndim,
              maskHolder.pointer,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }

      ternaryOp<dynamic, dynamic, dynamic, dynamic>(
        result,
        broadcastA,
        broadcastMin,
        broadcastMax,
        commonShape,
        broadcastA.strides,
        broadcastMin.strides,
        broadcastMax.strides,
        result.strides,
        0,
        broadcastA.offsetElements,
        broadcastMin.offsetElements,
        broadcastMax.offsetElements,
        result.offsetElements,
        (x, mn, mx) =>
            castValue((x as num).clamp(mn as num, mx as num), a.dtype),
        maskHolder.pointer,
      );

      return result;
    } finally {
      maskHolder.dispose();
    }
  } finally {
    if (ownsMin) minArr.dispose();
    if (ownsMax) maxArr.dispose();
  }
}

num _getMinLimit(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return double.negativeInfinity;
    case DType.int64:
      return -9223372036854775808;
    case DType.int32:
      return -2147483648;
    case DType.int16:
      return -32768;
    case DType.int8:
      return -128;
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      return 0;
    default:
      return double.negativeInfinity;
  }
}

num _getMaxLimit(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      return double.infinity;
    case DType.int64:
      return 9223372036854775807;
    case DType.int32:
      return 2147483647;
    case DType.int16:
      return 32767;
    case DType.int8:
      return 127;
    case DType.uint64:
      return 9223372036854775807;
    case DType.uint32:
      return 4294967295;
    case DType.uint16:
      return 65535;
    case DType.uint8:
      return 255;
    default:
      return double.infinity;
  }
}
