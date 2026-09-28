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

import '../ndarray.dart';
import '../ndarray_extensions_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports

/// Repeats elements of an array.
///
/// Repeats each element of an array [a] along the given [axis] a number of times
/// specified by [repeats].
///
/// If [axis] is null, [a] is flattened first.
///
/// **Preconditions:**
/// - If [axis] is specified, it must be within the range `[-rank, rank - 1]`.
/// - [repeats] must be an `NDArray<int>`.
/// - Its length must match the size of the
///   dimension along [axis].
/// - All values in [repeats] must be non-negative ($\ge 0$).
/// - If [out] is provided, it must have the correct shape and [DType] to store
///   the result.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
/// - [ArgumentError] if [repeats] length does not match
///   the dimension along [axis], or it contains negative values.
/// - [ArgumentError] if [out] shape or [DType] is incompatible.
///
/// **Performance considerations:**
/// - Time Complexity: $O(N)$ where $N$ is the total number of elements in the
///   output array.
/// - Space Complexity: $O(N)$ for the output array (unless [out] is provided).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final r = repeat(a, [3]);
/// print(r.toList()); // [1, 1, 1, 2, 2, 2]
/// ```
NDArray<T> repeat<T>(
  NDArray<T> a,
  List<int> repeats, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  NDArray<T> src = a;
  int normAxis;
  bool ownsSrc;

  if (axis == null) {
    src = a.flatten();
    normAxis = 0;
    ownsSrc = true;
  } else {
    final rank = a.rank;
    if (axis < -rank || axis >= rank) {
      throw RangeError.range(axis, -rank, rank - 1, 'axis');
    }
    normAxis = axis < 0 ? rank + axis : axis;
    if (!a.isContiguous) {
      src = a.copy();
      ownsSrc = true;
    } else {
      ownsSrc = false;
    }
  }

  try {
    List<int> repsList = repeats;
    if (repsList.length == 1) {
      repsList = List<int>.filled(src.shape[normAxis], repsList[0]);
    }

    if (repsList.length != src.shape[normAxis]) {
      throw ArgumentError(
        'repeats length (${repsList.length}) must match the dimension along axis ($normAxis) which is ${src.shape[normAxis]}',
      );
    }

    final bool hasNegative = repsList.any((x) => x < 0);
    if (hasNegative) {
      throw ArgumentError('repeats values must be non-negative');
    }

    final outputShape = List<int>.from(src.shape);
    final newDimSize = repsList.isEmpty ? 0 : repsList.reduce((x, y) => x + y);
    outputShape[normAxis] = newDimSize;

    final NDArray<T> result;
    if (out != null) {
      if (out.isDisposed) {
        throw StateError('Cannot access a disposed out NDArray.');
      }
      if (out.dtype != src.dtype) {
        throw ArgumentError('out buffer must have the same dtype as input');
      }
      if (out.shape.length != outputShape.length) {
        throw ArgumentError('out buffer shape length must match output shape');
      }
      for (var i = 0; i < outputShape.length; i++) {
        if (out.shape[i] != outputShape[i]) {
          throw ArgumentError('out buffer shape must match output shape');
        }
      }
      result = out;
    } else {
      result = NDArray<T>.create(outputShape, src.dtype);
    }

    if (result.size == 0) {
      return result;
    }

    final outer = src.shape.sublist(0, normAxis).fold<int>(1, (a, b) => a * b);
    final dim = src.shape[normAxis];
    final inner = src.shape.sublist(normAxis + 1).fold<int>(1, (a, b) => a * b);

    final destDim = result.shape[normAxis];

    var destOffset = 0;
    for (var i = 0; i < dim; i++) {
      final rep = repsList[i];
      if (rep == 0) continue;

      for (var o = 0; o < outer; o++) {
        final srcStart = (o * dim + i) * inner;
        final destStart = (o * destDim + destOffset) * inner;

        final srcView = NDArray<T>.view(
          src,
          shape: [rep, inner],
          strides: [0, 1],
          offsetElements: srcStart,
        );

        final destView = NDArray<T>.view(
          result,
          shape: [rep, inner],
          strides: [inner, 1],
          offsetElements: destStart,
        );

        srcView.copy(out: destView);
      }
      destOffset += rep;
    }

    return result;
  } finally {
    if (ownsSrc && !identical(src, a)) {
      src.dispose();
    }
  }
}

/// Constructs an array by repeating [a] the number of times given by [reps].
///
/// If [reps] has length `d`, the result will have dimension of `max(d, a.rank)`.
/// If `a.rank < d`, [a] is promoted to be d-dimensional by prepending new axes.
/// If `a.rank > d`, [reps] is promoted to `a.rank` by pre-pending 1's to it.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - All values in [reps] must be non-negative ($\ge 0$).
/// - If [out] is provided, it must not be disposed and must have the correct shape and [DType] to store the result.
/// - It is an error if [a] is disposed or [out] is disposed.
/// - It is an error if [reps] contains negative values.
/// - It is an error if [out] shape or [DType] is incompatible.
///
/// **Performance considerations:**
/// - Time Complexity: $O(N)$ where $N$ is the total number of elements in the
///   output array.
/// - Contiguous blocks are replicated directly in unmanaged C memory using fast exponential doubling `memcpy`.
/// - Space Complexity: $O(N)$ for the output array (unless [out] is provided).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final t = tile(a, [2]);
/// print(t.toList()); // [1, 2, 1, 2]
/// ```
///
/// Refer to the [NumPy tile reference](https://numpy.org/doc/stable/reference/generated/numpy.tile.html)
/// for details.
NDArray<T> tile<T extends Object>(
  NDArray<T> a,
  List<int> reps, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final bool hasNegative = reps.any((x) => x < 0);
  if (hasNegative) {
    throw ArgumentError('reps values must be non-negative');
  }

  NDArray<T> src = a;
  bool ownsSrc = false;
  List<int> tileReps = List<int>.from(reps);

  try {
    // Align dimensions
    if (src.rank < tileReps.length) {
      final newShape = [
        ...List<int>.filled(tileReps.length - src.rank, 1),
        ...src.shape,
      ];
      src = src.reshape(newShape);
      ownsSrc = !identical(src, a);
    } else if (src.rank > tileReps.length) {
      tileReps = [
        ...List<int>.filled(src.rank - tileReps.length, 1),
        ...tileReps,
      ];
    }

    final outputShape = List<int>.filled(src.rank, 0);
    for (var i = 0; i < src.rank; i++) {
      outputShape[i] = src.shape[i] * tileReps[i];
    }

    final NDArray<T> result;
    if (out != null) {
      if (out.dtype != src.dtype) {
        throw ArgumentError('out buffer must have the same dtype as input');
      }
      if (out.shape.length != outputShape.length) {
        throw ArgumentError('out buffer shape length must match output shape');
      }
      for (var i = 0; i < outputShape.length; i++) {
        if (out.shape[i] != outputShape[i]) {
          throw ArgumentError('out buffer shape must match output shape');
        }
      }
      result = out;
    } else {
      result = NDArray<T>.create(outputShape, src.dtype);
    }

    if (result.size == 0) {
      return result;
    }

    final rank = src.rank;
    if (rank == 0) {
      final marker = ScratchArena.marker;
      try {
        final cSrcShape = ScratchArena.copyInt64s(const <int>[]);
        final cReps = ScratchArena.copyInt64s(const <int>[]);
        final cOutShape = ScratchArena.copyInt64s(const <int>[]);
        native_tile_contiguous(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cReps,
          result.pointer.cast(),
          cOutShape,
          0,
        );
      } finally {
        ScratchArena.reset(marker);
      }
      return result;
    }

    final marker = ScratchArena.marker;
    try {
      final cSrcShape = ScratchArena.copyInt64s(src.shape);
      final cReps = ScratchArena.copyInt64s(tileReps);
      final cOutShape = ScratchArena.copyInt64s(outputShape);

      if (src.isContiguous && result.isContiguous) {
        native_tile_contiguous(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cReps,
          result.pointer.cast(),
          cOutShape,
          rank,
        );
      } else if (result.isContiguous) {
        final contigSrc = src.copy();
        try {
          native_tile_contiguous(
            contigSrc.dtype.index,
            contigSrc.pointer.cast(),
            cSrcShape,
            cReps,
            result.pointer.cast(),
            cOutShape,
            rank,
          );
        } finally {
          contigSrc.dispose();
        }
      } else {
        final cSrcStrides = ScratchArena.copyInt64s(src.strides);
        final cOutStrides = ScratchArena.copyInt64s(result.strides);
        native_tile_strided(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cSrcStrides,
          cReps,
          result.pointer.cast(),
          cOutShape,
          cOutStrides,
          rank,
        );
      }
    } finally {
      ScratchArena.reset(marker);
    }

    return result;
  } finally {
    if (ownsSrc && !identical(src, a)) {
      src.dispose();
    }
  }
}
