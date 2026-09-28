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

import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../ndarray_extensions_bindings.dart';
import '../scratch_arena.dart';

import 'helpers.dart';

/// Repeats elements of an array.
///
/// Repeats each element of an array [a] along the given [axis] a number of times
/// specified by [repeats].
///
/// If [axis] is null, [a] is flattened first.
///
/// **Preconditions:**
/// - If [axis] is specified, it must be within the range `[-rank, rank - 1]`.
/// - [repeats] must be an `int` or `List<int>`.
/// - Its length must match the size of the
///   dimension along [axis] (or be 1 / a scalar `int`).
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
NDArray<T> repeat<T extends DTypeTag>(
  NDArray<T> a,
  Object repeats, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot access a disposed out NDArray.');
  }

  final List<int> rawRepeats;
  if (repeats is int) {
    rawRepeats = [repeats];
  } else if (repeats is List<int>) {
    rawRepeats = repeats;
  } else {
    throw ArgumentError('repeats must be an int or a List<int>');
  }

  return NDArray.scope(() {
    NDArray<T> src = a;
    int normAxis;

    if (axis == null) {
      src = a.flatten();
      normAxis = 0;
    } else {
      final rank = a.rank;
      if (axis < -rank || axis >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axis');
      }
      normAxis = axis < 0 ? rank + axis : axis;
      if (!a.isContiguous) {
        src = a.copy();
      }
    }

    List<int> repsList = rawRepeats;
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
    }

    final bool useTempOut =
        out != null && (!out.isContiguous || sharesMemory(a, out));
    final NDArray<T> target = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(outputShape, src.dtype);

    if (target.size == 0) {
      if (out != null) {
        return out;
      }
      return target.detachToParentScope();
    }

    final outer = src.shape.sublist(0, normAxis).fold<int>(1, (x, y) => x * y);
    final dim = src.shape[normAxis];
    final inner = src.shape.sublist(normAxis + 1).fold<int>(1, (x, y) => x * y);

    final destDim = target.shape[normAxis];

    if (inner == 1) {
      switch (src.dtype) {
        case DType.float64:
          final srcList = src.pointer.cast<ffi.Double>().asTypedList(src.size);
          final dstList = target.pointer.cast<ffi.Double>().asTypedList(
            target.size,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim;
            var destPos = o * destDim;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final val = srcList[srcBase + i];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = val;
              }
            }
          }
        case DType.float32:
          final srcList = src.pointer.cast<ffi.Float>().asTypedList(src.size);
          final dstList = target.pointer.cast<ffi.Float>().asTypedList(
            target.size,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim;
            var destPos = o * destDim;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final val = srcList[srcBase + i];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = val;
              }
            }
          }
        case DType.int64:
        case DType.uint64:
          final srcList = src.pointer.cast<ffi.Int64>().asTypedList(src.size);
          final dstList = target.pointer.cast<ffi.Int64>().asTypedList(
            target.size,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim;
            var destPos = o * destDim;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final val = srcList[srcBase + i];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = val;
              }
            }
          }
        case DType.int32:
        case DType.uint32:
          final srcList = src.pointer.cast<ffi.Int32>().asTypedList(src.size);
          final dstList = target.pointer.cast<ffi.Int32>().asTypedList(
            target.size,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim;
            var destPos = o * destDim;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final val = srcList[srcBase + i];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = val;
              }
            }
          }
        case DType.int16:
        case DType.uint16:
        case DType.float16:
        case DType.bfloat16:
          final srcList = src.pointer.cast<ffi.Int16>().asTypedList(src.size);
          final dstList = target.pointer.cast<ffi.Int16>().asTypedList(
            target.size,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim;
            var destPos = o * destDim;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final val = srcList[srcBase + i];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = val;
              }
            }
          }
        case DType.int8:
        case DType.uint8:
        case DType.boolean:
          final srcList = src.pointer.cast<ffi.Int8>().asTypedList(src.size);
          final dstList = target.pointer.cast<ffi.Int8>().asTypedList(
            target.size,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim;
            var destPos = o * destDim;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final val = srcList[srcBase + i];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = val;
              }
            }
          }
        case DType.complex64:
          final srcList = src.pointer.cast<ffi.Int64>().asTypedList(src.size);
          final dstList = target.pointer.cast<ffi.Int64>().asTypedList(
            target.size,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim;
            var destPos = o * destDim;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final val = srcList[srcBase + i];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = val;
              }
            }
          }
        case DType.complex128:
          final srcList = src.pointer.cast<ffi.Double>().asTypedList(
            src.size * 2,
          );
          final dstList = target.pointer.cast<ffi.Double>().asTypedList(
            target.size * 2,
          );
          for (var o = 0; o < outer; o++) {
            final srcBase = o * dim * 2;
            var destPos = o * destDim * 2;
            for (var i = 0; i < dim; i++) {
              final rep = repsList[i];
              if (rep == 0) continue;
              final rVal = srcList[srcBase + i * 2];
              final iVal = srcList[srcBase + i * 2 + 1];
              for (var r = 0; r < rep; r++) {
                dstList[destPos++] = rVal;
                dstList[destPos++] = iVal;
              }
            }
          }
      }
    } else {
      final srcBytePtr = src.pointer.cast<ffi.Uint8>();
      final dstBytePtr = target.pointer.cast<ffi.Uint8>();
      final itemBytes = src.dtype.byteWidth;
      final sliceBytes = inner * itemBytes;
      final destDimBytes = destDim * sliceBytes;
      final dimBytes = dim * sliceBytes;
      for (var o = 0; o < outer; o++) {
        final srcOuterPtr = srcBytePtr + o * dimBytes;
        final dstOuterPtr = dstBytePtr + o * destDimBytes;
        var destByteOffset = 0;
        for (var i = 0; i < dim; i++) {
          final rep = repsList[i];
          if (rep == 0) continue;
          final sPtr = srcOuterPtr + i * sliceBytes;
          final dPtr = dstOuterPtr + destByteOffset;
          custom_memcpy(dPtr.cast(), sPtr.cast(), sliceBytes);
          var copied = sliceBytes;
          final totalBytes = rep * sliceBytes;
          while (copied < totalBytes) {
            final toCopy = (copied <= totalBytes - copied)
                ? copied
                : (totalBytes - copied);
            custom_memcpy((dPtr + copied).cast(), dPtr.cast(), toCopy);
            copied += toCopy;
          }
          destByteOffset += totalBytes;
        }
      }
    }

    if (out != null) {
      if (useTempOut) {
        target.copy(out: out);
      }
      return out;
    }
    return target.detachToParentScope();
  });
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
NDArray<T> tile<T extends DTypeTag>(
  NDArray<T> a,
  List<int> reps, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final List<int> rawReps = reps;

  final bool hasNegative = rawReps.any((x) => x < 0);
  if (hasNegative) {
    throw ArgumentError('reps values must be non-negative');
  }

  return NDArray.scope(() {
    NDArray<T> src = a;
    List<int> tileReps = List<int>.from(rawReps);

    // Align dimensions
    if (src.rank < tileReps.length) {
      final newShape = [
        ...List<int>.filled(tileReps.length - src.rank, 1),
        ...src.shape,
      ];
      src = src.reshape(newShape);
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
    }

    final bool useTempOut =
        out != null && (!out.isContiguous || sharesMemory(a, out));
    final NDArray<T> target = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(outputShape, src.dtype);

    if (target.size == 0) {
      if (out != null) {
        return out;
      }
      return target.detachToParentScope();
    }

    final rank = src.rank;
    if (rank == 0) {
      final marker = ScratchArena.marker;
      try {
        final cSrcShape = ScratchArena.copyInt64s(const <int>[]);
        final cReps = ScratchArena.copyInt64s(const <int>[]);
        final cOutShape = ScratchArena.copyInt64s(const <int>[]);
        final rc = native_tile_contiguous(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cReps,
          target.pointer.cast(),
          cOutShape,
          0,
        );
        if (rc != 0) {
          throw StateError('Native tile operation failed with code $rc');
        }
      } finally {
        ScratchArena.reset(marker);
      }
      if (out != null) {
        if (useTempOut) {
          target.copy(out: out);
        }
        return out;
      }
      return target.detachToParentScope();
    }

    final marker = ScratchArena.marker;
    try {
      final cSrcShape = ScratchArena.copyInt64s(src.shape);
      final cReps = ScratchArena.copyInt64s(tileReps);
      final cOutShape = ScratchArena.copyInt64s(outputShape);

      final int rc;
      if (src.isContiguous && target.isContiguous) {
        rc = native_tile_contiguous(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cReps,
          target.pointer.cast(),
          cOutShape,
          rank,
        );
      } else if (target.isContiguous) {
        final contigSrc = src.copy();
        rc = native_tile_contiguous(
          contigSrc.dtype.index,
          contigSrc.pointer.cast(),
          cSrcShape,
          cReps,
          target.pointer.cast(),
          cOutShape,
          rank,
        );
      } else {
        final cSrcStrides = ScratchArena.copyInt64s(src.strides);
        final cOutStrides = ScratchArena.copyInt64s(target.strides);
        rc = native_tile_strided(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cSrcStrides,
          cReps,
          target.pointer.cast(),
          cOutShape,
          cOutStrides,
          rank,
        );
      }
      if (rc != 0) {
        throw StateError('Native tile operation failed with code $rc');
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (out != null) {
      if (useTempOut) {
        target.copy(out: out);
      }
      return out;
    }
    return target.detachToParentScope();
  });
}
