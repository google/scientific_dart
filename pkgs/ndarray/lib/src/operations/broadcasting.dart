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

// Standalone operational relative cross-imports

/// Encapsulates the results of a shape broadcasting operation.
///
/// Contains the combined [shape], and the adjusted strides ([stridesA] and [stridesB])
/// for both operand arrays to enable aligned memory walks.
final class BroadcastResult {
  /// The combined broadcasted shape list representing the final target shape
  /// resulting from broadcasting matricial dimensions.
  final List<int> shape;

  /// The adjusted memory stride offsets for the first operand array,
  /// containing `0` on stretched dimensions.
  final List<int> stridesA;

  /// The adjusted memory stride offsets for the second operand array,
  /// containing `0` on stretched dimensions.
  final List<int> stridesB;

  /// Creates a new [BroadcastResult] representing aligned shape and strides.
  BroadcastResult(List<int> shape, List<int> stridesA, List<int> stridesB)
    : shape = List<int>.unmodifiable(shape),
      stridesA = List<int>.unmodifiable(stridesA),
      stridesB = List<int>.unmodifiable(stridesB);
}

/// Computes the broadcasted shape and strides given shapes and strides directly.
///
/// **Preconditions:**
/// - Trailing dimensions comparing from right-to-left must either be equal or one of them must be 1.
///
/// **Throws:**
/// - It is an error if matrix shapes are not compatible for broadcasting.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(D)$ where $D$ is the maximum rank dimension length.
BroadcastResult broadcastBinaryStrides(
  List<int> shapeA,
  List<int> stridesA,
  List<int> shapeB,
  List<int> stridesB,
) {
  if (listEquals(shapeA, shapeB)) {
    checkTotalSize(shapeA);
    return BroadcastResult(shapeA, stridesA, stridesB);
  }

  final maxLen = shapeA.length > shapeB.length ? shapeA.length : shapeB.length;
  final commonShape = List<int>.filled(maxLen, 1);
  final newStridesA = List<int>.filled(maxLen, 0);
  final newStridesB = List<int>.filled(maxLen, 0);

  for (var i = 0; i < maxLen; i++) {
    // Compare from right to left
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
      // newStridesA remains 0 for this dimension, effectively stretching it
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimB == 1) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
      // newStridesB remains 0 for this dimension, effectively stretching it
    } else {
      throw ArgumentError(
        'Shapes $shapeA and $shapeB are not compatible for broadcasting',
      );
    }
  }

  checkTotalSize(commonShape);
  return BroadcastResult(commonShape, newStridesA, newStridesB);
}

/// Calculates the broadcasted shape and strides for two matrices.
///
/// Compares dimensions starting from the trailing dimensions and working forward
/// according to standard NumPy shape broadcasting guidelines.
///
/// **Preconditions:**
/// - Trailing dimensions comparing from right-to-left must either be equal or one of them must be 1.
///
/// **Throws:**
/// - It is an error if [a] or [b] is disposed.
/// - It is an error if matrix shapes are not compatible for broadcasting.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(D)$ where $D$ is the maximum rank dimension length, executing
///   in zero unmanaged heap allocations.
///
/// **Example:**
/// {@example /example/broadcasting_example.dart lang=dart}
BroadcastResult broadcast(NDArray<DTypeTag> a, NDArray<DTypeTag> b) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute broadcast() on a disposed array.');
  }
  return broadcastBinaryStrides(a.shape, a.strides, b.shape, b.strides);
}

/// Broadcasts an array [a] to a new target shape [targetShape].
///
/// Returns a zero-allocation, zero-copy [NDArray] view sharing the exact same
/// backing unmanaged C memory as [a], but with adjusted shape and strides.
/// Dimensions that are stretched (i.e. size 1 broadcasted to size $N$) are
/// assigned a stride of `0`.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - [targetShape] must have at least as many dimensions as [a.shape] (rank is $\ge$ input rank).
/// - Trailing dimensions of [a.shape] and [targetShape] comparing from right-to-left
///   must be compatible (either equal, or the input dimension is 1).
///
/// **Throws:**
/// - It is an error if the array [a] has been disposed.
/// - It is an error if [targetShape] has fewer dimensions than [a.shape].
/// - It is an error if [targetShape] is incompatible with [a.shape] for broadcasting.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(D)$ and space complexity is $O(D)$ where $D$ is the rank of
///   [targetShape], as it only allocates the shape and strides metadata without copying any data.
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view.
///
/// **NumPy Counterpart:**
/// - Maps directly to NumPy's `np.broadcast_to`.
///
/// **Example:**
/// {@example /example/broadcasting_example.dart lang=dart}
///
/// Refer to the [NumPy broadcast_to reference](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html)
/// for additional details.
NDArray<T> broadcastTo<T extends DTypeTag>(
  NDArray<T> a,
  List<int> targetShape,
) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  checkTotalSize(targetShape);

  final shapeA = a.shape;
  final stridesA = a.strides;

  if (listEquals(shapeA, targetShape)) {
    return NDArray<T>.view(
      a,
      shape: targetShape,
      strides: stridesA,
      offsetElements: 0,
    );
  }

  if (targetShape.length < shapeA.length) {
    throw ArgumentError(
      'Cannot broadcast to a shape with fewer dimensions: '
      'input shape $shapeA, target shape $targetShape',
    );
  }

  final newStrides = List<int>.filled(targetShape.length, 0);

  for (var i = 0; i < targetShape.length; i++) {
    // Compare from right to left
    final idxA = shapeA.length - 1 - i;
    final idxT = targetShape.length - 1 - i;

    final dimA = idxA >= 0 ? shapeA[idxA] : 1;
    final dimT = targetShape[idxT];

    if (dimA == dimT) {
      if (idxA >= 0) {
        newStrides[idxT] = stridesA[idxA];
      } else {
        newStrides[idxT] = 0;
      }
    } else if (dimA == 1) {
      newStrides[idxT] = 0;
    } else {
      throw ArgumentError(
        'Shape $shapeA cannot be broadcast to target shape $targetShape',
      );
    }
  }

  return NDArray<T>.view(
    a,
    shape: targetShape,
    strides: newStrides,
    offsetElements: 0,
  );
}
