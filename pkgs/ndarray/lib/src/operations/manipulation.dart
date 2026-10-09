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
import 'dart:math' as math;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../ndarray_extensions_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'helpers.dart';
export 'helpers.dart' show castNDArray, sharesMemory;

/// Concatenates a list of arrays along a specified axis.
///
/// **Preconditions:**
/// - [arrays] must not be empty.
/// - All arrays in [arrays] must not be disposed.
/// - All arrays in [arrays] must have identical dtypes and compatible shapes.
/// - It is an error if [arrays] is empty.
/// - It is an error if any array in [arrays] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if arrays have mismatched dtypes or shapes.
/// - It is an error if [out] is provided with incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Time Complexity: $O(N)$ where $N$ is the total number of elements in the output array.
/// - Contiguous memory blocks are copied directly using native C `memcpy`.
/// - Space Complexity: $O(N)$ for newly allocated output (or $O(1)$ auxiliary if [out] is provided).
///
/// **Example:**
/// {@example /example/rearranging_example.dart lang=dart}
///
/// Refer to the [NumPy concatenate reference](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
/// for details.
NDArray<T> concatenate<T extends SelfOf<DTypeTag>>(
  List<NDArray<T>> arrays, {
  int axis = 0,
  NDArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(
      arrays,
      'arrays',
      'List of arrays must not be empty',
    );
  }

  for (final arr in arrays) {
    if (arr.isDisposed) {
      throw StateError('Cannot concatenate a disposed array.');
    }
  }

  final first = arrays.first;
  final rank = first.shape.length;
  final dtype = first.dtype;

  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.index(axis, first.shape, 'axis out of range');
  }

  for (var i = 1; i < arrays.length; i++) {
    final arr = arrays[i];
    if (arr.dtype != dtype) {
      throw ArgumentError.value(
        arr.dtype,
        'arrays',
        'All arrays must have the same DType',
      );
    }
    if (arr.shape.length != rank) {
      throw ArgumentError.value(
        arr.rank,
        'arrays',
        'All arrays must have the same rank',
      );
    }
    for (var j = 0; j < rank; j++) {
      if (j != normAxis && arr.shape[j] != first.shape[j]) {
        throw ArgumentError.value(
          arr.shape,
          'arrays',
          'Shapes must match except in dimension $normAxis',
        );
      }
    }
  }

  final targetShape = List<int>.from(first.shape);
  var totalAxisSize = 0;
  for (final arr in arrays) {
    totalAxisSize += arr.shape[normAxis];
  }
  targetShape[normAxis] = totalAxisSize;

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot concatenate into a disposed out array.');
    }
    validateOutBuffer(out);
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Incompatible out buffer shape or dtype',
      );
    }
  }

  return NDArray.scope(() {
    final bool useTempOut =
        out != null &&
        (!out.isContiguous || arrays.any((arr) => sharesMemory(arr, out)));
    final result = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(targetShape, dtype);
    if (result.size == 0) {
      if (out != null) {
        return out;
      }
      return result.detachToParentScope();
    }

    final outer = first.shape
        .sublist(0, normAxis)
        .fold<int>(1, (a, b) => a * b);
    final inner = first.shape
        .sublist(normAxis + 1)
        .fold<int>(1, (a, b) => a * b);
    final byteWidth = dtype.byteWidth;
    final destStrideBytes = totalAxisSize * inner * byteWidth;
    final destPtr = result.pointer.cast<ffi.Uint8>();

    var axisOffset = 0;
    for (var r = 0; r < arrays.length; r++) {
      final arr = arrays[r];
      final sizeAlongAxis = arr.shape[normAxis];
      if (sizeAlongAxis == 0) continue;

      final blockBytes = sizeAlongAxis * inner * byteWidth;
      final srcStrideBytes = sizeAlongAxis * inner * byteWidth;
      final destInitialOffsetBytes = axisOffset * inner * byteWidth;

      final srcContiguous = arr.isContiguous ? arr : arr.copy();
      try {
        final srcPtr = srcContiguous.pointer.cast<ffi.Uint8>();
        if (outer == 1) {
          custom_memcpy(
            (destPtr + destInitialOffsetBytes).cast(),
            srcPtr.cast(),
            blockBytes,
          );
        } else {
          for (var o = 0; o < outer; o++) {
            custom_memcpy(
              (destPtr + (o * destStrideBytes + destInitialOffsetBytes)).cast(),
              (srcPtr + (o * srcStrideBytes)).cast(),
              blockBytes,
            );
          }
        }
      } finally {
        if (!identical(srcContiguous, arr)) {
          srcContiguous.dispose();
        }
      }
      axisOffset += sizeAlongAxis;
    }

    if (out != null) {
      if (useTempOut) {
        result.copy(out: out);
      }
      return out;
    }
    return result.detachToParentScope();
  });
}

/// Join a sequence of arrays along a new axis.
///
/// Stacks the input [arrays] along a new dimension at [axis]. All arrays in the
/// list must have the exact same shape and `DType.`
///
/// **Preconditions:**
/// - Input list [arrays] must be non-empty.
/// - All arrays in [arrays] must not be disposed.
/// - All arrays in [arrays] must share identical shapes and DTypes.
/// - It is an error if [arrays] is empty.
/// - It is an error if any array in [arrays] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if shapes or DTypes mismatch.
/// - It is an error if [out] is provided with incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Time Complexity: $O(N)$ where $N$ is the total number of elements in the output array.
/// - Directly preallocates destination memory and copies buffer blocks using native C `memcpy`.
/// - Space Complexity: $O(N)$ for newly allocated output (or $O(1)$ auxiliary if [out] is provided).
///
/// **Example:**
/// {@example /example/rearranging_example.dart lang=dart}
///
/// Refer to the [NumPy stack reference](https://numpy.org/doc/stable/reference/generated/numpy.stack.html)
/// for details.
NDArray<T> stack<T extends SelfOf<DTypeTag>>(
  List<NDArray<T>> arrays, {
  int axis = 0,
  NDArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(
      arrays,
      'arrays',
      'List of arrays to stack must not be empty',
    );
  }

  for (final arr in arrays) {
    if (arr.isDisposed) {
      throw StateError('Cannot execute stack() on a disposed array.');
    }
  }

  final first = arrays.first;
  final rank = first.shape.length;
  final dtype = first.dtype;

  // Validate identical shapes and DTypes
  for (var i = 1; i < arrays.length; i++) {
    final arr = arrays[i];
    if (arr.dtype != dtype) {
      throw ArgumentError.value(
        arr.dtype,
        'arrays',
        'All arrays in stack must have identical DTypes',
      );
    }
    if (!listEquals(arr.shape, first.shape)) {
      throw ArgumentError.value(
        arr.shape,
        'arrays',
        'All arrays in stack must have identical shapes',
      );
    }
  }

  // Target axis can range from -(rank + 1) to rank
  final targetAxis = axis < 0 ? rank + 1 + axis : axis;
  if (targetAxis < 0 || targetAxis > rank) {
    throw RangeError.range(targetAxis, 0, rank, 'axis');
  }

  // Build stacked output shape by inserting arrays.length at targetAxis index
  final stackedShape = List<int>.from(first.shape);
  stackedShape.insert(targetAxis, arrays.length);

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot execute stack() with a disposed out array.');
    }
    validateOutBuffer(out);
    if (!out.isWriteable ||
        !listEquals(out.shape, stackedShape) ||
        out.dtype != dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Incompatible out buffer shape or dtype',
      );
    }
  }

  return NDArray.scope(() {
    final bool useTempOut =
        out != null &&
        (!out.isContiguous || arrays.any((arr) => sharesMemory(arr, out)));
    final result = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(stackedShape, dtype);
    if (result.size == 0) {
      if (out != null) {
        return out;
      }
      return result.detachToParentScope();
    }

    final numArrays = arrays.length;
    final outer = first.shape
        .sublist(0, targetAxis)
        .fold<int>(1, (a, b) => a * b);
    final inner = first.shape.sublist(targetAxis).fold<int>(1, (a, b) => a * b);
    final byteWidth = dtype.byteWidth;
    final blockBytes = inner * byteWidth;
    final destStrideBytes = numArrays * inner * byteWidth;
    final srcStrideBytes = inner * byteWidth;

    final destPtr = result.pointer.cast<ffi.Uint8>();

    for (var r = 0; r < numArrays; r++) {
      final arr = arrays[r];
      final srcContiguous = arr.isContiguous ? arr : arr.copy();
      try {
        final srcPtr = srcContiguous.pointer.cast<ffi.Uint8>();
        final destInitialOffsetBytes = r * inner * byteWidth;

        if (outer == 1) {
          custom_memcpy(
            (destPtr + destInitialOffsetBytes).cast(),
            srcPtr.cast(),
            blockBytes,
          );
        } else {
          for (var o = 0; o < outer; o++) {
            custom_memcpy(
              (destPtr + (o * destStrideBytes + destInitialOffsetBytes)).cast(),
              (srcPtr + (o * srcStrideBytes)).cast(),
              blockBytes,
            );
          }
        }
      } finally {
        if (!identical(srcContiguous, arr)) {
          srcContiguous.dispose();
        }
      }
    }

    if (out != null) {
      if (useTempOut) {
        result.copy(out: out);
      }
      return out;
    }
    return result.detachToParentScope();
  });
}

/// Expand the shape of an array by inserting a new axis of size 1.
///
/// Inserts a new dimension of size 1 at the specified [axis] position.
/// If [axis] is negative, it is normalized relative to the rank of [a] plus 1.
///
/// **Preconditions:**
/// - [axis] normalized value must be between `0` and the rank of [a] inclusive.
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Expand Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> expand_dims<T extends DTypeTag>(NDArray<T> a, int axis) {
  if (a.isDisposed) {
    throw StateError('Cannot execute expand_dims() on a disposed array.');
  }
  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + 1 + axis : axis;

  if (targetAxis < 0 || targetAxis > rank) {
    throw RangeError.range(
      axis,
      -rank - 1,
      rank,
      'axis',
      'Must be within valid rank range',
    );
  }

  final newShape = List<int>.from(a.shape);
  final newStrides = List<int>.from(a.strides);

  newShape.insert(targetAxis, 1);
  // Insert stride mapping: copy sibling stride or default to 1 if rank is 0
  final siblingStride = targetAxis < rank ? a.strides[targetAxis] : 1;
  newStrides.insert(targetAxis, siblingStride);

  return NDArray.view(
    a,
    shape: newShape,
    strides: newStrides,
    offsetElements: 0,
  );
}

/// Remove axes of size 1 from the shape of an array.
///
/// If [axis] is provided, only removes the specified dimensions (which must have size 1).
/// If [axis] is null, removes all dimensions of size 1.
///
/// **Preconditions:**
/// - Specified [axis] entries must indeed correspond to dimensions of size 1.
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds or targets a dimension whose size is not 1.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Squeeze Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html)
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> squeeze<T extends DTypeTag>(NDArray<T> a, {List<int>? axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute squeeze() on a disposed array.');
  }
  final rank = a.shape.length;
  final shape = a.shape;
  final strides = a.strides;

  final newShape = <int>[];
  final newStrides = <int>[];

  final squeezeAxes = <int>{};
  if (axis != null) {
    for (final ax in axis) {
      final targetAx = ax < 0 ? rank + ax : ax;
      if (targetAx < 0 || targetAx >= rank) {
        throw RangeError.range(
          ax,
          -rank,
          rank - 1,
          'axis',
          'Must be within valid rank range',
        );
      }
      if (shape[targetAx] != 1) {
        throw ArgumentError.value(
          ax,
          'axis',
          'Cannot squeeze axis $ax because its dimension size is ${shape[targetAx]} (must be 1)',
        );
      }
      squeezeAxes.add(targetAx);
    }
  } else {
    for (var i = 0; i < rank; i++) {
      if (shape[i] == 1) {
        squeezeAxes.add(i);
      }
    }
  }

  for (var i = 0; i < rank; i++) {
    if (!squeezeAxes.contains(i)) {
      newShape.add(shape[i]);
      newStrides.add(strides[i]);
    }
  }

  // Squeezing all dimensions of a 1D unit tensor (e.g. shape [1]) yields a 0D scalar shape []
  return NDArray.view(
    a,
    shape: newShape,
    strides: newStrides,
    offsetElements: 0,
  );
}

/// Create a sliding window view over several dimensions of an array.
///
/// This corresponds to NumPy's `lib.stride_tricks.sliding_window_view` function.
///
/// **Mathematical Mechanics**:
/// By manipulating strides, a sliding window of shape `windowShape` can be created
/// without copying any element data (completely zero-copy, copy-free, and zero-allocation).
///
/// For an input array `a` with shape $S = (s_0, \dots, s_{D-1})$ and strides $V = (v_0, \dots, v_{D-1})$:
/// - The axes to apply sliding windows are specified by [axis] (defaults to all axes).
/// - The output shape has the original dimensions reduced by `windowShape - 1`, with the window
///   dimensions appended at the end:
///   $$S_{\text{out}} = (s_0 - w_0 + 1, \dots, s_{k} - w_k + 1, \dots, w_0, \dots, w_k)$$
/// - The output strides has the original strides, with the original strides of the window axes appended at the end:
///   $$V_{\text{out}} = (v_0, \dots, v_{D-1}, v_{\text{axis}_0}, \dots, v_{\text{axis}_k})$$
///
/// **Preconditions:**
/// - [windowShape] length must match [axis] length.
/// - Each window dimension must be strictly positive and less than or equal to the corresponding axis size.
/// - It is an error if [a] is disposed.
/// - It is an error if axes are out of range, shapes mismatch, or window dimensions exceed axis sizes.
///
/// **Example:**
/// {@example /example/rearranging_example.dart lang=dart}
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> slidingWindowView<T extends DTypeTag>(
  NDArray<T> a,
  List<int> windowShape, {
  List<int>? axis,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute slidingWindowView() on a disposed array.');
  }
  final rank = a.shape.length;

  // 1. Resolve and validate target axes
  final targetAxes = <int>[];
  if (axis != null) {
    for (final ax in axis) {
      final resolved = ax < 0 ? rank + ax : ax;
      if (resolved < 0 || resolved >= rank) {
        throw RangeError.range(
          ax,
          -rank,
          rank - 1,
          'axis',
          'Must be within valid rank range',
        );
      }
      if (targetAxes.contains(resolved)) {
        throw ArgumentError.value(
          resolved,
          'axis',
          'Duplicate axis specified in sliding window: $resolved',
        );
      }
      targetAxes.add(resolved);
    }
  } else {
    // Default is all axes
    for (var i = 0; i < rank; i++) {
      targetAxes.add(i);
    }
  }

  if (windowShape.length != targetAxes.length) {
    throw ArgumentError.value(
      windowShape,
      'windowShape',
      'windowShape length (${windowShape.length}) must match axes length (${targetAxes.length})',
    );
  }

  // 2. Calculate output shape and strides
  final outShape = List<int>.from(a.shape);
  final outStrides = List<int>.from(a.strides);

  for (var i = 0; i < targetAxes.length; i++) {
    final ax = targetAxes[i];
    final wSize = windowShape[i];
    final aSize = a.shape[ax];

    if (wSize <= 0) {
      throw ArgumentError.value(
        wSize,
        'windowShape',
        'windowShape dimensions must be strictly positive (was $wSize)',
      );
    }
    if (wSize > aSize) {
      throw ArgumentError.value(
        wSize,
        'windowShape',
        'windowShape dimension ($wSize) cannot exceed axis size ($aSize) for axis $ax',
      );
    }

    outShape[ax] = aSize - wSize + 1;
  }

  // Append window dimensions and their corresponding strides at the end
  for (var i = 0; i < targetAxes.length; i++) {
    final ax = targetAxes[i];
    outShape.add(windowShape[i]);
    outStrides.add(a.strides[ax]);
  }

  // 3. Return the zero-copy NDArray view sharing backing unmanaged C memory
  return NDArray.view(a, shape: outShape, strides: outStrides);
}

/// Reverses the order of elements along the given axis/axes.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If [axis] is a list, all elements must be unique.
/// - Each axis must be a valid axis index for [a] (within `[-rank, rank - 1]`).
/// - It is an error if [a] is disposed.
/// - It is an error if any axis is out of bounds.
/// - It is an error if [axis] contains duplicate indices.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Reference:**
/// Refer to [NumPy flip documentation](https://numpy.org/doc/stable/reference/generated/numpy.flip.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> flip<T extends DTypeTag>(NDArray<T> a, {Object? axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot flip a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return NDArray.view(a, shape: [], strides: [], offsetElements: 0);
  }

  final List<int> axesToFlip;
  if (axis == null) {
    axesToFlip = List.generate(rank, (i) => i);
  } else if (axis is int) {
    final normAx = axis < 0 ? rank + axis : axis;
    if (normAx < 0 || normAx >= rank) {
      throw RangeError.range(
        axis,
        -rank,
        rank - 1,
        'axis',
        'Must be within valid rank range',
      );
    }
    axesToFlip = [normAx];
  } else if (axis is List<int>) {
    final uniqueAxes = <int>{};
    for (final ax in axis) {
      final normAx = ax < 0 ? rank + ax : ax;
      if (normAx < 0 || normAx >= rank) {
        throw RangeError.range(
          ax,
          -rank,
          rank - 1,
          'axis',
          'Must be within valid rank range',
        );
      }
      if (!uniqueAxes.add(normAx)) {
        throw ArgumentError.value(axis, 'axis', 'axes must be unique');
      }
    }
    axesToFlip = uniqueAxes.toList();
  } else {
    throw ArgumentError.value(
      axis,
      'axis',
      'axis must be null, an integer, or a list of integers',
    );
  }

  final newStrides = List<int>.from(a.strides);
  var offset = 0;

  for (final ax in axesToFlip) {
    newStrides[ax] = a.strides[ax] * -1;
    if (a.shape[ax] > 0) {
      offset += (a.shape[ax] - 1) * a.strides[ax];
    }
  }

  return NDArray.view(
    a,
    shape: a.shape,
    strides: newStrides,
    offsetElements: offset,
  );
}

/// Flips the array in the left/right direction (column-wise).
///
/// Equivalent to `flip(a, axis: 1)`.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have a rank of at least 2.
/// - It is an error if [a] is disposed.
/// - It is an error if [a] rank is less than 2.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view.
///
/// **Reference:**
/// Refer to [NumPy fliplr documentation](https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> fliplr<T extends DTypeTag>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot fliplr a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(a, 'a', 'Input must be >= 2-D');
  }
  return flip(a, axis: 1);
}

/// Flips the array in the up/down direction (row-wise).
///
/// Equivalent to `flip(a, axis: 0)`.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have a rank of at least 1.
/// - It is an error if [a] is disposed.
/// - It is an error if [a] rank is less than 1.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view.
///
/// **Reference:**
/// Refer to [NumPy flipud documentation](https://numpy.org/doc/stable/reference/generated/numpy.flipud.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> flipud<T extends DTypeTag>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot flipud a disposed array.');
  }
  if (a.rank < 1) {
    throw ArgumentError.value(a, 'a', 'Input must be >= 1-D');
  }
  return flip(a, axis: 0);
}

/// Rotate an array by 90 degrees in the plane specified by axes.
///
/// Rotation direction is from the first towards the second axis.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have rank >= 2.
/// - [axes] must be a list of 2 distinct axes.
///
/// Refer to [NumPy rot90 documentation](https://numpy.org/doc/stable/reference/generated/numpy.rot90.html).
NDArray<T> rot90<T extends DTypeTag>(
  NDArray<T> a, [
  int k = 1,
  List<int> axes = const [0, 1],
]) {
  if (a.isDisposed) {
    throw StateError('Cannot rot90 a disposed array.');
  }
  final rank = a.rank;
  if (rank < 2) {
    throw ArgumentError.value(a, 'a', 'Must be >= 2-D (was rank $rank).');
  }
  if (axes.length != 2) {
    throw ArgumentError.value(axes, 'axes', 'len(axes) must be 2');
  }
  final ax0 = axes[0] < 0 ? rank + axes[0] : axes[0];
  final ax1 = axes[1] < 0 ? rank + axes[1] : axes[1];
  if (ax0 < 0 || ax0 >= rank || ax1 < 0 || ax1 >= rank) {
    throw RangeError('axes out of range for array of rank $rank.');
  }
  if (ax0 == ax1) {
    throw ArgumentError.value(axes, 'axes', 'Axes must be different');
  }

  var rotK = k % 4;
  if (rotK < 0) rotK += 4;
  if (rotK == 0) {
    return NDArray.view(
      a,
      shape: a.shape,
      strides: a.strides,
      offsetElements: 0,
    );
  }
  if (rotK == 2) {
    return flip(flip(a, axis: ax0), axis: ax1);
  }

  final axesList = List.generate(rank, (i) => i);
  axesList[ax0] = ax1;
  axesList[ax1] = ax0;

  if (rotK == 1) {
    return flip(a.transpose(axesList), axis: ax0);
  } else {
    // rotK == 3
    return flip(a.transpose(axesList), axis: ax1);
  }
}

/// Stacks arrays in sequence vertically (row wise).
///
/// It is an error if [arrays] is empty, any array is disposed, or array shapes/dtypes mismatch.
NDArray<T> vstack<T extends SelfOf<DTypeTag>>(
  List<NDArray<T>> arrays, {
  NDArray<T>? out,
}) {
  return concatenate(arrays, axis: 0, out: out);
}

/// Stacks arrays in sequence horizontally (column wise).
///
/// This is equivalent to concatenation along the second axis (axis 1),
/// except for 1-D arrays where it concatenates along the first axis (axis 0).
///
/// It is an error if [arrays] is empty, any array is disposed, or array shapes/dtypes mismatch.
NDArray<T> hstack<T extends SelfOf<DTypeTag>>(
  List<NDArray<T>> arrays, {
  NDArray<T>? out,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError.value(arrays, 'arrays', 'arrays cannot be empty');
  }
  if (arrays.first.rank == 1) {
    return concatenate(arrays, axis: 0, out: out);
  }
  return concatenate(arrays, axis: 1, out: out);
}

/// Returns a deep, C-contiguous copy of the given array.
///
/// The copy preserves the logical order and values of the elements defined by
/// [a]'s shape and strides. However, the physical memory layout of the returned
/// array is always contiguous (and its strides are reset to standard C-contiguous
/// strides).
///
/// This function corresponds to NumPy's `copy` function.
/// **Preconditions:**
/// - It is an error if the array [a] is already disposed.
///
///
/// **Example:**
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> copy<T extends DTypeTag, Out extends T>(
  NDArray<T> a, {
  NDArray<Out>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute copy() on a disposed array.');
  }
  return a.copy(out: out);
}

/// Extracts a diagonal or constructs a diagonal array.
///
/// **Memory Ownership & Lifetime Warning:**
/// - **If [v] is 2D (extracting diagonal):** Returns a **zero-copy metadata view** sharing the underlying C memory page. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [v] will invalidate the returned view. Calling [dispose] on the view does nothing.
/// - **If [v] is 1D (constructing diagonal):** Allocates a new 2D array on the unmanaged C heap (unless [out] is provided). **The caller takes full ownership** of this memory and **must explicitly call [dispose]** to prevent native leaks.
///
/// **Example:**
/// {@example /example/diag_example.dart lang=dart}
///
/// Reference: [Diagonal Matrix](https://en.wikipedia.org/wiki/Diagonal_matrix)
NDArray<T> diag<T extends DTypeTag, Out extends T>(
  NDArray<T> v, {
  int k = 0,
  NDArray<Out>? out,
}) {
  if (v.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute diag() on a disposed array.');
  }
  if (v.shape.length == 2) {
    final m = v.shape[0];
    final n = v.shape[1];

    int startRow;
    int startCol;
    int len;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
      len = math.max(0, math.min(m, n - k));
    } else {
      startRow = -k;
      startCol = 0;
      len = math.max(0, math.min(m + k, n));
    }

    if (len <= 0) {
      if (out != null) {
        validateOutBuffer(out);
        if (!out.isWriteable ||
            !listEquals(out.shape, [0]) ||
            out.dtype != v.dtype) {
          throw ArgumentError.value(
            out,
            'out',
            'Provided out buffer has incompatible shape or dtype',
          );
        }
        out.fill(castValue(0, v.dtype));
        return out;
      }
      return NDArray<T>.view(
        v,
        shape: [0],
        strides: [v.strides[0] + v.strides[1]],
        offsetElements: 0,
      );
    }

    final offsetElements = startRow * v.strides[0] + startCol * v.strides[1];
    final diagStride = v.strides[0] + v.strides[1];

    final view = NDArray<T>.view(
      v,
      shape: [len],
      strides: [diagStride],
      offsetElements: offsetElements,
    );
    if (out != null) {
      validateOutBuffer(out);
      view.copy(out: out);
      view.dispose();
      return out;
    }
    return view;
  } else if (v.shape.length == 1) {
    final n = v.shape[0];
    final size = n + k.abs();
    final targetShape = [size, size];

    if (out != null) {
      validateOutBuffer(out);
      if (!out.isWriteable ||
          !listEquals(out.shape, targetShape) ||
          out.dtype != v.dtype) {
        throw ArgumentError.value(
          out,
          'out',
          'Provided out buffer has incompatible shape or dtype',
        );
      }
    }

    return NDArray.scope(() {
      final bool useTempOut = out != null && sharesMemory(v, out);
      final result = (out != null && !useTempOut)
          ? out
          : NDArray<T>.zeros(targetShape, v.dtype);
      if (out != null && !useTempOut) {
        result.fill(castValue(0, v.dtype));
      }

      int startRow;
      int startCol;

      if (k >= 0) {
        startRow = 0;
        startCol = k;
      } else {
        startRow = -k;
        startCol = 0;
      }

      final resStride0 = result.strides[0];
      final resStride1 = result.strides[1];
      for (var i = 0; i < n; i++) {
        final flatOffset =
            (startRow + i) * resStride0 + (startCol + i) * resStride1;
        result.setCellRaw(result.offsetElements + flatOffset, v.getCellFlat(i));
      }

      if (out != null) {
        if (useTempOut) {
          result.copy(out: out);
        }
        return out;
      }
      return result.detachToParentScope();
    });
  } else {
    throw ArgumentError.value(
      v,
      'v',
      'Input array must be 1- or 2-dimensional',
    );
  }
}

/// Extract a lower triangular matrix (on and below the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
/// - It is an error if [a] is disposed.
/// - It is an error if [a] has rank < 2.
/// - It is an error if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> tril<T extends DTypeTag, Out extends T>(
  NDArray<T> a, {
  int k = 0,
  NDArray<Out>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute tril() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError.value(a, 'a', 'Input array must have rank >= 2');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Provided out buffer has incompatible shape or dtype',
      );
    }
  }

  return NDArray.scope(() {
    final bool useTempOut = out != null && sharesMemory(a, out);
    final result = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(a.shape, a.dtype);

    final rank = a.shape.length;
    final rows = a.shape[rank - 2];
    final cols = a.shape[rank - 1];

    final batchCount = a.shape.isEmpty
        ? 1
        : a.shape.sublist(0, rank - 2).isEmpty
        ? 1
        : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

    bool handledByFastKernel = false;
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_tril_double(
            a.pointer.cast(),
            result.pointer.cast(),
            batchCount,
            rows,
            cols,
            k,
          );
          handledByFastKernel = true;
        case DType.float32:
          v_tril_float(
            a.pointer.cast(),
            result.pointer.cast(),
            batchCount,
            rows,
            cols,
            k,
          );
          handledByFastKernel = true;
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

    if (!handledByFastKernel) {
      final zeroVal = castValue(0, a.dtype);
      final coords = List<int>.filled(rank, 0);

      void walk(int dim) {
        if (dim == rank - 2) {
          for (var r = 0; r < rows; r++) {
            coords[rank - 2] = r;
            for (var c = 0; c < cols; c++) {
              coords[rank - 1] = c;
              if (c <= r + k) {
                result.setCell(coords, a.getCell(coords));
              } else {
                result.setCell(coords, zeroVal);
              }
            }
          }
          return;
        }
        for (var i = 0; i < a.shape[dim]; i++) {
          coords[dim] = i;
          walk(dim + 1);
        }
      }

      walk(0);
    }

    if (out != null) {
      if (useTempOut) {
        result.copy(out: out);
      }
      return out;
    }
    return result.detachToParentScope();
  });
}

/// Extract an upper triangular matrix (on and above the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
/// - It is an error if [a] is disposed.
/// - It is an error if [a] has rank < 2.
/// - It is an error if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> triu<T extends DTypeTag, Out extends T>(
  NDArray<T> a, {
  int k = 0,
  NDArray<Out>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute triu() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError.value(a, 'a', 'Input array must have rank >= 2');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Provided out buffer has incompatible shape or dtype',
      );
    }
  }

  return NDArray.scope(() {
    final bool useTempOut = out != null && sharesMemory(a, out);
    final result = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(a.shape, a.dtype);

    final rank = a.shape.length;
    final rows = a.shape[rank - 2];
    final cols = a.shape[rank - 1];

    final batchCount = a.shape.isEmpty
        ? 1
        : a.shape.sublist(0, rank - 2).isEmpty
        ? 1
        : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

    bool handledByFastKernel = false;
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_triu_double(
            a.pointer.cast(),
            result.pointer.cast(),
            batchCount,
            rows,
            cols,
            k,
          );
          handledByFastKernel = true;
        case DType.float32:
          v_triu_float(
            a.pointer.cast(),
            result.pointer.cast(),
            batchCount,
            rows,
            cols,
            k,
          );
          handledByFastKernel = true;
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

    if (!handledByFastKernel) {
      final zeroVal = castValue(0, a.dtype);
      final coords = List<int>.filled(rank, 0);

      void walk(int dim) {
        if (dim == rank - 2) {
          for (var r = 0; r < rows; r++) {
            coords[rank - 2] = r;
            for (var c = 0; c < cols; c++) {
              coords[rank - 1] = c;
              if (c >= r + k) {
                result.setCell(coords, a.getCell(coords));
              } else {
                result.setCell(coords, zeroVal);
              }
            }
          }
          return;
        }
        for (var i = 0; i < a.shape[dim]; i++) {
          coords[dim] = i;
          walk(dim + 1);
        }
      }

      walk(0);
    }

    if (out != null) {
      if (useTempOut) {
        result.copy(out: out);
      }
      return out;
    }
    return result.detachToParentScope();
  });
}

/// Calculates the n-th discrete difference along the given axis.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [n] must be non-negative.
/// - If provided, [out] must match the calculated shape and dtype.
/// - It is an error if [a] is disposed.
/// - It is an error if [n] is negative.
/// - It is an error if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> diff<T extends DTypeTag, Out extends T>(
  NDArray<T> a, {
  int n = 1,
  int axis = -1,
  NDArray<Out>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute diff() on a disposed array.');
  }
  if (n < 0) {
    throw ArgumentError.value(
      n,
      'n',
      'Order of difference n must be >= 0 (was $n)',
    );
  }
  if (n == 0) {
    return a.copy(out: out);
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(
      axis,
      -rank,
      rank - 1,
      'axis',
      'Must be within valid rank range',
    );
  }

  if (n >= a.shape[targetAxis]) {
    final emptyShape = List<int>.from(a.shape);
    emptyShape[targetAxis] = 0;
    if (out != null) {
      validateOutBuffer(out);
      if (!out.isWriteable ||
          !listEquals(out.shape, emptyShape) ||
          out.dtype != a.dtype) {
        throw ArgumentError.value(
          out,
          'out',
          'Incompatible out buffer shape or dtype for diff',
        );
      }
      return out;
    }
    return NDArray<T>.create(emptyShape, a.dtype);
  }

  final targetShape = List<int>.from(a.shape);
  targetShape[targetAxis] = a.shape[targetAxis] - n;

  if (out != null) {
    validateOutBuffer(out);
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Provided out buffer has incompatible shape or dtype',
      );
    }
  }

  return NDArray.scope(() {
    final bool useTempOut = out != null && sharesMemory(a, out);
    final result = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(targetShape, a.dtype);

    if (n > 1) {
      final step = diff(a, n: n - 1, axis: targetAxis);
      diff(step, n: 1, axis: targetAxis, out: result);
    } else {
      final rank = a.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(a.shape);
        final cStridesA = ScratchArena.copyInts(a.strides);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_diff_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.float32:
            s_diff_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.int64:
            s_diff_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.int32:
            s_diff_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.int16:
            s_diff_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.int8:
            s_diff_int8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.uint64:
            s_diff_uint64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.uint32:
            s_diff_uint32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.uint16:
            s_diff_uint16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.uint8:
            s_diff_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.boolean:
            s_diff_boolean(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.complex128:
            s_diff_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.complex64:
            s_diff_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          case DType.float16:
          case DType.bfloat16:
            final intA = castNDArray(a, DType.float64);
            final intRes = NDArray<Float64>.create(targetShape, DType.float64);
            final cStridesIntA = ScratchArena.copyInts(intA.strides);
            final cStridesIntRes = ScratchArena.copyInts(intRes.strides);

            s_diff_double(
              intA.pointer.cast(),
              cStridesIntA,
              intRes.pointer.cast(),
              cStridesIntRes,
              cShape,
              rank,
              targetAxis,
            );

            final castedRes = castNDArray(intRes, a.dtype);
            castedRes.copy(out: result);
            castedRes.dispose();
            intA.dispose();
            intRes.dispose();
        }
        checkNativeOom();
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (out != null) {
      if (useTempOut) {
        result.copy(out: out);
      }
      return out;
    }
    return result.detachToParentScope();
  });
}

/// Roll array elements along a given axis.
///
/// Elements that roll beyond the last position are re-introduced at the first.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - [shift] must be an integer, or a list of integers matching the length of [axis].
/// - [axis] must be null, an integer, or a list of integers within `[-rank, rank - 1]`.
/// - If [out] is provided, it must not be disposed and must have identical shape and dtype to [a].
/// - It is an error if [a] or [out] is disposed.
/// - It is an error if the shift/axis arguments are invalid or mismatched.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements.
/// - For 1D and contiguous N-D arrays, native C `memcpy` blocks are executed with zero Dart allocation overhead.
/// - Space Complexity is $O(N)$ for newly allocated output (or $O(1)$ auxiliary if [out] is provided).
///
/// **Reference:**
/// Refer to [NumPy roll documentation](https://numpy.org/doc/stable/reference/generated/numpy.roll.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> roll<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  Object shift, {
  Object? axis,
  NDArray<Out>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute roll() on a disposed array.');
  }

  // Parse shifts and axes
  final List<int> shifts;
  final List<int>? axes;

  if (axis == null) {
    if (shift is int) {
      shifts = [shift];
      axes = null;
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError.value(
          shift,
          'shift',
          'shift must be an integer when axis is null',
        );
      }
      shifts = shift;
      axes = null;
    } else {
      throw ArgumentError.value(
        shift,
        'shift',
        'shift must be an integer or a list of integers',
      );
    }
  } else if (axis is int) {
    if (shift is int) {
      shifts = [shift];
      axes = [axis];
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError.value(
          shift,
          'shift',
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = [axis];
    } else {
      throw ArgumentError.value(
        shift,
        'shift',
        'shift must be an integer or a list of integers',
      );
    }
  } else if (axis is List<int>) {
    if (shift is int) {
      shifts = List<int>.filled(axis.length, shift);
      axes = axis;
    } else if (shift is List<int>) {
      if (shift.length != axis.length) {
        throw ArgumentError.value(
          shift,
          'shift',
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = axis;
    } else {
      throw ArgumentError.value(
        shift,
        'shift',
        'shift must be an integer or a list of integers',
      );
    }
  } else {
    throw ArgumentError.value(
      axis,
      'axis',
      'axis must be null, an integer, or a list of integers',
    );
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Incompatible out buffer shape or dtype',
      );
    }
  }

  if (axes != null) {
    for (final ax in axes) {
      final normAx = ax < 0 ? a.rank + ax : ax;
      if (normAx < 0 || normAx >= a.rank) {
        throw RangeError.range(
          ax,
          -a.rank,
          a.rank - 1,
          'axis',
          'Must be within valid rank range',
        );
      }
    }
  }

  if (out != null && sharesMemory(a, out)) {
    final temp = roll(a, shift, axis: axis);
    try {
      temp.copy(out: out);
      return out;
    } finally {
      temp.dispose();
    }
  }

  if (a.rank == 0 || a.size == 0) {
    return a.copy(out: out);
  }

  if (axes == null) {
    if (a.rank == 1) {
      return _roll1D(a, shifts[0], out: out);
    }
    if (a.isContiguous) {
      final targetResult = out ?? NDArray<T>.create(a.shape, a.dtype);
      if (targetResult.isContiguous) {
        final rc = native_roll_1d(
          a.dtype.index,
          a.pointer.cast(),
          a.size,
          shifts[0],
          targetResult.pointer.cast(),
        );
        if (rc != 0) {
          if (out == null) targetResult.dispose();
          if (rc == -4) throw OutOfMemoryError();
          throw StateError('Native roll operation failed with code $rc');
        }
        return targetResult;
      } else {
        final contigRes = NDArray<T>.create(a.shape, a.dtype);
        try {
          final rc = native_roll_1d(
            a.dtype.index,
            a.pointer.cast(),
            a.size,
            shifts[0],
            contigRes.pointer.cast(),
          );
          if (rc != 0) {
            if (out == null) targetResult.dispose();
            if (rc == -4) throw OutOfMemoryError();
            throw StateError('Native roll operation failed with code $rc');
          }
          contigRes.copy(out: targetResult);
          return targetResult;
        } finally {
          contigRes.dispose();
        }
      }
    } else {
      final contigA = a.copy();
      try {
        final targetResult = out ?? NDArray<T>.create(a.shape, a.dtype);
        if (targetResult.isContiguous) {
          final rc = native_roll_1d(
            contigA.dtype.index,
            contigA.pointer.cast(),
            contigA.size,
            shifts[0],
            targetResult.pointer.cast(),
          );
          if (rc != 0) {
            if (out == null) targetResult.dispose();
            if (rc == -4) throw OutOfMemoryError();
            throw StateError('Native roll operation failed with code $rc');
          }
        } else {
          final contigRes = NDArray<T>.create(a.shape, a.dtype);
          try {
            final rc = native_roll_1d(
              contigA.dtype.index,
              contigA.pointer.cast(),
              contigA.size,
              shifts[0],
              contigRes.pointer.cast(),
            );
            if (rc != 0) {
              if (out == null) targetResult.dispose();
              if (rc == -4) throw OutOfMemoryError();
              throw StateError('Native roll operation failed with code $rc');
            }
            contigRes.copy(out: targetResult);
          } finally {
            contigRes.dispose();
          }
        }
        return targetResult;
      } finally {
        contigA.dispose();
      }
    }
  }

  final nonNullAxes = axes;
  if (nonNullAxes.isEmpty) {
    return a.copy(out: out);
  }
  if (nonNullAxes.length == 1) {
    return _rollSingleND(a, shifts[0], nonNullAxes[0], out: out);
  }

  if (nonNullAxes.length == 2) {
    final temp = NDArray<T>.create(a.shape, a.dtype);
    try {
      _rollSingleND(a, shifts[0], nonNullAxes[0], out: temp);
      final targetResult = out ?? NDArray<T>.create(a.shape, a.dtype);
      try {
        _rollSingleND(temp, shifts[1], nonNullAxes[1], out: targetResult);
      } catch (_) {
        if (out == null) targetResult.dispose();
        rethrow;
      }
      return targetResult;
    } finally {
      temp.dispose();
    }
  }

  return NDArray.scope(() {
    NDArray<T> current = a;
    for (var i = 0; i < nonNullAxes.length; i++) {
      final isLast = i == nonNullAxes.length - 1;
      final stepOut = isLast ? out : null;
      final next = _rollSingleND(
        current,
        shifts[i],
        nonNullAxes[i],
        out: stepOut,
      );
      current = next;
    }
    return out != null ? current : current.detachToParentScope();
  });
}

NDArray<T> _roll1D<T extends DTypeTag>(
  NDArray<T> a,
  int shift, {
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute _roll1D() on a disposed array.');
  }
  final size = a.size;
  if (size == 0) {
    return a.copy(out: out);
  }
  final s = shift % size;
  final realShift = s < 0 ? size + s : s;
  if (realShift == 0) {
    return a.copy(out: out);
  }

  final targetResult = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (a.isContiguous && targetResult.isContiguous) {
    final rc = native_roll_1d(
      a.dtype.index,
      a.pointer.cast(),
      size,
      realShift,
      targetResult.pointer.cast(),
    );
    if (rc != 0) {
      if (out == null) targetResult.dispose();
      if (rc == -4) throw OutOfMemoryError();
      throw StateError('Native roll operation failed with code $rc');
    }
    return targetResult;
  }

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.copyInt64s(a.shape);
    final cSrcStrides = ScratchArena.copyInt64s(a.strides);
    final cDestStrides = ScratchArena.copyInt64s(targetResult.strides);
    final rc = native_roll_nd(
      a.dtype.index,
      a.pointer.cast(),
      cShape,
      cSrcStrides,
      1,
      realShift,
      0,
      targetResult.pointer.cast(),
      cDestStrides,
    );
    if (rc != 0) {
      if (out == null) targetResult.dispose();
      if (rc == -4) throw OutOfMemoryError();
      throw StateError('Native roll operation failed with code $rc');
    }
  } finally {
    ScratchArena.reset(marker);
  }
  return targetResult;
}

NDArray<T> _rollSingleND<T extends DTypeTag>(
  NDArray<T> a,
  int shift,
  int axis, {
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute _rollSingleND() on a disposed array.');
  }
  final rank = a.rank;
  final normAx = axis < 0 ? rank + axis : axis;
  if (normAx < 0 || normAx >= rank) {
    throw RangeError.range(
      axis,
      -rank,
      rank - 1,
      'axis',
      'Must be within valid rank range',
    );
  }

  final dimSize = a.shape[normAx];
  if (dimSize == 0 || a.size == 0) {
    return a.copy(out: out);
  }

  final s = shift % dimSize;
  final realShift = s < 0 ? dimSize + s : s;
  if (realShift == 0) {
    return a.copy(out: out);
  }

  final targetResult = out ?? NDArray<T>.create(a.shape, a.dtype);

  if (rank == 1 && a.isContiguous && targetResult.isContiguous) {
    final rc = native_roll_1d(
      a.dtype.index,
      a.pointer.cast(),
      a.size,
      realShift,
      targetResult.pointer.cast(),
    );
    if (rc != 0) {
      if (out == null) targetResult.dispose();
      if (rc == -4) throw OutOfMemoryError();
      throw StateError('Native roll operation failed with code $rc');
    }
    return targetResult;
  }

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.copyInt64s(a.shape);
    final int rc;
    if (a.isContiguous && targetResult.isContiguous) {
      rc = native_roll_nd(
        a.dtype.index,
        a.pointer.cast(),
        cShape,
        ffi.nullptr,
        rank,
        realShift,
        normAx,
        targetResult.pointer.cast(),
        ffi.nullptr,
      );
    } else {
      final cSrcStrides = ScratchArena.copyInt64s(a.strides);
      final cDestStrides = ScratchArena.copyInt64s(targetResult.strides);
      rc = native_roll_nd(
        a.dtype.index,
        a.pointer.cast(),
        cShape,
        cSrcStrides,
        rank,
        realShift,
        normAx,
        targetResult.pointer.cast(),
        cDestStrides,
      );
    }
    if (rc != 0) {
      if (out == null) targetResult.dispose();
      if (rc == -4) throw OutOfMemoryError();
      throw StateError('Native roll operation failed with code $rc');
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return targetResult;
}

/// Returns a copy of array [a] cast to the specified [targetDType].
///
/// If [copy] is `false` and [targetDType] matches [a.dtype],
/// returns [a] directly without copying. Otherwise, allocates
/// and returns a new [NDArray] of type [R].
///
/// **Preconditions:**
/// - [a] must not be disposed.
///
/// **Performance considerations:**
/// - If [copy] is `false` and dtypes match, returns this in $O(1)$ time and $O(1)$ memory.
/// - Otherwise, complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Throws:**
/// - It is an error if [a] is already disposed.
///
/// **Example:**
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<R> astype<R extends DTypeTag>(
  NDArray<DTypeTag> a,
  DType<R> targetDType, {
  bool copy = true,
}) {
  return a.astype<R>(targetDType, copy: copy);
}
