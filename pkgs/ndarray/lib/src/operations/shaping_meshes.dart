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
import 'broadcasting.dart';
import 'helpers.dart';
import 'manipulation.dart';
import 'spacers.dart';

/// Represents a range specification for meshgrid operations.
///
/// Generates evenly-spaced coordinate values along a grid dimension.
///
/// **Preconditions:**
/// - [step] must be non-zero.
/// - If [numPoints] is specified, it must be strictly positive ($\ge 1$).
///
/// - It is an error if [step] is 0.
/// - It is an error if [numPoints] is negative or 0.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
final class GridRange {
  /// The starting value of the range (inclusive).
  final double start;

  /// The ending value of the range.
  ///
  /// If [numPoints] is specified, [stop] is inclusive.
  /// Otherwise, [stop] is exclusive (standard half-open range).
  final double stop;

  /// The step size between adjacent points.
  /// Only active if [numPoints] is null. Defaults to 1.0.
  final double step;

  /// The total number of points to generate.
  /// If not null, the points are generated evenly spaced between [start]
  /// and [stop] (inclusive), and [step] is ignored.
  final int? numPoints;

  /// Creates a new grid range specification.
  GridRange(this.start, this.stop, {this.step = 1.0, this.numPoints}) {
    if (step == 0.0) {
      throw ArgumentError.value(step, 'step', 'Step cannot be zero');
    }
    if (numPoints != null && numPoints! <= 0) {
      throw ArgumentError.value(
        numPoints,
        'numPoints',
        'numPoints must be positive',
      );
    }
  }

  /// Creates a grid range specification using NumPy-style parameters.
  ///
  /// If [step] is a [Complex] number, the magnitude of [step] (truncated to integer)
  /// is interpreted as specifying the [numPoints] count between [start] and [stop],
  /// where [stop] is inclusive (matching NumPy's complex step behavior).
  /// Otherwise, if [step] is a real [num], it is treated as the step size
  /// where [stop] is exclusive.
  factory GridRange.numpy(double start, double stop, Object step) {
    if (step is Complex) {
      return GridRange(start, stop, numPoints: step.abs.toInt());
    } else if (step is num) {
      return GridRange(start, stop, step: step.toDouble());
    } else {
      throw ArgumentError.value(
        step,
        'step',
        'step must be a num or a Complex number',
      );
    }
  }
}

/// Helper to generate a 1D coordinate array from a [GridRange].
NDArray<Float64> _generate1DCoordinate(GridRange range, DType<Float64> dtype) {
  if (range.numPoints != null) {
    return linspace<Float64>(
      range.start,
      range.stop,
      range.numPoints!,
      dtype: dtype,
    );
  } else {
    return NDArray<Float64>.arange(
      range.start,
      range.stop,
      step: range.step,
      dtype: dtype,
    );
  }
}

/// Creates a view of [x] with the given [shape] and [strides].
///
/// [shape] and [strides] default to the array's shape and strides if not provided.
///
/// **Memory Safety Warning:**
/// This function is extremely low-level and does not perform bounds safety checks
/// on memory accesses. The user must ensure that the specified shape and strides
/// do not reference memory outside of the underlying C buffer. Accessing elements
/// out of bounds will result in memory corruption, undefined behavior, or crashes.
///
/// **Preconditions:**
/// - [x] must not be disposed.
/// - If [shape] is provided, all its dimensions must be positive.
/// - If [strides] is provided, its length must match the length of [shape].
///
/// - It is an error if [x] has been disposed.
/// - It is an error if [shape] and [strides] lengths do not match.
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [x] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// Refer to the [NumPy as_strided reference](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)
/// for details.
NDArray<T> asStrided<T extends DTypeTag>(
  NDArray<T> x, {
  List<int>? shape,
  List<int>? strides,
}) {
  if (x.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  final targetShape = shape ?? x.shape;
  final targetStrides = strides ?? x.strides;

  if (targetShape.length != targetStrides.length) {
    throw ArgumentError.value(
      targetStrides,
      'strides',
      'Shape length (${targetShape.length}) must match strides length (${targetStrides.length})',
    );
  }

  checkTotalSize(targetShape);

  return NDArray<T>.view(x, shape: targetShape, strides: targetStrides);
}

/// Returns an open multi-dimensional mesh-grid.
///
/// Returns a list of broadcastable [NDArray]s, one for each dimension range
/// in [ranges]. Each returned array has shape `[1, ..., N_i, ..., 1]` with length
/// greater than 1 in only the $i$-th dimension, enabling memory-efficient broadcasting.
///
/// Specify dimension ranges using [GridRange] or [GridRange.numpy].
/// If a [GridRange] is specified with a [Complex] step via [GridRange.numpy],
/// the magnitude of the step is interpreted as the total number of points (`numPoints`)
/// between `start` and `stop` (inclusive). Otherwise, real steps create standard
/// half-open ranges (exclusive of `stop`).
///
/// **Preconditions:**
/// - [ranges] must not be empty.
///
/// - It is an error if [ranges] is empty.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// Refer to the [NumPy ogrid reference](https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html)
/// for details.
List<NDArray<Float64>> ogrid(
  List<GridRange> ranges, {
  DType<Float64> dtype = DType.float64,
  List<NDArray<Float64>>? out,
}) {
  if (ranges.isEmpty) {
    throw ArgumentError.value(ranges, 'ranges', 'ranges must not be empty');
  }
  if (out != null) {
    if (out.length != ranges.length) {
      throw ArgumentError.value(
        out,
        'out',
        'Length of out (${out.length}) must match length of ranges (${ranges.length})',
      );
    }
    for (var i = 0; i < ranges.length; i++) {
      if (out[i].isDisposed) {
        throw StateError(
          'Cannot write ogrid result to a disposed output array.',
        );
      }
      validateOutBuffer(out[i], 'out[$i]');
    }
  }

  return NDArray.scope(() {
    final k = ranges.length;
    final results = <NDArray<Float64>>[];

    for (var i = 0; i < k; i++) {
      final arr1D = _generate1DCoordinate(ranges[i], dtype);
      final shape = List<int>.filled(k, 1);
      shape[i] = arr1D.size;

      if (out != null) {
        final targetOut = out[i];
        if (!listEquals(targetOut.shape, shape) || targetOut.dtype != dtype) {
          throw ArgumentError.value(
            targetOut,
            'out',
            'Incompatible out buffer shape or dtype',
          );
        }
        final reshaped = arr1D.reshape(shape);
        reshaped.copy(out: targetOut);
        results.add(targetOut);
      } else {
        final reshaped = arr1D.reshape(shape);
        results.add(reshaped.copy().detachToParentScope());
      }
    }

    return results;
  });
}

/// Returns a dense multi-dimensional mesh-grid.
///
/// Returns a single contiguous [NDArray] containing the fleshed-out coordinate grids.
/// The output array has shape `[k, d1, d2, ..., dk]`, where `k = ranges.length`.
///
/// Dimension ranges are specified using [GridRange] or [GridRange.numpy].
/// If a [GridRange] is specified with a [Complex] step via [GridRange.numpy],
/// the magnitude of the step is interpreted as the number of points (`numPoints`)
/// between `start` and `stop` (inclusive). Otherwise, real steps create standard
/// half-open ranges (exclusive of `stop`).
///
/// **Preconditions:**
/// - [ranges] must not be empty.
///
/// - It is an error if [ranges] is empty.
/// - It is an error if [out] is provided with incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Materializes full dense coordinates in memory with time and space complexity $O(k \prod d_i)$.
/// - For memory-efficient operations without materializing dense arrays, consider [ogrid].
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// Refer to the [NumPy mgrid reference](https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html)
/// for details.
/// If [out] is provided, writes the resulting grid array into it.
NDArray<Float64> mgrid(
  List<GridRange> ranges, {
  DType<Float64> dtype = DType.float64,
  NDArray<Float64>? out,
}) {
  if (ranges.isEmpty) {
    throw ArgumentError.value(ranges, 'ranges', 'ranges must not be empty');
  }
  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot write mgrid result to a disposed output array.');
    }
    validateOutBuffer(out);
  }

  return NDArray.scope(() {
    final k = ranges.length;

    // 1. Generate 1D coordinates to determine shape of the grid
    final allCoords = <NDArray<Float64>>[];
    final gridShape = <int>[];

    for (var i = 0; i < k; i++) {
      final arr1D = _generate1DCoordinate(ranges[i], dtype);
      allCoords.add(arr1D);
      gridShape.add(arr1D.size);
    }

    // 2. Broadcast each 1D coordinate to gridShape
    final broadcastedGrids = <NDArray<Float64>>[];
    for (var i = 0; i < k; i++) {
      final shape1D = List<int>.filled(k, 1);
      shape1D[i] = gridShape[i];
      final reshaped = allCoords[i].reshape(shape1D);
      final broadcasted = broadcastTo(reshaped, gridShape);
      broadcastedGrids.add(broadcasted);
    }

    // 3. Stack all broadcasted grids along axis 0
    final result = stack<Float64>(broadcastedGrids, axis: 0, out: out);
    if (out != null) {
      return out;
    }
    return result.detachToParentScope();
  });
}
