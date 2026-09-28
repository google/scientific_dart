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

import 'backend/compute_engine.dart';

/// Base class for all tensor index and slice specifiers.
sealed class SliceSpec {
  /// Creates a [SliceSpec].
  const SliceSpec();
}

/// A slice range along an axis `[start:stop:step]`.
final class Slice extends SliceSpec {
  /// The inclusive start index of the slice range, or `null` for the axis default.
  final int? start;

  /// The exclusive stop index of the slice range, or `null` for the axis default.
  final int? stop;

  /// The step size of the slice range (defaults to `1`).
  final int step;

  /// Creates a [Slice] range `[start:stop:step]`.
  ///
  /// The [step] must not be zero.
  const Slice([this.start, this.stop, this.step = 1])
    : assert(step != 0, 'Slice step cannot be 0');

  /// Convenience constructor for a full axis slice `[:]`.
  const Slice.all() : start = null, stop = null, step = 1;

  @override
  String toString() => 'Slice($start, $stop, $step)';
}

/// Selects a single integer index along an axis, reducing the tensor rank by 1.
final class Index extends SliceSpec {
  /// The integer index along the axis (negative values count from the end).
  final int index;

  /// Creates an [Index] specifier for [index].
  const Index(this.index);

  @override
  String toString() => 'Index($index)';
}

/// Selects all elements along an axis (`:`).
final class All extends SliceSpec {
  /// Creates an [All] specifier.
  const All();

  @override
  String toString() => 'All()';
}

/// Introduces a new axis of size 1 at the specified position.
final class NewAxis extends SliceSpec {
  /// Creates a [NewAxis] specifier.
  const NewAxis();

  @override
  String toString() => 'NewAxis()';
}

/// Expands to fill any unspecified intermediate dimensions with [All].
final class Ellipsis extends SliceSpec {
  /// Creates an [Ellipsis] specifier.
  const Ellipsis();

  @override
  String toString() => 'Ellipsis()';
}

/// Computed geometry descriptor for a strided tensor subview.
final class SliceViewResult {
  /// Unmodifiable dimensions of the resulting view.
  final List<int> shape;

  /// Unmodifiable element strides of the resulting view.
  final List<int> strides;

  /// Offset in elements from the start of the underlying buffer.
  final int offsetElements;

  /// Whether the resulting view is C-contiguous in memory.
  final bool isContiguous;

  /// Creates a [SliceViewResult] with unmodifiable [shape] and [strides].
  SliceViewResult({
    required List<int> shape,
    required List<int> strides,
    required this.offsetElements,
    required this.isContiguous,
  }) : shape = List<int>.unmodifiable(shape),
       strides = List<int>.unmodifiable(strides);
}

/// Computes the shape, strides, and element offset of a subview defined by [specs].
///
/// It is an error if [specs] contains more than one [Ellipsis], if any [Slice]
/// has a `step` of `0`, if more non-[NewAxis] specifiers are given than the
/// rank of [shape], or if an integer [Index] is out of bounds for its axis.
SliceViewResult computeSliceView({
  required List<int> shape,
  required List<int> strides,
  required int offsetElements,
  required List<Object?> specs,
}) {
  final rank = shape.length;

  var ellipsisCount = 0;
  for (final spec in specs) {
    if (spec is Ellipsis) ellipsisCount++;
  }
  if (ellipsisCount > 1) {
    throw ArgumentError.value(
      specs,
      'specs',
      'Must contain at most one Ellipsis (...).',
    );
  }

  final normalizedSpecs = <Object?>[];
  var axisCount = 0;
  for (final spec in specs) {
    if (spec is! NewAxis && spec is! Ellipsis) {
      axisCount++;
    }
  }
  if (axisCount > rank) {
    throw RangeError('Too many indices ($axisCount) for array of rank $rank.');
  }

  final missingDims = rank - axisCount;
  for (final spec in specs) {
    if (spec is Ellipsis) {
      for (var i = 0; i < missingDims; i++) {
        normalizedSpecs.add(const All());
      }
      axisCount += missingDims;
    } else {
      normalizedSpecs.add(spec);
    }
  }

  while (axisCount < rank) {
    normalizedSpecs.add(const All());
    axisCount++;
  }

  final newShape = <int>[];
  final newStrides = <int>[];
  var newOffset = offsetElements;
  var currentAxis = 0;

  for (final spec in normalizedSpecs) {
    if (spec is NewAxis) {
      newShape.add(1);
      newStrides.add(0);
      continue;
    }

    if (currentAxis >= rank) {
      throw RangeError('Too many indices for array of rank $rank.');
    }

    final dim = shape[currentAxis];
    final stride = strides[currentAxis];

    if (spec is int || spec is Index) {
      final rawIndex = (spec is Index) ? spec.index : (spec as int);
      final resolvedIndex = rawIndex < 0 ? rawIndex + dim : rawIndex;
      if (resolvedIndex < 0 || resolvedIndex >= dim) {
        throw RangeError.index(
          rawIndex,
          shape,
          'axis $currentAxis',
          'Index $rawIndex is out of bounds for axis $currentAxis with size $dim.',
          dim,
        );
      }
      newOffset += resolvedIndex * stride;
      currentAxis++;
    } else if (spec is All) {
      newShape.add(dim);
      newStrides.add(stride);
      currentAxis++;
    } else if (spec is Slice) {
      final step = spec.step;
      if (step == 0) {
        throw ArgumentError.value(step, 'step', 'Must not be zero.');
      }
      int start;
      int stop;

      if (step > 0) {
        start = spec.start ?? 0;
        stop = spec.stop ?? dim;

        if (start < 0) start += dim;
        if (stop < 0) stop += dim;

        start = start.clamp(0, dim);
        stop = stop.clamp(0, dim);

        if (start >= stop) {
          newShape.add(0);
          newStrides.add(stride * step);
        } else {
          final count = ((stop - start - 1) ~/ step) + 1;
          newShape.add(count);
          newStrides.add(stride * step);
          newOffset += start * stride;
        }
      } else {
        start = spec.start ?? (dim - 1);
        stop = spec.stop ?? -1;

        if (start < 0) start += dim;
        if (spec.stop != null && stop < 0) stop += dim;

        start = start.clamp(-1, dim - 1);
        stop = stop.clamp(-1, dim - 1);

        if (start <= stop) {
          newShape.add(0);
          newStrides.add(stride * step);
        } else {
          final count = ((start - stop - 1) ~/ (-step)) + 1;
          newShape.add(count);
          newStrides.add(stride * step);
          newOffset += start * stride;
        }
      }
      currentAxis++;
    } else {
      throw ArgumentError.value(
        spec,
        'specs',
        'Must be an int, Index, Slice, All, NewAxis, or Ellipsis.',
      );
    }
  }

  while (currentAxis < rank) {
    newShape.add(shape[currentAxis]);
    newStrides.add(strides[currentAxis]);
    currentAxis++;
  }

  final isContiguous = isContiguousLayout(newShape, newStrides);

  return SliceViewResult(
    shape: newShape,
    strides: newStrides,
    offsetElements: newOffset,
    isContiguous: isContiguous,
  );
}
