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

/// Base class for all tensor index/slice specifiers.
sealed class SliceSpec {
  const SliceSpec();
}

/// A slice range along an axis `[start:stop:step]`.
final class Slice extends SliceSpec {
  /// The start index of the slice range.
  final int? start;

  /// The stop index of the slice range (exclusive).
  final int? stop;

  /// The step size of the slice range (default is 1).
  final int step;

  const Slice([this.start, this.stop, this.step = 1])
    : assert(step != 0, 'Slice step cannot be 0');

  /// Convenience for a full axis slice `[:]`.
  const Slice.all() : start = null, stop = null, step = 1;

  @override
  String toString() => 'Slice($start, $stop, $step)';
}

/// Selects a single integer index along an axis, reducing the rank by 1.
final class Index extends SliceSpec {
  final int index;
  const Index(this.index);

  @override
  String toString() => 'Index($index)';
}

/// Represents selecting all elements along an axis (`:`).
final class All extends SliceSpec {
  const All();

  @override
  String toString() => 'All()';
}

/// Introduces a new axis of size 1 at the specified position.
final class NewAxis extends SliceSpec {
  const NewAxis();

  @override
  String toString() => 'NewAxis()';
}

/// Expands to fill any unspecified intermediate dimensions.
final class Ellipsis extends SliceSpec {
  const Ellipsis();

  @override
  String toString() => 'Ellipsis()';
}

/// Computed geometry descriptor for a strided subview.
final class SliceViewResult {
  final List<int> shape;
  final List<int> strides;
  final int offsetElements;
  final bool isContiguous;

  const SliceViewResult({
    required this.shape,
    required this.strides,
    required this.offsetElements,
    required this.isContiguous,
  });
}

/// Computes the shape, strides, and element offset of a subview defined by [specs].
SliceViewResult computeSliceView({
  required List<int> shape,
  required List<int> strides,
  required int offsetElements,
  required List<dynamic> specs,
}) {
  final rank = shape.length;

  // 1. Expand Ellipsis if present
  var ellipsisCount = 0;
  for (final spec in specs) {
    if (spec is Ellipsis) ellipsisCount++;
  }
  if (ellipsisCount > 1) {
    throw ArgumentError('An index can only have a single ellipsis (...)');
  }

  final normalizedSpecs = <dynamic>[];
  var axisCount = 0;
  for (final spec in specs) {
    if (spec is! NewAxis && spec is! Ellipsis) {
      axisCount++;
    }
  }

  final missingDims = rank - axisCount;
  for (final spec in specs) {
    if (spec is Ellipsis) {
      for (var i = 0; i < missingDims; i++) {
        normalizedSpecs.add(const All());
      }
    } else {
      normalizedSpecs.add(spec);
    }
  }

  // Pad remaining axes with All() if fewer specs than rank
  while (axisCount < rank && normalizedSpecs.length < rank) {
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
      throw RangeError('Too many indices for array of rank $rank');
    }

    final dim = shape[currentAxis];
    final stride = strides[currentAxis];

    if (spec is int || spec is Index) {
      var idx = (spec is Index) ? spec.index : (spec as int);
      if (idx < 0) idx += dim;
      if (idx < 0 || idx >= dim) {
        throw IndexError.withLength(
          idx,
          dim,
          indexable: shape,
          name: 'axis $currentAxis',
        );
      }
      newOffset += idx * stride;
      currentAxis++;
    } else if (spec is All) {
      newShape.add(dim);
      newStrides.add(stride);
      currentAxis++;
    } else if (spec is Slice) {
      final step = spec.step;
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
      throw ArgumentError(
        'Unsupported slice specifier: $spec (${spec.runtimeType})',
      );
    }
  }

  // Trailing dimensions not explicitly specified
  while (currentAxis < rank) {
    newShape.add(shape[currentAxis]);
    newStrides.add(strides[currentAxis]);
    currentAxis++;
  }

  final isContig = ShapeUtils.isContiguous(newShape, newStrides);

  return SliceViewResult(
    shape: List.unmodifiable(newShape),
    strides: List.unmodifiable(newStrides),
    offsetElements: newOffset,
    isContiguous: isContig,
  );
}
