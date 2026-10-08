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

import 'package:meta/meta.dart';

import 'ndarray.dart';

/// A high-performance, zero-allocation multi-dimensional iterator for [NDArray].
///
/// Iterates over the multi-dimensional coordinates and corresponding memory
/// offsets of one or more [NDArray] objects in standard lexicographical (C-contiguous)
/// order.
///
/// To achieve maximum performance and zero heap allocation during iteration,
/// [NDIter] reuses the same list of coordinates and updates it in-place.
/// Therefore, the coordinates returned by [coords] must not be stored or
/// modified by the consumer.
///
/// Supports iterating over a single array, or multiple arrays simultaneously
/// by broadcasting their shapes to a compatible common shape.
///
/// **Preconditions:**
/// - When broadcasting multiple arrays, all shapes must be compatible for broadcasting.
///
/// It is an error if any array passed to the iterator has been disposed.
/// It is an error if the list of arrays is empty, or if shapes are incompatible.
///
/// **Example:**
/// {@example /example/ndarray_example.dart lang=dart}
final class NDIter {
  final List<int> _shape;
  final int _rank;
  final List<int> _coords;

  final int _numArrays;
  final List<List<int>> _strides;
  final List<int> _offsets;
  final List<int> _absoluteOffsets;

  bool _isStarted = false;
  bool _hasMore = true;

  /// Internal constructor holding the unified initialization logic.
  NDIter._internal(List<NDArray<DTypeTag>> arrays, List<int> commonShape)
    : _shape = List<int>.from(commonShape),
      _rank = commonShape.length,
      _coords = List<int>.filled(commonShape.length, 0),
      _numArrays = arrays.length,
      _offsets = List<int>.filled(arrays.length, 0),
      _absoluteOffsets = arrays.map((e) => e.offsetElements).toList(),
      _strides = arrays.map((a) {
        if (a.isDisposed) {
          throw StateError('Cannot construct NDIter on a disposed array.');
        }
        return NDIter._broadcastStrides(a.shape, a.strides, commonShape);
      }).toList() {
    if (arrays.isEmpty) {
      throw ArgumentError.value(
        arrays,
        'arrays',
        'Must provide at least one array for NDIter',
      );
    }
    if (_shape.any((dim) => dim == 0)) {
      _hasMore = false;
    }
  }

  /// Creates an iterator over a single [array].
  ///
  /// **Performance considerations:**
  /// - Iteration (calling [moveNext]) is zero-allocation.
  /// - Construction allocates internal helper lists to track state.
  ///
  NDIter(NDArray<DTypeTag> array) : this._internal([array], array.shape);

  /// Creates an iterator that iterates over two arrays simultaneously,
  /// broadcasting their shapes to a common compatible shape.
  ///
  /// **Performance considerations:**
  /// - Iteration (calling [moveNext]) is zero-allocation.
  /// - Construction allocates internal helper lists to track state.
  ///
  NDIter.broadcast2(NDArray<DTypeTag> a, NDArray<DTypeTag> b)
    : this._internal([a, b], NDIter._broadcastShapes(a.shape, b.shape));

  /// Creates an iterator that iterates over three arrays simultaneously,
  /// broadcasting their shapes to a common compatible shape.
  NDIter.broadcast3(
    NDArray<DTypeTag> a,
    NDArray<DTypeTag> b,
    NDArray<DTypeTag> c,
  ) : this._internal(
        [a, b, c],
        NDIter._broadcastShapes(
          NDIter._broadcastShapes(a.shape, b.shape),
          c.shape,
        ),
      );

  /// Creates an iterator that iterates over a list of [arrays] simultaneously,
  /// broadcasting their shapes to a common compatible shape.
  ///
  /// **Performance considerations:**
  /// - Iteration (calling [moveNext]) is zero-allocation.
  /// - Construction allocates internal helper lists to track state.
  ///
  factory NDIter.broadcast(List<NDArray<DTypeTag>> arrays) {
    final copy = List<NDArray<DTypeTag>>.of(arrays);
    if (copy.isEmpty) {
      throw ArgumentError.value(arrays, 'arrays', 'Must not be empty');
    }
    return NDIter._internal(
      copy,
      copy.map((a) => a.shape).reduce(NDIter._broadcastShapes),
    );
  }

  /// Moves the iterator to the next multi-dimensional element position.
  ///
  /// Returns `true` if the iterator successfully advanced to the next element,
  /// or `false` if the iteration is complete.
  bool moveNext() {
    if (!_hasMore) return false;
    if (!_isStarted) {
      _isStarted = true;
      return true;
    }

    for (var d = _rank - 1; d >= 0; d--) {
      _coords[d]++;
      if (_coords[d] < _shape[d]) {
        for (var i = 0; i < _numArrays; i++) {
          _offsets[i] += _strides[i][d];
        }
        return true;
      }
      _coords[d] = 0;
      for (var i = 0; i < _numArrays; i++) {
        _offsets[i] -= (_shape[d] - 1) * _strides[i][d];
      }
    }

    _hasMore = false;
    return false;
  }

  /// The broadcasted shape of the iteration space.
  List<int> get shape => List<int>.unmodifiable(_shape);

  /// The current multi-dimensional coordinates of the iteration.
  ///
  /// **Warning:** The returned list is mutated in-place by [moveNext].
  /// Do not store or modify it.
  List<int> get coords => _coords;

  /// The current flat index in the underlying array's data buffer.
  ///
  /// This is the absolute physical storage index in the underlying buffer for the first array.
  int get index => _absoluteOffsets[0] + _offsets[0];

  /// Returns the current flat index in the underlying data buffer for the array at [arrayIndex].
  ///
  /// **Preconditions:**
  /// - [arrayIndex] must be greater than or equal to 0 and less than the number of arrays being iterated.
  ///
  /// It is an error if [arrayIndex] is out of bounds.
  int getIndex(int arrayIndex) {
    if (arrayIndex < 0 || arrayIndex >= _numArrays) {
      throw RangeError.range(arrayIndex, 0, _numArrays - 1, 'arrayIndex');
    }
    return _absoluteOffsets[arrayIndex] + _offsets[arrayIndex];
  }

  /// The number of arrays being iterated simultaneously.
  int get numArrays => _numArrays;

  /// Helper to broadcast two shapes to a compatible common shape.
  static List<int> _broadcastShapes(List<int> shapeA, List<int> shapeB) {
    final maxLen = shapeA.length > shapeB.length
        ? shapeA.length
        : shapeB.length;
    final commonShape = List<int>.filled(maxLen, 1);
    for (var i = 0; i < maxLen; i++) {
      final dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
      final dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;
      if (dimA == dimB) {
        commonShape[maxLen - 1 - i] = dimA;
      } else if (dimA == 1) {
        commonShape[maxLen - 1 - i] = dimB;
      } else if (dimB == 1) {
        commonShape[maxLen - 1 - i] = dimA;
      } else {
        throw ArgumentError.value(
          shapeB,
          'shapeB',
          'Must be compatible for broadcasting with $shapeA (got $shapeB)',
        );
      }
    }
    return commonShape;
  }

  /// Helper to compute broadcasted strides for a target shape.
  static List<int> _broadcastStrides(
    List<int> shape,
    List<int> strides,
    List<int> targetShape,
  ) {
    if (shape.length > targetShape.length) {
      throw ArgumentError.value(
        targetShape,
        'targetShape',
        'Must be compatible for broadcasting from shape $shape to targetShape $targetShape',
      );
    }
    final newStrides = List<int>.filled(targetShape.length, 0);
    for (var i = 0; i < shape.length; i++) {
      final targetDimIdx = targetShape.length - 1 - i;
      final origDimIdx = shape.length - 1 - i;
      final dimSize = shape[origDimIdx];
      if (dimSize == targetShape[targetDimIdx]) {
        newStrides[targetDimIdx] = strides[origDimIdx];
      } else if (dimSize == 1) {
        newStrides[targetDimIdx] = 0;
      } else {
        throw ArgumentError.value(
          shape,
          'shape',
          'Must be compatible for broadcasting from shape $shape to targetShape $targetShape',
        );
      }
    }
    return newStrides;
  }
}

/// A high-performance zero-allocation multi-dimensional enumeration helper.
///
/// Yields multi-dimensional coordinates and cell values of an [NDArray]
/// in standard lexicographical (C-contiguous) order.
///
/// It is an error if the array has been disposed.
///
/// **Example:**
/// {@example /example/ndarray_example.dart lang=dart}
final class NDEnumerate<T extends DTypeTag> {
  final NDArray<T> _array;
  final NDIter _iter;

  /// Creates an enumeration over the specified [array].
  ///
  NDEnumerate(NDArray<T> array) : _array = array, _iter = NDIter(array);

  /// Advances to the next element.
  ///
  /// Returns `true` if another element is available, or `false` if the
  /// enumeration is complete.
  bool moveNext() => _iter.moveNext();

  /// The current multi-dimensional coordinates of the enumeration.
  ///
  /// **Warning:** The returned list is mutated in-place by [moveNext].
  /// Do not store or modify it.
  List<int> get coords => _iter.coords;

  /// The current element value, untyped.
  ///
  /// The tag [T] does not name the element type, so this getter cannot be
  /// typed. Use [NDEnumerateElements.value] for the element type implied by
  /// the tag.
  @internal
  Object? get valueRaw => _array.getCellRawUntyped(_iter.index);
}

/// The element-typed view of an [NDEnumerate] whose dtype tag implements
/// [ElementOf].
///
/// The element type [E] is recovered from the enumerated array's dtype tag
/// [T] through its [ElementOf] bound, so `NDEnumerate<Float64>.value` has
/// static type `double` and `NDEnumerate<Int32>.value` has static type `int`.
///
/// Enumerations whose static tag does not specify an element type (such as
/// `NDEnumerate<DTypeTag>` or `NDEnumerate<BitwiseDType>`) fall back to
/// [NDEnumerateBaseElements], where `value` is typed as `dynamic`.
extension NDEnumerateElements<T extends ElementOf<E>, E> on NDEnumerate<T> {
  /// The current element value.
  E get value => valueRaw as E;
}

/// Element access for an [NDEnumerate] typed as the base [DTypeTag].
extension NDEnumerateBaseElements on NDEnumerate<DTypeTag> {
  /// The current element value.
  dynamic get value => valueRaw;
}
