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
import '../ndarray.dart';
import 'helpers.dart';

// Standalone operational relative cross-imports

/// Helper to slice an array along a specific axis between [start] and [stop].
NDArray<T> _sliceAlongAxis<T extends DTypeTag>(
  NDArray<T> a,
  int axis,
  int start,
  int stop,
) {
  final selectors = List<Selector>.filled(a.shape.length, Slice.all());
  selectors[axis] = Slice(start: start, stop: stop);
  return a.slice(selectors);
}

/// Splits an array into multiple sub-arrays of as equal size as possible.
///
/// The array is divided into [sections] along [axis]. If such a split is not possible,
/// the first $L \pmod{\text{sections}}$ sub-arrays will have size $L // \text{sections} + 1$,
/// and the rest will have size $L // \text{sections}$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [sections] must be strictly positive ($\ge 1$).
/// - [axis] must be within bounds `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [sections] is less than or equal to 0.
/// - It is an error if [axis] is out of bounds.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy array_split reference](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html)
/// for details.
List<NDArray<T>> array_split<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  int sections, {
  int axis = 0,
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (out != null && out.length != sections) {
    throw ArgumentError.value(
      out,
      'out',
      'Length of out (${out.length}) must match sections ($sections)',
    );
  }
  if (sections <= 0) {
    throw ArgumentError.value(
      sections,
      'sections',
      'Number of sections must be positive',
    );
  }

  final rank = a.rank;
  if (axis < -rank || axis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  final L = a.shape[normAxis];

  final List<NDArray<T>> results = [];

  final S_0 = L ~/ sections;
  final rem = L % sections;
  var currentIdx = 0;

  final bool needsCopy =
      out != null &&
      out.any((outSub) => !outSub.isDisposed && sharesMemory(a, outSub));
  final NDArray<T> aSrc = needsCopy ? a.copy() : a;
  try {
    for (var i = 0; i < sections; i++) {
      final size = i < rem ? S_0 + 1 : S_0;
      final start = currentIdx;
      final stop = currentIdx + size;
      currentIdx = stop;

      final sub = _sliceAlongAxis(aSrc, normAxis, start, stop);
      if (out != null) {
        try {
          final outSub = out[i];
          if (outSub.isDisposed) {
            throw StateError('Cannot write to a disposed out array.');
          }
          validateOutBuffer(outSub, 'out[$i]');
          if (!listEquals(outSub.shape, sub.shape) ||
              outSub.dtype != sub.dtype) {
            throw ArgumentError.value(
              outSub,
              'out',
              'Incompatible out buffer shape or dtype for split item $i',
            );
          }
          sub.copy(out: outSub);
          results.add(outSub);
        } finally {
          sub.dispose();
        }
      } else {
        results.add(sub);
      }
    }
  } finally {
    if (needsCopy) {
      aSrc.dispose();
    }
  }

  return results;
}

/// Splits an array along [axis] at the coordinate points specified in [indices].
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [axis] must be within bounds `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy array_split reference](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html)
/// for details.
List<NDArray<T>> array_split_at<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  List<int> indices, {
  int axis = 0,
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final rank = a.rank;
  if (axis < -rank || axis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  final L = a.shape[normAxis];

  final boundaries = <int>[0];
  for (final p in indices) {
    boundaries.add(p.clamp(0, L));
  }
  boundaries.add(L);

  if (out != null && out.length != boundaries.length - 1) {
    throw ArgumentError.value(
      out,
      'out',
      'Length of out (${out.length}) must match split sections count (${boundaries.length - 1})',
    );
  }

  final List<NDArray<T>> results = [];

  final bool needsCopy =
      out != null &&
      out.any((outSub) => !outSub.isDisposed && sharesMemory(a, outSub));
  final NDArray<T> aSrc = needsCopy ? a.copy() : a;
  try {
    for (var i = 0; i < boundaries.length - 1; i++) {
      final start = boundaries[i];
      final stop = boundaries[i + 1];
      final sub = _sliceAlongAxis(aSrc, normAxis, start, stop);
      if (out != null) {
        try {
          final outSub = out[i];
          if (outSub.isDisposed) {
            throw StateError('Cannot write to a disposed out array.');
          }
          validateOutBuffer(outSub, 'out[$i]');
          if (!listEquals(outSub.shape, sub.shape) ||
              outSub.dtype != sub.dtype) {
            throw ArgumentError.value(
              outSub,
              'out',
              'Incompatible out buffer shape or dtype for split item $i',
            );
          }
          sub.copy(out: outSub);
          results.add(outSub);
        } finally {
          sub.dispose();
        }
      } else {
        results.add(sub);
      }
    }
  } finally {
    if (needsCopy) {
      aSrc.dispose();
    }
  }

  return results;
}

/// Splits an array into multiple sub-arrays of strictly equal size.
///
/// The array is divided into [sections] along [axis]. Throws [ArgumentError]
/// if the split does not result in an equal division.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [sections] must be strictly positive ($\ge 1$).
/// - [axis] must be within bounds `[-rank, rank - 1]`.
/// - The dimension size along [axis] must be divisible by [sections].
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [sections] is less than or equal to 0.
/// - It is an error if division is not equal.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy split reference](https://numpy.org/doc/stable/reference/generated/numpy.split.html)
/// for details.
List<NDArray<T>> split<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  int sections, {
  int axis = 0,
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final rank = a.rank;
  if (axis < -rank || axis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  final L = a.shape[normAxis];

  if (sections <= 0) {
    throw ArgumentError.value(
      sections,
      'sections',
      'Number of sections must be positive',
    );
  }
  if (L % sections != 0) {
    throw ArgumentError.value(
      sections,
      'sections',
      'array split does not result in an equal division: '
          'dimension size along axis $normAxis is $L, which is not divisible by $sections',
    );
  }

  return array_split(a, sections, axis: normAxis, out: out);
}

/// Splits an array along [axis] at the coordinate points specified in [indices].
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [axis] must be within bounds `[-rank, rank - 1]`.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [axis] is out of bounds.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy split reference](https://numpy.org/doc/stable/reference/generated/numpy.split.html)
/// for details.
List<NDArray<T>> split_at<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  List<int> indices, {
  int axis = 0,
  List<NDArray<Out>>? out,
}) {
  return array_split_at(a, indices, axis: axis, out: out);
}

/// Splits an array horizontally (column-wise) into [sections] equal sub-arrays.
///
/// Equivalent to [split] with `axis = 1` for arrays of rank $\ge 2$, and
/// `axis = 0` for 1D arrays.
///
/// **Preconditions:**
/// - [a] must not be disposed.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [a] is 0-dimensional or if the split is invalid.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy hsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html)
/// for details.
List<NDArray<T>> hsplit<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  int sections, {
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank == 0) {
    throw ArgumentError.value(a, 'a', 'Cannot hsplit a 0D array');
  }
  final axis = a.rank == 1 ? 0 : 1;
  return split(a, sections, axis: axis, out: out);
}

/// Splits an array horizontally (column-wise) at the coordinate points specified in [indices].
///
/// Equivalent to [split_at] with `axis = 1` for arrays of rank $\ge 2$, and
/// `axis = 0` for 1D arrays.
///
/// **Preconditions:**
/// - [a] must not be disposed.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [a] is 0-dimensional or if the split is invalid.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy hsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html)
/// for details.
List<NDArray<T>> hsplit_at<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  List<int> indices, {
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank == 0) {
    throw ArgumentError.value(a, 'a', 'Cannot hsplit a 0D array');
  }
  final axis = a.rank == 1 ? 0 : 1;
  return split_at(a, indices, axis: axis, out: out);
}

/// Splits an array vertically (row-wise) into [sections] equal sub-arrays.
///
/// Equivalent to [split] with `axis = 0`. Requires rank $\ge 2$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have rank $\ge 2$.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [a] has rank $< 2$.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy vsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html)
/// for details.
List<NDArray<T>> vsplit<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  int sections, {
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a,
      'a',
      'vsplit only supports arrays of rank >= 2',
    );
  }
  return split(a, sections, axis: 0, out: out);
}

/// Splits an array vertically (row-wise) at the coordinate points specified in [indices].
///
/// Equivalent to [split_at] with `axis = 0`. Requires rank $\ge 2$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have rank $\ge 2$.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [a] has rank $< 2$.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy vsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html)
/// for details.
List<NDArray<T>> vsplit_at<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  List<int> indices, {
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank < 2) {
    throw ArgumentError.value(
      a,
      'a',
      'vsplit only supports arrays of rank >= 2',
    );
  }
  return split_at(a, indices, axis: 0, out: out);
}

/// Splits an array into multiple sub-arrays of strictly equal size along the 3rd axis (axis 2).
///
/// Equivalent to [split] with `axis = 2`. Requires rank $\ge 3$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have rank $\ge 3$.
/// - [sections] must be strictly positive ($\ge 1$).
/// - The dimension size along axis 2 must be divisible by [sections].
///
/// - It is an error if [a] is disposed.
/// - It is an error if [a] has rank $< 3$.
/// - It is an error if [sections] is less than or equal to 0.
/// - It is an error if the split does not result in an equal division.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy dsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html)
/// for details.
List<NDArray<T>> dsplit<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  int sections, {
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank < 3) {
    throw ArgumentError.value(
      a,
      'a',
      'dsplit only supports arrays of rank >= 3',
    );
  }
  return split(a, sections, axis: 2, out: out);
}

/// Splits an array along the 3rd axis (axis 2) at the coordinate points specified in [indices].
///
/// Equivalent to [split_at] with `axis = 2`. Requires rank $\ge 3$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have rank $\ge 3$.
///
/// - It is an error if [a] is disposed.
/// - It is an error if [a] has rank $< 3$.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart lang=dart}
///
/// Refer to the [NumPy dsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html)
/// for details.
List<NDArray<T>> dsplit_at<T extends DTypeTag, Out extends T>(
  NDArray<T> a,
  List<int> indices, {
  List<NDArray<Out>>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank < 3) {
    throw ArgumentError.value(
      a,
      'a',
      'dsplit_at only supports arrays of rank >= 3',
    );
  }
  return split_at(a, indices, axis: 2, out: out);
}
