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
import 'dart:math' as math;

import 'package:openblas/openblas.dart';

import '../../ndarray.dart';

import 'dart:ffi' as ffi;

import '../../nditer.dart';
import '../helpers.dart';

/// Configure the number of parallel execution threads used by OpenBLAS at runtime.
///
/// **Preconditions:**
/// - [numThreads] must be greater than or equal to 1.
///
/// It is an error if [numThreads] is less than 1 (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// setNumThreads(1); // Disable multi-threading to bypass overhead on small matrices
/// ```
void setNumThreads(int numThreads) {
  if (numThreads < 1) {
    throw ArgumentError(
      'Number of threads must be at least 1 (was $numThreads)',
    );
  }
  openblas_set_num_threads(numThreads);
}

/// Enumerates elements of a multidimensional array yielding coordinates and values.
///
/// Yields records containing the coordinate list and the element value at that coordinate
/// in standard C-contiguous order.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
///
/// It is an error if [a] has been disposed (throws [StateError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
/// for (final entry in ndenumerate(a)) {
///   print('coord: ${entry.$1}, value: ${entry.$2}');
/// }
/// // Yields:
/// // ([0, 0], 10)
/// // ([0, 1], 20)
/// // ([1, 0], 30)
/// // ([1, 1], 40)
/// ```
Iterable<(List<int> coordinate, T value)> ndenumerate<T>(NDArray<T> a) sync* {
  if (a.isDisposed) {
    throw StateError('Cannot execute ndenumerate() on a disposed array.');
  }

  final shape = a.shape;
  final strides = a.strides;
  final totalSize = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);

  if (shape.isEmpty) {
    yield ([], a.getCellFlat(a.offsetElements));
    return;
  }

  final coord = List<int>.filled(shape.length, 0);
  int offset = a.offsetElements;

  for (int el = 0; el < totalSize; el++) {
    // Yield a copy of the coordinate list so that users don't receive the same mutated buffer!
    yield (List<int>.from(coord), a.getCellRaw(offset));

    // Advance odometer multidimensional coordinate odometer walk!
    for (int d = shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < shape[d]) {
        offset += strides[d];
        break;
      }
      coord[d] = 0;
      offset -= (shape[d] - 1) * strides[d];
    }
  }
}

/// Replace NaN with zero and infinity with large finite numbers.
///
/// By default, maps NaN to [nan] (which defaults to 0.0), maps positive infinity
/// to [posinf] (or the maximum finite float value if null), and maps negative infinity
/// to [neginf] (or the minimum finite float value if null).
///
/// **Preconditions:**
/// - Input [a] must be a numeric array.
///
/// It is an error if the provided [out] buffer has an incompatible shape (throws [ArgumentError]).
///
/// **Example:**
/// {@example /example/nan_to_num_example.dart lang=dart}
///
/// Reference: [Replace NaN and Infinities](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
NDArray nan_to_num(
  NDArray a, {
  double nan = 0.0,
  double? posinf,
  double? neginf,
  NDArray<dynamic>? where,
  NDArray? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute nan_to_num() on a disposed array.');
  }
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for nan_to_num.',
      );
    }
  }

  final resultCopy = out ?? NDArray.create(a.shape, a.dtype);

  final maxLimit = a.dtype == DType.float32
      ? 3.4028234663852886e+38
      : double.maxFinite;
  final minLimit = -maxLimit;

  final targetPosInf = posinf ?? maxLimit;
  final targetNegInf = neginf ?? minLimit;

  final maskHolder = prepareMask(where, resultCopy.shape);
  try {
    final iter = NDIter.broadcast2(resultCopy, a);
    final maskPtr = maskHolder.pointer;
    var flatIdx = 0;
    while (iter.moveNext()) {
      final idxRes = iter.getIndex(0);
      final idxA = iter.getIndex(1);
      if (maskPtr == ffi.nullptr || maskPtr[flatIdx] != 0) {
        final val = a.getCellRaw(idxA);

        if (val is Complex) {
          var r = val.real;
          var img = val.imag;

          if (r.isNaN) r = nan;
          if (r == double.infinity) r = targetPosInf;
          if (r == double.negativeInfinity) r = targetNegInf;

          if (img.isNaN) img = nan;
          if (img == double.infinity) img = targetPosInf;
          if (img == double.negativeInfinity) img = targetNegInf;

          resultCopy.setCellRaw(idxRes, Complex(r, img));
        } else {
          var dVal = (val as num).toDouble();

          if (dVal.isNaN) {
            dVal = nan;
          } else if (dVal == double.infinity) {
            dVal = targetPosInf;
          } else if (dVal == double.negativeInfinity) {
            dVal = targetNegInf;
          }

          resultCopy.setCellRaw(idxRes, castValue(dVal, resultCopy.dtype));
        }
      } else if (out == null) {
        resultCopy.setCellRaw(idxRes, a.getCellRaw(idxA));
      }
      flatIdx++;
    }

    return resultCopy;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the common broadcasted shape resulting from broadcasting shapes [s1] and [s2].
///
/// Follows standard NumPy broadcasting rules: shapes are aligned from the trailing
/// dimension backwards. For each dimension, the dimensions are compatible if they are
/// equal, or if one of them is 1.
///
/// **Preconditions:**
/// - It is an error if [s1] and [s2] have incompatible dimensions for broadcasting.
///
/// It is an error if [s1] and [s2] cannot be broadcast together (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final common = broadcastShapes([2, 1, 4], [3, 4]); // [2, 3, 4]
/// ```
///
/// Reference: [NumPy broadcast_shapes](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html)
List<int> broadcastShapes(List<int> s1, List<int> s2) {
  final len = math.max(s1.length, s2.length);
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;

    final int target;
    if (dim1 == dim2) {
      target = dim1;
    } else if (dim1 == 1) {
      target = dim2;
    } else if (dim2 == 1) {
      target = dim1;
    } else {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}
