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
import 'helpers.dart';
import 'sorting.dart';

/// Finds the unique elements of an array.
///
/// Returns the sorted unique elements of an array.
///
/// If [ar] is not 1D, it is flattened first.
///
/// It is an error if [ar] has an unsupported dtype.
///
/// It is an error if [ar] is disposed.
dynamic unique<T extends Object>(
  NDArray<T> ar, {
  bool returnIndex = false,
  bool returnInverse = false,
  bool returnCounts = false,
  NDArray<T>? out,
}) {
  if (ar.isDisposed) {
    throw StateError('Cannot execute unique on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write unique result to a disposed output array.');
  }
  if (out != null && out.dtype != ar.dtype) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  final flat = (ar.rank == 1 && ar.isContiguous) ? ar : ar.flatten();
  NDArray<T>? dest;
  NDArray<int>? outIndex;
  NDArray<int>? outInverse;
  NDArray<int>? outCounts;

  try {
    dest = NDArray<T>.create(flat.shape, flat.dtype);

    if (returnIndex) {
      outIndex = NDArray<int>.create([flat.size], DType.int64);
    }
    if (returnInverse) {
      outInverse = NDArray<int>.create([flat.size], DType.int64);
    }
    if (returnCounts) {
      outCounts = NDArray<int>.create([flat.size], DType.int64);
    }

    final pIndex = outIndex != null
        ? outIndex.pointer.cast<ffi.Int64>()
        : ffi.Pointer<ffi.Int64>.fromAddress(0);
    final pInverse = outInverse != null
        ? outInverse.pointer.cast<ffi.Int64>()
        : ffi.Pointer<ffi.Int64>.fromAddress(0);
    final pCounts = outCounts != null
        ? outCounts.pointer.cast<ffi.Int64>()
        : ffi.Pointer<ffi.Int64>.fromAddress(0);

    final uniqueCount = ndarray_unique(
      flat.pointer.cast(),
      dest.pointer.cast(),
      flat.size,
      encodeDType(flat.dtype),
      pIndex,
      pInverse,
      pCounts,
    );

    if (uniqueCount == 0) {
      final empty = out ?? NDArray<T>.create([0], flat.dtype);
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }

      if (returnIndex || returnInverse || returnCounts) {
        return (
          values: empty,
          index: returnIndex ? NDArray<int>.create([0], DType.int64) : null,
          inverse: returnInverse ? NDArray<int>.create([0], DType.int64) : null,
          counts: returnCounts ? NDArray<int>.create([0], DType.int64) : null,
        );
      }
      return empty;
    }

    if (out != null && !listEquals(out.shape, [uniqueCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final NDArray<T> result;
    if (out != null) {
      custom_memcpy(
        out.pointer.cast(),
        dest.pointer.cast(),
        uniqueCount * ar.dtype.byteWidth,
      );
      result = out;
    } else {
      result = NDArray<T>.create([uniqueCount], flat.dtype);
      custom_memcpy(
        result.pointer.cast(),
        dest.pointer.cast(),
        uniqueCount * ar.dtype.byteWidth,
      );
    }

    NDArray<int>? indexResult;
    if (outIndex != null) {
      indexResult = NDArray<int>.create([uniqueCount], DType.int64);
      custom_memcpy(
        indexResult.pointer.cast(),
        outIndex.pointer.cast(),
        uniqueCount * 8,
      );
    }

    NDArray<int>? inverseResult;
    if (outInverse != null) {
      inverseResult = outInverse.copy();
    }

    NDArray<int>? countsResult;
    if (outCounts != null) {
      countsResult = NDArray<int>.create([uniqueCount], DType.int64);
      custom_memcpy(
        countsResult.pointer.cast(),
        outCounts.pointer.cast(),
        uniqueCount * 8,
      );
    }

    if (returnIndex || returnInverse || returnCounts) {
      return (
        values: result,
        index: indexResult,
        inverse: inverseResult,
        counts: countsResult,
      );
    }

    return result;
  } finally {
    dest?.dispose();
    outIndex?.dispose();
    outInverse?.dispose();
    outCounts?.dispose();
    if (flat != ar) {
      flat.dispose();
    }
  }
}

/// Finds the intersection of two arrays.
///
/// Returns the sorted, unique values that are in both of the input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> intersect1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute intersect1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write intersect1d result to a disposed output array.',
    );
  }
  if (out != null && out.dtype != ar1.dtype) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  NDArray<T>? u1;
  NDArray<T>? u2;
  NDArray<T>? dest;

  try {
    if (assumeUnique) {
      u1 = sort<T>(flat1);
      u2 = sort<T>(flat2);
    } else {
      u1 = unique<T>(flat1) as NDArray<T>;
      u2 = unique<T>(flat2) as NDArray<T>;
    }

    final maxDstSize = u1.size < u2.size ? u1.size : u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    dest = NDArray<T>.create([maxDstSize], ar1.dtype);

    final intersectionCount = ndarray_intersect1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(ar1.dtype),
    );

    if (intersectionCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    if (out != null && !listEquals(out.shape, [intersectionCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final NDArray<T> result;
    if (out != null) {
      custom_memcpy(
        out.pointer.cast(),
        dest.pointer.cast(),
        intersectionCount * ar1.dtype.byteWidth,
      );
      result = out;
    } else {
      result = NDArray<T>.create([intersectionCount], ar1.dtype);
      custom_memcpy(
        result.pointer.cast(),
        dest.pointer.cast(),
        intersectionCount * ar1.dtype.byteWidth,
      );
    }
    return result;
  } finally {
    dest?.dispose();
    u1?.dispose();
    u2?.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
  }
}

/// Finds the set difference of two arrays.
///
/// Returns the unique values in [ar1] that are not in [ar2].
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> setdiff1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setdiff1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write setdiff1d result to a disposed output array.',
    );
  }
  if (out != null && out.dtype != ar1.dtype) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  NDArray<T>? u1;
  NDArray<T>? u2;
  NDArray<T>? dest;

  try {
    if (assumeUnique) {
      u1 = sort<T>(flat1);
      u2 = sort<T>(flat2);
    } else {
      u1 = unique<T>(flat1) as NDArray<T>;
      u2 = unique<T>(flat2) as NDArray<T>;
    }

    final maxDstSize = u1.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    dest = NDArray<T>.create([maxDstSize], ar1.dtype);

    final diffCount = ndarray_setdiff1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(ar1.dtype),
    );

    if (diffCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    if (out != null && !listEquals(out.shape, [diffCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final NDArray<T> result;
    if (out != null) {
      custom_memcpy(
        out.pointer.cast(),
        dest.pointer.cast(),
        diffCount * ar1.dtype.byteWidth,
      );
      result = out;
    } else {
      result = NDArray<T>.create([diffCount], ar1.dtype);
      custom_memcpy(
        result.pointer.cast(),
        dest.pointer.cast(),
        diffCount * ar1.dtype.byteWidth,
      );
    }
    return result;
  } finally {
    dest?.dispose();
    u1?.dispose();
    u2?.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
  }
}

/// Finds the set exclusive-or of two arrays.
///
/// Returns the sorted, unique values that are in only one (not both) of the input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> setxor1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setxor1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write setxor1d result to a disposed output array.',
    );
  }
  if (out != null && out.dtype != ar1.dtype) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  NDArray<T>? u1;
  NDArray<T>? u2;
  NDArray<T>? dest;

  try {
    if (assumeUnique) {
      u1 = sort<T>(flat1);
      u2 = sort<T>(flat2);
    } else {
      u1 = unique<T>(flat1) as NDArray<T>;
      u2 = unique<T>(flat2) as NDArray<T>;
    }

    final maxDstSize = u1.size + u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    dest = NDArray<T>.create([maxDstSize], ar1.dtype);

    final xorCount = ndarray_setxor1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(ar1.dtype),
    );

    if (xorCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    if (out != null && !listEquals(out.shape, [xorCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final NDArray<T> result;
    if (out != null) {
      custom_memcpy(
        out.pointer.cast(),
        dest.pointer.cast(),
        xorCount * ar1.dtype.byteWidth,
      );
      result = out;
    } else {
      result = NDArray<T>.create([xorCount], ar1.dtype);
      custom_memcpy(
        result.pointer.cast(),
        dest.pointer.cast(),
        xorCount * ar1.dtype.byteWidth,
      );
    }
    return result;
  } finally {
    dest?.dispose();
    u1?.dispose();
    u2?.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
  }
}

/// Finds the union of two arrays.
///
/// Returns the unique, sorted array of values that are in either of the two input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> union1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute union1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write union1d result to a disposed output array.');
  }
  if (out != null && out.dtype != ar1.dtype) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  NDArray<T>? u1;
  NDArray<T>? u2;
  NDArray<T>? dest;

  try {
    u1 = unique<T>(flat1) as NDArray<T>;
    u2 = unique<T>(flat2) as NDArray<T>;

    final maxDstSize = u1.size + u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    dest = NDArray<T>.create([maxDstSize], ar1.dtype);

    final unionCount = ndarray_union1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(ar1.dtype),
    );

    if (unionCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ?? NDArray<T>.create([0], ar1.dtype);
    }

    if (out != null && !listEquals(out.shape, [unionCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final NDArray<T> result;
    if (out != null) {
      custom_memcpy(
        out.pointer.cast(),
        dest.pointer.cast(),
        unionCount * ar1.dtype.byteWidth,
      );
      result = out;
    } else {
      result = NDArray<T>.create([unionCount], ar1.dtype);
      custom_memcpy(
        result.pointer.cast(),
        dest.pointer.cast(),
        unionCount * ar1.dtype.byteWidth,
      );
    }
    return result;
  } finally {
    dest?.dispose();
    u1?.dispose();
    u2?.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
  }
}

/// Tests whether each element of an array is also present in a second array.
///
/// Returns a boolean array of the same shape as [element] that is `true` where an element of [element] is in [testElements] and `false` otherwise.
///
/// It is an error if [element] or [testElements] is disposed.
NDArray<bool> isin<T extends Object>(
  NDArray<T> element,
  NDArray<T> testElements, {
  bool assumeUnique = false,
  bool invert = false,
  NDArray<bool>? out,
}) {
  if (element.isDisposed || testElements.isDisposed) {
    throw StateError('Cannot execute isin on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write isin result to a disposed output array.');
  }
  if (out != null) {
    if (!listEquals(out.shape, element.shape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final flatTest = (testElements.rank == 1 && testElements.isContiguous)
      ? testElements
      : testElements.flatten();
  NDArray<T>? uTest;
  NDArray<T>? contigElement;

  try {
    if (assumeUnique) {
      uTest = sort<T>(flatTest);
    } else {
      uTest = unique<T>(flatTest) as NDArray<T>;
    }

    if (element.isContiguous) {
      contigElement = element;
    } else {
      contigElement = element.copy();
    }

    final dest = out ?? NDArray<bool>.create(element.shape, DType.boolean);

    ndarray_isin(
      contigElement.pointer.cast(),
      element.size,
      uTest.pointer.cast(),
      uTest.size,
      dest.pointer.cast(),
      encodeDType(element.dtype),
      invert ? 1 : 0,
    );

    return dest;
  } finally {
    if (contigElement != null && contigElement != element) {
      contigElement.dispose();
    }
    uTest?.dispose();
    if (flatTest != testElements) {
      flatTest.dispose();
    }
  }
}
