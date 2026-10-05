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
import 'dart:typed_data';
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';
import 'helpers.dart';
import 'sorting.dart';

/// Finds the unique elements of an array.
///
/// Returns the sorted unique elements of [ar].
///
/// If [ar] is not 1-D, it is flattened first.
///
/// If [out] is provided, the result is written into it and returned.
///
/// The [ar] must not be disposed.
/// If provided, [out] must not be disposed and must have compatible dtype and shape.
///
/// {@example /example/set_operations_example.dart lang=dart}
NDArray<T> unique<T extends DTypeTag>(NDArray<T> ar, {NDArray<T>? out}) =>
    _uniqueImpl(ar, out: out).values;

/// Finds the unique elements of an array and the indices of their first occurrences.
///
/// Returns a record `(values: ..., index: ...)` containing:
/// - `values`: The sorted unique elements of [ar].
/// - `index`: The indices of the first occurrences of the unique values in the
///   (flattened) original array [ar].
///
/// If [ar] is not 1-D, it is flattened first.
///
/// If [out] is provided, the unique values are written into it.
///
/// The [ar] must not be disposed.
/// If provided, [out] must not be disposed and must have compatible dtype and shape.
///
/// {@example /example/set_operations_example.dart lang=dart}
({NDArray<T> values, NDArray<Int64> index}) uniqueWithIndex<T extends DTypeTag>(
  NDArray<T> ar, {
  NDArray<T>? out,
}) {
  final res = _uniqueImpl(ar, returnIndex: true, out: out);
  return (values: res.values, index: res.index!);
}

/// Finds the unique elements of an array and the indices to reconstruct the original array.
///
/// Returns a record `(values: ..., inverse: ...)` containing:
/// - `values`: The sorted unique elements of [ar].
/// - `inverse`: The indices to reconstruct the (flattened) original array [ar]
///   from the unique values.
///
/// If [ar] is not 1-D, it is flattened first.
///
/// If [out] is provided, the unique values are written into it.
///
/// The [ar] must not be disposed.
/// If provided, [out] must not be disposed and must have compatible dtype and shape.
///
/// {@example /example/set_operations_example.dart lang=dart}
({NDArray<T> values, NDArray<Int64> inverse})
uniqueWithInverse<T extends DTypeTag>(NDArray<T> ar, {NDArray<T>? out}) {
  final res = _uniqueImpl(ar, returnInverse: true, out: out);
  return (values: res.values, inverse: res.inverse!);
}

/// Finds the unique elements of an array and the number of times each element appears.
///
/// Returns a record `(values: ..., counts: ...)` containing:
/// - `values`: The sorted unique elements of [ar].
/// - `counts`: The number of times each of the unique values comes up in [ar].
///
/// If [ar] is not 1-D, it is flattened first.
///
/// If [out] is provided, the unique values are written into it.
///
/// The [ar] must not be disposed.
/// If provided, [out] must not be disposed and must have compatible dtype and shape.
///
/// {@example /example/set_operations_example.dart lang=dart}
({NDArray<T> values, NDArray<Int64> counts})
uniqueWithCounts<T extends DTypeTag>(NDArray<T> ar, {NDArray<T>? out}) {
  final res = _uniqueImpl(ar, returnCounts: true, out: out);
  return (values: res.values, counts: res.counts!);
}

/// Finds the unique elements of an array along with indices, inverse indices, and counts.
///
/// Returns a record `(values: ..., index: ..., inverse: ..., counts: ...)` containing:
/// - `values`: The sorted unique elements of [ar].
/// - `index`: The indices of the first occurrences of the unique values in the
///   (flattened) original array [ar].
/// - `inverse`: The indices to reconstruct the (flattened) original array [ar]
///   from the unique values.
/// - `counts`: The number of times each of the unique values comes up in [ar].
///
/// If [ar] is not 1-D, it is flattened first.
///
/// If [out] is provided, the unique values are written into it.
///
/// The [ar] must not be disposed.
/// If provided, [out] must not be disposed and must have compatible dtype and shape.
///
/// {@example /example/set_operations_example.dart lang=dart}
({
  NDArray<T> values,
  NDArray<Int64> index,
  NDArray<Int64> inverse,
  NDArray<Int64> counts,
})
uniqueAll<T extends DTypeTag>(NDArray<T> ar, {NDArray<T>? out}) {
  final res = _uniqueImpl(
    ar,
    returnIndex: true,
    returnInverse: true,
    returnCounts: true,
    out: out,
  );
  return (
    values: res.values,
    index: res.index!,
    inverse: res.inverse!,
    counts: res.counts!,
  );
}

({
  NDArray<T> values,
  NDArray<Int64>? index,
  NDArray<Int64>? inverse,
  NDArray<Int64>? counts,
})
_uniqueImpl<T extends DTypeTag>(
  NDArray<T> ar, {
  bool returnIndex = false,
  bool returnInverse = false,
  bool returnCounts = false,
  NDArray<T>? out,
}) {
  if (ar.isDisposed) {
    throw StateError('Cannot execute unique on a disposed array.');
  }
  if (out != null) {
    if (out.isDisposed) {
      throw StateError(
        'Cannot write unique result to a disposed output array.',
      );
    }
    validateOutBuffer(out);
    if (out.dtype != ar.dtype) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must have the same dtype as ar (${ar.dtype})',
      );
    }
  }

  return NDArray.scope(() {
    final flat = (ar.rank == 1 && ar.isContiguous) ? ar : ar.flatten();

    if (!returnIndex &&
        !returnInverse &&
        flat.dtype.isInteger &&
        flat.dtype != DType.uint64 &&
        flat.size > 64) {
      final tableRes = _tryUniqueTable<T>(
        flat,
        returnCounts: returnCounts,
        out: out,
      );
      if (tableRes != null) {
        return (
          values: tableRes.values,
          index: null,
          inverse: null,
          counts: tableRes.counts,
        );
      }
    }

    final dest = NDArray<T>.create(flat.shape, flat.dtype);
    final outIndex = returnIndex
        ? NDArray<Int64>.create([flat.size], DType.int64)
        : null;
    final outInverse = returnInverse
        ? NDArray<Int64>.create([flat.size], DType.int64)
        : null;
    final outCounts = returnCounts
        ? NDArray<Int64>.create([flat.size], DType.int64)
        : null;

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
    checkNativeOom();

    if (uniqueCount < 0) {
      throw OutOfMemoryError();
    }

    if (uniqueCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(
          out.shape,
          'out',
          'Must have shape [0] for empty unique result',
        );
      }
      final empty =
          out ?? (NDArray<T>.create([0], flat.dtype)..detachToParentScope());

      return (
        values: empty,
        index: returnIndex
            ? (NDArray<Int64>.create([0], DType.int64)..detachToParentScope())
            : null,
        inverse: returnInverse
            ? (NDArray<Int64>.create([0], DType.int64)..detachToParentScope())
            : null,
        counts: returnCounts
            ? (NDArray<Int64>.create([0], DType.int64)..detachToParentScope())
            : null,
      );
    }

    if (out != null && !listEquals(out.shape, [uniqueCount])) {
      throw ArgumentError.value(
        out.shape,
        'out',
        'Must have shape [$uniqueCount] to match unique result',
      );
    }

    final validView = dest.slice([Slice(start: 0, stop: uniqueCount)]);
    final NDArray<T> result;
    if (out != null) {
      validView.copy(out: out);
      result = out;
    } else {
      result = validView.copy()..detachToParentScope();
    }

    NDArray<Int64>? indexResult;
    if (outIndex != null) {
      indexResult = outIndex.slice([Slice(start: 0, stop: uniqueCount)]).copy()
        ..detachToParentScope();
    }

    NDArray<Int64>? inverseResult;
    if (outInverse != null) {
      inverseResult = outInverse.copy()..detachToParentScope();
    }

    NDArray<Int64>? countsResult;
    if (outCounts != null) {
      countsResult = outCounts.slice([
        Slice(start: 0, stop: uniqueCount),
      ]).copy()..detachToParentScope();
    }

    return (
      values: result,
      index: indexResult,
      inverse: inverseResult,
      counts: countsResult,
    );
  });
}

/// Finds the intersection of two arrays.
///
/// Returns the sorted, unique values that are in both of the input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
///
/// {@example /example/set_operations_example.dart lang=dart}
NDArray<T> intersect1d<T extends DTypeTag>(
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
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != commonDType) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer dtype');
    }
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray<T> u1 = assumeUnique ? sort(flat1) : unique(flat1);
    final NDArray<T> u2 = assumeUnique ? sort(flat2) : unique(flat2);

    final maxDstSize = u1.size < u2.size ? u1.size : u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final intersectionCount = ndarray_intersect1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (intersectionCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [intersectionCount])) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
    }

    final validView = dest.slice([Slice(start: 0, stop: intersectionCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Finds the set difference of two arrays.
///
/// Returns the unique values in [ar1] that are not in [ar2].
///
/// It is an error if [ar1] or [ar2] is disposed.
///
/// {@example /example/set_operations_example.dart lang=dart}
NDArray<T> setdiff1d<T extends DTypeTag>(
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
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != commonDType) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer dtype');
    }
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray<T> u1 = assumeUnique ? sort(flat1) : unique(flat1);
    final NDArray<T> u2 = assumeUnique ? sort(flat2) : unique(flat2);

    final maxDstSize = u1.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final diffCount = ndarray_setdiff1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (diffCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [diffCount])) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
    }

    final validView = dest.slice([Slice(start: 0, stop: diffCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Finds the set exclusive-or of two arrays.
///
/// Returns the sorted, unique values that are in only one (not both) of the input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
///
/// {@example /example/set_operations_example.dart lang=dart}
NDArray<T> setxor1d<T extends DTypeTag>(
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
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != commonDType) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer dtype');
    }
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray<T> u1 = assumeUnique ? sort(flat1) : unique(flat1);
    final NDArray<T> u2 = assumeUnique ? sort(flat2) : unique(flat2);

    final maxDstSize = u1.size + u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final xorCount = ndarray_setxor1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (xorCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [xorCount])) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
    }

    final validView = dest.slice([Slice(start: 0, stop: xorCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Finds the union of two arrays.
///
/// Returns the unique, sorted array of values that are in either of the two input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
///
/// {@example /example/set_operations_example.dart lang=dart}
NDArray<T> union1d<T extends DTypeTag>(
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
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != commonDType) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer dtype');
    }
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray<T> u1 = unique(flat1);
    final NDArray<T> u2 = unique(flat2);

    final maxDstSize = u1.size + u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final unionCount = ndarray_union1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (unionCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [unionCount])) {
      throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
    }

    final validView = dest.slice([Slice(start: 0, stop: unionCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Tests whether each element of an array is also present in a second array.
///
/// Returns a boolean array of the same shape as [element] that is `true` where an element of [element] is in [testElements] and `false` otherwise.
///
/// It is an error if [element] or [testElements] is disposed.
///
/// {@example /example/set_operations_example.dart lang=dart}
NDArray<Boolean> isin<T extends DTypeTag>(
  NDArray<T> element,
  NDArray<T> testElements, {
  bool assumeUnique = false,
  bool invert = false,
  NDArray<Boolean>? out,
}) {
  if (element.isDisposed || testElements.isDisposed) {
    throw StateError('Cannot execute isin on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write isin result to a disposed output array.');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, element.shape) || out.dtype != DType.boolean) {
      throw ArgumentError.value(
        out,
        'out',
        'Incompatible out buffer shape or dtype',
      );
    }
  }

  final DType<T> commonDType =
      (element.dtype == testElements.dtype
              ? element.dtype
              : resolveDType(element.dtype, testElements.dtype))
          as DType<T>;

  final bool useTempOut =
      out != null &&
      (!out.isContiguous ||
          sharesMemory(element, out) ||
          sharesMemory(testElements, out));

  return NDArray.scope(() {
    final NDArray<T> cElement = element.dtype == commonDType
        ? element
        : castNDArray<T>(element, commonDType);
    final NDArray<T> cTest = testElements.dtype == commonDType
        ? testElements
        : castNDArray<T>(testElements, commonDType);

    final dest = (out != null && !useTempOut)
        ? out
        : NDArray<Boolean>.create(element.shape, DType.boolean);

    if (element.size == 0) {
      // Empty input array, result is empty boolean array.
    } else if (testElements.size == 0) {
      dest.fill(invert);
    } else {
      final NDArray<T> contigElement = cElement.isContiguous
          ? cElement
          : cElement.copy();
      final NDArray<T> flatTest = (cTest.rank == 1 && cTest.isContiguous)
          ? cTest
          : cTest.flatten();

      if (!_tryIsinTable<T>(
        contigElement,
        flatTest,
        dest,
        commonDType,
        invert,
      )) {
        final NDArray<T> uTest = assumeUnique
            ? sort(flatTest)
            : unique(flatTest);

        ndarray_isin(
          contigElement.pointer.cast(),
          element.size,
          uTest.pointer.cast(),
          uTest.size,
          dest.pointer.cast(),
          encodeDType(commonDType),
          invert ? 1 : 0,
        );
      }
    }

    if (useTempOut) {
      dest.copy(out: out);
      return out;
    }
    if (out == null) {
      dest.detachToParentScope();
    }
    return dest;
  });
}

bool _tryIsinTable<T extends DTypeTag>(
  NDArray<T> contigElement,
  NDArray<T> flatTest,
  NDArray<Boolean> dest,
  DType<T> dtype,
  bool invert,
) {
  final elemSize = contigElement.size;
  final testSize = flatTest.size;
  const maxTableRange = 10000000;

  switch (dtype) {
    case DType.int32:
      final pTest = flatTest.pointer.cast<ffi.Int32>();
      final minVal = r_min_int32_t(pTest, testSize);
      final maxVal = r_max_int32_t(pTest, testSize);
      if (maxVal < minVal) return false;
      final range = maxVal - minVal + 1;
      if (range > maxTableRange || range <= 0) return false;
      final table = Uint8List(range);
      for (var i = 0; i < testSize; i++) {
        table[pTest[i] - minVal] = 1;
      }
      final pElem = contigElement.pointer.cast<ffi.Int32>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (invert) {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 0
              : 1;
        }
      } else {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 1
              : 0;
        }
      }
      return true;

    case DType.int64:
      final pTest = flatTest.pointer.cast<ffi.Int64>();
      final minVal = r_min_int64_t(pTest, testSize);
      final maxVal = r_max_int64_t(pTest, testSize);
      if (maxVal < minVal) return false;
      final diff = maxVal - minVal;
      if (diff < 0 || diff >= maxTableRange) return false;
      final range = diff + 1;
      final table = Uint8List(range);
      for (var i = 0; i < testSize; i++) {
        table[pTest[i] - minVal] = 1;
      }
      final pElem = contigElement.pointer.cast<ffi.Int64>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (invert) {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 0
              : 1;
        }
      } else {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 1
              : 0;
        }
      }
      return true;

    case DType.int16:
      final pTest = flatTest.pointer.cast<ffi.Int16>();
      final minVal = r_min_int16_t(pTest, testSize);
      final maxVal = r_max_int16_t(pTest, testSize);
      if (maxVal < minVal) return false;
      final range = maxVal - minVal + 1;
      if (range > 65536 || range <= 0) return false;
      final table = Uint8List(range);
      for (var i = 0; i < testSize; i++) {
        table[pTest[i] - minVal] = 1;
      }
      final pElem = contigElement.pointer.cast<ffi.Int16>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (invert) {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 0
              : 1;
        }
      } else {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 1
              : 0;
        }
      }
      return true;

    case DType.int8:
      final pTest = flatTest.pointer.cast<ffi.Int8>();
      var minVal = pTest[0];
      var maxVal = pTest[0];
      for (var i = 1; i < testSize; i++) {
        final v = pTest[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      final range = maxVal - minVal + 1;
      final table = Uint8List(range);
      for (var i = 0; i < testSize; i++) {
        table[pTest[i] - minVal] = 1;
      }
      final pElem = contigElement.pointer.cast<ffi.Int8>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (invert) {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 0
              : 1;
        }
      } else {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 1
              : 0;
        }
      }
      return true;

    case DType.uint8:
      final pTest = flatTest.pointer.cast<ffi.Uint8>();
      final minVal = r_min_uint8_t(pTest, testSize);
      final maxVal = r_max_uint8_t(pTest, testSize);
      final range = maxVal - minVal + 1;
      final table = Uint8List(range);
      for (var i = 0; i < testSize; i++) {
        table[pTest[i] - minVal] = 1;
      }
      final pElem = contigElement.pointer.cast<ffi.Uint8>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (invert) {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 0
              : 1;
        }
      } else {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 1
              : 0;
        }
      }
      return true;

    case DType.uint16:
      final pTest = flatTest.pointer.cast<ffi.Uint16>();
      var minVal = pTest[0];
      var maxVal = pTest[0];
      for (var i = 1; i < testSize; i++) {
        final v = pTest[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      final range = maxVal - minVal + 1;
      final table = Uint8List(range);
      for (var i = 0; i < testSize; i++) {
        table[pTest[i] - minVal] = 1;
      }
      final pElem = contigElement.pointer.cast<ffi.Uint16>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (invert) {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 0
              : 1;
        }
      } else {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 1
              : 0;
        }
      }
      return true;

    case DType.uint32:
      final pTest = flatTest.pointer.cast<ffi.Uint32>();
      var minVal = pTest[0];
      var maxVal = pTest[0];
      for (var i = 1; i < testSize; i++) {
        final v = pTest[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      if (maxVal < minVal) return false;
      final range = maxVal - minVal + 1;
      if (range > maxTableRange || range <= 0) return false;
      final table = Uint8List(range);
      for (var i = 0; i < testSize; i++) {
        table[pTest[i] - minVal] = 1;
      }
      final pElem = contigElement.pointer.cast<ffi.Uint32>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (invert) {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 0
              : 1;
        }
      } else {
        for (var i = 0; i < elemSize; i++) {
          final v = pElem[i];
          pDest[i] = (v >= minVal && v <= maxVal && table[v - minVal] == 1)
              ? 1
              : 0;
        }
      }
      return true;

    case DType.boolean:
      final pTest = flatTest.pointer.cast<ffi.Uint8>();
      var hasZero = false;
      var hasOne = false;
      for (var i = 0; i < testSize; i++) {
        if (pTest[i] == 0) hasZero = true;
        if (pTest[i] != 0) hasOne = true;
        if (hasZero && hasOne) break;
      }
      final pElem = contigElement.pointer.cast<ffi.Uint8>();
      final pDest = dest.pointer.cast<ffi.Uint8>();
      if (hasZero && hasOne) {
        final fillVal = invert ? 0 : 1;
        for (var i = 0; i < elemSize; i++) {
          pDest[i] = fillVal;
        }
      } else if (hasOne) {
        if (invert) {
          for (var i = 0; i < elemSize; i++) {
            pDest[i] = pElem[i] != 0 ? 0 : 1;
          }
        } else {
          for (var i = 0; i < elemSize; i++) {
            pDest[i] = pElem[i] != 0 ? 1 : 0;
          }
        }
      } else if (hasZero) {
        if (invert) {
          for (var i = 0; i < elemSize; i++) {
            pDest[i] = pElem[i] == 0 ? 0 : 1;
          }
        } else {
          for (var i = 0; i < elemSize; i++) {
            pDest[i] = pElem[i] == 0 ? 1 : 0;
          }
        }
      } else {
        final fillVal = invert ? 1 : 0;
        for (var i = 0; i < elemSize; i++) {
          pDest[i] = fillVal;
        }
      }
      return true;

    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.uint64:
    case DType.complex128:
    case DType.complex64:
      return false;
  }
}

(int, int)? _minMaxInt<T extends DTypeTag>(NDArray<T> values) {
  final size = values.size;
  if (size == 0) return null;
  final ptr = values.pointer;
  switch (values.dtype) {
    case DType.int32:
      final p = ptr.cast<ffi.Int32>();
      var minVal = p[0];
      var maxVal = minVal;
      for (var i = 1; i < size; i++) {
        final v = p[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      return (minVal, maxVal);
    case DType.int64:
      final p = ptr.cast<ffi.Int64>();
      var minVal = p[0];
      var maxVal = minVal;
      for (var i = 1; i < size; i++) {
        final v = p[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      return (minVal, maxVal);
    case DType.int16:
      final p = ptr.cast<ffi.Int16>();
      var minVal = p[0];
      var maxVal = minVal;
      for (var i = 1; i < size; i++) {
        final v = p[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      return (minVal, maxVal);
    case DType.int8:
      final p = ptr.cast<ffi.Int8>();
      var minVal = p[0];
      var maxVal = minVal;
      for (var i = 1; i < size; i++) {
        final v = p[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      return (minVal, maxVal);
    case DType.uint32:
      final p = ptr.cast<ffi.Uint32>();
      var minVal = p[0];
      var maxVal = minVal;
      for (var i = 1; i < size; i++) {
        final v = p[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      return (minVal, maxVal);
    case DType.uint16:
      final p = ptr.cast<ffi.Uint16>();
      var minVal = p[0];
      var maxVal = minVal;
      for (var i = 1; i < size; i++) {
        final v = p[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      return (minVal, maxVal);
    case DType.uint8:
      final p = ptr.cast<ffi.Uint8>();
      var minVal = p[0];
      var maxVal = minVal;
      for (var i = 1; i < size; i++) {
        final v = p[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      return (minVal, maxVal);
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.uint64:
    case DType.boolean:
    case DType.complex128:
    case DType.complex64:
      return null;
  }
}

({NDArray<T> values, NDArray<Int64>? counts})? _tryUniqueTable<
  T extends DTypeTag
>(NDArray<T> values, {required bool returnCounts, NDArray<T>? out}) {
  final mm = _minMaxInt(values);
  if (mm == null) return null;
  final (minVal, maxVal) = mm;
  if (maxVal < minVal) return null;
  final span = maxVal - minVal;
  const maxSpan = 16777216;
  if (span < 0 || span > maxSpan) return null;
  final maxAllowedSpan = values.size * 16 > 262144 ? values.size * 16 : 262144;
  if (span > maxAllowedSpan) return null;

  final size = values.size;
  final tableSize = span + 1;
  final ptr = values.pointer;
  final marker = ScratchArena.marker;

  try {
    if (!returnCounts) {
      final tablePtr = ScratchArena.allocate<ffi.Uint8>(tableSize);
      for (var i = 0; i < tableSize; i++) {
        tablePtr[i] = 0;
      }

      var uniqueCount = 0;
      switch (values.dtype) {
        case DType.int32:
          final p = ptr.cast<ffi.Int32>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              tablePtr[idx] = 1;
              uniqueCount++;
            }
          }
        case DType.int64:
          final p = ptr.cast<ffi.Int64>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              tablePtr[idx] = 1;
              uniqueCount++;
            }
          }
        case DType.int16:
          final p = ptr.cast<ffi.Int16>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              tablePtr[idx] = 1;
              uniqueCount++;
            }
          }
        case DType.int8:
          final p = ptr.cast<ffi.Int8>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              tablePtr[idx] = 1;
              uniqueCount++;
            }
          }
        case DType.uint32:
          final p = ptr.cast<ffi.Uint32>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              tablePtr[idx] = 1;
              uniqueCount++;
            }
          }
        case DType.uint16:
          final p = ptr.cast<ffi.Uint16>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              tablePtr[idx] = 1;
              uniqueCount++;
            }
          }
        case DType.uint8:
          final p = ptr.cast<ffi.Uint8>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              tablePtr[idx] = 1;
              uniqueCount++;
            }
          }
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.uint64:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          return null;
      }

      if (out != null && !listEquals(out.shape, [uniqueCount])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }

      final bool useTempOut =
          out != null && (!out.isContiguous || sharesMemory(values, out));
      final NDArray<T> res = (out != null && !useTempOut)
          ? out
          : NDArray<T>.create([uniqueCount], values.dtype);

      final resPtr = res.pointer;
      var outIdx = 0;
      switch (values.dtype) {
        case DType.int32:
          final pRes = resPtr.cast<ffi.Int32>();
          for (var idx = 0; idx <= span; idx++) {
            if (tablePtr[idx] != 0) {
              pRes[outIdx++] = minVal + idx;
            }
          }
        case DType.int64:
          final pRes = resPtr.cast<ffi.Int64>();
          for (var idx = 0; idx <= span; idx++) {
            if (tablePtr[idx] != 0) {
              pRes[outIdx++] = minVal + idx;
            }
          }
        case DType.int16:
          final pRes = resPtr.cast<ffi.Int16>();
          for (var idx = 0; idx <= span; idx++) {
            if (tablePtr[idx] != 0) {
              pRes[outIdx++] = minVal + idx;
            }
          }
        case DType.int8:
          final pRes = resPtr.cast<ffi.Int8>();
          for (var idx = 0; idx <= span; idx++) {
            if (tablePtr[idx] != 0) {
              pRes[outIdx++] = minVal + idx;
            }
          }
        case DType.uint32:
          final pRes = resPtr.cast<ffi.Uint32>();
          for (var idx = 0; idx <= span; idx++) {
            if (tablePtr[idx] != 0) {
              pRes[outIdx++] = minVal + idx;
            }
          }
        case DType.uint16:
          final pRes = resPtr.cast<ffi.Uint16>();
          for (var idx = 0; idx <= span; idx++) {
            if (tablePtr[idx] != 0) {
              pRes[outIdx++] = minVal + idx;
            }
          }
        case DType.uint8:
          final pRes = resPtr.cast<ffi.Uint8>();
          for (var idx = 0; idx <= span; idx++) {
            if (tablePtr[idx] != 0) {
              pRes[outIdx++] = minVal + idx;
            }
          }
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.uint64:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          return null;
      }

      if (useTempOut) {
        res.copy(out: out);
        return (values: out, counts: null);
      }
      if (out == null) {
        res.detachToParentScope();
      }
      return (values: res, counts: null);
    } else {
      final tableBytes = tableSize * 4;
      final tablePtr = ScratchArena.allocate<ffi.Int32>(tableBytes);
      for (var i = 0; i < tableSize; i++) {
        tablePtr[i] = 0;
      }

      var uniqueCount = 0;
      switch (values.dtype) {
        case DType.int32:
          final p = ptr.cast<ffi.Int32>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              uniqueCount++;
            }
            tablePtr[idx]++;
          }
        case DType.int64:
          final p = ptr.cast<ffi.Int64>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              uniqueCount++;
            }
            tablePtr[idx]++;
          }
        case DType.int16:
          final p = ptr.cast<ffi.Int16>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              uniqueCount++;
            }
            tablePtr[idx]++;
          }
        case DType.int8:
          final p = ptr.cast<ffi.Int8>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              uniqueCount++;
            }
            tablePtr[idx]++;
          }
        case DType.uint32:
          final p = ptr.cast<ffi.Uint32>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              uniqueCount++;
            }
            tablePtr[idx]++;
          }
        case DType.uint16:
          final p = ptr.cast<ffi.Uint16>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              uniqueCount++;
            }
            tablePtr[idx]++;
          }
        case DType.uint8:
          final p = ptr.cast<ffi.Uint8>();
          for (var i = 0; i < size; i++) {
            final idx = p[i] - minVal;
            if (tablePtr[idx] == 0) {
              uniqueCount++;
            }
            tablePtr[idx]++;
          }
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.uint64:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          return null;
      }

      if (out != null && !listEquals(out.shape, [uniqueCount])) {
        throw ArgumentError.value(out, 'out', 'Incompatible out buffer shape');
      }

      final bool useTempOut =
          out != null && (!out.isContiguous || sharesMemory(values, out));
      final NDArray<T> res = (out != null && !useTempOut)
          ? out
          : NDArray<T>.create([uniqueCount], values.dtype);
      final counts = NDArray<Int64>.create([uniqueCount], DType.int64);

      final resPtr = res.pointer;
      final pCounts = counts.pointer.cast<ffi.Int64>();
      var outIdx = 0;
      switch (values.dtype) {
        case DType.int32:
          final pRes = resPtr.cast<ffi.Int32>();
          for (var idx = 0; idx <= span; idx++) {
            final c = tablePtr[idx];
            if (c != 0) {
              pRes[outIdx] = minVal + idx;
              pCounts[outIdx] = c;
              outIdx++;
            }
          }
        case DType.int64:
          final pRes = resPtr.cast<ffi.Int64>();
          for (var idx = 0; idx <= span; idx++) {
            final c = tablePtr[idx];
            if (c != 0) {
              pRes[outIdx] = minVal + idx;
              pCounts[outIdx] = c;
              outIdx++;
            }
          }
        case DType.int16:
          final pRes = resPtr.cast<ffi.Int16>();
          for (var idx = 0; idx <= span; idx++) {
            final c = tablePtr[idx];
            if (c != 0) {
              pRes[outIdx] = minVal + idx;
              pCounts[outIdx] = c;
              outIdx++;
            }
          }
        case DType.int8:
          final pRes = resPtr.cast<ffi.Int8>();
          for (var idx = 0; idx <= span; idx++) {
            final c = tablePtr[idx];
            if (c != 0) {
              pRes[outIdx] = minVal + idx;
              pCounts[outIdx] = c;
              outIdx++;
            }
          }
        case DType.uint32:
          final pRes = resPtr.cast<ffi.Uint32>();
          for (var idx = 0; idx <= span; idx++) {
            final c = tablePtr[idx];
            if (c != 0) {
              pRes[outIdx] = minVal + idx;
              pCounts[outIdx] = c;
              outIdx++;
            }
          }
        case DType.uint16:
          final pRes = resPtr.cast<ffi.Uint16>();
          for (var idx = 0; idx <= span; idx++) {
            final c = tablePtr[idx];
            if (c != 0) {
              pRes[outIdx] = minVal + idx;
              pCounts[outIdx] = c;
              outIdx++;
            }
          }
        case DType.uint8:
          final pRes = resPtr.cast<ffi.Uint8>();
          for (var idx = 0; idx <= span; idx++) {
            final c = tablePtr[idx];
            if (c != 0) {
              pRes[outIdx] = minVal + idx;
              pCounts[outIdx] = c;
              outIdx++;
            }
          }
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.uint64:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          return null;
      }

      if (useTempOut) {
        res.copy(out: out);
      } else if (out == null) {
        res.detachToParentScope();
      }
      counts.detachToParentScope();
      return (values: out ?? res, counts: counts);
    }
  } finally {
    ScratchArena.reset(marker);
  }
}
