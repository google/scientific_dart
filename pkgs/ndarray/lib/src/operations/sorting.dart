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
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';
import '../float16_utils.dart';

// Standalone operational relative cross-imports
import 'spacers.dart';
import 'helpers.dart';

NDArray<DTypeTag> _createNDArrayTyped(List<int> shape, DType<DTypeTag> dtype) {
  switch (dtype) {
    case DType.float64:
      return NDArray<Float64>.create(shape, DType.float64);
    case DType.float32:
      return NDArray<Float32>.create(shape, DType.float32);
    case DType.float16:
      return NDArray<Float16>.create(shape, DType.float16);
    case DType.bfloat16:
      return NDArray<BFloat16>.create(shape, DType.bfloat16);
    case DType.int64:
      return NDArray<Int64>.create(shape, DType.int64);
    case DType.int32:
      return NDArray<Int32>.create(shape, DType.int32);
    case DType.int16:
      return NDArray<Int16>.create(shape, DType.int16);
    case DType.int8:
      return NDArray<Int8>.create(shape, DType.int8);
    case DType.uint64:
      return NDArray<Uint64>.create(shape, DType.uint64);
    case DType.uint32:
      return NDArray<Uint32>.create(shape, DType.uint32);
    case DType.uint16:
      return NDArray<Uint16>.create(shape, DType.uint16);
    case DType.uint8:
      return NDArray<Uint8>.create(shape, DType.uint8);
    case DType.complex128:
      return NDArray<Complex128>.create(shape, DType.complex128);
    case DType.complex64:
      return NDArray<Complex64>.create(shape, DType.complex64);
    case DType.boolean:
      return NDArray<Boolean>.create(shape, DType.boolean);
  }
}

/// Returns a sorted copy of an array along a specified [axis].
///
/// This function corresponds to NumPy's `sort` function.
///
/// It uses native ANSI C `qsort` to perform in-place sorting
/// on the C heap for contiguous last-axis rows.
///
/// Complex numbers are sorted lexicographically: by their real parts first,
/// and by their imaginary parts if the real parts are equal.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
/// - It is an error if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<T> sort<T extends DTypeTag>(
  NDArray<T> a, {
  int axis = -1,
  SortKind kind = SortKind.quicksort,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot sort a disposed array.');
  }
  if (out != null) {
    validateOutArray(out, a.shape, a.dtype, name: 'out');
  }
  if (a.size == 0) {
    return out ?? NDArray<T>.create(a.shape, a.dtype);
  }
  final rank = a.shape.length;
  if (rank == 0) {
    if (out != null) {
      out.setCellFlat(0, a.scalar);
      return out;
    }
    return NDArray<T>.scalar(a.scalar, dtype: a.dtype);
  }

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

  if (targetAxis != rank - 1) {
    return NDArray.scope(() {
      final swappedView = a.swapaxes(targetAxis, rank - 1);
      final sortedView = sort(swappedView, axis: rank - 1, kind: kind);
      final resultSwapped = sortedView.swapaxes(targetAxis, rank - 1);
      if (out != null) {
        resultSwapped.copy(out: out);
        return out;
      } else {
        final result = resultSwapped.copy();
        return result.detachToParentScope();
      }
    });
  }
  NDArray<T> src = a;
  final bool needsDisposeSrc = !a.isContiguous;
  if (needsDisposeSrc) {
    src = a.copy();
  }

  NDArray<T>? tempResult;
  try {
    final NDArray<T> result;
    if (out != null && !out.isContiguous) {
      tempResult = src.copy();
      result = tempResult;
    } else if (out != null) {
      if (!identical(out, src)) {
        src.copy(out: out);
      }
      result = out;
    } else {
      result = src.copy();
    }

    NDArray<T> finish() {
      if (tempResult != null && out != null) {
        tempResult.copy(out: out);
        return out;
      }
      return result;
    }

    final n = src.shape.last;
    final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
    final numRows = totalSize ~/ n;

    if (src.dtype == DType.boolean) {
      // Sort boolean rows in $O(N)$
      for (var r = 0; r < numRows; r++) {
        final rowStart = r * n;
        var falses = 0;
        for (var i = 0; i < n; i++) {
          if (!(result.getCellFlat(rowStart + i) as bool)) falses++;
        }
        for (var i = 0; i < falses; i++) {
          result.setCellFlat(rowStart + i, false);
        }
        for (var i = falses; i < n; i++) {
          result.setCellFlat(rowStart + i, true);
        }
      }
      return finish();
    }

    final elementSizeInBytes = src.dtype.byteWidth;

    final baseCast = result.pointer.cast<ffi.Uint8>();
    final rowSizeInBytes = n * elementSizeInBytes;
    final nativeKind = mapSortKind(kind);

    for (var r = 0; r < numRows; r++) {
      final rowPtr = baseCast + (r * rowSizeInBytes);

      // High-speed direct C sorters bypassing FFI context switches
      switch (src.dtype) {
        case DType.float64:
          native_sort_double(rowPtr.cast<ffi.Double>(), n, nativeKind);
        case DType.float32:
          native_sort_float(rowPtr.cast<ffi.Float>(), n, nativeKind);
        case DType.int64:
          native_sort_int64(rowPtr.cast<ffi.LongLong>(), n, nativeKind);
        case DType.int32:
          native_sort_int32(rowPtr.cast<ffi.Int>(), n, nativeKind);
        case DType.int16:
          native_sort_int16(rowPtr.cast<ffi.Int16>(), n, nativeKind);
        case DType.uint8:
          native_sort_uint8(rowPtr.cast<ffi.Uint8>(), n, nativeKind);
        case DType.complex128:
          native_sort_complex128(rowPtr.cast<ffi.Double>(), n, nativeKind);
        case DType.complex64:
          native_sort_complex64(rowPtr.cast<ffi.Float>(), n, nativeKind);
        case DType.boolean:
          break; // Handled above in O(N)
        case DType.int8:
          native_sort_int8(rowPtr.cast<ffi.Int8>(), n, nativeKind);
        case DType.uint16:
          native_sort_uint16(rowPtr.cast<ffi.Uint16>(), n, nativeKind);
        case DType.uint32:
          native_sort_uint32(rowPtr.cast<ffi.Uint32>(), n, nativeKind);
        case DType.uint64:
          native_sort_uint64(rowPtr.cast<ffi.Uint64>(), n, nativeKind);
        case DType.float16:
          native_sort_float16(rowPtr.cast<ffi.Uint16>(), n, nativeKind);
        case DType.bfloat16:
          native_sort_bfloat16(rowPtr.cast<ffi.Uint16>(), n, nativeKind);
      }
    }

    checkNativeOom();
    return finish();
  } finally {
    tempResult?.dispose();
    if (needsDisposeSrc) {
      src.dispose();
    }
  }
}

/// Returns the indices that would sort an array along a specified [axis].
///
/// This function corresponds to NumPy's `argsort` function.
///
/// It performs indirect index sorting, fully supporting complex numbers
/// lexicographical logic and internal axes reorientation via the axis-swapping pipeline.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
/// - It is an error if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<Int64> argsort<T extends DTypeTag>(
  NDArray<T> a, {
  int axis = -1,
  SortKind kind = SortKind.quicksort,
  NDArray<Int64>? out,
}) => argsortAs<T, Int64>(a, DType.int64, axis: axis, kind: kind, out: out);

/// Returns the indices that would sort an array [a], stored in the specified
/// integer [dtype] (`DType.int32` or `DType.int64`).
///
/// Refer to [argsort] for full details.
NDArray<R> argsortAs<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  DType<R> dtype, {
  int axis = -1,
  SortKind kind = SortKind.quicksort,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute argsort() on a disposed array.');
  }
  if (dtype != DType.int32 && dtype != DType.int64) {
    throw ArgumentError(
      'dtype must be DType.int32 or DType.int64, got $dtype.',
    );
  }
  if (out != null) {
    validateOutArray(out, a.shape, dtype, name: 'out');
  }
  final rank = a.shape.length;
  if (rank == 0) {
    if (out != null) {
      out.setCellFlat(0, 0);
      return out;
    }
    return NDArray<R>.scalar(0, dtype: dtype);
  }
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
  if (a.size == 0) {
    return out ?? NDArray<R>.create(a.shape, dtype);
  }

  if (targetAxis != rank - 1) {
    return NDArray.scope(() {
      final swappedView = a.swapaxes(targetAxis, rank - 1);
      final sortedIndicesView = argsortAs<T, R>(
        swappedView,
        dtype,
        axis: rank - 1,
        kind: kind,
      );
      final resultSwapped = sortedIndicesView.swapaxes(targetAxis, rank - 1);
      if (out != null) {
        resultSwapped.copy(out: out);
        return out;
      }
      final res = resultSwapped.copy();
      res.detachToParentScope();
      return res;
    });
  }

  return NDArray.scope(() {
    final NDArray<T> src = a.isContiguous ? a : a.copy();

    final bool needsTempOut =
        out != null && (!out.isContiguous || sharesMemory(a, out));
    final NDArray<R>? tempResult = needsTempOut
        ? NDArray<R>.create(src.shape, out.dtype)
        : null;
    final NDArray<R> result =
        tempResult ?? (out ?? NDArray<R>.create(src.shape, dtype));

    NDArray<R> finish() {
      if (tempResult != null && out != null) {
        tempResult.copy(out: out);
        return out;
      }
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    final n = src.shape.last;
    final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
    final numRows = totalSize ~/ n;

    final nativeKind = mapSortKind(kind);

    final is32 = result.dtype == DType.int32;
    final ScratchMarker? marker = is32 ? ScratchArena.marker : null;
    try {
      final ffi.Pointer<ffi.Int64> resPtr = is32
          ? ScratchArena.allocate<ffi.Int64>(
              totalSize * ffi.sizeOf<ffi.Int64>(),
            )
          : result.pointer.cast<ffi.Int64>();

      void copyIf32() {
        if (is32) {
          final outPtr = result.pointer.cast<ffi.Int32>();
          for (var i = 0; i < totalSize; i++) {
            outPtr[i] = resPtr[i];
          }
        }
      }

      switch (src.dtype) {
        case DType.float64:
          final dataPtr = src.pointer.cast<ffi.Double>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_double(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.float32:
          final dataPtr = src.pointer.cast<ffi.Float>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_float(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.int64:
          final dataPtr = src.pointer.cast<ffi.LongLong>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_int64(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.int32:
          final dataPtr = src.pointer.cast<ffi.Int>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_int32(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.int16:
          final dataPtr = src.pointer.cast<ffi.Int16>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_int16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.uint8:
          final dataPtr = src.pointer.cast<ffi.Uint8>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_uint8(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.complex128:
        case DType.complex64:
          for (var r = 0; r < numRows; r++) {
            final rowStart = r * n;
            final indices = List<int>.generate(n, (i) => i);
            indices.sort((i, j) {
              final cA = src.getCellFlat(rowStart + i) as Complex;
              final cB = src.getCellFlat(rowStart + j) as Complex;
              final cmpReal = cA.real.compareTo(cB.real);
              if (cmpReal != 0) return cmpReal;
              return cA.imag.compareTo(cB.imag);
            });
            for (var i = 0; i < n; i++) {
              result.setCellFlat(rowStart + i, indices[i]);
            }
          }
          return finish();
        case DType.boolean:
          for (var r = 0; r < numRows; r++) {
            final rowStart = r * n;
            final indices = List<int>.generate(n, (i) => i);
            indices.sort((i, j) {
              final bA = src.getCellFlat(rowStart + i) as bool;
              final bB = src.getCellFlat(rowStart + j) as bool;
              if (bA == bB) return 0;
              return bA ? 1 : -1;
            });
            for (var i = 0; i < n; i++) {
              result.setCellFlat(rowStart + i, indices[i]);
            }
          }
          return finish();
        case DType.int8:
          final dataPtr = src.pointer.cast<ffi.Int8>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_int8(dataPtr + r * n, resPtr + r * n, n, nativeKind);
          }
          copyIf32();
          return finish();
        case DType.uint16:
          final dataPtr = src.pointer.cast<ffi.Uint16>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_uint16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.uint32:
          final dataPtr = src.pointer.cast<ffi.Uint32>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_uint32(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.uint64:
          final dataPtr = src.pointer.cast<ffi.Uint64>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_uint64(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.float16:
          final dataPtr = src.pointer.cast<ffi.Uint16>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_float16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
        case DType.bfloat16:
          final dataPtr = src.pointer.cast<ffi.Uint16>();
          for (var r = 0; r < numRows; r++) {
            native_argsort_bfloat16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              nativeKind,
            );
          }
          copyIf32();
          return finish();
      }
    } finally {
      if (marker != null) {
        ScratchArena.reset(marker);
      }
    }
  });
}

/// Rearranges the elements of the array along a specified [axis] such that
/// the value of the element at [kth] position is in the position it would be
/// in a sorted array.
///
/// This function corresponds to NumPy's `partition` function.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
/// - [kth] must be an `int` or `List<int>` containing indices within `[0, axis_size - 1]`.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [kth] is invalid.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<T> partition<T extends DTypeTag>(
  NDArray<T> a,
  Object kth, {
  int axis = -1,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot partition a disposed array.');
  }
  if (out != null) {
    validateOutArray(out, a.shape, a.dtype, name: 'out');
  }
  final rank = a.shape.length;
  if (rank == 0) {
    if (out != null) {
      out.setCellFlat(0, a.scalar);
      return out;
    }
    return NDArray<T>.scalar(a.scalar, dtype: a.dtype);
  }

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

  final n = a.shape[targetAxis];

  // Parse and validate kth
  final List<int> kList = [];
  if (kth is int) {
    final val = kth < 0 ? n + kth : kth;
    if (val < 0 || val >= n) {
      throw RangeError.value(kth, 'kth');
    }
    kList.add(val);
  } else if (kth is Iterable<int>) {
    for (final k in kth) {
      final val = k < 0 ? n + k : k;
      if (val < 0 || val >= n) {
        throw RangeError.value(k, 'kth element');
      }
      kList.add(val);
    }
  } else {
    throw ArgumentError('kth must be an int or an Iterable of ints.');
  }

  final uniqueK = kList.toSet().toList()..sort();

  if (targetAxis != rank - 1) {
    return NDArray.scope(() {
      final swappedView = a.swapaxes(targetAxis, rank - 1);
      final partitionedView = partition(swappedView, uniqueK, axis: rank - 1);
      final resultSwapped = partitionedView.swapaxes(targetAxis, rank - 1);
      if (out != null) {
        resultSwapped.copy(out: out);
        return out;
      }
      final res = resultSwapped.copy();
      res.detachToParentScope();
      return res;
    });
  }

  return NDArray.scope(() {
    final NDArray<T> src = a.isContiguous ? a : a.copy();
    final NDArray<T>? tempResult = (out != null && !out.isContiguous)
        ? src.copy()
        : null;
    final NDArray<T> result;
    if (tempResult != null) {
      result = tempResult;
    } else {
      result = out ?? NDArray<T>.create(src.shape, src.dtype);
      if (!identical(result, src)) {
        src.copy(out: result);
      }
    }

    NDArray<T> finish() {
      if (tempResult != null && out != null) {
        tempResult.copy(out: out);
        return out;
      }
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    if (uniqueK.isEmpty) {
      return finish();
    }

    final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
    final numRows = totalSize ~/ n;

    if (src.dtype == DType.boolean) {
      // A boolean partition is sorted
      sort(result, axis: rank - 1, out: result);
      return finish();
    }

    final elementSizeInBytes = src.dtype.byteWidth;

    final baseCast = result.pointer.cast<ffi.Uint8>();
    final rowSizeInBytes = n * elementSizeInBytes;

    final marker = ScratchArena.marker;
    try {
      final cKList = ScratchArena.allocate<ffi.Int64>(
        uniqueK.length * ffi.sizeOf<ffi.Int64>(),
      );
      for (var i = 0; i < uniqueK.length; i++) {
        cKList[i] = uniqueK[i];
      }

      for (var r = 0; r < numRows; r++) {
        final rowPtr = baseCast + (r * rowSizeInBytes);

        switch (src.dtype) {
          case DType.float64:
            native_partition_double(
              rowPtr.cast<ffi.Double>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.float32:
            native_partition_float(
              rowPtr.cast<ffi.Float>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.int64:
            native_partition_int64(
              rowPtr.cast<ffi.LongLong>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.int32:
            native_partition_int32(
              rowPtr.cast<ffi.Int>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.int16:
            native_partition_int16(
              rowPtr.cast<ffi.Int16>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.uint8:
            native_partition_uint8(
              rowPtr.cast<ffi.Uint8>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.complex128:
            native_partition_complex128(
              rowPtr.cast<ffi.Double>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.complex64:
            native_partition_complex64(
              rowPtr.cast<ffi.Float>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.int8:
            native_partition_int8(
              rowPtr.cast<ffi.Int8>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.uint16:
            native_partition_uint16(
              rowPtr.cast<ffi.Uint16>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.uint32:
            native_partition_uint32(
              rowPtr.cast<ffi.Uint32>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.uint64:
            native_partition_uint64(
              rowPtr.cast<ffi.Uint64>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.float16:
            native_partition_float16(
              rowPtr.cast<ffi.Uint16>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.bfloat16:
            native_partition_bfloat16(
              rowPtr.cast<ffi.Uint16>(),
              n,
              cKList,
              uniqueK.length,
            );
          case DType.boolean:
            sort(result, axis: rank - 1, out: result);
            return finish();
        }
      }
    } finally {
      ScratchArena.reset(marker);
    }

    return finish();
  });
}

/// Returns the indices that would partition an array along a specified [axis].
///
/// This function corresponds to NumPy's `argpartition` function.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
/// - [kth] must be an `int` or `List<int>` containing indices within `[0, axis_size - 1]`.
/// - It is an error if [axis] is out of bounds.
/// - It is an error if [kth] is invalid.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<Int64> argpartition<T extends DTypeTag>(
  NDArray<T> a,
  Object kth, {
  int axis = -1,
  NDArray<Int64>? out,
}) => argpartitionAs<T, Int64>(a, kth, DType.int64, axis: axis, out: out);

/// Returns the indices that would partition an array along [axis], stored in
/// the specified integer [dtype] (`DType.int32` or `DType.int64`).
///
/// Refer to [argpartition] for full details.
NDArray<R> argpartitionAs<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  Object kth,
  DType<R> dtype, {
  int axis = -1,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute argpartition() on a disposed array.');
  }
  if (dtype != DType.int32 && dtype != DType.int64) {
    throw ArgumentError(
      'dtype must be DType.int32 or DType.int64, got $dtype.',
    );
  }
  if (out != null) {
    validateOutArray(out, a.shape, dtype, name: 'out');
  }
  final rank = a.shape.length;
  if (rank == 0) {
    if (out != null) {
      out.setCellFlat(0, 0);
      return out;
    }
    return NDArray<R>.scalar(0, dtype: dtype);
  }

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

  final n = a.shape[targetAxis];

  // Parse and validate kth
  final List<int> kList = [];
  if (kth is int) {
    final val = kth < 0 ? n + kth : kth;
    if (val < 0 || val >= n) {
      throw RangeError.value(kth, 'kth');
    }
    kList.add(val);
  } else if (kth is Iterable<int>) {
    for (final k in kth) {
      final val = k < 0 ? n + k : k;
      if (val < 0 || val >= n) {
        throw RangeError.value(k, 'kth element');
      }
      kList.add(val);
    }
  } else {
    throw ArgumentError('kth must be an int or an Iterable of ints.');
  }

  final uniqueK = kList.toSet().toList()..sort();

  if (targetAxis != rank - 1) {
    return NDArray.scope(() {
      final swappedView = a.swapaxes(targetAxis, rank - 1);
      final partitionedIndicesView = argpartitionAs<T, R>(
        swappedView,
        uniqueK,
        dtype,
        axis: rank - 1,
      );
      final resultSwapped = partitionedIndicesView.swapaxes(
        targetAxis,
        rank - 1,
      );
      if (out != null) {
        resultSwapped.copy(out: out);
        return out;
      }
      final res = resultSwapped.copy();
      res.detachToParentScope();
      return res;
    });
  }

  return NDArray.scope(() {
    final NDArray<T> src = a.isContiguous ? a : a.copy();
    final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
    final numRows = totalSize ~/ n;

    final bool needsTempOut =
        out != null && (!out.isContiguous || sharesMemory(a, out));
    final NDArray<R>? tempResult = needsTempOut
        ? NDArray<R>.create(src.shape, out.dtype)
        : null;
    final NDArray<R> result =
        tempResult ?? (out ?? NDArray<R>.create(src.shape, dtype));

    NDArray<R> finish() {
      if (tempResult != null && out != null) {
        tempResult.copy(out: out);
        return out;
      }
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    if (uniqueK.isEmpty) {
      for (var i = 0; i < totalSize; i++) {
        result.setCellFlat(i, i % n);
      }
      return finish();
    }

    final is32 = result.dtype == DType.int32;
    final marker = ScratchArena.marker;
    try {
      final ffi.Pointer<ffi.Int64> resPtr = is32
          ? ScratchArena.allocate<ffi.Int64>(
              totalSize * ffi.sizeOf<ffi.Int64>(),
            )
          : result.pointer.cast<ffi.Int64>();

      final cKList = ScratchArena.allocate<ffi.Int64>(
        uniqueK.length * ffi.sizeOf<ffi.Int64>(),
      );
      for (var i = 0; i < uniqueK.length; i++) {
        cKList[i] = uniqueK[i];
      }

      switch (src.dtype) {
        case DType.float64:
          final dataPtr = src.pointer.cast<ffi.Double>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_double(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.float32:
          final dataPtr = src.pointer.cast<ffi.Float>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_float(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.int64:
          final dataPtr = src.pointer.cast<ffi.LongLong>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_int64(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.int32:
          final dataPtr = src.pointer.cast<ffi.Int>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_int32(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.int16:
          final dataPtr = src.pointer.cast<ffi.Int16>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_int16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.uint8:
          final dataPtr = src.pointer.cast<ffi.Uint8>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_uint8(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.complex128:
          final dataPtr = src.pointer.cast<ffi.Double>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_complex128(
              dataPtr + r * n * 2,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.complex64:
          final dataPtr = src.pointer.cast<ffi.Float>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_complex64(
              dataPtr + r * n * 2,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.boolean:
          for (var r = 0; r < numRows; r++) {
            final rowStart = r * n;
            final indices = List<int>.generate(n, (i) => i);
            indices.sort((i, j) {
              final bA = src.getCellFlat(rowStart + i) as bool;
              final bB = src.getCellFlat(rowStart + j) as bool;
              if (bA == bB) return 0;
              return bA ? 1 : -1;
            });
            for (var i = 0; i < n; i++) {
              result.setCellFlat(rowStart + i, indices[i]);
            }
          }
        case DType.int8:
          final dataPtr = src.pointer.cast<ffi.Int8>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_int8(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.uint16:
          final dataPtr = src.pointer.cast<ffi.Uint16>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_uint16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.uint32:
          final dataPtr = src.pointer.cast<ffi.Uint32>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_uint32(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.uint64:
          final dataPtr = src.pointer.cast<ffi.Uint64>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_uint64(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.float16:
          final dataPtr = src.pointer.cast<ffi.Uint16>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_float16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
        case DType.bfloat16:
          final dataPtr = src.pointer.cast<ffi.Uint16>();
          for (var r = 0; r < numRows; r++) {
            native_argpartition_bfloat16(
              dataPtr + r * n,
              resPtr + r * n,
              n,
              cKList,
              uniqueK.length,
            );
          }
      }

      if (is32 && src.dtype != DType.boolean) {
        final outPtr = result.pointer.cast<ffi.Int32>();
        for (var i = 0; i < totalSize; i++) {
          outPtr[i] = resPtr[i];
        }
      }
      return finish();
    } finally {
      ScratchArena.reset(marker);
    }
  });
}

/// Finds indices where elements of [v] should be inserted to maintain order in a sorted 1-D array [a].
///
/// This function corresponds to NumPy's `searchsorted` function.
///
/// Binary search is performed using native pointers, supporting
/// multi-dimensional shapes for the query array [v]. The returned index
/// array will have the exact same shape as [v].
///
/// **Preconditions:**
/// - [a] must be a 1-D array. If [sorter] is `null`, [a] must be sorted in ascending order.
/// - [v] must have a matching data type to [a].
/// - [sorter] (optional) must be a 1-D integer array of the same size as [a]
/// - It is an error if [a] is not 1-D, [sorter] shape/size is invalid, or if data types mismatch.
/// - It is an error if any input array is already disposed.
///   containing indices that sort [a] into ascending order. If provided, binary search
///   is performed indirectly using the sorter indices, completely copy-free.
///
/// **Performance considerations:**
/// - **Time Complexity**: $O(M \log N)$ where $N$ is the size of [a] and $M$ is the size of [v].
/// - **Memory Complexity**: $O(M)$ to hold the returned N-dimensional shape. The C search runs in $O(1)$ auxiliary space.
///
/// **Example:**
/// ```dart
/// import 'package:ndarray/ndarray.dart';
///
/// void main() {
///   final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
///   final v = NDArray.fromList([15.0, 30.0, 5.0, 35.0], [2, 2], DType.float64);
///
///   // Finds insertion indices for the entire multi-dimensional grid v:
///   final indices = searchsorted(a, v, side: SearchSide.left);
///
///   print(indices.shape); // [2, 2] (matches shape of v)
///   print(indices.toList()); // [[1, 2], [0, 3]]
/// }
/// ```
NDArray<Int64> searchsorted<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> v, {
  SearchSide side = SearchSide.left,
  NDArray<DTypeTag>? sorter,
  NDArray<Int64>? out,
}) => searchsortedAs<T, Int64>(
  a,
  v,
  DType.int64,
  side: side,
  sorter: sorter,
  out: out,
);

/// Finds indices where elements of [v] should be inserted to maintain order in
/// a sorted 1-D array [a], stored in the specified integer [dtype]
/// (`DType.int32` or `DType.int64`).
///
/// Refer to [searchsorted] for full details.
NDArray<R> searchsortedAs<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> v,
  DType<R> dtype, {
  SearchSide side = SearchSide.left,
  NDArray<DTypeTag>? sorter,
  NDArray<R>? out,
}) {
  if (a.isDisposed || v.isDisposed) {
    throw StateError('Cannot execute searchsorted() on a disposed array.');
  }
  if (sorter != null && sorter.isDisposed) {
    throw StateError('Cannot use a disposed sorter array.');
  }
  if (dtype != DType.int32 && dtype != DType.int64) {
    throw ArgumentError(
      'dtype must be DType.int32 or DType.int64, got $dtype.',
    );
  }
  if (a.shape.length != 1) {
    throw ArgumentError('a must be a 1-D array.');
  }
  if (v.dtype != a.dtype) {
    throw ArgumentError(
      'v and a must have the same DType (got v: ${v.dtype}, a: ${a.dtype}).',
    );
  }

  if (sorter != null) {
    if (sorter.shape.length != 1 || sorter.shape[0] != a.shape[0]) {
      throw ArgumentError('sorter must be a 1-D array of the same size as a.');
    }
    if (!sorter.dtype.isInteger) {
      throw ArgumentError(
        'sorter must have an integer dtype, got ${sorter.dtype}.',
      );
    }
  }

  if (out != null) {
    validateOutArray(out, v.shape, dtype, name: 'out');
    if (!out.isContiguous ||
        sharesMemory(a, out) ||
        sharesMemory(v, out) ||
        (sorter != null && sharesMemory(sorter, out))) {
      return NDArray.scope(() {
        final targetOut = NDArray<R>.create(v.shape, out.dtype);
        searchsortedAs<T, R>(
          a,
          v,
          dtype,
          side: side,
          sorter: sorter,
          out: targetOut,
        );
        targetOut.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    NDArray<DTypeTag>? srcSorter;
    if (sorter != null) {
      if (sorter.dtype != DType.int64) {
        srcSorter = castNDArray(sorter, DType.int64);
      } else if (!sorter.isContiguous) {
        srcSorter = sorter.copy();
      } else {
        srcSorter = sorter;
      }
      final aSize = a.shape[0];
      final sorterPtr = srcSorter.pointer.cast<ffi.Int64>();
      for (var i = 0; i < aSize; i++) {
        final idx = sorterPtr[i];
        if (idx < 0 || idx >= aSize) {
          throw IndexError.withLength(idx, aSize, name: 'sorter[$i]');
        }
      }
    }

    final result = out ?? NDArray<R>.create(v.shape, dtype);

    if (v.size == 0) {
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    final NDArray<T> srcA = a.isContiguous ? a : a.copy();
    final NDArray<T> srcV = v.isContiguous ? v : v.copy();

    final size = srcA.shape[0];
    final numValues = srcV.size;
    final sideLeft = side == SearchSide.left ? 1 : 0;

    final ffi.Pointer<ffi.Int64> cSorter = (srcSorter != null)
        ? srcSorter.pointer.cast<ffi.Int64>()
        : ffi.Pointer<ffi.Int64>.fromAddress(0);

    final is32 = result.dtype == DType.int32;
    final ScratchMarker? marker = is32 ? ScratchArena.marker : null;
    try {
      final ffi.Pointer<ffi.Int64> resPtr = is32
          ? ScratchArena.allocate<ffi.Int64>(
              numValues * ffi.sizeOf<ffi.Int64>(),
            )
          : result.pointer.cast<ffi.Int64>();

      var wroteResultDirectly = false;
      switch (srcA.dtype) {
        case DType.float64:
          native_searchsorted_double(
            srcA.pointer.cast<ffi.Double>(),
            size,
            srcV.pointer.cast<ffi.Double>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.float32:
          native_searchsorted_float(
            srcA.pointer.cast<ffi.Float>(),
            size,
            srcV.pointer.cast<ffi.Float>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.int64:
          native_searchsorted_int64(
            srcA.pointer.cast<ffi.LongLong>(),
            size,
            srcV.pointer.cast<ffi.LongLong>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.int32:
          native_searchsorted_int32(
            srcA.pointer.cast<ffi.Int>(),
            size,
            srcV.pointer.cast<ffi.Int>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.int16:
          native_searchsorted_int16(
            srcA.pointer.cast<ffi.Int16>(),
            size,
            srcV.pointer.cast<ffi.Int16>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.uint8:
          native_searchsorted_uint8(
            srcA.pointer.cast<ffi.Uint8>(),
            size,
            srcV.pointer.cast<ffi.Uint8>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.complex128:
          native_searchsorted_complex128(
            srcA.pointer.cast<ffi.Double>(),
            size,
            srcV.pointer.cast<ffi.Double>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.complex64:
          native_searchsorted_complex64(
            srcA.pointer.cast<ffi.Float>(),
            size,
            srcV.pointer.cast<ffi.Float>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.boolean:
          wroteResultDirectly = true;
          bool getElement(int idx) {
            return srcSorter != null
                ? srcA.getCellFlat(srcSorter.getCellFlat(idx) as int) as bool
                : srcA.getCellFlat(idx) as bool;
          }

          for (var vIdx = 0; vIdx < numValues; vIdx++) {
            final val = srcV.getCellFlat(vIdx) as bool;
            var low = 0;
            var high = size;
            while (low < high) {
              final mid = low + ((high - low) >> 1);
              final midVal = getElement(mid);

              int comp;
              if (midVal == val) {
                comp = 0;
              } else {
                comp = midVal ? 1 : -1; // false < true
              }

              if (side == SearchSide.left) {
                if (comp < 0) {
                  low = mid + 1;
                } else {
                  high = mid;
                }
              } else {
                if (comp <= 0) {
                  low = mid + 1;
                } else {
                  high = mid;
                }
              }
            }
            result.setCellFlat(vIdx, low);
          }
        case DType.int8:
          native_searchsorted_int8(
            srcA.pointer.cast<ffi.Int8>(),
            size,
            srcV.pointer.cast<ffi.Int8>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.uint16:
          native_searchsorted_uint16(
            srcA.pointer.cast<ffi.Uint16>(),
            size,
            srcV.pointer.cast<ffi.Uint16>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.uint32:
          native_searchsorted_uint32(
            srcA.pointer.cast<ffi.Uint32>(),
            size,
            srcV.pointer.cast<ffi.Uint32>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.uint64:
          native_searchsorted_uint64(
            srcA.pointer.cast<ffi.Uint64>(),
            size,
            srcV.pointer.cast<ffi.Uint64>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.float16:
          native_searchsorted_float16(
            srcA.pointer.cast<ffi.Uint16>(),
            size,
            srcV.pointer.cast<ffi.Uint16>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
        case DType.bfloat16:
          native_searchsorted_bfloat16(
            srcA.pointer.cast<ffi.Uint16>(),
            size,
            srcV.pointer.cast<ffi.Uint16>(),
            resPtr,
            numValues,
            sideLeft,
            cSorter,
          );
      }
      if (is32 && !wroteResultDirectly) {
        final outPtr = result.pointer.cast<ffi.Int32>();
        for (var i = 0; i < numValues; i++) {
          outPtr[i] = resPtr[i];
        }
      }
    } finally {
      if (marker != null) {
        ScratchArena.reset(marker);
      }
    }

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Returns elements chosen from [x] or [y] depending on [condition], or the indices where [condition] is true.
///
/// This function implements a ternary conditional selector or coordinate selector depending on its arguments:
/// 1. **Ternary Select (Three Arguments):**
///    If both [x] and [y] are provided, returns a new array with elements from [x] where [condition] is true, and
///    elements from [y] where it is false. Broadcasting rules apply across [condition], [x], and [y].
/// 2. **Coordinate Selector (One Argument):**
///    If both [x] and [y] are omitted, returns a list of 1-D index coordinate arrays representing the indices
///    where [condition] evaluates to true.
///
/// **Preconditions:**
/// - [condition] must be a boolean array (`condition.dtype == DType.boolean`).
/// - Either both or neither of [x] and [y] must be provided.
/// - If provided, [condition], [x], and [y] shapes must be mutually broadcast-compatible.
/// - It is an error if only one of [x] or [y] is provided.
/// - It is an error if [out] is specified when [x] and [y] are omitted.
/// - It is an error if the shapes are not broadcast-compatible.
/// - It is an error if the [out] recycler has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ where $N$ is the broadcasted result size.
/// - If all arrays are contiguous, of `float32`/`float64`/`int32`/`int64` types, and C-contiguous,
///   uses vectorized C operations (`s_where_double`/`s_where_float`).
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array (or list of arrays) on the unmanaged C heap. **The caller takes full ownership** of this memory and **must explicitly call [dispose]** on all returned arrays to prevent native leaks, unless executing inside a managed [NDArray.scope()].
///
/// **NumPy Counterpart:**
/// - Maps directly to NumPy's `np.where`.
///
/// **Example:**
/// ```dart
/// final cond = NDArray.fromList([true, false, true], [3], DType.boolean);
/// final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
/// final y = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
/// final result = where(cond, x, y) as NDArray<Float64>;
/// print(result.toList()); // [1.0, 20.0, 3.0]
/// result.dispose();
/// ```
///
/// **Return Value Type Behavior:**
/// - If [x] and [y] are provided, returns a single `NDArray<T>` of the resolved common type.
/// - If [x] and [y] are omitted, returns a `List<NDArray<DTypeTag>>` containing coordinates where the condition is true.
dynamic where<T extends DTypeTag>(
  NDArray<Boolean> condition, [
  NDArray<T>? x,
  NDArray<T>? y,
  NDArray<T>? out,
]) {
  if (condition.isDisposed) {
    throw StateError('Cannot execute where() on a disposed condition array.');
  }
  if (x != null && x.isDisposed) {
    throw StateError('Cannot execute where() with a disposed x array.');
  }
  if (y != null && y.isDisposed) {
    throw StateError('Cannot execute where() with a disposed y array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write where() result to a disposed output array.');
  }
  if (condition.dtype != DType.boolean) {
    throw ArgumentError('condition must be a boolean array');
  }
  if (x == null && y == null) {
    if (out != null) {
      throw ArgumentError(
        'out buffer cannot be provided when x and y are omitted.',
      );
    }
    return nonzero(condition);
  }

  if ((x == null && y != null) || (x != null && y == null)) {
    throw ArgumentError('Either both or neither of x and y must be given');
  }

  // Calculate target common shape via high-speed 3-way broadcast matching
  final commonShape = broadcast3Shapes(condition.shape, x!.shape, y!.shape);
  if (commonShape.length > 8) {
    throw UnsupportedError('where only supports up to 8 dimensions.');
  }

  final DType<DTypeTag> targetDType = resolveDType(x.dtype, y.dtype);

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for where() result.',
      );
    }
    if (sharesMemory(condition, out) ||
        sharesMemory(x, out) ||
        sharesMemory(y, out)) {
      return NDArray.scope(() {
        final temp = where<T>(condition, x, y) as NDArray<T>;
        temp.copy(out: out);
        return out;
      });
    }
  }

  return NDArray.scope(() {
    final xCast = castNDArray(x, targetDType);
    final yCast = castNDArray(y, targetDType);

    // Compute precise broadcasted strides for each operand independently to commonShape
    final stridesCond = broadcastStrides(condition, commonShape);
    final stridesX = broadcastStrides(xCast, commonShape);
    final stridesY = broadcastStrides(yCast, commonShape);

    final NDArray<DTypeTag> result =
        out ?? _createNDArrayTyped(commonShape, targetDType);
    final resultStrides = result.strides;

    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int64>(
        commonShape.length * ffi.sizeOf<ffi.Int64>(),
      );
      final cStridesCond = ScratchArena.allocate<ffi.Int64>(
        stridesCond.length * ffi.sizeOf<ffi.Int64>(),
      );
      final cStridesX = ScratchArena.allocate<ffi.Int64>(
        stridesX.length * ffi.sizeOf<ffi.Int64>(),
      );
      final cStridesY = ScratchArena.allocate<ffi.Int64>(
        stridesY.length * ffi.sizeOf<ffi.Int64>(),
      );
      final cStridesRes = ScratchArena.allocate<ffi.Int64>(
        resultStrides.length * ffi.sizeOf<ffi.Int64>(),
      );

      for (var i = 0; i < commonShape.length; i++) {
        cShape[i] = commonShape[i];
        cStridesCond[i] = stridesCond[i];
        cStridesX[i] = stridesX[i];
        cStridesY[i] = stridesY[i];
        cStridesRes[i] = resultStrides[i];
      }

      switch (targetDType) {
        case DType.float64:
          s_where_double(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
        case DType.float32:
          s_where_float(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
        case DType.int64:
        case DType.uint64:
          s_where_int64(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
        case DType.int32:
        case DType.uint32:
          s_where_int32(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
        case DType.uint8:
        case DType.int8:
        case DType.boolean:
          s_where_uint8(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
        case DType.int16:
        case DType.uint16:
        case DType.float16:
        case DType.bfloat16:
          s_where_int16(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
        case DType.complex128:
          s_where_complex128(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
        case DType.complex64:
          s_where_complex64(
            condition.pointer.cast(),
            cStridesCond,
            xCast.pointer.cast(),
            cStridesX,
            yCast.pointer.cast(),
            cStridesY,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            commonShape.length,
          );
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Returns the indices of the elements that are non-zero.
///
/// Returns a `List<NDArray<Int64>>` containing 1D integer arrays, one for each dimension
/// of [a], which give the coordinates of the non-zero elements along that dimension.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
List<NDArray<Int64>> nonzero<T extends DTypeTag>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute nonzero() on a disposed array.');
  }
  final rank = a.shape.length;
  final count = count_nonzero<T>(a).scalar;
  final results = List.generate(
    rank,
    (_) => NDArray<Int64>.create([count], DType.int64, zeroInit: true),
  );

  if (count == 0 || rank == 0) {
    return results;
  }

  // Convert input to a contiguous boolean mask using type-specialized native C intrinsics!
  final cond = NDArray.scope(() {
    final res = NDArray<Boolean>.create(a.shape, DType.boolean);
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int64>(
        rank * ffi.sizeOf<ffi.Int64>(),
      );
      final cStrides = ScratchArena.allocate<ffi.Int64>(
        rank * ffi.sizeOf<ffi.Int64>(),
      );
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStrides[i] = a.strides[i];
      }

      final isContiguousVal = a.isContiguous ? 1 : 0;

      switch (a.dtype) {
        case DType.float64:
          native_to_bool_mask_double(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.float32:
          native_to_bool_mask_float(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.int64:
          native_to_bool_mask_int64(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.int32:
          native_to_bool_mask_int32(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.complex128:
          native_to_bool_mask_complex128(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.complex64:
          native_to_bool_mask_complex64(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.boolean:
        case DType.uint8:
          native_to_bool_mask_uint8(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.int16:
          native_to_bool_mask_int16(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          final doubleA = castNDArray(a, DType.float64);
          final doubleStrides = ScratchArena.allocate<ffi.Int64>(
            rank * ffi.sizeOf<ffi.Int64>(),
          );
          for (var i = 0; i < rank; i++) {
            doubleStrides[i] = doubleA.strides[i];
          }
          native_to_bool_mask_double(
            doubleA.pointer.cast(),
            doubleA.size,
            cShape,
            doubleStrides,
            rank,
            doubleA.isContiguous ? 1 : 0,
            res.pointer.cast(),
          );
          doubleA.dispose();
      }
    } finally {
      ScratchArena.reset(marker);
    }
    return res.detachToParentScope();
  });

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cStrides = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    for (var i = 0; i < rank; i++) {
      cShape[i] = cond.shape[i];
      cStrides[i] = cond.strides[i];
    }

    final outCoords = ScratchArena.allocate<ffi.Pointer<ffi.Int64>>(
      rank * ffi.sizeOf<ffi.Pointer<ffi.Int64>>(),
    );
    for (var d = 0; d < rank; d++) {
      outCoords[d] = results[d].pointer.cast<ffi.Int64>();
    }

    native_collect_nonzero_coords(
      cond.pointer.cast(),
      cond.size,
      cShape,
      cStrides,
      rank,
      outCoords,
    );
  } finally {
    ScratchArena.reset(marker);
    cond.dispose();
  }

  return results;
}

/// Returns indices that are non-zero in the flattened version of [a].
///
/// Equivalent to `nonzero(a.reshape([a.size]))[0]` for contiguous arrays or
/// `nonzero(a.flatten())[0]` for strided arrays.
///
/// **Preconditions:**
/// - [a] (and [out] if provided) must not be disposed.
/// - If [out] is provided, it must be writeable with dtype [DType.int64] and
///   1-D shape matching the number of non-zero elements in [a].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([-2, -1, 0, 1, 2], [5], DType.int32);
/// final idx = flatnonzero(a); // [0, 1, 3, 4] (DType.int64)
/// ```
NDArray<Int64> flatnonzero<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<Int64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute flatnonzero() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write flatnonzero() result to a disposed output array.',
    );
  }
  return NDArray.scope(() {
    final flat = a.isContiguous
        ? a.reshape([a.size])
        : a.copy().reshape([a.size]);
    final coords = nonzero<T>(flat);
    final res = coords[0];
    if (out != null) {
      if (!out.isWriteable ||
          !listEquals(out.shape, res.shape) ||
          out.dtype != DType.int64) {
        throw ArgumentError.value(
          out,
          'out',
          'Must be writeable with shape ${res.shape} and dtype int64',
        );
      }
      res.copy(out: out);
      return out;
    }
    return res.detachToParentScope();
  });
}

/// Find the indices of array elements that are non-zero, grouped by element.
///
/// Returns a 2D array of shape `[M, N]` where `M` is the number of non-zero
/// elements, and `N` is the rank of [a].
///
/// It is an error if [a] is disposed.
NDArray<Int64> argwhere<T extends DTypeTag>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute argwhere() on a disposed array.');
  }
  final rank = a.shape.length;

  return NDArray.scope(() {
    final count = count_nonzero<T>(a).scalar;

    if (rank == 0) {
      if (count > 0) {
        return NDArray<Int64>.create([1, 0], DType.int64).detachToParentScope();
      } else {
        return NDArray<Int64>.create([0, 0], DType.int64).detachToParentScope();
      }
    }

    final result = NDArray<Int64>.create(
      [count, rank],
      DType.int64,
      zeroInit: true,
    );
    if (count == 0) {
      return result.detachToParentScope();
    }

    // Convert input to a contiguous boolean mask using type-specialized native C intrinsics!
    final cond = NDArray<Boolean>.create(a.shape, DType.boolean);
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int64>(
        rank * ffi.sizeOf<ffi.Int64>(),
      );
      final cStrides = ScratchArena.allocate<ffi.Int64>(
        rank * ffi.sizeOf<ffi.Int64>(),
      );
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStrides[i] = a.strides[i];
      }

      final isContiguousVal = a.isContiguous ? 1 : 0;

      switch (a.dtype) {
        case DType.float64:
          native_to_bool_mask_double(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.float32:
          native_to_bool_mask_float(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.int64:
          native_to_bool_mask_int64(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.int32:
          native_to_bool_mask_int32(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.complex128:
          native_to_bool_mask_complex128(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.complex64:
          native_to_bool_mask_complex64(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.boolean:
        case DType.uint8:
          native_to_bool_mask_uint8(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.int16:
          native_to_bool_mask_int16(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            cond.pointer.cast(),
          );
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          final doubleA = castNDArray(a, DType.float64);
          final doubleStrides = ScratchArena.allocate<ffi.Int64>(
            rank * ffi.sizeOf<ffi.Int64>(),
          );
          for (var i = 0; i < rank; i++) {
            doubleStrides[i] = doubleA.strides[i];
          }
          native_to_bool_mask_double(
            doubleA.pointer.cast(),
            doubleA.size,
            cShape,
            doubleStrides,
            rank,
            doubleA.isContiguous ? 1 : 0,
            cond.pointer.cast(),
          );
          doubleA.dispose();
      }

      final cCondShape = ScratchArena.allocate<ffi.Int64>(
        rank * ffi.sizeOf<ffi.Int64>(),
      );
      final cCondStrides = ScratchArena.allocate<ffi.Int64>(
        rank * ffi.sizeOf<ffi.Int64>(),
      );
      for (var i = 0; i < rank; i++) {
        cCondShape[i] = cond.shape[i];
        cCondStrides[i] = cond.strides[i];
      }

      native_collect_nonzero_coords_grouped(
        cond.pointer.cast(),
        cond.size,
        cCondShape,
        cCondStrides,
        rank,
        result.pointer.cast<ffi.Int64>(),
      );
    } finally {
      ScratchArena.reset(marker);
    }

    return result.detachToParentScope();
  });
}

void _dispatchCountNonzeroFFI(
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<ffi.Int64> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
  int isContig,
  DType dtype,
) {
  switch (dtype) {
    case DType.float64:
      native_count_nonzero_double(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.float32:
      native_count_nonzero_float(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.int64:
      native_count_nonzero_int64(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.int32:
      native_count_nonzero_int32(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.boolean:
    case DType.uint8:
      native_count_nonzero_uint8(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.int16:
      native_count_nonzero_int16(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.complex128:
      native_count_nonzero_complex128(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.complex64:
      native_count_nonzero_complex64(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.float16:
    case DType.bfloat16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
      native_count_nonzero_double(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
  }
}

/// Count the number of non-zero elements in the array [a].
///
/// If [axis] is provided, counts along that axis and returns a new array.
/// Otherwise, counts all elements globally and returns a 0-dimensional [NDArray]
/// whose value can be accessed via [scalar].
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<Int64> count_nonzero<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  NDArray<DTypeTag>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot count non-zero elements on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write count_nonzero result to a disposed output array.',
    );
  }

  final rank = a.shape.length;

  final int? normAxis;
  if (axis != null) {
    normAxis = axis < 0 ? rank + axis : axis;
    if (normAxis < 0 || normAxis >= rank) {
      throw RangeError.range(
        axis,
        -rank,
        rank - 1,
        'axis',
        'Must be within valid rank range',
      );
    }
  } else {
    normAxis = null;
  }

  final targetShape = normAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(normAxis));

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, targetShape) ||
        out.dtype != DType.int64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Int64>.create(out.shape, DType.int64);
        count_nonzero<T>(a, axis: axis, out: temp);
        return temp.copy(out: out as NDArray<Int64>);
      });
    }
  }

  if (a.dtype == DType.float16 ||
      a.dtype == DType.bfloat16 ||
      a.dtype == DType.int8 ||
      a.dtype == DType.uint64 ||
      a.dtype == DType.uint32 ||
      a.dtype == DType.uint16) {
    final doubleA = castNDArray(a, DType.float64);
    try {
      return count_nonzero(doubleA, axis: axis, out: out);
    } finally {
      doubleA.dispose();
    }
  }

  if (normAxis == null) {
    // Global flat reduction
    final isContig = a.isContiguous;
    final NDArray<T> src;
    if (!isContig) {
      src = a.copy();
    } else {
      src = a;
    }

    final result =
        (out ?? NDArray<Int64>.create([], DType.int64)) as NDArray<Int64>;
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int64>(ffi.sizeOf<ffi.Int64>());
      cShape[0] = src.size;
      final cStrides = ScratchArena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>(),
      );
      cStrides[0] = 1;

      _dispatchCountNonzeroFFI(
        src.pointer,
        cStrides,
        result.pointer.cast<ffi.Int64>(),
        cStrides,
        cShape,
        1,
        -1,
        1,
        src.dtype,
      );
    } finally {
      ScratchArena.reset(marker);
      if (!isContig) src.dispose();
    }

    return result;
  }

  // Axis reduction
  final targetAxis = normAxis;

  final result =
      (out ?? NDArray<Int64>.create(targetShape, DType.int64))
          as NDArray<Int64>;

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cStridesSrc = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    final rankDest = targetShape.length;
    final cStridesDest = ScratchArena.allocate<ffi.Int64>(
      (rankDest > 0 ? rankDest : 1) * ffi.sizeOf<ffi.Int64>(),
    );
    for (var i = 0; i < rankDest; i++) {
      cStridesDest[i] = result.strides[i];
    }

    _dispatchCountNonzeroFFI(
      a.pointer,
      cStridesSrc,
      result.pointer.cast<ffi.Int64>(),
      cStridesDest,
      cShape,
      rank,
      targetAxis,
      0,
      a.dtype,
    );
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

void _dispatchArgMinMaxFFI(
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<ffi.Int64> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
  int isMax,
  int isContig,
  DType dtype,
) {
  switch (dtype) {
    case DType.float64:
      native_argminmax_double(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.float32:
      native_argminmax_float(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.int64:
      native_argminmax_int64(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.int32:
      native_argminmax_int32(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.boolean:
    case DType.uint8:
      native_argminmax_uint8(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.int16:
      native_argminmax_int16(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.float16:
    case DType.bfloat16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
      native_argminmax_double(
        src,
        stridesSrc,
        dest,
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.complex128:
    case DType.complex64:
      throw UnsupportedError('Unsupported data type for argmin/argmax: $dtype');
  }
}

NDArray<Int64> _uint64ArgMinMax(
  NDArray<DTypeTag> a,
  int? normAxis,
  bool isMax, {
  bool keepdims = false,
  NDArray<DTypeTag>? out,
  required List<int> targetShape,
}) {
  final result =
      (out ?? NDArray<Int64>.create(targetShape, DType.int64))
          as NDArray<Int64>;
  final rank = a.shape.length;

  if (normAxis == null) {
    if (a.size == 0) {
      throw ArgumentError('attempt to get argmin/argmax of an empty sequence');
    }
    var bestIdx = 0;
    var bestVal = a.getCellFlat(0) as int;
    for (var i = 1; i < a.size; i++) {
      final val = a.getCellFlat(i) as int;
      final comp = uint64Compare(val, bestVal);
      if (isMax ? comp > 0 : comp < 0) {
        bestVal = val;
        bestIdx = i;
      }
    }
    result.setCellFlat(0, bestIdx);
    return result;
  }

  final axisLen = a.shape[normAxis];
  if (axisLen == 0) {
    throw ArgumentError('attempt to get argmin/argmax of an empty sequence');
  }

  final squeezedDestStrides = keepdims
      ? (List<int>.from(result.strides)..removeAt(normAxis))
      : result.strides;
  final outShape = List<int>.from(a.shape)..removeAt(normAxis);
  final outRank = outShape.length;
  final outSize = result.size;

  if (outRank == 0) {
    var bestIdx = 0;
    var bestVal = a.getCellFlat(0) as int;
    for (var i = 1; i < axisLen; i++) {
      final val = a.getCellFlat(i) as int;
      final comp = uint64Compare(val, bestVal);
      if (isMax ? comp > 0 : comp < 0) {
        bestVal = val;
        bestIdx = i;
      }
    }
    result.setCellFlat(0, bestIdx);
    return result;
  }

  final outCoords = List<int>.filled(outRank, 0);
  final aCoords = List<int>.filled(rank, 0);

  for (var outIdx = 0; outIdx < outSize; outIdx++) {
    var c = 0;
    for (var d = 0; d < rank; d++) {
      if (d == normAxis) continue;
      aCoords[d] = outCoords[c++];
    }

    aCoords[normAxis] = 0;
    var bestIdx = 0;
    var bestVal = a.getCell(aCoords) as int;
    for (var i = 1; i < axisLen; i++) {
      aCoords[normAxis] = i;
      final val = a.getCell(aCoords) as int;
      final comp = uint64Compare(val, bestVal);
      if (isMax ? comp > 0 : comp < 0) {
        bestVal = val;
        bestIdx = i;
      }
    }

    var destOffset = result.offsetElements;
    for (var d = 0; d < outRank; d++) {
      destOffset += outCoords[d] * squeezedDestStrides[d];
    }
    result.setCellRaw(destOffset, bestIdx);

    for (var d = outRank - 1; d >= 0; d--) {
      outCoords[d]++;
      if (outCoords[d] < outShape[d]) break;
      outCoords[d] = 0;
    }
  }

  return result;
}

NDArray<Int64> _argminmaxFFI<T extends DTypeTag>(
  NDArray<T> a,
  int? axis,
  bool isMax, {
  bool keepdims = false,
  NDArray<DTypeTag>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate reduction on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported.');
  }

  final rank = a.shape.length;
  final isMaxVal = isMax ? 1 : 0;

  final int? normAxis;
  if (axis != null) {
    normAxis = axis < 0 ? rank + axis : axis;
    if (normAxis < 0 || normAxis >= rank) {
      throw RangeError.range(axis, -rank, rank - 1, 'axis');
    }
  } else {
    normAxis = null;
  }

  if (normAxis == null ? a.size == 0 : a.shape[normAxis] == 0) {
    throw ArgumentError('Cannot compute reduction on an empty array.');
  }

  final targetShape = normAxis == null
      ? (keepdims ? List<int>.filled(rank, 1) : <int>[])
      : (keepdims
            ? (List<int>.from(a.shape)..[normAxis] = 1)
            : (List<int>.from(a.shape)..removeAt(normAxis)));

  if (out != null) {
    validateOutArray(out, targetShape, DType.int64, name: 'out');
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = NDArray<Int64>.create(out.shape, DType.int64);
        _argminmaxFFI<T>(a, axis, isMax, keepdims: keepdims, out: temp);
        return temp.copy(out: out as NDArray<Int64>);
      });
    }
  }

  if (a.size == 0) {
    if (out != null) return out as NDArray<Int64>;
    return NDArray<Int64>.create(targetShape, DType.int64);
  }

  if (a.dtype == DType.float16 ||
      a.dtype == DType.bfloat16 ||
      a.dtype == DType.int8 ||
      a.dtype == DType.uint32 ||
      a.dtype == DType.uint16) {
    final doubleA = castNDArray(a, DType.float64);
    try {
      return _argminmaxFFI(doubleA, axis, isMax, keepdims: keepdims, out: out);
    } finally {
      doubleA.dispose();
    }
  }

  if (a.dtype == DType.uint64) {
    return _uint64ArgMinMax(
      a,
      normAxis,
      isMax,
      keepdims: keepdims,
      out: out,
      targetShape: targetShape,
    );
  }

  if (normAxis == null) {
    // Global flat reduction
    final isContig = a.isContiguous;
    final NDArray<T> src;
    if (!isContig) {
      src = a.copy();
    } else {
      src = a;
    }

    final result =
        (out ?? NDArray<Int64>.create(targetShape, DType.int64))
            as NDArray<Int64>;
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int64>(ffi.sizeOf<ffi.Int64>());
      cShape[0] = src.size;
      final cStrides = ScratchArena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>(),
      );
      cStrides[0] = 1;

      _dispatchArgMinMaxFFI(
        src.pointer,
        cStrides,
        result.pointer.cast<ffi.Int64>(),
        cStrides, // dummy contiguous dest strides
        cShape,
        1, // dummy rank
        -1, // global reduction flag
        isMaxVal,
        1, // isContig flat
        src.dtype,
      );
    } finally {
      ScratchArena.reset(marker);
      if (!isContig) src.dispose();
    }

    return result;
  }

  // Axis reduction
  final targetAxis = normAxis;

  final result =
      (out ?? NDArray<Int64>.create(targetShape, DType.int64))
          as NDArray<Int64>;

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cStridesSrc = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    final squeezedDestStrides = keepdims
        ? (List<int>.from(result.strides)..removeAt(targetAxis))
        : result.strides;
    final rankDest = squeezedDestStrides.length;
    final cStridesDest = ScratchArena.allocate<ffi.Int64>(
      (rankDest > 0 ? rankDest : 1) * ffi.sizeOf<ffi.Int64>(),
    );
    for (var i = 0; i < rankDest; i++) {
      cStridesDest[i] = squeezedDestStrides[i];
    }

    _dispatchArgMinMaxFFI(
      a.pointer,
      cStridesSrc,
      result.pointer.cast<ffi.Int64>(),
      cStridesDest,
      cShape,
      rank,
      targetAxis,
      isMaxVal,
      0, // axis-based strided reduction
      a.dtype,
    );
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Returns the indices of the maximum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat 0-dimensional [NDArray]
/// (or shape with 1s if [keepdims] is true) whose value can be accessed via [scalar].
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<Int64> argmax<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<DTypeTag>? out,
}) {
  return _argminmaxFFI<T>(a, axis, true, keepdims: keepdims, out: out);
}

/// Returns the indices of the minimum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat 0-dimensional [NDArray]
/// (or shape with 1s if [keepdims] is true) whose value can be accessed via [scalar].
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<Int64> argmin<T extends DTypeTag>(
  NDArray<T> a, {
  int? axis,
  bool keepdims = false,
  NDArray<DTypeTag>? out,
}) {
  return _argminmaxFFI<T>(a, axis, false, keepdims: keepdims, out: out);
}

/// Comparison operators for search operations.
enum CompareOp {
  /// Equal to (`==`)
  equal,

  /// Not equal to (`!=`)
  notEqual,

  /// Less than (`<`)
  less,

  /// Less than or equal to (`<=`)
  lessEqual,

  /// Greater than (`>`)
  greater,

  /// Greater than or equal to (`>=`)
  greaterEqual,
}

/// Finds the coordinates of the first element in [a] that satisfies the comparison [op] with [target].
///
/// Traverses the array element-by-element starting from [startCoords] (which defaults to
/// `0` for dimensions with positive direction, or `shape[i] - 1` for negative) and steps in the
/// directions specified by [directions] (which defaults to forward `1` for all dimensions).
///
/// Returns a list of coordinates (e.g. `[row, col]`) on match, or `null` if no matching element is found.
/// For 0-dimensional arrays, returns `[]` on match.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - [target] must match the Dart representation of the [DType] of [a] (e.g., [double] for [DType.float64],
/// - It is an error if [a] is disposed.
/// - It is an error if the length of [startCoords] or [directions] does not match the rank of [a].
/// - It is an error if [directions] contains any values other than `1` or `-1`.
/// - It is an error if any coordinate in [startCoords] is out of bounds for the array's shape.
/// - It is an error if [a] has a complex data type and [op] is an inequality operator (e.g., [CompareOp.less]).
///   [int] for integer types, [Complex] for complex types).
/// - Complex types only support [CompareOp.equal] and [CompareOp.notEqual].
/// - If [startCoords] is provided, its length must match [a.shape.length] (the rank of the array),
///   and each coordinate `startCoords[i]` must satisfy `0 <= startCoords[i] < a.shape[i]`.
/// - If [directions] is provided, its length must match [a.shape.length], and it must only
///   contain `1` (forward search) or `-1` (backward search) for each dimension.
///
/// **Performance considerations:**
/// - Complexity is $O(N)$ in the worst case where $N$ is the number of elements in [a].
/// - It performs a linear search with early-exit (short-circuiting) implemented in native C, which
///   avoids allocating temporary boolean masks or intermediate coordinate arrays.
///   No intermediate Dart objects are allocated for coordinate tracking during search.
/// - Runs in $O(1)$ memory overhead (allocates FFI arguments via [ScratchArena]).
///
/// **Equivalent NumPy Operations:**
/// - In NumPy, coordinates of matching elements are typically found using `np.argwhere(cond)`.
///   However, `np.argwhere` evaluates the condition on the entire array and returns all matches, which
///   allocates memory. This function is analogous to a version that returns only the
///   first matching coordinate index list: `np.argwhere(op(a, target))[0]` (if one exists).
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
List<int>? findIndex<T extends DTypeTag>(
  NDArray<T> a,
  CompareOp op,
  Object target, {
  List<int>? startCoords,
  List<int>? directions,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute findIndex() on a disposed array.');
  }

  if (a.size == 0) {
    return null;
  }

  final rank = a.shape.length;

  if (startCoords != null && startCoords.length != rank) {
    throw ArgumentError.value(
      startCoords,
      'startCoords',
      'Length must match NDArray rank ($rank)',
    );
  }
  if (directions != null && directions.length != rank) {
    throw ArgumentError.value(
      directions,
      'directions',
      'Length must match NDArray rank ($rank)',
    );
  }

  final resolvedDirections = directions ?? List<int>.filled(rank, 1);
  for (var i = 0; i < rank; i++) {
    if (resolvedDirections[i] != 1 && resolvedDirections[i] != -1) {
      throw ArgumentError.value(
        directions,
        'directions',
        'Direction values must be 1 or -1',
      );
    }
  }

  final resolvedStart =
      startCoords ??
      List<int>.generate(
        rank,
        (i) => resolvedDirections[i] >= 0 ? 0 : a.shape[i] - 1,
      );

  for (var i = 0; i < rank; i++) {
    if (resolvedStart[i] < 0 || resolvedStart[i] >= a.shape[i]) {
      throw RangeError.range(
        resolvedStart[i],
        0,
        a.shape[i] - 1,
        'startCoords[$i]',
      );
    }
  }

  final opVal = _mapCompareOp(op);
  if (a.dtype.isComplex && opVal != CMP_OP_EQ && opVal != CMP_OP_NE) {
    throw UnsupportedError(
      'Complex types only support CompareOp.equal and CompareOp.notEqual',
    );
  }

  final marker = ScratchArena.marker;
  try {
    final cShape = a.shape.isEmpty
        ? ffi.nullptr
        : ScratchArena.copyInts(a.shape);
    final cStridesA = a.strides.isEmpty
        ? ffi.nullptr
        : ScratchArena.copyInts(a.strides);

    final cStartCoords = rank == 0
        ? ffi.nullptr
        : ScratchArena.copyInts(resolvedStart);
    final cDirections = rank == 0
        ? ffi.nullptr
        : ScratchArena.copyInts(resolvedDirections);

    final cMatchCoords = rank == 0
        ? ffi.nullptr
        : ScratchArena.allocate<ffi.Int64>(rank * ffi.sizeOf<ffi.Int64>());

    final cTarget = _allocateTarget(target, a.dtype);

    final success = ndarray_find_index(
      opVal,
      a.dtype.index,
      a.pointer.cast(),
      cStridesA,
      cShape,
      rank,
      cTarget,
      cStartCoords,
      cDirections,
      cMatchCoords,
    );

    if (success == 1) {
      if (rank == 0) {
        return [];
      }
      return List<int>.generate(rank, (i) => cMatchCoords[i]);
    } else {
      return null;
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

int _mapCompareOp(CompareOp op) {
  switch (op) {
    case CompareOp.equal:
      return CMP_OP_EQ;
    case CompareOp.notEqual:
      return CMP_OP_NE;
    case CompareOp.less:
      return CMP_OP_LT;
    case CompareOp.lessEqual:
      return CMP_OP_LE;
    case CompareOp.greater:
      return CMP_OP_GT;
    case CompareOp.greaterEqual:
      return CMP_OP_GE;
  }
}

ffi.Pointer<ffi.Void> _allocateTarget(Object value, DType<DTypeTag> dtype) {
  switch (dtype) {
    case DType.float64:
      final ptr = ScratchArena.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>());
      ptr.value = value as double;
      return ptr.cast();
    case DType.float32:
      final ptr = ScratchArena.allocate<ffi.Float>(ffi.sizeOf<ffi.Float>());
      ptr.value = value as double;
      return ptr.cast();
    case DType.int32:
      final ptr = ScratchArena.allocate<ffi.Int32>(ffi.sizeOf<ffi.Int32>());
      ptr.value = value as int;
      return ptr.cast();
    case DType.int64:
      final ptr = ScratchArena.allocate<ffi.Int64>(ffi.sizeOf<ffi.Int64>());
      ptr.value = value as int;
      return ptr.cast();
    case DType.uint8:
      final ptr = ScratchArena.allocate<ffi.Uint8>(ffi.sizeOf<ffi.Uint8>());
      ptr.value = value as int;
      return ptr.cast();
    case DType.boolean:
      final ptr = ScratchArena.allocate<ffi.Uint8>(ffi.sizeOf<ffi.Uint8>());
      ptr.value = (value as bool) ? 1 : 0;
      return ptr.cast();
    case DType.int16:
      final ptr = ScratchArena.allocate<ffi.Int16>(ffi.sizeOf<ffi.Int16>());
      ptr.value = value as int;
      return ptr.cast();
    case DType.complex128:
      final cVal = value as Complex;
      final ptr = ScratchArena.allocate<ffi.Double>(
        ffi.sizeOf<ffi.Double>() * 2,
      );
      ptr[0] = cVal.real;
      ptr[1] = cVal.imag;
      return ptr.cast();
    case DType.complex64:
      final cVal = value as Complex;
      final ptr = ScratchArena.allocate<ffi.Float>(ffi.sizeOf<ffi.Float>() * 2);
      ptr[0] = cVal.real;
      ptr[1] = cVal.imag;
      return ptr.cast();
    case DType.float16:
      final ptr = ScratchArena.allocate<ffi.Float>(ffi.sizeOf<ffi.Float>());
      ptr.value = Float16Utils.decodeFloat16(
        Float16Utils.encodeFloat16((value as num).toDouble()),
      );
      return ptr.cast();
    case DType.bfloat16:
      final ptr = ScratchArena.allocate<ffi.Float>(ffi.sizeOf<ffi.Float>());
      ptr.value = Float16Utils.decodeBFloat16(
        Float16Utils.encodeBFloat16((value as num).toDouble()),
      );
      return ptr.cast();
    case DType.int8:
      final ptr = ScratchArena.allocate<ffi.Int8>(ffi.sizeOf<ffi.Int8>());
      ptr.value = value as int;
      return ptr.cast();
    case DType.uint64:
      final ptr = ScratchArena.allocate<ffi.Uint64>(ffi.sizeOf<ffi.Uint64>());
      ptr.value = value as int;
      return ptr.cast();
    case DType.uint32:
      final ptr = ScratchArena.allocate<ffi.Uint32>(ffi.sizeOf<ffi.Uint32>());
      ptr.value = value as int;
      return ptr.cast();
    case DType.uint16:
      final ptr = ScratchArena.allocate<ffi.Uint16>(ffi.sizeOf<ffi.Uint16>());
      ptr.value = value as int;
      return ptr.cast();
  }
}
