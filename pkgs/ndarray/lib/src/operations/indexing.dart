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

import '../ndarray.dart';
import '../ndarray_extensions_bindings.dart';
import '../nditer.dart';
import '../scratch_arena.dart';
import 'helpers.dart';

/// Modes for handling out-of-bounds choice indices in [choose].
enum ChooseMode {
  /// It is an error if an index is out of bounds (default).
  raise,

  /// Wraps indices using modulo arithmetic (`(idx % N + N) % N`).
  wrap,

  /// Clamps indices to the valid choice range `[0, N - 1]`.
  clip,
}

/// Helper function to broadcast a list of shapes into a common compatible shape.
List<int> _broadcastMultiShapes(List<List<int>> shapes) {
  if (shapes.isEmpty) return [];
  var maxLen = 0;
  for (final s in shapes) {
    if (s.length > maxLen) maxLen = s.length;
  }
  final result = List<int>.filled(maxLen, 1);
  for (var i = 0; i < maxLen; i++) {
    var maxDim = 1;
    for (final s in shapes) {
      final dim = i < s.length ? s[s.length - 1 - i] : 1;
      if (dim != 1) {
        if (maxDim != 1 && maxDim != dim) {
          throw ArgumentError('Incompatible shapes for broadcasting: $shapes');
        }
        maxDim = dim;
      }
    }
    result[maxLen - 1 - i] = maxDim;
  }
  return result;
}

/// Helper function to map target coordinate to array coordinate based on broadcasting in-place.
void _mapCoordInPlace(
  List<int> targetCoord,
  List<int> arrShape,
  List<int> outCoord,
) {
  final rank = arrShape.length;
  final targetRank = targetCoord.length;
  for (var i = 0; i < rank; i++) {
    final dim = arrShape[rank - 1 - i];
    final targetDim = targetCoord[targetRank - 1 - i];
    outCoord[rank - 1 - i] = dim == 1 ? 0 : targetDim;
  }
}

/// Extracts elements from an array along a specified [axis] using coordinate index arrays.
///
/// This function corresponds to NumPy's `take_along_axis`.
///
/// **Preconditions:**
/// - It is an error if [arr], [indices], or [out] (if provided) is disposed.
/// - It is an error if [arr] and [indices] do not have the same rank (`arr.rank == indices.rank`).
/// - It is an error if [axis] is not within `[-arr.rank, arr.rank - 1]`.
/// - It is an error if non-axis dimensions of [arr] and [indices] are not broadcast-compatible.
/// - It is an error if index values in [indices] are invalid 1D indices along [axis] of [arr].
/// - It is an error if [out] is provided and its shape does not match the target broadcast shape or its dtype does not match [arr.dtype].
///
/// **Throws:**
/// - It is an error if [arr], [indices], or [out] is disposed.
/// - It is an error if ranks don't match, shapes are incompatible, or [out] shape/dtype is invalid.
/// - It is an error if [axis] or an index value in [indices] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Float64>.fromList([10, 20, 30, 40, 50, 60], [2, 3], DType.float64);
/// final indices = NDArray<DTypeTag>.fromList([2, 0, 1, 1], [2, 2], DType.int32);
/// final result = take_along_axis(a, indices, 1);
/// ```
NDArray<T> take_along_axis<T extends DTypeTag>(
  NDArray<T> arr,
  NDArray<DTypeTag> indices,
  int axis, {
  NDArray<T>? out,
}) {
  if (arr.isDisposed || indices.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute take_along_axis on a disposed array.');
  }
  final rank = arr.shape.length;
  if (indices.shape.length != rank) {
    throw ArgumentError(
      'arr and indices must have the same rank (arr.ndim=${arr.shape.length}, indices.ndim=${indices.shape.length})',
    );
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.filled(rank, 0);
  for (var i = 0; i < rank; i++) {
    if (i == normAxis) {
      targetShape[i] = indices.shape[i];
    } else {
      final dimA = arr.shape[i];
      final dimI = indices.shape[i];
      if (dimA != dimI && dimA != 1 && dimI != 1) {
        throw ArgumentError(
          'Incompatible shapes along dimension $i: arr.shape[i]=$dimA vs indices.shape[i]=$dimI',
        );
      }
      targetShape[i] = dimA > dimI ? dimA : dimI;
    }
  }

  if (out != null) {
    if (out.dtype != arr.dtype) {
      throw ArgumentError(
        'out dtype (${out.dtype}) must match arr dtype (${arr.dtype})',
      );
    }
    if (!out.isWriteable || !listEquals(out.shape, targetShape)) {
      throw ArgumentError(
        'out shape (${out.shape}) must match target shape ($targetShape)',
      );
    }
    if (sharesMemory(arr, out) || sharesMemory(indices, out)) {
      return NDArray.scope(() {
        final temp = take_along_axis(arr, indices, axis);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final result = out ?? NDArray<T>.create(targetShape, arr.dtype);
  final marker = ScratchArena.marker;
  try {
    final cArrShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cArrStrides = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cIdxShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cIdxStrides = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cOutShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cOutStrides = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cOutErrorIdx = ScratchArena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>(),
    );

    for (var i = 0; i < rank; i++) {
      cArrShape[i] = arr.shape[i];
      cArrStrides[i] = arr.strides[i];
      cIdxShape[i] = indices.shape[i];
      cIdxStrides[i] = indices.strides[i];
      cOutShape[i] = targetShape[i];
      cOutStrides[i] = result.strides[i];
    }

    final status = switch (arr.dtype) {
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 ||
      DType.int64 ||
      DType.int32 ||
      DType.int16 ||
      DType.int8 ||
      DType.uint64 ||
      DType.uint32 ||
      DType.uint16 ||
      DType.uint8 ||
      DType.boolean ||
      DType.complex128 ||
      DType.complex64 => native_take_along_axis(
        arr.dtype.index,
        indices.dtype.index,
        arr.pointer,
        cArrShape,
        cArrStrides,
        indices.pointer,
        cIdxShape,
        cIdxStrides,
        result.pointer,
        cOutShape,
        cOutStrides,
        rank,
        normAxis,
        cOutErrorIdx,
      ),
    };

    if (status != 0) {
      if (out == null) {
        result.dispose();
      }
      if (status == -1) {
        final badIdx = cOutErrorIdx.value;
        final axisSize = arr.shape[normAxis];
        throw RangeError.range(
          badIdx,
          0,
          axisSize - 1,
          'index along axis $normAxis',
        );
      }
      throw ArgumentError('take_along_axis failed with status $status');
    }

    return result;
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Puts values into an array along a specified [axis] using 1D index arrays.
///
/// This function corresponds to NumPy's `put_along_axis`.
/// Modifies [arr] in-place (or writes to [out] if provided).
///
/// **Preconditions:**
/// - It is an error if [arr], [indices], or [values] (or [out] if provided) is disposed.
/// - It is an error if [arr], [indices], and [values] do not have compatible ranks and shapes.
/// - It is an error if [axis] is not within `[-arr.rank, arr.rank - 1]`.
/// - It is an error if index values in [indices] are invalid indices along [axis] of [arr].
/// - It is an error if [out] is provided and its shape does not match [arr.shape] or its dtype does not match [arr.dtype].
///
/// **Throws:**
/// - It is an error if any input array is disposed.
/// - It is an error if shapes are incompatible or [out] is invalid.
/// - It is an error if [axis] or index values in [indices] are out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Float64>.fromList([10, 20, 30, 40, 50, 60], [2, 3], DType.float64);
/// final indices = NDArray<DTypeTag>.fromList([2, 0, 1, 1], [2, 2], DType.int32);
/// final values = NDArray<Float64>.fromList([99, 88, 77, 66], [2, 2], DType.float64);
/// put_along_axis(a, indices, values, 1);
/// ```
NDArray<T> put_along_axis<T extends DTypeTag>(
  NDArray<T> arr,
  NDArray<DTypeTag> indices,
  Object values,
  int axis, {
  NDArray<T>? out,
}) {
  if (arr.isDisposed || indices.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute put_along_axis on a disposed array.');
  }
  final rank = arr.shape.length;
  if (indices.shape.length != rank) {
    throw ArgumentError('arr and indices must have the same rank');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final bool valuesAllocated = values is! NDArray || values.dtype != arr.dtype;
  final NDArray<T> valuesArr = toNDArray<T>(values, arr.dtype);
  if (valuesArr.isDisposed) {
    if (valuesAllocated) valuesArr.dispose();
    throw StateError('Cannot execute put_along_axis with disposed values.');
  }

  final valRank = valuesArr.shape.length;
  if (valRank > rank) {
    if (valuesAllocated) valuesArr.dispose();
    throw ArgumentError(
      'values rank ($valRank) cannot be greater than arr rank ($rank)',
    );
  }

  final NDArray<T> target;
  if (out != null) {
    if (out.dtype != arr.dtype) {
      if (valuesAllocated) valuesArr.dispose();
      throw ArgumentError('out dtype must match arr dtype');
    }
    if (!out.isWriteable || !listEquals(out.shape, arr.shape)) {
      if (valuesAllocated) valuesArr.dispose();
      throw ArgumentError('out shape must match arr shape');
    }
    if (sharesMemory(arr, out) ||
        sharesMemory(indices, out) ||
        sharesMemory(valuesArr, out)) {
      try {
        return NDArray.scope(() {
          final temp = NDArray<T>.create(arr.shape, arr.dtype);
          put_along_axis(arr, indices, valuesArr, axis, out: temp);
          temp.copy(out: out);
          return out;
        });
      } finally {
        if (valuesAllocated) {
          valuesArr.dispose();
        }
      }
    }
    target = out;
  } else {
    if (!arr.isWriteable) {
      if (valuesAllocated) valuesArr.dispose();
      throw ArgumentError(
        'Assignment destination is a read-only broadcast view.',
      );
    }
    if (sharesMemory(arr, indices) || sharesMemory(arr, valuesArr)) {
      try {
        return put_along_axis(arr, indices, valuesArr, axis, out: arr);
      } finally {
        if (valuesAllocated) {
          valuesArr.dispose();
        }
      }
    }
    target = arr;
  }

  final marker = ScratchArena.marker;
  try {
    final cTargetShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cTargetStrides = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cIdxShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cIdxStrides = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cValShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cValStrides = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cOutErrorIdx = ScratchArena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>(),
    );

    for (var i = 0; i < rank; i++) {
      cTargetShape[i] = target.shape[i];
      cTargetStrides[i] = target.strides[i];
      cIdxShape[i] = indices.shape[i];
      cIdxStrides[i] = indices.strides[i];

      final valDimIndex = i - (rank - valRank);
      if (valDimIndex < 0) {
        cValShape[i] = 1;
        cValStrides[i] = 0;
      } else {
        final valDim = valuesArr.shape[valDimIndex];
        final idxDim = indices.shape[i];
        if (valDim != idxDim && valDim != 1) {
          throw ArgumentError(
            'Incompatible shapes for put_along_axis: indices shape ${indices.shape} and values shape ${valuesArr.shape}',
          );
        }
        cValShape[i] = valDim;
        cValStrides[i] = valuesArr.strides[valDimIndex];
      }
    }

    return NDArray.scope(() {
      final tempTarget = arr.copy();
      final cTempStrides = ScratchArena.allocate<ffi.Int64>(
        rank * ffi.sizeOf<ffi.Int64>(),
      );
      for (var i = 0; i < rank; i++) {
        cTempStrides[i] = tempTarget.strides[i];
      }

      final status = switch (arr.dtype) {
        DType.float64 ||
        DType.float32 ||
        DType.float16 ||
        DType.bfloat16 ||
        DType.int64 ||
        DType.int32 ||
        DType.int16 ||
        DType.int8 ||
        DType.uint64 ||
        DType.uint32 ||
        DType.uint16 ||
        DType.uint8 ||
        DType.boolean ||
        DType.complex128 ||
        DType.complex64 => native_put_along_axis(
          arr.dtype.index,
          indices.dtype.index,
          tempTarget.pointer,
          cTargetShape,
          cTempStrides,
          indices.pointer,
          cIdxShape,
          cIdxStrides,
          valuesArr.pointer,
          cValShape,
          cValStrides,
          rank,
          normAxis,
          cOutErrorIdx,
        ),
      };

      if (status != 0) {
        if (status == -1) {
          final badIdx = cOutErrorIdx.value;
          final axisSize = target.shape[normAxis];
          throw RangeError.range(
            badIdx,
            0,
            axisSize - 1,
            'index along axis $normAxis',
          );
        }
        throw ArgumentError('put_along_axis failed with status $status');
      }

      tempTarget.copy(out: target);
      return target;
    });
  } finally {
    ScratchArena.reset(marker);
    if (valuesAllocated) {
      valuesArr.dispose();
    }
  }
}

/// Constructs an array from an index array ([a]) and a list of arrays or scalars ([choices]).
///
/// This function corresponds to NumPy's `choose`.
///
/// **Preconditions:**
/// - It is an error if [a] or any choice item in [choices] (or [out] if provided) is disposed.
/// - It is an error if [choices] is empty.
/// - It is an error if shapes of [a] and all choice items are not broadcast-compatible.
/// - It is an error if index values in [a] are out of bounds and [mode] is [ChooseMode.raise].
/// - It is an error if [out] is provided and its shape does not match the broadcast shape or its dtype does not match resolved choices dtype.
///
/// **Throws:**
/// - It is an error if any input array is disposed.
/// - It is an error if [choices] is empty or shapes cannot be broadcast.
/// - It is an error if index values in [a] are out of bounds and [mode] is [ChooseMode.raise].
///
/// **Example:**
/// ```dart
/// final choices = [
///   NDArray<Float64>.fromList([0, 1, 2, 3], [2, 2], DType.float64),
///   NDArray<Float64>.fromList([10, 11, 12, 13], [2, 2], DType.float64),
/// ];
/// final a = NDArray<DTypeTag>.fromList([0, 1, 1, 0], [2, 2], DType.int32);
/// final result = choose(a, choices);
/// ```
NDArray<T> choose<T extends DTypeTag>(
  NDArray<DTypeTag> a,
  List<Object> choices, {
  NDArray<T>? out,
  ChooseMode mode = ChooseMode.raise,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute choose on a disposed array.');
  }
  if (choices.isEmpty) {
    throw ArgumentError('choices list must not be empty');
  }

  for (var i = 0; i < choices.length; i++) {
    final c = choices[i];
    if (c is NDArray && c.isDisposed) {
      throw StateError(
        'Cannot execute choose with a disposed choice array at index $i.',
      );
    }
  }

  return NDArray.scope(() {
    final hasArray = choices.any((c) => c is NDArray);
    DType getItemDType(Object item) {
      if (item is NDArray) return item.dtype;
      if (item is int) {
        if (hasArray) {
          final arrayIntDTypes = choices
              .whereType<NDArray>()
              .map((a) => a.dtype)
              .where((dt) => dt.isInteger);
          if (arrayIntDTypes.isNotEmpty) {
            return arrayIntDTypes.first;
          }
        }
        return DType.int64;
      }
      if (item is bool) return DType.boolean;
      if (item is Complex) return DType.complex128;
      return DType.float64;
    }

    final resolvedDType =
        (out?.dtype) ??
        (() {
          DType dt = getItemDType(choices.first);
          for (var i = 1; i < choices.length; i++) {
            dt = resolveDType(dt, getItemDType(choices[i]));
          }
          return dt as DType<T>;
        })();

    final choiceArrays = choices
        .map((c) => toNDArray<T>(c, resolvedDType))
        .toList();

    final allShapes = <List<int>>[a.shape, ...choiceArrays.map((c) => c.shape)];
    final targetShape = _broadcastMultiShapes(allShapes);

    if (out != null) {
      if (out.dtype != resolvedDType) {
        throw ArgumentError('out dtype must match resolved choices dtype');
      }
      if (!out.isWriteable || !listEquals(out.shape, targetShape)) {
        throw ArgumentError(
          'out shape must match broadcast shape ($targetShape)',
        );
      }
    }

    final bool needsTemp =
        out != null &&
        (!out.isContiguous ||
            sharesMemory(a, out) ||
            choices.any((c) => c is NDArray && sharesMemory(c, out)) ||
            choiceArrays.any((c) => sharesMemory(c, out)));
    final result = needsTemp || out == null
        ? NDArray<T>.create(targetShape, resolvedDType)
        : out;
    final nChoices = choiceArrays.length;

    // Fast path: contiguous arrays or scalar choice/index arrays
    final canFastPath =
        result.isContiguous &&
        ((a.isContiguous && a.size == result.size) || a.size == 1) &&
        choiceArrays.every(
          (c) => (c.isContiguous && c.size == result.size) || c.size == 1,
        );

    if (canFastPath) {
      final totalElements = result.size;
      final aIsScalar = a.size == 1;

      // Extract a pointer reader function based on a.dtype
      int Function(int) getIdx;
      switch (a.dtype) {
        case DType.int64:
          final ptr = a.pointer.cast<ffi.Int64>();
          getIdx = aIsScalar ? ((_) => ptr[0]) : ((i) => ptr[i]);
          break;
        case DType.int32:
          final ptr = a.pointer.cast<ffi.Int32>();
          getIdx = aIsScalar ? ((_) => ptr[0]) : ((i) => ptr[i]);
          break;
        case DType.int16:
          final ptr = a.pointer.cast<ffi.Int16>();
          getIdx = aIsScalar ? ((_) => ptr[0]) : ((i) => ptr[i]);
          break;
        case DType.int8:
          final ptr = a.pointer.cast<ffi.Int8>();
          getIdx = aIsScalar ? ((_) => ptr[0]) : ((i) => ptr[i]);
          break;
        case DType.uint64:
          final ptr = a.pointer.cast<ffi.Uint64>();
          final twoPow63Mod = ((1 << 62) % nChoices) * 2;
          int normalizeUint64(int raw) {
            if (raw >= 0) return raw;
            return mode == ChooseMode.wrap
                ? ((raw & 0x7FFFFFFFFFFFFFFF) % nChoices + twoPow63Mod) %
                      nChoices
                : nChoices;
          }
          getIdx = aIsScalar
              ? ((_) => normalizeUint64(ptr[0]))
              : ((i) => normalizeUint64(ptr[i]));
          break;
        case DType.uint32:
          final ptr = a.pointer.cast<ffi.Uint32>();
          getIdx = aIsScalar ? ((_) => ptr[0]) : ((i) => ptr[i]);
          break;
        case DType.uint16:
          final ptr = a.pointer.cast<ffi.Uint16>();
          getIdx = aIsScalar ? ((_) => ptr[0]) : ((i) => ptr[i]);
          break;
        case DType.uint8:
          final ptr = a.pointer.cast<ffi.Uint8>();
          getIdx = aIsScalar ? ((_) => ptr[0]) : ((i) => ptr[i]);
          break;
        case DType.float64:
        case DType.float32:
        case DType.float16:
        case DType.bfloat16:
        case DType.boolean:
        case DType.complex128:
        case DType.complex64:
          throw UnsupportedError('Unsupported index dtype: ${a.dtype}');
      }

      final isScalarChoice = choiceArrays.map((c) => c.size == 1).toList();

      switch (resolvedDType) {
        case DType.float64:
          final resPtr = result.pointer.cast<ffi.Double>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Double>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            resPtr[i] = isScalarChoice[idx] ? srcPtr[0] : srcPtr[i];
          }
          break;

        case DType.float32:
          final resPtr = result.pointer.cast<ffi.Float>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Float>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            resPtr[i] = isScalarChoice[idx] ? srcPtr[0] : srcPtr[i];
          }
          break;

        case DType.int64 || DType.uint64:
          final resPtr = result.pointer.cast<ffi.Int64>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Int64>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            resPtr[i] = isScalarChoice[idx] ? srcPtr[0] : srcPtr[i];
          }
          break;

        case DType.int32 || DType.uint32:
          final resPtr = result.pointer.cast<ffi.Int32>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Int32>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            resPtr[i] = isScalarChoice[idx] ? srcPtr[0] : srcPtr[i];
          }
          break;

        case DType.int16 || DType.uint16 || DType.float16 || DType.bfloat16:
          final resPtr = result.pointer.cast<ffi.Int16>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Int16>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            resPtr[i] = isScalarChoice[idx] ? srcPtr[0] : srcPtr[i];
          }
          break;

        case DType.int8 || DType.uint8 || DType.boolean:
          final resPtr = result.pointer.cast<ffi.Uint8>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Uint8>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            resPtr[i] = isScalarChoice[idx] ? srcPtr[0] : srcPtr[i];
          }
          break;

        case DType.complex128:
          final resPtr = result.pointer.cast<ffi.Double>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Double>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            final srcIdx = isScalarChoice[idx] ? 0 : (i << 1);
            final dstIdx = i << 1;
            resPtr[dstIdx] = srcPtr[srcIdx];
            resPtr[dstIdx + 1] = srcPtr[srcIdx + 1];
          }
          break;

        case DType.complex64:
          final resPtr = result.pointer.cast<ffi.Float>();
          final cPtrs = choiceArrays
              .map((c) => c.pointer.cast<ffi.Float>())
              .toList();
          for (var i = 0; i < totalElements; i++) {
            var idx = getIdx(i);
            switch (mode) {
              case ChooseMode.raise:
                if (idx < 0 || idx >= nChoices) {
                  throw RangeError.range(idx, 0, nChoices - 1, 'choice index');
                }
                break;
              case ChooseMode.wrap:
                idx = idx % nChoices;
                if (idx < 0) idx += nChoices;
                break;
              case ChooseMode.clip:
                if (idx < 0) {
                  idx = 0;
                } else if (idx >= nChoices) {
                  idx = nChoices - 1;
                }
                break;
            }
            final srcPtr = cPtrs[idx];
            final srcIdx = isScalarChoice[idx] ? 0 : (i << 1);
            final dstIdx = i << 1;
            resPtr[dstIdx] = srcPtr[srcIdx];
            resPtr[dstIdx + 1] = srcPtr[srcIdx + 1];
          }
          break;
      }

      if (out != null) {
        if (needsTemp) {
          result.copy(out: out);
        }
        return out;
      }
      return result.detachToParentScope();
    }

    final marker = ScratchArena.marker;
    try {
      final aCoord = List<int>.filled(a.shape.length, 0);
      final choiceCoords = choiceArrays
          .map((c) => List<int>.filled(c.shape.length, 0))
          .toList();

      final twoPow63Mod = ((1 << 62) % nChoices) * 2;
      final isUint64 = a.dtype == DType.uint64;
      final iter = NDIter(result);
      while (iter.moveNext()) {
        final coords = iter.coords;
        _mapCoordInPlace(coords, a.shape, aCoord);
        var idxVal = a.getCell(aCoord) as int;
        if (isUint64 && idxVal < 0) {
          idxVal = mode == ChooseMode.wrap
              ? ((idxVal & 0x7FFFFFFFFFFFFFFF) % nChoices + twoPow63Mod) %
                    nChoices
              : nChoices;
        }

        switch (mode) {
          case ChooseMode.raise:
            if (idxVal < 0 || idxVal >= nChoices) {
              throw RangeError.range(idxVal, 0, nChoices - 1, 'choice index');
            }
            break;
          case ChooseMode.wrap:
            idxVal = idxVal % nChoices;
            if (idxVal < 0) idxVal += nChoices;
            break;
          case ChooseMode.clip:
            if (idxVal < 0) {
              idxVal = 0;
            } else if (idxVal >= nChoices) {
              idxVal = nChoices - 1;
            }
            break;
        }

        final choiceArr = choiceArrays[idxVal];
        final choiceCoord = choiceCoords[idxVal];
        _mapCoordInPlace(coords, choiceArr.shape, choiceCoord);
        final val = choiceArr.getCell(choiceCoord);
        result.setCell(coords, val);
      }

      if (out != null) {
        if (needsTemp) {
          result.copy(out: out);
        }
        return out;
      }
      return result.detachToParentScope();
    } finally {
      ScratchArena.reset(marker);
    }
  });
}

/// Returns an array drawn from elements in [choicelist], depending on conditions in [condlist].
///
/// This function corresponds to NumPy's `select`.
///
/// **Preconditions:**
/// - It is an error if any array in [condlist] or item in [choicelist] (or [out] if provided) is disposed.
/// - It is an error if [condlist] and [choicelist] do not have the same non-zero length.
/// - It is an error if [condlist] and [choicelist] items are not broadcast-compatible.
/// - It is an error if [out] is provided and its shape does not match the broadcast shape or its dtype does not match resolved dtype.
///
/// **Throws:**
/// - It is an error if any input array is disposed.
/// - It is an error if list lengths don't match, lists are empty, or shapes/dtypes are incompatible.
///
/// **Example:**
/// ```dart
/// final x = NDArray<Float64>.fromList([1, 2, 3, 4, 5], [5], DType.float64);
/// final conds = [x < 2, x > 3];
/// final choices = [x * 10, x * 100];
/// final result = select(conds, choices, defaultValue: -1.0);
/// ```
NDArray<T> select<T extends DTypeTag>(
  List<NDArray<Boolean>> condlist,
  List<Object> choicelist, {
  Object? defaultValue,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (out != null && out.isDisposed) {
    throw StateError('Cannot execute select with a disposed out array.');
  }
  if (condlist.isEmpty || choicelist.isEmpty) {
    throw ArgumentError('condlist and choicelist must not be empty');
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError(
      'condlist (${condlist.length}) and choicelist (${choicelist.length}) must have the same length',
    );
  }

  for (var i = 0; i < condlist.length; i++) {
    if (condlist[i].isDisposed) {
      throw StateError(
        'Cannot execute select with a disposed condition array at index $i.',
      );
    }
  }

  return NDArray.scope(() {
    final resolvedDType =
        dtype ??
        (out?.dtype) ??
        (() {
          DType getItemDType(Object item) {
            if (item is NDArray) return item.dtype;
            if (item is int) return DType.int32;
            if (item is bool) return DType.boolean;
            if (item is Complex) return DType.complex128;
            return DType.float64;
          }

          DType dt = getItemDType(choicelist.first);
          for (var i = 1; i < choicelist.length; i++) {
            dt = resolveDType(dt, getItemDType(choicelist[i]));
          }
          if (defaultValue != null) {
            dt = resolveDType(dt, getItemDType(defaultValue));
          }
          return dt as DType<T>;
        })();

    for (var i = 0; i < choicelist.length; i++) {
      final c = choicelist[i];
      if (c is NDArray && c.isDisposed) {
        throw StateError(
          'Cannot execute select with a disposed choice array at index $i.',
        );
      }
    }
    if (defaultValue is NDArray && defaultValue.isDisposed) {
      throw StateError(
        'Cannot execute select with a disposed defaultValue array.',
      );
    }

    final choiceArrays = choicelist
        .map((c) => toNDArray<T>(c, resolvedDType))
        .toList();

    final defaultValObj = defaultValue ?? 0;
    final defaultArr = toNDArray<T>(defaultValObj, resolvedDType);
    if (defaultArr.isDisposed) {
      throw StateError('Cannot execute select with a disposed default array.');
    }

    final allShapes = <List<int>>[
      ...condlist.map((c) => c.shape),
      ...choiceArrays.map((c) => c.shape),
      defaultArr.shape,
    ];
    final targetShape = _broadcastMultiShapes(allShapes);

    if (out != null) {
      if (out.isDisposed) {
        throw StateError('Cannot use a disposed out array.');
      }
      if (out.dtype != resolvedDType) {
        throw ArgumentError('out dtype must match resolved dtype');
      }
      if (!out.isWriteable || !listEquals(out.shape, targetShape)) {
        throw ArgumentError(
          'out shape must match broadcast shape ($targetShape)',
        );
      }
    }

    final bool needsTemp =
        out != null &&
        (condlist.any((c) => sharesMemory(c, out)) ||
            choicelist.any((c) => c is NDArray && sharesMemory(c, out)) ||
            choiceArrays.any((c) => sharesMemory(c, out)) ||
            (defaultValue is NDArray && sharesMemory(defaultValue, out)) ||
            sharesMemory(defaultArr, out));
    final result = needsTemp || out == null
        ? NDArray<T>.create(targetShape, resolvedDType)
        : out;
    final nConds = condlist.length;
    final marker = ScratchArena.marker;
    try {
      final condCoords = condlist
          .map((c) => List<int>.filled(c.shape.length, 0))
          .toList();
      final choiceCoords = choiceArrays
          .map((c) => List<int>.filled(c.shape.length, 0))
          .toList();
      final defaultCoord = List<int>.filled(defaultArr.shape.length, 0);
      final resDType = result.dtype;

      final iter = NDIter(result);
      while (iter.moveNext()) {
        final coords = iter.coords;
        var selectedIdx = -1;
        for (var i = 0; i < nConds; i++) {
          final condArr = condlist[i];
          final condCoord = condCoords[i];
          _mapCoordInPlace(coords, condArr.shape, condCoord);
          if (condArr.getCell(condCoord)) {
            selectedIdx = i;
            break;
          }
        }

        if (selectedIdx != -1) {
          final choiceArr = choiceArrays[selectedIdx];
          final choiceCoord = choiceCoords[selectedIdx];
          _mapCoordInPlace(coords, choiceArr.shape, choiceCoord);
          final val = choiceArr.getCell(choiceCoord);
          result.setCell(coords, castValue(val, resDType));
        } else {
          _mapCoordInPlace(coords, defaultArr.shape, defaultCoord);
          final val = defaultArr.getCell(defaultCoord);
          result.setCell(coords, castValue(val, resDType));
        }
      }

      if (out != null) {
        if (needsTemp) {
          result.copy(out: out);
        }
        return out;
      }
      return result.detachToParentScope();
    } finally {
      ScratchArena.reset(marker);
    }
  });
}
