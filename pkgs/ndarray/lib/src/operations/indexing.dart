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
import 'sorting.dart';

/// Modes for handling out-of-bounds choice indices in [choose] and
/// [ravel_multi_index].
enum ChooseMode {
  /// It is an error if an index is out of bounds (default).
  raise,

  /// Wraps indices using modulo arithmetic (`(idx % N + N) % N`).
  wrap,

  /// Clamps indices to the valid choice range `[0, N - 1]`.
  clip,
}

/// Memory layout order for multi-dimensional index conversion in
/// [unravel_index] and [ravel_multi_index].
enum IndexOrder {
  /// Row-major (C-style) order, where the last axis index varies fastest.
  c,

  /// Column-major (Fortran-style) order, where the first axis index varies
  /// fastest.
  f,
}

bool _scalarFitsIntDType(int value, DType dtype) => switch (dtype) {
  DType.int8 => value >= -128 && value <= 127,
  DType.uint8 => value >= 0 && value <= 255,
  DType.int16 => value >= -32768 && value <= 32767,
  DType.uint16 => value >= 0 && value <= 65535,
  DType.int32 => value >= -2147483648 && value <= 2147483647,
  DType.uint32 => value >= 0 && value <= 4294967295,
  DType.int64 => true,
  DType.uint64 => value >= 0,
  _ => false,
};

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
          throw ArgumentError.value(
            shapes,
            'shapes',
            'Incompatible shapes for broadcasting: $shapes',
          );
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
/// {@example /example/indexing_example.dart lang=dart}
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
  if (rank == 0) {
    throw ArgumentError.value(
      arr,
      'arr',
      'Must have at least 1 dimension for take_along_axis',
    );
  }
  if (indices.shape.length != rank) {
    throw ArgumentError.value(
      indices,
      'indices',
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
        throw ArgumentError.value(
          indices,
          'indices',
          'Incompatible shapes along dimension $i: arr.shape[i]=$dimA vs indices.shape[i]=$dimI',
        );
      }
      targetShape[i] = dimA > dimI ? dimA : dimI;
    }
  }

  if (out != null) {
    validateOutBuffer(out);
    if (out.dtype != arr.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'out dtype (${out.dtype}) must match arr dtype (${arr.dtype})',
      );
    }
    if (!out.isWriteable || !listEquals(out.shape, targetShape)) {
      throw ArgumentError.value(
        out,
        'out',
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
      throw ArgumentError.value(
        status,
        'status',
        'take_along_axis failed with status $status',
      );
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
/// {@example /example/indexing_example.dart lang=dart}
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
  if (rank == 0) {
    throw ArgumentError.value(
      arr,
      'arr',
      'Must have at least 1 dimension for put_along_axis',
    );
  }
  if (indices.shape.length != rank) {
    throw ArgumentError.value(
      indices,
      'indices',
      'arr and indices must have the same rank',
    );
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
    throw ArgumentError.value(
      values,
      'values',
      'values rank ($valRank) cannot be greater than arr rank ($rank)',
    );
  }

  final NDArray<T> target;
  if (out != null) {
    try {
      validateOutBuffer(out);
    } catch (_) {
      if (valuesAllocated) valuesArr.dispose();
      rethrow;
    }
    if (out.dtype != arr.dtype) {
      if (valuesAllocated) valuesArr.dispose();
      throw ArgumentError.value(out, 'out', 'out dtype must match arr dtype');
    }
    if (!out.isWriteable || !listEquals(out.shape, arr.shape)) {
      if (valuesAllocated) valuesArr.dispose();
      throw ArgumentError.value(out, 'out', 'out shape must match arr shape');
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
    try {
      validateOutBuffer(arr, 'arr');
    } catch (_) {
      if (valuesAllocated) valuesArr.dispose();
      rethrow;
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
          throw ArgumentError.value(
            values,
            'values',
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
        throw ArgumentError.value(
          status,
          'status',
          'put_along_axis failed with status $status',
        );
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
/// {@example /example/indexing_example.dart lang=dart}
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
    throw ArgumentError.value(
      choices,
      'choices',
      'choices list must not be empty',
    );
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
    final arrayIntDTypes = choices
        .whereType<NDArray>()
        .map((a) => a.dtype)
        .where((dt) => dt.isInteger)
        .toList();
    DType getItemDType(Object item) {
      if (item is NDArray) return item.dtype;
      if (item is int) {
        for (final dt in arrayIntDTypes) {
          if (_scalarFitsIntDType(item, dt)) {
            return dt;
          }
        }
        return DType.int64;
      }
      if (item is bool) return DType.boolean;
      if (item is Complex) return DType.complex128;
      return DType.float64;
    }

    final resolvedDType = (() {
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
      validateOutBuffer(out);
      if (out.dtype != resolvedDType) {
        throw ArgumentError.value(
          out,
          'out',
          'out dtype must match resolved choices dtype',
        );
      }
      if (!out.isWriteable || !listEquals(out.shape, targetShape)) {
        throw ArgumentError.value(
          out,
          'out',
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
/// {@example /example/indexing_example.dart lang=dart}
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
    throw ArgumentError.value(
      condlist,
      'condlist',
      'condlist and choicelist must not be empty',
    );
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError.value(
      choicelist,
      'choicelist',
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
    final allChoiceItems = <Object>[...choicelist, ?defaultValue];
    final arrayIntDTypes = allChoiceItems
        .whereType<NDArray>()
        .map((a) => a.dtype)
        .where((dt) => dt.isInteger)
        .toList();
    final resolvedDType =
        dtype ??
        (() {
          DType getItemDType(Object item) {
            if (item is NDArray) return item.dtype;
            if (item is int) {
              for (final dt in arrayIntDTypes) {
                if (_scalarFitsIntDType(item, dt)) {
                  return dt;
                }
              }
              return DType.int64;
            }
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
      validateOutBuffer(out);
      if (out.dtype != resolvedDType) {
        throw ArgumentError.value(
          out,
          'out',
          'out dtype must match resolved dtype',
        );
      }
      if (!out.isWriteable || !listEquals(out.shape, targetShape)) {
        throw ArgumentError.value(
          out,
          'out',
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

/// Converts a flat index or array of flat indices into a list of
/// multi-dimensional coordinate arrays.
///
/// Each returned coordinate array has the same shape as [indices] and
/// data type [DType.int64], with one array per dimension in [shape].
///
/// The [shape] must be non-empty and must not contain negative dimensions.
/// It is an error if any index in [indices] is negative or greater than or
/// equal to the total number of elements implied by [shape].
///
/// Parameters:
/// - `indices`: An integer [NDArray] whose elements are flat indices into an
///   array of dimensions [shape].
/// - `shape`: The shape of the array used for unraveling [indices].
/// - `order`: Determines whether the indices should be viewed as indexing in
///   row-major ([IndexOrder.c], default) or column-major ([IndexOrder.f])
///   order.
/// - `out`: Optional list of pre-allocated [NDArray]s of dtype [DType.int64]
///   and shape `indices.shape`, with length equal to `shape.length`.
///
/// Returns a list of [NDArray]s of dtype [DType.int64], where the `i`-th
/// array contains the coordinates along axis `i`.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
List<NDArray<Int64>> unravel_index<T extends DTypeTag>(
  NDArray<T> indices,
  List<int> shape, {
  IndexOrder order = IndexOrder.c,
  List<NDArray<Int64>>? out,
}) {
  if (indices.isDisposed) {
    throw StateError('Cannot execute unravel_index on a disposed array.');
  }
  if (!indices.dtype.isInteger) {
    throw ArgumentError.value(
      indices.dtype,
      'indices',
      'Must have an integer dtype',
    );
  }
  if (shape.isEmpty) {
    throw ArgumentError.value(shape, 'shape', 'Must not be empty');
  }
  for (var i = 0; i < shape.length; i++) {
    if (shape[i] < 0) {
      throw ArgumentError.value(
        shape,
        'shape',
        'Must not contain negative dimensions',
      );
    }
  }

  final ndims = shape.length;
  if (out != null) {
    if (out.length != ndims) {
      throw ArgumentError.value(
        out.length,
        'out',
        'Must have length equal to shape.length ($ndims)',
      );
    }
    for (var i = 0; i < ndims; i++) {
      final o = out[i];
      if (o.isDisposed) {
        throw StateError(
          'Cannot write unravel_index result to a disposed out array at index $i.',
        );
      }
      validateOutBuffer(o, 'out[$i]');
      if (!o.isWriteable) {
        throw ArgumentError.value(o, 'out[$i]', 'Must be writeable');
      }
      if (o.dtype != DType.int64) {
        throw ArgumentError.value(o.dtype, 'out[$i]', 'Must have dtype int64');
      }
      if (!listEquals(o.shape, indices.shape)) {
        throw ArgumentError.value(
          o.shape,
          'out[$i]',
          'Must match indices.shape (${indices.shape})',
        );
      }
    }
  }

  return NDArray.scope(() {
    var needsTemp = false;
    if (out != null) {
      for (var i = 0; i < ndims; i++) {
        if (sharesMemory(indices, out[i])) {
          needsTemp = true;
          break;
        }
        for (var j = 0; j < i; j++) {
          if (sharesMemory(out[i], out[j])) {
            needsTemp = true;
            break;
          }
        }
        if (needsTemp) break;
      }
    }

    final workOut = (out == null || needsTemp)
        ? List<NDArray<Int64>>.generate(
            ndims,
            (_) => NDArray<Int64>.create(indices.shape, DType.int64),
          )
        : out;

    final isContig =
        indices.isContiguous && workOut.every((a) => a.isContiguous);
    final rank = indices.shape.length;
    final orderCode = order == IndexOrder.c ? 0 : 1;

    final marker = ScratchArena.marker;
    try {
      final dimsPtr = ScratchArena.copyInts(shape);
      final outPtrs = ScratchArena.allocate<ffi.Pointer<ffi.Int64>>(
        ndims * ffi.sizeOf<ffi.Pointer<ffi.Int64>>(),
      );
      for (var d = 0; d < ndims; d++) {
        outPtrs[d] = workOut[d].pointer.cast<ffi.Int64>();
      }

      final inPtr = indices.pointer;

      ffi.Pointer<ffi.Int64> inShapePtr = ffi.nullptr;
      ffi.Pointer<ffi.Int64> inStridesPtr = ffi.nullptr;
      ffi.Pointer<ffi.Int64> outStridesFlat = ffi.nullptr;
      if (!isContig) {
        inShapePtr = ScratchArena.copyInts(indices.shape);
        inStridesPtr = ScratchArena.copyInts(indices.strides);
        final strideCount = ndims * (rank > 0 ? rank : 1);
        outStridesFlat = ScratchArena.allocate<ffi.Int64>(
          strideCount * ffi.sizeOf<ffi.Int64>(),
        );
        for (var d = 0; d < ndims; d++) {
          final ostrides = workOut[d].strides;
          for (var r = 0; r < rank; r++) {
            outStridesFlat[d * rank + r] = ostrides[r];
          }
        }
      }

      final errIdxPtr = ScratchArena.allocate<ffi.Int64>(
        1 * ffi.sizeOf<ffi.Int64>(),
      );
      errIdxPtr.value = 0;

      final status = native_unravel_index(
        indices.dtype.index,
        inPtr,
        inShapePtr,
        inStridesPtr,
        rank,
        indices.size,
        dimsPtr,
        ndims,
        orderCode,
        outPtrs,
        outStridesFlat,
        isContig ? 1 : 0,
        errIdxPtr,
      );

      if (status == -2) {
        throw RangeError(
          'Index ${errIdxPtr.value} is out of bounds for array with shape $shape',
        );
      }
      if (status == -4) {
        throw OutOfMemoryError();
      }
      if (status != 0) {
        throw StateError('native_unravel_index failed with status $status');
      }

      if (out != null) {
        if (needsTemp) {
          for (var d = 0; d < ndims; d++) {
            workOut[d].copy(out: out[d]);
          }
        }
        return out;
      }

      return [for (final arr in workOut) arr.detachToParentScope()];
    } finally {
      ScratchArena.reset(marker);
    }
  });
}

/// Alias for [unravel_index].
List<NDArray<Int64>> unravelIndex<T extends DTypeTag>(
  NDArray<T> indices,
  List<int> shape, {
  IndexOrder order = IndexOrder.c,
  List<NDArray<Int64>>? out,
}) => unravel_index(indices, shape, order: order, out: out);

/// Converts a list of coordinate arrays into an array of flat indices.
///
/// The coordinate arrays in [multi_index] are broadcast to a common shape,
/// and each tuple of coordinates is mapped to a 64-bit flat index into an
/// array of dimensions [dims].
///
/// Parameters:
/// - `multi_index`: A non-empty list of integer [NDArray]s, one for each
///   dimension in [dims].
/// - `dims`: The dimensions of the array into which the indices from
///   [multi_index] apply. Every dimension must be positive, and their product
///   must fit in a 64-bit signed integer.
/// - `mode`: Specifies how out-of-bounds indices are handled. May be either a
///   single [ChooseMode] applied to all dimensions or a `List<ChooseMode>` of
///   length `dims.length` specifying a mode per dimension. Defaults to
///   [ChooseMode.raise].
/// - `order`: Determines whether the multi-indices should be viewed as
///   indexing in row-major ([IndexOrder.c], default) or column-major
///   ([IndexOrder.f]) order.
/// - `out`: Optional pre-allocated [NDArray] of dtype [DType.int64] with shape
///   matching the broadcast shape of [multi_index].
///
/// Returns an [NDArray] of dtype [DType.int64] containing the flat indices.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
NDArray<Int64> ravel_multi_index(
  List<NDArray<DTypeTag>> multi_index,
  List<int> dims, {
  Object mode = ChooseMode.raise,
  IndexOrder order = IndexOrder.c,
  NDArray<Int64>? out,
}) {
  if (multi_index.isEmpty) {
    throw ArgumentError.value(multi_index, 'multi_index', 'Must not be empty');
  }
  if (multi_index.length != dims.length) {
    throw ArgumentError.value(
      multi_index.length,
      'multi_index',
      'Must have length equal to dims.length (${dims.length})',
    );
  }
  for (var i = 0; i < multi_index.length; i++) {
    final c = multi_index[i];
    if (c.isDisposed) {
      throw StateError(
        'Cannot execute ravel_multi_index with a disposed coordinate array at index $i.',
      );
    }
    if (!c.dtype.isInteger) {
      throw ArgumentError.value(
        c.dtype,
        'multi_index[$i]',
        'Must have an integer dtype',
      );
    }
  }

  var maxFlat = 1;
  for (var i = 0; i < dims.length; i++) {
    final d = dims[i];
    if (d <= 0) {
      throw ArgumentError.value(
        dims,
        'dims',
        'Must contain positive dimensions',
      );
    }
    if (maxFlat > 0x7fffffffffffffff ~/ d) {
      throw ArgumentError.value(
        dims,
        'dims',
        'Must have a product that fits in a 64-bit signed integer',
      );
    }
    maxFlat *= d;
  }

  final List<ChooseMode> modesList;
  if (mode is ChooseMode) {
    modesList = List<ChooseMode>.filled(dims.length, mode);
  } else if (mode is List<ChooseMode>) {
    if (mode.length != dims.length) {
      throw ArgumentError.value(
        mode.length,
        'mode',
        'Must have length equal to dims.length (${dims.length})',
      );
    }
    modesList = mode;
  } else {
    throw ArgumentError.value(
      mode,
      'mode',
      'Must be a ChooseMode or List<ChooseMode>',
    );
  }

  return NDArray.scope(() {
    final ndims = dims.length;
    final coordArrays = <NDArray<Int64>>[];
    for (var d = 0; d < ndims; d++) {
      final c = multi_index[d];
      if (identical(c.dtype, DType.uint64)) {
        final copy = c.astype(DType.int64);
        final dimSize = dims[d];
        final dimMode = modesList[d];
        for (var i = 0; i < copy.size; i++) {
          final raw = copy.getCellFlat(i);
          if (raw < 0) {
            switch (dimMode) {
              case ChooseMode.raise:
                throw RangeError(
                  'Coordinate out of bounds for array with dimensions $dims',
                );
              case ChooseMode.clip:
                copy.setCellFlat(i, dimSize - 1);
              case ChooseMode.wrap:
                final wrapped = BigInt.from(
                  raw,
                ).toUnsigned(64).remainder(BigInt.from(dimSize)).toInt();
                copy.setCellFlat(i, wrapped);
            }
          }
        }
        coordArrays.add(copy);
      } else if (identical(c.dtype, DType.int64)) {
        coordArrays.add(c as NDArray<Int64>);
      } else {
        coordArrays.add(c.astype(DType.int64));
      }
    }

    final targetShape = _broadcastMultiShapes(
      coordArrays.map((c) => c.shape).toList(),
    );

    if (out != null) {
      if (out.isDisposed) {
        throw StateError(
          'Cannot write ravel_multi_index result to a disposed out array.',
        );
      }
      validateOutBuffer(out);
      if (!out.isWriteable) {
        throw ArgumentError.value(out, 'out', 'Must be writeable');
      }
      if (out.dtype != DType.int64) {
        throw ArgumentError.value(out.dtype, 'out', 'Must have dtype int64');
      }
      if (!listEquals(out.shape, targetShape)) {
        throw ArgumentError.value(
          out.shape,
          'out',
          'Must match broadcast shape ($targetShape)',
        );
      }
    }

    final needsTemp =
        out != null &&
        (multi_index.any((c) => sharesMemory(c, out)) ||
            coordArrays.any((c) => sharesMemory(c, out)));
    final workOut = (out == null || needsTemp)
        ? NDArray<Int64>.create(targetShape, DType.int64)
        : out;

    final isContig =
        workOut.isContiguous &&
        coordArrays.every(
          (c) => listEquals(c.shape, targetShape) && c.isContiguous,
        );
    final targetRank = targetShape.length;
    final orderCode = order == IndexOrder.c ? 0 : 1;

    final marker = ScratchArena.marker;
    try {
      final coordsPtrs = ScratchArena.allocate<ffi.Pointer<ffi.Int64>>(
        ndims * ffi.sizeOf<ffi.Pointer<ffi.Int64>>(),
      );
      for (var d = 0; d < ndims; d++) {
        coordsPtrs[d] = coordArrays[d].pointer.cast<ffi.Int64>();
      }

      final dimsPtr = ScratchArena.copyInts(dims);
      final modesPtr = ScratchArena.allocate<ffi.Int>(
        ndims * ffi.sizeOf<ffi.Int>(),
      );
      for (var d = 0; d < ndims; d++) {
        modesPtr[d] = switch (modesList[d]) {
          ChooseMode.raise => 0,
          ChooseMode.wrap => 1,
          ChooseMode.clip => 2,
        };
      }

      ffi.Pointer<ffi.Int64> coordsStridesFlat = ffi.nullptr;
      ffi.Pointer<ffi.Int64> targetShapePtr = ffi.nullptr;
      ffi.Pointer<ffi.Int64> outStridesPtr = ffi.nullptr;
      if (!isContig) {
        final strideCount = ndims * (targetRank > 0 ? targetRank : 1);
        coordsStridesFlat = ScratchArena.allocate<ffi.Int64>(
          strideCount * ffi.sizeOf<ffi.Int64>(),
        );
        for (var d = 0; d < ndims; d++) {
          final c = coordArrays[d];
          final cShape = c.shape;
          final cStrides = c.strides;
          final offsetRank = targetRank - cShape.length;
          for (var r = 0; r < targetRank; r++) {
            final cAxis = r - offsetRank;
            coordsStridesFlat[d * targetRank + r] =
                (cAxis < 0 || cShape[cAxis] == 1) ? 0 : cStrides[cAxis];
          }
        }
        targetShapePtr = ScratchArena.copyInts(targetShape);
        outStridesPtr = ScratchArena.copyInts(workOut.strides);
      }

      final outPtr = workOut.pointer.cast<ffi.Int64>();
      final errValPtr = ScratchArena.allocate<ffi.Int64>(
        1 * ffi.sizeOf<ffi.Int64>(),
      );
      errValPtr.value = 0;

      final status = native_ravel_multi_index(
        coordsPtrs,
        coordsStridesFlat,
        targetShapePtr,
        targetRank,
        workOut.size,
        dimsPtr,
        modesPtr,
        ndims,
        orderCode,
        outPtr,
        outStridesPtr,
        isContig ? 1 : 0,
        errValPtr,
      );

      if (status == -2) {
        throw RangeError(
          'Coordinate ${errValPtr.value} is out of bounds for dimensions $dims',
        );
      }
      if (status == -3) {
        throw ArgumentError.value(
          dims,
          'dims',
          'Must not overflow 64-bit signed integer product',
        );
      }
      if (status == -4) {
        throw OutOfMemoryError();
      }
      if (status != 0) {
        throw StateError('native_ravel_multi_index failed with status $status');
      }

      if (out != null) {
        if (needsTemp) {
          workOut.copy(out: out);
        }
        return out;
      }
      return workOut.detachToParentScope();
    } finally {
      ScratchArena.reset(marker);
    }
  });
}

/// Alias for [ravel_multi_index].
NDArray<Int64> ravelMultiIndex(
  List<NDArray<DTypeTag>> multiIndex,
  List<int> dims, {
  Object mode = ChooseMode.raise,
  IndexOrder order = IndexOrder.c,
  NDArray<Int64>? out,
}) => ravel_multi_index(multiIndex, dims, mode: mode, order: order, out: out);

/// Returns an array representing the indices of a grid.
///
/// Computes an array where the subarrays contain index values `0, 1, ...`
/// varying only along the corresponding axis. For a grid with [dimensions]
/// `[d_0, ..., d_{N-1}]`, the returned array has shape
/// `[N, d_0, ..., d_{N-1}]`.
///
/// For the sparse open-grid representation (equivalent to `sparse=True` in
/// NumPy), use [sparse_indices].
///
/// Parameters:
/// - `dimensions`: The shape of the grid. Every dimension must be
///   non-negative.
/// - `dtype`: Data type of the result. Defaults to [DType.int64].
/// - `out`: Optional pre-allocated [NDArray] with shape
///   `[dimensions.length, ...dimensions]`.
///
/// Returns an [NDArray] of grid indices with shape
/// `[dimensions.length, ...dimensions]`.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
NDArray<T> indices<T extends DTypeTag>(
  List<int> dimensions, {
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  for (var i = 0; i < dimensions.length; i++) {
    if (dimensions[i] < 0) {
      throw ArgumentError.value(
        dimensions,
        'dimensions',
        'Must not contain negative dimensions',
      );
    }
  }

  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  final ndims = dimensions.length;
  final targetShape = <int>[ndims, ...dimensions];

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot write indices result to a disposed out array.');
    }
    validateOutBuffer(out);
    if (!out.isWriteable) {
      throw ArgumentError.value(out, 'out', 'Must be writeable');
    }
    if (out.dtype != resolvedDType) {
      throw ArgumentError.value(
        out.dtype,
        'out',
        'Must match resolved dtype ($resolvedDType)',
      );
    }
    if (!listEquals(out.shape, targetShape)) {
      throw ArgumentError.value(
        out.shape,
        'out',
        'Must match target shape ($targetShape)',
      );
    }
  }

  return NDArray.scope(() {
    final needsContigTemp =
        out != null &&
        (!out.isContiguous || !identical(resolvedDType, DType.int64));
    final int64Grid =
        (identical(resolvedDType, DType.int64) &&
            out != null &&
            !needsContigTemp)
        ? out as NDArray<Int64>
        : NDArray<Int64>.create(targetShape, DType.int64);

    var sliceSize = 1;
    for (final d in dimensions) {
      sliceSize *= d;
    }

    if (ndims > 0 && sliceSize > 0) {
      final marker = ScratchArena.marker;
      try {
        final dimsPtr = ScratchArena.copyInts(dimensions);
        final outPtr = int64Grid.pointer.cast<ffi.Int64>();
        final status = native_indices_int64(dimsPtr, ndims, sliceSize, outPtr);
        if (status == -4) {
          throw OutOfMemoryError();
        }
        if (status != 0) {
          throw StateError('native_indices_int64 failed with status $status');
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (identical(resolvedDType, DType.int64)) {
      if (out != null) {
        if (needsContigTemp) {
          (int64Grid as NDArray<T>).copy(out: out);
        }
        return out;
      }
      return (int64Grid as NDArray<T>).detachToParentScope();
    }

    final casted = int64Grid.astype(resolvedDType);
    if (out != null) {
      casted.copy(out: out);
      return out;
    }
    return casted.detachToParentScope();
  });
}

/// Returns a list of sparse coordinate arrays representing the indices of a
/// grid (equivalent to `np.indices(dimensions, sparse=True)`).
///
/// For a grid with [dimensions] `[d_0, ..., d_{N-1}]`, returns `N` arrays
/// where the `i`-th array has shape `[1, ..., d_i, ..., 1]` (of rank `N`)
/// with values `0, 1, ..., d_i - 1` along axis `i`.
///
/// Parameters:
/// - `dimensions`: The shape of the grid. Every dimension must be
///   non-negative.
/// - `dtype`: Data type of the returned coordinate arrays. Defaults to
///   [DType.int64].
///
/// Returns a `List<NDArray<T>>` of length `dimensions.length`.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
List<NDArray<T>> sparse_indices<T extends DTypeTag>(
  List<int> dimensions, {
  DType<T>? dtype,
}) {
  for (var i = 0; i < dimensions.length; i++) {
    if (dimensions[i] < 0) {
      throw ArgumentError.value(
        dimensions,
        'dimensions',
        'Must not contain negative dimensions',
      );
    }
  }
  final resolvedDType = dtype ?? DType.int64 as DType<T>;
  final ndims = dimensions.length;
  return NDArray.scope(() {
    final result = <NDArray<T>>[];
    for (var i = 0; i < ndims; i++) {
      final sparseShape = List<int>.filled(ndims, 1);
      sparseShape[i] = dimensions[i];
      final vec = NDArray<T>.arange(
        0,
        dimensions[i].toDouble(),
        dtype: resolvedDType,
      );
      result.add(vec.reshape(sparseShape).copy().detachToParentScope());
    }
    return result;
  });
}

/// Alias for [sparse_indices].
List<NDArray<T>> sparseIndices<T extends DTypeTag>(
  List<int> dimensions, {
  DType<T>? dtype,
}) => sparse_indices(dimensions, dtype: dtype);

/// Returns the indices to access the main diagonal of an array.
///
/// Returns a list of [ndim] 1-D coordinate arrays of dtype [DType.int64],
/// each containing `0, 1, ..., n - 1`, suitable for indexing the main
/// diagonal of an [ndim]-dimensional array of shape `[n, n, ..., n]`.
///
/// The [n] must be non-negative and [ndim] must be at least `1`.
///
/// Parameters:
/// - `n`: The size along each dimension of the array for which the diagonal
///   indices are returned.
/// - `ndim`: The number of dimensions (defaults to `2`).
///
/// Returns a `List<NDArray<Int64>>` of length [ndim], each of shape `[n]`.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
List<NDArray<Int64>> diag_indices(int n, {int ndim = 2}) {
  if (n < 0) {
    throw ArgumentError.value(n, 'n', 'Must be non-negative');
  }
  if (ndim < 1) {
    throw ArgumentError.value(ndim, 'ndim', 'Must be at least 1');
  }
  return List<NDArray<Int64>>.generate(
    ndim,
    (_) => NDArray<Int64>.arange(0, n.toDouble(), dtype: DType.int64),
  );
}

/// Alias for [diag_indices].
List<NDArray<Int64>> diagIndices(int n, {int ndim = 2}) =>
    diag_indices(n, ndim: ndim);

/// Returns the indices to access the main diagonal of [arr].
///
/// The [arr] must be at least 2-dimensional and have equal length along all
/// dimensions.
///
/// Parameters:
/// - `arr`: The input [NDArray], which must be at least 2-D with equal-length
///   dimensions.
///
/// Returns a `List<NDArray<Int64>>` of length `arr.shape.length`.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
List<NDArray<Int64>> diag_indices_from<T extends DTypeTag>(NDArray<T> arr) {
  if (arr.isDisposed) {
    throw StateError('Cannot execute diag_indices_from on a disposed array.');
  }
  if (arr.shape.length < 2) {
    throw ArgumentError.value(
      arr.shape,
      'arr',
      'Must be at least 2-dimensional',
    );
  }
  final n = arr.shape[0];
  for (var i = 1; i < arr.shape.length; i++) {
    if (arr.shape[i] != n) {
      throw ArgumentError.value(
        arr.shape,
        'arr',
        'Must have equal length along all dimensions',
      );
    }
  }
  return diag_indices(n, ndim: arr.shape.length);
}

/// Alias for [diag_indices_from].
List<NDArray<Int64>> diagIndicesFrom<T extends DTypeTag>(NDArray<T> arr) =>
    diag_indices_from(arr);

int _trilCount(int n, int m, int k) {
  if (n <= 0 || m <= 0) return 0;
  final negK = -k;
  final i0 = negK < 0 ? 0 : (negK > n ? n : negK);
  final limit = m - k - 1;
  final clampedLimit = limit < 0 ? 0 : (limit > n ? n : limit);
  final i1 = clampedLimit < i0 ? i0 : clampedLimit;
  final l1 = i1 - i0;
  final sum1 = l1 > 0 ? (l1 * ((i0 + k + 1) + (i1 + k))) ~/ 2 : 0;
  final l2 = n - i1;
  final sum2 = l2 * m;
  return sum1 + sum2;
}

/// Returns the indices for the lower-triangle of an `(n, m)` array.
///
/// Both [n] and [m] (when provided) must be non-negative.
///
/// Parameters:
/// - `n`: The row dimension of the arrays for which the returned indices will
///   be valid.
/// - `k`: Diagonal offset (defaults to `0`, the main diagonal; `k < 0` is
///   below and `k > 0` is above the main diagonal).
/// - `m`: The column dimension of the arrays for which the returned arrays
///   will be valid. By default, [m] is taken equal to [n].
///
/// Returns a record `({NDArray<Int64> row, NDArray<Int64> col})` of `(rowIndices, colIndices)`
/// in row-major order.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
({NDArray<Int64> row, NDArray<Int64> col}) tril_indices(
  int n, {
  int k = 0,
  int? m,
}) {
  if (n < 0) {
    throw ArgumentError.value(n, 'n', 'Must be non-negative');
  }
  final cols = m ?? n;
  if (cols < 0) {
    throw ArgumentError.value(m, 'm', 'Must be non-negative');
  }
  final count = _trilCount(n, cols, k);
  return NDArray.scope(() {
    final rowArr = NDArray<Int64>.create([count], DType.int64);
    final colArr = NDArray<Int64>.create([count], DType.int64);
    if (count > 0) {
      final rowPtr = rowArr.pointer.cast<ffi.Int64>();
      final colPtr = colArr.pointer.cast<ffi.Int64>();
      final status = native_tril_indices(n, cols, k, rowPtr, colPtr);
      if (status == -4) {
        throw OutOfMemoryError();
      }
      if (status != 0) {
        throw StateError('native_tril_indices failed with status $status');
      }
    }
    return (
      row: rowArr.detachToParentScope(),
      col: colArr.detachToParentScope(),
    );
  });
}

/// Alias for [tril_indices].
({NDArray<Int64> row, NDArray<Int64> col}) trilIndices(
  int n, {
  int k = 0,
  int? m,
}) => tril_indices(n, k: k, m: m);

/// Returns the indices for the lower-triangle of [arr].
///
/// The [arr] must be a 2-dimensional array.
///
/// Parameters:
/// - `arr`: A 2-dimensional [NDArray].
/// - `k`: Diagonal offset (defaults to `0`).
///
/// Returns a record `({NDArray<Int64> row, NDArray<Int64> col})` of `(rowIndices, colIndices)`.
({NDArray<Int64> row, NDArray<Int64> col})
tril_indices_from<T extends DTypeTag>(NDArray<T> arr, {int k = 0}) {
  if (arr.isDisposed) {
    throw StateError('Cannot execute tril_indices_from on a disposed array.');
  }
  if (arr.shape.length != 2) {
    throw ArgumentError.value(arr.shape, 'arr', 'Must be 2-dimensional');
  }
  return tril_indices(arr.shape[0], k: k, m: arr.shape[1]);
}

/// Alias for [tril_indices_from].
({NDArray<Int64> row, NDArray<Int64> col}) trilIndicesFrom<T extends DTypeTag>(
  NDArray<T> arr, {
  int k = 0,
}) => tril_indices_from(arr, k: k);

/// Returns the indices for the upper-triangle of an `(n, m)` array.
///
/// Both [n] and [m] (when provided) must be non-negative.
///
/// Parameters:
/// - `n`: The row dimension of the arrays for which the returned indices will
///   be valid.
/// - `k`: Diagonal offset (defaults to `0`, the main diagonal; `k < 0` is
///   below and `k > 0` is above the main diagonal).
/// - `m`: The column dimension of the arrays for which the returned arrays
///   will be valid. By default, [m] is taken equal to [n].
///
/// Returns a record `({NDArray<Int64> row, NDArray<Int64> col})` of `(rowIndices, colIndices)`
/// in row-major order.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
({NDArray<Int64> row, NDArray<Int64> col}) triu_indices(
  int n, {
  int k = 0,
  int? m,
}) {
  if (n < 0) {
    throw ArgumentError.value(n, 'n', 'Must be non-negative');
  }
  final cols = m ?? n;
  if (cols < 0) {
    throw ArgumentError.value(m, 'm', 'Must be non-negative');
  }
  final count = n * cols - _trilCount(n, cols, k - 1);
  return NDArray.scope(() {
    final rowArr = NDArray<Int64>.create([count], DType.int64);
    final colArr = NDArray<Int64>.create([count], DType.int64);
    if (count > 0) {
      final rowPtr = rowArr.pointer.cast<ffi.Int64>();
      final colPtr = colArr.pointer.cast<ffi.Int64>();
      final status = native_triu_indices(n, cols, k, rowPtr, colPtr);
      if (status == -4) {
        throw OutOfMemoryError();
      }
      if (status != 0) {
        throw StateError('native_triu_indices failed with status $status');
      }
    }
    return (
      row: rowArr.detachToParentScope(),
      col: colArr.detachToParentScope(),
    );
  });
}

/// Alias for [triu_indices].
({NDArray<Int64> row, NDArray<Int64> col}) triuIndices(
  int n, {
  int k = 0,
  int? m,
}) => triu_indices(n, k: k, m: m);

/// Returns the indices for the upper-triangle of [arr].
///
/// The [arr] must be a 2-dimensional array.
///
/// Parameters:
/// - `arr`: A 2-dimensional [NDArray].
/// - `k`: Diagonal offset (defaults to `0`).
///
/// Returns a record `({NDArray<Int64> row, NDArray<Int64> col})` of `(rowIndices, colIndices)`.
({NDArray<Int64> row, NDArray<Int64> col})
triu_indices_from<T extends DTypeTag>(NDArray<T> arr, {int k = 0}) {
  if (arr.isDisposed) {
    throw StateError('Cannot execute triu_indices_from on a disposed array.');
  }
  if (arr.shape.length != 2) {
    throw ArgumentError.value(arr.shape, 'arr', 'Must be 2-dimensional');
  }
  return triu_indices(arr.shape[0], k: k, m: arr.shape[1]);
}

/// Alias for [triu_indices_from].
({NDArray<Int64> row, NDArray<Int64> col}) triuIndicesFrom<T extends DTypeTag>(
  NDArray<T> arr, {
  int k = 0,
}) => triu_indices_from(arr, k: k);

/// Returns the indices to access `(n, n)` arrays, given a masking function.
///
/// Assume [mask_func] is a function that, for a square array `m` of size
/// `(n, n)` with a possible offset argument `k`, when called as
/// `mask_func(m, k: k)` returns a new 2-D array with zeros in certain
/// locations (functions like `triu` or `tril` match this signature). This
/// function returns the indices where the non-zero values would be located.
///
/// The [n] must be non-negative.
///
/// Parameters:
/// - `n`: The square matrix dimension `(n, n)`.
/// - `mask_func`: A function whose call signature matches `(m, {int k})`
///   (such as `triu` or `tril`) and returns a 2-dimensional mask [NDArray].
/// - `k`: An optional diagonal offset argument passed through to [mask_func].
///
/// Returns a record `({NDArray<Int64> row, NDArray<Int64> col})` of `(rowIndices, colIndices)`
/// corresponding to the non-zero positions of `mask_func(ones([n, n]), k: k)`.
///
/// Example:
/// {@example /example/indexing_example.dart lang=dart}
({NDArray<Int64> row, NDArray<Int64> col}) mask_indices<T extends DTypeTag>(
  int n,
  NDArray<T> Function(NDArray<Int64> m, {int k}) mask_func, {
  int k = 0,
}) {
  if (n < 0) {
    throw ArgumentError.value(n, 'n', 'Must be non-negative');
  }
  return NDArray.scope(() {
    final m = NDArray<Int64>.ones([n, n], DType.int64);
    final mask = mask_func(m, k: k);
    if (mask.isDisposed) {
      throw StateError('mask_func returned a disposed array.');
    }
    if (mask.shape.length != 2) {
      throw ArgumentError.value(
        mask.shape,
        'mask_func',
        'Must return a 2-dimensional mask array',
      );
    }
    final coords = nonzero(mask);
    return (
      row: coords[0].detachToParentScope(),
      col: coords[1].detachToParentScope(),
    );
  });
}

/// Alias for [mask_indices].
({NDArray<Int64> row, NDArray<Int64> col}) maskIndices<T extends DTypeTag>(
  int n,
  NDArray<T> Function(NDArray<Int64> m, {int k}) maskFunc, {
  int k = 0,
}) => mask_indices(n, maskFunc, k: k);
