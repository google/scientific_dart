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

import '../ndarray.dart';

import 'dart:ffi' as ffi;

import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'spacers.dart';
import 'broadcasting.dart';

int mapSortKind(SortKind kind) {
  switch (kind) {
    case SortKind.quicksort:
      return 0;
    case SortKind.mergesort:
    case SortKind.stable:
      return 1;
    case SortKind.heapsort:
      return 2;
  }
}

DType resolveDType(DType a, DType b) {
  if (a == DType.boolean && b == DType.boolean) return DType.uint8;
  if (a == b) return a;
  if (a == DType.boolean) return b;
  if (b == DType.boolean) return a;

  // Complex promotion
  if (a == DType.complex128 || b == DType.complex128) return DType.complex128;
  if (a == DType.complex64 || b == DType.complex64) {
    final other = (a == DType.complex64) ? b : a;
    if (other == DType.float64 ||
        other == DType.int64 ||
        other == DType.uint64 ||
        other == DType.complex128) {
      return DType.complex128;
    }
    return DType.complex64;
  }

  // Floating point promotion
  if (a == DType.float64 || b == DType.float64) return DType.float64;
  if (a == DType.float32 || b == DType.float32) {
    final other = (a == DType.float32) ? b : a;
    if (other == DType.int64 ||
        other == DType.uint64 ||
        other == DType.int32 ||
        other == DType.uint32) {
      return DType.float64;
    }
    return DType.float32;
  }
  if ((a == DType.float16 && b == DType.bfloat16) ||
      (a == DType.bfloat16 && b == DType.float16)) {
    return DType.float32;
  }
  if (a == DType.float16 || b == DType.float16) {
    final other = (a == DType.float16) ? b : a;
    if (other == DType.int64 || other == DType.uint64) return DType.float64;
    if (other == DType.int32 ||
        other == DType.uint32 ||
        other == DType.int16 ||
        other == DType.uint16) {
      return DType.float32;
    }
    return DType.float16;
  }
  if (a == DType.bfloat16 || b == DType.bfloat16) {
    final other = (a == DType.bfloat16) ? b : a;
    if (other == DType.int64 || other == DType.uint64) return DType.float64;
    if (other == DType.int32 ||
        other == DType.uint32 ||
        other == DType.int16 ||
        other == DType.uint16) {
      return DType.float32;
    }
    return DType.bfloat16;
  }

  // Integer promotions
  final isASigned =
      a == DType.int64 ||
      a == DType.int32 ||
      a == DType.int16 ||
      a == DType.int8;
  final isBSigned =
      b == DType.int64 ||
      b == DType.int32 ||
      b == DType.int16 ||
      b == DType.int8;

  if (isASigned && isBSigned) {
    final maxBytes = math.max(a.byteWidth, b.byteWidth);
    if (maxBytes >= 8) return DType.int64;
    if (maxBytes >= 4) return DType.int32;
    if (maxBytes >= 2) return DType.int16;
    return DType.int8;
  }

  if (!isASigned && !isBSigned) {
    final maxBytes = math.max(a.byteWidth, b.byteWidth);
    if (maxBytes >= 8) return DType.uint64;
    if (maxBytes >= 4) return DType.uint32;
    if (maxBytes >= 2) return DType.uint16;
    return DType.uint8;
  }

  // Mixed signed and unsigned
  final signed = isASigned ? a : b;
  final unsigned = isASigned ? b : a;

  if (signed.byteWidth > unsigned.byteWidth) {
    return signed;
  }
  if (unsigned.byteWidth == 1) return DType.int16;
  if (unsigned.byteWidth == 2) return DType.int32;
  if (unsigned.byteWidth == 4) return DType.int64;
  return DType.float64;
}

DType<T> defaultDType<T>() {
  if (T == Complex) return DType.complex128 as DType<T>;
  if (T == int) return DType.int64 as DType<T>;
  if (T == bool) return DType.boolean as DType<T>;
  return DType.float64 as DType<T>;
}

Object normalizeScalar(Object o, DType dtype) {
  switch (dtype) {
    case DType.complex64:
    case DType.complex128:
      if (o is Complex) return o;
      if (o is num) return Complex(o.toDouble(), 0.0);
      if (o is bool) return Complex(o ? 1.0 : 0.0, 0.0);
      return Complex((o as dynamic).toDouble() as double, 0.0);
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      if (o is num) return o.toDouble();
      if (o is bool) return o ? 1.0 : 0.0;
      if (o is Complex) return o.real;
      return (o as dynamic).toDouble() as double;
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      if (o is num) return o.toInt();
      if (o is bool) return o ? 1 : 0;
      if (o is Complex) return o.real.toInt();
      return (o as dynamic).toInt() as int;
    case DType.boolean:
      if (o is bool) return o;
      if (o is num) return o != 0;
      if (o is Complex) return o.real != 0 || o.imag != 0;
      return o != 0;
  }
}

NDArray<T> toNDArray<T>(Object o, DType<T> dtype) {
  if (o is NDArray) {
    if (o.isDisposed) {
      throw StateError('Cannot convert a disposed NDArray to NDArray.');
    }
    if (o.dtype == dtype) return o as NDArray<T>;
    return castNDArray(o, dtype);
  }
  final normalized = normalizeScalar(o, dtype);
  return NDArray<T>.scalar(normalized as T, dtype: dtype);
}

({NDArray<T> samples, T step}) linspaceInternal<T>(
  T start,
  T stop,
  int numSamples, {
  bool endpoint = true,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (numSamples < 0) throw ArgumentError('numSamples must be non-negative');

  final resolvedDType = dtype ?? defaultDType<T>();

  if (numSamples == 0) {
    final arr = NDArray<T>.create([0], resolvedDType);
    final step = normalizeScalar(double.nan, resolvedDType) as T;
    return (samples: arr, step: step);
  }

  final div = endpoint ? (numSamples - 1) : numSamples;
  if (out != null) {
    if (out.isDisposed) throw StateError('Cannot write to disposed out array');
    if (!listEquals(out.shape, [numSamples]) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out array shape or dtype');
    }
  }
  final arr = out ?? NDArray<T>.create([numSamples], resolvedDType);
  T step;

  switch (resolvedDType) {
    case DType.float64:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_double(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.float32:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_float(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.complex128:
      final s = normalizeScalar(start as Object, DType.complex128) as Complex;
      final e = normalizeScalar(stop as Object, DType.complex128) as Complex;
      final stp = numSamples <= 1 ? Complex(0, 0) : (e - s) / div;
      v_linspace_complex128(
        arr.pointer.cast(),
        s.real,
        s.imag,
        stp.real,
        stp.imag,
        numSamples,
      );
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.complex64:
      final s = normalizeScalar(start as Object, DType.complex128) as Complex;
      final e = normalizeScalar(stop as Object, DType.complex128) as Complex;
      final stp = numSamples <= 1 ? Complex(0, 0) : (e - s) / div;
      v_linspace_complex64(
        arr.pointer.cast(),
        s.real,
        s.imag,
        stp.real,
        stp.imag,
        numSamples,
      );
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.int64:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_int64(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.int32:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_int32(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.int16:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_int16(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.uint8:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_uint8(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.float16:
    case DType.bfloat16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      final temp = NDArray<Float64>.create([numSamples], DType.float64);
      v_linspace_double(temp.pointer.cast(), s, stp, numSamples);
      final casted = castNDArray(temp, resolvedDType);
      casted.copy(out: arr);
      temp.dispose();
      casted.dispose();
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.boolean:
      throw UnsupportedError('linspace not supported for boolean arrays');
  }

  return (samples: arr, step: step);
}

void elementWiseOp<Ta, Tb, Tr>(
  NDArray<Tr> result,
  NDArray<Ta> a,
  NDArray<Tb> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  Tr Function(Ta, Tb) op, [
  ffi.Pointer<ffi.Uint8>? whereMask,
  int flatIndex = 0,
]) {
  if (dim == shape.length) {
    if (whereMask == null ||
        whereMask == ffi.nullptr ||
        whereMask[flatIndex] != 0) {
      result.setCellRaw(
        offsetResult,
        op(a.getCellRaw(offsetA), b.getCellRaw(offsetB)),
      );
    }
    return;
  }

  var currentFlat = flatIndex;
  final strideNext = dim + 1 < shape.length
      ? shape.sublist(dim + 1).reduce((x, y) => x * y)
      : 1;
  final limit = shape[dim];
  final strideA = stridesA[dim];
  final strideB = stridesB[dim];
  final strideResult = stridesResult[dim];

  for (var i = 0; i < limit; i++) {
    elementWiseOp<Ta, Tb, Tr>(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * strideA,
      offsetB + i * strideB,
      offsetResult + i * strideResult,
      op,
      whereMask,
      currentFlat,
    );
    currentFlat += strideNext;
  }
}

/// Computes the broadcasted shape for the batch/stack dimensions of two arrays.
List<int> broadcastStackShapes(List<int> sA, List<int> sB) {
  final lenA = sA.length;
  final lenB = sB.length;
  final maxLen = math.max(lenA, lenB);
  final result = List<int>.filled(maxLen, 0);

  for (var i = 0; i < maxLen; i++) {
    final dimA = (lenA - 1 - i >= 0) ? sA[lenA - 1 - i] : 1;
    final dimB = (lenB - 1 - i >= 0) ? sB[lenB - 1 - i] : 1;

    if (dimA == dimB) {
      result[maxLen - 1 - i] = dimA;
    } else if (dimA == 1) {
      result[maxLen - 1 - i] = dimB;
    } else if (dimB == 1) {
      result[maxLen - 1 - i] = dimA;
    } else {
      throw ArgumentError(
        'Incompatible stack shapes for broadcasting in matmul: $sA and $sB',
      );
    }
  }
  return result;
}

int encodeDType(DType type) => type.index;

NDArray<Float64> promoteToDouble(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute promoteToDouble on a disposed array.');
  }
  final res = NDArray<Float64>.create(a.shape, DType.float64);
  final ndim = a.shape.length;
  final marker = ScratchArena.marker;

  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesSrc = cBuffer + ndim;

    for (var i = 0; i < ndim; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    s_cast_generic(
      a.pointer.cast(),
      cStridesSrc,
      encodeDType(a.dtype),
      res.pointer.cast(),
      encodeDType(DType.float64),
      cShape,
      ndim,
    );
  } finally {
    ScratchArena.reset(marker);
  }
  return res;
}

NDArray<Complex> promoteToComplex(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute promoteToComplex on a disposed array.');
  }
  final res = NDArray<Complex>.create(a.shape, DType.complex128);
  final ndim = a.shape.length;
  final marker = ScratchArena.marker;

  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesSrc = cBuffer + ndim;

    for (var i = 0; i < ndim; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    s_cast_generic(
      a.pointer.cast(),
      cStridesSrc,
      encodeDType(a.dtype),
      res.pointer.cast(),
      encodeDType(DType.complex128),
      cShape,
      ndim,
    );
  } finally {
    ScratchArena.reset(marker);
  }
  return res;
}

/// Recursive helper to accumulate sum and count of non-NaN elements along an axis.
void nanReduceRecursive<T>(
  NDArray<T> a,
  NDArray<T> result,
  NDArray<int> counts,
  List<int> coordA,
  List<int> coordRes,
  int axis,
  int dim, {
  bool keepdims = false,
}) {
  if (dim == a.shape.length) {
    final val = a.getCell(coordA);
    if (val is double && val.isNaN) return;
    if (val is Complex && (val.real.isNaN || val.imag.isNaN)) return;
    final current = result.getCell(coordRes);
    result.setCell(coordRes, ((current as dynamic) + val) as T);
    counts.setCell(coordRes, counts.getCell(coordRes) + 1);
    return;
  }
  if (dim == axis) {
    if (keepdims) coordRes[axis] = 0;
    for (var i = 0; i < a.shape[axis]; i++) {
      coordA[dim] = i;
      nanReduceRecursive<T>(
        a,
        result,
        counts,
        coordA,
        coordRes,
        axis,
        dim + 1,
        keepdims: keepdims,
      );
    }
  } else {
    final resDim = keepdims ? dim : (dim < axis ? dim : dim - 1);
    for (var i = 0; i < a.shape[dim]; i++) {
      coordA[dim] = i;
      coordRes[resDim] = i;
      nanReduceRecursive<T>(
        a,
        result,
        counts,
        coordA,
        coordRes,
        axis,
        dim + 1,
        keepdims: keepdims,
      );
    }
  }
}

/// Recursive helper to traverse the leading stack dimensions of a multi-dimensional array.
///
/// Generates multidimensional coordinates of the stack/batch dimensions.
void walkStackCoords(
  List<int> stackShape,
  List<int> currentCoords,
  int dim,
  void Function(List<int> coords) leafCallback,
) {
  if (dim == stackShape.length) {
    leafCallback(currentCoords);
    return;
  }
  final limit = stackShape[dim];
  for (var i = 0; i < limit; i++) {
    currentCoords[dim] = i;
    walkStackCoords(stackShape, currentCoords, dim + 1, leafCallback);
  }
}

/// Recursive helper to traverse and reduce an array along an axis.
void reduceRecursive<S extends Object, D extends Object>(
  NDArray<S> src,
  NDArray<D> dest,
  List<int> currentPos,
  List<int> destPos,
  int targetAxis,
  int currentDim,
  D Function(D acc, S val) op, {
  List<int>? destStrides,
}) {
  if (currentDim == src.shape.length) {
    // Calculate flat index for src
    var srcOffset = src.offsetElements;
    for (var i = 0; i < src.shape.length; i++) {
      srcOffset += currentPos[i] * src.strides[i];
    }

    // Calculate flat index for dest
    var destOffset = dest.offsetElements;
    final strides = destStrides ?? dest.strides;
    for (var i = 0; i < destPos.length; i++) {
      destOffset += destPos[i] * strides[i];
    }

    dest.setCellRaw(
      destOffset,
      op(dest.getCellRaw(destOffset), src.getCellRaw(srcOffset)),
    );
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    currentPos[currentDim] = i;
    if (currentDim < targetAxis) {
      destPos[currentDim] = i;
    } else if (currentDim > targetAxis) {
      destPos[currentDim - 1] = i;
    }
    reduceRecursive<S, D>(
      src,
      dest,
      currentPos,
      destPos,
      targetAxis,
      currentDim + 1,
      op,
      destStrides: destStrides,
    );
  }
}

void unaryOp<Ta, Tr>(
  NDArray<Tr> result,
  NDArray<Ta> a,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetResult,
  Tr Function(Ta) op, [
  ffi.Pointer<ffi.Uint8>? whereMask,
  int flatIndex = 0,
]) {
  if (dim == shape.length) {
    if (whereMask == null ||
        whereMask == ffi.nullptr ||
        whereMask[flatIndex] != 0) {
      result.setCellRaw(offsetResult, op(a.getCellRaw(offsetA)));
    }
    return;
  }

  var currentFlat = flatIndex;
  final strideNext = dim + 1 < shape.length
      ? shape.sublist(dim + 1).reduce((x, y) => x * y)
      : 1;
  for (var i = 0; i < shape[dim]; i++) {
    unaryOp<Ta, Tr>(
      result,
      a,
      shape,
      stridesA,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetResult + i * stridesResult[dim],
      op,
      whereMask,
      currentFlat,
    );
    currentFlat += strideNext;
  }
}

void ternaryOp<Ta, Tb, Tc, Tr>(
  NDArray<Tr> result,
  NDArray<Ta> a,
  NDArray<Tb> b,
  NDArray<Tc> c,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesC,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetC,
  int offsetResult,
  Tr Function(Ta, Tb, Tc) op, [
  ffi.Pointer<ffi.Uint8>? whereMask,
  int flatIndex = 0,
]) {
  if (dim == shape.length) {
    if (whereMask == null ||
        whereMask == ffi.nullptr ||
        whereMask[flatIndex] != 0) {
      result.setCellRaw(
        offsetResult,
        op(a.getCellRaw(offsetA), b.getCellRaw(offsetB), c.getCellRaw(offsetC)),
      );
    }
    return;
  }

  var currentFlat = flatIndex;
  final strideNext = dim + 1 < shape.length
      ? shape.sublist(dim + 1).reduce((x, y) => x * y)
      : 1;
  final limit = shape[dim];
  final strideA = stridesA[dim];
  final strideB = stridesB[dim];
  final strideC = stridesC[dim];
  final strideResult = stridesResult[dim];

  for (var i = 0; i < limit; i++) {
    ternaryOp<Ta, Tb, Tc, Tr>(
      result,
      a,
      b,
      c,
      shape,
      stridesA,
      stridesB,
      stridesC,
      stridesResult,
      dim + 1,
      offsetA + i * strideA,
      offsetB + i * strideB,
      offsetC + i * strideC,
      offsetResult + i * strideResult,
      op,
      whereMask,
      currentFlat,
    );
    currentFlat += strideNext;
  }
}

bool isTrueHelper(dynamic x) {
  if (x is bool) {
    return x;
  } else if (x is Complex) {
    return x.real != 0.0 || x.imag != 0.0;
  } else if (x is num) {
    return x != 0;
  }
  return false;
}

List<int> broadcastStrides(NDArray a, List<int> targetShape) {
  final strides = List<int>.filled(targetShape.length, 0);
  final offset = targetShape.length - a.shape.length;
  for (var i = 0; i < a.shape.length; i++) {
    final targetDim = targetShape[i + offset];
    final aDim = a.shape[i];
    if (aDim == targetDim) {
      strides[i + offset] = a.strides[i];
    } else if (aDim == 1) {
      strides[i + offset] = 0;
    } else {
      throw ArgumentError('Cannot broadcast shape ${a.shape} to $targetShape');
    }
  }
  return strides;
}

List<int> broadcast3Shapes(List<int> s1, List<int> s2, List<int> s3) {
  final len = math.max(s1.length, math.max(s2.length, s3.length));
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;
    final dim3 = s3.length - 1 - i >= 0 ? s3[s3.length - 1 - i] : 1;

    var target = 1;
    for (final d in [dim1, dim2, dim3]) {
      if (d != 1) {
        if (target == 1) {
          target = d;
        } else if (target != d) {
          throw ArgumentError('Incompatible shapes for broadcasting');
        }
      }
    }
    common[len - 1 - i] = target;
  }
  return common;
}

dynamic castValue(dynamic val, DType dtype) {
  switch (dtype) {
    case DType.complex128:
    case DType.complex64:
      if (val is Complex) return val;
      if (val is num) return Complex(val.toDouble(), 0.0);
      return Complex(0.0, 0.0);
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      if (val is num) return val.toDouble();
      if (val is Complex) return val.real;
      if (val is bool) return val ? 1.0 : 0.0;
      return 0.0;
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      if (val is num) return val.toInt();
      if (val is Complex) return val.real.toInt();
      if (val is bool) return val ? 1 : 0;
      return 0;
    case DType.boolean:
      if (val is bool) return val;
      if (val is num) return val != 0;
      if (val is Complex) return val.real != 0.0 || val.imag != 0.0;
      return false;
  }
}

enum CumOpType { sum, prod, min, max }

NDArray<R> cumOpFFI<T, R>(
  NDArray<T> a,
  int axis,
  NDArray<R> result,
  CumOpType opType,
) {
  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  final cShape = ScratchArena.copyInts(a.shape);
  final cStridesA = ScratchArena.copyInts(a.strides);
  final cStridesRes = ScratchArena.copyInts(result.strides);

  try {
    switch (opType) {
      case CumOpType.sum:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cumsum_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cumsum_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cumsum_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cumsum_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex128:
            s_cumsum_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex64:
            s_cumsum_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float16:
          case DType.bfloat16:
          case DType.int16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            _cumOpFallbackHelper(a, result, axis, s_cumsum_double);
        }

      case CumOpType.prod:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cumprod_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cumprod_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cumprod_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cumprod_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex128:
            s_cumprod_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex64:
            s_cumprod_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float16:
          case DType.bfloat16:
          case DType.int16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            _cumOpFallbackHelper(a, result, axis, s_cumprod_double);
        }

      case CumOpType.min:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cummin_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cummin_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cummin_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cummin_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float16:
          case DType.bfloat16:
          case DType.int16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            _cumOpFallbackHelper(a, result, axis, s_cummin_double);
          case DType.complex128:
          case DType.complex64:
            throw ArgumentError(
              'Cumulative minimum is not defined for complex numbers.',
            );
        }

      case CumOpType.max:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cummax_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cummax_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cummax_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cummax_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float16:
          case DType.bfloat16:
          case DType.int16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            _cumOpFallbackHelper(a, result, axis, s_cummax_double);
          case DType.complex128:
          case DType.complex64:
            throw ArgumentError(
              'Cumulative maximum is not defined for complex numbers.',
            );
        }
    }
  } finally {
    ScratchArena.reset(marker);
  }
  return result;
}

NDArray<R> castNDArray<R>(NDArray a, DType<R> targetDType) {
  if (a.dtype == targetDType && a is NDArray<R>) return a;
  final list = a.toList();
  if (targetDType.isFloating) {
    final doubleList = list.map((e) {
      if (e is num) return e.toDouble();
      if (e is bool) return e ? 1.0 : 0.0;
      if (e is Complex) return e.real;
      return (e as dynamic).toDouble() as double;
    }).toList();
    return NDArray<R>.fromList(doubleList, a.shape, targetDType);
  } else if (targetDType.isInteger) {
    final intList = list.map((e) {
      if (e is num) return e.toInt();
      if (e is bool) return e ? 1 : 0;
      if (e is Complex) return e.real.toInt();
      return (e as dynamic).toInt() as int;
    }).toList();
    return NDArray<R>.fromList(intList, a.shape, targetDType);
  } else if (targetDType == DType.boolean) {
    final boolList = list.map((e) {
      if (e is bool) return e;
      if (e is num) return e != 0;
      if (e is Complex) return e.real != 0 || e.imag != 0;
      return e != 0;
    }).toList();
    return NDArray<R>.fromList(boolList, a.shape, targetDType);
  } else if (targetDType.isComplex) {
    final cpxList = list.map((e) {
      if (e is Complex) return e;
      if (e is num) return Complex(e.toDouble(), 0.0);
      if (e is bool) return Complex(e ? 1.0 : 0.0, 0.0);
      return Complex((e as dynamic).toDouble() as double, 0.0);
    }).toList();
    return NDArray<R>.fromList(cpxList, a.shape, targetDType);
  }
  return NDArray<R>.fromList(list.cast<R>(), a.shape, targetDType);
}

void _cumOpFallbackHelper<T, R>(
  NDArray<T> a,
  NDArray<R> result,
  int axis,
  void Function(
    ffi.Pointer<ffi.Double> src,
    ffi.Pointer<ffi.Int> srcStrides,
    ffi.Pointer<ffi.Double> dest,
    ffi.Pointer<ffi.Int> destStrides,
    ffi.Pointer<ffi.Int> shape,
    int rank,
    int axis,
  )
  ffiFunc,
) {
  NDArray.scope(() {
    final doubleA = castNDArray(a, DType.float64);
    final doubleRes = NDArray<Float64>.create(doubleA.shape, DType.float64);
    final marker = ScratchArena.marker;
    try {
      final cStridesDoubleA = ScratchArena.copyInts(doubleA.strides);
      final cStridesDoubleRes = ScratchArena.copyInts(doubleRes.strides);
      final cShape = ScratchArena.copyInts(doubleA.shape);

      ffiFunc(
        doubleA.pointer.cast(),
        cStridesDoubleA,
        doubleRes.pointer.cast(),
        cStridesDoubleRes,
        cShape,
        doubleA.shape.length,
        axis,
      );
    } finally {
      ScratchArena.reset(marker);
    }

    final convertedRes = castNDArray<R>(doubleRes, result.dtype);
    convertedRes.copy(out: result);
  });
}

/// A holder for a mask FFI pointer and any transient allocated arrays.
///
/// Wraps an FFI pointer ([pointer]) and manages disposal of intermediate
/// broadcasted or contiguous mask allocations via [dispose].
final class MaskHolder {
  /// The native memory pointer passed to FFI ufunc kernels.
  final ffi.Pointer<ffi.Uint8> pointer;

  /// Optional temporary array allocation requiring disposal after kernel execution.
  final NDArray<dynamic>? _tempAllocated;

  /// Creates a new [MaskHolder] with the given [pointer] and optional [_tempAllocated].
  MaskHolder(this.pointer, [this._tempAllocated]);

  /// Disposes any temporary array allocations created during mask preparation.
  void dispose() {
    _tempAllocated?.dispose();
  }
}

/// Prepares a [where] mask array for native elementwise ufunc execution.
///
/// Validates that [where] is not disposed, has boolean or uint8 data type, and
/// broadcasts [where] to [targetShape] if necessary.
///
/// - It is an error if [where] does not have [DType.boolean] or [DType.uint8].
/// - It is an error if [where] is disposed.
///
/// Returns a [MaskHolder] containing the native pointer and any transient allocations.
/// Time complexity is $O(1)$ for contiguous masks of matching shape, or $O(N)$
/// where $N$ is the element count when broadcasting or copying strided masks.
MaskHolder prepareMask(NDArray<dynamic>? where, List<int> targetShape) {
  if (where == null) {
    return MaskHolder(ffi.nullptr);
  }
  if (where.isDisposed) {
    throw StateError('Cannot execute operation with a disposed where array.');
  }
  if (where.dtype != DType.boolean && where.dtype != DType.uint8) {
    throw ArgumentError('where mask must have boolean or uint8 dtype.');
  }
  final aligned = listEquals(where.shape, targetShape)
      ? where
      : broadcastTo(where, targetShape);
  if (aligned.isContiguous) {
    return MaskHolder(
      aligned.pointer.cast(),
      !identical(aligned, where) ? aligned : null,
    );
  } else {
    final temp = aligned.copy();
    if (!identical(aligned, where)) {
      aligned.dispose();
    }
    return MaskHolder(temp.pointer.cast(), temp);
  }
}
