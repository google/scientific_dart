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

import 'dart:math' as math;
import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'spacers.dart';
import 'broadcasting.dart';

/// Throws [OutOfMemoryError] if a native C/C++ kernel signaled an allocation failure.
void checkNativeOom() {
  if (ndarray_consume_oom_flag() != 0) {
    throw OutOfMemoryError();
  }
}

/// Checks if two arrays share the same underlying memory buffer.
bool sharesMemory(NDArray x, NDArray y) {
  if (identical(x, y)) return true;
  if (x.pointer.address != 0 &&
      y.pointer.address != 0 &&
      x.size > 0 &&
      y.size > 0) {
    int minRel(NDArray arr) {
      int m = 0;
      for (int i = 0; i < arr.rank; i++) {
        if (arr.shape[i] > 1 && arr.strides[i] < 0) {
          m += (arr.shape[i] - 1) * arr.strides[i];
        }
      }
      return m;
    }

    int maxRel(NDArray arr) {
      int m = 0;
      for (int i = 0; i < arr.rank; i++) {
        if (arr.shape[i] > 1 && arr.strides[i] > 0) {
          m += (arr.shape[i] - 1) * arr.strides[i];
        }
      }
      return m;
    }

    final xStart = x.pointer.address + minRel(x) * x.dtype.byteWidth;
    final xEnd = x.pointer.address + (maxRel(x) + 1) * x.dtype.byteWidth;
    final yStart = y.pointer.address + minRel(y) * y.dtype.byteWidth;
    final yEnd = y.pointer.address + (maxRel(y) + 1) * y.dtype.byteWidth;
    if (xStart < yEnd && yStart < xEnd) return true;
  }
  return false;
}

bool _pointerOverlapsArray(
  ffi.Pointer<ffi.Uint8>? ptr,
  int byteLen,
  NDArray arr,
) {
  if (ptr == null ||
      ptr == ffi.nullptr ||
      byteLen <= 0 ||
      arr.pointer.address == 0 ||
      arr.size == 0) {
    return false;
  }
  int minRel = 0;
  int maxRel = 0;
  for (int i = 0; i < arr.rank; i++) {
    if (arr.shape[i] > 1) {
      if (arr.strides[i] < 0) {
        minRel += (arr.shape[i] - 1) * arr.strides[i];
      } else {
        maxRel += (arr.shape[i] - 1) * arr.strides[i];
      }
    }
  }
  final arrStart = arr.pointer.address + minRel * arr.dtype.byteWidth;
  final arrEnd = arr.pointer.address + (maxRel + 1) * arr.dtype.byteWidth;
  final ptrStart = ptr.address;
  final ptrEnd = ptr.address + byteLen;
  return ptrStart < arrEnd && arrStart < ptrEnd;
}

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

int saturatingDoubleToInt(double val, DType dtype) {
  if (val.isNaN) return 0;
  switch (dtype) {
    case DType.int8:
      if (val >= 127.0) return 127;
      if (val <= -128.0) return -128;
      return val.truncate();
    case DType.uint8:
      if (val >= 255.0) return 255;
      if (val <= 0.0) return 0;
      return val.truncate();
    case DType.int16:
      if (val >= 32767.0) return 32767;
      if (val <= -32768.0) return -32768;
      return val.truncate();
    case DType.uint16:
      if (val >= 65535.0) return 65535;
      if (val <= 0.0) return 0;
      return val.truncate();
    case DType.int32:
      if (val >= 2147483647.0) return 2147483647;
      if (val <= -2147483648.0) return -2147483648;
      return val.truncate();
    case DType.uint32:
      if (val >= 4294967295.0) return 4294967295;
      if (val <= 0.0) return 0;
      return val.truncate();
    case DType.int64:
      if (val >= 9223372036854775808.0) return 0x7FFFFFFFFFFFFFFF;
      if (val <= -9223372036854775808.0) return -0x8000000000000000;
      return val.truncate();
    case DType.uint64:
      if (val <= 0.0) return 0;
      if (val >= 18446744073709551616.0) return -1;
      if (val >= 9223372036854775808.0) {
        return BigInt.from(val).toSigned(64).toInt();
      }
      return val.truncate();
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
    case DType.complex128:
    case DType.complex64:
    case DType.boolean:
      return val.truncate();
  }
}

Object normalizeScalar(Object o, DType dtype) {
  switch (dtype) {
    case DType.complex64:
    case DType.complex128:
      if (o is Complex) return o;
      if (o is double && o.isNaN) return Complex(double.nan, double.nan);
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
      if (o is int) return o;
      if (o is double) return saturatingDoubleToInt(o, dtype);
      if (o is bool) return o ? 1 : 0;
      if (o is Complex) return saturatingDoubleToInt(o.real, dtype);
      if (o is num) return o.toInt();
      return (o as dynamic).toInt() as int;
    case DType.boolean:
      if (o is bool) return o;
      if (o is num) return o != 0;
      if (o is Complex) return o.real != 0 || o.imag != 0;
      return o != 0;
  }
}

NDArray<T> toNDArray<T extends DTypeTag>(Object o, DType<T> dtype) {
  if (o is NDArray) {
    if (o.isDisposed) {
      throw StateError('Cannot convert a disposed NDArray to NDArray.');
    }
    if (o.dtype == dtype) {
      if (o is NDArray<T>) return o;
      return NDArray<T>.view(o, shape: o.shape, strides: o.strides);
    }
    return castNDArray(o, dtype);
  }
  final normalized = normalizeScalar(o, dtype);
  return NDArray<T>.scalar(normalized, dtype: dtype);
}

({NDArray<T> samples, dynamic step}) linspaceInternal<T extends DTypeTag>(
  Object? start,
  Object? stop,
  int numSamples, {
  bool endpoint = true,
  required DType<T> dtype,
  NDArray<T>? out,
}) {
  if (numSamples < 0) throw ArgumentError('numSamples must be non-negative');

  final resolvedDType = dtype;

  if (out != null) {
    if (out.isDisposed) throw StateError('Cannot write to disposed out array');
    if (!listEquals(out.shape, [numSamples]) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out array shape or dtype');
    }
  }

  if (numSamples == 0) {
    final arr = out ?? NDArray<T>.create([0], resolvedDType);
    final step = normalizeScalar(double.nan, resolvedDType);
    return (samples: arr, step: step);
  }

  final div = endpoint ? (numSamples - 1) : numSamples;
  final bool useTempOut = out != null && !out.isContiguous;

  return NDArray.scope(() {
    final arr = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create([numSamples], resolvedDType);
    dynamic step;

    switch (resolvedDType) {
      case DType.float64:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_linspace_double(arr.pointer.cast(), s, stp, numSamples);
        if (endpoint && numSamples > 1) {
          arr.pointer.cast<ffi.Double>()[numSamples - 1] = e;
        }
        step = normalizeScalar(stp, resolvedDType);
      case DType.float32:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_linspace_float(arr.pointer.cast(), s, stp, numSamples);
        if (endpoint && numSamples > 1) {
          arr.pointer.cast<ffi.Float>()[numSamples - 1] = e;
        }
        step = normalizeScalar(stp, resolvedDType);
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
        if (endpoint && numSamples > 1) {
          arr.setCellRaw(numSamples - 1, e);
        }
        step = normalizeScalar(stp, resolvedDType);
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
        if (endpoint && numSamples > 1) {
          arr.setCellRaw(numSamples - 1, e);
        }
        step = normalizeScalar(stp, resolvedDType);
      case DType.int64:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_linspace_int64(arr.pointer.cast(), s, stp, numSamples);
        if (endpoint && numSamples > 1) {
          arr.setCellRaw(
            numSamples - 1,
            normalizeScalar(e.floor(), resolvedDType),
          );
        }
        step = normalizeScalar(stp, resolvedDType);
      case DType.int32:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_linspace_int32(arr.pointer.cast(), s, stp, numSamples);
        if (endpoint && numSamples > 1) {
          arr.setCellRaw(
            numSamples - 1,
            normalizeScalar(e.floor(), resolvedDType),
          );
        }
        step = normalizeScalar(stp, resolvedDType);
      case DType.int16:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_linspace_int16(arr.pointer.cast(), s, stp, numSamples);
        if (endpoint && numSamples > 1) {
          arr.setCellRaw(
            numSamples - 1,
            normalizeScalar(e.floor(), resolvedDType),
          );
        }
        step = normalizeScalar(stp, resolvedDType);
      case DType.uint8:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        v_linspace_uint8(arr.pointer.cast(), s, stp, numSamples);
        if (endpoint && numSamples > 1) {
          arr.setCellRaw(
            numSamples - 1,
            normalizeScalar(e.floor(), resolvedDType),
          );
        }
        step = normalizeScalar(stp, resolvedDType);
      case DType.float16:
      case DType.bfloat16:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        final temp = NDArray<Float64>.create([numSamples], DType.float64);
        v_linspace_double(temp.pointer.cast(), s, stp, numSamples);
        if (endpoint && numSamples > 1) {
          temp.pointer.cast<ffi.Double>()[numSamples - 1] = e;
        }
        final casted = castNDArray(temp, resolvedDType);
        casted.copy(out: arr);
        step = normalizeScalar(stp, resolvedDType);
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        final s = (start as num).toDouble();
        final e = (stop as num).toDouble();
        final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
        final temp = NDArray<Float64>.create([numSamples], DType.float64);
        v_linspace_double(temp.pointer.cast(), s, stp, numSamples);
        final tempPtr = temp.pointer.cast<ffi.Double>();
        for (var i = 0; i < numSamples; i++) {
          tempPtr[i] = tempPtr[i].floorToDouble();
        }
        if (endpoint && numSamples > 1) {
          tempPtr[numSamples - 1] = e.floorToDouble();
        }
        final casted = castNDArray(temp, resolvedDType);
        casted.copy(out: arr);
        step = normalizeScalar(stp, resolvedDType);
      case DType.boolean:
        throw UnsupportedError('linspace not supported for boolean arrays');
    }

    if (useTempOut) {
      arr.copy(out: out);
      return (samples: out, step: step);
    }
    if (out == null) {
      arr.detachToParentScope();
    }
    return (samples: arr, step: step);
  });
}

void
elementWiseOp<Ta extends DTypeTag, Tb extends DTypeTag, Tr extends DTypeTag>(
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
  dynamic Function(dynamic, dynamic) op, [
  ffi.Pointer<ffi.Uint8>? whereMask,
  int flatIndex = 0,
]) {
  if (dim == 0) {
    final totalElements = shape.fold<int>(1, (acc, v) => acc * v);
    if (totalElements > 0 &&
        (sharesMemory(a, result) ||
            sharesMemory(b, result) ||
            _pointerOverlapsArray(whereMask, totalElements, result))) {
      final tempOut = NDArray<Tr>.create(shape, result.dtype);
      try {
        if (whereMask != null && whereMask != ffi.nullptr) {
          unaryOp<Tr, Tr>(
            tempOut,
            result,
            shape,
            stridesResult,
            tempOut.strides,
            0,
            offsetResult,
            0,
            (v) => v,
          );
        }
        elementWiseOp<Ta, Tb, Tr>(
          tempOut,
          a,
          b,
          shape,
          stridesA,
          stridesB,
          tempOut.strides,
          0,
          offsetA,
          offsetB,
          0,
          op,
          whereMask,
          flatIndex,
        );
        unaryOp<Tr, Tr>(
          result,
          tempOut,
          shape,
          tempOut.strides,
          stridesResult,
          0,
          0,
          offsetResult,
          (v) => v,
        );
      } finally {
        tempOut.dispose();
      }
      return;
    }
  }
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
  checkTotalSize(result);
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

NDArray<DTypeTag> promoteToComplex(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute promoteToComplex on a disposed array.');
  }
  final res = NDArray<DTypeTag>.create(a.shape, DType.complex128);
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
void nanReduceRecursive<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> result,
  NDArray<DTypeTag> counts,
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
    result.setCell(coordRes, ((current as dynamic) + val));
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
void reduceRecursive<S extends DTypeTag, D extends DTypeTag>(
  NDArray<S> src,
  NDArray<D> dest,
  List<int> currentPos,
  List<int> destPos,
  int targetAxis,
  int currentDim,
  dynamic Function(dynamic acc, dynamic val) op, {
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

void unaryOp<Ta extends DTypeTag, Tr extends DTypeTag>(
  NDArray<Tr> result,
  NDArray<Ta> a,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetResult,
  dynamic Function(dynamic) op, [
  ffi.Pointer<ffi.Uint8>? whereMask,
  int flatIndex = 0,
]) {
  if (dim == 0) {
    final totalElements = shape.fold<int>(1, (acc, v) => acc * v);
    if (totalElements > 0 &&
        (sharesMemory(a, result) ||
            _pointerOverlapsArray(whereMask, totalElements, result))) {
      final tempOut = NDArray<Tr>.create(shape, result.dtype);
      try {
        if (whereMask != null && whereMask != ffi.nullptr) {
          unaryOp<Tr, Tr>(
            tempOut,
            result,
            shape,
            stridesResult,
            tempOut.strides,
            0,
            offsetResult,
            0,
            (v) => v,
          );
        }
        unaryOp<Ta, Tr>(
          tempOut,
          a,
          shape,
          stridesA,
          tempOut.strides,
          0,
          offsetA,
          0,
          op,
          whereMask,
          flatIndex,
        );
        unaryOp<Tr, Tr>(
          result,
          tempOut,
          shape,
          tempOut.strides,
          stridesResult,
          0,
          0,
          offsetResult,
          (v) => v,
        );
      } finally {
        tempOut.dispose();
      }
      return;
    }
  }
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

void ternaryOp<
  Ta extends DTypeTag,
  Tb extends DTypeTag,
  Tc extends DTypeTag,
  Tr extends DTypeTag
>(
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
  dynamic Function(dynamic, dynamic, dynamic) op, [
  ffi.Pointer<ffi.Uint8>? whereMask,
  int flatIndex = 0,
]) {
  if (dim == 0) {
    final totalElements = shape.fold<int>(1, (acc, v) => acc * v);
    if (totalElements > 0 &&
        (sharesMemory(a, result) ||
            sharesMemory(b, result) ||
            sharesMemory(c, result) ||
            _pointerOverlapsArray(whereMask, totalElements, result))) {
      final tempOut = NDArray<Tr>.create(shape, result.dtype);
      try {
        if (whereMask != null && whereMask != ffi.nullptr) {
          unaryOp<Tr, Tr>(
            tempOut,
            result,
            shape,
            stridesResult,
            tempOut.strides,
            0,
            offsetResult,
            0,
            (v) => v,
          );
        }
        ternaryOp<Ta, Tb, Tc, Tr>(
          tempOut,
          a,
          b,
          c,
          shape,
          stridesA,
          stridesB,
          stridesC,
          tempOut.strides,
          0,
          offsetA,
          offsetB,
          offsetC,
          0,
          op,
          whereMask,
          flatIndex,
        );
        unaryOp<Tr, Tr>(
          result,
          tempOut,
          shape,
          tempOut.strides,
          stridesResult,
          0,
          0,
          offsetResult,
          (v) => v,
        );
      } finally {
        tempOut.dispose();
      }
      return;
    }
  }
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
  checkTotalSize(common);
  return common;
}

/// Maximum dimension or leading stride supported by LP64 32-bit `blasint` OpenBLAS/LAPACK kernels.
const int maxBlasInt = 0x7fffffff;

/// Validates that an OpenBLAS/LAPACK matrix or vector dimension fits within LP64 32-bit signed `blasint`.
void checkBlasIntDim(int value, String name, String operation) {
  if (value > maxBlasInt || value < -maxBlasInt) {
    throw UnsupportedError(
      '$operation dimension $name ($value) exceeds 32-bit OpenBLAS/LAPACK blasint limit ($maxBlasInt).',
    );
  }
}

/// Validates that an OpenBLAS/LAPACK stride or leading dimension fits within LP64 32-bit signed `blasint`.
void checkBlasIntStride(int stride, String name, String operation) {
  if (stride > maxBlasInt || stride < -maxBlasInt) {
    throw UnsupportedError(
      '$operation stride $name ($stride) exceeds 32-bit OpenBLAS/LAPACK blasint limit ($maxBlasInt).',
    );
  }
}

dynamic castValue(dynamic val, DType dtype, {DType? sourceDType}) {
  switch (dtype) {
    case DType.complex128:
    case DType.complex64:
      if (val is Complex) return val;
      if (val is int && sourceDType == DType.uint64) {
        return Complex(BigInt.from(val).toUnsigned(64).toDouble(), 0.0);
      }
      if (val is num) return Complex(val.toDouble(), 0.0);
      return Complex(0.0, 0.0);
    case DType.float64:
    case DType.float32:
    case DType.float16:
    case DType.bfloat16:
      if (val is int && sourceDType == DType.uint64) {
        return BigInt.from(val).toUnsigned(64).toDouble();
      }
      if (val is num) return val.toDouble();
      if (val is Complex) return val.real;
      if (val is bool) return val ? 1.0 : 0.0;
      return 0.0;
    case DType.uint64:
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      if (val is int) return val;
      if (val is double) return saturatingDoubleToInt(val, dtype);
      if (val is Complex) return saturatingDoubleToInt(val.real, dtype);
      if (val is bool) return val ? 1 : 0;
      if (val is num) return val.toInt();
      return 0;
    case DType.boolean:
      if (val is bool) return val;
      if (val is num) return val != 0;
      if (val is Complex) return val.real != 0.0 || val.imag != 0.0;
      return false;
  }
}

enum CumOpType { sum, prod, min, max }

NDArray<R> cumOpFFI<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  int axis,
  NDArray<R> result,
  CumOpType opType,
) {
  if (!result.isWriteable) {
    throw ArgumentError(
      'Assignment destination is a read-only broadcast view.',
    );
  }
  if (sharesMemory(a, result)) {
    return NDArray.scope(() {
      final temp = NDArray<R>.create(result.shape, result.dtype);
      cumOpFFI<T, R>(a, axis, temp, opType);
      return temp.copy(out: result);
    });
  }
  if (a.dtype != result.dtype) {
    if (result.dtype == DType.boolean) {
      _cumOpFallbackHelper(
        a,
        result,
        axis,
        opType == CumOpType.sum ? s_cumsum_double : s_cumprod_double,
        opType,
      );
      return result;
    }
    return NDArray.scope(() {
      final casted = castNDArray<R>(a, result.dtype);
      return cumOpFFI<R, R>(casted, axis, result, opType);
    });
  }
  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.copyInts(a.shape);
    final cStridesA = ScratchArena.copyInts(a.strides);
    final cStridesRes = ScratchArena.copyInts(result.strides);

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
          case DType.uint64:
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
          case DType.uint32:
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
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            _cumOpFallbackHelper(
              a,
              result,
              axis,
              s_cumsum_double,
              CumOpType.sum,
            );
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
          case DType.uint64:
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
          case DType.uint32:
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
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
            _cumOpFallbackHelper(
              a,
              result,
              axis,
              s_cumprod_double,
              CumOpType.prod,
            );
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
            _cumOpFallbackHelper(
              a,
              result,
              axis,
              s_cummin_double,
              CumOpType.min,
            );
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
            _cumOpFallbackHelper(
              a,
              result,
              axis,
              s_cummax_double,
              CumOpType.max,
            );
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

int _dtypeToCode(DType dtype) {
  switch (dtype) {
    case DType.float64:
      return 0;
    case DType.float32:
      return 1;
    case DType.float16:
      return 2;
    case DType.bfloat16:
      return 3;
    case DType.int64:
      return 4;
    case DType.int32:
      return 5;
    case DType.int16:
      return 6;
    case DType.int8:
      return 7;
    case DType.uint64:
      return 8;
    case DType.uint32:
      return 9;
    case DType.uint16:
      return 10;
    case DType.uint8:
      return 11;
    case DType.complex128:
      return 12;
    case DType.complex64:
      return 13;
    case DType.boolean:
      return 14;
  }
}

/// Casts [a] to the specified [targetDType], returning a new [NDArray] (or a view if [a] already has [targetDType]).
NDArray<R> castNDArray<R extends DTypeTag>(NDArray a, DType<R> targetDType) {
  if (a.dtype == targetDType) {
    if (a is NDArray<R>) return a;
    return NDArray<R>.view(a, shape: a.shape, strides: a.strides);
  }

  final result = NDArray<R>.create(a.shape, targetDType);
  if (a.size == 0) {
    return result;
  }

  if (a.rank == 0) {
    final scalarVal = a.scalar;
    final converted = castValue(scalarVal, targetDType, sourceDType: a.dtype);
    result.setCellRaw(0, converted);
    return result;
  }

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rank, 3);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);

    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    final srcDTypeCode = _dtypeToCode(a.dtype);
    final dstDTypeCode = _dtypeToCode(targetDType);

    s_cast_generic(
      a.pointer.cast(),
      cStridesA,
      srcDTypeCode,
      result.pointer.cast(),
      dstDTypeCode,
      cShape,
      rank,
    );
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

void _cumOpFallbackHelper<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<R> result,
  int axis,
  void Function(
    ffi.Pointer<ffi.Double> src,
    ffi.Pointer<ffi.Int64> srcStrides,
    ffi.Pointer<ffi.Double> dest,
    ffi.Pointer<ffi.Int64> destStrides,
    ffi.Pointer<ffi.Int64> shape,
    int rank,
    int axis,
  )
  ffiFunc,
  CumOpType opType,
) {
  if (a.dtype.isInteger ||
      result.dtype.isInteger ||
      a.dtype == DType.boolean ||
      result.dtype == DType.boolean) {
    final shape = a.shape;
    if (shape.isEmpty || a.size == 0) return;
    final rank = shape.length;
    final axisLen = shape[axis];
    final outerCount = a.size ~/ axisLen;
    final coord = List<int>.filled(rank, 0);
    final isUnsigned = a.dtype == DType.uint64 || result.dtype == DType.uint64;
    final accSourceDType = isUnsigned ? DType.uint64 : DType.int64;

    int toIntVal(dynamic v) {
      if (v is bool) return v ? 1 : 0;
      if (v is int) return v;
      if (v is double) {
        if (isUnsigned) {
          if (v.isNaN || v <= 0.0) return 0;
          if (v.isInfinite || v >= 18446744073709551616.0) return -1;
          if (v >= 9223372036854775808.0) {
            return BigInt.from(v).toSigned(64).toInt();
          }
          return v.toInt();
        }
        return v.toInt();
      }
      return (v as num).toInt();
    }

    for (int outer = 0; outer < outerCount; outer++) {
      int baseOffsetA = a.offsetElements;
      int baseOffsetRes = result.offsetElements;
      for (int d = 0; d < rank; d++) {
        if (d != axis) {
          baseOffsetA += coord[d] * a.strides[d];
          baseOffsetRes += coord[d] * result.strides[d];
        }
      }

      int acc = 0;
      final resDType = result.dtype;
      final resIsBool = resDType == DType.boolean;
      final resIsIntOrBool = resDType.isInteger || resIsBool;
      for (int i = 0; i < axisLen; i++) {
        final val = a.getCellRaw(baseOffsetA + i * a.strides[axis]);
        final vInt = toIntVal(val);
        if (i == 0) {
          acc = vInt;
        } else {
          switch (opType) {
            case CumOpType.sum:
              if (resIsBool) {
                acc = (acc != 0 || vInt != 0) ? 1 : 0;
              } else {
                acc = acc + vInt;
              }
            case CumOpType.prod:
              if (resIsBool) {
                acc = (acc != 0 && vInt != 0) ? 1 : 0;
              } else {
                acc = acc * vInt;
              }
            case CumOpType.min:
              if (resIsBool) {
                acc = (acc != 0 && vInt != 0) ? 1 : 0;
              } else if (isUnsigned) {
                if (uint64Compare(vInt, acc) < 0) acc = vInt;
              } else {
                if (vInt < acc) acc = vInt;
              }
            case CumOpType.max:
              if (resIsBool) {
                acc = (acc != 0 || vInt != 0) ? 1 : 0;
              } else if (isUnsigned) {
                if (uint64Compare(vInt, acc) > 0) acc = vInt;
              } else {
                if (vInt > acc) acc = vInt;
              }
          }
        }
        final resIdx = baseOffsetRes + i * result.strides[axis];
        result.setCellRaw(
          resIdx,
          castValue(acc, resDType, sourceDType: accSourceDType),
        );
        if (resIsIntOrBool) {
          acc = toIntVal(result.getCellRaw(resIdx));
        }
      }

      for (int d = rank - 1; d >= 0; d--) {
        if (d == axis) continue;
        coord[d]++;
        if (coord[d] < shape[d]) break;
        coord[d] = 0;
      }
    }
    return;
  }

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
  final NDArray<DTypeTag>? _tempAllocated;

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
MaskHolder prepareMask(NDArray<DTypeTag>? where, List<int> targetShape) {
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
