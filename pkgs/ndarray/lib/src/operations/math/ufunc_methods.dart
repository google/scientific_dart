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

import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../nditer.dart';
import '../../scratch_arena.dart';
import '../broadcasting.dart';
import '../helpers.dart';
import 'binary_op.dart';
import 'arithmetic.dart';
import 'bitwise.dart';
import 'complex.dart';
import 'exponential.dart';
import 'logical.dart';
import 'floating_point.dart';
import 'trigonometric.dart';
import 'utility.dart';

/// Extension methods for generalized ufunc operations on [NDArray].
extension UfuncNDArrayExtension<T extends DTypeTag> on NDArray<T> {
  /// Reduces this array along [axis] using [op].
  ///
  /// **Preconditions:**
  /// - It is an error if [op] is not reducible ([op.isReducible] is false).
  /// - It is an error if this array or [out] (if provided) is disposed.
  /// - It is an error if [axis] is not within `[-rank, rank - 1]`.
  /// - It is an error if this array is empty without [initial].
  /// - It is an error if [out] (if provided) has incompatible shape or dtype.
  NDArray<T> reduce({
    required BinaryOp op,
    int? axis,
    bool keepdims = false,
    NDArray<T>? out,
    Object? initial,
  }) => reduceUfunc(
    this,
    op: op,
    axis: axis,
    keepdims: keepdims,
    out: out,
    initial: initial,
  );

  /// Performs a cumulative operation on this array along [axis] using [op].
  ///
  /// **Preconditions:**
  /// - It is an error if [op] is not reducible ([op.isReducible] is false).
  /// - It is an error if this array or [out] (if provided) is disposed.
  /// - It is an error if [axis] is not within `[-rank, rank - 1]`.
  /// - It is an error if [out] (if provided) has incompatible shape or dtype.
  NDArray<T> accumulate({
    required BinaryOp op,
    int axis = 0,
    NDArray<T>? out,
  }) => accumulateUfunc(this, op: op, axis: axis, out: out);

  /// Performs slice reductions along [axis] for intervals defined by [indices] using [op].
  ///
  /// **Preconditions:**
  /// - It is an error if [op] is not reducible ([op.isReducible] is false).
  /// - It is an error if this array, [indices], or [out] is disposed.
  /// - It is an error if [axis] is not within `[-rank, rank - 1]`.
  /// - It is an error if [out] (if provided) has incompatible shape or dtype.
  NDArray<T> reduceat(
    NDArray<DTypeTag> indices, {
    required BinaryOp op,
    int axis = 0,
    NDArray<T>? out,
  }) => reduceatUfunc(this, indices, op: op, axis: axis, out: out);

  /// Performs an outer binary operation between this array and [b] using [op].
  ///
  /// **Preconditions:**
  /// - It is an error if this array, [b], or [out] is disposed.
  /// - It is an error if [out] (if provided) has incompatible shape or dtype.
  NDArray<R> outer<R extends DTypeTag>(
    NDArray<T> b, {
    BinaryOp op = BinaryOp.multiply,
    NDArray<DTypeTag>? where,
    NDArray<R>? out,
  }) => outerUfunc(this, b, op: op, where: where, out: out);

  /// Performs unbuffered in-place scatter updates on this array at [indices] using [b] and [op].
  ///
  /// **Preconditions:**
  /// - It is an error if this array, [indices], or [b] is disposed.
  void at(
    NDArray<DTypeTag> indices,
    NDArray<DTypeTag> b, {
    required BinaryOp op,
  }) => atUfunc(this, indices, b, op: op);
}

NDArray<U> _asView<U extends DTypeTag>(NDArray a) {
  if (a is NDArray<U>) return a;
  return NDArray<U>.view(
    a,
    shape: a.shape,
    strides: a.strides,
    offsetElements: 0,
  );
}

R _withView<U extends DTypeTag, R>(NDArray a, R Function(NDArray<U> view) fn) {
  if (a is NDArray<U>) return fn(a);
  if (a.isDisposed) {
    throw StateError('Cannot operate on a disposed array.');
  }
  if (a.dtype is! DType<U>) {
    throw ArgumentError.value(a, 'out', 'Must have compatible dtype');
  }
  final v = NDArray<U>.view(
    a,
    shape: a.shape,
    strides: a.strides,
    offsetElements: 0,
  );
  try {
    return fn(v);
  } finally {
    v.dispose();
  }
}

R _withViewNullable<U extends DTypeTag, R>(
  NDArray? a,
  R Function(NDArray<U>? view) fn,
) {
  if (a == null) return fn(null);
  validateOutBuffer(a);
  return _withView<U, R>(a, fn);
}

NDArray<R> _coerceOwned<R extends DTypeTag>(NDArray res) {
  if (res is NDArray<R>) return res;
  try {
    final typed = _createTyped<R>(res.shape, res.dtype);
    res.copy(out: typed);
    return typed;
  } finally {
    res.dispose();
  }
}

NDArray<R> _createTyped<R extends DTypeTag>(
  List<int> shape,
  DType dtype, {
  bool zeroInit = false,
}) {
  final NDArray arr = switch (dtype) {
    DType.float64 => NDArray<Float64>.create(
      shape,
      DType.float64,
      zeroInit: zeroInit,
    ),
    DType.float32 => NDArray<Float32>.create(
      shape,
      DType.float32,
      zeroInit: zeroInit,
    ),
    DType.float16 => NDArray<Float16>.create(
      shape,
      DType.float16,
      zeroInit: zeroInit,
    ),
    DType.bfloat16 => NDArray<BFloat16>.create(
      shape,
      DType.bfloat16,
      zeroInit: zeroInit,
    ),
    DType.int64 => NDArray<Int64>.create(
      shape,
      DType.int64,
      zeroInit: zeroInit,
    ),
    DType.int32 => NDArray<Int32>.create(
      shape,
      DType.int32,
      zeroInit: zeroInit,
    ),
    DType.int16 => NDArray<Int16>.create(
      shape,
      DType.int16,
      zeroInit: zeroInit,
    ),
    DType.int8 => NDArray<Int8>.create(shape, DType.int8, zeroInit: zeroInit),
    DType.uint64 => NDArray<Uint64>.create(
      shape,
      DType.uint64,
      zeroInit: zeroInit,
    ),
    DType.uint32 => NDArray<Uint32>.create(
      shape,
      DType.uint32,
      zeroInit: zeroInit,
    ),
    DType.uint16 => NDArray<Uint16>.create(
      shape,
      DType.uint16,
      zeroInit: zeroInit,
    ),
    DType.uint8 => NDArray<Uint8>.create(
      shape,
      DType.uint8,
      zeroInit: zeroInit,
    ),
    DType.boolean => NDArray<Boolean>.create(
      shape,
      DType.boolean,
      zeroInit: zeroInit,
    ),
    DType.complex64 => NDArray<Complex64>.create(
      shape,
      DType.complex64,
      zeroInit: zeroInit,
    ),
    DType.complex128 => NDArray<Complex128>.create(
      shape,
      DType.complex128,
      zeroInit: zeroInit,
    ),
  };
  return _asView<R>(arr);
}

/// Evaluates binary operation [op] element-wise between [a] and [b].
NDArray<R> binaryUfunc<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  required BinaryOp op,
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute binaryUfunc() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  switch (op) {
    case BinaryOp.add:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => add(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.subtract:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => subtract(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.multiply:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => multiply(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.divide:
      final res = divide<T, T, DTypeTag>(a, b, where: where, out: out);
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.floorDivide:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => floorDivide(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.remainder:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => remainder(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.fmod:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => fmod(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.gcd:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => gcd(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.lcm:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => lcm(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.heaviside:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => heaviside(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.power:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => power(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.floatPower:
      if (a.dtype.isComplex || b.dtype.isComplex) {
        final aCpx = castNDArray<Complex128>(a, DType.complex128);
        final bCpx = castNDArray<Complex128>(b, DType.complex128);
        try {
          final res = _withViewNullable<Complex128, NDArray<Complex128>>(
            out,
            (outView) =>
                power<Complex128>(aCpx, bCpx, where: where, out: outView),
          );
          return out ?? _coerceOwned<R>(res);
        } finally {
          if (!identical(aCpx, a)) aCpx.dispose();
          if (!identical(bCpx, b)) bCpx.dispose();
        }
      } else {
        final aFloat = castNDArray<Float64>(a, DType.float64);
        final bFloat = castNDArray<Float64>(b, DType.float64);
        try {
          final res = _withViewNullable<Float64, NDArray<Float64>>(
            out,
            (outView) =>
                power<Float64>(aFloat, bFloat, where: where, out: outView),
          );
          return out ?? _coerceOwned<R>(res);
        } finally {
          if (!identical(aFloat, a)) aFloat.dispose();
          if (!identical(bFloat, b)) bFloat.dispose();
        }
      }
    case BinaryOp.logaddexp:
      final res = _withView<AnySpec, NDArray<DTypeTag>>(
        a,
        (aSpec) => _withView<AnySpec, NDArray<DTypeTag>>(
          b,
          (bSpec) => logaddexp<DTypeTag>(aSpec, bSpec, where: where, out: out),
        ),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.logaddexp2:
      final res = _withView<AnySpec, NDArray<DTypeTag>>(
        a,
        (aSpec) => _withView<AnySpec, NDArray<DTypeTag>>(
          b,
          (bSpec) => logaddexp2<DTypeTag>(aSpec, bSpec, where: where, out: out),
        ),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.arctan2:
      final res = _withView<AnySpec, NDArray<DTypeTag>>(
        a,
        (aSpec) => _withView<AnySpec, NDArray<DTypeTag>>(
          b,
          (bSpec) => atan2<DTypeTag>(aSpec, bSpec, where: where, out: out),
        ),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.hypot:
      final res = _withView<AnySpec, NDArray<DTypeTag>>(
        a,
        (aSpec) => _withView<AnySpec, NDArray<DTypeTag>>(
          b,
          (bSpec) => hypot<DTypeTag>(aSpec, bSpec, where: where, out: out),
        ),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.copysign:
      final res = _withViewNullable<T, NDArray<T>>(
        out,
        (outView) => copysign<T>(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.bitwiseAnd:
      final res = bitwiseAnd<DTypeTag>(a, b, where: where, out: out);
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.bitwiseOr:
      final res = bitwiseOr<DTypeTag>(a, b, where: where, out: out);
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.bitwiseXor:
      final res = bitwiseXor<DTypeTag>(a, b, where: where, out: out);
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.leftShift:
      final res = leftShift<DTypeTag>(a, b, where: where, out: out);
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.rightShift:
      final res = rightShift<DTypeTag>(a, b, where: where, out: out);
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.logicalAnd:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => logicalAnd(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.logicalOr:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => logicalOr(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.logicalXor:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => logicalXor(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.minimum:
      if (where == null) {
        return _nativeMinMax(a, b, opCode: 2, out: out);
      }
      return _elementwiseMinMax(
        a,
        b,
        isMax: false,
        ignoreNaN: false,
        whereMask: where,
        out: out,
      );
    case BinaryOp.fmin:
      if (where == null) {
        return _nativeMinMax(a, b, opCode: 4, out: out);
      }
      return _elementwiseMinMax(
        a,
        b,
        isMax: false,
        ignoreNaN: true,
        whereMask: where,
        out: out,
      );
    case BinaryOp.maximum:
      if (where == null) {
        return _nativeMinMax(a, b, opCode: 3, out: out);
      }
      return _elementwiseMinMax(
        a,
        b,
        isMax: true,
        ignoreNaN: false,
        whereMask: where,
        out: out,
      );
    case BinaryOp.fmax:
      if (where == null) {
        return _nativeMinMax(a, b, opCode: 5, out: out);
      }
      return _elementwiseMinMax(
        a,
        b,
        isMax: true,
        ignoreNaN: true,
        whereMask: where,
        out: out,
      );
    case BinaryOp.equal:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => equal(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.notEqual:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => notEqual(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.greater:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => greater(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.greaterEqual:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => greaterEqual(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.less:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => less(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
    case BinaryOp.lessEqual:
      final res = _withViewNullable<Boolean, NDArray<Boolean>>(
        out,
        (outView) => lessEqual(a, b, where: where, out: outView),
      );
      return out ?? _coerceOwned<R>(res);
  }
}

bool _isValueNaN(dynamic v) {
  if (v is double) return v.isNaN;
  if (v is Complex) return v.real.isNaN || v.imag.isNaN;
  return false;
}

int _compareValues(dynamic a, dynamic b, DType dtype) {
  if (dtype == DType.boolean) {
    final ba = a as bool;
    final bb = b as bool;
    if (ba == bb) return 0;
    return ba ? 1 : -1;
  }
  if (dtype == DType.uint64) {
    return uint64Compare(a as int, b as int);
  }
  if (dtype.isComplex) {
    final ca = a as Complex;
    final cb = b as Complex;
    final cmpReal = ca.real.compareTo(cb.real);
    if (cmpReal != 0) return cmpReal;
    return ca.imag.compareTo(cb.imag);
  }
  return (a as num).compareTo(b as num);
}

NDArray<R> _nativeMinMax<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  required int opCode,
  NDArray<R>? out,
}) {
  final targetShape = broadcastShapes(a.shape, b.shape);
  final targetDType = out?.dtype ?? a.dtype;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, targetShape)) {
      throw ArgumentError.value(
        out,
        'out',
        'Output array shape ${out.shape} must match broadcast shape $targetShape',
      );
    }
  }
  if (targetDType != a.dtype) {
    // If output dtype differs, fall back to casted elementwise
    return _elementwiseMinMax(
      a,
      b,
      isMax: opCode == 3 || opCode == 5,
      ignoreNaN: opCode == 4 || opCode == 5,
      out: out,
    );
  }
  if (out != null && (sharesMemory(a, out) || sharesMemory(b, out))) {
    return NDArray.scope(() {
      final temp = _createTyped<R>(targetShape, targetDType);
      _nativeMinMax<T, R>(a, b, opCode: opCode, out: temp);
      temp.copy(out: out);
      return out;
    });
  }

  final result = out ?? _createTyped<R>(targetShape, targetDType);
  if (result.size == 0) return result;

  final dtypeCode = switch (a.dtype) {
    DType.float64 => 0,
    DType.float32 => 1,
    DType.float16 => 2,
    DType.bfloat16 => 3,
    DType.int64 => 4,
    DType.int32 => 5,
    DType.int16 => 6,
    DType.int8 => 7,
    DType.uint64 => 8,
    DType.uint32 => 9,
    DType.uint16 => 10,
    DType.uint8 => 11,
    DType.complex128 => 12,
    DType.complex64 => 13,
    DType.boolean => 14,
  };

  // Contiguous same-shape fast path
  if (a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, targetShape) &&
      listEquals(b.shape, targetShape)) {
    v_binary_minmax(
      opCode,
      dtypeCode,
      a.pointer.cast(),
      b.pointer.cast(),
      result.pointer.cast(),
      result.size,
    );
    checkNativeOom();
    return result;
  }

  // Strided N-D broadcast path
  final aBroadcast = listEquals(a.shape, targetShape)
      ? a
      : broadcastTo(a, targetShape);
  final bBroadcast = listEquals(b.shape, targetShape)
      ? b
      : broadcastTo(b, targetShape);
  try {
    final rank = targetShape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank, 4);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesOut = cBuffer + (rank * 3);
      for (var i = 0; i < rank; i++) {
        cShape[i] = targetShape[i];
        cStridesA[i] = aBroadcast.strides[i];
        cStridesB[i] = bBroadcast.strides[i];
        cStridesOut[i] = result.strides[i];
      }
      s_binary_minmax(
        opCode,
        dtypeCode,
        rank,
        cShape.cast(),
        aBroadcast.pointer.cast(),
        cStridesA.cast(),
        bBroadcast.pointer.cast(),
        cStridesB.cast(),
        result.pointer.cast(),
        cStridesOut.cast(),
      );
      checkNativeOom();
    } finally {
      ScratchArena.reset(marker);
    }
    return result;
  } finally {
    if (!identical(aBroadcast, a)) aBroadcast.dispose();
    if (!identical(bBroadcast, b)) bBroadcast.dispose();
  }
}

NDArray<R> _elementwiseMinMax<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  required bool isMax,
  required bool ignoreNaN,
  NDArray<DTypeTag>? whereMask,
  NDArray<R>? out,
}) {
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  final targetShape = broadcastShapes(a.shape, b.shape);
  final targetDType = out?.dtype ?? a.dtype;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, targetShape)) {
      throw ArgumentError.value(
        out,
        'out',
        'Output array shape ${out.shape} must match broadcast shape $targetShape',
      );
    }
  }
  if (out != null &&
      (sharesMemory(a, out) ||
          sharesMemory(b, out) ||
          (whereMask != null && sharesMemory(whereMask, out)))) {
    return NDArray.scope(() {
      final temp = whereMask != null
          ? out.copy()
          : _createTyped<R>(targetShape, targetDType);
      _elementwiseMinMax<T, R>(
        a,
        b,
        isMax: isMax,
        ignoreNaN: ignoreNaN,
        whereMask: whereMask,
        out: temp,
      );
      temp.copy(out: out);
      return out;
    });
  }
  final aCasted = a.dtype != targetDType
      ? castNDArray<R>(a, targetDType as DType<R>)
      : _asView<R>(a);
  final bCasted = b.dtype != targetDType
      ? castNDArray<R>(b, targetDType as DType<R>)
      : _asView<R>(b);
  NDArray<Boolean>? wBool;
  NDArray<Boolean>? wBroadcast;
  try {
    if (whereMask != null) {
      wBool = whereMask.dtype == DType.boolean
          ? _asView<Boolean>(whereMask)
          : castNDArray<Boolean>(whereMask, DType.boolean);
      wBroadcast = broadcastTo(wBool, targetShape);
    }
    final result =
        out ??
        _createTyped<R>(targetShape, targetDType, zeroInit: whereMask != null);
    final iter = NDIter.broadcast3(result, aCasted, bCasted);
    final wIter = wBroadcast != null ? NDIter(wBroadcast) : null;
    while (iter.moveNext()) {
      if (wIter != null) {
        wIter.moveNext();
        if (!wBroadcast!.getCellRaw(wIter.index)) continue;
      }
      final idxRes = iter.getIndex(0);
      final idxA = iter.getIndex(1);
      final idxB = iter.getIndex(2);
      final valA = aCasted.getCellRaw(idxA);
      final valB = bCasted.getCellRaw(idxB);
      final nanA = _isValueNaN(valA);
      final nanB = _isValueNaN(valB);
      final Object? chosen;
      if (nanA || nanB) {
        if (ignoreNaN) {
          if (nanA && nanB) {
            chosen = valA;
          } else if (nanA) {
            chosen = valB;
          } else {
            chosen = valA;
          }
        } else {
          chosen = nanA ? valA : valB;
        }
      } else {
        final cmp = _compareValues(valA, valB, targetDType);
        if (isMax) {
          chosen = cmp >= 0 ? valA : valB;
        } else {
          chosen = cmp <= 0 ? valA : valB;
        }
      }
      result.setCellRaw(idxRes, chosen);
    }
    return result;
  } finally {
    if (!identical(aCasted, a)) aCasted.dispose();
    if (!identical(bCasted, b)) bCasted.dispose();
    if (wBroadcast != null && !identical(wBroadcast, whereMask)) {
      wBroadcast.dispose();
    }
    if (wBool != null &&
        !identical(wBool, whereMask) &&
        !identical(wBool, wBroadcast)) {
      wBool.dispose();
    }
  }
}

/// Reduces [a] along [axis] using [op].
///
/// **Preconditions:**
/// - It is an error if [op] is not reducible ([op.isReducible] is false).
/// - It is an error if [a] or [out] (if provided) is disposed.
/// - It is an error if [axis] is not within `[-rank, rank - 1]`.
/// - It is an error if [a] is empty without [initial].
/// - It is an error if [out] (if provided) has incompatible shape or dtype.
NDArray<T> reduce<T extends DTypeTag>(
  NDArray<T> a, {
  required BinaryOp op,
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
  Object? initial,
}) => reduceUfunc(
  a,
  op: op,
  axis: axis,
  keepdims: keepdims,
  out: out,
  initial: initial,
);

/// Performs a cumulative operation on [a] along [axis] using [op].
///
/// **Preconditions:**
/// - It is an error if [op] is not reducible ([op.isReducible] is false).
/// - It is an error if [a] or [out] (if provided) is disposed.
/// - It is an error if [axis] is not within `[-rank, rank - 1]`.
/// - It is an error if [out] (if provided) has incompatible shape or dtype.
NDArray<T> accumulate<T extends DTypeTag>(
  NDArray<T> a, {
  required BinaryOp op,
  int axis = 0,
  NDArray<T>? out,
}) => accumulateUfunc(a, op: op, axis: axis, out: out);

/// Performs slice reductions on [a] along [axis] for intervals defined by [indices] using [op].
///
/// **Preconditions:**
/// - It is an error if [op] is not reducible ([op.isReducible] is false).
/// - It is an error if [a], [indices], or [out] is disposed.
/// - It is an error if [axis] is not within `[-rank, rank - 1]`.
/// - It is an error if [out] (if provided) has incompatible shape or dtype.
NDArray<T> reduceat<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<DTypeTag> indices, {
  required BinaryOp op,
  int axis = 0,
  NDArray<T>? out,
}) => reduceatUfunc(a, indices, op: op, axis: axis, out: out);

/// Performs unbuffered in-place scatter updates on [a] at [indices] using [b] and [op].
///
/// **Preconditions:**
/// - It is an error if [a], [indices], or [b] is disposed.
void at<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<DTypeTag> indices,
  NDArray<DTypeTag> b, {
  required BinaryOp op,
}) => atUfunc(a, indices, b, op: op);

/// Generalized ufunc reduction function.
NDArray<T> reduceUfunc<T extends DTypeTag>(
  NDArray<T> a, {
  required BinaryOp op,
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
  Object? initial,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute reduce on a disposed array.');
  }
  if (op == BinaryOp.subtract && a.dtype == DType.boolean) {
    throw UnsupportedError(
      "Boolean subtract, the '-' operator, is not supported; use logicalXor or bitwiseXor instead.",
    );
  }
  if (!op.isReducible) {
    throw ArgumentError.value(
      op,
      'op',
      'Operation ${op.name} is not reducible',
    );
  }

  if (axis == null) {
    // Global reduction
    if (a.size == 0 && initial == null) {
      throw ArgumentError.value(
        a,
        'a',
        'Cannot reduce an empty array without an initial value',
      );
    }
    final targetShape = keepdims ? List.filled(a.rank, 1) : <int>[];
    final NDArray<T> result;
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape and dtype for reduce',
        );
      }
      if (sharesMemory(a, out)) {
        return NDArray.scope(() {
          final temp = reduceUfunc<T>(
            a,
            op: op,
            axis: axis,
            keepdims: keepdims,
            initial: initial,
          );
          temp.copy(out: out);
          return out;
        });
      }
      result = out;
    } else {
      result = _createTyped<T>(targetShape, a.dtype);
    }

    if (a.size == 0) {
      result.fill(initial);
      return result;
    }

    if (a.isContiguous && initial == null) {
      bool handled = false;
      switch (op) {
        case BinaryOp.add:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_sum_double(a.pointer.cast(), a.size));
              handled = true;
            case DType.float32:
              result.fill(r_sum_float(a.pointer.cast(), a.size));
              handled = true;
            case DType.int64:
              result.fill(r_sum_int64(a.pointer.cast(), a.size));
              handled = true;
            case DType.int32:
              result.fill(r_sum_int32(a.pointer.cast(), a.size));
              handled = true;
            case DType.uint8:
              result.fill(r_sum_uint8(a.pointer.cast(), a.size));
              handled = true;
            case DType.int16:
              result.fill(r_sum_int16(a.pointer.cast(), a.size));
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.multiply:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_prod_double(a.pointer.cast(), a.size));
              handled = true;
            case DType.float32:
              result.fill(r_prod_float(a.pointer.cast(), a.size));
              handled = true;
            case DType.int64:
              result.fill(r_prod_int64(a.pointer.cast(), a.size));
              handled = true;
            case DType.int32:
              result.fill(r_prod_int32(a.pointer.cast(), a.size));
              handled = true;
            case DType.uint8:
              result.fill(r_prod_uint8(a.pointer.cast(), a.size));
              handled = true;
            case DType.int16:
              result.fill(r_prod_int16(a.pointer.cast(), a.size));
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.minimum:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_min_double(a.pointer.cast(), a.size));
              handled = true;
            case DType.float32:
              result.fill(r_min_float(a.pointer.cast(), a.size));
              handled = true;
            case DType.int64:
              result.fill(r_min_int64_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.int32:
              result.fill(r_min_int32_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.uint8:
              result.fill(r_min_uint8_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.int16:
              result.fill(r_min_int16_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.maximum:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_max_double(a.pointer.cast(), a.size));
              handled = true;
            case DType.float32:
              result.fill(r_max_float(a.pointer.cast(), a.size));
              handled = true;
            case DType.int64:
              result.fill(r_max_int64_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.int32:
              result.fill(r_max_int32_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.uint8:
              result.fill(r_max_uint8_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.int16:
              result.fill(r_max_int16_t(a.pointer.cast(), a.size));
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.bitwiseAnd:
          switch (a.dtype) {
            case DType.int64:
            case DType.uint64:
              result.fill(r_bitwise_and_int64(a.pointer.cast(), a.size));
              handled = true;
            case DType.int32:
            case DType.uint32:
              result.fill(r_bitwise_and_int32(a.pointer.cast(), a.size));
              handled = true;
            case DType.uint8:
            case DType.int8:
              result.fill(r_bitwise_and_uint8(a.pointer.cast(), a.size));
              handled = true;
            case DType.int16:
            case DType.uint16:
              result.fill(r_bitwise_and_int16(a.pointer.cast(), a.size));
              handled = true;
            case DType.float64:
            case DType.float32:
            case DType.float16:
            case DType.bfloat16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.bitwiseOr:
          switch (a.dtype) {
            case DType.int64:
            case DType.uint64:
              result.fill(r_bitwise_or_int64(a.pointer.cast(), a.size));
              handled = true;
            case DType.int32:
            case DType.uint32:
              result.fill(r_bitwise_or_int32(a.pointer.cast(), a.size));
              handled = true;
            case DType.uint8:
            case DType.int8:
              result.fill(r_bitwise_or_uint8(a.pointer.cast(), a.size));
              handled = true;
            case DType.int16:
            case DType.uint16:
              result.fill(r_bitwise_or_int16(a.pointer.cast(), a.size));
              handled = true;
            case DType.float64:
            case DType.float32:
            case DType.float16:
            case DType.bfloat16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.bitwiseXor:
          switch (a.dtype) {
            case DType.int64:
            case DType.uint64:
              result.fill(r_bitwise_xor_int64(a.pointer.cast(), a.size));
              handled = true;
            case DType.int32:
            case DType.uint32:
              result.fill(r_bitwise_xor_int32(a.pointer.cast(), a.size));
              handled = true;
            case DType.uint8:
            case DType.int8:
              result.fill(r_bitwise_xor_uint8(a.pointer.cast(), a.size));
              handled = true;
            case DType.int16:
            case DType.uint16:
              result.fill(r_bitwise_xor_int16(a.pointer.cast(), a.size));
              handled = true;
            case DType.float64:
            case DType.float32:
            case DType.float16:
            case DType.bfloat16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.logicalAnd:
          if (a.dtype == DType.boolean) {
            result.fill((r_logical_and(a.pointer.cast(), a.size) != 0));
            handled = true;
          }
        case BinaryOp.logicalOr:
          if (a.dtype == DType.boolean) {
            result.fill((r_logical_or(a.pointer.cast(), a.size) != 0));
            handled = true;
          }
        case BinaryOp.logicalXor:
          if (a.dtype == DType.boolean) {
            result.fill((r_logical_xor(a.pointer.cast(), a.size) != 0));
            handled = true;
          }
        default:
          break;
      }
      if (handled) return result;
    }

    // Fallback global reduction via flat view iteration
    final flat = a.ravel();
    final axisRes = reduceUfunc(flat, op: op, axis: 0, initial: initial);
    flat.dispose();
    result.fill(axisRes.scalar);
    axisRes.dispose();
    return result;
  }

  // Axis reduction
  final normAxis = axis < 0 ? axis + a.rank : axis;
  if (normAxis < 0 || normAxis >= a.rank) {
    throw RangeError.range(normAxis, 0, a.rank - 1, 'axis');
  }

  final resShape = List<int>.from(a.shape);
  if (keepdims) {
    resShape[normAxis] = 1;
  } else {
    resShape.removeAt(normAxis);
  }

  final NDArray<T> result;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, resShape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for reduce',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = reduceUfunc<T>(
          a,
          op: op,
          axis: axis,
          keepdims: keepdims,
          initial: initial,
        );
        temp.copy(out: out);
        return out;
      });
    }
    result = out;
  } else {
    result = _createTyped<T>(resShape, a.dtype);
  }

  if (a.shape[normAxis] == 0) {
    if (initial != null) {
      result.fill(initial);
      return result;
    }
    throw ArgumentError.value(
      a,
      'a',
      'Cannot reduce array of size 0 along axis $axis without an initial value',
    );
  }
  if (result.size == 0) return result;

  bool handled = false;
  if (initial == null) {
    final marker = ScratchArena.marker;
    try {
      final rank = a.rank;
      final cBuffer = ScratchArena.getStridedBuffer(rank * 3);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
      }
      if (rank > 1) {
        if (keepdims) {
          var resIdx = 0;
          for (var i = 0; i < rank; i++) {
            if (i != normAxis) {
              cStridesRes[resIdx++] = result.strides[i];
            }
          }
        } else {
          for (var i = 0; i < rank - 1; i++) {
            cStridesRes[i] = result.strides[i];
          }
        }
      }

      switch (op) {
        case BinaryOp.add:
          switch (a.dtype) {
            case DType.float64:
              s_sum_double(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float32:
              s_sum_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int64:
              s_sum_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int32:
              s_sum_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.uint8:
              s_sum_uint8(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int16:
              s_sum_int16(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.multiply:
          switch (a.dtype) {
            case DType.float64:
              s_prod_double(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float32:
              s_prod_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int64:
              s_prod_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int32:
              s_prod_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.uint8:
              s_prod_uint8(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int16:
              s_prod_int16(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.complex128:
              s_prod_complex128(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.complex64:
              s_prod_complex64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
              break;
          }
        case BinaryOp.minimum:
          switch (a.dtype) {
            case DType.float64:
              s_min_double(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float32:
              s_min_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int64:
              s_min_int64_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int32:
              s_min_int32_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.uint8:
              s_min_uint8_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int16:
              s_min_int16_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.maximum:
          switch (a.dtype) {
            case DType.float64:
              s_max_double(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float32:
              s_max_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int64:
              s_max_int64_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int32:
              s_max_int32_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.uint8:
              s_max_uint8_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int16:
              s_max_int16_t(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.bitwiseAnd:
          switch (a.dtype) {
            case DType.int64:
            case DType.uint64:
              s_bitwise_and_red_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int32:
            case DType.uint32:
              s_bitwise_and_red_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.uint8:
            case DType.int8:
              s_bitwise_and_red_uint8(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int16:
            case DType.uint16:
              s_bitwise_and_red_int16(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float64:
            case DType.float32:
            case DType.float16:
            case DType.bfloat16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.bitwiseOr:
          switch (a.dtype) {
            case DType.int64:
            case DType.uint64:
              s_bitwise_or_red_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int32:
            case DType.uint32:
              s_bitwise_or_red_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.uint8:
            case DType.int8:
              s_bitwise_or_red_uint8(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int16:
            case DType.uint16:
              s_bitwise_or_red_int16(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float64:
            case DType.float32:
            case DType.float16:
            case DType.bfloat16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.bitwiseXor:
          switch (a.dtype) {
            case DType.int64:
            case DType.uint64:
              s_bitwise_xor_red_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int32:
            case DType.uint32:
              s_bitwise_xor_red_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.uint8:
            case DType.int8:
              s_bitwise_xor_red_uint8(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.int16:
            case DType.uint16:
              s_bitwise_xor_red_int16(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                normAxis,
              );
              handled = true;
            case DType.float64:
            case DType.float32:
            case DType.float16:
            case DType.bfloat16:
            case DType.boolean:
            case DType.complex128:
            case DType.complex64:
              break;
          }
        case BinaryOp.logicalAnd:
          if (a.dtype == DType.boolean) {
            s_logical_and_red(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          }
        case BinaryOp.logicalOr:
          if (a.dtype == DType.boolean) {
            s_logical_or_red(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          }
        case BinaryOp.logicalXor:
          if (a.dtype == DType.boolean) {
            s_logical_xor_red(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          }
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }

  if (handled) {
    checkNativeOom();
    return result;
  }

  // Fallback axis reduction via Index slicing and binary ufunc
  final axisLen = a.shape[normAxis];
  NDArray<T> current;
  if (initial != null) {
    final squeezedShape = List<int>.from(a.shape)..removeAt(normAxis);
    current = _createTyped<T>(squeezedShape, a.dtype);
    current.fill(initial);
    for (var i = 0; i < axisLen; i++) {
      final selectors = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final sub = a.slice(selectors);
      final stepRes = binaryUfunc(current, sub, op: op);
      sub.dispose();
      current.dispose();
      if (stepRes.dtype == a.dtype) {
        current = stepRes as NDArray<T>;
      } else {
        current = castNDArray<T>(stepRes, a.dtype);
        stepRes.dispose();
      }
    }
  } else {
    final selectors0 = List<Selector>.generate(
      a.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    final slice0 = a.slice(selectors0);
    current = slice0.copy();
    slice0.dispose();
    for (var i = 1; i < axisLen; i++) {
      final selectorsI = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final sub = a.slice(selectorsI);
      final stepRes = binaryUfunc(current, sub, op: op);
      current.dispose();
      sub.dispose();
      if (stepRes.dtype == a.dtype) {
        current = stepRes as NDArray<T>;
      } else {
        current = castNDArray<T>(stepRes, a.dtype);
        stepRes.dispose();
      }
    }
  }
  if (!listEquals(current.shape, result.shape)) {
    final reshaped = current.reshape(result.shape);
    try {
      reshaped.copy(out: result);
    } finally {
      reshaped.dispose();
    }
  } else {
    current.copy(out: result);
  }
  current.dispose();
  return result;
}

/// Generalized ufunc accumulation function.
NDArray<T> accumulateUfunc<T extends DTypeTag>(
  NDArray<T> a, {
  required BinaryOp op,
  int axis = 0,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute accumulate on a disposed array.');
  }
  if (op == BinaryOp.subtract && a.dtype == DType.boolean) {
    throw UnsupportedError(
      "Boolean subtract, the '-' operator, is not supported; use logicalXor or bitwiseXor instead.",
    );
  }
  if (!op.isReducible) {
    throw ArgumentError.value(
      op,
      'op',
      'Operation ${op.name} is not reducible',
    );
  }

  final normAxis = axis < 0 ? axis + a.rank : axis;
  if (normAxis < 0 || normAxis >= a.rank) {
    throw RangeError.range(normAxis, 0, a.rank - 1, 'axis');
  }

  final NDArray<T> result;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for accumulate',
      );
    }
    if (sharesMemory(a, out)) {
      return NDArray.scope(() {
        final temp = accumulateUfunc<T>(a, op: op, axis: axis);
        temp.copy(out: out);
        return out;
      });
    }
    result = out;
  } else {
    result = _createTyped<T>(a.shape, a.dtype);
  }
  if (result.size == 0) return result;

  bool handled = false;
  final marker = ScratchArena.marker;
  try {
    final rank = a.rank;
    final cBuffer = ScratchArena.getStridedBuffer(rank * 3);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    switch (op) {
      case BinaryOp.add:
        switch (a.dtype) {
          case DType.float64:
            s_cumsum_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float32:
            s_cumsum_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int64:
            s_cumsum_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int32:
            s_cumsum_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int16:
            s_cumsum_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.uint8:
            s_cumsum_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.complex128:
            s_cumsum_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.complex64:
            s_cumsum_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float16:
          case DType.bfloat16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.boolean:
            break;
        }
      case BinaryOp.multiply:
        switch (a.dtype) {
          case DType.float64:
            s_cumprod_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float32:
            s_cumprod_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int64:
            s_cumprod_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int32:
            s_cumprod_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int16:
            s_cumprod_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.uint8:
            s_cumprod_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.complex128:
            s_cumprod_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.complex64:
            s_cumprod_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float16:
          case DType.bfloat16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.boolean:
            break;
        }
      case BinaryOp.minimum:
        switch (a.dtype) {
          case DType.float64:
            s_cummin_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float32:
            s_cummin_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int64:
            s_cummin_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int32:
            s_cummin_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float16:
          case DType.bfloat16:
          case DType.int16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
          case DType.complex128:
          case DType.complex64:
            break;
        }
      case BinaryOp.maximum:
        switch (a.dtype) {
          case DType.float64:
            s_cummax_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float32:
            s_cummax_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int64:
            s_cummax_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int32:
            s_cummax_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float16:
          case DType.bfloat16:
          case DType.int16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
          case DType.uint8:
          case DType.boolean:
          case DType.complex128:
          case DType.complex64:
            break;
        }
      case BinaryOp.bitwiseAnd:
        switch (a.dtype) {
          case DType.int64:
          case DType.uint64:
            s_cumbitwise_and_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int32:
          case DType.uint32:
            s_cumbitwise_and_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.uint8:
          case DType.int8:
            s_cumbitwise_and_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int16:
          case DType.uint16:
            s_cumbitwise_and_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.boolean:
          case DType.complex128:
          case DType.complex64:
            break;
        }
      case BinaryOp.bitwiseOr:
        switch (a.dtype) {
          case DType.int64:
          case DType.uint64:
            s_cumbitwise_or_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int32:
          case DType.uint32:
            s_cumbitwise_or_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.uint8:
          case DType.int8:
            s_cumbitwise_or_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int16:
          case DType.uint16:
            s_cumbitwise_or_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.boolean:
          case DType.complex128:
          case DType.complex64:
            break;
        }
      case BinaryOp.bitwiseXor:
        switch (a.dtype) {
          case DType.int64:
          case DType.uint64:
            s_cumbitwise_xor_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int32:
          case DType.uint32:
            s_cumbitwise_xor_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.uint8:
          case DType.int8:
            s_cumbitwise_xor_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.int16:
          case DType.uint16:
            s_cumbitwise_xor_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              normAxis,
            );
            handled = true;
          case DType.float64:
          case DType.float32:
          case DType.float16:
          case DType.bfloat16:
          case DType.boolean:
          case DType.complex128:
          case DType.complex64:
            break;
        }
      case BinaryOp.logicalAnd:
        if (a.dtype == DType.boolean) {
          s_cumlogical_and(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
          handled = true;
        }
      case BinaryOp.logicalOr:
        if (a.dtype == DType.boolean) {
          s_cumlogical_or(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
          handled = true;
        }
      case BinaryOp.logicalXor:
        if (a.dtype == DType.boolean) {
          s_cumlogical_xor(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            normAxis,
          );
          handled = true;
        }
      default:
        break;
    }
  } finally {
    ScratchArena.reset(marker);
  }

  if (handled) {
    checkNativeOom();
    return result;
  }

  // Fallback accumulation
  final axisLen = a.shape[normAxis];
  if (axisLen > 0) {
    final sel0 = List<Selector>.generate(
      a.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    final firstSlice = a.slice(sel0);
    final selRes0 = List<Selector>.generate(
      result.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    final resSlice0 = result.slice(selRes0);
    firstSlice.copy(out: resSlice0);
    resSlice0.dispose();
    firstSlice.dispose();

    for (var i = 1; i < axisLen; i++) {
      final selPrev = List<Selector>.generate(
        result.rank,
        (d) => d == normAxis ? Index(i - 1) : Slice(),
      );
      final prev = result.slice(selPrev);
      final selCurr = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final curr = a.slice(selCurr);
      final stepRes = binaryUfunc(prev, curr, op: op);
      final selResI = List<Selector>.generate(
        result.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final resSliceI = result.slice(selResI);
      if (stepRes.dtype == result.dtype) {
        stepRes.copy(out: resSliceI);
      } else {
        final casted = castNDArray<T>(stepRes, result.dtype);
        casted.copy(out: resSliceI);
        casted.dispose();
      }
      resSliceI.dispose();
      prev.dispose();
      curr.dispose();
      stepRes.dispose();
    }
  }
  return result;
}

/// Generalized ufunc reduceat function.
NDArray<T> reduceatUfunc<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<DTypeTag> indices, {
  required BinaryOp op,
  int axis = 0,
  NDArray<T>? out,
}) {
  if (a.isDisposed || indices.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute reduceat on a disposed array.');
  }
  if (op == BinaryOp.subtract && a.dtype == DType.boolean) {
    throw UnsupportedError(
      "Boolean subtract, the '-' operator, is not supported; use logicalXor or bitwiseXor instead.",
    );
  }
  if (!op.isReducible) {
    throw ArgumentError.value(
      op,
      'op',
      'Operation ${op.name} is not reducible',
    );
  }

  final normAxis = axis < 0 ? axis + a.rank : axis;
  if (normAxis < 0 || normAxis >= a.rank) {
    throw RangeError.range(normAxis, 0, a.rank - 1, 'axis');
  }

  final numIndices = indices.size;
  final axisLen = a.shape[normAxis];
  final resShape = List<int>.from(a.shape);
  resShape[normAxis] = numIndices;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, resShape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for reduceat',
      );
    }
  }

  if (numIndices > 0 && axisLen == 0) {
    throw RangeError(
      'Cannot execute reduceat with non-empty indices on an empty axis of length 0.',
    );
  }

  final opCode = op.index;
  if (numIndices == 0) {
    if (out != null) {
      return out;
    }
    return _createTyped<T>(resShape, a.dtype);
  }

  final marker = ScratchArena.marker;
  try {
    final indicesPtr = ScratchArena.allocate<ffi.Int64>(
      numIndices * ffi.sizeOf<ffi.Int64>(),
    );
    if (indices.isContiguous && indices.dtype == DType.int64) {
      final rawIdxPtr = indices.pointer.cast<ffi.Int64>();
      for (var i = 0; i < numIndices; i++) {
        var idx = rawIdxPtr[i];
        if (idx < -axisLen || idx >= axisLen) {
          throw RangeError.range(idx, -axisLen, axisLen - 1, 'indices');
        }
        if (idx < 0) idx += axisLen;
        indicesPtr[i] = idx;
      }
    } else {
      for (var i = 0; i < numIndices; i++) {
        var idx = (indices.getCellFlat(i) as num).toInt();
        if (indices.dtype == DType.uint64 && idx < 0) {
          throw RangeError.range(idx, -axisLen, axisLen - 1, 'indices');
        }
        if (idx < -axisLen || idx >= axisLen) {
          throw RangeError.range(idx, -axisLen, axisLen - 1, 'indices');
        }
        if (idx < 0) idx += axisLen;
        indicesPtr[i] = idx;
      }
    }

    if (out != null &&
        (sharesMemory(a, out) ||
            sharesMemory(indices, out) ||
            !out.isContiguous)) {
      return NDArray.scope(() {
        final temp = reduceatUfunc<T>(a, indices, op: op, axis: axis);
        temp.copy(out: out);
        return out;
      });
    }

    final NDArray<T> result = out ?? _createTyped<T>(resShape, a.dtype);
    final isBitwiseOrWrapCompatible =
        op == BinaryOp.bitwiseAnd ||
        op == BinaryOp.bitwiseOr ||
        op == BinaryOp.bitwiseXor ||
        op == BinaryOp.add ||
        op == BinaryOp.subtract ||
        op == BinaryOp.multiply;

    if (a.rank == 1 && a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_reduceat_double(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          return result;
        case DType.float32:
          v_reduceat_float(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          return result;
        case DType.int64:
        case DType.uint64 when isBitwiseOrWrapCompatible:
          v_reduceat_int64(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          if (get_and_reset_division_error() == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        case DType.int32:
        case DType.uint32 when isBitwiseOrWrapCompatible:
          v_reduceat_int32(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          if (get_and_reset_division_error() == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        case DType.int16:
        case DType.uint16 when isBitwiseOrWrapCompatible:
          v_reduceat_int16(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          if (get_and_reset_division_error() == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        case DType.uint8:
        case DType.int8 when isBitwiseOrWrapCompatible:
          v_reduceat_uint8(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          if (get_and_reset_division_error() == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        case DType.boolean:
          v_reduceat_boolean(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          return result;
        case DType.complex128:
          v_reduceat_complex128(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          return result;
        case DType.complex64:
          v_reduceat_complex64(
            a.pointer.cast(),
            axisLen,
            indicesPtr,
            numIndices,
            result.pointer.cast(),
            opCode,
          );
          checkNativeOom();
          return result;
        case DType.float16:
        case DType.bfloat16:
          NDArray.scope(() {
            final doubleA = castNDArray<Float64>(a, DType.float64);
            final doubleRes = NDArray<Float64>.create(
              result.shape,
              DType.float64,
            );
            v_reduceat_double(
              doubleA.pointer.cast(),
              axisLen,
              indicesPtr,
              numIndices,
              doubleRes.pointer.cast(),
              opCode,
            );
            checkNativeOom();
            final casted = castNDArray(doubleRes, result.dtype);
            casted.copy(out: result);
          });
          return result;
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          break;
      }
    }

    final rank = a.rank;
    final cStridesA = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cStridesRes = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    final cShape = ScratchArena.allocate<ffi.Int64>(
      rank * ffi.sizeOf<ffi.Int64>(),
    );
    for (var i = 0; i < rank; i++) {
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
      cShape[i] = a.shape[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_reduceat_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        return result;
      case DType.float32:
        s_reduceat_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        return result;
      case DType.int64:
      case DType.uint64 when isBitwiseOrWrapCompatible:
        s_reduceat_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        if (get_and_reset_division_error() == 1) {
          throw UnsupportedError('Integer division by zero');
        }
        return result;
      case DType.int32:
      case DType.uint32 when isBitwiseOrWrapCompatible:
        s_reduceat_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        if (get_and_reset_division_error() == 1) {
          throw UnsupportedError('Integer division by zero');
        }
        return result;
      case DType.int16:
      case DType.uint16 when isBitwiseOrWrapCompatible:
        s_reduceat_int16(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        if (get_and_reset_division_error() == 1) {
          throw UnsupportedError('Integer division by zero');
        }
        return result;
      case DType.uint8:
      case DType.int8 when isBitwiseOrWrapCompatible:
        s_reduceat_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        if (get_and_reset_division_error() == 1) {
          throw UnsupportedError('Integer division by zero');
        }
        return result;
      case DType.boolean:
        s_reduceat_boolean(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        return result;
      case DType.complex128:
        s_reduceat_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        return result;
      case DType.complex64:
        s_reduceat_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          normAxis,
          indicesPtr,
          numIndices,
          opCode,
        );
        checkNativeOom();
        return result;
      case DType.float16:
      case DType.bfloat16:
        NDArray.scope(() {
          final doubleA = castNDArray<Float64>(a, DType.float64);
          final doubleRes = NDArray<Float64>.create(
            result.shape,
            DType.float64,
          );
          final cStridesDoubleA = ScratchArena.copyInts(doubleA.strides);
          final cStridesDoubleRes = ScratchArena.copyInts(doubleRes.strides);
          s_reduceat_double(
            doubleA.pointer.cast(),
            cStridesDoubleA,
            doubleRes.pointer.cast(),
            cStridesDoubleRes,
            cShape,
            rank,
            normAxis,
            indicesPtr,
            numIndices,
            opCode,
          );
          checkNativeOom();
          final casted = castNDArray(doubleRes, result.dtype);
          casted.copy(out: result);
        });
        return result;
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        NDArray.scope(() {
          final sliceShape = List<int>.of(a.shape)..[normAxis] = 1;
          for (var i = 0; i < numIndices; i++) {
            var start = indicesPtr[i];
            if (start < 0) start += axisLen;
            var end = (i < numIndices - 1) ? indicesPtr[i + 1] : axisLen;
            if (end < 0) end += axisLen;
            if (start < 0) start = 0;
            if (start >= axisLen) start = axisLen - 1;
            if (end > axisLen) end = axisLen;

            final outSlice = NDArray<T>.view(
              result,
              shape: sliceShape,
              strides: result.strides,
              offsetElements: i * result.strides[normAxis],
            );
            final firstSlice = NDArray<T>.view(
              a,
              shape: sliceShape,
              strides: a.strides,
              offsetElements: start * a.strides[normAxis],
            );
            firstSlice.copy(out: outSlice);
            for (var j = start + 1; j < end; j++) {
              final nextSlice = NDArray<T>.view(
                a,
                shape: sliceShape,
                strides: a.strides,
                offsetElements: j * a.strides[normAxis],
              );
              final stepRes = binaryUfunc(outSlice, nextSlice, op: op);
              if (stepRes.dtype == a.dtype) {
                stepRes.copy(out: outSlice);
              } else {
                final casted = castNDArray<T>(stepRes, a.dtype);
                casted.copy(out: outSlice);
              }
            }
          }
        });
        return result;
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Generalized ufunc outer operation.
NDArray<R> outerUfunc<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  BinaryOp op = BinaryOp.multiply,
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute outer on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }

  if (out != null) {
    validateOutBuffer(out);
  }
  if (out != null &&
      (sharesMemory(a, out) ||
          sharesMemory(b, out) ||
          (where != null && sharesMemory(where, out)))) {
    return NDArray.scope(() {
      final temp = where != null
          ? out.copy()
          : _createTyped<R>([...a.shape, ...b.shape], out.dtype);
      outerUfunc<T, R>(a, b, op: op, where: where, out: temp);
      temp.copy(out: out);
      return out;
    });
  }

  final aReshaped = a.reshape([...a.shape, ...List.filled(b.rank, 1)]);
  final bReshaped = b.reshape([...List.filled(a.rank, 1), ...b.shape]);
  try {
    return binaryUfunc<T, R>(
      aReshaped,
      bReshaped,
      op: op,
      where: where,
      out: out,
    );
  } finally {
    aReshaped.dispose();
    bReshaped.dispose();
  }
}

/// Generalized ufunc at operation.
void atUfunc<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<DTypeTag> indices,
  NDArray<DTypeTag> b, {
  required BinaryOp op,
}) {
  if (a.isDisposed || indices.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute at on a disposed array.');
  }
  validateOutBuffer(a, 'a');
  if (a.rank == 0) {
    throw ArgumentError.value(
      a,
      'a',
      'Cannot execute at on a 0-dimensional array',
    );
  }
  if (op == BinaryOp.subtract && a.dtype == DType.boolean) {
    throw UnsupportedError(
      "Boolean subtract, the '-' operator, is not supported; use logicalXor or bitwiseXor instead.",
    );
  }

  if ((a.dtype.isFloating || a.dtype.isComplex) &&
      (op == BinaryOp.gcd ||
          op == BinaryOp.lcm ||
          op == BinaryOp.bitwiseAnd ||
          op == BinaryOp.bitwiseOr ||
          op == BinaryOp.bitwiseXor ||
          op == BinaryOp.leftShift ||
          op == BinaryOp.rightShift)) {
    throw UnsupportedError(
      'Binary operation ${op.name} is not supported on dtype ${a.dtype}',
    );
  }

  final opCode = op.index;
  final rankA = a.rank;
  final axis0Len = a.shape[0];
  final numIndices = indices.size;

  if (numIndices > 0 && axis0Len == 0) {
    throw RangeError(
      'Cannot execute at with non-empty indices on an empty axis of length 0.',
    );
  }

  NDArray.scope(() {
    final marker = ScratchArena.marker;
    try {
      final ffi.Pointer<ffi.Int64> idxPtr = numIndices > 0
          ? ScratchArena.allocate<ffi.Int64>(
              numIndices * ffi.sizeOf<ffi.Int64>(),
            )
          : ffi.nullptr;
      if (numIndices > 0) {
        if (indices.isContiguous && indices.dtype == DType.int64) {
          final rawIdxPtr = indices.pointer.cast<ffi.Int64>();
          for (var i = 0; i < numIndices; i++) {
            var idx = rawIdxPtr[i];
            if (idx < -axis0Len || idx >= axis0Len) {
              throw RangeError.range(idx, -axis0Len, axis0Len - 1, 'indices');
            }
            if (idx < 0) idx += axis0Len;
            idxPtr[i] = idx;
          }
        } else {
          for (var i = 0; i < numIndices; i++) {
            var idx = (indices.getCellFlat(i) as num).toInt();
            if (indices.dtype == DType.uint64 && idx < 0) {
              throw RangeError.range(idx, -axis0Len, axis0Len - 1, 'indices');
            }
            if (idx < -axis0Len || idx >= axis0Len) {
              throw RangeError.range(idx, -axis0Len, axis0Len - 1, 'indices');
            }
            if (idx < 0) idx += axis0Len;
            idxPtr[i] = idx;
          }
        }
      }
      const effectiveStrideIdx = 1;

      final NDArray<T> bTyped = b.dtype == a.dtype
          ? _asView<T>(b)
          : castNDArray<T>(b, a.dtype);
      final expectedBShape = <int>[numIndices, ...a.shape.sublist(1)];
      final NDArray<T> bReshaped =
          (indices.rank > 1 &&
              bTyped.rank == indices.rank + rankA - 1 &&
              listEquals(bTyped.shape.sublist(0, indices.rank), indices.shape))
          ? bTyped.reshape(<int>[
              numIndices,
              ...bTyped.shape.sublist(indices.rank),
            ])
          : bTyped;
      var bReady = broadcastTo<T>(bReshaped, expectedBShape);
      if (numIndices == 0) {
        return;
      }
      if (sharesMemory(a, bReady)) {
        bReady = bReady.copy();
      }

      final rankB = bReady.rank;
      final cBuffer = ScratchArena.getStridedBuffer(rankA * 2 + rankB * 2);
      final cStridesA = cBuffer;
      final cShapeA = cBuffer + rankA;
      final cStridesB = cBuffer + (rankA * 2);
      final cShapeB = cBuffer + (rankA * 2) + rankB;

      for (var i = 0; i < rankA; i++) {
        cStridesA[i] = a.strides[i];
        cShapeA[i] = a.shape[i];
      }
      for (var i = 0; i < rankB; i++) {
        cStridesB[i] = bReady.strides[i];
        cShapeB[i] = bReady.shape[i];
      }

      final isBitwiseOrWrapCompatible =
          op == BinaryOp.bitwiseAnd ||
          op == BinaryOp.bitwiseOr ||
          op == BinaryOp.bitwiseXor ||
          op == BinaryOp.leftShift ||
          op == BinaryOp.add ||
          op == BinaryOp.subtract ||
          op == BinaryOp.multiply;

      switch (a.dtype) {
        case DType.float64:
          s_at_double(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.float32:
          s_at_float(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.int64:
        case DType.uint64 when isBitwiseOrWrapCompatible:
          s_at_int64(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.int32:
        case DType.uint32 when isBitwiseOrWrapCompatible:
          s_at_int32(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.uint8:
        case DType.int8 when isBitwiseOrWrapCompatible:
          s_at_uint8(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.int16:
        case DType.uint16 when isBitwiseOrWrapCompatible:
          s_at_int16(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.complex128:
          s_at_complex128(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.complex64:
          s_at_complex64(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.boolean:
          s_at_boolean(
            a.pointer.cast(),
            cStridesA,
            cShapeA,
            rankA,
            idxPtr,
            numIndices,
            effectiveStrideIdx,
            bReady.pointer.cast(),
            cStridesB,
            cShapeB,
            rankB,
            opCode,
          );
        case DType.float16:
        case DType.bfloat16:
          NDArray.scope(() {
            final doubleA = castNDArray<Float64>(a, DType.float64);
            final doubleB = castNDArray<Float64>(bReady, DType.float64);
            final cStridesDoubleA = ScratchArena.copyInts(doubleA.strides);
            final cStridesDoubleB = ScratchArena.copyInts(doubleB.strides);
            s_at_double(
              doubleA.pointer.cast(),
              cStridesDoubleA,
              cShapeA,
              rankA,
              idxPtr,
              numIndices,
              effectiveStrideIdx,
              doubleB.pointer.cast(),
              cStridesDoubleB,
              cShapeB,
              rankB,
              opCode,
            );
            checkNativeOom();
            final castedBack = castNDArray(doubleA, a.dtype);
            castedBack.copy(out: a);
          });
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          final sliceShape = a.shape.sublist(1);
          final sliceStridesA = a.strides.sublist(1);
          final sliceStridesB = bReady.strides.sublist(1);
          for (var i = 0; i < numIndices; i++) {
            final idx = idxPtr[i];
            final aSlice = NDArray<T>.view(
              a,
              shape: sliceShape,
              strides: sliceStridesA,
              offsetElements: idx * a.strides[0],
            );
            final bSlice = NDArray<T>.view(
              bReady,
              shape: sliceShape,
              strides: sliceStridesB,
              offsetElements: i * bReady.strides[0],
            );
            final stepRes = binaryUfunc(aSlice, bSlice, op: op);
            if (stepRes.dtype == a.dtype) {
              stepRes.copy(out: aSlice);
            } else {
              final casted = castNDArray<T>(stepRes, a.dtype);
              casted.copy(out: aSlice);
            }
          }
      }
      checkNativeOom();
    } finally {
      ScratchArena.reset(marker);
    }
    if (a.dtype.isInteger) {
      if (get_and_reset_division_error() == 1) {
        throw UnsupportedError('Integer division by zero');
      }
    }
  });
}

/// Evaluates unary operation [op] element-wise on [x].
NDArray<R> unaryUfunc<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> x, {
  required UnaryOp op,
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute unaryUfunc() on a disposed array.');
  }
  return _withView<AnySpec, NDArray<R>>(x, (xSpec) {
    switch (op) {
      case UnaryOp.invert:
      case UnaryOp.bitwiseNot:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => invert(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.negative:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => negative(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.positive:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => positive(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.absolute:
      case UnaryOp.abs:
      case UnaryOp.fabs:
        final res = abs<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.rint:
        final res = rint<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.sign:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => sign(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.conj:
      case UnaryOp.conjugate:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => conj(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.exp:
        final res = exp<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.exp2:
        return NDArray.scope(() {
          final targetDType =
              (x.dtype == DType.complex128 ||
                  x.dtype == DType.complex64 ||
                  x.dtype == DType.float32)
              ? x.dtype
              : DType.float64;
          final xCast = x.dtype == targetDType
              ? x
              : castNDArray(x, targetDType);
          final base = NDArray.scalar(
            targetDType.isComplex ? Complex(2.0, 0.0) : 2.0,
            dtype: targetDType,
          );
          final res = power<DTypeTag>(base, xCast, where: where, out: out);
          return out ?? _coerceOwned<R>(res).detachToParentScope();
        });
      case UnaryOp.log:
        final res = log<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.log2:
        final res = log2<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.log10:
        final res = log10<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.expm1:
        final res = expm1<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.log1p:
        final res = log1p<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.sqrt:
        final res = sqrt<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.square:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => square(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.cbrt:
        if (x.dtype.isComplex) {
          throw UnsupportedError('cbrt is not supported for complex numbers.');
        }
        return NDArray.scope(() {
          final targetDType = x.dtype == DType.float32
              ? DType.float32
              : DType.float64;
          final xCast = x.dtype == targetDType
              ? x
              : castNDArray(x, targetDType);
          final absX = _withView<AnySpec, NDArray<DTypeTag>>(
            xCast,
            (xView) => abs<DTypeTag>(xView),
          );
          final expScalar = NDArray.scalar(1.0 / 3.0, dtype: targetDType);
          final mag = power<DTypeTag>(absX, expScalar);
          final res = copysign<DTypeTag>(mag, xCast, where: where, out: out);
          return out ?? _coerceOwned<R>(res).detachToParentScope();
        });
      case UnaryOp.reciprocal:
        final res = reciprocal<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.sin:
        final res = sin<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.cos:
        final res = cos<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.tan:
        final res = tan<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.arcsin:
        final res = asin<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.arccos:
        final res = acos<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.arctan:
        final res = atan<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.sinh:
        final res = sinh<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.cosh:
        final res = cosh<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.tanh:
        final res = tanh<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.arcsinh:
        final res = asinh<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.arccosh:
        final res = acosh<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.arctanh:
        final res = atanh<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.degrees:
      case UnaryOp.rad2deg:
        final res = rad2deg<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.radians:
      case UnaryOp.deg2rad:
        final res = deg2rad<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.logicalNot:
        final res = _withViewNullable<Boolean, NDArray<Boolean>>(
          out,
          (outView) => logicalNot(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.isnan:
        final res = _withViewNullable<Boolean, NDArray<Boolean>>(
          out,
          (outView) => isnan(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.isinf:
        final res = _withViewNullable<Boolean, NDArray<Boolean>>(
          out,
          (outView) => isinf(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.isfinite:
        final res = _withViewNullable<Boolean, NDArray<Boolean>>(
          out,
          (outView) => isfinite(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.signbit:
        if (x.dtype.isComplex) {
          throw UnsupportedError(
            'signbit is not supported for complex numbers.',
          );
        }
        return NDArray.scope(() {
          if (x.dtype == DType.boolean ||
              x.dtype == DType.uint64 ||
              x.dtype == DType.uint32 ||
              x.dtype == DType.uint16 ||
              x.dtype == DType.uint8) {
            final falseArr = NDArray<Boolean>.zeros(x.shape, DType.boolean);
            final res = _withViewNullable<Boolean, NDArray<Boolean>>(
              out,
              (outView) =>
                  logicalAnd(falseArr, falseArr, where: where, out: outView),
            );
            return out ?? _coerceOwned<R>(res).detachToParentScope();
          }
          final zero = NDArray.scalar(0, dtype: x.dtype);
          final src = x.dtype.isFloating
              ? copysign(NDArray.scalar(1, dtype: x.dtype), x)
              : x;
          final res = _withViewNullable<Boolean, NDArray<Boolean>>(
            out,
            (outView) => less(src, zero, where: where, out: outView),
          );
          return out ?? _coerceOwned<R>(res).detachToParentScope();
        });
      case UnaryOp.floor:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => floor(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.ceil:
        final res = _withViewNullable<T, NDArray<T>>(
          out,
          (outView) => ceil(x, where: where, out: outView),
        );
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.trunc:
        final res = trunc<DTypeTag>(xSpec, where: where, out: out);
        return out ?? _coerceOwned<R>(res);
      case UnaryOp.spacing:
        if (x.dtype.isComplex) {
          throw UnsupportedError(
            'spacing is not supported for complex numbers.',
          );
        }
        return NDArray.scope(() {
          final targetDType = x.dtype == DType.float32
              ? DType.float32
              : DType.float64;
          if (out != null) {
            validateOutBuffer(out);
            if (!listEquals(out.shape, x.shape) || out.dtype != targetDType) {
              throw ArgumentError.value(
                out,
                'out',
                'Must have compatible shape and dtype',
              );
            }
          }
          final maskHolder = prepareMask(where, x.shape);
          try {
            final temp = _createTyped<R>(x.shape, targetDType);
            if (where != null) {
              if (out != null) {
                out.copy(out: temp);
              } else {
                temp.fill(0.0);
              }
            }
            final bd = ByteData(8);
            final isF32 = targetDType == DType.float32;
            final xIter = NDIter(x);
            final tIter = NDIter(temp);
            final maskPtr = maskHolder.pointer;
            var flatIdx = 0;
            while (xIter.moveNext() && tIter.moveNext()) {
              final curIdx = flatIdx++;
              if (maskPtr != ffi.nullptr && maskPtr[curIdx] == 0) {
                continue;
              }
              final rawVal = x.getCellRaw(xIter.index);
              final double v = rawVal is bool
                  ? (rawVal ? 1.0 : 0.0)
                  : (rawVal as num).toDouble();
              double ulp;
              if (v.isNaN || v.isInfinite) {
                ulp = double.nan;
              } else if (isF32) {
                bd.setFloat32(0, v);
                final bits = bd.getUint32(0);
                final signBit = bits & 0x80000000;
                final expBits = (bits >> 23) & 0xFF;
                if (expBits == 0xFF) {
                  ulp = double.nan;
                } else {
                  final ulpBits = expBits <= 24 ? 1 : ((expBits - 23) << 23);
                  bd.setUint32(0, signBit | ulpBits);
                  ulp = bd.getFloat32(0);
                }
              } else {
                bd.setFloat64(0, v);
                final bits = bd.getUint64(0);
                final signBit = bits & 0x8000000000000000;
                final expBits = (bits >> 52) & 0x7FF;
                if (expBits == 0x7FF) {
                  ulp = double.nan;
                } else {
                  final ulpBits = expBits <= 53 ? 1 : ((expBits - 52) << 52);
                  bd.setUint64(0, signBit | ulpBits);
                  ulp = bd.getFloat64(0);
                }
              }
              temp.setCellRaw(tIter.index, ulp);
            }
            if (out != null) {
              temp.copy(out: out);
              return out;
            }
            return temp.detachToParentScope();
          } finally {
            maskHolder.dispose();
          }
        });
    }
  });
}
