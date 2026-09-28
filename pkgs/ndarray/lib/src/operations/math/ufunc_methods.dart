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

import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import 'binary_op.dart';
import 'arithmetic.dart';
import 'bitwise.dart';
import 'logical.dart';
import 'floating_point.dart';
import 'trigonometric.dart';
import '../sorting.dart' show where;

/// Extension methods for generalized ufunc operations on [NDArray].
extension UfuncNDArrayExtension<T extends Object> on NDArray<T> {
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
    T? initial,
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
    NDArray<int> indices, {
    required BinaryOp op,
    int axis = 0,
    NDArray<T>? out,
  }) => reduceatUfunc(this, indices, op: op, axis: axis, out: out);

  /// Performs an outer binary operation between this array and [b] using [op].
  ///
  /// **Preconditions:**
  /// - It is an error if this array, [b], or [out] is disposed.
  /// - It is an error if [out] (if provided) has incompatible shape or dtype.
  NDArray<T> outer(
    NDArray<T> b, {
    BinaryOp op = BinaryOp.multiply,
    NDArray<dynamic>? where,
    NDArray<T>? out,
  }) => outerUfunc(this, b, op: op, where: where, out: out);

  /// Performs unbuffered in-place scatter updates on this array at [indices] using [b] and [op].
  ///
  /// **Preconditions:**
  /// - It is an error if this array, [indices], or [b] is disposed.
  void at(NDArray<int> indices, NDArray<T> b, {required BinaryOp op}) =>
      atUfunc(this, indices, b, op: op);
}

/// Evaluates binary operation [op] element-wise between [a] and [b].
NDArray<R> binaryUfunc<T extends Object, R extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  required BinaryOp op,
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  switch (op) {
    case BinaryOp.add:
      return add(a, b, where: where, out: out as NDArray<T>?) as NDArray<R>;
    case BinaryOp.subtract:
      return subtract(a, b, where: where, out: out as NDArray<T>?)
          as NDArray<R>;
    case BinaryOp.multiply:
      return multiply(a, b, where: where, out: out as NDArray<T>?)
          as NDArray<R>;
    case BinaryOp.divide:
      return divide(a, b, where: where, out: out as NDArray<double>?)
          as NDArray<R>;
    case BinaryOp.floorDivide:
      return floor_divide(a, b, where: where, out: out as NDArray<T>?)
          as NDArray<R>;
    case BinaryOp.remainder:
      return remainder(a, b, where: where, out: out as NDArray<T>?)
          as NDArray<R>;
    case BinaryOp.fmod:
      return fmod(a, b, where: where, out: out as NDArray<T>?) as NDArray<R>;
    case BinaryOp.gcd:
      return gcd(a, b, where: where, out: out as NDArray<T>?) as NDArray<R>;
    case BinaryOp.lcm:
      return lcm(a, b, where: where, out: out as NDArray<T>?) as NDArray<R>;
    case BinaryOp.heaviside:
      return heaviside(a, b, where: where, out: out as NDArray<T>?)
          as NDArray<R>;
    case BinaryOp.power:
      return power(a, b, where: where, out: out as NDArray<T>?) as NDArray<R>;
    case BinaryOp.floatPower:
      return power(a, b, where: where, out: out as NDArray<double>?)
          as NDArray<R>;
    case BinaryOp.logaddexp:
      return logaddexp<num, num>(
        a as NDArray<num>,
        b as NDArray<num>,
        where: where,
        out: out as NDArray<double>?,
      ) as NDArray<R>;
    case BinaryOp.logaddexp2:
      return logaddexp2<num, num>(
        a as NDArray<num>,
        b as NDArray<num>,
        where: where,
        out: out as NDArray<double>?,
      ) as NDArray<R>;
    case BinaryOp.arctan2:
      return atan2<num, num>(
        a as NDArray<num>,
        b as NDArray<num>,
        where: where,
        out: out as NDArray<double>?,
      ) as NDArray<R>;
    case BinaryOp.hypot:
      return hypot<dynamic, dynamic, R>(a, b, where: where, out: out);
    case BinaryOp.copysign:
      return copysign<R>(
        a as NDArray<R>,
        b as NDArray<R>,
        where: where,
        out: out,
      );
    case BinaryOp.bitwiseAnd:
      return bitwise_and<dynamic, dynamic, R>(a, b, where: where, out: out);
    case BinaryOp.bitwiseOr:
      return bitwise_or<dynamic, dynamic, R>(a, b, where: where, out: out);
    case BinaryOp.bitwiseXor:
      return bitwise_xor<dynamic, dynamic, R>(a, b, where: where, out: out);
    case BinaryOp.leftShift:
      return left_shift<dynamic, dynamic, R>(a, b, where: where, out: out);
    case BinaryOp.rightShift:
      return right_shift<dynamic, dynamic, R>(a, b, where: where, out: out);
    case BinaryOp.logicalAnd:
      return logical_and(a, b, where: where, out: out as NDArray<bool>?)
          as NDArray<R>;
    case BinaryOp.logicalOr:
      return logical_or(a, b, where: where, out: out as NDArray<bool>?)
          as NDArray<R>;
    case BinaryOp.logicalXor:
      return logical_xor(a, b, where: where, out: out as NDArray<bool>?)
          as NDArray<R>;
    case BinaryOp.minimum:
    case BinaryOp.fmin:
      return _elementwiseMin(a, b, whereMask: where, out: out);
    case BinaryOp.maximum:
    case BinaryOp.fmax:
      return _elementwiseMax(a, b, whereMask: where, out: out);
    default:
      throw UnsupportedError(
        'Binary operation ${op.name} is not implemented for binaryUfunc.',
      );
  }
}

NDArray<R> _elementwiseMin<T extends Object, R extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<dynamic>? whereMask,
  NDArray<R>? out,
}) {
  final outCopy = out?.copy();
  final cond = less(a, b);
  final result = where(cond, a, b, out as NDArray<T>?);
  cond.dispose();
  if (whereMask != null) {
    if (outCopy != null) {
      where(whereMask as NDArray<bool>, result, outCopy, out);
      outCopy.dispose();
    } else {
      where(whereMask as NDArray<bool>, result, a, result);
    }
  }
  return result as NDArray<R>;
}

NDArray<R> _elementwiseMax<T extends Object, R extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<dynamic>? whereMask,
  NDArray<R>? out,
}) {
  final outCopy = out?.copy();
  final cond = greater(a, b);
  final result = where(cond, a, b, out as NDArray<T>?);
  cond.dispose();
  if (whereMask != null) {
    if (outCopy != null) {
      where(whereMask as NDArray<bool>, result, outCopy, out);
      outCopy.dispose();
    } else {
      where(whereMask as NDArray<bool>, result, a, result);
    }
  }
  return result as NDArray<R>;
}

/// Reduces [a] along [axis] using [op].
///
/// **Preconditions:**
/// - It is an error if [op] is not reducible ([op.isReducible] is false).
/// - It is an error if [a] or [out] (if provided) is disposed.
/// - It is an error if [axis] is not within `[-rank, rank - 1]`.
/// - It is an error if [a] is empty without [initial].
/// - It is an error if [out] (if provided) has incompatible shape or dtype.
NDArray<T> reduce<T extends Object>(
  NDArray<T> a, {
  required BinaryOp op,
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
  T? initial,
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
NDArray<T> accumulate<T extends Object>(
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
NDArray<T> reduceat<T extends Object>(
  NDArray<T> a,
  NDArray<int> indices, {
  required BinaryOp op,
  int axis = 0,
  NDArray<T>? out,
}) => reduceatUfunc(a, indices, op: op, axis: axis, out: out);

/// Performs unbuffered in-place scatter updates on [a] at [indices] using [b] and [op].
///
/// **Preconditions:**
/// - It is an error if [a], [indices], or [b] is disposed.
void at<T extends Object>(
  NDArray<T> a,
  NDArray<int> indices,
  NDArray<T> b, {
  required BinaryOp op,
}) => atUfunc(a, indices, b, op: op);

/// Generalized ufunc reduction function.
NDArray<T> reduceUfunc<T extends Object>(
  NDArray<T> a, {
  required BinaryOp op,
  int? axis,
  bool keepdims = false,
  NDArray<T>? out,
  T? initial,
}) {
  if (!op.isReducible) {
    throw ArgumentError('Operation ${op.name} is not reducible.');
  }
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute reduce on a disposed array.');
  }

  if (axis == null) {
    // Global reduction
    if (a.size == 0 && initial == null) {
      throw ArgumentError(
        'Cannot reduce an empty array without an initial value.',
      );
    }
    final targetShape = keepdims ? List.filled(a.rank, 1) : <int>[];
    final NDArray<T> result;
    if (out != null) {
      if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype for reduce.',
        );
      }
      result = out;
    } else {
      result = NDArray.create(targetShape, a.dtype);
    }

    if (a.size == 0) {
      result.fill(initial!);
      return result;
    }

    if (a.isContiguous && initial == null) {
      bool handled = false;
      switch (op) {
        case BinaryOp.add:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_sum_double(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.float32:
              result.fill(r_sum_float(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int64:
              result.fill(r_sum_int64(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int32:
              result.fill(r_sum_int32(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.uint8:
              result.fill(r_sum_uint8(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int16:
              result.fill(r_sum_int16(a.pointer.cast(), a.size) as T);
              handled = true;
            default:
              break;
          }
        case BinaryOp.multiply:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_prod_double(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.float32:
              result.fill(r_prod_float(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int64:
              result.fill(r_prod_int64(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int32:
              result.fill(r_prod_int32(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.uint8:
              result.fill(r_prod_uint8(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int16:
              result.fill(r_prod_int16(a.pointer.cast(), a.size) as T);
              handled = true;
            default:
              break;
          }
        case BinaryOp.minimum:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_min_double(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.float32:
              result.fill(r_min_float(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int64:
              result.fill(r_min_int64_t(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int32:
              result.fill(r_min_int32_t(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.uint8:
              result.fill(r_min_uint8_t(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int16:
              result.fill(r_min_int16_t(a.pointer.cast(), a.size) as T);
              handled = true;
            default:
              break;
          }
        case BinaryOp.maximum:
          switch (a.dtype) {
            case DType.float64:
              result.fill(r_max_double(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.float32:
              result.fill(r_max_float(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int64:
              result.fill(r_max_int64_t(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int32:
              result.fill(r_max_int32_t(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.uint8:
              result.fill(r_max_uint8_t(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int16:
              result.fill(r_max_int16_t(a.pointer.cast(), a.size) as T);
              handled = true;
            default:
              break;
          }
        case BinaryOp.bitwiseAnd:
          switch (a.dtype) {
            case DType.int64:
              result.fill(r_bitwise_and_int64(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int32:
              result.fill(r_bitwise_and_int32(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.uint8:
              result.fill(r_bitwise_and_uint8(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int16:
              result.fill(r_bitwise_and_int16(a.pointer.cast(), a.size) as T);
              handled = true;
            default:
              break;
          }
        case BinaryOp.bitwiseOr:
          switch (a.dtype) {
            case DType.int64:
              result.fill(r_bitwise_or_int64(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int32:
              result.fill(r_bitwise_or_int32(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.uint8:
              result.fill(r_bitwise_or_uint8(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int16:
              result.fill(r_bitwise_or_int16(a.pointer.cast(), a.size) as T);
              handled = true;
            default:
              break;
          }
        case BinaryOp.bitwiseXor:
          switch (a.dtype) {
            case DType.int64:
              result.fill(r_bitwise_xor_int64(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int32:
              result.fill(r_bitwise_xor_int32(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.uint8:
              result.fill(r_bitwise_xor_uint8(a.pointer.cast(), a.size) as T);
              handled = true;
            case DType.int16:
              result.fill(r_bitwise_xor_int16(a.pointer.cast(), a.size) as T);
              handled = true;
            default:
              break;
          }
        case BinaryOp.logicalAnd:
          if (a.dtype == DType.boolean) {
            result.fill((r_logical_and(a.pointer.cast(), a.size) != 0) as T);
            handled = true;
          }
        case BinaryOp.logicalOr:
          if (a.dtype == DType.boolean) {
            result.fill((r_logical_or(a.pointer.cast(), a.size) != 0) as T);
            handled = true;
          }
        case BinaryOp.logicalXor:
          if (a.dtype == DType.boolean) {
            result.fill((r_logical_xor(a.pointer.cast(), a.size) != 0) as T);
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
    if (!listEquals(out.shape, resShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for reduce.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(resShape, a.dtype);
  }

  if (a.shape[normAxis] == 0) {
    if (initial != null) {
      result.fill(initial);
      return result;
    }
    throw ArgumentError(
      'Cannot reduce array of size 0 along axis $axis without an initial value.',
    );
  }

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
            default:
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
            default:
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
            default:
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
            default:
              break;
          }
        case BinaryOp.bitwiseAnd:
          switch (a.dtype) {
            case DType.int64:
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
            default:
              break;
          }
        case BinaryOp.bitwiseOr:
          switch (a.dtype) {
            case DType.int64:
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
            default:
              break;
          }
        case BinaryOp.bitwiseXor:
          switch (a.dtype) {
            case DType.int64:
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
            default:
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

  if (handled) return result;

  // Fallback axis reduction via Index slicing and binary ufunc
  final axisLen = a.shape[normAxis];
  NDArray<T> current;
  if (initial != null) {
    current = NDArray.create(resShape, a.dtype);
    current.fill(initial);
    for (var i = 0; i < axisLen; i++) {
      final selectors = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final sub = a.slice(selectors);
      binaryUfunc(current, sub, op: op, out: current);
      sub.dispose();
    }
  } else {
    final selectors0 = List<Selector>.generate(
      a.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    current = a.slice(selectors0).copy();
    for (var i = 1; i < axisLen; i++) {
      final selectorsI = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final sub = a.slice(selectorsI);
      final next = binaryUfunc(current, sub, op: op) as NDArray<T>;
      current.dispose();
      sub.dispose();
      current = next;
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
NDArray<T> accumulateUfunc<T extends Object>(
  NDArray<T> a, {
  required BinaryOp op,
  int axis = 0,
  NDArray<T>? out,
}) {
  if (!op.isReducible) {
    throw ArgumentError('Operation ${op.name} is not reducible.');
  }
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute accumulate on a disposed array.');
  }

  final normAxis = axis < 0 ? axis + a.rank : axis;
  if (normAxis < 0 || normAxis >= a.rank) {
    throw RangeError.range(normAxis, 0, a.rank - 1, 'axis');
  }

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for accumulate.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }

  final marker = ScratchArena.marker;
  bool handled = false;
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
          default:
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
          default:
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
          default:
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
          default:
            break;
        }
      case BinaryOp.bitwiseAnd:
        switch (a.dtype) {
          case DType.int64:
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
          default:
            break;
        }
      case BinaryOp.bitwiseOr:
        switch (a.dtype) {
          case DType.int64:
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
          default:
            break;
        }
      case BinaryOp.bitwiseXor:
        switch (a.dtype) {
          case DType.int64:
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
          default:
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

  if (handled) return result;

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
      stepRes.copy(out: resSliceI);
      resSliceI.dispose();
      prev.dispose();
      curr.dispose();
      stepRes.dispose();
    }
  }
  return result;
}

/// Generalized ufunc reduceat function.
NDArray<T> reduceatUfunc<T extends Object>(
  NDArray<T> a,
  NDArray<int> indices, {
  required BinaryOp op,
  int axis = 0,
  NDArray<T>? out,
}) {
  if (!op.isReducible) {
    throw ArgumentError('Operation ${op.name} is not reducible.');
  }
  if (a.isDisposed || indices.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute reduceat on a disposed array.');
  }

  final normAxis = axis < 0 ? axis + a.rank : axis;
  if (normAxis < 0 || normAxis >= a.rank) {
    throw RangeError.range(normAxis, 0, a.rank - 1, 'axis');
  }

  final numIndices = indices.size;
  final axisLen = a.shape[normAxis];
  final resShape = List<int>.from(a.shape);
  resShape[normAxis] = numIndices;

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, resShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for reduceat.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(resShape, a.dtype);
  }

  final opCode = op.index;
  if (numIndices > 0) {
    final marker = ScratchArena.marker;
    try {
      final ffi.Pointer<ffi.Int64> indicesPtr;
      if (indices.isContiguous && indices.dtype == DType.int64) {
        indicesPtr = indices.pointer.cast<ffi.Int64>();
      } else {
        indicesPtr = ScratchArena.allocate<ffi.Int64>(
          numIndices * ffi.sizeOf<ffi.Int64>(),
        );
        final ptr = indices.pointer;
        if (indices.dtype == DType.int32) {
          final p32 = ptr.cast<ffi.Int32>();
          for (var i = 0; i < numIndices; i++) {
            indicesPtr[i] = p32[i];
          }
        } else {
          final p64 = ptr.cast<ffi.Int64>();
          for (var i = 0; i < numIndices; i++) {
            indicesPtr[i] = p64[i];
          }
        }
      }

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
            return result;
          case DType.int64:
            v_reduceat_int64(
              a.pointer.cast(),
              axisLen,
              indicesPtr,
              numIndices,
              result.pointer.cast(),
              opCode,
            );
            return result;
          case DType.int32:
            v_reduceat_int32(
              a.pointer.cast(),
              axisLen,
              indicesPtr,
              numIndices,
              result.pointer.cast(),
              opCode,
            );
            return result;
          case DType.int16:
            v_reduceat_int16(
              a.pointer.cast(),
              axisLen,
              indicesPtr,
              numIndices,
              result.pointer.cast(),
              opCode,
            );
            return result;
          case DType.uint8:
          case DType.boolean:
            v_reduceat_uint8(
              a.pointer.cast(),
              axisLen,
              indicesPtr,
              numIndices,
              result.pointer.cast(),
              opCode,
            );
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
            return result;
          case DType.float16:
          case DType.bfloat16:
          case DType.int8:
          case DType.uint64:
          case DType.uint32:
          case DType.uint16:
            final doubleA = NDArray.fromList(
              a.toList().cast<num>().map((e) => e.toDouble()).toList(),
              a.shape,
              DType.float64,
            );
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
            final casted = NDArray.fromList(
              doubleRes.toList(),
              doubleRes.shape,
              result.dtype,
            );
            casted.copy(out: result);
            doubleA.dispose();
            doubleRes.dispose();
            casted.dispose();
            return result;
        }
      }

      final rank = a.rank;
      final cStridesA = ScratchArena.allocate<ffi.Int>(
        rank * ffi.sizeOf<ffi.Int>(),
      );
      final cStridesRes = ScratchArena.allocate<ffi.Int>(
        rank * ffi.sizeOf<ffi.Int>(),
      );
      final cShape = ScratchArena.allocate<ffi.Int>(
        rank * ffi.sizeOf<ffi.Int>(),
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
          return result;
        case DType.int64:
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
          return result;
        case DType.int32:
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
          return result;
        case DType.int16:
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
          return result;
        case DType.uint8:
        case DType.boolean:
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
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
          final doubleA = NDArray.fromList(
            a.toList().cast<num>().map((e) => e.toDouble()).toList(),
            a.shape,
            DType.float64,
          );
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
          final casted = NDArray.fromList(
            doubleRes.toList(),
            doubleRes.shape,
            result.dtype,
          );
          casted.copy(out: result);
          doubleA.dispose();
          doubleRes.dispose();
          casted.dispose();
          return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }

  final idxView = indices.reshape([indices.size]);
  for (var i = 0; i < numIndices; i++) {
    var start = (idxView.getCell([i]) as num).toInt();
    if (start < 0) start += axisLen;
    var end = (i < numIndices - 1)
        ? (idxView.getCell([i + 1]) as num).toInt()
        : axisLen;
    if (end < 0) end += axisLen;

    final destSelectors = List<Selector>.generate(
      result.rank,
      (d) => d == normAxis ? Slice(start: i, stop: i + 1) : Slice(),
    );
    final destSlice = result.slice(destSelectors);

    if (start >= end) {
      final selectors = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Slice(start: start, stop: start + 1) : Slice(),
      );
      final single = a.slice(selectors);
      single.copy(out: destSlice);
      single.dispose();
    } else {
      final selectors = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Slice(start: start, stop: end) : Slice(),
      );
      final rangeSlice = a.slice(selectors);
      final reduced = reduceUfunc(
        rangeSlice,
        op: op,
        axis: normAxis,
        keepdims: true,
      );
      reduced.copy(out: destSlice);
      rangeSlice.dispose();
      reduced.dispose();
    }
    destSlice.dispose();
  }

  return result;
}

/// Generalized ufunc outer operation.
NDArray<T> outerUfunc<T extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  BinaryOp op = BinaryOp.multiply,
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute outer on a disposed array.');
  }

  final aReshaped = a.reshape([...a.shape, ...List.filled(b.rank, 1)]);
  final bReshaped = b.reshape([...List.filled(a.rank, 1), ...b.shape]);

  final result = binaryUfunc(
    aReshaped,
    bReshaped,
    op: op,
    where: where,
    out: out,
  );

  aReshaped.dispose();
  bReshaped.dispose();

  return result;
}

/// Generalized ufunc at operation.
void atUfunc<T extends Object>(
  NDArray<T> a,
  NDArray<int> indices,
  NDArray<T> b, {
  required BinaryOp op,
}) {
  if (a.isDisposed || indices.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute at on a disposed array.');
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
  final rankB = b.rank;
  final numIndices = indices.size;
  final strideIdx = indices.strides.isEmpty ? 1 : indices.strides[0];

  final marker = ScratchArena.marker;
  try {
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
      cStridesB[i] = b.strides[i];
      cShapeB[i] = b.shape[i];
    }
    final ffi.Pointer<ffi.Int64> idxPtr;
    final int effectiveStrideIdx;
    if (indices.isContiguous && indices.dtype == DType.int64) {
      idxPtr = indices.pointer.cast<ffi.Int64>();
      effectiveStrideIdx = strideIdx;
    } else {
      idxPtr = ScratchArena.allocate<ffi.Int64>(numIndices);
      for (var i = 0; i < numIndices; i++) {
        final val = indices.getCell([i]) as num;
        idxPtr[i] = val.toInt();
      }
      effectiveStrideIdx = 1;
    }

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
          b.pointer.cast(),
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
          b.pointer.cast(),
          cStridesB,
          cShapeB,
          rankB,
          opCode,
        );
      case DType.int64:
        s_at_int64(
          a.pointer.cast(),
          cStridesA,
          cShapeA,
          rankA,
          idxPtr,
          numIndices,
          effectiveStrideIdx,
          b.pointer.cast(),
          cStridesB,
          cShapeB,
          rankB,
          opCode,
        );
      case DType.int32:
        s_at_int32(
          a.pointer.cast(),
          cStridesA,
          cShapeA,
          rankA,
          idxPtr,
          numIndices,
          effectiveStrideIdx,
          b.pointer.cast(),
          cStridesB,
          cShapeB,
          rankB,
          opCode,
        );
      case DType.uint8:
        s_at_uint8(
          a.pointer.cast(),
          cStridesA,
          cShapeA,
          rankA,
          idxPtr,
          numIndices,
          effectiveStrideIdx,
          b.pointer.cast(),
          cStridesB,
          cShapeB,
          rankB,
          opCode,
        );
      case DType.int16:
        s_at_int16(
          a.pointer.cast(),
          cStridesA,
          cShapeA,
          rankA,
          idxPtr,
          numIndices,
          effectiveStrideIdx,
          b.pointer.cast(),
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
          b.pointer.cast(),
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
          b.pointer.cast(),
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
          b.pointer.cast(),
          cStridesB,
          cShapeB,
          rankB,
          opCode,
        );
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
        final doubleA = NDArray.fromList(
          a.toList().cast<num>().map((e) => e.toDouble()).toList(),
          a.shape,
          DType.float64,
        );
        final doubleB = NDArray.fromList(
          b.toList().cast<num>().map((e) => e.toDouble()).toList(),
          b.shape,
          DType.float64,
        );
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
        final castedBack = NDArray.fromList(
          doubleA.toList(),
          doubleA.shape,
          a.dtype,
        );
        castedBack.copy(out: a);
        doubleA.dispose();
        doubleB.dispose();
        castedBack.dispose();
    }
  } finally {
    ScratchArena.reset(marker);
  }
}
