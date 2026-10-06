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
import 'dart:ffi' as ffi;
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';

import '../broadcasting.dart';
import '../helpers.dart';
import '../native_pointer.dart';
import '../stats.dart';
import 'logical.dart';

/// Computes the element-wise square root of the array.
///
/// Returns a new array with the results.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// **Edge cases:**
/// - Negative values will result in [double.nan].
NDArray<R> sqrt<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute sqrt() on a disposed array.');
  }
  final DType<R> targetDType;
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    targetDType = a.dtype as DType<R>;
  } else {
    targetDType =
        ((a.dtype as DType<DTypeTag>) == DType.float32
                ? DType.float32
                : DType.float64)
            as DType<R>;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for sqrt',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        sqrt<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ?? NDArray.create(a.shape, targetDType, zeroInit: where != null);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_sqrt_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_sqrt_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_sqrt_complex128(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_sqrt_complex64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
          break;
      }
    }

    if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
        (a.dtype as DType<DTypeTag>) == DType.complex64) {
      final rank = a.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesRes = cBuffer + (rank * 2);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
          cStridesRes[i] = result.strides[i];
        }
        if ((a.dtype as DType<DTypeTag>) == DType.complex128) {
          s_sqrt_complex128(
            a.typedPointer(),
            cStridesA,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        } else {
          s_sqrt_complex64(
            a.typedPointer(),
            cStridesA,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        }
        return result;
      } finally {
        ScratchArena.reset(marker);
      }
    }

    final temp = a.isContiguous ? a : a.copy();

    double toDoubleUnsigned(Object? val) {
      if ((temp.dtype as DType<DTypeTag>) == DType.uint64 && val is int) {
        return BigInt.from(val).toUnsigned(64).toDouble();
      }
      if (val is bool) {
        return val ? 1.0 : 0.0;
      }
      return (val as num).toDouble();
    }

    if (result.isContiguous &&
        !sharesMemory(temp, result) &&
        (where == null || !sharesMemory(where, result))) {
      for (var i = 0; i < temp.size; i++) {
        if (maskHolder.pointer == ffi.nullptr || maskHolder.pointer[i] != 0) {
          result.setCellFlat(
            i,
            castValue(
              math.sqrt(toDoubleUnsigned(temp.getCellFlat(i))),
              result.dtype,
            ),
          );
        }
      }
    } else {
      final tempOut = result.copy();
      for (var i = 0; i < temp.size; i++) {
        if (maskHolder.pointer == ffi.nullptr || maskHolder.pointer[i] != 0) {
          tempOut.setCellFlat(
            i,
            castValue(
              math.sqrt(toDoubleUnsigned(temp.getCellFlat(i))),
              result.dtype,
            ),
          );
        }
      }
      tempOut.copy(out: result);
      tempOut.dispose();
    }

    if (!identical(temp, a)) {
      temp.dispose();
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

Complex _complexExpm1(Complex z) {
  final a = z.real;
  final b = z.imag;
  double expm1Val(double x) {
    if (x.abs() < 1e-5) return x + 0.5 * x * x + (1.0 / 6.0) * x * x * x;
    return math.exp(x) - 1.0;
  }

  final ea = expm1Val(a);
  final expa = ea + 1.0;
  final realPart =
      ea * math.cos(b) - 2.0 * math.sin(b / 2.0) * math.sin(b / 2.0);
  final imagPart = expa * math.sin(b);
  return Complex(realPart, imagPart);
}

Complex _complexLog1p(Complex z) {
  final x = z.real;
  final y = z.imag;
  final absVal = math.sqrt(x * x + y * y);
  double log1pVal(double v) {
    if (v.abs() < 1e-5) return v - 0.5 * v * v + (1.0 / 3.0) * v * v * v;
    return math.log(1.0 + v);
  }

  if (absVal < 0.375) {
    return Complex(
      0.5 * log1pVal(2.0 * x + x * x + y * y),
      math.atan2(y, 1.0 + x),
    );
  } else {
    final rx = 1.0 + x;
    final ry = y;
    return Complex(math.log(math.sqrt(rx * rx + ry * ry)), math.atan2(ry, rx));
  }
}

double _logaddexp(double x, double y) {
  if (x.isNaN || y.isNaN) return double.nan;
  if (x == double.negativeInfinity && y == double.negativeInfinity) {
    return double.negativeInfinity;
  }
  if (x == y) return x + 0.6931471805599453;
  final maxVal = x > y ? x : y;
  final minVal = x > y ? y : x;
  double log1pVal(double v) {
    if (v.abs() < 1e-5) return v - 0.5 * v * v + (1.0 / 3.0) * v * v * v;
    return math.log(1.0 + v);
  }

  return maxVal + log1pVal(math.exp(minVal - maxVal));
}

double _logaddexp2(double x, double y) {
  if (x.isNaN || y.isNaN) return double.nan;
  if (x == double.negativeInfinity && y == double.negativeInfinity) {
    return double.negativeInfinity;
  }
  if (x == y) return x + 1.0;
  final maxVal = x > y ? x : y;
  final minVal = x > y ? y : x;
  final ln2 = 0.6931471805599453;
  double log1pVal(double v) {
    if (v.abs() < 1e-5) return v - 0.5 * v * v + (1.0 / 3.0) * v * v * v;
    return math.log(1.0 + v);
  }

  return maxVal + log1pVal(math.exp((minVal - maxVal) * ln2)) / ln2;
}

/// Computes the exponential minus one ($e^x - 1$) element-wise.
NDArray<R> expm1<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute expm1() on a disposed array.');
  }
  final DType<DTypeTag> targetDType;
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
        ? DType.float32
        : DType.float64;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for expm1',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        expm1<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_expm1_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_expm1_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_expm1_complex128(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_expm1_complex64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
          break;
      }
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        final marker = ScratchArena.marker;
        try {
          final cBuffer = ScratchArena.getStridedBuffer(rank);
          final cShape = cBuffer;
          final cStridesA = cBuffer + rank;
          final cStridesRes = cBuffer + (rank * 2);
          for (var i = 0; i < rank; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesRes[i] = result.strides[i];
          }

          switch (a.dtype) {
            case DType.float64:
              s_expm1_double(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_expm1_float(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex128:
              s_expm1_complex128(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex64:
              s_expm1_complex64(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float16:
            case DType.bfloat16:
            case DType.int64:
            case DType.int32:
            case DType.int16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.uint8:
            case DType.boolean:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
        (a.dtype as DType<DTypeTag>) == DType.complex64) {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => _complexExpm1(x as Complex),
        maskHolder.pointer,
      );
    } else if (a.dtype.isInteger) {
      final isUint64 = (a.dtype as DType<DTypeTag>) == DType.uint64;
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) {
          final dx = isUint64
              ? BigInt.from(x as int).toUnsigned(64).toDouble()
              : (x as num).toDouble();
          if (dx.abs() < 1e-5) {
            return dx + 0.5 * dx * dx + (1.0 / 6.0) * dx * dx * dx;
          }
          return math.exp(dx) - 1.0;
        },
        maskHolder.pointer,
      );
    } else {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) {
          final dx = (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
          if (dx.abs() < 1e-5) {
            return dx + 0.5 * dx * dx + (1.0 / 6.0) * dx * dx * dx;
          }
          return math.exp(dx) - 1.0;
        },
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes $\ln(1+x)$ element-wise.
NDArray<R> log1p<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute log1p() on a disposed array.');
  }
  final DType<DTypeTag> targetDType;
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
        ? DType.float32
        : DType.float64;
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for log1p',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(out.shape, out.dtype);
        log1p<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_log1p_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_log1p_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_log1p_complex128(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_log1p_complex64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
        case DType.int16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.uint8:
        case DType.boolean:
          break;
      }
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        final marker = ScratchArena.marker;
        try {
          final cBuffer = ScratchArena.getStridedBuffer(rank);
          final cShape = cBuffer;
          final cStridesA = cBuffer + rank;
          final cStridesRes = cBuffer + (rank * 2);
          for (var i = 0; i < rank; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesRes[i] = result.strides[i];
          }

          switch (a.dtype) {
            case DType.float64:
              s_log1p_double(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_log1p_float(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex128:
              s_log1p_complex128(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex64:
              s_log1p_complex64(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float16:
            case DType.bfloat16:
            case DType.int64:
            case DType.int32:
            case DType.int16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.uint8:
            case DType.boolean:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
        (a.dtype as DType<DTypeTag>) == DType.complex64) {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => _complexLog1p(x as Complex),
        maskHolder.pointer,
      );
    } else if (a.dtype.isInteger) {
      final isUint64 = (a.dtype as DType<DTypeTag>) == DType.uint64;
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) {
          final dx = isUint64
              ? BigInt.from(x as int).toUnsigned(64).toDouble()
              : (x as num).toDouble();
          if (dx.abs() < 1e-5) {
            return dx - 0.5 * dx * dx + (1.0 / 3.0) * dx * dx * dx;
          }
          return math.log(1.0 + dx);
        },
        maskHolder.pointer,
      );
    } else {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) {
          final dx = (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
          if (dx.abs() < 1e-5) {
            return dx - 0.5 * dx * dx + (1.0 / 3.0) * dx * dx * dx;
          }
          return math.log(1.0 + dx);
        },
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes $\log(e^{x_1} + e^{x_2})$ element-wise.
NDArray<R> logaddexp<T extends DTypeTag, R extends DTypeTag>(
  NDArray<DTypeSpec<T, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>> x1,
  NDArray<DTypeSpec<T, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>>
  x2, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logaddexp() on a disposed array.');
  }
  final DType<DTypeTag> x1DType = x1.dtype;
  final DType<DTypeTag> x2DType = x2.dtype;
  if (x1DType == DType.complex128 ||
      x1DType == DType.complex64 ||
      x2DType == DType.complex128 ||
      x2DType == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for logaddexp');
  }
  if (x1DType != x2DType) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType<R> targetDType =
      ((x1DType == DType.float32 && x2DType == DType.float32)
              ? DType.float32
              : DType.float64)
          as DType<R>;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for logaddexp',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(shape, targetDType);
        logaddexp<T, R>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, shape);
  try {
    final NDArray<R> result =
        out ?? NDArray<R>.create(shape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch (targetDType) {
        case DType.float64:
          if (x1DType == DType.float64 && x2DType == DType.float64) {
            v_logaddexp_double(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1DType == DType.float32 && x2DType == DType.float32) {
            v_logaddexp_float(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
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
    }

    final stridesX1 = broadcastResult.stridesA;
    final stridesX2 = broadcastResult.stridesB;

    if (shape.length <= 8) {
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(shape);
        final cStridesX1 = ScratchArena.copyInts(stridesX1);
        final cStridesX2 = ScratchArena.copyInts(stridesX2);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        switch (targetDType) {
          case DType.float64:
            if (x1DType == DType.float64 && x2DType == DType.float64) {
              s_logaddexp_double(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                shape.length,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float32:
            if (x1DType == DType.float32 && x2DType == DType.float32) {
              s_logaddexp_float(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                shape.length,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float16:
          case DType.bfloat16:
          case DType.int64:
          case DType.int32:
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    final isUint64 = x1DType == DType.uint64;
    elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
      result,
      x1,
      x2,
      shape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      result.strides,
      0,
      x1.offsetElements,
      x2.offsetElements,
      result.offsetElements,
      (a, b) => _logaddexp(
        isUint64
            ? BigInt.from(a as int).toUnsigned(64).toDouble()
            : (a is bool ? (a ? 1.0 : 0.0) : (a as num).toDouble()),
        isUint64
            ? BigInt.from(b as int).toUnsigned(64).toDouble()
            : (b is bool ? (b ? 1.0 : 0.0) : (b as num).toDouble()),
      ),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes $\log_2(2^{x_1} + 2^{x_2})$ element-wise.
NDArray<R> logaddexp2<T extends DTypeTag, R extends DTypeTag>(
  NDArray<DTypeSpec<T, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>> x1,
  NDArray<DTypeSpec<T, Object?, DTypeTag, DTypeTag, R, DTypeTag, DTypeTag>>
  x2, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logaddexp2() on a disposed array.');
  }
  final DType<DTypeTag> x1DType = x1.dtype;
  final DType<DTypeTag> x2DType = x2.dtype;
  if (x1DType == DType.complex128 ||
      x1DType == DType.complex64 ||
      x2DType == DType.complex128 ||
      x2DType == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for logaddexp2');
  }
  if (x1DType != x2DType) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType<R> targetDType =
      ((x1DType == DType.float32 && x2DType == DType.float32)
              ? DType.float32
              : DType.float64)
          as DType<R>;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for logaddexp2',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(shape, targetDType);
        logaddexp2<T, R>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, shape);
  try {
    final NDArray<R> result =
        out ?? NDArray<R>.create(shape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch (targetDType) {
        case DType.float64:
          if (x1DType == DType.float64 && x2DType == DType.float64) {
            v_logaddexp2_double(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1DType == DType.float32 && x2DType == DType.float32) {
            v_logaddexp2_float(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
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
    }

    final stridesX1 = broadcastResult.stridesA;
    final stridesX2 = broadcastResult.stridesB;

    if (shape.length <= 8) {
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(shape);
        final cStridesX1 = ScratchArena.copyInts(stridesX1);
        final cStridesX2 = ScratchArena.copyInts(stridesX2);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        switch (targetDType) {
          case DType.float64:
            if (x1DType == DType.float64 && x2DType == DType.float64) {
              s_logaddexp2_double(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                shape.length,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float32:
            if (x1DType == DType.float32 && x2DType == DType.float32) {
              s_logaddexp2_float(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                shape.length,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float16:
          case DType.bfloat16:
          case DType.int64:
          case DType.int32:
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    final isUint64 = x1DType == DType.uint64;
    elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
      result,
      x1,
      x2,
      shape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      result.strides,
      0,
      x1.offsetElements,
      x2.offsetElements,
      result.offsetElements,
      (a, b) => _logaddexp2(
        isUint64
            ? BigInt.from(a as int).toUnsigned(64).toDouble()
            : (a is bool ? (a ? 1.0 : 0.0) : (a as num).toDouble()),
        isUint64
            ? BigInt.from(b as int).toUnsigned(64).toDouble()
            : (b is bool ? (b ? 1.0 : 0.0) : (b as num).toDouble()),
      ),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Rounds elements of the array to the nearest integer.
NDArray<R> rint<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, R, DTypeTag, DTypeTag, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute rint() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for rint');
  }
  final targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
      ? DType.float32
      : DType.float64;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for rint',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(a.shape, targetDType as DType<R>);
        rint<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_rint_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_rint_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
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
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        final marker = ScratchArena.marker;
        try {
          final cBuffer = ScratchArena.getStridedBuffer(rank);
          final cShape = cBuffer;
          final cStridesA = cBuffer + rank;
          final cStridesRes = cBuffer + (rank * 2);
          for (var i = 0; i < rank; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesRes[i] = result.strides[i];
          }

          switch (a.dtype) {
            case DType.float64:
              s_rint_double(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_rint_float(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float16:
            case DType.bfloat16:
            case DType.int64:
            case DType.int32:
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
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if (a.dtype.isInteger) {
      final isUint64 = (a.dtype as DType<DTypeTag>) == DType.uint64;
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => isUint64
            ? BigInt.from(x as int).toUnsigned(64).toDouble()
            : (x as num).toDouble(),
        maskHolder.pointer,
      );
    } else {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) {
          final dx = (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
          if (dx.isInfinite || dx.isNaN) return dx;
          final floorVal = dx.floorToDouble();
          final ceilVal = dx.ceilToDouble();
          final distFloor = dx - floorVal;
          final distCeil = ceilVal - dx;
          if (distFloor < distCeil) return floorVal;
          if (distCeil < distFloor) return ceilVal;
          return (floorVal % 2 == 0) ? floorVal : ceilVal;
        },
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Rounds elements of the array to the nearest integer towards zero.
NDArray<R> trunc<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, R, DTypeTag, DTypeTag, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute trunc() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == DType.complex128 ||
      (a.dtype as DType<DTypeTag>) == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for trunc');
  }
  final targetDType = (a.dtype as DType<DTypeTag>) == DType.float32
      ? DType.float32
      : DType.float64;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for trunc',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(a.shape, targetDType as DType<R>);
        trunc<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ??
        (NDArray.create(a.shape, targetDType, zeroInit: where != null)
            as NDArray<R>);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_trunc_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_trunc_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
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
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        final marker = ScratchArena.marker;
        try {
          final cBuffer = ScratchArena.getStridedBuffer(rank);
          final cShape = cBuffer;
          final cStridesA = cBuffer + rank;
          final cStridesRes = cBuffer + (rank * 2);
          for (var i = 0; i < rank; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesRes[i] = result.strides[i];
          }

          switch (a.dtype) {
            case DType.float64:
              s_trunc_double(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_trunc_float(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float16:
            case DType.bfloat16:
            case DType.int64:
            case DType.int32:
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
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if (a.dtype.isInteger) {
      final isUint64 = (a.dtype as DType<DTypeTag>) == DType.uint64;
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => isUint64
            ? BigInt.from(x as int).toUnsigned(64).toDouble()
            : (x as num).toDouble(),
        maskHolder.pointer,
      );
    } else {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble())
            .truncateToDouble(),
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Rounds elements of the array to the nearest integer towards zero.
///
/// Synonym for [trunc].
NDArray<R> fix<R extends DTypeTag>(
  NDArray<
    DTypeSpec<DTypeTag, Object?, R, DTypeTag, DTypeTag, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) => trunc(a, where: where, out: out);

/// Computes the element-wise square of the input array.
///
/// It is an error if the array has been disposed (throws [StateError]), or if the provided [out] buffer shape or dtype is incompatible (throws [ArgumentError]).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<T> square<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute square() on a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for square',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        square<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_square_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_square_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
          v_square_float16(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.bfloat16:
          v_square_bfloat16(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_square_int64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_square_int32(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          v_square_int16(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int8:
          v_square_int8(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint64:
          v_square_uint64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint32:
          v_square_uint32(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint16:
          v_square_uint16(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          v_square_uint8(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.boolean:
          v_square_boolean(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_square_complex128(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_square_complex64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
      }
    } else {
      final rank = a.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesRes = cBuffer + (rank * 2);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
          cStridesRes[i] = result.strides[i];
        }
        switch (a.dtype) {
          case DType.float64:
            s_square_double(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.float32:
            s_square_float(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.float16:
            s_square_float16(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.bfloat16:
            s_square_bfloat16(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int64:
            s_square_int64(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int32:
            s_square_int32(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int16:
            s_square_int16(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int8:
            s_square_int8(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.uint64:
            s_square_uint64(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.uint32:
            s_square_uint32(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.uint16:
            s_square_uint16(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.uint8:
            s_square_uint8(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.boolean:
            s_square_boolean(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex128:
            s_square_complex128(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex64:
            s_square_complex64(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise reciprocal ($1/x$) of the array.
NDArray<T> reciprocal<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute reciprocal() on a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for reciprocal',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        reciprocal<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray.create(a.shape, a.dtype, zeroInit: where != null);
    var isInt = false;
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_reciprocal_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_reciprocal_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_reciprocal_complex128(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_reciprocal_complex64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_reciprocal_int64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        case DType.int32:
          v_reciprocal_int32(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        case DType.int16:
          v_reciprocal_int16(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        case DType.uint8:
          v_reciprocal_uint8(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.boolean:
          break;
      }
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        final marker = ScratchArena.marker;
        try {
          final cBuffer = ScratchArena.getStridedBuffer(rank);
          final cShape = cBuffer;
          final cStridesA = cBuffer + rank;
          final cStridesRes = cBuffer + (rank * 2);
          for (var i = 0; i < rank; i++) {
            cShape[i] = a.shape[i];
            cStridesA[i] = a.strides[i];
            cStridesRes[i] = result.strides[i];
          }

          switch (a.dtype) {
            case DType.float64:
              s_reciprocal_double(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_reciprocal_float(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex128:
              s_reciprocal_complex128(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex64:
              s_reciprocal_complex64(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.int64:
              s_reciprocal_int64(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            case DType.int32:
              s_reciprocal_int32(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            case DType.int16:
              s_reciprocal_int16(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            case DType.uint8:
              s_reciprocal_uint8(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            case DType.float16:
            case DType.bfloat16:
            case DType.int8:
            case DType.uint64:
            case DType.uint32:
            case DType.uint16:
            case DType.boolean:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if (isInt) {
      final err = get_and_reset_division_error();
      if (err == 1) {
        throw UnsupportedError('Integer division by zero');
      }
      return result;
    }

    final isUnsignedInt =
        a.dtype == DType.uint64 ||
        a.dtype == DType.uint32 ||
        a.dtype == DType.uint16 ||
        a.dtype == DType.uint8;
    unaryOp<T, T>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) {
        if (x is Complex) {
          return (Complex(1.0, 0.0) / x);
        } else if (x is double) {
          return castValue(1.0 / x, a.dtype);
        } else if (x is int) {
          if (x == 0) throw UnsupportedError('Integer division by zero');
          if (isUnsignedInt) {
            return x == 1 ? 1 : 0;
          }
          return castValue(1 ~/ x, a.dtype);
        }
        throw UnsupportedError('Unsupported type for reciprocal');
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Numerical positive, element-wise.
///
/// Returns a copy of [a] for all numeric types.
///
/// **Example:**
/// {@example /example/easy_ufuncs_example.dart lang=dart}
NDArray<T> positive<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute positive() on a disposed array.');
  }
  if (a.dtype == DType.boolean) {
    throw UnsupportedError('Boolean arrays do not support positive operator');
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for positive',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        positive<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray.create(a.shape, a.dtype, zeroInit: where != null);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_positive_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_positive_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_positive_complex128(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_positive_complex64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
        case DType.uint64:
          v_positive_int64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
        case DType.uint32:
          v_positive_int32(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
        case DType.uint16:
        case DType.float16:
        case DType.bfloat16:
          v_positive_int16(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
        case DType.int8:
          v_positive_uint8(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.boolean:
          break;
      }
    } else {
      final rank = a.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesRes = cBuffer + (rank * 2);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
          cStridesRes[i] = result.strides[i];
        }

        switch (a.dtype) {
          case DType.float64:
            s_positive_double(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.float32:
            s_positive_float(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex128:
            s_positive_complex128(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex64:
            s_positive_complex64(
              a.typedPointer(),
              cStridesA,
              result.typedPointer(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int64:
          case DType.uint64:
            s_positive_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int32:
          case DType.uint32:
            s_positive_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int16:
          case DType.uint16:
          case DType.float16:
          case DType.bfloat16:
            s_positive_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.uint8:
          case DType.int8:
            s_positive_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.boolean:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    unaryOp<T, T>(
      result,
      a,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => x,
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// First array elements raised to powers from second array elements, element-wise.
///
/// Raise each base in [x1] to the positionally corresponding power in [x2].
/// Both [x1] and [x2] must have the same [DType].
///
/// Preconditions:
/// - [x1] and [x2] must not be disposed.
/// - [x1] and [x2] must have matching [DType].
/// - [x1] and [x2] shapes must be broadcastable.
///
/// Parameters:
/// - [x1]: First input array of bases.
/// - [x2]: Second input array of exponents. Must have same [DType] as [x1].
/// - [out]: Optional output array buffer to store results.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - integer bases are raised to negative integer powers (throws [ArgumentError]).
///
/// Performance Considerations:
/// - Contiguous arrays leverage vector sweeps (`v_pow_*`).
/// - Strided broadcasting uses multi-dimensional FFI iterators (`s_pow_*`).
NDArray<T> power<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute power() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final dtype = x1.dtype;

  final isSignedInt =
      dtype == DType.int64 ||
      dtype == DType.int32 ||
      dtype == DType.int16 ||
      dtype == DType.int8;
  if (isSignedInt && x2.size > 0) {
    final NDArray<DTypeTag> x2Num = x2;
    try {
      if (x2Num.rank == 0) {
        if ((x2Num.scalar as num) < 0) {
          throw ArgumentError.value(
            x2Num.scalar,
            'x2',
            'Integers to negative integer powers are not allowed',
          );
        }
      } else {
        final minArr = min(x2Num);
        final minVal = minArr.scalar as num;
        minArr.dispose();
        if (minVal < 0) {
          throw ArgumentError.value(
            minVal,
            'x2',
            'Integers to negative integer powers are not allowed',
          );
        }
      }
    } finally {
      if (!identical(x2Num, x2)) x2Num.dispose();
    }
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, shape) || out.dtype != dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for power',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(shape, dtype);
        power<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, shape);
  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(shape, dtype, zeroInit: where != null);
    final isContig =
        x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous;

    if (isContig) {
      switch (dtype) {
        case DType.float64:
          v_pow_double(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_pow_float(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
          v_pow_float16(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.bfloat16:
          v_pow_bfloat16(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_pow_int64(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_pow_int32(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          v_pow_int16(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int8:
          v_pow_int8(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint64:
          v_pow_uint64(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint32:
          v_pow_uint32(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint16:
          v_pow_uint16(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          v_pow_uint8(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.boolean:
          v_pow_boolean(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_pow_complex128(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_pow_complex64(
            x1.typedPointer(),
            x2.typedPointer(),
            result.typedPointer(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
      }
    }

    final rank = shape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank * 4);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);
      for (var i = 0; i < rank; i++) {
        cShape[i] = shape[i];
        cStridesA[i] = broadcastResult.stridesA[i];
        cStridesB[i] = broadcastResult.stridesB[i];
        cStridesRes[i] = result.strides[i];
      }
      switch (dtype) {
        case DType.float64:
          s_pow_double(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          s_pow_float(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
          s_pow_float16(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.bfloat16:
          s_pow_bfloat16(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          s_pow_int64(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          s_pow_int32(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          s_pow_int16(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int8:
          s_pow_int8(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.uint64:
          s_pow_uint64(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.uint32:
          s_pow_uint32(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.uint16:
          s_pow_uint16(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          s_pow_uint8(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.boolean:
          s_pow_boolean(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          s_pow_complex128(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          s_pow_complex64(
            x1.typedPointer(),
            cStridesA,
            x2.typedPointer(),
            cStridesB,
            result.typedPointer(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the numerical negative of [a] element-wise (`-a`).
///
/// If [where] is provided, elements where [where] is truthy receive `-a` and
/// remaining elements are untouched (when [out] is supplied) or zero-initialized.
/// If [out] is provided, the result is written into [out] and returned.
///
/// The [out] array must match the shape and dtype of [a].
/// None of [a], [where], or [out] may be disposed.
NDArray<T> negative<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute negative() on a disposed array.');
  }
  if (a.dtype == DType.boolean) {
    throw UnsupportedError('Boolean arrays do not support negative operator');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for negative',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        negative<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => -(x as Complex),
          maskHolder.pointer,
        );
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => castValue(-(x as num), a.dtype),
          maskHolder.pointer,
        );
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => castValue((-(x as num)).toInt(), a.dtype),
          maskHolder.pointer,
        );
      case DType.boolean:
        throw UnsupportedError(
          'Boolean arrays do not support negative operator',
        );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Element-wise floor division with broadcasting support.
///
/// Both [x1] and [x2] must have the same [DType].
///
/// **Division by Zero:**
/// - **Integer arrays**: Division by zero is an error.
/// - **Floating-point arrays**: Follows IEEE 754 rules (`±Infinity` for non-zero divided by zero, `NaN` for `0.0 / 0.0`).
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed and must have the same [DType].
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - for integer arrays, the divisor [x2] contains any `0` elements (throws [UnsupportedError]).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<T> floorDivide<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute floorDivide() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support floor division');
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for floorDivide',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(commonShape, targetDType);
        floorDivide<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, commonShape);
  try {
    final NDArray<T> result =
        out ??
        NDArray<T>.create(commonShape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_floordiv_double(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_floordiv_float(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_floordiv_int64(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_floordiv_int32(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
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
    } else if (commonShape.length <= 8) {
      final rank = commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);
        for (var i = 0; i < rank; i++) {
          cShape[i] = commonShape[i];
          cStridesA[i] = stridesA[i];
          cStridesB[i] = stridesB[i];
          cStridesRes[i] = result.strides[i];
        }
        switch (targetDType) {
          case DType.float64:
            if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
              s_floordiv_double(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float32:
            if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
              s_floordiv_float(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.int64:
            if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
              s_floordiv_int64(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              final err = get_and_reset_division_error();
              if (err == 1) {
                throw UnsupportedError('Integer division by zero');
              }
              return result;
            }
          case DType.int32:
            if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
              s_floordiv_int32(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              final err = get_and_reset_division_error();
              if (err == 1) {
                throw UnsupportedError('Integer division by zero');
              }
              return result;
            }
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    double doubleFloorDiv(double a, double b) {
      if (b == 0.0) {
        if (a == 0.0 || a.isNaN) return double.nan;
        final signA = a.isNegative ? -1.0 : 1.0;
        final signB = b.isNegative ? -1.0 : 1.0;
        return signA * signB * double.infinity;
      }
      if (a.isNaN || b.isNaN || a.isInfinite) return double.nan;
      if (b.isInfinite) {
        if (a == 0.0) return (a.isNegative != b.isNegative) ? -0.0 : 0.0;
        return (a.isNegative == b.isNegative) ? 0.0 : -1.0;
      }
      if (a == 0.0) {
        return (a.isNegative != b.isNegative) ? -0.0 : 0.0;
      }
      var mod = a.remainder(b);
      var div = (a - mod) / b;
      if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) {
          mod += b;
          div -= 1.0;
        }
      }
      if (div != 0.0) {
        var floordiv = div.floorToDouble();
        if (div - floordiv > 0.5) {
          floordiv += 1.0;
        }
        return floordiv;
      } else {
        return (a.isNegative != b.isNegative) ? -0.0 : 0.0;
      }
    }

    int intFloorDiv(int x, int y) {
      if (y == 0) {
        throw UnsupportedError('Integer division by zero');
      }
      if (targetDType == DType.uint64) {
        final ux = BigInt.from(x).toUnsigned(64);
        final uy = BigInt.from(y).toUnsigned(64);
        return (ux ~/ uy).toSigned(64).toInt();
      }
      final res = x ~/ y;
      final rem = x % y;
      if (rem != 0 && ((x < 0) ^ (y < 0))) {
        return res - 1;
      }
      return res;
    }

    if (targetDType.isFloating) {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
        result,
        x1,
        x2,
        commonShape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) => castValue(
          doubleFloorDiv(
            (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble()),
            (y is bool ? (y ? 1.0 : 0.0) : (y as num).toDouble()),
          ),
          targetDType,
        ),
        maskHolder.pointer,
      );
    } else {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
        result,
        x1,
        x2,
        commonShape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) => castValue(
          intFloorDiv(
            (x is bool ? (x ? 1 : 0) : (x as num).toInt()),
            (y is bool ? (y ? 1 : 0) : (y as num).toInt()),
          ),
          targetDType,
        ),
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise remainder of division of two arrays (`x1 - floor_divide(x1, x2) * x2`).
///
/// The sign of the result matches the divisor [x2]. For C-style remainder where
/// the sign matches the dividend [x1], use [fmod].
///
/// **Division by Zero:**
/// - **Integer arrays**: Division by zero is an error.
/// - **Floating-point arrays**: Returns `double.nan` silently.
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - for integer arrays, the divisor [x2] contains any `0` elements (throws [UnsupportedError]).
NDArray<T> remainder<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute remainder() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support remainder');
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for remainder',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(commonShape, targetDType);
        remainder<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, commonShape);
  try {
    final NDArray<T> result =
        out ??
        NDArray<T>.create(commonShape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_remainder_double(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_remainder_float(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_remainder_int64(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_remainder_int32(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
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
    } else if (commonShape.length <= 8) {
      final rank = commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);
        for (var i = 0; i < rank; i++) {
          cShape[i] = commonShape[i];
          cStridesA[i] = stridesA[i];
          cStridesB[i] = stridesB[i];
          cStridesRes[i] = result.strides[i];
        }
        switch (targetDType) {
          case DType.float64:
            if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
              s_remainder_double(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float32:
            if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
              s_remainder_float(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.int64:
            if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
              s_remainder_int64(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              final err = get_and_reset_division_error();
              if (err == 1) {
                throw UnsupportedError('Integer division by zero');
              }
              return result;
            }
          case DType.int32:
            if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
              s_remainder_int32(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              final err = get_and_reset_division_error();
              if (err == 1) {
                throw UnsupportedError('Integer division by zero');
              }
              return result;
            }
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    double doubleMod(double x, double y) {
      if (y == 0.0 || x.isNaN || y.isNaN || x.isInfinite) return double.nan;
      var rem = x.remainder(y);
      if (rem != 0.0) {
        if ((rem < 0.0) != (y < 0.0)) {
          rem += y;
        }
      } else {
        rem = y.isNegative ? -0.0 : 0.0;
      }
      return rem;
    }

    int intMod(int x, int y) {
      if (y == 0) {
        throw UnsupportedError('Integer division by zero');
      }
      if (targetDType == DType.uint64) {
        final ux = BigInt.from(x).toUnsigned(64);
        final uy = BigInt.from(y).toUnsigned(64);
        return (ux % uy).toSigned(64).toInt();
      }
      final rem = x % y;
      if (rem != 0 && ((rem < 0) != (y < 0))) {
        return rem + y;
      }
      return rem;
    }

    if (targetDType.isFloating) {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
        result,
        x1,
        x2,
        commonShape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) => castValue(
          doubleMod(
            (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble()),
            (y is bool ? (y ? 1.0 : 0.0) : (y as num).toDouble()),
          ),
          targetDType,
        ),
        maskHolder.pointer,
      );
    } else {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
        result,
        x1,
        x2,
        commonShape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) => castValue(
          intMod(
            (x is bool ? (x ? 1 : 0) : (x as num).toInt()),
            (y is bool ? (y ? 1 : 0) : (y as num).toInt()),
          ),
          targetDType,
        ),
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Alias for [remainder].
///
/// **Division by Zero:**
/// - **Integer arrays**: Division by zero is an error.
/// - **Floating-point arrays**: Returns `double.nan` silently.
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed and must have the same [DType].
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - for integer arrays, the divisor [x2] contains any `0` elements (throws [UnsupportedError]).
NDArray<T> mod<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) => remainder<T>(x1, x2, where: where, out: out);

/// Element-wise floor division and remainder simultaneously (`floor_divide(x1, x2)`, `remainder(x1, x2)`).
///
/// Returns a named record containing:
/// - `quotient`: The floor division result ([floorDivide]).
/// - `remainder`: The remainder of division ([remainder]).
///
/// **Division by Zero:**
/// - **Integer arrays**: Division by zero is an error.
/// - **Floating-point arrays**: Follows IEEE 754 rules for floor division and returns `double.nan` for remainder.
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed and must have the same [DType].
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1] or [x2] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - for integer arrays, the divisor [x2] contains any `0` elements (throws [UnsupportedError]).
({NDArray<T> quotient, NDArray<T> remainder}) divmod<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out1,
  NDArray<T>? out2,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out1 != null && out1.isDisposed) ||
      (out2 != null && out2.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute divmod() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support divmod');
  }
  if (out1 != null) {
    validateOutBuffer(out1, 'out1');
    if (!listEquals(out1.shape, commonShape) || out1.dtype != targetDType) {
      throw ArgumentError.value(
        out1,
        'out1',
        'Must have compatible shape and dtype for divmod',
      );
    }
  }
  if (out2 != null) {
    validateOutBuffer(out2, 'out2');
    if (!listEquals(out2.shape, commonShape) || out2.dtype != targetDType) {
      throw ArgumentError.value(
        out2,
        'out2',
        'Must have compatible shape and dtype for divmod',
      );
    }
  }
  if (out1 != null && out2 != null && sharesMemory(out1, out2)) {
    throw ArgumentError.value(out2, 'out2', 'Must not share memory with out1.');
  }
  if (out1 != null &&
      (sharesMemory(x1, out1) ||
          sharesMemory(x2, out1) ||
          (where != null && sharesMemory(where, out1)))) {
    return NDArray.scope(() {
      final temp1 = where != null
          ? out1.copy()
          : NDArray<T>.create(commonShape, targetDType);
      floorDivide<T>(x1, x2, where: where, out: temp1);
      final rem = remainder<T>(x1, x2, where: where, out: out2);
      temp1.copy(out: out1);
      return (quotient: out1, remainder: out2 ?? rem.detachToParentScope());
    });
  }
  return (
    quotient: floorDivide<T>(x1, x2, where: where, out: out1),
    remainder: remainder<T>(x1, x2, where: where, out: out2),
  );
}

/// Element-wise C-style modulo / remainder of division (`x1 % x2`).
///
/// Unlike [remainder] / [mod], the sign of the result matches the dividend [x1].
///
/// **Division by Zero:**
/// - **Integer arrays**: Division by zero is an error.
/// - **Floating-point arrays**: Returns `double.nan` silently.
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed and must have the same [DType].
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - for integer arrays, the divisor [x2] contains any `0` elements (throws [UnsupportedError]).
NDArray<T> fmod<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute fmod() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support fmod');
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for fmod',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(commonShape, targetDType);
        fmod<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, commonShape);
  try {
    final NDArray<T> result =
        out ??
        NDArray<T>.create(commonShape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_fmod_double(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_fmod_float(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_fmod_int64(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_fmod_int32(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
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
    } else if (commonShape.length <= 8) {
      final rank = commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank * 3);
        final cShape = cBuffer;
        final cStridesX1 = cBuffer + rank;
        final cStridesX2 = cBuffer + (rank * 2);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        for (var i = 0; i < rank; i++) {
          cShape[i] = commonShape[i];
          cStridesX1[i] = stridesA[i];
          cStridesX2[i] = stridesB[i];
        }
        switch (targetDType) {
          case DType.float64:
            if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
              s_fmod_double(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float32:
            if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
              s_fmod_float(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.int64:
            if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
              s_fmod_int64(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              final err = get_and_reset_division_error();
              if (err == 1) {
                throw UnsupportedError('Integer division by zero');
              }
              return result;
            }
          case DType.int32:
            if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
              s_fmod_int32(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              final err = get_and_reset_division_error();
              if (err == 1) {
                throw UnsupportedError('Integer division by zero');
              }
              return result;
            }
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (targetDType.isFloating) {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
        result,
        x1,
        x2,
        commonShape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) {
          final dy = (y is bool ? (y ? 1.0 : 0.0) : (y as num).toDouble());
          final dx = (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
          final val = dy == 0.0 ? double.nan : dx.remainder(dy);
          return castValue(val, targetDType);
        },
        maskHolder.pointer,
      );
    } else {
      elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
        result,
        x1,
        x2,
        commonShape,
        stridesA,
        stridesB,
        result.strides,
        0,
        x1.offsetElements,
        x2.offsetElements,
        result.offsetElements,
        (x, y) {
          final iy = (y is bool ? (y ? 1 : 0) : (y as num).toInt());
          final ix = (x is bool ? (x ? 1 : 0) : (x as num).toInt());
          if (iy == 0) throw UnsupportedError('Integer division by zero');
          if (targetDType == DType.uint64) {
            final ux = BigInt.from(ix).toUnsigned(64);
            final uy = BigInt.from(iy).toUnsigned(64);
            return (ux % uy).toSigned(64).toInt();
          }
          return castValue(ix.remainder(iy), targetDType);
        },
        maskHolder.pointer,
      );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Element-wise greatest common divisor (`gcd(x1, x2)`).
///
/// Operates on integer arrays. Always returns a non-negative greatest common divisor.
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed, must have the same [DType], and must have integer dtypes.
/// - If [out] is provided, it must not be disposed and must have compatible shape and integer dtype.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - [x1] or [x2] has a non-integer dtype (throws [UnsupportedError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
NDArray<T> gcd<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute gcd() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  if (!x1.dtype.isInteger || !x2.dtype.isInteger) {
    throw UnsupportedError('gcd only supports integer arrays.');
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for gcd',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(commonShape, targetDType);
        gcd<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, commonShape);
  try {
    final NDArray<T> result =
        out ??
        NDArray<T>.create(commonShape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_gcd_int64(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_gcd_int32(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float64:
        case DType.float32:
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
    } else if (commonShape.length <= 8) {
      final rank = commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank * 3);
        final cShape = cBuffer;
        final cStridesX1 = cBuffer + rank;
        final cStridesX2 = cBuffer + (rank * 2);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        for (var i = 0; i < rank; i++) {
          cShape[i] = commonShape[i];
          cStridesX1[i] = stridesA[i];
          cStridesX2[i] = stridesB[i];
        }
        switch (targetDType) {
          case DType.int64:
            if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
              s_gcd_int64(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.int32:
            if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
              s_gcd_int32(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float64:
          case DType.float32:
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    BigInt calcGcd(BigInt a, BigInt b) {
      var u = a.abs();
      var v = b.abs();
      while (v != BigInt.zero) {
        final t = v;
        v = u % v;
        u = t;
      }
      return u;
    }

    elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
      result,
      x1,
      x2,
      commonShape,
      stridesA,
      stridesB,
      result.strides,
      0,
      x1.offsetElements,
      x2.offsetElements,
      result.offsetElements,
      (x, y) {
        final ix = (x as num).toInt();
        final iy = (y as num).toInt();
        final ua = targetDType == DType.uint64
            ? BigInt.from(ix).toUnsigned(64)
            : BigInt.from(ix);
        final ub = targetDType == DType.uint64
            ? BigInt.from(iy).toUnsigned(64)
            : BigInt.from(iy);
        final g = calcGcd(ua, ub).toSigned(64).toInt();
        return targetDType == DType.uint64 ? g : castValue(g, targetDType);
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Element-wise least common multiple (`lcm(x1, x2)`).
///
/// Returns the lowest common multiple of `|x1|` and `|x2|`.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<T> lcm<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute lcm() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  if (!x1.dtype.isInteger || !x2.dtype.isInteger) {
    throw UnsupportedError('lcm only supports integer arrays.');
  }
  final DType<DTypeTag> targetDType = resolveDType(x1.dtype, x2.dtype);
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for lcm',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(commonShape, targetDType as DType<T>);
        lcm<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, commonShape);

  try {
    final NDArray<T> result =
        out ??
        NDArray<T>.create(
          commonShape,
          targetDType as DType<T>,
          zeroInit: where != null,
        );
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch (targetDType) {
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_lcm_int64(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              result.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_lcm_int32(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              result.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float64:
        case DType.float32:
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
    } else if (commonShape.length <= 8) {
      final rank = commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);

        for (var i = 0; i < rank; i++) {
          cShape[i] = commonShape[i];
          cStridesA[i] = stridesA[i];
          cStridesB[i] = stridesB[i];
          cStridesRes[i] = result.strides[i];
        }

        switch (targetDType) {
          case DType.int64:
            if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
              s_lcm_int64(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.int32:
            if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
              s_lcm_int32(
                x1.typedPointer(),
                cStridesA,
                x2.typedPointer(),
                cStridesB,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float64:
          case DType.float32:
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    BigInt calcGcd(BigInt a, BigInt b) {
      var u = a.abs();
      var v = b.abs();
      while (v != BigInt.zero) {
        final t = v;
        v = u % v;
        u = t;
      }
      return u;
    }

    elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
      result,
      x1,
      x2,
      commonShape,
      stridesA,
      stridesB,
      result.strides,
      0,
      x1.offsetElements,
      x2.offsetElements,
      result.offsetElements,
      (x, y) {
        final ix = (x as num).toInt();
        final iy = (y as num).toInt();
        final ua = targetDType == DType.uint64
            ? BigInt.from(ix).toUnsigned(64)
            : BigInt.from(ix);
        final ub = targetDType == DType.uint64
            ? BigInt.from(iy).toUnsigned(64)
            : BigInt.from(iy);
        if (ua == BigInt.zero || ub == BigInt.zero) {
          return castValue(0, targetDType);
        }
        final l = (ua.abs() ~/ calcGcd(ua, ub)) * ub.abs();
        final resVal = l.toSigned(64).toInt();
        return targetDType == DType.uint64
            ? resVal
            : castValue(resVal, targetDType);
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Element-wise Heaviside step function (`heaviside(x1, x2)`).
///
/// Computes:
/// - `0` if `x1 < 0`
/// - `x2` if `x1 == 0`
/// - `1` if `x1 > 0`
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed and must have the same [DType].
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [x1] and [x2] have different dtypes (throws [ArgumentError]).
/// - [x1] or [x2] has a complex dtype (throws [UnsupportedError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
NDArray<T> heaviside<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute heaviside() on a disposed array.');
  }
  if (x1.dtype != x2.dtype) {
    throw ArgumentError.value(
      x2.dtype,
      'x2',
      'Must have the same dtype as x1 (${x1.dtype})',
    );
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support heaviside');
  }

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for heaviside',
      );
    }
    if (sharesMemory(x1, out) ||
        sharesMemory(x2, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(commonShape, targetDType);
        heaviside<T>(x1, x2, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }

  final maskHolder = prepareMask(where, commonShape);
  try {
    final NDArray<T> result =
        out ??
        NDArray<T>.create(commonShape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_heaviside_double(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_heaviside_float(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_heaviside_int64(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_heaviside_int32(
              x1.typedPointer(),
              x2.typedPointer(),
              result.typedPointer(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
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
    } else if (commonShape.length <= 8) {
      final rank = commonShape.length;
      final marker = ScratchArena.marker;
      try {
        final cBuffer = ScratchArena.getStridedBuffer(rank * 3);
        final cShape = cBuffer;
        final cStridesX1 = cBuffer + rank;
        final cStridesX2 = cBuffer + (rank * 2);
        final cStridesRes = ScratchArena.copyInts(result.strides);
        for (var i = 0; i < rank; i++) {
          cShape[i] = commonShape[i];
          cStridesX1[i] = stridesA[i];
          cStridesX2[i] = stridesB[i];
        }
        switch (targetDType) {
          case DType.float64:
            if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
              s_heaviside_double(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.float32:
            if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
              s_heaviside_float(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.int64:
            if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
              s_heaviside_int64(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          case DType.int32:
            if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
              s_heaviside_int32(
                x1.typedPointer(),
                cStridesX1,
                x2.typedPointer(),
                cStridesX2,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
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
      } finally {
        ScratchArena.reset(marker);
      }
    }

    elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
      result,
      x1,
      x2,
      commonShape,
      stridesA,
      stridesB,
      result.strides,
      0,
      x1.offsetElements,
      x2.offsetElements,
      result.offsetElements,
      (x, y) {
        if (targetDType.isFloating) {
          final dx = (x is bool ? (x ? 1.0 : 0.0) : (x as num).toDouble());
          if (dx.isNaN) return castValue(dx, targetDType);
          if (dx < 0.0) return castValue(0.0, targetDType);
          if (dx > 0.0) return castValue(1.0, targetDType);
          return castValue(
            (y is bool ? (y ? 1.0 : 0.0) : (y as num).toDouble()),
            targetDType,
          );
        } else {
          final ix = (x is bool ? (x ? 1 : 0) : (x as num).toInt());
          if (targetDType == DType.uint64) {
            if (ix != 0) return 1;
            return (y is bool ? (y ? 1 : 0) : (y as num).toInt());
          }
          if (ix < 0) return castValue(0, targetDType);
          if (ix > 0) return castValue(1, targetDType);
          return castValue(
            (y is bool ? (y ? 1 : 0) : (y as num).toInt()),
            targetDType,
          );
        }
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the absolute value (or magnitude for complex inputs) of [a] element-wise.
///
/// For real and integer arrays, the output has the same dtype as [a]. For
/// [Complex64] and [Complex128] arrays, the output is the Euclidean magnitude
/// with dtype [Float32] and [Float64], respectively.
/// If [where] is provided, only elements where [where] is truthy are updated.
/// If [out] is provided, the result is written into [out] and returned.
NDArray<R> abs<R extends DTypeTag>(
  NDArray<
    DTypeSpec<R, Object?, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>
  >
  a, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute abs() on a disposed array.');
  }
  if ((a.dtype as DType<DTypeTag>) == DType.boolean) {
    throw UnsupportedError('Unsupported DType for abs: ${a.dtype}');
  }
  final targetDType = switch (a.dtype) {
    DType.complex64 => DType.float32,
    DType.complex128 => DType.float64,
    _ => a.dtype,
  };

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for abs',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(a.shape, targetDType as DType<R>);
        abs<R>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ??
        NDArray.create(
          a.shape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_abs_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_abs_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_abs_complex128(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_abs_complex64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_abs_int64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint64:
          v_positive_int64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_abs_int32(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint32:
          v_positive_int32(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          v_abs_int16(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint16:
          v_positive_int16(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          v_abs_uint8(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int8:
        case DType.boolean:
          break;
      }
    }
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
      case DType.int64:
      case DType.uint64:
      case DType.int32:
      case DType.uint32:
      case DType.int16:
      case DType.uint16:
      case DType.uint8:
        final rank = a.shape.length;
        if (rank <= 8) {
          final marker = ScratchArena.marker;
          try {
            final cBuffer = ScratchArena.getStridedBuffer(rank);
            final cShape = cBuffer;
            final cStridesA = cBuffer + rank;
            final cStridesRes = cBuffer + (rank * 2);
            for (var i = 0; i < rank; i++) {
              cShape[i] = a.shape[i];
              cStridesA[i] = a.strides[i];
              cStridesRes[i] = result.strides[i];
            }
            switch (a.dtype) {
              case DType.complex128:
                s_abs_complex128(
                  a.typedPointer(),
                  cStridesA,
                  result.typedPointer(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.complex64:
                s_abs_complex64(
                  a.typedPointer(),
                  cStridesA,
                  result.typedPointer(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.int64:
                s_abs_int64(
                  a.typedPointer(),
                  cStridesA,
                  result.typedPointer(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.uint64:
                s_positive_int64(
                  a.pointer.cast(),
                  cStridesA,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.int32:
                s_abs_int32(
                  a.typedPointer(),
                  cStridesA,
                  result.typedPointer(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.uint32:
                s_positive_int32(
                  a.pointer.cast(),
                  cStridesA,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.int16:
                s_abs_int16(
                  a.typedPointer(),
                  cStridesA,
                  result.typedPointer(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.uint16:
                s_positive_int16(
                  a.pointer.cast(),
                  cStridesA,
                  result.pointer.cast(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.uint8:
                s_abs_uint8(
                  a.typedPointer(),
                  cStridesA,
                  result.typedPointer(),
                  cStridesRes,
                  cShape,
                  rank,
                  maskHolder.pointer,
                );
                return result;
              case DType.float64:
              case DType.float32:
              case DType.float16:
              case DType.bfloat16:
              case DType.int8:
              case DType.boolean:
                break;
            }
          } finally {
            ScratchArena.reset(marker);
          }
        }
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.boolean:
        break;
    }

    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        unaryOp<DTypeTag, R>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (c) {
            final z = c as Complex;
            var x = z.real.abs();
            var y = z.imag.abs();
            if (x.isInfinite || y.isInfinite) return double.infinity;
            if (x < y) {
              final t = x;
              x = y;
              y = t;
            }
            if (x == 0.0) return 0.0;
            final r = y / x;
            return castValue(x * math.sqrt(1.0 + r * r), result.dtype);
          },
          maskHolder.pointer,
        );
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => x,
          maskHolder.pointer,
        );
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.int8:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => castValue((x as num).abs().toInt(), a.dtype),
          maskHolder.pointer,
        );
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => castValue((x as num).abs().toDouble(), a.dtype),
          maskHolder.pointer,
        );
      case DType.boolean:
        throw UnsupportedError('Unsupported DType for abs: ${a.dtype}');
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise sign of the array.
///
/// For real numbers, returns:
/// - -1 if x < 0
/// - 0 if x == 0
/// - 1 if x > 0
/// - nan if x is nan
///
/// For complex numbers, returns `x / |x|` (or 0 if x is 0).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<T> sign<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute sign() on a disposed array.');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for sign',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        sign<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        unaryOp<T, T>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (c) {
            final z = c as Complex;
            if (z.real.isNaN || z.imag.isNaN) {
              return Complex(double.nan, double.nan);
            }
            if (z.real == 0.0 && z.imag == 0.0) return Complex(0.0, 0.0);
            var x = z.real.abs();
            var y = z.imag.abs();
            if (x.isInfinite || y.isInfinite) {
              return Complex(double.nan, double.nan);
            }
            if (x < y) {
              final t = x;
              x = y;
              y = t;
            }
            final r = y / x;
            final mag = x * math.sqrt(1.0 + r * r);
            return Complex(z.real / mag, z.imag / mag);
          },
          maskHolder.pointer,
        );
      case DType.uint64:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => (x as int) == 0 ? 0 : 1,
          maskHolder.pointer,
        );
      case DType.boolean:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => x,
          maskHolder.pointer,
        );
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.int8:
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => castValue((x as num).sign.toInt(), a.dtype),
          maskHolder.pointer,
        );
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => castValue((x as num).sign.toDouble(), a.dtype),
          maskHolder.pointer,
        );
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes element-wise ceiling of the array.
///
/// It is an error if [a], [where], or [out] is disposed (throws [StateError]),
/// if [a] has a complex dtype (throws [UnsupportedError]),
/// or if [out] has an incompatible shape or dtype (throws [ArgumentError]).
NDArray<T> ceil<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute ceil() on a disposed array.');
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for ceil');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for ceil',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        ceil<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_ceil_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_ceil_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float16:
        case DType.bfloat16:
        case DType.int64:
        case DType.int32:
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
    }

    if (a.dtype.isInteger || a.dtype == DType.boolean) {
      if (where == null) {
        a.copy(out: result);
      } else {
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => x,
          maskHolder.pointer,
        );
      }
    } else if (a.dtype.isFloating) {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => castValue((x as num).ceilToDouble(), a.dtype),
        maskHolder.pointer,
      );
    } else {
      throw UnsupportedError('Unsupported dtype for ceil: ${a.dtype}');
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes element-wise floor of the array.
///
/// It is an error if [a], [where], or [out] is disposed (throws [StateError]),
/// if [a] has a complex dtype (throws [UnsupportedError]),
/// or if [out] has an incompatible shape or dtype (throws [ArgumentError]).
NDArray<T> floor<T extends DTypeTag>(
  NDArray<T> a, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute floor() on a disposed array.');
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for floor');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for floor',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        floor<T>(a, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    switch (a.dtype) {
      case DType.float64:
        if (a.isContiguous && result.isContiguous) {
          v_floor_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        }
      case DType.float32:
        if (a.isContiguous && result.isContiguous) {
          v_floor_float(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        }
      case DType.float16:
      case DType.bfloat16:
      case DType.int64:
      case DType.int32:
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

    if (a.dtype.isInteger || a.dtype == DType.boolean) {
      if (where == null) {
        a.copy(out: result);
      } else {
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => x,
          maskHolder.pointer,
        );
      }
    } else if (a.dtype.isFloating) {
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => castValue((x as num).floorToDouble(), a.dtype),
        maskHolder.pointer,
      );
    } else {
      throw UnsupportedError('Unsupported dtype for floor: ${a.dtype}');
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes element-wise round of the array to the given number of [decimals].
///
/// Rounds values to the nearest even value for halfway cases (round half to even,
/// matching `numpy.round` / `numpy.around`).
///
/// It is an error if [a], [where], or [out] is disposed (throws [StateError]),
/// if [a] has a complex dtype (throws [UnsupportedError]),
/// or if [out] has an incompatible shape or dtype (throws [ArgumentError]).
NDArray<T> round<T extends DTypeTag>(
  NDArray<T> a, {
  int decimals = 0,
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute round() on a disposed array.');
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for round');
  }
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for round',
      );
    }
    if (sharesMemory(a, out) || (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<T>.create(a.shape, a.dtype);
        round<T>(a, decimals: decimals, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    if (decimals == 0) {
      if (a.isContiguous && result.isContiguous) {
        switch (a.dtype) {
          case DType.float64:
            v_round_double(
              a.typedPointer(),
              result.typedPointer(),
              a.size,
              maskHolder.pointer,
            );
            return result;
          case DType.float32:
            v_round_float(
              a.typedPointer(),
              result.typedPointer(),
              a.size,
              maskHolder.pointer,
            );
            return result;
          case DType.float16:
          case DType.bfloat16:
          case DType.int64:
          case DType.int32:
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
      } else {
        final rank = a.shape.length;
        if (rank <= 8 &&
            (a.dtype == DType.float64 || a.dtype == DType.float32)) {
          final marker = ScratchArena.marker;
          try {
            final cBuffer = ScratchArena.getStridedBuffer(rank);
            final cShape = cBuffer;
            final cStridesA = cBuffer + rank;
            final cStridesRes = cBuffer + (rank * 2);
            for (var i = 0; i < rank; i++) {
              cShape[i] = a.shape[i];
              cStridesA[i] = a.strides[i];
              cStridesRes[i] = result.strides[i];
            }
            if (a.dtype == DType.float64) {
              s_rint_double(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            } else {
              s_rint_float(
                a.typedPointer(),
                cStridesA,
                result.typedPointer(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          } finally {
            ScratchArena.reset(marker);
          }
        }
      }
    }

    double roundHalfToEven(double dx) {
      if (dx.isInfinite || dx.isNaN || dx == 0.0) return dx;
      final floorVal = dx.floorToDouble();
      final ceilVal = dx.ceilToDouble();
      final distFloor = dx - floorVal;
      final distCeil = ceilVal - dx;
      if (distFloor < distCeil) return floorVal;
      if (distCeil < distFloor) return ceilVal;
      final evenVal = (floorVal % 2.0 == 0.0) ? floorVal : ceilVal;
      return evenVal == 0.0 ? (dx.isNegative ? -0.0 : 0.0) : evenVal;
    }

    if (a.dtype.isInteger || a.dtype == DType.boolean) {
      if (decimals >= 0) {
        if (where == null) {
          a.copy(out: result);
        } else {
          unaryOp<DTypeTag, DTypeTag>(
            result,
            a,
            a.shape,
            a.strides,
            result.strides,
            0,
            a.offsetElements,
            result.offsetElements,
            (x) => x,
            maskHolder.pointer,
          );
        }
      } else {
        final invFactor = math.pow(10.0, -decimals).toDouble();
        unaryOp<DTypeTag, DTypeTag>(
          result,
          a,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) {
            if (a.dtype == DType.boolean) return false;
            final dx = a.dtype == DType.uint64
                ? uint64ToDouble(x as int)
                : (x as num).toDouble();
            if (invFactor.isInfinite) return castValue(0, a.dtype);
            final rounded = (roundHalfToEven(dx / invFactor) * invFactor)
                .round();
            return castValue(rounded, a.dtype);
          },
          maskHolder.pointer,
        );
      }
    } else if (a.dtype.isFloating) {
      final factor = decimals >= 0
          ? math.pow(10.0, decimals).toDouble()
          : math.pow(10.0, -decimals).toDouble();
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) {
          final dx = (x as num).toDouble();
          if (decimals == 0) {
            return castValue(roundHalfToEven(dx), a.dtype);
          }
          if (dx.isInfinite || dx.isNaN || dx == 0.0) {
            return castValue(dx, a.dtype);
          }
          if (decimals > 0) {
            final scaled = dx * factor;
            if (scaled.isInfinite) return castValue(dx, a.dtype);
            return castValue(roundHalfToEven(scaled) / factor, a.dtype);
          } else {
            if (factor.isInfinite) {
              return castValue(dx.isNegative ? -0.0 : 0.0, a.dtype);
            }
            return castValue(roundHalfToEven(dx / factor) * factor, a.dtype);
          }
        },
        maskHolder.pointer,
      );
    } else {
      throw UnsupportedError('Unsupported dtype for round: ${a.dtype}');
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Element-wise addition of two arrays.
///
/// Both [a] and [b] must have the same [DType]. For [DType.boolean], computes
/// logical OR (`a | b`), matching `numpy.add`.
NDArray<T> add<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute add() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  return _addKernel<T, T, T>(a, b, a.dtype, where: where, out: out);
}

NDArray<R>
_addKernel<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> targetDType, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype',
      );
    }
    if (sharesMemory(a, out) ||
        sharesMemory(b, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(commonShape, targetDType);
        _addKernel<Ta, Tb, R>(a, b, targetDType, where: where, out: temp);
        temp.copy(out: out);
        return out;
      });
    }
  }
  if (a.dtype == DType.boolean &&
      b.dtype == DType.boolean &&
      targetDType == DType.boolean) {
    if (out != null) {
      final outView = NDArray<Boolean>.view(
        out,
        shape: out.shape,
        strides: out.strides,
      );
      logicalOr<Boolean>(
        a as NDArray<Boolean>,
        b as NDArray<Boolean>,
        where: where,
        out: outView,
      );
      return out;
    }
    return logicalOr<Boolean>(
          a as NDArray<Boolean>,
          b as NDArray<Boolean>,
          where: where,
        )
        as NDArray<R>;
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<R> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<R>.create(commonShape, targetDType, zeroInit: where != null);
    // Specialized paths for Float64 (as in original extensions.dart)
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int64> cShape;
    late final ffi.Pointer<ffi.Int64> cStridesA;
    late final ffi.Pointer<ffi.Int64> cStridesB;
    late final ffi.Pointer<ffi.Int64> cStridesRes;
    if (!isContig) {
      final cBuffer = ScratchArena.getStridedBuffer(ndim);
      cShape = cBuffer;
      cStridesA = cBuffer + ndim;
      cStridesB = cBuffer + (ndim * 2);
      cStridesRes = cBuffer + (ndim * 3);

      for (var i = 0; i < commonShape.length; i++) {
        cShape[i] = commonShape[i];
        cStridesA[i] = stridesA[i];
        cStridesB[i] = stridesB[i];
        cStridesRes[i] = result.strides[i];
      }
    }
    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_add_double_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_add_double_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32) when isContig:
        v_add_double_float_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_add_double_float_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64) when isContig:
        v_add_double_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_add_double_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32) when isContig:
        v_add_double_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_add_double_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_add_double_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_add_double_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16) when isContig:
        v_add_double_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_add_double_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128) when isContig:
        v_add_double_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_add_double_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64) when isContig:
        v_add_double_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_add_double_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64) when isContig:
        v_add_double_float_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_add_double_float_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32) when isContig:
        v_add_float_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_add_float_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64)
          when isContig && result.dtype == DType.float32:
        v_add_float_int64_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64) when result.dtype == DType.float32:
        s_add_float_int64_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32)
          when isContig && result.dtype == DType.float32:
        v_add_float_int32_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32) when result.dtype == DType.float32:
        s_add_float_int32_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_add_float_uint8_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_add_float_uint8_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16) when isContig:
        v_add_float_int16_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_add_float_int16_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128) when isContig:
        v_add_float_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_add_float_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64) when isContig:
        v_add_float_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_add_float_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64) when isContig:
        v_add_double_int64_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_add_double_int64_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_add_float_int64_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32) when result.dtype == DType.float32:
        s_add_float_int64_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64) when isContig:
        v_add_int64_int64_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
        s_add_int64_int64_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32) when isContig:
        v_add_int64_int32_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_add_int64_int32_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_add_int64_uint8_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_add_int64_uint8_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16) when isContig:
        v_add_int64_int16_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_add_int64_int16_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128) when isContig:
        v_add_int64_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_add_int64_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64)
          when isContig && result.dtype == DType.complex64:
        v_add_int64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64) when result.dtype == DType.complex64:
        s_add_int64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64) when isContig:
        v_add_double_int32_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_add_double_int32_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_add_float_int32_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32) when result.dtype == DType.float32:
        s_add_float_int32_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64) when isContig:
        v_add_int64_int32_int64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_add_int64_int32_int64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32) when isContig:
        v_add_int32_int32_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
        s_add_int32_int32_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_add_int32_uint8_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_add_int32_uint8_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16) when isContig:
        v_add_int32_int16_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_add_int32_int16_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128) when isContig:
        v_add_int32_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_add_int32_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64) when isContig:
        v_add_int32_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_add_int32_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_add_double_uint8_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_add_double_uint8_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_add_float_uint8_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_add_float_uint8_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_add_int64_uint8_int64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_add_int64_uint8_int64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_add_int32_uint8_int32(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_add_int32_uint8_int32(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean) when isContig:
      case (DType.boolean, DType.uint8) when isContig:
      case (DType.uint8, DType.boolean) when isContig:
      case (DType.uint8, DType.uint8) when isContig:
        v_add_uint8_uint8_uint8(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
        s_add_uint8_uint8_uint8(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_add_uint8_int16_int16(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_add_uint8_int16_int16(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_add_uint8_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_add_uint8_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_add_uint8_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_add_uint8_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64) when isContig:
        v_add_double_int16_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_add_double_int16_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32) when isContig:
        v_add_float_int16_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_add_float_int16_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64) when isContig:
        v_add_int64_int16_int64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_add_int64_int16_int64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32) when isContig:
        v_add_int32_int16_int32(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_add_int32_int16_int32(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_add_uint8_int16_int16(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_add_uint8_int16_int16(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16) when isContig:
        v_add_int16_int16_int16(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
        s_add_int16_int16_int16(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128) when isContig:
        v_add_int16_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_add_int16_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64) when isContig:
        v_add_int16_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_add_int16_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64) when isContig:
        v_add_double_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_add_double_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32) when isContig:
        v_add_float_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_add_float_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64) when isContig:
        v_add_int64_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_add_int64_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32) when isContig:
        v_add_int32_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_add_int32_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_add_uint8_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_add_uint8_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16) when isContig:
        v_add_int16_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_add_int16_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128) when isContig:
        v_add_cpx_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_add_cpx_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64) when isContig:
        v_add_cpx_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_add_cpx_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64) when isContig:
        v_add_double_cpx64_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_add_double_cpx64_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32) when isContig:
        v_add_float_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_add_float_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64)
          when isContig && result.dtype == DType.complex64:
        v_add_int64_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64) when result.dtype == DType.complex64:
        s_add_int64_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32) when isContig:
        v_add_int32_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_add_int32_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_add_uint8_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_add_uint8_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16) when isContig:
        v_add_int16_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_add_int16_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128) when isContig:
        v_add_cpx_cpx64_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_add_cpx_cpx64_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64) when isContig:
        v_add_cpx64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_add_cpx64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint64, DType.uint64) when isContig:
        v_add_int64_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint64, DType.uint64):
        s_add_int64_int64_int64(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint32, DType.uint32) when isContig:
        v_add_int32_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint32, DType.uint32):
        s_add_int32_int32_int32(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint16, DType.uint16) when isContig:
        v_add_int16_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint16, DType.uint16):
        s_add_int16_int16_int16(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int8, DType.int8) when isContig:
        v_add_uint8_uint8_uint8(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int8, DType.int8):
        s_add_uint8_uint8_uint8(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      default:
        if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
          final cpxA = castNDArray(a, DType.complex128);
          final cpxB = castNDArray(b, DType.complex128);
          final cpxRes = add<Complex128>(cpxA, cpxB, where: where);
          final casted = castNDArray(cpxRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(cpxA, a)) cpxA.dispose();
          if (!identical(cpxB, b)) cpxB.dispose();
          cpxRes.dispose();
          if (!identical(casted, cpxRes)) casted.dispose();
          return result;
        } else if (result.dtype.isInteger) {
          final intA = castNDArray(a, DType.int64);
          final intB = castNDArray(b, DType.int64);
          final intRes = add<Int64>(intA, intB, where: where);
          final casted = castNDArray(intRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(intA, a)) intA.dispose();
          if (!identical(intB, b)) intB.dispose();
          intRes.dispose();
          if (!identical(casted, intRes)) casted.dispose();
          return result;
        } else {
          final doubleA = castNDArray(a, DType.float64);
          final doubleB = castNDArray(b, DType.float64);
          final doubleRes = add<Float64>(doubleA, doubleB, where: where);
          final casted = castNDArray(doubleRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(doubleA, a)) doubleA.dispose();
          if (!identical(doubleB, b)) doubleB.dispose();
          doubleRes.dispose();
          if (!identical(casted, doubleRes)) casted.dispose();
          return result;
        }
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
}

/// Element-wise subtraction of two arrays.
NDArray<T> subtract<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute subtract() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  return _subtractKernel<T, T, T>(a, b, a.dtype, where: where, out: out);
}

NDArray<R>
_subtractKernel<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> targetDType, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (targetDType == DType.boolean) {
    throw UnsupportedError(
      "Boolean subtract, the '-' operator, is not supported; use logicalXor or bitwiseXor instead.",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype',
      );
    }
    if (sharesMemory(a, out) ||
        sharesMemory(b, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(commonShape, out.dtype);
        _subtractKernel<Ta, Tb, R>(a, b, targetDType, where: where, out: temp);
        return temp.copy(out: out);
      });
    }
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<R> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<R>.create(commonShape, targetDType, zeroInit: where != null);
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int64> cShape;
    late final ffi.Pointer<ffi.Int64> cStridesA;
    late final ffi.Pointer<ffi.Int64> cStridesB;
    late final ffi.Pointer<ffi.Int64> cStridesRes;
    if (!isContig) {
      final cBuffer = ScratchArena.getStridedBuffer(ndim);
      cShape = cBuffer;
      cStridesA = cBuffer + ndim;
      cStridesB = cBuffer + (ndim * 2);
      cStridesRes = cBuffer + (ndim * 3);

      for (var i = 0; i < commonShape.length; i++) {
        cShape[i] = commonShape[i];
        cStridesA[i] = stridesA[i];
        cStridesB[i] = stridesB[i];
        cStridesRes[i] = result.strides[i];
      }
    }
    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_sub_double_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_sub_double_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32) when isContig:
        v_sub_double_float_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_sub_double_float_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64) when isContig:
        v_sub_double_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_sub_double_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32) when isContig:
        v_sub_double_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_sub_double_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_sub_double_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_sub_double_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16) when isContig:
        v_sub_double_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_sub_double_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128) when isContig:
        v_sub_double_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_sub_double_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64) when isContig:
        v_sub_double_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_sub_double_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64) when isContig:
        v_sub_float_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_sub_float_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32) when isContig:
        v_sub_float_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_sub_float_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64)
          when isContig && result.dtype == DType.float32:
        v_sub_float_int64_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64) when result.dtype == DType.float32:
        s_sub_float_int64_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32)
          when isContig && result.dtype == DType.float32:
        v_sub_float_int32_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32) when result.dtype == DType.float32:
        s_sub_float_int32_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_sub_float_uint8_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_sub_float_uint8_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16) when isContig:
        v_sub_float_int16_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_sub_float_int16_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128) when isContig:
        v_sub_float_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_sub_float_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64) when isContig:
        v_sub_float_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_sub_float_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64) when isContig:
        v_sub_int64_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_sub_int64_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_sub_int64_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32) when result.dtype == DType.float32:
        s_sub_int64_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64) when isContig:
        v_sub_int64_int64_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
        s_sub_int64_int64_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32) when isContig:
        v_sub_int64_int32_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_sub_int64_int32_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_sub_int64_uint8_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_sub_int64_uint8_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16) when isContig:
        v_sub_int64_int16_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_sub_int64_int16_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128) when isContig:
        v_sub_int64_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_sub_int64_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64)
          when isContig && result.dtype == DType.complex64:
        v_sub_int64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64) when result.dtype == DType.complex64:
        s_sub_int64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64) when isContig:
        v_sub_int32_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_sub_int32_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_sub_int32_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32) when result.dtype == DType.float32:
        s_sub_int32_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64) when isContig:
        v_sub_int32_int64_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_sub_int32_int64_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32) when isContig:
        v_sub_int32_int32_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
        s_sub_int32_int32_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_sub_int32_uint8_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_sub_int32_uint8_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16) when isContig:
        v_sub_int32_int16_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_sub_int32_int16_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128) when isContig:
        v_sub_int32_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_sub_int32_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64) when isContig:
        v_sub_int32_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_sub_int32_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_sub_uint8_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_sub_uint8_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_sub_uint8_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_sub_uint8_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_sub_uint8_int64_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_sub_uint8_int64_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_sub_uint8_int32_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_sub_uint8_int32_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean) when isContig:
      case (DType.boolean, DType.uint8) when isContig:
      case (DType.uint8, DType.boolean) when isContig:
      case (DType.uint8, DType.uint8) when isContig:
        v_sub_uint8_uint8_uint8(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
        s_sub_uint8_uint8_uint8(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_sub_uint8_int16_int16(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_sub_uint8_int16_int16(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_sub_uint8_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_sub_uint8_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_sub_uint8_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_sub_uint8_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64) when isContig:
        v_sub_int16_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_sub_int16_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32) when isContig:
        v_sub_int16_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_sub_int16_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64) when isContig:
        v_sub_int16_int64_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_sub_int16_int64_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32) when isContig:
        v_sub_int16_int32_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_sub_int16_int32_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_sub_int16_uint8_int16(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_sub_int16_uint8_int16(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16) when isContig:
        v_sub_int16_int16_int16(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
        s_sub_int16_int16_int16(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128) when isContig:
        v_sub_int16_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_sub_int16_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64) when isContig:
        v_sub_int16_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_sub_int16_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64) when isContig:
        v_sub_cpx_double_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_sub_cpx_double_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32) when isContig:
        v_sub_cpx_float_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_sub_cpx_float_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64) when isContig:
        v_sub_cpx_int64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_sub_cpx_int64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32) when isContig:
        v_sub_cpx_int32_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_sub_cpx_int32_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_sub_cpx_uint8_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_sub_cpx_uint8_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16) when isContig:
        v_sub_cpx_int16_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_sub_cpx_int16_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128) when isContig:
        v_sub_cpx_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_sub_cpx_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64) when isContig:
        v_sub_cpx_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_sub_cpx_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64) when isContig:
        v_sub_cpx64_double_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_sub_cpx64_double_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32) when isContig:
        v_sub_cpx64_float_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_sub_cpx64_float_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64)
          when isContig && result.dtype == DType.complex64:
        v_sub_cpx64_int64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64) when result.dtype == DType.complex64:
        s_sub_cpx64_int64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32) when isContig:
        v_sub_cpx64_int32_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_sub_cpx64_int32_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_sub_cpx64_uint8_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_sub_cpx64_uint8_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16) when isContig:
        v_sub_cpx64_int16_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_sub_cpx64_int16_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128) when isContig:
        v_sub_cpx64_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_sub_cpx64_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64) when isContig:
        v_sub_cpx64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_sub_cpx64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint64, DType.uint64) when isContig:
        v_sub_int64_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint64, DType.uint64):
        s_sub_int64_int64_int64(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint32, DType.uint32) when isContig:
        v_sub_int32_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint32, DType.uint32):
        s_sub_int32_int32_int32(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint16, DType.uint16) when isContig:
        v_sub_int16_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint16, DType.uint16):
        s_sub_int16_int16_int16(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int8, DType.int8) when isContig:
        v_sub_uint8_uint8_uint8(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int8, DType.int8):
        s_sub_uint8_uint8_uint8(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      default:
        if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
          final cpxA = castNDArray(a, DType.complex128);
          final cpxB = castNDArray(b, DType.complex128);
          final cpxRes = subtract<Complex128>(cpxA, cpxB, where: where);
          final casted = castNDArray(cpxRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(cpxA, a)) cpxA.dispose();
          if (!identical(cpxB, b)) cpxB.dispose();
          cpxRes.dispose();
          if (!identical(casted, cpxRes)) casted.dispose();
          return result;
        } else if (result.dtype.isInteger) {
          final intA = castNDArray(a, DType.int64);
          final intB = castNDArray(b, DType.int64);
          final intRes = subtract<Int64>(intA, intB, where: where);
          final casted = castNDArray(intRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(intA, a)) intA.dispose();
          if (!identical(intB, b)) intB.dispose();
          intRes.dispose();
          if (!identical(casted, intRes)) casted.dispose();
          return result;
        } else {
          final doubleA = castNDArray(a, DType.float64);
          final doubleB = castNDArray(b, DType.float64);
          final doubleRes = subtract<Float64>(doubleA, doubleB, where: where);
          final casted = castNDArray(doubleRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(doubleA, a)) doubleA.dispose();
          if (!identical(doubleB, b)) doubleB.dispose();
          doubleRes.dispose();
          if (!identical(casted, doubleRes)) casted.dispose();
          return result;
        }
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
}

/// Element-wise multiplication of two arrays with full broadcasting support.
///
/// Both [a] and [b] must have the same [DType]. For [DType.boolean], computes
/// logical AND (`a & b`), matching `numpy.multiply`.
///
/// **Overflow behavior:**
/// - **Integer arrays** (`int32`, `int64`, etc.) overflow silently wrapping around via standard two's complement.
/// - **Floating-point arrays** (`float32`, `float64`) overflow silently to `double.infinity` or `double.negativeInfinity` per IEEE 754.
NDArray<T> multiply<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute multiply() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  return _multiplyKernel<T, T, T>(a, b, a.dtype, where: where, out: out);
}

NDArray<R>
_multiplyKernel<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> targetDType, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype',
      );
    }
    if (sharesMemory(a, out) ||
        sharesMemory(b, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(commonShape, out.dtype);
        _multiplyKernel<Ta, Tb, R>(a, b, targetDType, where: where, out: temp);
        return temp.copy(out: out);
      });
    }
  }
  if (a.dtype == DType.boolean &&
      b.dtype == DType.boolean &&
      targetDType == DType.boolean) {
    if (out != null) {
      final outView = NDArray<Boolean>.view(
        out,
        shape: out.shape,
        strides: out.strides,
      );
      logicalAnd<Boolean>(
        a as NDArray<Boolean>,
        b as NDArray<Boolean>,
        where: where,
        out: outView,
      );
      return out;
    }
    return logicalAnd<Boolean>(
          a as NDArray<Boolean>,
          b as NDArray<Boolean>,
          where: where,
        )
        as NDArray<R>;
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<R> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<R>.create(commonShape, targetDType, zeroInit: where != null);
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int64> cShape;
    late final ffi.Pointer<ffi.Int64> cStridesA;
    late final ffi.Pointer<ffi.Int64> cStridesB;
    late final ffi.Pointer<ffi.Int64> cStridesRes;
    if (!isContig) {
      final cBuffer = ScratchArena.getStridedBuffer(ndim);
      cShape = cBuffer;
      cStridesA = cBuffer + ndim;
      cStridesB = cBuffer + (ndim * 2);
      cStridesRes = cBuffer + (ndim * 3);

      for (var i = 0; i < commonShape.length; i++) {
        cShape[i] = commonShape[i];
        cStridesA[i] = stridesA[i];
        cStridesB[i] = stridesB[i];
        cStridesRes[i] = result.strides[i];
      }
    }

    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_mul_double_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_mul_double_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32) when isContig:
        v_mul_double_float_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_mul_double_float_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64) when isContig:
        v_mul_double_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_mul_double_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32) when isContig:
        v_mul_double_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_mul_double_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_mul_double_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_mul_double_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16) when isContig:
        v_mul_double_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_mul_double_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128) when isContig:
        v_mul_double_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_mul_double_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64) when isContig:
        v_mul_double_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_mul_double_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64) when isContig:
        v_mul_double_float_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_mul_double_float_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32) when isContig:
        v_mul_float_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_mul_float_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64)
          when isContig && result.dtype == DType.float32:
        v_mul_float_int64_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64) when result.dtype == DType.float32:
        s_mul_float_int64_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32)
          when isContig && result.dtype == DType.float32:
        v_mul_float_int32_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32) when result.dtype == DType.float32:
        s_mul_float_int32_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_mul_float_uint8_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_mul_float_uint8_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16) when isContig:
        v_mul_float_int16_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_mul_float_int16_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128) when isContig:
        v_mul_float_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_mul_float_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64) when isContig:
        v_mul_float_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_mul_float_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64) when isContig:
        v_mul_double_int64_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_mul_double_int64_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_mul_float_int64_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32) when result.dtype == DType.float32:
        s_mul_float_int64_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64) when isContig:
        v_mul_int64_int64_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
        s_mul_int64_int64_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32) when isContig:
        v_mul_int64_int32_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_mul_int64_int32_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_mul_int64_uint8_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_mul_int64_uint8_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16) when isContig:
        v_mul_int64_int16_int64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_mul_int64_int16_int64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128) when isContig:
        v_mul_int64_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_mul_int64_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64)
          when isContig && result.dtype == DType.complex64:
        v_mul_int64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64) when result.dtype == DType.complex64:
        s_mul_int64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64) when isContig:
        v_mul_double_int32_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_mul_double_int32_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_mul_float_int32_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32) when result.dtype == DType.float32:
        s_mul_float_int32_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64) when isContig:
        v_mul_int64_int32_int64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_mul_int64_int32_int64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32) when isContig:
        v_mul_int32_int32_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
        s_mul_int32_int32_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_mul_int32_uint8_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_mul_int32_uint8_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16) when isContig:
        v_mul_int32_int16_int32(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_mul_int32_int16_int32(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128) when isContig:
        v_mul_int32_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_mul_int32_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64) when isContig:
        v_mul_int32_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_mul_int32_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_mul_double_uint8_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_mul_double_uint8_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_mul_float_uint8_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_mul_float_uint8_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_mul_int64_uint8_int64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_mul_int64_uint8_int64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_mul_int32_uint8_int32(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_mul_int32_uint8_int32(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean) when isContig:
      case (DType.boolean, DType.uint8) when isContig:
      case (DType.uint8, DType.boolean) when isContig:
      case (DType.uint8, DType.uint8) when isContig:
        v_mul_uint8_uint8_uint8(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
        s_mul_uint8_uint8_uint8(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_mul_uint8_int16_int16(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_mul_uint8_int16_int16(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_mul_uint8_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_mul_uint8_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_mul_uint8_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_mul_uint8_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64) when isContig:
        v_mul_double_int16_double(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_mul_double_int16_double(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32) when isContig:
        v_mul_float_int16_float(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_mul_float_int16_float(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64) when isContig:
        v_mul_int64_int16_int64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_mul_int64_int16_int64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32) when isContig:
        v_mul_int32_int16_int32(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_mul_int32_int16_int32(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_mul_uint8_int16_int16(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_mul_uint8_int16_int16(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16) when isContig:
        v_mul_int16_int16_int16(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
        s_mul_int16_int16_int16(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128) when isContig:
        v_mul_int16_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_mul_int16_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64) when isContig:
        v_mul_int16_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_mul_int16_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64) when isContig:
        v_mul_double_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_mul_double_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32) when isContig:
        v_mul_float_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_mul_float_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64) when isContig:
        v_mul_int64_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_mul_int64_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32) when isContig:
        v_mul_int32_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_mul_int32_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_mul_uint8_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_mul_uint8_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16) when isContig:
        v_mul_int16_cpx_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_mul_int16_cpx_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128) when isContig:
        v_mul_cpx_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_mul_cpx_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64) when isContig:
        v_mul_cpx_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_mul_cpx_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64) when isContig:
        v_mul_double_cpx64_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_mul_double_cpx64_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32) when isContig:
        v_mul_float_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_mul_float_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64)
          when isContig && result.dtype == DType.complex64:
        v_mul_int64_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64) when result.dtype == DType.complex64:
        s_mul_int64_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32) when isContig:
        v_mul_int32_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_mul_int32_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_mul_uint8_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_mul_uint8_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16) when isContig:
        v_mul_int16_cpx64_cpx64(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_mul_int16_cpx64_cpx64(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128) when isContig:
        v_mul_cpx_cpx64_cpx(
          b.typedPointer(),
          a.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_mul_cpx_cpx64_cpx(
          b.typedPointer(),
          cStridesB,
          a.typedPointer(),
          cStridesA,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64) when isContig:
        v_mul_cpx64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_mul_cpx64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint64, DType.uint64) when isContig:
        v_mul_int64_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint64, DType.uint64):
        s_mul_int64_int64_int64(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint32, DType.uint32) when isContig:
        v_mul_int32_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint32, DType.uint32):
        s_mul_int32_int32_int32(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint16, DType.uint16) when isContig:
        v_mul_int16_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.uint16, DType.uint16):
        s_mul_int16_int16_int16(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int8, DType.int8) when isContig:
        v_mul_uint8_uint8_uint8(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int8, DType.int8):
        s_mul_uint8_uint8_uint8(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      default:
        if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
          final cpxA = castNDArray(a, DType.complex128);
          final cpxB = castNDArray(b, DType.complex128);
          final cpxRes = multiply<Complex128>(cpxA, cpxB, where: where);
          final casted = castNDArray(cpxRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(cpxA, a)) cpxA.dispose();
          if (!identical(cpxB, b)) cpxB.dispose();
          cpxRes.dispose();
          if (!identical(casted, cpxRes)) casted.dispose();
          return result;
        } else if (result.dtype.isInteger) {
          final intA = castNDArray(a, DType.int64);
          final intB = castNDArray(b, DType.int64);
          final intRes = multiply<Int64>(intA, intB, where: where);
          final casted = castNDArray(intRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(intA, a)) intA.dispose();
          if (!identical(intB, b)) intB.dispose();
          intRes.dispose();
          if (!identical(casted, intRes)) casted.dispose();
          return result;
        } else {
          final doubleA = castNDArray(a, DType.float64);
          final doubleB = castNDArray(b, DType.float64);
          final doubleRes = multiply<Float64>(doubleA, doubleB, where: where);
          final casted = castNDArray(doubleRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(doubleA, a)) doubleA.dispose();
          if (!identical(doubleB, b)) doubleB.dispose();
          doubleRes.dispose();
          if (!identical(casted, doubleRes)) casted.dispose();
          return result;
        }
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
}

/// Element-wise division of two arrays with full broadcasting support.
///
/// Both [a] and [b] must have the same [DType]. Always upcasts integer and
/// boolean operands to [DType.float64] and performs floating-point division.
///
/// **Division by Zero:**
/// Division by zero is handled silently under IEEE 754 floating-point rules:
/// - Dividing a non-zero value by zero results in `double.infinity` or `double.negativeInfinity`.
/// - Dividing zero by zero results in `double.nan`.
///
/// **Preconditions:**
/// - The input arrays [a] and [b] must not be disposed and must have the same [DType].
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
///
/// It is an error if:
/// - [a], [b], or [out] is disposed (throws [StateError]).
/// - [a] and [b] have different dtypes (throws [ArgumentError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
NDArray<R> divide<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute divide() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  var targetDType = resolveDType(a.dtype, b.dtype);
  if (targetDType.isInteger || targetDType == DType.boolean) {
    targetDType = DType.float64;
  }
  return _divideKernel<T, T, R>(
    a,
    b,
    targetDType as DType<R>,
    where: where,
    out: out,
  );
}

NDArray<R>
_divideKernel<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> targetDType, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype',
      );
    }
    if (sharesMemory(a, out) ||
        sharesMemory(b, out) ||
        (where != null && sharesMemory(where, out))) {
      return NDArray.scope(() {
        final temp = where != null
            ? out.copy()
            : NDArray<R>.create(commonShape, out.dtype);
        _divideKernel<Ta, Tb, R>(a, b, targetDType, where: where, out: temp);
        return temp.copy(out: out);
      });
    }
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<R> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<R>.create(commonShape, targetDType, zeroInit: where != null);
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int64> cShape;
    late final ffi.Pointer<ffi.Int64> cStridesA;
    late final ffi.Pointer<ffi.Int64> cStridesB;
    late final ffi.Pointer<ffi.Int64> cStridesRes;
    if (!isContig) {
      final cBuffer = ScratchArena.getStridedBuffer(ndim);
      cShape = cBuffer;
      cStridesA = cBuffer + ndim;
      cStridesB = cBuffer + (ndim * 2);
      cStridesRes = cBuffer + (ndim * 3);

      for (var i = 0; i < commonShape.length; i++) {
        cShape[i] = commonShape[i];
        cStridesA[i] = stridesA[i];
        cStridesB[i] = stridesB[i];
        cStridesRes[i] = result.strides[i];
      }
    }
    switch ((a.dtype, b.dtype)) {
      // DIV cases
      case (DType.float64, DType.float64) when isContig:
        v_div_double_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_div_double_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32) when isContig:
        v_div_double_float_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_div_double_float_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64) when isContig:
        v_div_double_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_div_double_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32) when isContig:
        v_div_double_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_div_double_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_div_double_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_div_double_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16) when isContig:
        v_div_double_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_div_double_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128) when isContig:
        v_div_double_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_div_double_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64) when isContig:
        v_div_double_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_div_double_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64) when isContig:
        v_div_float_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_div_float_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32) when isContig:
        v_div_float_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_div_float_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64)
          when isContig && result.dtype == DType.float32:
        v_div_float_int64_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64) when result.dtype == DType.float32:
        s_div_float_int64_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32)
          when isContig && result.dtype == DType.float32:
        v_div_float_int32_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32) when result.dtype == DType.float32:
        s_div_float_int32_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_div_float_uint8_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_div_float_uint8_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16) when isContig:
        v_div_float_int16_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_div_float_int16_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128) when isContig:
        v_div_float_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_div_float_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64) when isContig:
        v_div_float_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_div_float_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64) when isContig:
        v_div_int64_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_div_int64_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_div_int64_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32) when result.dtype == DType.float32:
        s_div_int64_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64) when isContig:
        v_div_int64_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
        s_div_int64_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32) when isContig:
        v_div_int64_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_div_int64_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_div_int64_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_div_int64_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16) when isContig:
        v_div_int64_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_div_int64_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128) when isContig:
        v_div_int64_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_div_int64_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64)
          when isContig && result.dtype == DType.complex64:
        v_div_int64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64) when result.dtype == DType.complex64:
        s_div_int64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64) when isContig:
        v_div_int32_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_div_int32_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32)
          when isContig && result.dtype == DType.float32:
        v_div_int32_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32) when result.dtype == DType.float32:
        s_div_int32_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64) when isContig:
        v_div_int32_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_div_int32_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32) when isContig:
        v_div_int32_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
        s_div_int32_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_div_int32_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_div_int32_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16) when isContig:
        v_div_int32_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_div_int32_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128) when isContig:
        v_div_int32_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_div_int32_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64) when isContig:
        v_div_int32_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_div_int32_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_div_uint8_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_div_uint8_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_div_uint8_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_div_uint8_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_div_uint8_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_div_uint8_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_div_uint8_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_div_uint8_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean) when isContig:
      case (DType.boolean, DType.uint8) when isContig:
      case (DType.uint8, DType.boolean) when isContig:
      case (DType.uint8, DType.uint8) when isContig:
        v_div_uint8_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
        s_div_uint8_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_div_uint8_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_div_uint8_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_div_uint8_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_div_uint8_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_div_uint8_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_div_uint8_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64) when isContig:
        v_div_int16_double_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_div_int16_double_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32) when isContig:
        v_div_int16_float_float(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_div_int16_float_float(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64) when isContig:
        v_div_int16_int64_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_div_int16_int64_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32) when isContig:
        v_div_int16_int32_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_div_int16_int32_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_div_int16_uint8_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_div_int16_uint8_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16) when isContig:
        v_div_int16_int16_double(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
        s_div_int16_int16_double(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128) when isContig:
        v_div_int16_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_div_int16_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64) when isContig:
        v_div_int16_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_div_int16_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64) when isContig:
        v_div_cpx_double_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_div_cpx_double_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32) when isContig:
        v_div_cpx_float_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_div_cpx_float_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64) when isContig:
        v_div_cpx_int64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_div_cpx_int64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32) when isContig:
        v_div_cpx_int32_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_div_cpx_int32_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_div_cpx_uint8_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_div_cpx_uint8_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16) when isContig:
        v_div_cpx_int16_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_div_cpx_int16_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128) when isContig:
        v_div_cpx_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_div_cpx_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64) when isContig:
        v_div_cpx_cpx64_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_div_cpx_cpx64_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64) when isContig:
        v_div_cpx64_double_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_div_cpx64_double_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32) when isContig:
        v_div_cpx64_float_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_div_cpx64_float_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64)
          when isContig && result.dtype == DType.complex64:
        v_div_cpx64_int64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64) when result.dtype == DType.complex64:
        s_div_cpx64_int64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32) when isContig:
        v_div_cpx64_int32_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_div_cpx64_int32_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_div_cpx64_uint8_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_div_cpx64_uint8_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16) when isContig:
        v_div_cpx64_int16_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_div_cpx64_int16_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128) when isContig:
        v_div_cpx64_cpx_cpx(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_div_cpx64_cpx_cpx(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64) when isContig:
        v_div_cpx64_cpx64_cpx64(
          a.typedPointer(),
          b.typedPointer(),
          result.typedPointer(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_div_cpx64_cpx64_cpx64(
          a.typedPointer(),
          cStridesA,
          b.typedPointer(),
          cStridesB,
          result.typedPointer(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      default:
        if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
          final cpxA = castNDArray(a, DType.complex128);
          final cpxB = castNDArray(b, DType.complex128);
          final cpxRes = divide<Complex128, Complex128>(
            cpxA,
            cpxB,
            where: where,
          );
          final casted = castNDArray(cpxRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(cpxA, a)) cpxA.dispose();
          if (!identical(cpxB, b)) cpxB.dispose();
          cpxRes.dispose();
          if (!identical(casted, cpxRes)) casted.dispose();
          return result;
        } else {
          final doubleA = castNDArray(a, DType.float64);
          final doubleB = castNDArray(b, DType.float64);
          final doubleRes = divide<Float64, Float64>(
            doubleA,
            doubleB,
            where: where,
          );
          final casted = castNDArray(doubleRes, result.dtype);
          _copyMaskedResult(casted, result, where);
          if (!identical(doubleA, a)) doubleA.dispose();
          if (!identical(doubleB, b)) doubleB.dispose();
          doubleRes.dispose();
          if (!identical(casted, doubleRes)) casted.dispose();
          return result;
        }
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
}

void _copyMaskedResult(NDArray src, NDArray dest, NDArray<DTypeTag>? where) {
  if (where == null && dest.isContiguous && src.isContiguous) {
    custom_memcpy(dest.pointer, src.pointer, dest.size * dest.dtype.byteWidth);
    return;
  }
  final maskHolder = prepareMask(where, dest.shape);
  try {
    unaryOp<DTypeTag, DTypeTag>(
      dest,
      src,
      dest.shape,
      src.strides,
      dest.strides,
      0,
      src.offsetElements,
      dest.offsetElements,
      (x) => x,
      maskHolder.pointer,
    );
  } finally {
    maskHolder.dispose();
  }
}

bool _isNativeMixedKernelInput(DType<DTypeTag> dt) => switch (dt) {
  DType.float64 ||
  DType.float32 ||
  DType.int64 ||
  DType.int32 ||
  DType.int16 ||
  DType.uint8 ||
  DType.boolean ||
  DType.complex128 ||
  DType.complex64 => true,
  DType.float16 ||
  DType.bfloat16 ||
  DType.int8 ||
  DType.uint64 ||
  DType.uint32 ||
  DType.uint16 => false,
};

DType<DTypeTag>? _nativeMixedKernelOutputDType(
  DType<DTypeTag> a,
  DType<DTypeTag> b, {
  required bool isDivide,
}) {
  if (!isDivide &&
      a == b &&
      (a == DType.uint64 ||
          a == DType.uint32 ||
          a == DType.uint16 ||
          a == DType.int8)) {
    return a;
  }
  if (!_isNativeMixedKernelInput(a) || !_isNativeMixedKernelInput(b)) {
    return null;
  }
  if (a == DType.boolean && b == DType.boolean) {
    return isDivide ? DType.float64 : DType.boolean;
  }
  if (a == DType.complex128 || b == DType.complex128) {
    return DType.complex128;
  }
  if (a == DType.complex64 || b == DType.complex64) {
    return (a == DType.float64 || b == DType.float64)
        ? DType.complex128
        : DType.complex64;
  }
  if (a == DType.float64 || b == DType.float64) {
    return DType.float64;
  }
  if (a == DType.float32 || b == DType.float32) {
    return DType.float32;
  }
  if (isDivide) {
    return DType.float64;
  }
  if (a == DType.int64 || b == DType.int64) {
    return DType.int64;
  }
  if (a == DType.int32 || b == DType.int32) {
    return DType.int32;
  }
  if (a == DType.int16 || b == DType.int16) {
    return DType.int16;
  }
  return DType.uint8;
}

/// Element-wise addition of [a] and [b] computed into the specified target [dtype].
///
/// Accepts operands [a] and [b] of any compatible [DType] (including mixed
/// dtypes), dispatching directly to single-pass mixed-type SIMD kernels when
/// available or casting operands to [dtype]. Returns an [NDArray<R>] whose
/// static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy add](https://numpy.org/doc/stable/reference/generated/numpy.add.html)
NDArray<R> addAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute addAs() on a disposed array.');
  }
  if (_nativeMixedKernelOutputDType(a.dtype, b.dtype, isDivide: false) ==
      dtype) {
    return _addKernel<Ta, Tb, R>(a, b, dtype, where: where, out: out);
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = _addKernel<R, R, R>(
      aCast,
      bCast,
      dtype,
      where: where,
      out: out,
    );
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise subtraction of [a] and [b] computed into the specified target [dtype].
///
/// Accepts operands [a] and [b] of any compatible [DType] (including mixed
/// dtypes), dispatching directly to single-pass mixed-type SIMD kernels when
/// available or casting operands to [dtype]. Returns an [NDArray<R>] whose
/// static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy subtract](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html)
NDArray<R>
subtractAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute subtractAs() on a disposed array.');
  }
  if (_nativeMixedKernelOutputDType(a.dtype, b.dtype, isDivide: false) ==
      dtype) {
    return _subtractKernel<Ta, Tb, R>(a, b, dtype, where: where, out: out);
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = _subtractKernel<R, R, R>(
      aCast,
      bCast,
      dtype,
      where: where,
      out: out,
    );
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise multiplication of [a] and [b] computed into the specified target [dtype].
///
/// Accepts operands [a] and [b] of any compatible [DType] (including mixed
/// dtypes), dispatching directly to single-pass mixed-type SIMD kernels when
/// available or casting operands to [dtype]. Returns an [NDArray<R>] whose
/// static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy multiply](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
NDArray<R>
multiplyAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute multiplyAs() on a disposed array.');
  }
  if (_nativeMixedKernelOutputDType(a.dtype, b.dtype, isDivide: false) ==
      dtype) {
    return _multiplyKernel<Ta, Tb, R>(a, b, dtype, where: where, out: out);
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = _multiplyKernel<R, R, R>(
      aCast,
      bCast,
      dtype,
      where: where,
      out: out,
    );
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise true division of [a] by [b] computed into the specified target [dtype].
///
/// Accepts operands [a] and [b] of any compatible [DType] (including mixed
/// dtypes), dispatching directly to single-pass mixed-type SIMD kernels when
/// available or casting operands/result to [dtype]. Returns an [NDArray<R>]
/// whose static type [R] is inferred from [dtype].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy divide](https://numpy.org/doc/stable/reference/generated/numpy.divide.html)
NDArray<R>
divideAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute divideAs() on a disposed array.');
  }
  if (_nativeMixedKernelOutputDType(a.dtype, b.dtype, isDivide: true) ==
      dtype) {
    return _divideKernel<Ta, Tb, R>(a, b, dtype, where: where, out: out);
  }
  return NDArray.scope(() {
    if (!dtype.isInteger && dtype != DType.boolean) {
      final aCast = castNDArray<R>(a, dtype);
      final bCast = castNDArray<R>(b, dtype);
      final res = _divideKernel<R, R, R>(
        aCast,
        bCast,
        dtype,
        where: where,
        out: out,
      );
      return out ?? res.detachToParentScope();
    }
    final broadcastResult = broadcast(a, b);
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, broadcastResult.shape) || out.dtype != dtype) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape and dtype',
        );
      }
    }
    final NDArray<DTypeTag> divRes;
    if (a.dtype.isComplex || b.dtype.isComplex) {
      final aCast = castNDArray<Complex128>(a, DType.complex128);
      final bCast = castNDArray<Complex128>(b, DType.complex128);
      divRes = _divideKernel<Complex128, Complex128, Complex128>(
        aCast,
        bCast,
        DType.complex128,
        where: where,
      );
    } else {
      final aCast = castNDArray<Float64>(a, DType.float64);
      final bCast = castNDArray<Float64>(b, DType.float64);
      divRes = _divideKernel<Float64, Float64, Float64>(
        aCast,
        bCast,
        DType.float64,
        where: where,
      );
    }
    final casted = castNDArray<R>(divRes, dtype);
    if (out != null) {
      _copyMaskedResult(casted, out, where);
      return out;
    }
    return casted.detachToParentScope();
  });
}

/// Element-wise floor division of [a] by [b] computed into the specified target [dtype].
///
/// Casts [a] and [b] to [dtype] and computes `floorDivide` into [NDArray<R>].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy floor_divide](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html)
NDArray<R>
floorDivideAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute floorDivideAs() on a disposed array.');
  }
  if (a.dtype == dtype && b.dtype == dtype) {
    return floorDivide<R>(
      a as NDArray<R>,
      b as NDArray<R>,
      where: where,
      out: out,
    );
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = floorDivide<R>(aCast, bCast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise remainder (modulo) of [a] divided by [b] computed into the specified target [dtype].
///
/// Casts [a] and [b] to [dtype] and computes `remainder` into [NDArray<R>].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy remainder](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html)
NDArray<R>
remainderAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute remainderAs() on a disposed array.');
  }
  if (a.dtype == dtype && b.dtype == dtype) {
    return remainder<R>(
      a as NDArray<R>,
      b as NDArray<R>,
      where: where,
      out: out,
    );
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = remainder<R>(aCast, bCast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Alias for [remainderAs] matching `numpy.mod`.
///
/// Reference: [NumPy mod](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)
NDArray<R> modAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) => remainderAs<Ta, Tb, R>(a, b, dtype, where: where, out: out);

/// Element-wise C-style `fmod` remainder of [x1] divided by [x2] computed into the specified target [dtype].
///
/// Casts [x1] and [x2] to [dtype] and computes `fmod` into [NDArray<R>].
///
/// **Preconditions:**
/// - It is an error if [x1], [x2], [where], or [out] is disposed.
/// - [x1] and [x2] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy fmod](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html)
NDArray<R> fmodAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> x1,
  NDArray<Tb> x2,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute fmodAs() on a disposed array.');
  }
  if (x1.dtype == dtype && x2.dtype == dtype) {
    return fmod<R>(x1 as NDArray<R>, x2 as NDArray<R>, where: where, out: out);
  }
  return NDArray.scope(() {
    final x1Cast = castNDArray<R>(x1, dtype);
    final x2Cast = castNDArray<R>(x2, dtype);
    final res = fmod<R>(x1Cast, x2Cast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Computes element-wise quotient and remainder simultaneously into the specified target [dtype].
///
/// Casts [x1] and [x2] to [dtype] and returns `({NDArray<R> quotient, NDArray<R> remainder})`.
///
/// **Preconditions:**
/// - It is an error if [x1] or [x2] is disposed.
/// - [x1] and [x2] must have broadcast-compatible shapes.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy divmod](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html)
({NDArray<R> quotient, NDArray<R> remainder}) divmodAs<
  Ta extends DTypeTag,
  Tb extends DTypeTag,
  R extends DTypeTag
>(NDArray<Ta> x1, NDArray<Tb> x2, DType<R> dtype) {
  if (x1.isDisposed || x2.isDisposed) {
    throw StateError('Cannot execute divmodAs() on a disposed array.');
  }
  if (x1.dtype == dtype && x2.dtype == dtype) {
    return divmod<R>(x1 as NDArray<R>, x2 as NDArray<R>);
  }
  return NDArray.scope(() {
    final x1Cast = castNDArray<R>(x1, dtype);
    final x2Cast = castNDArray<R>(x2, dtype);
    final res = divmod<R>(x1Cast, x2Cast);
    return (
      quotient: res.quotient.detachToParentScope(),
      remainder: res.remainder.detachToParentScope(),
    );
  });
}

/// Element-wise exponentiation $a^b$ computed into the specified target [dtype].
///
/// Casts [a] and [b] to [dtype] and computes `power` into [NDArray<R>].
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy power](https://numpy.org/doc/stable/reference/generated/numpy.power.html)
NDArray<R>
powerAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute powerAs() on a disposed array.');
  }
  if (a.dtype == dtype && b.dtype == dtype) {
    return power<R>(a as NDArray<R>, b as NDArray<R>, where: where, out: out);
  }
  return NDArray.scope(() {
    final aCast = castNDArray<R>(a, dtype);
    final bCast = castNDArray<R>(b, dtype);
    final res = power<R>(aCast, bCast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// First array elements raised to powers from second array, element-wise,
/// promoting to at least double precision ([DType.float64] or [DType.complex128]).
///
/// Both [a] and [b] must have the same [DType]. Real and integer inputs are
/// promoted to [DType.float64]; complex inputs are promoted to [DType.complex128],
/// matching `numpy.float_power`.
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have the same [DType] and broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal the promoted output dtype ([DType.float64] or [DType.complex128]).
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy float_power](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html)
NDArray<R> floatPower<T extends DTypeTag, R extends DTypeTag>(
  NDArray<DTypeSpec<T, Object?, DTypeTag, DTypeTag, DTypeTag, DTypeTag, R>> a,
  NDArray<DTypeSpec<T, Object?, DTypeTag, DTypeTag, DTypeTag, DTypeTag, R>> b, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute floatPower() on a disposed array.');
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError.value(
      b.dtype,
      'b',
      'Must have the same dtype as a (${a.dtype})',
    );
  }
  final targetDType =
      (a.dtype.isComplex ? DType.complex128 : DType.float64) as DType<R>;
  return floatPowerAs<DTypeTag, DTypeTag, R>(
    a,
    b,
    targetDType,
    where: where,
    out: out,
  );
}

/// Element-wise float exponentiation of [a] to [b] computed in double precision
/// ([DType.float64] or [DType.complex128]) and converted to the specified target [dtype].
///
/// Accepts operands [a] and [b] of any compatible [DType] (including mixed
/// dtypes).
///
/// **Preconditions:**
/// - It is an error if [a], [b], [where], or [out] is disposed.
/// - [a] and [b] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy float_power](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html)
NDArray<R>
floatPowerAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      b.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute floatPowerAs() on a disposed array.');
  }
  final computeDType =
      (a.dtype.isComplex || b.dtype.isComplex || dtype.isComplex)
      ? DType.complex128
      : DType.float64;
  if (computeDType == dtype) {
    return powerAs<Ta, Tb, R>(a, b, dtype, where: where, out: out);
  }
  final broadcastResult = broadcast(a, b);
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, broadcastResult.shape) || out.dtype != dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype',
      );
    }
  }
  return NDArray.scope(() {
    final computed = powerAs<Ta, Tb, DTypeTag>(
      a,
      b,
      computeDType,
      where: where,
    );
    final casted = castNDArray<R>(computed, dtype);
    if (out != null) {
      _copyMaskedResult(casted, out, where);
      return out;
    }
    return casted.detachToParentScope();
  });
}

/// Element-wise greatest common divisor of [x1] and [x2] computed into the specified target [dtype].
///
/// [dtype] must be an integer [DType]. Casts [x1] and [x2] to [dtype] and
/// computes `gcd` into [NDArray<R>].
///
/// **Preconditions:**
/// - It is an error if [x1], [x2], [where], or [out] is disposed.
/// - [dtype] must be an integer [DType].
/// - [x1] and [x2] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N \log(\min(|x_1|, |x_2|)))$.
///
/// Reference: [NumPy gcd](https://numpy.org/doc/stable/reference/generated/numpy.gcd.html)
NDArray<R> gcdAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> x1,
  NDArray<Tb> x2,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute gcdAs() on a disposed array.');
  }
  if (!dtype.isInteger) {
    throw UnsupportedError('gcdAs requires an integer target DType');
  }
  if (x1.dtype == dtype && x2.dtype == dtype) {
    return gcd<R>(x1 as NDArray<R>, x2 as NDArray<R>, where: where, out: out);
  }
  return NDArray.scope(() {
    final x1Cast = castNDArray<R>(x1, dtype);
    final x2Cast = castNDArray<R>(x2, dtype);
    final res = gcd<R>(x1Cast, x2Cast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise least common multiple of [x1] and [x2] computed into the specified target [dtype].
///
/// [dtype] must be an integer [DType]. Casts [x1] and [x2] to [dtype] and
/// computes `lcm` into [NDArray<R>].
///
/// **Preconditions:**
/// - It is an error if [x1], [x2], [where], or [out] is disposed.
/// - [dtype] must be an integer [DType].
/// - [x1] and [x2] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N \log(\min(|x_1|, |x_2|)))$.
///
/// Reference: [NumPy lcm](https://numpy.org/doc/stable/reference/generated/numpy.lcm.html)
NDArray<R> lcmAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> x1,
  NDArray<Tb> x2,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute lcmAs() on a disposed array.');
  }
  if (!dtype.isInteger) {
    throw UnsupportedError('lcmAs requires an integer target DType');
  }
  if (x1.dtype == dtype && x2.dtype == dtype) {
    return lcm<R>(x1 as NDArray<R>, x2 as NDArray<R>, where: where, out: out);
  }
  return NDArray.scope(() {
    final x1Cast = castNDArray<R>(x1, dtype);
    final x2Cast = castNDArray<R>(x2, dtype);
    final res = lcm<R>(x1Cast, x2Cast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Computes the Heaviside step function of [x1] with step value [x2] into the specified target [dtype].
///
/// Casts [x1] and [x2] to [dtype] (or [DType.float64] when [dtype] is not
/// [DType.float32] or [DType.float64]) and returns an [NDArray<R>].
///
/// **Preconditions:**
/// - It is an error if [x1], [x2], [where], or [out] is disposed.
/// - [x1] and [x2] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy heaviside](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html)
NDArray<R>
heavisideAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> x1,
  NDArray<Tb> x2,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute heavisideAs() on a disposed array.');
  }
  if (x1.dtype.isComplex || x2.dtype.isComplex || dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for heaviside');
  }
  if (x1.dtype == dtype && x2.dtype == dtype) {
    return heaviside<R>(
      x1 as NDArray<R>,
      x2 as NDArray<R>,
      where: where,
      out: out,
    );
  }
  return NDArray.scope(() {
    final x1Cast = castNDArray<R>(x1, dtype);
    final x2Cast = castNDArray<R>(x2, dtype);
    final res = heaviside<R>(x1Cast, x2Cast, where: where, out: out);
    return out ?? res.detachToParentScope();
  });
}

/// Computes $\log(e^{x_1} + e^{x_2})$ element-wise into the specified target [dtype].
///
/// Accepts operands [x1] and [x2] of any real or integer [DType] (including
/// mixed dtypes).
///
/// **Preconditions:**
/// - It is an error if [x1], [x2], [where], or [out] is disposed.
/// - [x1] and [x2] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
NDArray<R>
logaddexpAs<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> x1,
  NDArray<Tb> x2,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logaddexpAs() on a disposed array.');
  }
  if (x1.dtype.isComplex || x2.dtype.isComplex || dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for logaddexp');
  }
  final broadcastResult = broadcast(x1, x2);
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, broadcastResult.shape) || out.dtype != dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for logaddexpAs',
      );
    }
  }
  return NDArray.scope(() {
    if (dtype == DType.float32) {
      final aCast = castNDArray<Float32>(x1, DType.float32);
      final bCast = castNDArray<Float32>(x2, DType.float32);
      final res = logaddexp<Float32, Float32>(
        aCast,
        bCast,
        where: where,
        out: out as NDArray<Float32>?,
      );
      return out ?? (res.detachToParentScope() as NDArray<R>);
    }
    final aCast = castNDArray<Float64>(x1, DType.float64);
    final bCast = castNDArray<Float64>(x2, DType.float64);
    if (dtype == DType.float64) {
      final res = logaddexp<Float64, Float64>(
        aCast,
        bCast,
        where: where,
        out: out as NDArray<Float64>?,
      );
      return out ?? (res.detachToParentScope() as NDArray<R>);
    }
    final res = logaddexp<Float64, Float64>(aCast, bCast, where: where);
    final casted = castNDArray<R>(res, dtype);
    if (out != null) {
      _copyMaskedResult(casted, out, where);
      return out;
    }
    return casted.detachToParentScope();
  });
}

/// Computes $\log_2(2^{x_1} + 2^{x_2})$ element-wise into the specified target [dtype].
///
/// Accepts operands [x1] and [x2] of any real or integer [DType] (including
/// mixed dtypes).
///
/// **Preconditions:**
/// - It is an error if [x1], [x2], [where], or [out] is disposed.
/// - [x1] and [x2] must have broadcast-compatible shapes.
/// - If [out] is provided, its shape must match the broadcasted shape and its
///   dtype must equal [dtype].
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the broadcasted element count.
///
/// Reference: [NumPy logaddexp2](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html)
NDArray<R>
logaddexp2As<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> x1,
  NDArray<Tb> x2,
  DType<R> dtype, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logaddexp2As() on a disposed array.');
  }
  if (x1.dtype.isComplex || x2.dtype.isComplex || dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for logaddexp2');
  }
  final broadcastResult = broadcast(x1, x2);
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, broadcastResult.shape) || out.dtype != dtype) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for logaddexp2As',
      );
    }
  }
  return NDArray.scope(() {
    if (dtype == DType.float32) {
      final aCast = castNDArray<Float32>(x1, DType.float32);
      final bCast = castNDArray<Float32>(x2, DType.float32);
      final res = logaddexp2<Float32, Float32>(
        aCast,
        bCast,
        where: where,
        out: out as NDArray<Float32>?,
      );
      return out ?? (res.detachToParentScope() as NDArray<R>);
    }
    final aCast = castNDArray<Float64>(x1, DType.float64);
    final bCast = castNDArray<Float64>(x2, DType.float64);
    if (dtype == DType.float64) {
      final res = logaddexp2<Float64, Float64>(
        aCast,
        bCast,
        where: where,
        out: out as NDArray<Float64>?,
      );
      return out ?? (res.detachToParentScope() as NDArray<R>);
    }
    final res = logaddexp2<Float64, Float64>(aCast, bCast, where: where);
    final casted = castNDArray<R>(res, dtype);
    if (out != null) {
      _copyMaskedResult(casted, out, where);
      return out;
    }
    return casted.detachToParentScope();
  });
}
