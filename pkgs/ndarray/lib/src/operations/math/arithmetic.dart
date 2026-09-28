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

/// Computes the element-wise square root of the array.
///
/// Returns a new array with the results.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 4.0, 9.0], [3], DType.float64);
/// final b = sqrt(a);
/// print(b.toList()); // [1.0, 2.0, 3.0]
/// ```
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
  if (a.isDisposed || (out != null && out.isDisposed)) {
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sqrt.',
      );
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for expm1.',
      );
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for log1p.',
      );
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
NDArray<DTypeTag> logaddexp<T1 extends DTypeTag, T2 extends DTypeTag>(
  NDArray<T1> x1,
  NDArray<T2> x2, {
  NDArray<DTypeTag>? where,
  NDArray<DTypeTag>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logaddexp() on a disposed array.');
  }
  if (x1.dtype == DType.complex128 ||
      x1.dtype == DType.complex64 ||
      x2.dtype == DType.complex128 ||
      x2.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for logaddexp');
  }
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType targetDType =
      (x1.dtype == DType.float32 && x2.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for logaddexp.',
      );
    }
  }

  final maskHolder = prepareMask(where, shape);
  try {
    final NDArray<DTypeTag> result =
        out ??
        NDArray<DTypeTag>.create(shape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
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
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
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
            if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
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
            if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
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
        (a is bool ? (a ? 1.0 : 0.0) : (a as num).toDouble()),
        (b is bool ? (b ? 1.0 : 0.0) : (b as num).toDouble()),
      ),
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes $\log_2(2^{x_1} + 2^{x_2})$ element-wise.
NDArray<DTypeTag> logaddexp2<T1 extends DTypeTag, T2 extends DTypeTag>(
  NDArray<T1> x1,
  NDArray<T2> x2, {
  NDArray<DTypeTag>? where,
  NDArray<DTypeTag>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute logaddexp2() on a disposed array.');
  }
  if (x1.dtype == DType.complex128 ||
      x1.dtype == DType.complex64 ||
      x2.dtype == DType.complex128 ||
      x2.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for logaddexp2');
  }
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType targetDType =
      (x1.dtype == DType.float32 && x2.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for logaddexp2.',
      );
    }
  }

  final maskHolder = prepareMask(where, shape);
  try {
    final NDArray<DTypeTag> result =
        out ??
        NDArray<DTypeTag>.create(shape, targetDType, zeroInit: where != null);
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
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
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
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
            if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
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
            if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
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
        (a is bool ? (a ? 1.0 : 0.0) : (a as num).toDouble()),
        (b is bool ? (b ? 1.0 : 0.0) : (b as num).toDouble()),
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for rint.',
      );
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
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => (x as num).toDouble().roundToDouble(),
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for trunc.',
      );
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
      unaryOp<DTypeTag, DTypeTag>(
        result,
        a,
        a.shape,
        a.strides,
        result.strides,
        0,
        a.offsetElements,
        result.offsetElements,
        (x) => (x as num).toDouble().truncateToDouble(),
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
/// ```dart
/// final a = NDArray.fromList([2.0, 3.0], [2], DType.float64);
/// final b = square(a); // [4.0, 9.0]
/// ```
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for square.',
      );
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for reciprocal.',
      );
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
          return (1.0 / x);
        } else if (x is int) {
          if (x == 0) throw UnsupportedError('Integer division by zero');
          return (1 ~/ x);
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for positive.',
      );
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
          v_positive_int64(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_positive_int32(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          v_positive_int16(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          v_positive_uint8(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
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
            s_positive_int64(
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
            s_positive_int32(
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
            s_positive_int16(
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
            s_positive_uint8(
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
    throw ArgumentError(
      'Operands x1 and x2 must have the same dtype for power (was ${x1.dtype} and ${x2.dtype}). Perform an explicit cast first.',
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
          throw ArgumentError(
            'Integers to negative integer powers are not allowed.',
          );
        }
      } else {
        final minArr = min(x2Num);
        final minVal = minArr.scalar as num;
        minArr.dispose();
        if (minVal < 0) {
          throw ArgumentError(
            'Integers to negative integer powers are not allowed.',
          );
        }
      }
    } finally {
      if (!identical(x2Num, x2)) x2Num.dispose();
    }
  }

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, shape) ||
        out.dtype != dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for power.',
      );
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
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for negative.',
      );
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

/// Element-wise floor division with broadcasting and dtype upcasting support.
///
/// Corresponds to Dart's `~/` operator.
///
/// **Division by Zero:**
/// - **Integer arrays**: Division by zero is an error.
/// - **Floating-point arrays**: Returns `double.nan` silently without throwing exceptions.
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - for integer arrays, the divisor [x2] contains any `0` elements (throws [UnsupportedError]).
///
/// **Example:**
/// ```dart
/// final c = floorDivide(a, b);
/// ```
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
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support floor division');
  }

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for floorDivide.',
      );
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

    double doubleFloorDiv(double x, double y) {
      if (y == 0.0) return double.nan;
      return (x / y).floorToDouble();
    }

    int intFloorDiv(int x, int y) {
      if (y == 0) {
        throw UnsupportedError('Integer division by zero');
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
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support remainder');
  }

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for remainder.',
      );
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
      if (y == 0.0) return double.nan;
      final rem = x % y;
      if (rem != 0.0 && ((rem < 0.0) != (y < 0.0))) {
        return rem + y;
      }
      return rem;
    }

    int intMod(int x, int y) {
      if (y == 0) {
        throw UnsupportedError('Integer division by zero');
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
/// - The input arrays [x1] and [x2] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
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
/// **Division by Zero:**
/// - **Integer arrays**: Division by zero is an error.
/// - **Floating-point arrays**: Returns `double.nan` silently.
///
/// **Preconditions:**
/// - The input arrays [x1] and [x2] must not be disposed.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1] or [x2] is disposed (throws [StateError]).
/// - for integer arrays, the divisor [x2] contains any `0` elements (throws [UnsupportedError]).
(NDArray<T> div, NDArray<T> mod) divmod<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2,
) {
  return (floorDivide<T>(x1, x2), remainder<T>(x1, x2));
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
/// - The input arrays [x1] and [x2] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
/// - For integer arrays, the divisor [x2] must not contain any `0` elements.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
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
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support fmod');
  }

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for fmod.',
      );
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
/// - The input arrays [x1] and [x2] must not be disposed and must have integer dtypes.
/// - If [out] is provided, it must not be disposed and must have compatible shape and integer dtype.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
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
  if (!x1.dtype.isInteger || !x2.dtype.isInteger) {
    throw UnsupportedError('gcd only supports integer arrays.');
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for gcd.',
      );
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

    int calcGcd(int a, int b) {
      var u = a.abs();
      var v = b.abs();
      while (v != 0) {
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
      (x, y) => castValue(
        calcGcd((x as num).toInt(), (y as num).toInt()),
        targetDType,
      ),
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
/// ```dart
/// final a = NDArray.fromList([12, 15], [2], DType.int32);
/// final b = NDArray.fromList([18, 20], [2], DType.int32);
/// final c = lcm(a, b);
/// print(c.toList()); // [36, 60]
/// ```
NDArray<T> lcm<T extends DTypeTag>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed || x2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute lcm() on a disposed array.');
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
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for lcm.',
      );
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

    int calcGcd(int a, int b) {
      var u = a.abs();
      var v = b.abs();
      while (v != 0) {
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
        final a = (x as num).toInt();
        final b = (y as num).toInt();
        if (a == 0 || b == 0) return castValue(0, targetDType);
        return castValue((a.abs() ~/ calcGcd(a, b)) * b.abs(), targetDType);
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
/// - The input arrays [x1] and [x2] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
///
/// It is an error if:
/// - [x1], [x2], or [out] is disposed (throws [StateError]).
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
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support heaviside');
  }

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for heaviside.',
      );
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
  final targetDType = switch (a.dtype) {
    DType.complex64 => DType.float32,
    DType.complex128 => DType.float64,
    _ => a.dtype,
  };

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for abs.',
      );
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
        case DType.int32:
          v_abs_int32(
            a.typedPointer(),
            result.typedPointer(),
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
        case DType.uint64:
        case DType.uint32:
        case DType.uint16:
        case DType.boolean:
          break;
      }
    }
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
      case DType.int64:
      case DType.int32:
      case DType.int16:
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
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
      case DType.int8:
      case DType.uint64:
      case DType.uint32:
      case DType.uint16:
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
            return math.sqrt(z.real * z.real + z.imag * z.imag);
          },
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
          (x) => (x as num).abs().toInt(),
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
          (x) => (x as num).abs().toDouble(),
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
/// ```dart
/// final s = sign(a);
/// ```
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sign.',
      );
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
            if (z.real == 0 && z.imag == 0) return Complex(0, 0);
            final mag = math.sqrt(z.real * z.real + z.imag * z.imag);
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for ceil.',
      );
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for floor.',
      );
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

/// Computes element-wise round of the array.
///
/// It is an error if [a], [where], or [out] is disposed (throws [StateError]),
/// if [a] has a complex dtype (throws [UnsupportedError]),
/// or if [out] has an incompatible shape or dtype (throws [ArgumentError]).
NDArray<T> round<T extends DTypeTag>(
  NDArray<T> a, {
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
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for round.',
      );
    }
  }
  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<T> result =
        out ?? NDArray<T>.create(a.shape, a.dtype, zeroInit: where != null);
    switch (a.dtype) {
      case DType.float64:
        if (a.isContiguous && result.isContiguous) {
          v_round_double(
            a.typedPointer(),
            result.typedPointer(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        }
      case DType.float32:
        if (a.isContiguous && result.isContiguous) {
          v_round_float(
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
        (x) => castValue((x as num).roundToDouble(), a.dtype),
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

/// Signature for C function strided binary operations.
typedef StridedBinaryOp =
    void Function(
      ffi.Pointer<ffi.Void> a,
      ffi.Pointer<ffi.Int> stridesA,
      ffi.Pointer<ffi.Void> b,
      ffi.Pointer<ffi.Int> stridesB,
      ffi.Pointer<ffi.Void> result,
      ffi.Pointer<ffi.Int> stridesResult,
      ffi.Pointer<ffi.Int> shape,
      int rank,
    );

/// Element-wise addition of two arrays.
///
/// Returns a new array with the promoted data type.
NDArray<T> add<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute add() on a disposed array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<T> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<T>.create(
          commonShape,
          targetDType as DType<T>,
          zeroInit: where != null,
        );
    // Specialized paths for Float64 (as in original extensions.dart)
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int> cShape;
    late final ffi.Pointer<ffi.Int> cStridesA;
    late final ffi.Pointer<ffi.Int> cStridesB;
    late final ffi.Pointer<ffi.Int> cStridesRes;
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
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute subtract() on a disposed array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<T> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<T>.create(
          commonShape,
          targetDType as DType<T>,
          zeroInit: where != null,
        );
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int> cShape;
    late final ffi.Pointer<ffi.Int> cStridesA;
    late final ffi.Pointer<ffi.Int> cStridesB;
    late final ffi.Pointer<ffi.Int> cStridesRes;
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
/// **Overflow behavior:**
/// - **Integer arrays** (`int32`, `int64`, etc.) overflow silently wrapping around via standard two's complement.
/// - **Floating-point arrays** (`float32`, `float64`) overflow silently to `double.infinity` or `double.negativeInfinity` per IEEE 754.
NDArray<T> multiply<T extends DTypeTag>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<DTypeTag>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute multiply() on a disposed array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<T> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<T>.create(
          commonShape,
          targetDType as DType<T>,
          zeroInit: where != null,
        );
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int> cShape;
    late final ffi.Pointer<ffi.Int> cStridesA;
    late final ffi.Pointer<ffi.Int> cStridesB;
    late final ffi.Pointer<ffi.Int> cStridesRes;
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
/// Always upcasts integer operands to [DType.float64] and performs floating-point division.
///
/// **Division by Zero:**
/// Division by zero is handled silently under IEEE 754 floating-point rules:
/// - Dividing a non-zero value by zero results in `double.infinity` or `double.negativeInfinity`.
/// - Dividing zero by zero results in `double.nan`.
///
/// **Preconditions:**
/// - The input arrays [a] and [b] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have compatible shape and dtype.
///
/// It is an error if:
/// - [a], [b], or [out] is disposed (throws [StateError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
NDArray<R> divide<Ta extends DTypeTag, Tb extends DTypeTag, R extends DTypeTag>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<DTypeTag>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute divide() on a disposed array.');
  }
  var targetDType = resolveDType(a.dtype, b.dtype);
  if (targetDType.isInteger) {
    targetDType = DType.float64;
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, commonShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }
  final maskHolder = prepareMask(where, commonShape);
  late final NDArray<R> result;

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    result =
        out ??
        NDArray<R>.create(
          commonShape,
          targetDType as DType<R>,
          zeroInit: where != null,
        );
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        listEquals(a.shape, b.shape);

    late final ffi.Pointer<ffi.Int> cShape;
    late final ffi.Pointer<ffi.Int> cStridesA;
    late final ffi.Pointer<ffi.Int> cStridesB;
    late final ffi.Pointer<ffi.Int> cStridesRes;
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
          final cpxRes = divide<Complex128, Complex128, Complex128>(
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
          final doubleRes = divide<Float64, Float64, Float64>(
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

/// Element-wise addition of [a] and [b] computed into the specified target [dtype].
///
/// Accepts two arrays of potentially different data types ([Ta] and [Tb]) and
/// returns an [NDArray<R>] whose static type [R] is inferred from [dtype].
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
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute addAs() on a disposed array.');
  }
  if (resolveDType(a.dtype, b.dtype) == dtype) {
    return add<DTypeTag>(a, b, where: where, out: out) as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCanPromoteToTarget =
        a.dtype == dtype || resolveDType(a.dtype, dtype) == dtype;
    final bCanPromoteToTarget =
        b.dtype == dtype || resolveDType(dtype, b.dtype) == dtype;
    final NDArray aIn = aCanPromoteToTarget && !bCanPromoteToTarget
        ? a
        : (a.dtype == dtype ? a : castNDArray<R>(a, dtype));
    final NDArray bIn = bCanPromoteToTarget && !aCanPromoteToTarget
        ? b
        : (b.dtype == dtype ? b : castNDArray<R>(b, dtype));
    final NDArray aFinal = resolveDType(aIn.dtype, bIn.dtype) == dtype
        ? aIn
        : (aIn.dtype == dtype ? aIn : castNDArray<R>(aIn, dtype));
    final NDArray bFinal = resolveDType(aFinal.dtype, bIn.dtype) == dtype
        ? bIn
        : (bIn.dtype == dtype ? bIn : castNDArray<R>(bIn, dtype));
    final res =
        add<DTypeTag>(aFinal, bFinal, where: where, out: out) as NDArray<R>;
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise subtraction of [a] and [b] computed into the specified target [dtype].
///
/// Accepts two arrays of potentially different data types ([Ta] and [Tb]) and
/// returns an [NDArray<R>] whose static type [R] is inferred from [dtype].
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
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute subtractAs() on a disposed array.');
  }
  if (resolveDType(a.dtype, b.dtype) == dtype) {
    return subtract<DTypeTag>(a, b, where: where, out: out) as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCanPromoteToTarget =
        a.dtype == dtype || resolveDType(a.dtype, dtype) == dtype;
    final bCanPromoteToTarget =
        b.dtype == dtype || resolveDType(dtype, b.dtype) == dtype;
    final NDArray aIn = aCanPromoteToTarget && !bCanPromoteToTarget
        ? a
        : (a.dtype == dtype ? a : castNDArray<R>(a, dtype));
    final NDArray bIn = bCanPromoteToTarget && !aCanPromoteToTarget
        ? b
        : (b.dtype == dtype ? b : castNDArray<R>(b, dtype));
    final NDArray aFinal = resolveDType(aIn.dtype, bIn.dtype) == dtype
        ? aIn
        : (aIn.dtype == dtype ? aIn : castNDArray<R>(aIn, dtype));
    final NDArray bFinal = resolveDType(aFinal.dtype, bIn.dtype) == dtype
        ? bIn
        : (bIn.dtype == dtype ? bIn : castNDArray<R>(bIn, dtype));
    final res =
        subtract<DTypeTag>(aFinal, bFinal, where: where, out: out)
            as NDArray<R>;
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise multiplication of [a] and [b] computed into the specified target [dtype].
///
/// Accepts two arrays of potentially different data types ([Ta] and [Tb]) and
/// returns an [NDArray<R>] whose static type [R] is inferred from [dtype].
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
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute multiplyAs() on a disposed array.');
  }
  if (resolveDType(a.dtype, b.dtype) == dtype) {
    return multiply<DTypeTag>(a, b, where: where, out: out) as NDArray<R>;
  }
  return NDArray.scope(() {
    final aCanPromoteToTarget =
        a.dtype == dtype || resolveDType(a.dtype, dtype) == dtype;
    final bCanPromoteToTarget =
        b.dtype == dtype || resolveDType(dtype, b.dtype) == dtype;
    final NDArray aIn = aCanPromoteToTarget && !bCanPromoteToTarget
        ? a
        : (a.dtype == dtype ? a : castNDArray<R>(a, dtype));
    final NDArray bIn = bCanPromoteToTarget && !aCanPromoteToTarget
        ? b
        : (b.dtype == dtype ? b : castNDArray<R>(b, dtype));
    final NDArray aFinal = resolveDType(aIn.dtype, bIn.dtype) == dtype
        ? aIn
        : (aIn.dtype == dtype ? aIn : castNDArray<R>(aIn, dtype));
    final NDArray bFinal = resolveDType(aFinal.dtype, bIn.dtype) == dtype
        ? bIn
        : (bIn.dtype == dtype ? bIn : castNDArray<R>(bIn, dtype));
    final res =
        multiply<DTypeTag>(aFinal, bFinal, where: where, out: out)
            as NDArray<R>;
    return out ?? res.detachToParentScope();
  });
}

/// Element-wise true division of [a] by [b] computed into the specified target [dtype].
///
/// Accepts two arrays of potentially different data types ([Ta] and [Tb])
/// and returns an [NDArray<R>] whose static type [R] is inferred from [dtype].
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
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute divideAs() on a disposed array.');
  }
  var resolved = resolveDType(a.dtype, b.dtype);
  if (resolved.isInteger) {
    resolved = DType.float64;
  }
  if (resolved == dtype) {
    return divide<Ta, Tb, R>(a, b, where: where, out: out);
  }
  return NDArray.scope(() {
    if (!dtype.isInteger && dtype != DType.boolean) {
      final aCanPromote =
          a.dtype == dtype ||
          (!a.dtype.isInteger && resolveDType(a.dtype, dtype) == dtype);
      final bCanPromote =
          b.dtype == dtype ||
          (!b.dtype.isInteger && resolveDType(dtype, b.dtype) == dtype);
      final NDArray aIn = aCanPromote ? a : castNDArray<R>(a, dtype);
      final NDArray bIn = bCanPromote ? b : castNDArray<R>(b, dtype);
      var inResolved = resolveDType(aIn.dtype, bIn.dtype);
      if (inResolved.isInteger) inResolved = DType.float64;
      final NDArray aFinal = inResolved == dtype
          ? aIn
          : (aIn.dtype == dtype ? aIn : castNDArray<R>(aIn, dtype));
      final NDArray bFinal = inResolved == dtype
          ? bIn
          : (bIn.dtype == dtype ? bIn : castNDArray<R>(bIn, dtype));
      final res = divide<DTypeTag, DTypeTag, R>(
        aFinal,
        bFinal,
        where: where,
        out: out,
      );
      return out ?? res.detachToParentScope();
    }
    final divF64 = divide<Ta, Tb, Float64>(a, b, where: where);
    final casted = castNDArray<R>(divF64, dtype);
    if (out != null) {
      casted.copy(out: out);
      return out;
    }
    return casted.detachToParentScope();
  });
}
