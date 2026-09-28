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
import 'dart:ffi' as ffi;

import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';

import '../broadcasting.dart';
import '../helpers.dart';
import '../stats.dart';

/// Computes the element-wise square root of the array.
///
/// Returns a new array with the results.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 4.0, 9.0], [3], DType.float64);
/// final b = sqrt(a);
/// print(b.data); // [1.0, 2.0, 3.0]
/// ```
///
/// **Edge cases:**
/// - Negative values will result in [double.nan].
NDArray<R> sqrt<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute sqrt() on a disposed array.');
  }
  final DType<R> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype as DType<R>;
  } else {
    targetDType =
        (a.dtype == DType.float32 ? DType.float32 : DType.float64) as DType<R>;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sqrt.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_sqrt_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_sqrt_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_sqrt_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_sqrt_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    }

    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
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
        if (a.dtype == DType.complex128) {
          s_sqrt_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        } else {
          s_sqrt_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
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

    if (result.isContiguous) {
      final offset = temp.offsetElements;
      final resOffset = result.offsetElements;
      for (var i = 0; i < temp.size; i++) {
        if (maskHolder.pointer == ffi.nullptr || maskHolder.pointer[i] != 0) {
          result.setCellFlat(
            resOffset + i,
            castValue(
              math.sqrt((temp.getCellFlat(offset + i) as num).toDouble()),
              result.dtype,
            ),
          );
        }
      }
    } else {
      final tempOut = result.copy();
      final offset = temp.offsetElements;
      for (var i = 0; i < temp.size; i++) {
        if (maskHolder.pointer == ffi.nullptr || maskHolder.pointer[i] != 0) {
          tempOut.setCellFlat(
            i,
            castValue(
              math.sqrt((temp.getCellFlat(offset + i) as num).toDouble()),
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
NDArray<R> expm1<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute expm1() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for expm1.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_expm1_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_expm1_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_expm1_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_expm1_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
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
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_expm1_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex128:
              s_expm1_complex128(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex64:
              s_expm1_complex64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            default:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
      unaryOp<dynamic, dynamic>(
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
      unaryOp<dynamic, dynamic>(
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
      unaryOp<dynamic, dynamic>(
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
NDArray<R> log1p<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute log1p() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for log1p.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_log1p_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_log1p_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_log1p_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_log1p_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
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
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_log1p_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex128:
              s_log1p_complex128(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex64:
              s_log1p_complex64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            default:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
      unaryOp<dynamic, dynamic>(
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
      unaryOp<dynamic, dynamic>(
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
      unaryOp<dynamic, dynamic>(
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
NDArray<double> logaddexp<T1, T2>(
  NDArray<T1> x1,
  NDArray<T2> x2, {
  NDArray<dynamic>? where,
  NDArray<double>? out,
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
  final DType<double> targetDType =
      (x1.dtype == DType.float32 && x2.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  final NDArray<double> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for logaddexp.',
      );
    }
    result = out;
  } else {
    result = NDArray<double>.create(shape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch ((x1.dtype, x2.dtype)) {
        case (DType.float64, DType.float64):
          v_logaddexp_double(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case (DType.float32, DType.float32):
          v_logaddexp_float(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    }

    final stridesX1 = broadcastResult.stridesA;
    final stridesX2 = broadcastResult.stridesB;

    if (shape.length <= 8) {
      final marker = ScratchArena.marker;
      final cShape = ScratchArena.copyInts(shape);
      final cStridesX1 = ScratchArena.copyInts(stridesX1);
      final cStridesX2 = ScratchArena.copyInts(stridesX2);
      final cStridesRes = ScratchArena.copyInts(result.strides);
      try {
        switch ((targetDType, x1.dtype, x2.dtype)) {
          case (DType.float64, DType.float64, DType.float64):
            s_logaddexp_double(
              x1.pointer.cast(),
              cStridesX1,
              x2.pointer.cast(),
              cStridesX2,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              shape.length,
              maskHolder.pointer,
            );
            return result;
          case (DType.float32, DType.float32, DType.float32):
            s_logaddexp_float(
              x1.pointer.cast(),
              cStridesX1,
              x2.pointer.cast(),
              cStridesX2,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              shape.length,
              maskHolder.pointer,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    elementWiseOp<dynamic, dynamic, double>(
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
NDArray<double> logaddexp2<T1, T2>(
  NDArray<T1> x1,
  NDArray<T2> x2, {
  NDArray<dynamic>? where,
  NDArray<double>? out,
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
  final DType<double> targetDType =
      (x1.dtype == DType.float32 && x2.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  final NDArray<double> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for logaddexp2.',
      );
    }
    result = out;
  } else {
    result = NDArray<double>.create(shape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch ((x1.dtype, x2.dtype)) {
        case (DType.float64, DType.float64):
          v_logaddexp2_double(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case (DType.float32, DType.float32):
          v_logaddexp2_float(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    }

    final stridesX1 = broadcastResult.stridesA;
    final stridesX2 = broadcastResult.stridesB;

    if (shape.length <= 8) {
      final marker = ScratchArena.marker;
      final cShape = ScratchArena.copyInts(shape);
      final cStridesX1 = ScratchArena.copyInts(stridesX1);
      final cStridesX2 = ScratchArena.copyInts(stridesX2);
      final cStridesRes = ScratchArena.copyInts(result.strides);
      try {
        switch ((targetDType, x1.dtype, x2.dtype)) {
          case (DType.float64, DType.float64, DType.float64):
            s_logaddexp2_double(
              x1.pointer.cast(),
              cStridesX1,
              x2.pointer.cast(),
              cStridesX2,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              shape.length,
              maskHolder.pointer,
            );
            return result;
          case (DType.float32, DType.float32, DType.float32):
            s_logaddexp2_float(
              x1.pointer.cast(),
              cStridesX1,
              x2.pointer.cast(),
              cStridesX2,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              shape.length,
              maskHolder.pointer,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    elementWiseOp<dynamic, dynamic, double>(
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
NDArray<R> rint<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute rint() on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for rint');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for rint.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_rint_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_rint_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
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
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_rint_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            default:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if (a.dtype.isInteger) {
      unaryOp<dynamic, dynamic>(
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
      unaryOp<dynamic, dynamic>(
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
NDArray<R> trunc<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute trunc() on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for trunc');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for trunc.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_trunc_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_trunc_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
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
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_trunc_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            default:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    if (a.dtype.isInteger) {
      unaryOp<dynamic, dynamic>(
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
      unaryOp<dynamic, dynamic>(
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
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Rounds elements of the array to the nearest integer towards zero.
///
/// Synonym for [trunc].
NDArray<R> fix<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
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
NDArray<T> square<T>(NDArray<T> a, {NDArray<dynamic>? where, NDArray<T>? out}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute square() on a disposed array.');
  }
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for square.',
      );
    }
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_square_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_square_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_square_int64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_square_int32(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_square_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_square_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.boolean:
          final maskPtr = maskHolder.pointer;
          for (var i = 0; i < a.size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              result.setCellFlat(i, a.getCellFlat(i));
            }
          }
          return result;
        case DType.uint8:
        case DType.int16:
          final maskPtr = maskHolder.pointer;
          for (var i = 0; i < a.size; i++) {
            if (maskPtr == ffi.nullptr || maskPtr[i] != 0) {
              final val = a.getCellFlat(i) as num;
              result.setCellFlat(i, (val * val) as T);
            }
          }
          return result;
        default:
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
            s_square_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.float32:
            s_square_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int64:
            s_square_int64(
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
            s_square_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex128:
            s_square_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex64:
            s_square_complex64(
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
            unaryOp<bool, bool>(
              result as NDArray<bool>,
              a as NDArray<bool>,
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
          case DType.uint8:
          case DType.int16:
            unaryOp<num, num>(
              result as NDArray<num>,
              a as NDArray<num>,
              a.shape,
              a.strides,
              result.strides,
              0,
              a.offsetElements,
              result.offsetElements,
              (x) => x * x,
              maskHolder.pointer,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }
    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the element-wise reciprocal ($1/x$) of the array.
NDArray<T> reciprocal<T extends Object>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute reciprocal() on a disposed array.');
  }
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for reciprocal.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    var isInt = false;
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_reciprocal_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_reciprocal_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_reciprocal_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_reciprocal_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_reciprocal_int64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        case DType.int32:
          v_reciprocal_int32(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        case DType.int16:
          v_reciprocal_int16(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        case DType.uint8:
          v_reciprocal_uint8(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          isInt = true;
          break;
        default:
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
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.float32:
              s_reciprocal_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex128:
              s_reciprocal_complex128(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex64:
              s_reciprocal_complex64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.int64:
              s_reciprocal_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            case DType.int32:
              s_reciprocal_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            case DType.int16:
              s_reciprocal_int16(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            case DType.uint8:
              s_reciprocal_uint8(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              isInt = true;
              break;
            default:
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
          return (Complex(1.0, 0.0) / x) as T;
        } else if (x is double) {
          return (1.0 / x) as T;
        } else if (x is int) {
          if (x == 0) throw UnsupportedError('Integer division by zero');
          return (1 ~/ x) as T;
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
NDArray<T> positive<T>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
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

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for positive.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_positive_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_positive_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_positive_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_positive_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_positive_int64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_positive_int32(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          v_positive_int16(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          v_positive_uint8(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
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
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.float32:
            s_positive_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex128:
            s_positive_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.complex64:
            s_positive_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int64:
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
          default:
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
NDArray<T> power<T>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
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

  if (dtype.isInteger) {
    final NDArray<num> x2Num = (x2 is NDArray<num>)
        ? (x2 as NDArray<num>)
        : castNDArray<num>(x2, x2.dtype as DType<num>);
    if (x2Num.rank == 0) {
      if (x2Num.scalar < 0) {
        throw ArgumentError(
          'Integers to negative integer powers are not allowed.',
        );
      }
    } else {
      if (min(x2Num).scalar < 0) {
        throw ArgumentError(
          'Integers to negative integer powers are not allowed.',
        );
      }
    }
  }

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for power.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(shape, dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    final isContig =
        x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous;

    if (isContig) {
      switch (dtype) {
        case DType.float64:
          v_pow_double(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_pow_float(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_pow_int64(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_pow_int32(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          v_pow_int16(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          v_pow_uint8(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_pow_complex128(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_pow_complex64(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            x1.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
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
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          s_pow_float(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          s_pow_int64(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          s_pow_int32(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          s_pow_int16(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          s_pow_uint8(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          s_pow_complex128(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          s_pow_complex64(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}

NDArray<T> negative<T>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute negative() on a disposed array.');
  }
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for negative.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(a.shape, a.dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        unaryOp<dynamic, dynamic>(
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
        unaryOp<dynamic, dynamic>(
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
        unaryOp<dynamic, dynamic>(
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
/// final c = floor_divide(a, b);
/// ```
NDArray<T> floor_divide<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (x1.isDisposed ||
      x2.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute floor_divide() on a disposed array.');
  }
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = resolveDType(x1.dtype, x2.dtype) as DType<T>;
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support floor division');
  }

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for floor_divide.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(commonShape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_floordiv_double(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_floordiv_float(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_floordiv_int64(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
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
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        default:
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
          default:
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
      elementWiseOp<dynamic, dynamic, dynamic>(
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
      elementWiseOp<dynamic, dynamic, dynamic>(
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
NDArray<T> remainder<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
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

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for remainder.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(commonShape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_remainder_double(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_remainder_float(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_remainder_int64(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
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
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        default:
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesA,
                x2.pointer.cast(),
                cStridesB,
                result.pointer.cast(),
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
          default:
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
      elementWiseOp<dynamic, dynamic, dynamic>(
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
      elementWiseOp<dynamic, dynamic, dynamic>(
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
NDArray<T> mod<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
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
(NDArray<T> div, NDArray<T> mod) divmod<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2,
) {
  return (floor_divide<T>(x1, x2), remainder<T>(x1, x2));
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
NDArray<T> fmod<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
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

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for fmod.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(commonShape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_fmod_double(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_fmod_float(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_fmod_int64(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
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
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        default:
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (targetDType.isFloating) {
      elementWiseOp<dynamic, dynamic, dynamic>(
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
      elementWiseOp<dynamic, dynamic, dynamic>(
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
          return castValue(ix % iy, targetDType);
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
NDArray<T> gcd<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
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
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for gcd.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(commonShape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_gcd_int64(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_gcd_int32(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        default:
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          default:
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

    elementWiseOp<dynamic, dynamic, dynamic>(
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
/// print(c.data); // [36, 60]
/// ```
NDArray<R> lcm<Ta, Tb, R>(
  NDArray<Ta> x1,
  NDArray<Tb> x2, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (x1.isDisposed || x2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute lcm() on a disposed array.');
  }
  if (!x1.dtype.isInteger || !x2.dtype.isInteger) {
    throw UnsupportedError('lcm only supports integer arrays.');
  }
  final DType<dynamic> targetDType = resolveDType(x1.dtype, x2.dtype);
  final broadcastResult = broadcast(x1, x2);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for lcm.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }
  final maskHolder = prepareMask(where, result.shape);

  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous &&
        listEquals(x1.shape, x2.shape)) {
      switch (targetDType) {
        case DType.int64:
          v_lcm_int64(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            result.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_lcm_int32(
            x1.pointer.cast(),
            x2.pointer.cast(),
            result.pointer.cast(),
            result.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    } else {
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
            s_lcm_int64(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          case DType.int32:
            s_lcm_int32(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
            return result;
          default:
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

    elementWiseOp<dynamic, dynamic, dynamic>(
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
NDArray<T> heaviside<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<dynamic>? where,
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

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for heaviside.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(commonShape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    if (x1.isContiguous &&
        x2.isContiguous &&
        listEquals(x1.shape, x2.shape) &&
        result.isContiguous) {
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            v_heaviside_double(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            v_heaviside_float(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            v_heaviside_int64(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            v_heaviside_int32(
              x1.pointer.cast(),
              x2.pointer.cast(),
              result.pointer.cast(),
              x1.size,
              maskHolder.pointer,
            );
            return result;
          }
        default:
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
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
                x1.pointer.cast(),
                cStridesX1,
                x2.pointer.cast(),
                cStridesX2,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            }
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    elementWiseOp<dynamic, dynamic, dynamic>(
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
          final dx = (x as num).toDouble();
          if (dx.isNaN) return castValue(dx, targetDType);
          if (dx < 0.0) return castValue(0.0, targetDType);
          if (dx > 0.0) return castValue(1.0, targetDType);
          return castValue((y as num).toDouble(), targetDType);
        } else {
          final ix = (x as num).toInt();
          if (ix < 0) return castValue(0, targetDType);
          if (ix > 0) return castValue(1, targetDType);
          return castValue((y as num).toInt(), targetDType);
        }
      },
      maskHolder.pointer,
    );
    return result;
  } finally {
    maskHolder.dispose();
  }
}

NDArray<R> abs<T, R>(NDArray<T> a, {NDArray<dynamic>? where, NDArray<R>? out}) {
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

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for abs.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType as DType<R>);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_abs_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_abs_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          v_abs_complex128(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          v_abs_complex64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int64:
          v_abs_int64(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int32:
          v_abs_int32(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.int16:
          v_abs_int16(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.uint8:
          v_abs_uint8(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    } else if (a.dtype == DType.complex128 ||
        a.dtype == DType.complex64 ||
        a.dtype == DType.int64 ||
        a.dtype == DType.int32 ||
        a.dtype == DType.int16 ||
        a.dtype == DType.uint8) {
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
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.complex64:
              s_abs_complex64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            case DType.int64:
              s_abs_int64(
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
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cStridesRes,
                cShape,
                rank,
                maskHolder.pointer,
              );
              return result;
            default:
              break;
          }
        } finally {
          ScratchArena.reset(marker);
        }
      }
    }

    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        unaryOp<Complex, double>(
          result as NDArray<double>,
          a as NDArray<Complex>,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (c) => math.sqrt(c.real * c.real + c.imag * c.imag),
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
        unaryOp<dynamic, dynamic>(
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
        unaryOp<dynamic, dynamic>(
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
      default:
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
NDArray<T> sign<T extends Object>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<T>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute sign() on a disposed array.');
  }
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sign.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(a.shape, a.dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    switch (a.dtype) {
      case DType.complex128:
      case DType.complex64:
        unaryOp<Complex, Complex>(
          result as NDArray<Complex>,
          a as NDArray<Complex>,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (c) {
            if (c.real == 0 && c.imag == 0) return Complex(0, 0);
            final mag = math.sqrt(c.real * c.real + c.imag * c.imag);
            return Complex(c.real / mag, c.imag / mag);
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
        unaryOp<dynamic, dynamic>(
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
        unaryOp<dynamic, dynamic>(
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
      case DType.boolean:
        a.copy(out: result);
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
NDArray<T> ceil<T extends Object>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
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
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for ceil.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(a.shape, a.dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    if (a.isContiguous && result.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_ceil_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          v_ceil_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    }

    if (a.dtype.isInteger || a.dtype == DType.boolean) {
      a.copy(out: result);
    } else if (a.dtype.isFloating) {
      unaryOp<dynamic, dynamic>(
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
NDArray<T> floor<T extends Object>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
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
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for floor.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(a.shape, a.dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    switch (a.dtype) {
      case DType.float64:
        if (a.isContiguous && result.isContiguous) {
          v_floor_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        }
      case DType.float32:
        if (a.isContiguous && result.isContiguous) {
          v_floor_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        }
      default:
        break;
    }

    if (a.dtype.isInteger || a.dtype == DType.boolean) {
      a.copy(out: result);
    } else if (a.dtype.isFloating) {
      unaryOp<dynamic, dynamic>(
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
NDArray<T> round<T extends Object>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
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
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for round.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(a.shape, a.dtype);
  }
  final maskHolder = prepareMask(where, result.shape);
  try {
    switch (a.dtype) {
      case DType.float64:
        if (a.isContiguous && result.isContiguous) {
          v_round_double(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        }
      case DType.float32:
        if (a.isContiguous && result.isContiguous) {
          v_round_float(
            a.pointer.cast(),
            result.pointer.cast(),
            a.size,
            maskHolder.pointer,
          );
          return result;
        }
      default:
        break;
    }

    if (a.dtype.isInteger || a.dtype == DType.boolean) {
      a.copy(out: result);
    } else if (a.dtype.isFloating) {
      unaryOp<dynamic, dynamic>(
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
typedef StridedBinaryOp = void Function(
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
NDArray<R> add<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute add() on a disposed array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }
  final maskHolder = prepareMask(where, result.shape);

  // Specialized paths for Float64 (as in original extensions.dart)
  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesB = cBuffer + (ndim * 2);
    final cStridesRes = cBuffer + (ndim * 3);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }
    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_add_double_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_add_double_double_double(
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
      case (DType.float64, DType.float32) when isContig:
        v_add_double_float_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_add_double_float_double(
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
      case (DType.float64, DType.int64) when isContig:
        v_add_double_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_add_double_int64_double(
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
      case (DType.float64, DType.int32) when isContig:
        v_add_double_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_add_double_int32_double(
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
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_add_double_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_add_double_uint8_double(
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
      case (DType.float64, DType.int16) when isContig:
        v_add_double_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_add_double_int16_double(
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
      case (DType.float64, DType.complex128) when isContig:
        v_add_double_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_add_double_cpx_cpx(
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
      case (DType.float64, DType.complex64) when isContig:
        v_add_double_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_add_double_cpx64_cpx(
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
      case (DType.float32, DType.float64) when isContig:
        v_add_double_float_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_add_double_float_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32) when isContig:
        v_add_float_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_add_float_float_float(
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
      case (DType.float32, DType.int64) when isContig:
        v_add_float_int64_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64):
        s_add_float_int64_float(
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
      case (DType.float32, DType.int32) when isContig:
        v_add_float_int32_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32):
        s_add_float_int32_float(
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
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_add_float_uint8_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_add_float_uint8_float(
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
      case (DType.float32, DType.int16) when isContig:
        v_add_float_int16_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_add_float_int16_float(
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
      case (DType.float32, DType.complex128) when isContig:
        v_add_float_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_add_float_cpx_cpx(
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
      case (DType.float32, DType.complex64) when isContig:
        v_add_float_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_add_float_cpx64_cpx64(
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
      case (DType.int64, DType.float64) when isContig:
        v_add_double_int64_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_add_double_int64_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32) when isContig:
        v_add_float_int64_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32):
        s_add_float_int64_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64) when isContig:
        v_add_int64_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
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
      case (DType.int64, DType.int32) when isContig:
        v_add_int64_int32_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_add_int64_int32_int64(
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
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_add_int64_uint8_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_add_int64_uint8_int64(
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
      case (DType.int64, DType.int16) when isContig:
        v_add_int64_int16_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_add_int64_int16_int64(
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
      case (DType.int64, DType.complex128) when isContig:
        v_add_int64_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_add_int64_cpx_cpx(
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
      case (DType.int64, DType.complex64) when isContig:
        v_add_int64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64):
        s_add_int64_cpx64_cpx64(
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
      case (DType.int32, DType.float64) when isContig:
        v_add_double_int32_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_add_double_int32_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32) when isContig:
        v_add_float_int32_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32):
        s_add_float_int32_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64) when isContig:
        v_add_int64_int32_int64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_add_int64_int32_int64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32) when isContig:
        v_add_int32_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
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
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_add_int32_uint8_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_add_int32_uint8_int32(
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
      case (DType.int32, DType.int16) when isContig:
        v_add_int32_int16_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_add_int32_int16_int32(
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
      case (DType.int32, DType.complex128) when isContig:
        v_add_int32_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_add_int32_cpx_cpx(
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
      case (DType.int32, DType.complex64) when isContig:
        v_add_int32_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_add_int32_cpx64_cpx64(
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
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_add_double_uint8_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_add_double_uint8_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_add_float_uint8_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_add_float_uint8_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_add_int64_uint8_int64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_add_int64_uint8_int64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_add_int32_uint8_int32(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_add_int32_uint8_int32(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
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
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
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
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_add_uint8_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_add_uint8_int16_int16(
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
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_add_uint8_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_add_uint8_cpx_cpx(
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
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_add_uint8_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_add_uint8_cpx64_cpx64(
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
      case (DType.int16, DType.float64) when isContig:
        v_add_double_int16_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_add_double_int16_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32) when isContig:
        v_add_float_int16_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_add_float_int16_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64) when isContig:
        v_add_int64_int16_int64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_add_int64_int16_int64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32) when isContig:
        v_add_int32_int16_int32(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_add_int32_int16_int32(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_add_uint8_int16_int16(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_add_uint8_int16_int16(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16) when isContig:
        v_add_int16_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
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
      case (DType.int16, DType.complex128) when isContig:
        v_add_int16_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_add_int16_cpx_cpx(
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
      case (DType.int16, DType.complex64) when isContig:
        v_add_int16_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_add_int16_cpx64_cpx64(
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
      case (DType.complex128, DType.float64) when isContig:
        v_add_double_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_add_double_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32) when isContig:
        v_add_float_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_add_float_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64) when isContig:
        v_add_int64_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_add_int64_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32) when isContig:
        v_add_int32_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_add_int32_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_add_uint8_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_add_uint8_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16) when isContig:
        v_add_int16_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_add_int16_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128) when isContig:
        v_add_cpx_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_add_cpx_cpx_cpx(
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
      case (DType.complex128, DType.complex64) when isContig:
        v_add_cpx_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_add_cpx_cpx64_cpx(
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
      case (DType.complex64, DType.float64) when isContig:
        v_add_double_cpx64_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_add_double_cpx64_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32) when isContig:
        v_add_float_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_add_float_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64) when isContig:
        v_add_int64_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64):
        s_add_int64_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32) when isContig:
        v_add_int32_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_add_int32_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_add_uint8_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_add_uint8_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16) when isContig:
        v_add_int16_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_add_int16_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128) when isContig:
        v_add_cpx_cpx64_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_add_cpx_cpx64_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64) when isContig:
        v_add_cpx64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_add_cpx64_cpx64_cpx64(
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
        break;
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
  if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
    final cpxA = castNDArray(a, DType.complex128);
    final cpxB = castNDArray(b, DType.complex128);
    final cpxRes = add<Complex, Complex, Complex>(cpxA, cpxB, where: where);
    final casted = castNDArray(cpxRes, result.dtype);
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(cpxA, a)) cpxA.dispose();
    if (!identical(cpxB, b)) cpxB.dispose();
    cpxRes.dispose();
    casted.dispose();
    return result;
  } else {
    final doubleA = castNDArray(a, DType.float64);
    final doubleB = castNDArray(b, DType.float64);
    final doubleRes = add<Float64, Float64, Float64>(
      doubleA,
      doubleB,
      where: where,
    );
    final casted = castNDArray(doubleRes, result.dtype);
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(doubleA, a)) doubleA.dispose();
    if (!identical(doubleB, b)) doubleB.dispose();
    doubleRes.dispose();
    casted.dispose();
    return result;
  }
}

/// Element-wise subtraction of two arrays.
NDArray<R> subtract<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute subtract() on a disposed array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }
  final maskHolder = prepareMask(where, result.shape);

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesB = cBuffer + (ndim * 2);
    final cStridesRes = cBuffer + (ndim * 3);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }
    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_sub_double_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_sub_double_double_double(
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
      case (DType.float64, DType.float32) when isContig:
        v_sub_double_float_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_sub_double_float_double(
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
      case (DType.float64, DType.int64) when isContig:
        v_sub_double_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_sub_double_int64_double(
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
      case (DType.float64, DType.int32) when isContig:
        v_sub_double_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_sub_double_int32_double(
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
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_sub_double_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_sub_double_uint8_double(
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
      case (DType.float64, DType.int16) when isContig:
        v_sub_double_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_sub_double_int16_double(
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
      case (DType.float64, DType.complex128) when isContig:
        v_sub_double_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_sub_double_cpx_cpx(
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
      case (DType.float64, DType.complex64) when isContig:
        v_sub_double_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_sub_double_cpx64_cpx(
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
      case (DType.float32, DType.float64) when isContig:
        v_sub_float_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_sub_float_double_double(
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
      case (DType.float32, DType.float32) when isContig:
        v_sub_float_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_sub_float_float_float(
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
      case (DType.float32, DType.int64) when isContig:
        v_sub_float_int64_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64):
        s_sub_float_int64_float(
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
      case (DType.float32, DType.int32) when isContig:
        v_sub_float_int32_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32):
        s_sub_float_int32_float(
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
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_sub_float_uint8_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_sub_float_uint8_float(
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
      case (DType.float32, DType.int16) when isContig:
        v_sub_float_int16_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_sub_float_int16_float(
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
      case (DType.float32, DType.complex128) when isContig:
        v_sub_float_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_sub_float_cpx_cpx(
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
      case (DType.float32, DType.complex64) when isContig:
        v_sub_float_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_sub_float_cpx64_cpx64(
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
      case (DType.int64, DType.float64) when isContig:
        v_sub_int64_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_sub_int64_double_double(
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
      case (DType.int64, DType.float32) when isContig:
        v_sub_int64_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32):
        s_sub_int64_float_float(
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
      case (DType.int64, DType.int64) when isContig:
        v_sub_int64_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
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
      case (DType.int64, DType.int32) when isContig:
        v_sub_int64_int32_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_sub_int64_int32_int64(
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
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_sub_int64_uint8_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_sub_int64_uint8_int64(
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
      case (DType.int64, DType.int16) when isContig:
        v_sub_int64_int16_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_sub_int64_int16_int64(
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
      case (DType.int64, DType.complex128) when isContig:
        v_sub_int64_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_sub_int64_cpx_cpx(
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
      case (DType.int64, DType.complex64) when isContig:
        v_sub_int64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64):
        s_sub_int64_cpx64_cpx64(
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
      case (DType.int32, DType.float64) when isContig:
        v_sub_int32_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_sub_int32_double_double(
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
      case (DType.int32, DType.float32) when isContig:
        v_sub_int32_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32):
        s_sub_int32_float_float(
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
      case (DType.int32, DType.int64) when isContig:
        v_sub_int32_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_sub_int32_int64_int64(
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
      case (DType.int32, DType.int32) when isContig:
        v_sub_int32_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
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
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_sub_int32_uint8_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_sub_int32_uint8_int32(
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
      case (DType.int32, DType.int16) when isContig:
        v_sub_int32_int16_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_sub_int32_int16_int32(
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
      case (DType.int32, DType.complex128) when isContig:
        v_sub_int32_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_sub_int32_cpx_cpx(
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
      case (DType.int32, DType.complex64) when isContig:
        v_sub_int32_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_sub_int32_cpx64_cpx64(
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
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_sub_uint8_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_sub_uint8_double_double(
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
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_sub_uint8_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_sub_uint8_float_float(
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
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_sub_uint8_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_sub_uint8_int64_int64(
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
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_sub_uint8_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_sub_uint8_int32_int32(
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
      case (DType.boolean, DType.boolean) when isContig:
      case (DType.boolean, DType.uint8) when isContig:
      case (DType.uint8, DType.boolean) when isContig:
      case (DType.uint8, DType.uint8) when isContig:
        v_sub_uint8_uint8_uint8(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
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
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_sub_uint8_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_sub_uint8_int16_int16(
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
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_sub_uint8_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_sub_uint8_cpx_cpx(
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
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_sub_uint8_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_sub_uint8_cpx64_cpx64(
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
      case (DType.int16, DType.float64) when isContig:
        v_sub_int16_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_sub_int16_double_double(
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
      case (DType.int16, DType.float32) when isContig:
        v_sub_int16_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_sub_int16_float_float(
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
      case (DType.int16, DType.int64) when isContig:
        v_sub_int16_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_sub_int16_int64_int64(
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
      case (DType.int16, DType.int32) when isContig:
        v_sub_int16_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_sub_int16_int32_int32(
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
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_sub_int16_uint8_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_sub_int16_uint8_int16(
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
      case (DType.int16, DType.int16) when isContig:
        v_sub_int16_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
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
      case (DType.int16, DType.complex128) when isContig:
        v_sub_int16_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_sub_int16_cpx_cpx(
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
      case (DType.int16, DType.complex64) when isContig:
        v_sub_int16_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_sub_int16_cpx64_cpx64(
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
      case (DType.complex128, DType.float64) when isContig:
        v_sub_cpx_double_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_sub_cpx_double_cpx(
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
      case (DType.complex128, DType.float32) when isContig:
        v_sub_cpx_float_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_sub_cpx_float_cpx(
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
      case (DType.complex128, DType.int64) when isContig:
        v_sub_cpx_int64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_sub_cpx_int64_cpx(
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
      case (DType.complex128, DType.int32) when isContig:
        v_sub_cpx_int32_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_sub_cpx_int32_cpx(
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
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_sub_cpx_uint8_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_sub_cpx_uint8_cpx(
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
      case (DType.complex128, DType.int16) when isContig:
        v_sub_cpx_int16_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_sub_cpx_int16_cpx(
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
      case (DType.complex128, DType.complex128) when isContig:
        v_sub_cpx_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_sub_cpx_cpx_cpx(
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
      case (DType.complex128, DType.complex64) when isContig:
        v_sub_cpx_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_sub_cpx_cpx64_cpx(
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
      case (DType.complex64, DType.float64) when isContig:
        v_sub_cpx64_double_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_sub_cpx64_double_cpx(
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
      case (DType.complex64, DType.float32) when isContig:
        v_sub_cpx64_float_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_sub_cpx64_float_cpx64(
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
      case (DType.complex64, DType.int64) when isContig:
        v_sub_cpx64_int64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64):
        s_sub_cpx64_int64_cpx64(
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
      case (DType.complex64, DType.int32) when isContig:
        v_sub_cpx64_int32_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_sub_cpx64_int32_cpx64(
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
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_sub_cpx64_uint8_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_sub_cpx64_uint8_cpx64(
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
      case (DType.complex64, DType.int16) when isContig:
        v_sub_cpx64_int16_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_sub_cpx64_int16_cpx64(
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
      case (DType.complex64, DType.complex128) when isContig:
        v_sub_cpx64_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_sub_cpx64_cpx_cpx(
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
      case (DType.complex64, DType.complex64) when isContig:
        v_sub_cpx64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_sub_cpx64_cpx64_cpx64(
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
        break;
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
  if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
    final cpxA = castNDArray(a, DType.complex128);
    final cpxB = castNDArray(b, DType.complex128);
    final cpxRes = subtract<Complex, Complex, Complex>(
      cpxA,
      cpxB,
      where: where,
    );
    final casted = castNDArray(cpxRes, result.dtype);
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(cpxA, a)) cpxA.dispose();
    if (!identical(cpxB, b)) cpxB.dispose();
    cpxRes.dispose();
    casted.dispose();
    return result;
  } else {
    final doubleA = castNDArray(a, DType.float64);
    final doubleB = castNDArray(b, DType.float64);
    final doubleRes = subtract<Float64, Float64, Float64>(
      doubleA,
      doubleB,
      where: where,
    );
    final casted = castNDArray(doubleRes, result.dtype);
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(doubleA, a)) doubleA.dispose();
    if (!identical(doubleB, b)) doubleB.dispose();
    doubleRes.dispose();
    casted.dispose();
    return result;
  }
}

/// Element-wise multiplication of two arrays with full broadcasting support.
///
/// **Overflow behavior:**
/// - **Integer arrays** (`int32`, `int64`, etc.) overflow silently wrapping around via standard two's complement.
/// - **Floating-point arrays** (`float32`, `float64`) overflow silently to `double.infinity` or `double.negativeInfinity` per IEEE 754.
NDArray<R> multiply<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute multiply() on a disposed array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }
  final maskHolder = prepareMask(where, result.shape);

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesB = cBuffer + (ndim * 2);
    final cStridesRes = cBuffer + (ndim * 3);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }

    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_mul_double_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_mul_double_double_double(
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
      case (DType.float64, DType.float32) when isContig:
        v_mul_double_float_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_mul_double_float_double(
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
      case (DType.float64, DType.int64) when isContig:
        v_mul_double_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_mul_double_int64_double(
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
      case (DType.float64, DType.int32) when isContig:
        v_mul_double_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_mul_double_int32_double(
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
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_mul_double_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_mul_double_uint8_double(
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
      case (DType.float64, DType.int16) when isContig:
        v_mul_double_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_mul_double_int16_double(
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
      case (DType.float64, DType.complex128) when isContig:
        v_mul_double_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_mul_double_cpx_cpx(
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
      case (DType.float64, DType.complex64) when isContig:
        v_mul_double_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_mul_double_cpx64_cpx(
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
      case (DType.float32, DType.float64) when isContig:
        v_mul_double_float_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_mul_double_float_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32) when isContig:
        v_mul_float_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_mul_float_float_float(
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
      case (DType.float32, DType.int64) when isContig:
        v_mul_float_int64_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64):
        s_mul_float_int64_float(
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
      case (DType.float32, DType.int32) when isContig:
        v_mul_float_int32_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32):
        s_mul_float_int32_float(
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
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_mul_float_uint8_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_mul_float_uint8_float(
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
      case (DType.float32, DType.int16) when isContig:
        v_mul_float_int16_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_mul_float_int16_float(
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
      case (DType.float32, DType.complex128) when isContig:
        v_mul_float_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_mul_float_cpx_cpx(
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
      case (DType.float32, DType.complex64) when isContig:
        v_mul_float_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_mul_float_cpx64_cpx64(
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
      case (DType.int64, DType.float64) when isContig:
        v_mul_double_int64_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_mul_double_int64_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32) when isContig:
        v_mul_float_int64_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32):
        s_mul_float_int64_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64) when isContig:
        v_mul_int64_int64_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
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
      case (DType.int64, DType.int32) when isContig:
        v_mul_int64_int32_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_mul_int64_int32_int64(
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
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_mul_int64_uint8_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_mul_int64_uint8_int64(
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
      case (DType.int64, DType.int16) when isContig:
        v_mul_int64_int16_int64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_mul_int64_int16_int64(
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
      case (DType.int64, DType.complex128) when isContig:
        v_mul_int64_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_mul_int64_cpx_cpx(
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
      case (DType.int64, DType.complex64) when isContig:
        v_mul_int64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64):
        s_mul_int64_cpx64_cpx64(
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
      case (DType.int32, DType.float64) when isContig:
        v_mul_double_int32_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_mul_double_int32_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32) when isContig:
        v_mul_float_int32_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32):
        s_mul_float_int32_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64) when isContig:
        v_mul_int64_int32_int64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_mul_int64_int32_int64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32) when isContig:
        v_mul_int32_int32_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
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
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_mul_int32_uint8_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_mul_int32_uint8_int32(
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
      case (DType.int32, DType.int16) when isContig:
        v_mul_int32_int16_int32(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_mul_int32_int16_int32(
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
      case (DType.int32, DType.complex128) when isContig:
        v_mul_int32_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_mul_int32_cpx_cpx(
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
      case (DType.int32, DType.complex64) when isContig:
        v_mul_int32_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_mul_int32_cpx64_cpx64(
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
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_mul_double_uint8_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_mul_double_uint8_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_mul_float_uint8_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_mul_float_uint8_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_mul_int64_uint8_int64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_mul_int64_uint8_int64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_mul_int32_uint8_int32(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_mul_int32_uint8_int32(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
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
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
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
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_mul_uint8_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_mul_uint8_int16_int16(
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
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_mul_uint8_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_mul_uint8_cpx_cpx(
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
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_mul_uint8_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_mul_uint8_cpx64_cpx64(
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
      case (DType.int16, DType.float64) when isContig:
        v_mul_double_int16_double(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_mul_double_int16_double(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32) when isContig:
        v_mul_float_int16_float(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_mul_float_int16_float(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64) when isContig:
        v_mul_int64_int16_int64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_mul_int64_int16_int64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32) when isContig:
        v_mul_int32_int16_int32(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_mul_int32_int16_int32(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_mul_uint8_int16_int16(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_mul_uint8_int16_int16(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16) when isContig:
        v_mul_int16_int16_int16(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
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
      case (DType.int16, DType.complex128) when isContig:
        v_mul_int16_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_mul_int16_cpx_cpx(
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
      case (DType.int16, DType.complex64) when isContig:
        v_mul_int16_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_mul_int16_cpx64_cpx64(
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
      case (DType.complex128, DType.float64) when isContig:
        v_mul_double_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_mul_double_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32) when isContig:
        v_mul_float_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_mul_float_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64) when isContig:
        v_mul_int64_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_mul_int64_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32) when isContig:
        v_mul_int32_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_mul_int32_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_mul_uint8_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_mul_uint8_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16) when isContig:
        v_mul_int16_cpx_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_mul_int16_cpx_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128) when isContig:
        v_mul_cpx_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_mul_cpx_cpx_cpx(
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
      case (DType.complex128, DType.complex64) when isContig:
        v_mul_cpx_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_mul_cpx_cpx64_cpx(
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
      case (DType.complex64, DType.float64) when isContig:
        v_mul_double_cpx64_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_mul_double_cpx64_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32) when isContig:
        v_mul_float_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_mul_float_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64) when isContig:
        v_mul_int64_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64):
        s_mul_int64_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32) when isContig:
        v_mul_int32_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_mul_int32_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_mul_uint8_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_mul_uint8_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16) when isContig:
        v_mul_int16_cpx64_cpx64(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_mul_int16_cpx64_cpx64(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128) when isContig:
        v_mul_cpx_cpx64_cpx(
          b.pointer.cast(),
          a.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_mul_cpx_cpx64_cpx(
          b.pointer.cast(),
          cStridesB,
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64) when isContig:
        v_mul_cpx64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_mul_cpx64_cpx64_cpx64(
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
        break;
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
  if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
    final cpxA = castNDArray(a, DType.complex128);
    final cpxB = castNDArray(b, DType.complex128);
    final cpxRes = multiply<Complex, Complex, Complex>(
      cpxA,
      cpxB,
      where: where,
    );
    final casted = castNDArray(cpxRes, result.dtype);
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(cpxA, a)) cpxA.dispose();
    if (!identical(cpxB, b)) cpxB.dispose();
    cpxRes.dispose();
    casted.dispose();
    return result;
  } else {
    final doubleA = castNDArray(a, DType.float64);
    final doubleB = castNDArray(b, DType.float64);
    final doubleRes = multiply<Float64, Float64, Float64>(
      doubleA,
      doubleB,
      where: where,
    );
    final casted = castNDArray(doubleRes, result.dtype);
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(doubleA, a)) doubleA.dispose();
    if (!identical(doubleB, b)) doubleB.dispose();
    doubleRes.dispose();
    casted.dispose();
    return result;
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
NDArray<R> divide<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<dynamic>? where,
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

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }
  final maskHolder = prepareMask(where, result.shape);

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesB = cBuffer + (ndim * 2);
    final cStridesRes = cBuffer + (ndim * 3);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }
    switch ((a.dtype, b.dtype)) {
      // DIV cases
      case (DType.float64, DType.float64) when isContig:
        v_div_double_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float64):
        s_div_double_double_double(
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
      case (DType.float64, DType.float32) when isContig:
        v_div_double_float_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.float32):
        s_div_double_float_double(
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
      case (DType.float64, DType.int64) when isContig:
        v_div_double_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int64):
        s_div_double_int64_double(
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
      case (DType.float64, DType.int32) when isContig:
        v_div_double_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int32):
        s_div_double_int32_double(
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
      case (DType.float64, DType.boolean) when isContig:
      case (DType.float64, DType.uint8) when isContig:
        v_div_double_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.boolean):
      case (DType.float64, DType.uint8):
        s_div_double_uint8_double(
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
      case (DType.float64, DType.int16) when isContig:
        v_div_double_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.int16):
        s_div_double_int16_double(
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
      case (DType.float64, DType.complex128) when isContig:
        v_div_double_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex128):
        s_div_double_cpx_cpx(
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
      case (DType.float64, DType.complex64) when isContig:
        v_div_double_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float64, DType.complex64):
        s_div_double_cpx64_cpx(
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
      case (DType.float32, DType.float64) when isContig:
        v_div_float_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float64):
        s_div_float_double_double(
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
      case (DType.float32, DType.float32) when isContig:
        v_div_float_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.float32):
        s_div_float_float_float(
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
      case (DType.float32, DType.int64) when isContig:
        v_div_float_int64_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int64):
        s_div_float_int64_float(
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
      case (DType.float32, DType.int32) when isContig:
        v_div_float_int32_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int32):
        s_div_float_int32_float(
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
      case (DType.float32, DType.boolean) when isContig:
      case (DType.float32, DType.uint8) when isContig:
        v_div_float_uint8_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.boolean):
      case (DType.float32, DType.uint8):
        s_div_float_uint8_float(
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
      case (DType.float32, DType.int16) when isContig:
        v_div_float_int16_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.int16):
        s_div_float_int16_float(
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
      case (DType.float32, DType.complex128) when isContig:
        v_div_float_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex128):
        s_div_float_cpx_cpx(
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
      case (DType.float32, DType.complex64) when isContig:
        v_div_float_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.float32, DType.complex64):
        s_div_float_cpx64_cpx64(
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
      case (DType.int64, DType.float64) when isContig:
        v_div_int64_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float64):
        s_div_int64_double_double(
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
      case (DType.int64, DType.float32) when isContig:
        v_div_int64_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.float32):
        s_div_int64_float_float(
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
      case (DType.int64, DType.int64) when isContig:
        v_div_int64_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int64):
        s_div_int64_int64_double(
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
      case (DType.int64, DType.int32) when isContig:
        v_div_int64_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int32):
        s_div_int64_int32_double(
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
      case (DType.int64, DType.boolean) when isContig:
      case (DType.int64, DType.uint8) when isContig:
        v_div_int64_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.boolean):
      case (DType.int64, DType.uint8):
        s_div_int64_uint8_double(
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
      case (DType.int64, DType.int16) when isContig:
        v_div_int64_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.int16):
        s_div_int64_int16_double(
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
      case (DType.int64, DType.complex128) when isContig:
        v_div_int64_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex128):
        s_div_int64_cpx_cpx(
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
      case (DType.int64, DType.complex64) when isContig:
        v_div_int64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int64, DType.complex64):
        s_div_int64_cpx64_cpx64(
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
      case (DType.int32, DType.float64) when isContig:
        v_div_int32_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float64):
        s_div_int32_double_double(
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
      case (DType.int32, DType.float32) when isContig:
        v_div_int32_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.float32):
        s_div_int32_float_float(
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
      case (DType.int32, DType.int64) when isContig:
        v_div_int32_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int64):
        s_div_int32_int64_double(
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
      case (DType.int32, DType.int32) when isContig:
        v_div_int32_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int32):
        s_div_int32_int32_double(
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
      case (DType.int32, DType.boolean) when isContig:
      case (DType.int32, DType.uint8) when isContig:
        v_div_int32_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.boolean):
      case (DType.int32, DType.uint8):
        s_div_int32_uint8_double(
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
      case (DType.int32, DType.int16) when isContig:
        v_div_int32_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.int16):
        s_div_int32_int16_double(
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
      case (DType.int32, DType.complex128) when isContig:
        v_div_int32_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex128):
        s_div_int32_cpx_cpx(
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
      case (DType.int32, DType.complex64) when isContig:
        v_div_int32_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int32, DType.complex64):
        s_div_int32_cpx64_cpx64(
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
      case (DType.boolean, DType.float64) when isContig:
      case (DType.uint8, DType.float64) when isContig:
        v_div_uint8_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float64):
      case (DType.uint8, DType.float64):
        s_div_uint8_double_double(
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
      case (DType.boolean, DType.float32) when isContig:
      case (DType.uint8, DType.float32) when isContig:
        v_div_uint8_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.float32):
      case (DType.uint8, DType.float32):
        s_div_uint8_float_float(
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
      case (DType.boolean, DType.int64) when isContig:
      case (DType.uint8, DType.int64) when isContig:
        v_div_uint8_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int64):
      case (DType.uint8, DType.int64):
        s_div_uint8_int64_double(
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
      case (DType.boolean, DType.int32) when isContig:
      case (DType.uint8, DType.int32) when isContig:
        v_div_uint8_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int32):
      case (DType.uint8, DType.int32):
        s_div_uint8_int32_double(
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
      case (DType.boolean, DType.boolean) when isContig:
      case (DType.boolean, DType.uint8) when isContig:
      case (DType.uint8, DType.boolean) when isContig:
      case (DType.uint8, DType.uint8) when isContig:
        v_div_uint8_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.boolean):
      case (DType.boolean, DType.uint8):
      case (DType.uint8, DType.boolean):
      case (DType.uint8, DType.uint8):
        s_div_uint8_uint8_double(
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
      case (DType.boolean, DType.int16) when isContig:
      case (DType.uint8, DType.int16) when isContig:
        v_div_uint8_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.int16):
      case (DType.uint8, DType.int16):
        s_div_uint8_int16_double(
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
      case (DType.boolean, DType.complex128) when isContig:
      case (DType.uint8, DType.complex128) when isContig:
        v_div_uint8_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex128):
      case (DType.uint8, DType.complex128):
        s_div_uint8_cpx_cpx(
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
      case (DType.boolean, DType.complex64) when isContig:
      case (DType.uint8, DType.complex64) when isContig:
        v_div_uint8_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.boolean, DType.complex64):
      case (DType.uint8, DType.complex64):
        s_div_uint8_cpx64_cpx64(
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
      case (DType.int16, DType.float64) when isContig:
        v_div_int16_double_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float64):
        s_div_int16_double_double(
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
      case (DType.int16, DType.float32) when isContig:
        v_div_int16_float_float(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.float32):
        s_div_int16_float_float(
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
      case (DType.int16, DType.int64) when isContig:
        v_div_int16_int64_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int64):
        s_div_int16_int64_double(
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
      case (DType.int16, DType.int32) when isContig:
        v_div_int16_int32_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int32):
        s_div_int16_int32_double(
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
      case (DType.int16, DType.boolean) when isContig:
      case (DType.int16, DType.uint8) when isContig:
        v_div_int16_uint8_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.boolean):
      case (DType.int16, DType.uint8):
        s_div_int16_uint8_double(
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
      case (DType.int16, DType.int16) when isContig:
        v_div_int16_int16_double(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.int16):
        s_div_int16_int16_double(
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
      case (DType.int16, DType.complex128) when isContig:
        v_div_int16_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex128):
        s_div_int16_cpx_cpx(
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
      case (DType.int16, DType.complex64) when isContig:
        v_div_int16_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.int16, DType.complex64):
        s_div_int16_cpx64_cpx64(
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
      case (DType.complex128, DType.float64) when isContig:
        v_div_cpx_double_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float64):
        s_div_cpx_double_cpx(
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
      case (DType.complex128, DType.float32) when isContig:
        v_div_cpx_float_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.float32):
        s_div_cpx_float_cpx(
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
      case (DType.complex128, DType.int64) when isContig:
        v_div_cpx_int64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int64):
        s_div_cpx_int64_cpx(
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
      case (DType.complex128, DType.int32) when isContig:
        v_div_cpx_int32_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int32):
        s_div_cpx_int32_cpx(
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
      case (DType.complex128, DType.boolean) when isContig:
      case (DType.complex128, DType.uint8) when isContig:
        v_div_cpx_uint8_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.boolean):
      case (DType.complex128, DType.uint8):
        s_div_cpx_uint8_cpx(
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
      case (DType.complex128, DType.int16) when isContig:
        v_div_cpx_int16_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.int16):
        s_div_cpx_int16_cpx(
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
      case (DType.complex128, DType.complex128) when isContig:
        v_div_cpx_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex128):
        s_div_cpx_cpx_cpx(
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
      case (DType.complex128, DType.complex64) when isContig:
        v_div_cpx_cpx64_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex128, DType.complex64):
        s_div_cpx_cpx64_cpx(
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
      case (DType.complex64, DType.float64) when isContig:
        v_div_cpx64_double_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float64):
        s_div_cpx64_double_cpx(
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
      case (DType.complex64, DType.float32) when isContig:
        v_div_cpx64_float_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.float32):
        s_div_cpx64_float_cpx64(
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
      case (DType.complex64, DType.int64) when isContig:
        v_div_cpx64_int64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int64):
        s_div_cpx64_int64_cpx64(
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
      case (DType.complex64, DType.int32) when isContig:
        v_div_cpx64_int32_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int32):
        s_div_cpx64_int32_cpx64(
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
      case (DType.complex64, DType.boolean) when isContig:
      case (DType.complex64, DType.uint8) when isContig:
        v_div_cpx64_uint8_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.boolean):
      case (DType.complex64, DType.uint8):
        s_div_cpx64_uint8_cpx64(
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
      case (DType.complex64, DType.int16) when isContig:
        v_div_cpx64_int16_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.int16):
        s_div_cpx64_int16_cpx64(
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
      case (DType.complex64, DType.complex128) when isContig:
        v_div_cpx64_cpx_cpx(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex128):
        s_div_cpx64_cpx_cpx(
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
      case (DType.complex64, DType.complex64) when isContig:
        v_div_cpx64_cpx64_cpx64(
          a.pointer.cast(),
          b.pointer.cast(),
          result.pointer.cast(),
          a.size,
          maskHolder.pointer,
        );
        return result;
      case (DType.complex64, DType.complex64):
        s_div_cpx64_cpx64_cpx64(
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
        break;
    }
  } finally {
    ScratchArena.reset(marker);
    maskHolder.dispose();
  }
  if (result.dtype.isComplex || a.dtype.isComplex || b.dtype.isComplex) {
    final cpxA = castNDArray(a, DType.complex128);
    final cpxB = castNDArray(b, DType.complex128);
    final cpxRes = divide<Complex, Complex, Complex>(cpxA, cpxB, where: where);
    final casted = castNDArray(cpxRes, result.dtype);
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(cpxA, a)) cpxA.dispose();
    if (!identical(cpxB, b)) cpxB.dispose();
    cpxRes.dispose();
    casted.dispose();
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
    custom_memcpy(
      result.pointer,
      casted.pointer,
      result.size * result.dtype.byteWidth,
    );
    if (!identical(doubleA, a)) doubleA.dispose();
    if (!identical(doubleB, b)) doubleB.dispose();
    doubleRes.dispose();
    casted.dispose();
    return result;
  }
}
