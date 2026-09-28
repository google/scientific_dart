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
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../helpers.dart';

/// Computes the zeroth order modified Bessel function of the first kind, $I_0(x)$ element-wise.
///
/// Supports float32, float64, complex64, complex128.
/// Integer and boolean types are promoted to float64.
///
/// **Mathematical Details:**
/// - For real types (float32, float64):
///   - For $|x| \le 3.75$, uses the Abramowitz-Stegun polynomial approximation.
///   - For $|x| > 3.75$, uses the asymptotic expansion.
/// - For complex types (complex64, complex128):
///   - For $|z| \le 15$, uses the power series expansion:
///     $$I_0(z) = \sum_{k=0}^{n} \frac{(z^2/4)^k}{(k!)^2}$$
///   - For $|z| > 15$, uses the two-term asymptotic expansion:
///     $$I_0(z) \approx \frac{e^z}{\sqrt{2\pi z}} A(z) + i \frac{e^{-z}}{\sqrt{2\pi z}} A(-z)$$
///     where $A(z) = 1 + \frac{1}{8z} + \frac{9}{128z^2} + \frac{75}{1024z^3} + \frac{1225}{32768z^4}$,
///     mapped to the first quadrant and conjugated appropriately to handle all quadrants.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have the same shape
///   and compatible dtype as the result.
///
/// It is an error if:
/// - [a] or [out] is disposed (throws [StateError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - the dtype of [a] is not supported (throws [ArgumentError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
/// final b = i0(a);
/// print(b.toList()); // [1.0, ~1.266066, ~2.279585]
/// ```
NDArray<R> i0<T, R>(NDArray<T> a, {NDArray<dynamic>? where, NDArray<R>? out}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError('Cannot execute i0() on a disposed array.');
  }

  // Handle integer and boolean types by promoting to float64 (double)
  if (a.dtype.isInteger || a.dtype == DType.boolean) {
    final promoted = promoteToDouble(a);
    final res = i0<double, double>(
      promoted,
      where: where,
      out: out as NDArray<double>?,
    );
    promoted.dispose();
    return res as NDArray<R>;
  }

  final DType<R> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype as DType<R>,
    DType.float32 => DType.float32 as DType<R>,
    _ => DType.float64 as DType<R>,
  };

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for i0.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    void dispatchContiguous(NDArray src, NDArray dest) {
      switch (src.dtype) {
        case DType.float64:
          v_i0_double(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
          break;
        case DType.float32:
          v_i0_float(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
          break;
        case DType.complex128:
          v_i0_complex128(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
          break;
        case DType.complex64:
          v_i0_complex64(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
          break;
        default:
          throw UnsupportedError(
            'Unsupported dtype for i0 contiguous dispatch: ${src.dtype}',
          );
      }
    }

    void dispatchStrided(NDArray src, NDArray dest) {
      final rank = src.shape.length;
      final marker = ScratchArena.marker;
      final cShape = ScratchArena.copyInts(src.shape);
      final cStridesSrc = ScratchArena.copyInts(src.strides);
      final cStridesDest = ScratchArena.copyInts(dest.strides);
      try {
        switch (src.dtype) {
          case DType.float64:
            s_i0_double(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
            break;
          case DType.float32:
            s_i0_float(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
            break;
          case DType.complex128:
            s_i0_complex128(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
            break;
          case DType.complex64:
            s_i0_complex64(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
            break;
          default:
            throw UnsupportedError(
              'Unsupported dtype for i0 strided dispatch: ${src.dtype}',
            );
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (a.isContiguous && result.isContiguous) {
      dispatchContiguous(a, result);
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        dispatchStrided(a, result);
      } else {
        final tempA = a.isContiguous ? a : a.copy();
        final tempResult = result.isContiguous
            ? result
            : NDArray.create(result.shape, result.dtype);

        dispatchContiguous(tempA, tempResult);

        if (!identical(tempResult, result)) {
          tempResult.copy(out: result);
          tempResult.dispose();
        }
        if (!identical(tempA, a)) {
          tempA.dispose();
        }
      }
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the gamma function, $\Gamma(x)$, element-wise.
///
/// Supports float32 and float64. Integer and boolean types are promoted to float64.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have the same shape
///   and compatible dtype as the result.
///
/// It is an error if:
/// - [a] or [out] is disposed (throws [StateError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - [a] has a complex dtype (throws [UnsupportedError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final b = gamma(a);
/// print(b.toList()); // [1.0, 1.0, 2.0, 6.0]
/// ```
NDArray<R> gamma<T, R>(
  NDArray<T> a, {
  NDArray<dynamic>? where,
  NDArray<R>? out,
}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute gamma() on a disposed array.");
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError("Complex numbers are not supported for gamma.");
  }

  if (a.dtype.isInteger || a.dtype == DType.boolean) {
    final promoted = promoteToDouble(a);
    final res = gamma<double, double>(
      promoted,
      where: where,
      out: out as NDArray<double>?,
    );
    promoted.dispose();
    return res as NDArray<R>;
  }

  final targetDType =
      (a.dtype == DType.float32 ? DType.float32 : DType.float64) as DType<R>;
  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype for gamma.",
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    void dispatchContiguous(NDArray src, NDArray dest) {
      switch (src.dtype) {
        case DType.float64:
          v_gamma_double(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
        case DType.float32:
          v_gamma_float(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
        default:
          throw UnsupportedError('Unsupported dtype: ${src.dtype}');
      }
    }

    void dispatchStrided(NDArray src, NDArray dest) {
      final rank = src.shape.length;
      final marker = ScratchArena.marker;
      final cShape = ScratchArena.copyInts(src.shape);
      final cStridesSrc = ScratchArena.copyInts(src.strides);
      final cStridesDest = ScratchArena.copyInts(dest.strides);
      try {
        switch (src.dtype) {
          case DType.float64:
            s_gamma_double(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.float32:
            s_gamma_float(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
          default:
            throw UnsupportedError('Unsupported dtype: ${src.dtype}');
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (a.isContiguous && result.isContiguous) {
      dispatchContiguous(a, result);
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        dispatchStrided(a, result);
      } else {
        final tempA = a.isContiguous ? a : a.copy();
        final tempResult = result.isContiguous
            ? result
            : NDArray.create(result.shape, result.dtype);

        dispatchContiguous(tempA, tempResult);

        if (!identical(tempResult, result)) {
          tempResult.copy(out: result);
          tempResult.dispose();
        }
        if (!identical(tempA, a)) {
          tempA.dispose();
        }
      }
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}

/// Computes the error function, $\text{erf}(x)$, element-wise.
///
/// Supports float32 and float64. Integer and boolean types are promoted to float64.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have the same shape
///   and compatible dtype as the result.
///
/// It is an error if:
/// - [a] or [out] is disposed (throws [StateError]).
/// - [out] has incompatible shape or dtype (throws [ArgumentError]).
/// - [a] has a complex dtype (throws [UnsupportedError]).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);
/// final b = erf(a);
/// print(b.toList()); // [0.0, ~0.8427]
/// ```
NDArray<R> erf<T, R>(NDArray<T> a, {NDArray<dynamic>? where, NDArray<R>? out}) {
  if (a.isDisposed ||
      (out != null && out.isDisposed) ||
      (where != null && where.isDisposed)) {
    throw StateError("Cannot execute erf() on a disposed array.");
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError("Complex numbers are not supported for erf.");
  }

  if (a.dtype.isInteger || a.dtype == DType.boolean) {
    final promoted = promoteToDouble(a);
    final res = erf<double, double>(
      promoted,
      where: where,
      out: out as NDArray<double>?,
    );
    promoted.dispose();
    return res as NDArray<R>;
  }

  final targetDType =
      (a.dtype == DType.float32 ? DType.float32 : DType.float64) as DType<R>;
  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype for erf.",
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }

  final maskHolder = prepareMask(where, result.shape);
  try {
    void dispatchContiguous(NDArray src, NDArray dest) {
      switch (src.dtype) {
        case DType.float64:
          v_erf_double(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
        case DType.float32:
          v_erf_float(
            src.pointer.cast(),
            dest.pointer.cast(),
            src.size,
            maskHolder.pointer,
          );
        default:
          throw UnsupportedError('Unsupported dtype: ${src.dtype}');
      }
    }

    void dispatchStrided(NDArray src, NDArray dest) {
      final rank = src.shape.length;
      final marker = ScratchArena.marker;
      final cShape = ScratchArena.copyInts(src.shape);
      final cStridesSrc = ScratchArena.copyInts(src.strides);
      final cStridesDest = ScratchArena.copyInts(dest.strides);
      try {
        switch (src.dtype) {
          case DType.float64:
            s_erf_double(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.float32:
            s_erf_float(
              src.pointer.cast(),
              cStridesSrc,
              dest.pointer.cast(),
              cStridesDest,
              cShape,
              rank,
              maskHolder.pointer,
            );
          default:
            throw UnsupportedError('Unsupported dtype: ${src.dtype}');
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }

    if (a.isContiguous && result.isContiguous) {
      dispatchContiguous(a, result);
    } else {
      final rank = a.shape.length;
      if (rank <= 8) {
        dispatchStrided(a, result);
      } else {
        final tempA = a.isContiguous ? a : a.copy();
        final tempResult = result.isContiguous
            ? result
            : NDArray.create(result.shape, result.dtype);

        dispatchContiguous(tempA, tempResult);

        if (!identical(tempResult, result)) {
          tempResult.copy(out: result);
          tempResult.dispose();
        }
        if (!identical(tempA, a)) {
          tempA.dispose();
        }
      }
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}
