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
NDArray<R> i0<R extends DTypeTag>(
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
    throw StateError('Cannot execute i0() on a disposed array.');
  }

  final DType<R> targetDType = switch (a.dtype) {
    DType.complex128 || DType.complex64 => a.dtype as DType<R>,
    DType.float32 => DType.float32 as DType<R>,
    _ => DType.float64 as DType<R>,
  };

  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for i0.',
      );
    }
  }

  // Handle integer, boolean, and half-precision float types by promoting to float64
  final aDType = a.dtype as DType<DTypeTag>;
  if (aDType.isInteger ||
      aDType == DType.boolean ||
      aDType == DType.float16 ||
      aDType == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      return i0<Float64>(promoted, where: where, out: out as NDArray<Float64>?)
          as NDArray<R>;
    } finally {
      if (!identical(promoted, a)) {
        promoted.dispose();
      }
    }
  }

  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ?? NDArray.create(a.shape, targetDType, zeroInit: where != null);

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
          throw UnsupportedError(
            'Unsupported dtype for i0 contiguous dispatch: ${src.dtype}',
          );
      }
    }

    void dispatchStrided(NDArray src, NDArray dest) {
      final rank = src.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(src.shape);
        final cStridesSrc = ScratchArena.copyInts(src.strides);
        final cStridesDest = ScratchArena.copyInts(dest.strides);
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
      if (rank <= 32) {
        dispatchStrided(a, result);
      } else {
        final tempA = a.isContiguous ? a : a.copy();
        try {
          final tempResult = result.isContiguous
              ? result
              : (where != null
                    ? result.copy()
                    : NDArray.create(result.shape, result.dtype));
          try {
            dispatchContiguous(tempA, tempResult);
            if (!identical(tempResult, result)) {
              tempResult.copy(out: result);
            }
          } finally {
            if (!identical(tempResult, result)) {
              tempResult.dispose();
            }
          }
        } finally {
          if (!identical(tempA, a)) {
            tempA.dispose();
          }
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
NDArray<R> gamma<R extends DTypeTag>(
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
    throw StateError("Cannot execute gamma() on a disposed array.");
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError("Complex numbers are not supported for gamma.");
  }

  final DType<R> targetDType = switch (a.dtype) {
    DType.float32 => DType.float32 as DType<R>,
    _ => DType.float64 as DType<R>,
  };
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype for gamma.",
      );
    }
  }

  final aDType = a.dtype as DType<DTypeTag>;
  if (aDType.isInteger ||
      aDType == DType.boolean ||
      aDType == DType.float16 ||
      aDType == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      return gamma<Float64>(
            promoted,
            where: where,
            out: out as NDArray<Float64>?,
          )
          as NDArray<R>;
    } finally {
      if (!identical(promoted, a)) {
        promoted.dispose();
      }
    }
  }

  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ?? NDArray.create(a.shape, targetDType, zeroInit: where != null);

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
          throw UnsupportedError('Unsupported dtype: ${src.dtype}');
      }
    }

    void dispatchStrided(NDArray src, NDArray dest) {
      final rank = src.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(src.shape);
        final cStridesSrc = ScratchArena.copyInts(src.strides);
        final cStridesDest = ScratchArena.copyInts(dest.strides);
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
      if (rank <= 32) {
        dispatchStrided(a, result);
      } else {
        final tempA = a.isContiguous ? a : a.copy();
        try {
          final tempResult = result.isContiguous
              ? result
              : (where != null
                    ? result.copy()
                    : NDArray.create(result.shape, result.dtype));
          try {
            dispatchContiguous(tempA, tempResult);
            if (!identical(tempResult, result)) {
              tempResult.copy(out: result);
            }
          } finally {
            if (!identical(tempResult, result)) {
              tempResult.dispose();
            }
          }
        } finally {
          if (!identical(tempA, a)) {
            tempA.dispose();
          }
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
NDArray<R> erf<R extends DTypeTag>(
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
    throw StateError("Cannot execute erf() on a disposed array.");
  }
  if (a.dtype.isComplex) {
    throw UnsupportedError("Complex numbers are not supported for erf.");
  }

  final DType<R> targetDType = switch (a.dtype) {
    DType.float32 => DType.float32 as DType<R>,
    _ => DType.float64 as DType<R>,
  };
  if (out != null) {
    if (!out.isWriteable ||
        !listEquals(out.shape, a.shape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype for erf.",
      );
    }
  }

  final aDType = a.dtype as DType<DTypeTag>;
  if (aDType.isInteger ||
      aDType == DType.boolean ||
      aDType == DType.float16 ||
      aDType == DType.bfloat16) {
    final promoted = promoteToDouble(a);
    try {
      return erf<Float64>(promoted, where: where, out: out as NDArray<Float64>?)
          as NDArray<R>;
    } finally {
      if (!identical(promoted, a)) {
        promoted.dispose();
      }
    }
  }

  final maskHolder = prepareMask(where, a.shape);
  try {
    final NDArray<R> result =
        out ?? NDArray.create(a.shape, targetDType, zeroInit: where != null);

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
          throw UnsupportedError('Unsupported dtype: ${src.dtype}');
      }
    }

    void dispatchStrided(NDArray src, NDArray dest) {
      final rank = src.shape.length;
      final marker = ScratchArena.marker;
      try {
        final cShape = ScratchArena.copyInts(src.shape);
        final cStridesSrc = ScratchArena.copyInts(src.strides);
        final cStridesDest = ScratchArena.copyInts(dest.strides);
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
      if (rank <= 32) {
        dispatchStrided(a, result);
      } else {
        final tempA = a.isContiguous ? a : a.copy();
        try {
          final tempResult = result.isContiguous
              ? result
              : (where != null
                    ? result.copy()
                    : NDArray.create(result.shape, result.dtype));
          try {
            dispatchContiguous(tempA, tempResult);
            if (!identical(tempResult, result)) {
              tempResult.copy(out: result);
            }
          } finally {
            if (!identical(tempResult, result)) {
              tempResult.dispose();
            }
          }
        } finally {
          if (!identical(tempA, a)) {
            tempA.dispose();
          }
        }
      }
    }

    return result;
  } finally {
    maskHolder.dispose();
  }
}
