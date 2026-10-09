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

// Orthogonal polynomial series (Chebyshev, Legendre, Hermite, Laguerre).
library;

import "../../ndarray.dart";
import "../../ndarray_bindings.dart";
import "../../scratch_arena.dart";
import "../helpers.dart";
import "../linalg.dart";

enum _OrthoKind { chebyshev, legendre, hermite, laguerre }

Object _addScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca + cb;
  }
  return (a as num).toDouble() + (b as num).toDouble();
}

Object _divScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca / cb;
  }
  return (a as num).toDouble() / (b as num).toDouble();
}

Object _mulScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca * cb;
  }
  return (a as num).toDouble() * (b as num).toDouble();
}

Object _subScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca - cb;
  }
  return (a as num).toDouble() - (b as num).toDouble();
}

Object _negScalar(Object a) {
  if (a is Complex) {
    return -a;
  }
  return -(a as num).toDouble();
}

bool _isZeroScalar(Object a) {
  if (a is Complex) {
    return a.real == 0.0 && a.imag == 0.0;
  }
  return (a as num) == 0;
}

NDArray<R> _ensureDType<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> a,
  DType<R> targetDType,
) {
  if (a.dtype == targetDType) {
    return a as NDArray<R>;
  }
  return castNDArray(a, targetDType);
}

void _copyInto<R extends DTypeTag>(NDArray src, NDArray<R> out) {
  src.copy(out: out);
}

/// Evaluates a Chebyshev series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate $p(x) = \sum c_i T_i(x)$.
///
/// [x] and [c] share the type parameter `T`, so both must have the same dtype;
/// the result dtype [R] is the inexact promotion of that dtype (`Float32` and
/// the complex dtypes are preserved, every other dtype evaluates in `Float64`).
/// Passing arrays of different concrete dtypes is a compile-time error; when
/// the dtype equality cannot be checked statically (for example with
/// `NDArray<AnySpec>` arguments), it is validated at runtime.
///
/// **Preconditions:**
/// - [x] and [c] must not be disposed.
/// - [c] must be 1-dimensional and non-empty.
/// - [c] must have the same dtype as [x].
/// - It is an error if any input or [out] buffer is disposed.
/// - It is an error if [c] is invalid or [out] buffer mismatches.
///
/// Reference: [NumPy chebval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval.html)
NDArray<R> chebval<
  T extends SelfOf<InexactOf<R>>,
  R extends DTypeTag,
  Out extends R
>(NDArray<T> x, NDArray<T> c, {NDArray<Out>? out}) {
  if (x.isDisposed || c.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  return _evalClenshaw(c, x, _OrthoKind.chebyshev, out: out);
}

/// Evaluates a Legendre series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate $p(x) = \sum c_i P_i(x)$.
///
/// [x] and [c] share the type parameter `T`, so both must have the same dtype;
/// the result dtype [R] is the inexact promotion of that dtype. Passing arrays
/// of different concrete dtypes is a compile-time error; otherwise the dtype
/// equality is validated at runtime. See [chebval] for details.
///
/// Reference: [NumPy legval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legval.html)
NDArray<R> legval<
  T extends SelfOf<InexactOf<R>>,
  R extends DTypeTag,
  Out extends R
>(NDArray<T> x, NDArray<T> c, {NDArray<Out>? out}) {
  if (x.isDisposed || c.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  return _evalClenshaw(c, x, _OrthoKind.legendre, out: out);
}

/// Evaluates a Hermite series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate $p(x) = \sum c_i H_i(x)$.
///
/// [x] and [c] share the type parameter `T`, so both must have the same dtype;
/// the result dtype [R] is the inexact promotion of that dtype. Passing arrays
/// of different concrete dtypes is a compile-time error; otherwise the dtype
/// equality is validated at runtime. See [chebval] for details.
///
/// Reference: [NumPy hermval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.hermval.html)
NDArray<R> hermval<
  T extends SelfOf<InexactOf<R>>,
  R extends DTypeTag,
  Out extends R
>(NDArray<T> x, NDArray<T> c, {NDArray<Out>? out}) {
  if (x.isDisposed || c.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  return _evalClenshaw(c, x, _OrthoKind.hermite, out: out);
}

/// Evaluates a Laguerre series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate $p(x) = \sum c_i L_i(x)$.
///
/// [x] and [c] share the type parameter `T`, so both must have the same dtype;
/// the result dtype [R] is the inexact promotion of that dtype. Passing arrays
/// of different concrete dtypes is a compile-time error; otherwise the dtype
/// equality is validated at runtime. See [chebval] for details.
///
/// Reference: [NumPy lagval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.laguerre.lagval.html)
NDArray<R> lagval<
  T extends SelfOf<InexactOf<R>>,
  R extends DTypeTag,
  Out extends R
>(NDArray<T> x, NDArray<T> c, {NDArray<Out>? out}) {
  if (x.isDisposed || c.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  return _evalClenshaw(c, x, _OrthoKind.laguerre, out: out);
}

NDArray<R> _evalClenshaw<T extends DTypeTag, R extends DTypeTag>(
  NDArray<T> c,
  NDArray<T> x,
  _OrthoKind kind, {
  NDArray<R>? out,
}) {
  if (c.isDisposed || x.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute series evaluation on a disposed array.");
  }
  if (c.shape.length != 1) {
    throw ArgumentError.value(
      c.shape,
      'c',
      'Must be 1-dimensional (got shape ${c.shape})',
    );
  }
  if (c.shape[0] == 0) {
    throw ArgumentError.value(c.shape[0], 'c', 'Must not be empty');
  }
  if (c.dtype != x.dtype) {
    throw ArgumentError.value(
      c.dtype,
      'c',
      'Must have the same dtype as x (${x.dtype})',
    );
  }

  DType<DTypeTag> resolved = x.dtype;
  if ((!resolved.isFloating && !resolved.isComplex) ||
      resolved == DType.float16 ||
      resolved == DType.bfloat16) {
    resolved = DType.float64;
  }
  final targetDType = resolved as DType<R>;
  if (out != null) {
    validateOutBuffer(out);
    if (!listEquals(out.shape, x.shape) || out.dtype != targetDType) {
      throw ArgumentError.value(
        out,
        'out',
        'Must have compatible shape and dtype for series evaluation (expected shape ${x.shape} and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype})',
      );
    }
  }

  return NDArray.scope(() {
    final cCast = _ensureDType(c, targetDType);
    final xCast = _ensureDType(x, targetDType);
    final aliased =
        out != null && (sharesMemory(c, out) || sharesMemory(x, out));
    final res = (out != null && !aliased)
        ? out
        : NDArray<R>.zeros(x.shape, targetDType);

    final isContiguous =
        cCast.isContiguous && xCast.isContiguous && res.isContiguous;
    final totalElements = xCast.shape.isEmpty
        ? 1
        : xCast.shape.reduce((a, b) => a * b);
    final nCoeffs = cCast.shape[0];
    final strideC = cCast.strides.isEmpty ? 1 : cCast.strides[0];

    final marker = ScratchArena.marker;
    try {
      if (isContiguous) {
        switch (targetDType) {
          case DType.float64:
            switch (kind) {
              case _OrthoKind.chebyshev:
                v_chebval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.legendre:
                v_legval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.hermite:
                v_hermval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.laguerre:
                v_lagval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
            }
          case DType.float32:
            switch (kind) {
              case _OrthoKind.chebyshev:
                v_chebval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.legendre:
                v_legval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.hermite:
                v_hermval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.laguerre:
                v_lagval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
            }
          case DType.complex128:
            switch (kind) {
              case _OrthoKind.chebyshev:
                v_chebval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.legendre:
                v_legval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.hermite:
                v_hermval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.laguerre:
                v_lagval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
            }
          case DType.complex64:
            switch (kind) {
              case _OrthoKind.chebyshev:
                v_chebval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.legendre:
                v_legval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.hermite:
                v_hermval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
              case _OrthoKind.laguerre:
                v_lagval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  res.pointer.cast(),
                  totalElements,
                );
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
            throw UnsupportedError(
              "Unsupported dtype $targetDType for orthogonal series evaluation.",
            );
        }
      } else {
        final ndim = xCast.shape.isEmpty ? 1 : xCast.shape.length;
        final cShape = ScratchArena.copyInts(
          xCast.shape.isEmpty ? [1] : xCast.shape,
        );
        final cStridesX = ScratchArena.copyInts(
          xCast.shape.isEmpty ? [0] : xCast.strides,
        );
        final cStridesRes = ScratchArena.copyInts(
          xCast.shape.isEmpty ? [0] : res.strides,
        );

        switch (targetDType) {
          case DType.float64:
            switch (kind) {
              case _OrthoKind.chebyshev:
                s_chebval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.legendre:
                s_legval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.hermite:
                s_hermval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.laguerre:
                s_lagval_double(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
            }
          case DType.float32:
            switch (kind) {
              case _OrthoKind.chebyshev:
                s_chebval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.legendre:
                s_legval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.hermite:
                s_hermval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.laguerre:
                s_lagval_float(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
            }
          case DType.complex128:
            switch (kind) {
              case _OrthoKind.chebyshev:
                s_chebval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.legendre:
                s_legval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.hermite:
                s_hermval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.laguerre:
                s_lagval_complex128(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
            }
          case DType.complex64:
            switch (kind) {
              case _OrthoKind.chebyshev:
                s_chebval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.legendre:
                s_legval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.hermite:
                s_hermval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
              case _OrthoKind.laguerre:
                s_lagval_complex64(
                  cCast.pointer.cast(),
                  strideC,
                  nCoeffs,
                  xCast.pointer.cast(),
                  cStridesX,
                  res.pointer.cast(),
                  cStridesRes,
                  cShape,
                  ndim,
                );
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
            throw UnsupportedError(
              "Unsupported dtype $targetDType for orthogonal series evaluation.",
            );
        }
        checkNativeOom();
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (out != null) {
      if (aliased) {
        res.copy(out: out);
      }
      return out;
    }
    return res.detachToParentScope();
  });
}

/// Finds roots of a Chebyshev series.
NDArray<C> chebroots<C extends DTypeTag, Out extends C>(
  NDArray<ComplexOf<C>> c, {
  NDArray<Out>? out,
}) {
  return _orthoRoots<C>(c, _OrthoKind.chebyshev, out: out);
}

/// Finds roots of a Legendre series.
NDArray<C> legroots<C extends DTypeTag, Out extends C>(
  NDArray<ComplexOf<C>> c, {
  NDArray<Out>? out,
}) {
  return _orthoRoots<C>(c, _OrthoKind.legendre, out: out);
}

/// Finds roots of a Hermite series.
NDArray<C> hermroots<C extends DTypeTag, Out extends C>(
  NDArray<ComplexOf<C>> c, {
  NDArray<Out>? out,
}) {
  return _orthoRoots<C>(c, _OrthoKind.hermite, out: out);
}

/// Finds roots of a Laguerre series.
NDArray<C> lagroots<C extends DTypeTag, Out extends C>(
  NDArray<ComplexOf<C>> c, {
  NDArray<Out>? out,
}) {
  return _orthoRoots<C>(c, _OrthoKind.laguerre, out: out);
}

NDArray<C> _orthoRoots<C extends DTypeTag>(
  NDArray<ComplexOf<C>> c,
  _OrthoKind kind, {
  NDArray<C>? out,
}) {
  if (c.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute root finding on a disposed array.");
  }
  if (c.shape.length != 1) {
    throw ArgumentError.value(
      c.shape,
      'c',
      'Must be 1-dimensional (got shape ${c.shape})',
    );
  }

  final DType<DTypeTag> cDType = c.dtype;
  final DType<C> targetComplexDType =
      ((cDType == DType.complex64 || cDType == DType.float32)
              ? DType.complex64
              : DType.complex128)
          as DType<C>;

  return NDArray.scope(() {
    var n = c.shape[0] - 1;
    while (n > 0) {
      if (!_isZeroScalar(c.getCellFlat(n) as Object)) break;
      n--;
    }
    final deg = n <= 0 ? 0 : n;
    if (out != null) {
      validateOutBuffer(out);
      if (!listEquals(out.shape, [deg]) || out.dtype != targetComplexDType) {
        throw ArgumentError.value(
          out,
          'out',
          'Must have compatible shape or dtype for roots result (expected shape [$deg] and dtype $targetComplexDType, got shape ${out.shape} and dtype ${out.dtype})',
        );
      }
      if (!out.isContiguous || sharesMemory(c, out)) {
        final temp = _orthoRoots<C>(c, kind);
        _copyInto(temp, out);
        return out;
      }
    }

    if (n <= 0) {
      final res = NDArray<C>.zeros([0], targetComplexDType);
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final cn = c.getCellFlat(n) as Object;
    if (n == 1) {
      final c0 = c.getCellFlat(0) as Object;
      Object rootVal;
      switch (kind) {
        case _OrthoKind.chebyshev:
        case _OrthoKind.legendre:
          rootVal = _divScalar(_negScalar(c0), cn);
          break;
        case _OrthoKind.hermite:
          rootVal = _divScalar(_negScalar(c0), _mulScalar(cn, 2.0));
          break;
        case _OrthoKind.laguerre:
          rootVal = _addScalar(1.0, _divScalar(c0, cn));
          break;
      }
      final complexRoot = rootVal is Complex
          ? rootVal
          : Complex((rootVal as num).toDouble(), 0.0);
      final res = NDArray<C>.fromList([complexRoot], [1], targetComplexDType);
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final bool isComp = cDType == DType.complex64 || cDType == DType.complex128;
    final NDArray cMat;
    switch (cDType) {
      case DType.complex64:
      case DType.complex128:
      case DType.float32:
        cMat = NDArray<DTypeTag>.zeros([n, n], cDType);
        break;
      case DType.float64:
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
        cMat = NDArray<Float64>.zeros([n, n], DType.float64);
        break;
    }
    final targetMatDType = cMat.dtype;

    switch (kind) {
      case _OrthoKind.chebyshev:
        cMat.setCellFlat(
          1 * n + 0,
          castValue(isComp ? Complex(1.0, 0.0) : 1.0, targetMatDType),
        );
        for (var i = 1; i < n - 1; i++) {
          cMat.setCellFlat(
            (i + 1) * n + i,
            castValue(isComp ? Complex(0.5, 0.0) : 0.5, targetMatDType),
          );
        }
        for (var i = 0; i < n - 1; i++) {
          cMat.setCellFlat(
            i * n + i + 1,
            castValue(isComp ? Complex(0.5, 0.0) : 0.5, targetMatDType),
          );
        }
        for (var i = 0; i < n; i++) {
          final ci = c.getCellFlat(i) as Object;
          final factor = (i == n - 1) ? 1.0 : 2.0;
          final denom = _mulScalar(cn, factor);
          final norm = _divScalar(ci, denom);
          final cur = cMat.getCellFlat(i * n + n - 1) as Object;
          final updated = _subScalar(cur, norm);
          cMat.setCellFlat(i * n + n - 1, castValue(updated, targetMatDType));
        }
        break;

      case _OrthoKind.legendre:
        for (var i = 0; i < n - 1; i++) {
          final sub = (i + 1) / (2 * i + 3);
          final sup = (i + 1) / (2 * i + 1);
          cMat.setCellFlat(
            (i + 1) * n + i,
            castValue(isComp ? Complex(sub, 0.0) : sub, targetMatDType),
          );
          cMat.setCellFlat(
            i * n + i + 1,
            castValue(isComp ? Complex(sup, 0.0) : sup, targetMatDType),
          );
        }
        final factor = (2 * n + 1) / n;
        for (var i = 0; i < n; i++) {
          final ci = c.getCellFlat(i) as Object;
          final denom = _mulScalar(cn, factor);
          final norm = _divScalar(ci, denom);
          final cur = cMat.getCellFlat(i * n + n - 1) as Object;
          final updated = _subScalar(cur, norm);
          cMat.setCellFlat(i * n + n - 1, castValue(updated, targetMatDType));
        }
        break;

      case _OrthoKind.hermite:
        for (var i = 0; i < n - 1; i++) {
          cMat.setCellFlat(
            (i + 1) * n + i,
            castValue(isComp ? Complex(0.5, 0.0) : 0.5, targetMatDType),
          );
          cMat.setCellFlat(
            i * n + i + 1,
            castValue(
              isComp ? Complex((i + 1).toDouble(), 0.0) : (i + 1).toDouble(),
              targetMatDType,
            ),
          );
        }
        for (var i = 0; i < n; i++) {
          final ci = c.getCellFlat(i) as Object;
          final denom = _mulScalar(cn, 2.0);
          final norm = _divScalar(ci, denom);
          final cur = cMat.getCellFlat(i * n + n - 1) as Object;
          final updated = _subScalar(cur, norm);
          cMat.setCellFlat(i * n + n - 1, castValue(updated, targetMatDType));
        }
        break;

      case _OrthoKind.laguerre:
        for (var i = 0; i < n; i++) {
          cMat.setCellFlat(
            i * n + i,
            castValue(
              isComp
                  ? Complex((2 * i + 1).toDouble(), 0.0)
                  : (2 * i + 1).toDouble(),
              targetMatDType,
            ),
          );
        }
        for (var i = 0; i < n - 1; i++) {
          cMat.setCellFlat(
            (i + 1) * n + i,
            castValue(
              isComp ? Complex(-(i + 1).toDouble(), 0.0) : -(i + 1).toDouble(),
              targetMatDType,
            ),
          );
          cMat.setCellFlat(
            i * n + i + 1,
            castValue(
              isComp ? Complex(-(i + 1).toDouble(), 0.0) : -(i + 1).toDouble(),
              targetMatDType,
            ),
          );
        }
        for (var i = 0; i < n; i++) {
          final ci = c.getCellFlat(i) as Object;
          final norm = _divScalar(_mulScalar(ci, n.toDouble()), cn);
          final cur = cMat.getCellFlat(i * n + n - 1) as Object;
          final updated = _subScalar(cur, norm);
          cMat.setCellFlat(i * n + n - 1, castValue(updated, targetMatDType));
        }
        break;
    }
    final res = eigvals<C, C>(cMat as NDArray<ComplexOf<C>>, out: out);
    if (out != null) return out;
    return res.detachToParentScope();
  });
}
