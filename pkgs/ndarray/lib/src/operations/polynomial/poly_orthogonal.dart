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
/// Uses backward Clenshaw recurrence to evaluate p(x) = sum(c[i] * T_i(x)).
/// Supports flexible argument order (c, x) or (x, c).
///
/// **Preconditions:**
/// - Input arrays must not be disposed.
/// - Coefficient array must be 1-dimensional and non-empty.
/// - It is an error if any input or [out] buffer is disposed.
/// - It is an error if coefficient array is invalid or [out] buffer mismatches.
///
/// Reference: [NumPy chebval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval.html)
NDArray<R> chebval<
  T1 extends DTypeTag,
  T2 extends DTypeTag,
  R extends DTypeTag
>(NDArray<T1> arg1, NDArray<T2> arg2, {NDArray<R>? out}) {
  if (arg1.isDisposed || arg2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  NDArray cArr;
  NDArray xArr;
  if (arg1.shape.length != 1 && arg2.shape.length == 1) {
    xArr = arg1;
    cArr = arg2;
  } else {
    cArr = arg1;
    xArr = arg2;
  }
  return _evalClenshaw(cArr, xArr, _OrthoKind.chebyshev, out: out);
}

/// Evaluates a Legendre series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate p(x) = sum(c[i] * P_i(x)).
/// Supports flexible argument order (c, x) or (x, c).
///
/// Reference: [NumPy legval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legval.html)
NDArray<R> legval<T1 extends DTypeTag, T2 extends DTypeTag, R extends DTypeTag>(
  NDArray<T1> arg1,
  NDArray<T2> arg2, {
  NDArray<R>? out,
}) {
  if (arg1.isDisposed || arg2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  NDArray cArr;
  NDArray xArr;
  if (arg1.shape.length != 1 && arg2.shape.length == 1) {
    xArr = arg1;
    cArr = arg2;
  } else {
    cArr = arg1;
    xArr = arg2;
  }
  return _evalClenshaw(cArr, xArr, _OrthoKind.legendre, out: out);
}

/// Evaluates a Hermite series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate p(x) = sum(c[i] * H_i(x)).
/// Supports flexible argument order (c, x) or (x, c).
///
/// Reference: [NumPy hermval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.hermval.html)
NDArray<R> hermval<
  T1 extends DTypeTag,
  T2 extends DTypeTag,
  R extends DTypeTag
>(NDArray<T1> arg1, NDArray<T2> arg2, {NDArray<R>? out}) {
  if (arg1.isDisposed || arg2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  NDArray cArr;
  NDArray xArr;
  if (arg1.shape.length != 1 && arg2.shape.length == 1) {
    xArr = arg1;
    cArr = arg2;
  } else {
    cArr = arg1;
    xArr = arg2;
  }
  return _evalClenshaw(cArr, xArr, _OrthoKind.hermite, out: out);
}

/// Evaluates a Laguerre series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate p(x) = sum(c[i] * L_i(x)).
/// Supports flexible argument order (c, x) or (x, c).
///
/// Reference: [NumPy lagval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.laguerre.lagval.html)
NDArray<R> lagval<T1 extends DTypeTag, T2 extends DTypeTag, R extends DTypeTag>(
  NDArray<T1> arg1,
  NDArray<T2> arg2, {
  NDArray<R>? out,
}) {
  if (arg1.isDisposed || arg2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  NDArray cArr;
  NDArray xArr;
  if (arg1.shape.length != 1 && arg2.shape.length == 1) {
    xArr = arg1;
    cArr = arg2;
  } else {
    cArr = arg1;
    xArr = arg2;
  }
  return _evalClenshaw(cArr, xArr, _OrthoKind.laguerre, out: out);
}

NDArray<R> _evalClenshaw<
  Tc extends DTypeTag,
  Tx extends DTypeTag,
  R extends DTypeTag
>(NDArray<Tc> c, NDArray<Tx> x, _OrthoKind kind, {NDArray<R>? out}) {
  if (c.isDisposed || x.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute series evaluation on a disposed array.");
  }
  if (c.shape.length != 1) {
    throw ArgumentError("Coefficient array c must be 1-dimensional.");
  }
  if (c.shape[0] == 0) {
    throw ArgumentError("Coefficient array c must not be empty.");
  }

  var resolved = resolveDType(c.dtype, x.dtype);
  if (!resolved.isFloating && !resolved.isComplex) {
    resolved = DType.float64;
  }
  final targetDType = resolved as DType<R>;
  if (out != null) {
    if (!listEquals(out.shape, x.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Incompatible out buffer shape or dtype for series evaluation.",
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
NDArray<DTypeTag> chebroots<T extends DTypeTag>(
  NDArray<T> c, {
  NDArray<DTypeTag>? out,
}) {
  return _orthoRoots(c, _OrthoKind.chebyshev, out: out);
}

/// Finds roots of a Legendre series.
NDArray<DTypeTag> legroots<T extends DTypeTag>(
  NDArray<T> c, {
  NDArray<DTypeTag>? out,
}) {
  return _orthoRoots(c, _OrthoKind.legendre, out: out);
}

/// Finds roots of a Hermite series.
NDArray<DTypeTag> hermroots<T extends DTypeTag>(
  NDArray<T> c, {
  NDArray<DTypeTag>? out,
}) {
  return _orthoRoots(c, _OrthoKind.hermite, out: out);
}

/// Finds roots of a Laguerre series.
NDArray<DTypeTag> lagroots<T extends DTypeTag>(
  NDArray<T> c, {
  NDArray<DTypeTag>? out,
}) {
  return _orthoRoots(c, _OrthoKind.laguerre, out: out);
}

NDArray<DTypeTag> _orthoRoots<T extends DTypeTag>(
  NDArray<T> c,
  _OrthoKind kind, {
  NDArray<DTypeTag>? out,
}) {
  if (c.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute root finding on a disposed array.");
  }
  if (c.shape.length != 1) {
    throw ArgumentError("Coefficient array c must be 1-dimensional.");
  }

  final DType<DTypeTag> targetComplexDType = c.dtype == DType.complex64
      ? DType.complex64
      : DType.complex128;

  return NDArray.scope(() {
    var n = c.shape[0] - 1;
    while (n > 0) {
      if (!_isZeroScalar(c.getCellFlat(n) as Object)) break;
      n--;
    }
    final deg = n <= 0 ? 0 : n;
    if (out != null) {
      if (!listEquals(out.shape, [deg]) || out.dtype != targetComplexDType) {
        throw ArgumentError(
          "Incompatible out buffer shape or dtype for roots result (expected shape [$deg] and dtype $targetComplexDType, got shape ${out.shape} and dtype ${out.dtype}).",
        );
      }
      if (!out.isContiguous || sharesMemory(c, out)) {
        final temp = _orthoRoots<T>(c, kind);
        _copyInto(temp, out);
        return out;
      }
    }

    if (n <= 0) {
      final res = NDArray<DTypeTag>.zeros([0], targetComplexDType);
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
      final res = NDArray<DTypeTag>.fromList(
        [complexRoot],
        [1],
        targetComplexDType,
      );
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final bool isComp =
        c.dtype == DType.complex64 || c.dtype == DType.complex128;
    final NDArray cMat;
    switch (c.dtype) {
      case DType.complex64:
      case DType.complex128:
        cMat = NDArray<DTypeTag>.zeros([n, n], c.dtype as DType<DTypeTag>);
        break;
      case DType.float64:
      case DType.float32:
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
    final res = eigvals(cMat as NDArray<AnySpec>, out: out);
    if (out != null) return out;
    return res.detachToParentScope();
  });
}
