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

// Standard polynomial operations (polyval, polyfit, roots).
library;

import "dart:ffi" as ffi;
import "package:openblas/openblas.dart";
import "../../ndarray.dart";
import "../../ndarray_bindings.dart";
import "../../scratch_arena.dart";
import "../helpers.dart";
import "../linalg.dart";

Object _divScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca / cb;
  }
  return (a as num).toDouble() / (b as num).toDouble();
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

/// Evaluates a polynomial with coefficients [c] at points [x].
///
/// If [c] has length N, this function evaluates:
/// p(x) = c[0] x^(N-1) + c[1] x^(N-2) + ... + c[N-1]
///
/// **Preconditions:**
/// - [c] and [x] must not be disposed.
/// - [c] must be a 1-dimensional array.
/// - [c] must not be empty.
/// - It is an error if any input array or [out] buffer is disposed.
/// - It is an error if [c] is not 1-dimensional, or if [c] is empty.
/// - It is an error if [out] shape or dtype is incompatible with [x].
///
/// Reference: [NumPy polyval](https://numpy.org/doc/stable/reference/generated/numpy.polyval.html)
NDArray<R> polyval<
  Tc extends DTypeTag,
  Tx extends DTypeTag,
  R extends DTypeTag
>(NDArray<Tc> c, NDArray<Tx> x, {NDArray<R>? out}) {
  if (c.isDisposed || x.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute polyval() on a disposed array.");
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
  if (out != null) {
    if (!listEquals(out.shape, x.shape) || out.dtype != resolved) {
      throw ArgumentError(
        "Incompatible out buffer shape or dtype for polyval.",
      );
    }
  }
  final targetDType = resolved as DType<R>;

  return NDArray.scope(() {
    final cCast = _ensureDType(c, targetDType);
    final xCast = _ensureDType(x, targetDType);
    final bool needsTemp =
        out != null && (sharesMemory(c, out) || sharesMemory(x, out));
    final res = needsTemp || out == null
        ? NDArray<R>.zeros(x.shape, targetDType)
        : out;

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
            v_polyval_double(
              cCast.pointer.cast(),
              strideC,
              nCoeffs,
              xCast.pointer.cast(),
              res.pointer.cast(),
              totalElements,
            );
          case DType.float32:
            v_polyval_float(
              cCast.pointer.cast(),
              strideC,
              nCoeffs,
              xCast.pointer.cast(),
              res.pointer.cast(),
              totalElements,
            );
          case DType.complex128:
            v_polyval_complex128(
              cCast.pointer.cast(),
              strideC,
              nCoeffs,
              xCast.pointer.cast(),
              res.pointer.cast(),
              totalElements,
            );
          case DType.complex64:
            v_polyval_complex64(
              cCast.pointer.cast(),
              strideC,
              nCoeffs,
              xCast.pointer.cast(),
              res.pointer.cast(),
              totalElements,
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
            throw UnsupportedError(
              "Unsupported dtype $targetDType for polyval.",
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
            s_polyval_double(
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
          case DType.float32:
            s_polyval_float(
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
          case DType.complex128:
            s_polyval_complex128(
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
          case DType.complex64:
            s_polyval_complex64(
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
              "Unsupported dtype $targetDType for polyval.",
            );
        }
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (out != null) {
      if (needsTemp) {
        res.copy(out: out);
      }
      return out;
    }
    return res.detachToParentScope();
  });
}

/// Least-squares fit of a polynomial to data points ([x], [y]).
///
/// Fits a polynomial of degree [deg] to points (x_i, y_i) by minimizing squared error.
/// Optionally weighted by [w].
/// Returns an array of coefficients of length deg + 1, ordered highest degree first.
///
/// **Preconditions:**
/// - [x], [y], and optional [w] must not be disposed.
/// - [x] and [y] must be 1-dimensional arrays of identical length.
/// - Length of [x] must be greater than [deg].
/// - Degree [deg] must be non-negative.
/// - It is an error if any input array or [out] buffer is disposed.
/// - It is an error if input arrays are not 1D, mismatch lengths, or if [deg] is invalid.
/// - It is an error if [out] shape or dtype is incompatible.
///
/// Reference: [NumPy polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)
NDArray<R> polyfit<
  Tx extends DTypeTag,
  Ty extends DTypeTag,
  Tw extends DTypeTag,
  R extends DTypeTag
>(
  NDArray<Tx> x,
  NDArray<Ty> y,
  int deg, {
  NDArray<Tw>? w,
  double? rcond,
  NDArray<R>? out,
}) {
  if (x.isDisposed ||
      y.isDisposed ||
      (w != null && w.isDisposed) ||
      (out != null && out.isDisposed)) {
    throw StateError("Cannot execute polyfit() on a disposed array.");
  }
  if (x.shape.length != 1 || y.shape.length != 1) {
    throw ArgumentError("Input arrays x and y must be 1-dimensional.");
  }
  if (x.shape[0] != y.shape[0]) {
    throw ArgumentError("Input arrays x and y must have equal length.");
  }
  if (deg < 0) {
    throw ArgumentError("Polynomial degree deg must be non-negative.");
  }
  final m = x.shape[0];
  if (m <= deg) {
    throw ArgumentError(
      "Number of data points ($m) must be greater than deg ($deg).",
    );
  }
  if (w != null && (w.shape.length != 1 || w.shape[0] != m)) {
    throw ArgumentError("Weights w must be a 1D array of same length as x.");
  }

  var resolvedType = resolveDType(x.dtype, y.dtype);
  if (w != null) {
    resolvedType = resolveDType(resolvedType, w.dtype);
  }
  if (!resolvedType.isFloating && !resolvedType.isComplex) {
    resolvedType = DType.float64;
  }
  final targetDType = resolvedType as DType<R>;

  if (out != null) {
    if (!listEquals(out.shape, [deg + 1]) || out.dtype != targetDType) {
      throw ArgumentError(
        "Incompatible out buffer shape or dtype for polyfit.",
      );
    }
  }

  return NDArray.scope(() {
    final n = deg + 1;
    final xCast = _ensureDType(x, targetDType);
    final yCast = _ensureDType(y, targetDType);
    final wCast = w != null ? _ensureDType(w, targetDType) : null;

    final vMat = NDArray<R>.create([m, n], targetDType);
    final rhs = NDArray<R>.create([m], targetDType);

    final isContig =
        xCast.isContiguous &&
        yCast.isContiguous &&
        (wCast == null || wCast.isContiguous);

    void fillVander() {
      if (isContig) {
        switch (targetDType) {
          case DType.float64:
            v_vander_fit_double(
              xCast.pointer.cast(),
              yCast.pointer.cast(),
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
            );
          case DType.float32:
            v_vander_fit_float(
              xCast.pointer.cast(),
              yCast.pointer.cast(),
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
            );
          case DType.complex128:
            v_vander_fit_complex128(
              xCast.pointer.cast(),
              yCast.pointer.cast(),
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
            );
          case DType.complex64:
            v_vander_fit_complex64(
              xCast.pointer.cast(),
              yCast.pointer.cast(),
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
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
            throw UnsupportedError(
              "Unsupported dtype $targetDType for polyfit.",
            );
        }
      } else {
        final strideX = xCast.strides.isEmpty ? 1 : xCast.strides[0];
        final strideY = yCast.strides.isEmpty ? 1 : yCast.strides[0];
        final strideW = (wCast == null || wCast.strides.isEmpty)
            ? 1
            : wCast.strides[0];
        switch (targetDType) {
          case DType.float64:
            s_vander_fit_double(
              xCast.pointer.cast(),
              strideX,
              yCast.pointer.cast(),
              strideY,
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              strideW,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
            );
          case DType.float32:
            s_vander_fit_float(
              xCast.pointer.cast(),
              strideX,
              yCast.pointer.cast(),
              strideY,
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              strideW,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
            );
          case DType.complex128:
            s_vander_fit_complex128(
              xCast.pointer.cast(),
              strideX,
              yCast.pointer.cast(),
              strideY,
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              strideW,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
            );
          case DType.complex64:
            s_vander_fit_complex64(
              xCast.pointer.cast(),
              strideX,
              yCast.pointer.cast(),
              strideY,
              wCast != null ? wCast.pointer.cast() : ffi.nullptr,
              strideW,
              vMat.pointer.cast(),
              rhs.pointer.cast(),
              m,
              deg,
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
            throw UnsupportedError(
              "Unsupported dtype $targetDType for polyfit.",
            );
        }
      }
    }

    fillVander();

    final marker = ScratchArena.marker;
    try {
      int info = 0;
      if (rcond == null) {
        // Fast QR solve via dgels/sgels/zgels/cgels
        switch (targetDType) {
          case DType.float64:
            info = LAPACKE_dgels(
              101, // LAPACK_ROW_MAJOR
              78, // 'N'
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Double>(),
              n,
              rhs.pointer.cast<ffi.Double>(),
              1,
            );
          case DType.float32:
            info = LAPACKE_sgels(
              101,
              78,
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Float>(),
              n,
              rhs.pointer.cast<ffi.Float>(),
              1,
            );
          case DType.complex128:
            info = LAPACKE_zgels(
              101,
              78,
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Double>(),
              n,
              rhs.pointer.cast<ffi.Double>(),
              1,
            );
          case DType.complex64:
            info = LAPACKE_cgels(
              101,
              78,
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Float>(),
              n,
              rhs.pointer.cast<ffi.Float>(),
              1,
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
            throw UnsupportedError(
              "Unsupported dtype $targetDType for polyfit.",
            );
        }

        if (info > 0) {
          // Rank deficiency: regenerate matrices and fallback to gelsy
          fillVander();
          final jpvt = ScratchArena.allocate<lapack_int>(
            n * ffi.sizeOf<lapack_int>(),
          );
          for (var k = 0; k < n; k++) {
            jpvt[k] = 0;
          }
          final rankPtr = ScratchArena.allocate<lapack_int>(
            ffi.sizeOf<lapack_int>(),
          );
          switch (targetDType) {
            case DType.float64:
              info = LAPACKE_dgelsy(
                101,
                m,
                n,
                1,
                vMat.pointer.cast<ffi.Double>(),
                n,
                rhs.pointer.cast<ffi.Double>(),
                1,
                jpvt,
                -1.0,
                rankPtr,
              );
            case DType.float32:
              info = LAPACKE_sgelsy(
                101,
                m,
                n,
                1,
                vMat.pointer.cast<ffi.Float>(),
                n,
                rhs.pointer.cast<ffi.Float>(),
                1,
                jpvt,
                -1.0,
                rankPtr,
              );
            case DType.complex128:
              info = LAPACKE_zgelsy(
                101,
                m,
                n,
                1,
                vMat.pointer.cast<ffi.Double>(),
                n,
                rhs.pointer.cast<ffi.Double>(),
                1,
                jpvt,
                -1.0,
                rankPtr,
              );
            case DType.complex64:
              info = LAPACKE_cgelsy(
                101,
                m,
                n,
                1,
                vMat.pointer.cast<ffi.Float>(),
                n,
                rhs.pointer.cast<ffi.Float>(),
                1,
                jpvt,
                -1.0,
                rankPtr,
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
              throw UnsupportedError(
                "Unsupported dtype $targetDType for polyfit.",
              );
          }
        }
      } else {
        // Explicit rcond: complete orthogonal factorization (gelsy)
        final jpvt = ScratchArena.allocate<lapack_int>(
          n * ffi.sizeOf<lapack_int>(),
        );
        for (var k = 0; k < n; k++) {
          jpvt[k] = 0;
        }
        final rankPtr = ScratchArena.allocate<lapack_int>(
          ffi.sizeOf<lapack_int>(),
        );
        switch (targetDType) {
          case DType.float64:
            info = LAPACKE_dgelsy(
              101,
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Double>(),
              n,
              rhs.pointer.cast<ffi.Double>(),
              1,
              jpvt,
              rcond,
              rankPtr,
            );
          case DType.float32:
            info = LAPACKE_sgelsy(
              101,
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Float>(),
              n,
              rhs.pointer.cast<ffi.Float>(),
              1,
              jpvt,
              rcond,
              rankPtr,
            );
          case DType.complex128:
            info = LAPACKE_zgelsy(
              101,
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Double>(),
              n,
              rhs.pointer.cast<ffi.Double>(),
              1,
              jpvt,
              rcond,
              rankPtr,
            );
          case DType.complex64:
            info = LAPACKE_cgelsy(
              101,
              m,
              n,
              1,
              vMat.pointer.cast<ffi.Float>(),
              n,
              rhs.pointer.cast<ffi.Float>(),
              1,
              jpvt,
              rcond,
              rankPtr,
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
            throw UnsupportedError(
              "Unsupported dtype $targetDType for polyfit.",
            );
        }
      }

      if (info < 0) {
        throw ArgumentError("Illegal parameter in LAPACK least-squares: $info");
      }
    } finally {
      ScratchArena.reset(marker);
    }

    final res = out ?? NDArray<R>.create([n], targetDType);
    rhs.slice([Slice(stop: n)]).copy(out: res);

    if (out != null) {
      return out;
    }
    return res.detachToParentScope();
  });
}

/// Computes the roots of a polynomial with coefficients [p].
///
/// The coefficient array [p] is ordered from highest degree to constant term.
/// Returns an `NDArray<DTypeTag>` containing the roots.
///
/// **Preconditions:**
/// - [p] and optional [out] must not be disposed.
/// - [p] must be a 1-dimensional array.
/// - It is an error if [p] or [out] buffer is disposed.
/// - It is an error if [p] is not 1-dimensional.
///
/// Reference: [NumPy roots](https://numpy.org/doc/stable/reference/generated/numpy.roots.html)
NDArray<DTypeTag> roots<T extends DTypeTag>(
  NDArray<T> p, {
  NDArray<DTypeTag>? out,
}) {
  if (p.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute roots() on a disposed array.");
  }
  if (p.shape.length != 1) {
    throw ArgumentError("Coefficient array p must be 1-dimensional.");
  }

  final DType<DTypeTag> targetComplexDType = p.dtype == DType.complex64
      ? DType.complex64
      : DType.complex128;

  return NDArray.scope(() {
    final size = p.shape[0];
    var firstNonZero = -1;
    for (var i = 0; i < size; i++) {
      if (!_isZeroScalar(p.getCellFlat(i) as Object)) {
        firstNonZero = i;
        break;
      }
    }

    final int deg = (firstNonZero == -1 || (size - firstNonZero) <= 1)
        ? 0
        : (size - firstNonZero - 1);

    if (out != null) {
      if (!listEquals(out.shape, [deg]) || out.dtype != targetComplexDType) {
        throw ArgumentError(
          "Incompatible out buffer shape or dtype for roots result (expected shape [$deg] and dtype $targetComplexDType, got shape ${out.shape} and dtype ${out.dtype}).",
        );
      }
      if (!out.isContiguous || sharesMemory(p, out)) {
        final temp = roots<T>(p);
        _copyInto(temp, out);
        return out;
      }
    }

    if (deg == 0) {
      final res = NDArray<DTypeTag>.zeros([0], targetComplexDType);
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    if (deg == 1) {
      final c0 = p.getCellFlat(firstNonZero) as Object;
      final c1 = p.getCellFlat(firstNonZero + 1) as Object;
      final rootVal = _divScalar(_negScalar(c1), c0);
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
        p.dtype == DType.complex64 || p.dtype == DType.complex128;
    final NDArray aMat;
    switch (p.dtype) {
      case DType.complex64:
      case DType.complex128:
        aMat = NDArray<DTypeTag>.zeros([deg, deg], p.dtype as DType<DTypeTag>);
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
        aMat = NDArray<Float64>.zeros([deg, deg], DType.float64);
        break;
    }

    final c0 = p.getCellFlat(firstNonZero) as Object;
    final targetMatDType = aMat.dtype;

    for (var j = 0; j < deg; j++) {
      final cj = p.getCellFlat(firstNonZero + j + 1) as Object;
      final val = _divScalar(_negScalar(cj), c0);
      aMat.setCellFlat(j, castValue(val, targetMatDType));
    }
    for (var i = 1; i < deg; i++) {
      final one = isComp ? Complex(1.0, 0.0) : 1.0;
      aMat.setCellFlat(i * deg + i - 1, castValue(one, targetMatDType));
    }

    final res = eigvals(aMat as NDArray<AnySpec>, out: out);
    if (out != null) return out;
    return res.detachToParentScope();
  });
}
