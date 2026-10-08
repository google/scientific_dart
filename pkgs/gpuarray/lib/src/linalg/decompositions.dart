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

import '../dtype.dart';
import '../gpu_array.dart';
import 'decomposition_kernels.dart';
import 'linalg_buffer_ops.dart';
import 'linalg_wgsl_df64.dart';
import 'spectral_kernels.dart';

/// Mode for [qr] decomposition.
enum QrMode {
  /// Returns `Q` of shape `[M, K]` and `R` of shape `[K, N]`, where
  /// `K = min(M, N)`.
  reduced,

  /// Returns `Q` of shape `[M, M]` and `R` of shape `[M, N]`.
  complete,

  /// Computes the upper-triangular `R` factor of shape `[K, N]` where
  /// `K = min(M, N)`.
  r,
}

/// Triangle selection for symmetric/Hermitian matrix operations ([eigh],
/// [eigvalsh]).
enum UpLo {
  /// Uses the lower-triangular part of the matrix.
  lower,

  /// Uses the upper-triangular part of the matrix.
  upper,
}

/// Alias for [UpLo] triangle selection.
typedef MatrixTriangle = UpLo;

/// Convenience disposal extension for [svd] output records.
extension SvdRecordDispose<U extends DTypeTag, S extends DTypeTag>
    on ({GpuArray<U> u, GpuArray<S> s, GpuArray<U> vt}) {
  /// Disposes all [GpuArray] fields (`u`, `s`, `vt`) in this record.
  void dispose() {
    u.dispose();
    s.dispose();
    vt.dispose();
  }
}

/// Convenience disposal extension for [qr] output records.
extension QrRecordDispose<T extends DTypeTag>
    on ({GpuArray<T> q, GpuArray<T> r}) {
  /// Disposes both [GpuArray] fields (`q`, `r`) in this record.
  void dispose() {
    q.dispose();
    r.dispose();
  }
}

/// Convenience disposal extension for [eigh] output records.
extension EighRecordDispose<W extends DTypeTag, V extends DTypeTag>
    on ({GpuArray<W> eigenvalues, GpuArray<V> eigenvectors}) {
  /// Disposes both [GpuArray] fields (`eigenvalues`, `eigenvectors`) in this
  /// record.
  void dispose() {
    eigenvalues.dispose();
    eigenvectors.dispose();
  }
}

/// Convenience disposal extension for [eig] output records.
extension EigRecordDispose<T extends AnySpec>
    on ({GpuArray<T> eigenvalues, GpuArray<T> eigenvectors}) {
  /// Disposes both [GpuArray] fields (`eigenvalues`, `eigenvectors`) in this
  /// record.
  void dispose() {
    eigenvalues.dispose();
    eigenvectors.dispose();
  }
}

/// Convenience disposal extension for [lu] output records.
extension LuRecordDispose<T extends DTypeTag>
    on ({GpuArray<T> p, GpuArray<T> l, GpuArray<T> u}) {
  /// Disposes all [GpuArray] fields (`p`, `l`, `u`) in this record.
  void dispose() {
    p.dispose();
    l.dispose();
    u.dispose();
  }
}

/// Convenience disposal extension for [luFactor] output records.
extension LuFactorRecordDispose<T extends DTypeTag, P extends DTypeTag>
    on ({GpuArray<T> lu, GpuArray<P> pivots}) {
  /// Disposes both [GpuArray] fields (`lu`, `pivots`) in this record.
  void dispose() {
    this.lu.dispose();
    pivots.dispose();
  }
}

void _require2d(GpuArray a, String functionName) {
  if (a.isDisposed) {
    throw StateError('Cannot execute $functionName on a disposed GpuArray.');
  }
  if (a.ndim != 2) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must be a 2-D matrix for $functionName, got ${a.ndim}-D.',
    );
  }
}

void _requireSquare2d(GpuArray a, String functionName) {
  _require2d(a, functionName);
  if (a.shape[0] != a.shape[1]) {
    throw ArgumentError.value(
      a.shape,
      'a',
      'Must be a square 2-D matrix for $functionName, got ${a.shape}.',
    );
  }
}

/// Singular Value Decomposition (`U`, `S`, `Vt`) of a 2-D matrix [a].
///
/// Factorizes an `[M, N]` matrix [a] into `U * diag(S) * Vt`, where:
/// - `u` is an orthogonal matrix of shape `[M, M]` (when [fullMatrices] is
///   `true`) or `[M, K]` (when [fullMatrices] is `false`), with `K = min(M, N)`.
/// - `s` is a 1-D array of shape `[K]` containing the singular values in
///   descending order.
/// - `vt` is an orthogonal matrix of shape `[N, N]` (when [fullMatrices] is
///   `true`) or `[K, N]` (when [fullMatrices] is `false`).
///
/// The [a] tensor must be a 2-D matrix. Optional [outU], [outS], and [outVt]
/// arrays must match the expected shapes, output dtypes, and device of [a].
({GpuArray<M> u, GpuArray<F> s, GpuArray<M> vt}) svd<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  bool fullMatrices = true,
  GpuArray<M>? outU,
  GpuArray<F>? outS,
  GpuArray<M>? outVt,
}) {
  _require2d(a, 'svd');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final uShape = fullMatrices ? <int>[m, m] : <int>[m, k];
  final sShape = <int>[k];
  final vtShape = fullMatrices ? <int>[n, n] : <int>[k, n];
  final mathDtype = linalgMathDType(a.dtype);
  final floatDtype = linalgFloatDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);

  validateLinalgOut(outU, a.device, uShape, mathDtype, paramName: 'outU');
  validateLinalgOut(outS, a.device, sShape, floatDtype, paramName: 'outS');
  validateLinalgOut(outVt, a.device, vtShape, mathDtype, paramName: 'outVt');

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchSvdGpu(
      a.device,
      inputBuffer,
      m,
      n,
      fullMatrices: fullMatrices,
      singlePrecision: single,
    );
    final uArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.u,
            uShape,
            mathDtype,
            out: outU,
            outParamName: 'outU',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.u,
            uShape,
            mathDtype,
            out: outU,
            outParamName: 'outU',
          );
    final sArray = single
        ? writeFloat32BufferToArray<F>(
            a.device,
            buffers.s,
            sShape,
            floatDtype,
            out: outS,
            outParamName: 'outS',
          )
        : writeFloat64BufferToArray<F>(
            a.device,
            buffers.s,
            sShape,
            floatDtype,
            out: outS,
            outParamName: 'outS',
          );
    final vtArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.vt,
            vtShape,
            mathDtype,
            out: outVt,
            outParamName: 'outVt',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.vt,
            vtShape,
            mathDtype,
            out: outVt,
            outParamName: 'outVt',
          );
    if (outU == null) uArray.detachToParentScope();
    if (outS == null) sArray.detachToParentScope();
    if (outVt == null) vtArray.detachToParentScope();
    return (u: uArray, s: sArray, vt: vtArray);
  });
}

/// Singular values of a 2-D matrix [a] in descending order.
///
/// For an `[M, N]` matrix [a], produces a 1-D [GpuArray] of shape `[min(M, N)]`.
///
/// The [a] tensor must be a 2-D matrix. If [out] is provided, it must have
/// shape `[min(M, N)]`, matching float dtype, and reside on `a.device`.
GpuArray<F> svdValues<F extends DTypeTag>(
  GpuArray<RealFloatOf<F>> a, {
  GpuArray<F>? out,
}) {
  _require2d(a, 'svdValues');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final sShape = <int>[k];
  final floatDtype = linalgFloatDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);
  validateLinalgOut(out, a.device, sShape, floatDtype);

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchSvdGpu(
      a.device,
      inputBuffer,
      m,
      n,
      fullMatrices: false,
      singlePrecision: single,
    );
    final sArray = single
        ? writeFloat32BufferToArray<F>(
            a.device,
            buffers.s,
            sShape,
            floatDtype,
            out: out,
          )
        : writeFloat64BufferToArray<F>(
            a.device,
            buffers.s,
            sShape,
            floatDtype,
            out: out,
          );
    if (out == null) sArray.detachToParentScope();
    return sArray;
  });
}

/// QR factorization of a 2-D matrix [a].
///
/// Factorizes an `[M, N]` matrix [a] into `Q * R`, where `Q` has orthonormal
/// columns and `R` is upper-triangular:
/// - For [QrMode.reduced] or [QrMode.r], `q` has shape `[M, K]` and `r` has
///   shape `[K, N]`, where `K = min(M, N)`.
/// - For [QrMode.complete], `q` has shape `[M, M]` and `r` has shape `[M, N]`.
///
/// The [a] tensor must be a 2-D matrix. Optional [outQ] and [outR] arrays must
/// match the expected shapes, output dtype, and reside on `a.device`.
({GpuArray<M> q, GpuArray<M> r}) qr<M extends DTypeTag>(
  GpuArray<InexactOf<M>> a, {
  QrMode mode = QrMode.reduced,
  GpuArray<M>? outQ,
  GpuArray<M>? outR,
}) {
  _require2d(a, 'qr');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final qCols = mode == QrMode.complete ? m : k;
  final rRows = mode == QrMode.complete ? m : k;
  final qShape = <int>[m, qCols];
  final rShape = <int>[rRows, n];
  final mathDtype = linalgMathDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);

  validateLinalgOut(outQ, a.device, qShape, mathDtype, paramName: 'outQ');
  validateLinalgOut(outR, a.device, rShape, mathDtype, paramName: 'outR');

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchQrGpu(
      a.device,
      inputBuffer,
      m,
      n,
      qCols: qCols,
      rRows: rRows,
      singlePrecision: single,
    );
    final qArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.q,
            qShape,
            mathDtype,
            out: outQ,
            outParamName: 'outQ',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.q,
            qShape,
            mathDtype,
            out: outQ,
            outParamName: 'outQ',
          );
    final rArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.r,
            rShape,
            mathDtype,
            out: outR,
            outParamName: 'outR',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.r,
            rShape,
            mathDtype,
            out: outR,
            outParamName: 'outR',
          );
    if (outQ == null) qArray.detachToParentScope();
    if (outR == null) rArray.detachToParentScope();
    return (q: qArray, r: rArray);
  });
}

/// Upper-triangular `R` factor of the QR decomposition of a 2-D matrix [a].
///
/// For an `[M, N]` matrix [a], produces an upper-triangular [GpuArray] of shape
/// `[min(M, N), N]`.
///
/// The [a] tensor must be a 2-D matrix. If [out] is provided, it must have
/// shape `[min(M, N), N]`, matching output dtype, and reside on `a.device`.
GpuArray<M> qrR<M extends DTypeTag>(
  GpuArray<InexactOf<M>> a, {
  GpuArray<M>? out,
}) {
  _require2d(a, 'qrR');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final rShape = <int>[k, n];
  final mathDtype = linalgMathDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);
  validateLinalgOut(out, a.device, rShape, mathDtype);

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchQrGpu(
      a.device,
      inputBuffer,
      m,
      n,
      qCols: k,
      rRows: k,
      singlePrecision: single,
    );
    final rArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.r,
            rShape,
            mathDtype,
            out: out,
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.r,
            rShape,
            mathDtype,
            out: out,
          );
    if (out == null) rArray.detachToParentScope();
    return rArray;
  });
}

/// Cholesky decomposition of a symmetric positive-definite 2-D matrix [a].
///
/// Factorizes [a] into `L * L^T` (when [upper] is `false` or [uplo] is
/// [UpLo.lower]) or `U^T * U` (when [upper] is `true` or [uplo] is
/// [UpLo.upper]), returning the triangular factor of shape `[N, N]`.
///
/// The [a] tensor must be a square 2-D matrix. If [out] is provided, it must
/// have shape `[N, N]`, matching output dtype, and reside on `a.device`.
GpuArray<M> cholesky<M extends DTypeTag>(
  GpuArray<InexactOf<M>> a, {
  bool upper = false,
  UpLo? uplo,
  GpuArray<M>? out,
}) {
  _requireSquare2d(a, 'cholesky');
  final n = a.shape[0];
  final mathDtype = linalgMathDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);
  validateLinalgOut(out, a.device, a.shape, mathDtype);
  final isUpper = uplo != null ? uplo == UpLo.upper : upper;

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final outBuffer = dispatchCholeskyGpu(
      a.device,
      inputBuffer,
      n,
      upper: isUpper,
      singlePrecision: single,
    );
    final result = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            outBuffer,
            a.shape,
            mathDtype,
            out: out,
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            outBuffer,
            a.shape,
            mathDtype,
            out: out,
          );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Eigenvalues and eigenvectors of a real symmetric 2-D matrix [a].
///
/// Produces `eigenvalues` of shape `[N]` in ascending order and `eigenvectors`
/// of shape `[N, N]` whose columns are the corresponding orthonormal
/// eigenvectors (`a * V = V * diag(w)`).
///
/// The [a] tensor must be a square 2-D matrix. Optional [outEigenvalues] and
/// [outEigenvectors] arrays must match shapes `[N]` and `[N, N]` and output
/// dtypes on `a.device`.
({GpuArray<F> eigenvalues, GpuArray<M> eigenvectors}) eigh<
  R extends DTypeTag,
  E,
  F extends DTypeTag,
  C extends DTypeTag,
  M extends DTypeTag,
  S extends DTypeTag,
  D extends DTypeTag
>(
  GpuArray<DTypeSpec<R, E, F, C, M, S, D, DTypeTag>> a, {
  UpLo uplo = UpLo.lower,
  GpuArray<F>? outEigenvalues,
  GpuArray<M>? outEigenvectors,
}) {
  _requireSquare2d(a, 'eigh');
  final n = a.shape[0];
  final wShape = <int>[n];
  final vShape = <int>[n, n];
  final floatDtype = linalgFloatDType(a.dtype);
  final mathDtype = linalgMathDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);
  validateLinalgOut(
    outEigenvalues,
    a.device,
    wShape,
    floatDtype,
    paramName: 'outEigenvalues',
  );
  validateLinalgOut(
    outEigenvectors,
    a.device,
    vShape,
    mathDtype,
    paramName: 'outEigenvectors',
  );

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchEighGpu(
      a.device,
      inputBuffer,
      n,
      useUpper: uplo == UpLo.upper,
      singlePrecision: single,
    );
    final wArray = single
        ? writeFloat32BufferToArray<F>(
            a.device,
            buffers.eigenvalues,
            wShape,
            floatDtype,
            out: outEigenvalues,
            outParamName: 'outEigenvalues',
          )
        : writeFloat64BufferToArray<F>(
            a.device,
            buffers.eigenvalues,
            wShape,
            floatDtype,
            out: outEigenvalues,
            outParamName: 'outEigenvalues',
          );
    final vArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.eigenvectors,
            vShape,
            mathDtype,
            out: outEigenvectors,
            outParamName: 'outEigenvectors',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.eigenvectors,
            vShape,
            mathDtype,
            out: outEigenvectors,
            outParamName: 'outEigenvectors',
          );
    if (outEigenvalues == null) wArray.detachToParentScope();
    if (outEigenvectors == null) vArray.detachToParentScope();
    return (eigenvalues: wArray, eigenvectors: vArray);
  });
}

/// Eigenvalues of a real symmetric 2-D matrix [a] in ascending order.
///
/// The [a] tensor must be a square 2-D matrix. If [out] is provided, it must
/// have shape `[N]`, matching float dtype, and reside on `a.device`.
GpuArray<F> eigvalsh<F extends DTypeTag>(
  GpuArray<RealFloatOf<F>> a, {
  UpLo uplo = UpLo.lower,
  GpuArray<F>? out,
}) {
  _requireSquare2d(a, 'eigvalsh');
  final n = a.shape[0];
  final wShape = <int>[n];
  final floatDtype = linalgFloatDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);
  validateLinalgOut(out, a.device, wShape, floatDtype);

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchEighGpu(
      a.device,
      inputBuffer,
      n,
      useUpper: uplo == UpLo.upper,
      singlePrecision: single,
    );
    final wArray = single
        ? writeFloat32BufferToArray<F>(
            a.device,
            buffers.eigenvalues,
            wShape,
            floatDtype,
            out: out,
          )
        : writeFloat64BufferToArray<F>(
            a.device,
            buffers.eigenvalues,
            wShape,
            floatDtype,
            out: out,
          );
    if (out == null) wArray.detachToParentScope();
    return wArray;
  });
}

/// Eigenvalues and right eigenvectors of a general square 2-D matrix [a].
///
/// Produces complex `eigenvalues` of shape `[N]` and normalized right
/// `eigenvectors` of shape `[N, N]` satisfying `a * v[:, i] = w[i] * v[:, i]`.
///
/// The [a] tensor must be a square 2-D matrix. Optional [outEigenvalues] and
/// [outEigenvectors] arrays must have matching complex dtype and shapes `[N]`
/// and `[N, N]` on `a.device`.
({GpuArray<C> eigenvalues, GpuArray<C> eigenvectors}) eig<C extends DTypeTag>(
  GpuArray<ComplexOf<C>> a, {
  GpuArray<C>? outEigenvalues,
  GpuArray<C>? outEigenvectors,
}) {
  _requireSquare2d(a, 'eig');
  final n = a.shape[0];
  final wShape = <int>[n];
  final vShape = <int>[n, n];
  final complexDtype = linalgComplexDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);
  validateLinalgOut(
    outEigenvalues,
    a.device,
    wShape,
    complexDtype,
    paramName: 'outEigenvalues',
  );
  validateLinalgOut(
    outEigenvectors,
    a.device,
    vShape,
    complexDtype,
    paramName: 'outEigenvectors',
  );

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchEigGpu(
      a.device,
      inputBuffer,
      n,
      computeVectors: true,
      singlePrecision: single,
    );
    final wArray = single
        ? writeComplex64BufferToArray<C>(
            a.device,
            buffers.eigenvalues,
            wShape,
            complexDtype,
            out: outEigenvalues,
            outParamName: 'outEigenvalues',
          )
        : writeComplex128BufferToArray<C>(
            a.device,
            buffers.eigenvalues,
            wShape,
            complexDtype,
            out: outEigenvalues,
            outParamName: 'outEigenvalues',
          );
    final vArray = single
        ? writeComplex64BufferToArray<C>(
            a.device,
            buffers.eigenvectors,
            vShape,
            complexDtype,
            out: outEigenvectors,
            outParamName: 'outEigenvectors',
          )
        : writeComplex128BufferToArray<C>(
            a.device,
            buffers.eigenvectors,
            vShape,
            complexDtype,
            out: outEigenvectors,
            outParamName: 'outEigenvectors',
          );
    if (outEigenvalues == null) wArray.detachToParentScope();
    if (outEigenvectors == null) vArray.detachToParentScope();
    return (eigenvalues: wArray, eigenvectors: vArray);
  });
}

/// Eigenvalues of a general square 2-D matrix [a].
///
/// Produces a 1-D [GpuArray] of shape `[N]` with complex dtype (`Complex64`
/// for single-precision inputs, `Complex128` otherwise).
///
/// The [a] tensor must be a square 2-D matrix. If [out] is provided, it must
/// have shape `[N]`, matching complex dtype, and reside on `a.device`.
GpuArray<C> eigvals<C extends DTypeTag>(
  GpuArray<ComplexOf<C>> a, {
  GpuArray<C>? out,
}) {
  _requireSquare2d(a, 'eigvals');
  final n = a.shape[0];
  final wShape = <int>[n];
  final complexDtype = linalgComplexDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);
  validateLinalgOut(out, a.device, wShape, complexDtype);

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchEigGpu(
      a.device,
      inputBuffer,
      n,
      computeVectors: false,
      singlePrecision: single,
    );
    final wArray = single
        ? writeComplex64BufferToArray<C>(
            a.device,
            buffers.eigenvalues,
            wShape,
            complexDtype,
            out: out,
          )
        : writeComplex128BufferToArray<C>(
            a.device,
            buffers.eigenvalues,
            wShape,
            complexDtype,
            out: out,
          );
    if (out == null) wArray.detachToParentScope();
    return wArray;
  });
}

/// Pivoted LU decomposition of a 2-D matrix [a].
///
/// Factorizes an `[M, N]` matrix [a] into `P * L * U`, where `P` is an
/// `[M, M]` permutation matrix, `L` is an `[M, K]` unit lower-triangular
/// matrix, and `U` is a `[K, N]` upper-triangular matrix, with `K = min(M, N)`.
///
/// The [a] tensor must be a 2-D matrix. Optional [outP], [outL], and [outU]
/// arrays must match the expected shapes and output dtype on `a.device`.
({GpuArray<M> p, GpuArray<M> l, GpuArray<M> u}) lu<M extends DTypeTag>(
  GpuArray<InexactOf<M>> a, {
  GpuArray<M>? outP,
  GpuArray<M>? outL,
  GpuArray<M>? outU,
}) {
  _require2d(a, 'lu');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final pShape = <int>[m, m];
  final lShape = <int>[m, k];
  final uShape = <int>[k, n];
  final mathDtype = linalgMathDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);

  validateLinalgOut(outP, a.device, pShape, mathDtype, paramName: 'outP');
  validateLinalgOut(outL, a.device, lShape, mathDtype, paramName: 'outL');
  validateLinalgOut(outU, a.device, uShape, mathDtype, paramName: 'outU');

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchLuGpu(
      a.device,
      inputBuffer,
      m,
      n,
      singlePrecision: single,
    );
    final pArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.p,
            pShape,
            mathDtype,
            out: outP,
            outParamName: 'outP',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.p,
            pShape,
            mathDtype,
            out: outP,
            outParamName: 'outP',
          );
    final lArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.l,
            lShape,
            mathDtype,
            out: outL,
            outParamName: 'outL',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.l,
            lShape,
            mathDtype,
            out: outL,
            outParamName: 'outL',
          );
    final uArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.u,
            uShape,
            mathDtype,
            out: outU,
            outParamName: 'outU',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.u,
            uShape,
            mathDtype,
            out: outU,
            outParamName: 'outU',
          );
    if (outP == null) pArray.detachToParentScope();
    if (outL == null) lArray.detachToParentScope();
    if (outU == null) uArray.detachToParentScope();
    return (p: pArray, l: lArray, u: uArray);
  });
}

/// Compact LU factorization of a square 2-D matrix [a] for use with [luSolve].
///
/// Produces a combined `lu` matrix of shape `[N, N]` containing the unit
/// lower-triangular `L` (below the diagonal) and upper-triangular `U` (on and
/// above the diagonal), together with a 1-D `pivots` array of shape `[N]`
/// ([DType.int32]).
///
/// The [a] tensor must be a square 2-D matrix. Optional [outLu] and [outPivots]
/// arrays must match the expected shapes and dtypes on `a.device`.
({GpuArray<M> lu, GpuArray<Int32> pivots}) luFactor<M extends DTypeTag>(
  GpuArray<InexactOf<M>> a, {
  GpuArray<M>? outLu,
  GpuArray<Int32>? outPivots,
}) {
  _requireSquare2d(a, 'luFactor');
  final n = a.shape[0];
  final luShape = <int>[n, n];
  final pivShape = <int>[n];
  final mathDtype = linalgMathDType(a.dtype);
  final single = isSinglePrecisionDType(a.dtype);

  validateLinalgOut(outLu, a.device, luShape, mathDtype, paramName: 'outLu');
  validateLinalgOut(
    outPivots,
    a.device,
    pivShape,
    DType.int32,
    paramName: 'outPivots',
  );

  return ResourceScope.scope(() {
    final inputBuffer = single
        ? toContiguousFloat32Buffer(a)
        : toContiguousFloat64Buffer(a);
    final buffers = dispatchLuGpu(
      a.device,
      inputBuffer,
      n,
      n,
      singlePrecision: single,
    );
    final luArray = single
        ? writeFloat32BufferToArray<M>(
            a.device,
            buffers.lu,
            luShape,
            mathDtype,
            out: outLu,
            outParamName: 'outLu',
          )
        : writeFloat64BufferToArray<M>(
            a.device,
            buffers.lu,
            luShape,
            mathDtype,
            out: outLu,
            outParamName: 'outLu',
          );
    final pivArray = single
        ? writeFloat32BufferToArray<Int32>(
            a.device,
            buffers.pivots,
            pivShape,
            DType.int32,
            out: outPivots,
            outParamName: 'outPivots',
          )
        : writeFloat64BufferToArray<Int32>(
            a.device,
            buffers.pivots,
            pivShape,
            DType.int32,
            out: outPivots,
            outParamName: 'outPivots',
          );
    if (outLu == null) luArray.detachToParentScope();
    if (outPivots == null) pivArray.detachToParentScope();
    return (lu: luArray, pivots: pivArray);
  });
}

/// Solves a linear system `A * x = b` given the [luFactor] decomposition
/// `(` [lu], [pivots] `)` and right-hand side [b].
///
/// The [lu] tensor must be a square 2-D matrix of shape `[N, N]`, [pivots] must
/// be a 1-D array of length `N`, and [b] must be a 1-D (`[N]`) or 2-D
/// (`[N, K]`) array on the same device.
GpuArray<M> luSolve<M extends DTypeTag>(
  GpuArray<InexactOf<M>> lu,
  GpuArray<DTypeTag> pivots,
  GpuArray<DTypeTag> b, {
  GpuArray<M>? out,
}) {
  _requireSquare2d(lu, 'luSolve');
  if (pivots.isDisposed) {
    throw StateError('Cannot execute luSolve with a disposed pivots GpuArray.');
  }
  if (b.isDisposed) {
    throw StateError('Cannot execute luSolve with a disposed b GpuArray.');
  }
  if (pivots.device != lu.device) {
    throw ArgumentError.value(
      pivots,
      'pivots',
      'Must reside on the same GpuDevice as lu.',
    );
  }
  if (b.device != lu.device) {
    throw ArgumentError.value(
      b,
      'b',
      'Must reside on the same GpuDevice as lu.',
    );
  }
  final n = lu.shape[0];
  if (pivots.ndim != 1 || pivots.shape[0] != n) {
    throw ArgumentError.value(
      pivots.shape,
      'pivots',
      'Must be a 1-D array of length $n, got ${pivots.shape}.',
    );
  }
  if (b.ndim != 1 && b.ndim != 2) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must be a 1-D or 2-D array for luSolve, got ${b.ndim}-D.',
    );
  }
  if (b.shape[0] != n) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must have first dimension $n matching lu, got ${b.shape[0]}.',
    );
  }

  final mathDtype = linalgMathDType(lu.dtype);
  final single = isSinglePrecisionDType(lu.dtype);
  validateLinalgOut(out, lu.device, b.shape, mathDtype);
  final nrhs = b.ndim == 1 ? 1 : b.shape[1];

  return ResourceScope.scope(() {
    final luBuffer = single
        ? toContiguousFloat32Buffer(lu)
        : toContiguousFloat64Buffer(lu);
    final pivotsBuffer = single
        ? toContiguousFloat32Buffer(pivots)
        : toContiguousFloat64Buffer(pivots);
    final bBuffer = single
        ? toContiguousFloat32Buffer(b)
        : toContiguousFloat64Buffer(b);
    final xBuffer = dispatchLuSolveGpu(
      lu.device,
      luBuffer,
      pivotsBuffer,
      bBuffer,
      n,
      nrhs,
      singlePrecision: single,
    );
    final result = single
        ? writeFloat32BufferToArray<M>(
            lu.device,
            xBuffer,
            b.shape,
            mathDtype,
            out: out,
          )
        : writeFloat64BufferToArray<M>(
            lu.device,
            xBuffer,
            b.shape,
            mathDtype,
            out: out,
          );
    if (out == null) result.detachToParentScope();
    return result;
  });
}
