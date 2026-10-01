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
extension SvdRecordDispose
    on ({GpuArray<Float64> u, GpuArray<Float64> s, GpuArray<Float64> vt}) {
  /// Disposes all [GpuArray] fields (`u`, `s`, `vt`) in this record.
  void dispose() {
    u.dispose();
    s.dispose();
    vt.dispose();
  }
}

/// Convenience disposal extension for [qr] output records.
extension QrRecordDispose on ({GpuArray<Float64> q, GpuArray<Float64> r}) {
  /// Disposes both [GpuArray] fields (`q`, `r`) in this record.
  void dispose() {
    q.dispose();
    r.dispose();
  }
}

/// Convenience disposal extension for [eigh] output records.
extension EighRecordDispose
    on ({GpuArray<Float64> eigenvalues, GpuArray<Float64> eigenvectors}) {
  /// Disposes both [GpuArray] fields (`eigenvalues`, `eigenvectors`) in this
  /// record.
  void dispose() {
    eigenvalues.dispose();
    eigenvectors.dispose();
  }
}

/// Convenience disposal extension for [eig] output records.
extension EigRecordDispose
    on ({GpuArray<Complex128> eigenvalues, GpuArray<Complex128> eigenvectors}) {
  /// Disposes both [GpuArray] fields (`eigenvalues`, `eigenvectors`) in this
  /// record.
  void dispose() {
    eigenvalues.dispose();
    eigenvectors.dispose();
  }
}

/// Convenience disposal extension for [lu] output records.
extension LuRecordDispose
    on ({GpuArray<Float64> p, GpuArray<Float64> l, GpuArray<Float64> u}) {
  /// Disposes all [GpuArray] fields (`p`, `l`, `u`) in this record.
  void dispose() {
    p.dispose();
    l.dispose();
    u.dispose();
  }
}

/// Convenience disposal extension for [luFactor] output records.
extension LuFactorRecordDispose<P extends DTypeTag>
    on ({GpuArray<Float64> lu, GpuArray<P> pivots}) {
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
/// arrays must match the expected shapes, `Float64` dtype, and device of [a].
({GpuArray<Float64> u, GpuArray<Float64> s, GpuArray<Float64> vt}) svd(
  GpuArray<DTypeTag> a, {
  bool fullMatrices = true,
  GpuArray<Float64>? outU,
  GpuArray<Float64>? outS,
  GpuArray<Float64>? outVt,
}) {
  _require2d(a, 'svd');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final uShape = fullMatrices ? <int>[m, m] : <int>[m, k];
  final sShape = <int>[k];
  final vtShape = fullMatrices ? <int>[n, n] : <int>[k, n];

  validateLinalgOut(outU, a.device, uShape, DType.float64, paramName: 'outU');
  validateLinalgOut(outS, a.device, sShape, DType.float64, paramName: 'outS');
  validateLinalgOut(
    outVt,
    a.device,
    vtShape,
    DType.float64,
    paramName: 'outVt',
  );

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchSvdGpu(
      a.device,
      inputF64,
      m,
      n,
      fullMatrices: fullMatrices,
    );
    final uArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.u,
      uShape,
      DType.float64,
      out: outU,
      outParamName: 'outU',
    );
    final sArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.s,
      sShape,
      DType.float64,
      out: outS,
      outParamName: 'outS',
    );
    final vtArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.vt,
      vtShape,
      DType.float64,
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
/// For an `[M, N]` matrix [a], produces a 1-D [GpuArray] of shape `[min(M, N)]`
/// with [DType.float64].
///
/// The [a] tensor must be a 2-D matrix. If [out] is provided, it must have
/// shape `[min(M, N)]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> svdvals(GpuArray<DTypeTag> a, {GpuArray<Float64>? out}) {
  _require2d(a, 'svdvals');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final sShape = <int>[k];
  validateLinalgOut(out, a.device, sShape, DType.float64);

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchSvdGpu(
      a.device,
      inputF64,
      m,
      n,
      fullMatrices: false,
    );
    final sArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.s,
      sShape,
      DType.float64,
      out: out,
    );
    if (out == null) sArray.detachToParentScope();
    return sArray;
  });
}

/// Alias for [svdvals] computing the singular values of [a].
///
/// The [a] tensor must be a 2-D matrix. If [out] is provided, it must have
/// shape `[min(M, N)]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> svdValues(GpuArray<DTypeTag> a, {GpuArray<Float64>? out}) =>
    svdvals(a, out: out);

/// QR factorization of a 2-D matrix [a].
///
/// Factorizes an `[M, N]` matrix [a] into `Q * R`, where `Q` has orthonormal
/// columns and `R` is upper-triangular:
/// - For [QrMode.reduced] or [QrMode.r], `q` has shape `[M, K]` and `r` has
///   shape `[K, N]`, where `K = min(M, N)`.
/// - For [QrMode.complete], `q` has shape `[M, M]` and `r` has shape `[M, N]`.
///
/// The [a] tensor must be a 2-D matrix. Optional [outQ] and [outR] arrays must
/// match the expected shapes, [DType.float64], and reside on `a.device`.
({GpuArray<Float64> q, GpuArray<Float64> r}) qr(
  GpuArray<DTypeTag> a, {
  QrMode mode = QrMode.reduced,
  GpuArray<Float64>? outQ,
  GpuArray<Float64>? outR,
}) {
  _require2d(a, 'qr');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final qCols = mode == QrMode.complete ? m : k;
  final rRows = mode == QrMode.complete ? m : k;
  final qShape = <int>[m, qCols];
  final rShape = <int>[rRows, n];

  validateLinalgOut(outQ, a.device, qShape, DType.float64, paramName: 'outQ');
  validateLinalgOut(outR, a.device, rShape, DType.float64, paramName: 'outR');

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchQrGpu(
      a.device,
      inputF64,
      m,
      n,
      qCols: qCols,
      rRows: rRows,
    );
    final qArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.q,
      qShape,
      DType.float64,
      out: outQ,
      outParamName: 'outQ',
    );
    final rArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.r,
      rShape,
      DType.float64,
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
/// `[min(M, N), N]` with [DType.float64].
///
/// The [a] tensor must be a 2-D matrix. If [out] is provided, it must have
/// shape `[min(M, N), N]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> qrR(GpuArray<DTypeTag> a, {GpuArray<Float64>? out}) {
  _require2d(a, 'qrR');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final rShape = <int>[k, n];
  validateLinalgOut(out, a.device, rShape, DType.float64);

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchQrGpu(a.device, inputF64, m, n, qCols: k, rRows: k);
    final rArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.r,
      rShape,
      DType.float64,
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
/// have shape `[N, N]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> cholesky(
  GpuArray<DTypeTag> a, {
  bool upper = false,
  UpLo? uplo,
  GpuArray<Float64>? out,
}) {
  _requireSquare2d(a, 'cholesky');
  final n = a.shape[0];
  validateLinalgOut(out, a.device, a.shape, DType.float64);
  final isUpper = uplo != null ? uplo == UpLo.upper : upper;

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final outBuffer = dispatchCholeskyGpu(
      a.device,
      inputF64,
      n,
      upper: isUpper,
    );
    final result = writeFloat64BufferToArray<Float64>(
      a.device,
      outBuffer,
      a.shape,
      DType.float64,
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
/// [outEigenvectors] arrays must match shapes `[N]` and `[N, N]` with
/// [DType.float64] on `a.device`.
({GpuArray<Float64> eigenvalues, GpuArray<Float64> eigenvectors}) eigh(
  GpuArray<DTypeTag> a, {
  UpLo uplo = UpLo.lower,
  GpuArray<Float64>? outEigenvalues,
  GpuArray<Float64>? outEigenvectors,
}) {
  _requireSquare2d(a, 'eigh');
  final n = a.shape[0];
  final wShape = <int>[n];
  final vShape = <int>[n, n];
  validateLinalgOut(
    outEigenvalues,
    a.device,
    wShape,
    DType.float64,
    paramName: 'outEigenvalues',
  );
  validateLinalgOut(
    outEigenvectors,
    a.device,
    vShape,
    DType.float64,
    paramName: 'outEigenvectors',
  );

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchEighGpu(
      a.device,
      inputF64,
      n,
      useUpper: uplo == UpLo.upper,
    );
    final wArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.eigenvalues,
      wShape,
      DType.float64,
      out: outEigenvalues,
      outParamName: 'outEigenvalues',
    );
    final vArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.eigenvectors,
      vShape,
      DType.float64,
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
/// have shape `[N]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> eigvalsh(
  GpuArray<DTypeTag> a, {
  UpLo uplo = UpLo.lower,
  GpuArray<Float64>? out,
}) {
  _requireSquare2d(a, 'eigvalsh');
  final n = a.shape[0];
  final wShape = <int>[n];
  validateLinalgOut(out, a.device, wShape, DType.float64);

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchEighGpu(
      a.device,
      inputF64,
      n,
      useUpper: uplo == UpLo.upper,
    );
    final wArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.eigenvalues,
      wShape,
      DType.float64,
      out: out,
    );
    if (out == null) wArray.detachToParentScope();
    return wArray;
  });
}

/// Eigenvalues and right eigenvectors of a general square 2-D matrix [a].
///
/// Produces complex `eigenvalues` of shape `[N]` ([DType.complex128]) and
/// normalized right `eigenvectors` of shape `[N, N]` ([DType.complex128])
/// satisfying `a * v[:, i] = w[i] * v[:, i]`.
///
/// The [a] tensor must be a square 2-D matrix. Optional [outEigenvalues] and
/// [outEigenvectors] arrays must have [DType.complex128] and shapes `[N]` and
/// `[N, N]` on `a.device`.
({GpuArray<Complex128> eigenvalues, GpuArray<Complex128> eigenvectors}) eig(
  GpuArray<DTypeTag> a, {
  GpuArray<Complex128>? outEigenvalues,
  GpuArray<Complex128>? outEigenvectors,
}) {
  _requireSquare2d(a, 'eig');
  final n = a.shape[0];
  final wShape = <int>[n];
  final vShape = <int>[n, n];
  validateLinalgOut(
    outEigenvalues,
    a.device,
    wShape,
    DType.complex128,
    paramName: 'outEigenvalues',
  );
  validateLinalgOut(
    outEigenvectors,
    a.device,
    vShape,
    DType.complex128,
    paramName: 'outEigenvectors',
  );

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchEigGpu(a.device, inputF64, n, computeVectors: true);
    final wArray = writeComplex128BufferToArray<Complex128>(
      a.device,
      buffers.eigenvalues,
      wShape,
      DType.complex128,
      out: outEigenvalues,
      outParamName: 'outEigenvalues',
    );
    final vArray = writeComplex128BufferToArray<Complex128>(
      a.device,
      buffers.eigenvectors,
      vShape,
      DType.complex128,
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
/// Produces a 1-D [GpuArray] of shape `[N]` with [DType.complex128].
///
/// The [a] tensor must be a square 2-D matrix. If [out] is provided, it must
/// have shape `[N]`, [DType.complex128], and reside on `a.device`.
GpuArray<Complex128> eigvals(
  GpuArray<DTypeTag> a, {
  GpuArray<Complex128>? out,
}) {
  _requireSquare2d(a, 'eigvals');
  final n = a.shape[0];
  final wShape = <int>[n];
  validateLinalgOut(out, a.device, wShape, DType.complex128);

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchEigGpu(
      a.device,
      inputF64,
      n,
      computeVectors: false,
    );
    final wArray = writeComplex128BufferToArray<Complex128>(
      a.device,
      buffers.eigenvalues,
      wShape,
      DType.complex128,
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
/// arrays must match the expected shapes and [DType.float64] on `a.device`.
({GpuArray<Float64> p, GpuArray<Float64> l, GpuArray<Float64> u}) lu(
  GpuArray<DTypeTag> a, {
  GpuArray<Float64>? outP,
  GpuArray<Float64>? outL,
  GpuArray<Float64>? outU,
}) {
  _require2d(a, 'lu');
  final m = a.shape[0];
  final n = a.shape[1];
  final k = math.min(m, n);
  final pShape = <int>[m, m];
  final lShape = <int>[m, k];
  final uShape = <int>[k, n];

  validateLinalgOut(outP, a.device, pShape, DType.float64, paramName: 'outP');
  validateLinalgOut(outL, a.device, lShape, DType.float64, paramName: 'outL');
  validateLinalgOut(outU, a.device, uShape, DType.float64, paramName: 'outU');

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchLuGpu(a.device, inputF64, m, n);
    final pArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.p,
      pShape,
      DType.float64,
      out: outP,
      outParamName: 'outP',
    );
    final lArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.l,
      lShape,
      DType.float64,
      out: outL,
      outParamName: 'outL',
    );
    final uArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.u,
      uShape,
      DType.float64,
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
({GpuArray<Float64> lu, GpuArray<Int32> pivots}) luFactor(
  GpuArray<DTypeTag> a, {
  GpuArray<Float64>? outLu,
  GpuArray<Int32>? outPivots,
}) {
  _requireSquare2d(a, 'luFactor');
  final n = a.shape[0];
  final luShape = <int>[n, n];
  final pivShape = <int>[n];

  validateLinalgOut(
    outLu,
    a.device,
    luShape,
    DType.float64,
    paramName: 'outLu',
  );
  validateLinalgOut(
    outPivots,
    a.device,
    pivShape,
    DType.int32,
    paramName: 'outPivots',
  );

  return ResourceScope.scope(() {
    final inputF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchLuGpu(a.device, inputF64, n, n);
    final luArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.lu,
      luShape,
      DType.float64,
      out: outLu,
      outParamName: 'outLu',
    );
    final pivArray = writeFloat64BufferToArray<Int32>(
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

/// Snake-case alias for [luFactor].
// ignore: non_constant_identifier_names
final lu_factor = luFactor;

/// Solves a linear system `A * x = b` given the [luFactor] decomposition
/// `(` [lu], [pivots] `)` and right-hand side [b].
///
/// The [lu] tensor must be a square 2-D matrix of shape `[N, N]`, [pivots] must
/// be a 1-D array of length `N`, and [b] must be a 1-D (`[N]`) or 2-D
/// (`[N, K]`) array on the same device.
GpuArray<Float64> luSolve(
  GpuArray<DTypeTag> lu,
  GpuArray<DTypeTag> pivots,
  GpuArray<DTypeTag> b, {
  GpuArray<Float64>? out,
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

  validateLinalgOut(out, lu.device, b.shape, DType.float64);
  final nrhs = b.ndim == 1 ? 1 : b.shape[1];

  return ResourceScope.scope(() {
    final luF64 = toContiguousFloat64Buffer(lu);
    final pivotsF64 = toContiguousFloat64Buffer(pivots);
    final bF64 = toContiguousFloat64Buffer(b);
    final xF64 = dispatchLuSolveGpu(lu.device, luF64, pivotsF64, bF64, n, nrhs);
    final result = writeFloat64BufferToArray<Float64>(
      lu.device,
      xF64,
      b.shape,
      DType.float64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Snake-case alias for [luSolve].
// ignore: non_constant_identifier_names
final lu_solve = luSolve;
