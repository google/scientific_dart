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
import 'linalg.dart';
import 'linalg_buffer_ops.dart';
import 'solver_kernels.dart';
import 'spectral_kernels.dart';
import 'tensor_kernels.dart';

/// Convenience disposal extension for [slogdet] output records.
extension SlogdetRecordDispose
    on ({GpuArray<Float64> sign, GpuArray<Float64> logabsdet}) {
  /// Disposes both [GpuArray] fields (`sign`, `logabsdet`) in this record.
  void dispose() {
    sign.dispose();
    logabsdet.dispose();
  }
}

/// Convenience disposal extension for [lstsq] output records.
extension LstsqResultDispose
    on
        ({
          GpuArray<Float64> solution,
          GpuArray<Float64> residuals,
          int rank,
          GpuArray<Float64> singularValues,
        }) {
  /// Disposes all [GpuArray] fields (`solution`, `residuals`,
  /// `singularValues`) in this record.
  void dispose() {
    solution.dispose();
    residuals.dispose();
    singularValues.dispose();
  }
}

/// Matrix or vector norm order specification for [norm] and [cond].
final class NormOrd {
  final int _kindCode;

  /// Numeric order value for general vector p-norms, or `null` for symbolic
  /// norms.
  final double? pValue;

  const NormOrd._(this._kindCode) : pValue = null;

  /// Frobenius matrix norm.
  static const NormOrd fro = NormOrd._(0);

  /// Alias for [fro] (Frobenius matrix norm).
  static const NormOrd frobenius = fro;

  /// Nuclear matrix norm (sum of singular values).
  static const NormOrd nuc = NormOrd._(1);

  /// Alias for [nuc] (nuclear matrix norm).
  static const NormOrd nuclear = nuc;

  /// Infinity norm (maximum absolute row sum for matrices, maximum absolute
  /// element for vectors).
  static const NormOrd inf = NormOrd._(2);

  /// Alias for [inf] (infinity norm).
  static const NormOrd infinity = inf;

  /// Negative infinity norm (minimum absolute row sum for matrices, minimum
  /// absolute element for vectors).
  static const NormOrd minusInf = NormOrd._(3);

  /// Alias for [minusInf] (negative infinity norm).
  static const NormOrd negativeInfinity = minusInf;

  /// Zero pseudo-norm (number of non-zero elements for vectors).
  static const NormOrd zero = NormOrd._(4);

  /// Alias for [zero] (L0 pseudo-norm).
  static const NormOrd l0 = zero;

  /// L1 norm (maximum absolute column sum for matrices, sum of absolute values
  /// for vectors).
  static const NormOrd one = NormOrd._(5);

  /// Alias for [one] (L1 norm).
  static const NormOrd l1 = one;

  /// Minimum absolute column sum for matrices, or `p = -1` for vectors.
  static const NormOrd minusOne = NormOrd._(6);

  /// L2 norm (largest singular value for matrices, Euclidean norm for vectors).
  static const NormOrd two = NormOrd._(7);

  /// Alias for [two] (L2 norm).
  static const NormOrd l2 = two;

  /// Smallest singular value for matrices, or `p = -2` for vectors.
  static const NormOrd minusTwo = NormOrd._(8);

  /// Arbitrary vector `p`-norm (`(sum |x_i|^p)^(1/p)`).
  const factory NormOrd.p(double p) = NormOrd._p;

  const NormOrd._p(double p) : _kindCode = 9, pValue = p;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is NormOrd &&
          other._kindCode == _kindCode &&
          other.pValue == pValue);

  @override
  int get hashCode => Object.hash(_kindCode, pValue);
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

/// Solves a linear matrix equation `a * x = b` for `x`.
///
/// Computes the exact solution `x` of the well-determined, full-rank linear
/// system defined by square coefficient matrix [a] (`[N, N]`) and ordinate
/// [b] (`[N]` or `[N, K]`).
///
/// The [a] tensor must be a square 2-D matrix, and [b] must be a 1-D or 2-D
/// array on the same [GpuDevice] with `b.shape[0] == a.shape[0]`.
GpuArray<Float64> solve(
  GpuArray<DTypeTag> a,
  GpuArray<DTypeTag> b, {
  GpuArray<Float64>? out,
}) {
  _requireSquare2d(a, 'solve');
  if (b.isDisposed) {
    throw StateError('Cannot execute solve with a disposed b GpuArray.');
  }
  if (b.device != a.device) {
    throw ArgumentError.value(
      b,
      'b',
      'Must reside on the same GpuDevice as a.',
    );
  }
  final n = a.shape[0];
  if (b.ndim != 1 && b.ndim != 2) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must be a 1-D or 2-D array for solve, got ${b.ndim}-D.',
    );
  }
  if (b.shape[0] != n) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must have first dimension $n matching a, got ${b.shape[0]}.',
    );
  }
  validateLinalgOut(out, a.device, b.shape, DType.float64);

  final nrhs = b.ndim == 1 ? 1 : b.shape[1];
  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    final bF64 = toContiguousFloat64Buffer(b);
    final luBuffers = dispatchLuGpu(a.device, aF64, n, n);
    final xF64 = dispatchLuSolveGpu(
      a.device,
      luBuffers.lu,
      luBuffers.pivots,
      bF64,
      n,
      nrhs,
    );
    final result = writeFloat64BufferToArray<Float64>(
      a.device,
      xF64,
      b.shape,
      DType.float64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Multiplicative inverse of a square 2-D matrix [a].
///
/// Given a square matrix [a] of shape `[N, N]`, produces the matrix `a_inv`
/// satisfying `dot(a, a_inv) = I`.
///
/// The [a] tensor must be a square 2-D matrix. If [out] is provided, it must
/// have shape `[N, N]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> inv(GpuArray<DTypeTag> a, {GpuArray<Float64>? out}) {
  _requireSquare2d(a, 'inv');
  final n = a.shape[0];
  validateLinalgOut(out, a.device, a.shape, DType.float64);

  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    final eyeF64 = dispatchIdentityF64Gpu(a.device, n);
    final luBuffers = dispatchLuGpu(a.device, aF64, n, n);
    final invF64 = dispatchLuSolveGpu(
      a.device,
      luBuffers.lu,
      luBuffers.pivots,
      eyeF64,
      n,
      n,
    );
    final result = writeFloat64BufferToArray<Float64>(
      a.device,
      invF64,
      a.shape,
      DType.float64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Moore-Penrose pseudo-inverse of a 2-D matrix [a].
///
/// Calculates the generalized inverse of an `[M, N]` matrix [a] using its
/// singular-value decomposition, producing a matrix of shape `[N, M]`. Singular
/// values smaller than `[rcond] * max(singular_values)` are treated as zero.
///
/// The [a] tensor must be a 2-D matrix. If [out] is provided, it must have
/// shape `[N, M]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> pinv(
  GpuArray<DTypeTag> a, {
  double? rcond,
  GpuArray<Float64>? out,
}) {
  _require2d(a, 'pinv');
  final m = a.shape[0];
  final n = a.shape[1];
  final outShape = <int>[n, m];
  validateLinalgOut(out, a.device, outShape, DType.float64);

  final effectiveRcond = rcond ?? (1e-12 * math.max(m, n));
  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    final svdBuffers = dispatchSvdGpu(
      a.device,
      aF64,
      m,
      n,
      fullMatrices: false,
    );
    final pinvBuffers = dispatchPinvAndLstsqFromSvdGpu(
      a.device,
      aF64: aF64,
      uF64: svdBuffers.u,
      sF64: svdBuffers.s,
      vtF64: svdBuffers.vt,
      bF64: aF64,
      m: m,
      n: n,
      nrhs: 1,
      rcond: effectiveRcond,
      computeLstsq: false,
    );
    final result = writeFloat64BufferToArray<Float64>(
      a.device,
      pinvBuffers.pinv,
      outShape,
      DType.float64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Least-squares solution to a linear matrix equation `a * x = b`.
///
/// Computes the vector or matrix `x` that minimizes the Euclidean 2-norm
/// `||b - a * x||_2`. Produces a record containing:
/// - `solution`: least-squares solution of shape `[N]` (if [b] is 1-D) or
///   `[N, K]` (if [b] is 2-D).
/// - `residuals`: sum of squared residuals per column of [b] (shape `[1]` or
///   `[K]` when `M > N` and `rank == N`, or `[0]` otherwise).
/// - `rank`: effective matrix rank of [a].
/// - `singularValues`: singular values of [a] of shape `[min(M, N)]`.
///
/// The [a] tensor must be a 2-D `[M, N]` matrix, and [b] must be a 1-D (`[M]`)
/// or 2-D (`[M, K]`) tensor on the same device.
({
  GpuArray<Float64> solution,
  GpuArray<Float64> residuals,
  int rank,
  GpuArray<Float64> singularValues,
})
lstsq(
  GpuArray<DTypeTag> a,
  GpuArray<DTypeTag> b, {
  double? rcond,
  GpuArray<Float64>? outSolution,
  GpuArray<Float64>? outResiduals,
  GpuArray<Float64>? outSingularValues,
}) {
  _require2d(a, 'lstsq');
  if (b.isDisposed) {
    throw StateError('Cannot execute lstsq with a disposed b GpuArray.');
  }
  if (b.device != a.device) {
    throw ArgumentError.value(
      b,
      'b',
      'Must reside on the same GpuDevice as a.',
    );
  }
  final m = a.shape[0];
  final n = a.shape[1];
  if (b.ndim != 1 && b.ndim != 2) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must be a 1-D or 2-D array for lstsq, got ${b.ndim}-D.',
    );
  }
  if (b.shape[0] != m) {
    throw ArgumentError.value(
      b.shape,
      'b',
      'Must have first dimension $m matching a, got ${b.shape[0]}.',
    );
  }

  final kMin = math.min(m, n);
  final nrhs = b.ndim == 1 ? 1 : b.shape[1];
  final solShape = b.ndim == 1 ? <int>[n] : <int>[n, nrhs];
  final sShape = <int>[kMin];

  validateLinalgOut(
    outSolution,
    a.device,
    solShape,
    DType.float64,
    paramName: 'outSolution',
  );
  validateLinalgOut(
    outSingularValues,
    a.device,
    sShape,
    DType.float64,
    paramName: 'outSingularValues',
  );

  final effectiveRcond = rcond ?? (1e-12 * math.max(m, n));
  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    final bF64 = toContiguousFloat64Buffer(b);
    final svdBuffers = dispatchSvdGpu(
      a.device,
      aF64,
      m,
      n,
      fullMatrices: false,
    );
    final lstsqBuffers = dispatchPinvAndLstsqFromSvdGpu(
      a.device,
      aF64: aF64,
      uF64: svdBuffers.u,
      sF64: svdBuffers.s,
      vtF64: svdBuffers.vt,
      bF64: bF64,
      m: m,
      n: n,
      nrhs: nrhs,
      rcond: effectiveRcond,
      computeLstsq: true,
    );
    final rankArray = writeFloat64BufferToArray<Int32>(
      a.device,
      lstsqBuffers.rank,
      const <int>[],
      DType.int32,
    );
    final rankValue = rankArray.scalar;
    final resShape = (m > n && rankValue == n) ? <int>[nrhs] : <int>[0];
    validateLinalgOut(
      outResiduals,
      a.device,
      resShape,
      DType.float64,
      paramName: 'outResiduals',
    );

    final solArray = writeFloat64BufferToArray<Float64>(
      a.device,
      lstsqBuffers.solution,
      solShape,
      DType.float64,
      out: outSolution,
      outParamName: 'outSolution',
    );
    final resArray = writeFloat64BufferToArray<Float64>(
      a.device,
      lstsqBuffers.residuals,
      resShape,
      DType.float64,
      out: outResiduals,
      outParamName: 'outResiduals',
    );
    final sArray = writeFloat64BufferToArray<Float64>(
      a.device,
      svdBuffers.s,
      sShape,
      DType.float64,
      out: outSingularValues,
      outParamName: 'outSingularValues',
    );
    if (outSolution == null) solArray.detachToParentScope();
    if (outResiduals == null) resArray.detachToParentScope();
    if (outSingularValues == null) sArray.detachToParentScope();
    return (
      solution: solArray,
      residuals: resArray,
      rank: rankValue,
      singularValues: sArray,
    );
  });
}

/// Determinant of a square 2-D matrix [a].
///
/// Produces a 0-D scalar [GpuArray] of [DType.float64].
///
/// The [a] tensor must be a square 2-D matrix. If [out] is provided, it must
/// have shape `[]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> det(GpuArray<DTypeTag> a, {GpuArray<Float64>? out}) {
  _requireSquare2d(a, 'det');
  final n = a.shape[0];
  validateLinalgOut(out, a.device, const <int>[], DType.float64);

  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchDetAndSlogdetGpu(a.device, aF64, n);
    final result = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.det,
      const <int>[],
      DType.float64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Sign and natural logarithm of the absolute determinant of a square 2-D
/// matrix [a].
///
/// If the determinant is zero, `sign` is `0.0` and `logabsdet` is `-infinity`.
/// Otherwise, `det(a) = sign * exp(logabsdet)`.
///
/// The [a] tensor must be a square 2-D matrix. Optional [outSign] and
/// [outLogAbsDet] arrays must be 0-D (`[]`) with [DType.float64] on `a.device`.
({GpuArray<Float64> sign, GpuArray<Float64> logabsdet}) slogdet(
  GpuArray<DTypeTag> a, {
  GpuArray<Float64>? outSign,
  GpuArray<Float64>? outLogAbsDet,
}) {
  _requireSquare2d(a, 'slogdet');
  final n = a.shape[0];
  validateLinalgOut(
    outSign,
    a.device,
    const <int>[],
    DType.float64,
    paramName: 'outSign',
  );
  validateLinalgOut(
    outLogAbsDet,
    a.device,
    const <int>[],
    DType.float64,
    paramName: 'outLogAbsDet',
  );

  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    final buffers = dispatchDetAndSlogdetGpu(a.device, aF64, n);
    final signArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.sign,
      const <int>[],
      DType.float64,
      out: outSign,
      outParamName: 'outSign',
    );
    final logArray = writeFloat64BufferToArray<Float64>(
      a.device,
      buffers.logabsdet,
      const <int>[],
      DType.float64,
      out: outLogAbsDet,
      outParamName: 'outLogAbsDet',
    );
    if (outSign == null) signArray.detachToParentScope();
    if (outLogAbsDet == null) logArray.detachToParentScope();
    return (sign: signArray, logabsdet: logArray);
  });
}

/// Raises a square 2-D matrix [a] to the integer power [n].
///
/// For `n == 0`, produces the identity matrix. For `n > 0`, uses binary
/// exponentiation by repeated matrix multiplication. For `n < 0`, computes the
/// matrix inverse and raises it to `|n|`.
///
/// The [a] tensor must be a square 2-D matrix. If [out] is provided, it must
/// match `a.shape`, `a.dtype`, and `a.device`.
GpuArray<T> matrixPower<T extends DTypeTag>(
  GpuArray<T> a,
  int n, {
  GpuArray<T>? out,
}) {
  _requireSquare2d(a, 'matrixPower');
  validateLinalgOut(out, a.device, a.shape, a.dtype);
  final size = a.shape[0];

  return ResourceScope.scope(() {
    if (n == 0) {
      final eyeBuffer = dispatchIdentityF64Gpu(a.device, size);
      final output = writeFloat64BufferToArray<T>(
        a.device,
        eyeBuffer,
        a.shape,
        a.dtype,
        out: out,
      );
      if (out == null) output.detachToParentScope();
      return output;
    }

    var currentBuffer = toContiguousFloat64Buffer(a);
    if (n < 0) {
      final eyeBuffer = dispatchIdentityF64Gpu(a.device, size);
      final luBuffers = dispatchLuGpu(a.device, currentBuffer, size, size);
      currentBuffer = dispatchLuSolveGpu(
        a.device,
        luBuffers.lu,
        luBuffers.pivots,
        eyeBuffer,
        size,
        size,
      );
    }

    var exponent = n.abs();
    var accumulatorBuffer = dispatchIdentityF64Gpu(a.device, size);
    while (exponent > 0) {
      if ((exponent & 1) != 0) {
        accumulatorBuffer = dispatchBatchedMatmulF64Gpu(
          a.device,
          accumulatorBuffer,
          currentBuffer,
          batchCount: 1,
          m: size,
          k: size,
          n: size,
        );
      }
      exponent >>= 1;
      if (exponent > 0) {
        currentBuffer = dispatchBatchedMatmulF64Gpu(
          a.device,
          currentBuffer,
          currentBuffer,
          batchCount: 1,
          m: size,
          k: size,
          n: size,
        );
      }
    }

    final output = writeFloat64BufferToArray<T>(
      a.device,
      accumulatorBuffer,
      a.shape,
      a.dtype,
      out: out,
    );
    if (out == null) output.detachToParentScope();
    return output;
  });
}

/// Snake-case alias for [matrixPower].
// ignore: non_constant_identifier_names
final matrix_power = matrixPower;

/// Numerical rank of a 2-D matrix [a] computed via SVD.
///
/// Counts the number of singular values strictly greater than [tol] (or
/// `max(M, N) * eps * max(S)` when [tol] is `null`). Produces a 0-D scalar
/// [GpuArray] of [DType.int64].
///
/// The [a] tensor must be a 2-D matrix. If [out] is provided, it must have
/// shape `[]`, [DType.int64], and reside on `a.device`.
GpuArray<Int64> matrixRank(
  GpuArray<DTypeTag> a, {
  double? tol,
  GpuArray<Int64>? out,
}) {
  _require2d(a, 'matrixRank');
  validateLinalgOut(out, a.device, const <int>[], DType.int64);
  final m = a.shape[0];
  final n = a.shape[1];
  final rcond = tol ?? (1e-12 * math.max(m, n));

  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    final svdBuffers = dispatchSvdGpu(
      a.device,
      aF64,
      m,
      n,
      fullMatrices: false,
    );
    final buffers = dispatchPinvAndLstsqFromSvdGpu(
      a.device,
      aF64: aF64,
      uF64: svdBuffers.u,
      sF64: svdBuffers.s,
      vtF64: svdBuffers.vt,
      bF64: aF64,
      m: m,
      n: n,
      nrhs: 1,
      rcond: rcond,
      computeLstsq: false,
    );
    final result = writeFloat64BufferToArray<Int64>(
      a.device,
      buffers.rank,
      const <int>[],
      DType.int64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Snake-case alias for [matrixRank].
// ignore: non_constant_identifier_names
final matrix_rank = matrixRank;

int _vectorNormMode(NormOrd? ord) {
  if (ord == null || ord == NormOrd.two || ord == NormOrd.fro) return 0;
  if (ord == NormOrd.one) return 1;
  if (ord == NormOrd.inf) return 2;
  if (ord == NormOrd.minusInf) return 3;
  if (ord == NormOrd.zero) return 4;
  if (ord == NormOrd.minusOne ||
      ord == NormOrd.minusTwo ||
      ord.pValue != null) {
    return 5;
  }
  throw ArgumentError.value(ord, 'ord', 'Must be a valid vector norm order.');
}

double _vectorNormP(NormOrd? ord) {
  if (ord == null) return 2.0;
  if (ord == NormOrd.minusOne) return -1.0;
  if (ord == NormOrd.minusTwo) return -2.0;
  return ord.pValue ?? 2.0;
}

/// Matrix or vector norm of [a].
///
/// Supports Frobenius, nuclear, L1, L2, infinity, and general `p`-norms via
/// [ord], optional reduction along [axis], and dimension retention via
/// [keepdims].
///
/// The [a] tensor must not be disposed. If [out] is provided, it must match the
/// output shape, [DType.float64], and `a.device`.
GpuArray<Float64> norm(
  GpuArray<DTypeTag> a, {
  NormOrd? ord,
  Object? axis,
  bool keepdims = false,
  GpuArray<Float64>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute norm of a disposed GpuArray.');
  }

  if (axis is int) {
    final normalizedAxis = axis < 0 ? a.ndim + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= a.ndim) {
      throw ArgumentError.value(
        axis,
        'axis',
        'Must be in range [-${a.ndim}, ${a.ndim - 1}].',
      );
    }
    var outerSize = 1;
    for (var i = 0; i < normalizedAxis; i++) {
      outerSize *= a.shape[i];
    }
    final axisLength = a.shape[normalizedAxis];
    var innerSize = 1;
    for (var i = normalizedAxis + 1; i < a.ndim; i++) {
      innerSize *= a.shape[i];
    }
    final outShape = keepdims
        ? <int>[
            for (var i = 0; i < a.ndim; i++)
              i == normalizedAxis ? 1 : a.shape[i],
          ]
        : <int>[
            for (var i = 0; i < a.ndim; i++)
              if (i != normalizedAxis) a.shape[i],
          ];
    validateLinalgOut(out, a.device, outShape, DType.float64);

    final mode = _vectorNormMode(ord);
    final pVal = _vectorNormP(ord);
    return ResourceScope.scope(() {
      final aF64 = toContiguousFloat64Buffer(a);
      final normF64 = dispatchNormGpu(
        a.device,
        aF64,
        outerSize: outerSize,
        axisLength: axisLength,
        innerSize: innerSize,
        normMode: mode,
        pValue: pVal,
      );
      final result = writeFloat64BufferToArray<Float64>(
        a.device,
        normF64,
        outShape,
        DType.float64,
        out: out,
      );
      if (out == null) result.detachToParentScope();
      return result;
    });
  }

  final outShape = keepdims ? List<int>.filled(a.ndim, 1) : const <int>[];
  validateLinalgOut(out, a.device, outShape, DType.float64);

  return ResourceScope.scope(() {
    final aF64 = toContiguousFloat64Buffer(a);
    if (a.ndim == 2 && ord != null && ord != NormOrd.fro) {
      final m = a.shape[0];
      final n = a.shape[1];
      if (ord == NormOrd.one ||
          ord == NormOrd.minusOne ||
          ord == NormOrd.inf ||
          ord == NormOrd.minusInf) {
        final mode = ord == NormOrd.one
            ? 10
            : (ord == NormOrd.minusOne ? 11 : (ord == NormOrd.inf ? 12 : 13));
        final normF64 = dispatchNormGpu(
          a.device,
          aF64,
          outerSize: 1,
          axisLength: a.size,
          innerSize: 1,
          normMode: mode,
          rows: m,
          cols: n,
        );
        final result = writeFloat64BufferToArray<Float64>(
          a.device,
          normF64,
          outShape,
          DType.float64,
          out: out,
        );
        if (out == null) result.detachToParentScope();
        return result;
      }
      if (ord == NormOrd.nuc || ord == NormOrd.two || ord == NormOrd.minusTwo) {
        final kMin = math.min(m, n);
        final svdBuffers = dispatchSvdGpu(
          a.device,
          aF64,
          m,
          n,
          fullMatrices: false,
        );
        final mode = ord == NormOrd.nuc ? 14 : (ord == NormOrd.two ? 15 : 16);
        final normF64 = dispatchNormGpu(
          a.device,
          svdBuffers.s,
          outerSize: 1,
          axisLength: kMin,
          innerSize: 1,
          normMode: mode,
        );
        final result = writeFloat64BufferToArray<Float64>(
          a.device,
          normF64,
          outShape,
          DType.float64,
          out: out,
        );
        if (out == null) result.detachToParentScope();
        return result;
      }
      throw ArgumentError.value(ord, 'ord', 'Must be a valid matrix norm.');
    }

    final mode = _vectorNormMode(ord);
    final pVal = _vectorNormP(ord);
    final normF64 = dispatchNormGpu(
      a.device,
      aF64,
      outerSize: 1,
      axisLength: a.size,
      innerSize: 1,
      normMode: mode,
      pValue: pVal,
    );
    final result = writeFloat64BufferToArray<Float64>(
      a.device,
      normF64,
      outShape,
      DType.float64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Condition number of a 2-D matrix [a] with respect to norm [p] (or [ord]).
///
/// Defaults to the 2-norm condition number (`max(S) / min(S)`).
///
/// The [a] tensor must be a non-empty 2-D matrix. If [out] is provided, it must
/// have shape `[]`, [DType.float64], and reside on `a.device`.
GpuArray<Float64> cond(
  GpuArray<DTypeTag> a, {
  NormOrd? p,
  NormOrd? ord,
  GpuArray<Float64>? out,
}) {
  _require2d(a, 'cond');
  final m = a.shape[0];
  final n = a.shape[1];
  if (m == 0 || n == 0) {
    throw ArgumentError.value(a.shape, 'a', 'Must not be an empty matrix.');
  }
  validateLinalgOut(out, a.device, const <int>[], DType.float64);
  final effectiveOrd = p ?? ord;

  return ResourceScope.scope(() {
    if (effectiveOrd == null ||
        effectiveOrd == NormOrd.two ||
        effectiveOrd == NormOrd.minusTwo) {
      final kMin = math.min(m, n);
      final aF64 = toContiguousFloat64Buffer(a);
      final svdBuffers = dispatchSvdGpu(
        a.device,
        aF64,
        m,
        n,
        fullMatrices: false,
      );
      final mode = (effectiveOrd == NormOrd.minusTwo) ? 18 : 17;
      final condF64 = dispatchNormGpu(
        a.device,
        svdBuffers.s,
        outerSize: 1,
        axisLength: kMin,
        innerSize: 1,
        normMode: mode,
      );
      final result = writeFloat64BufferToArray<Float64>(
        a.device,
        condF64,
        const <int>[],
        DType.float64,
        out: out,
      );
      if (out == null) result.detachToParentScope();
      return result;
    }

    _requireSquare2d(a, 'cond');
    final normA = norm(a, ord: effectiveOrd);
    final invA = inv(a);
    final normInvA = norm(invA, ord: effectiveOrd);
    final normAF64 = toContiguousFloat64Buffer(normA);
    final normInvAF64 = toContiguousFloat64Buffer(normInvA);
    final condF64 = dispatchScalarMulF64Gpu(a.device, normAF64, normInvAF64);
    final result = writeFloat64BufferToArray<Float64>(
      a.device,
      condF64,
      const <int>[],
      DType.float64,
      out: out,
    );
    if (out == null) result.detachToParentScope();
    return result;
  });
}

/// Chained matrix product of two or more [GpuArray] tensors in [arrays].
///
/// The [arrays] list must contain at least 2 tensors residing on the same
/// [GpuDevice] with matching [DType]s. If [out] is provided, it must match the
/// output shape, dtype, and device.
GpuArray<T> multiDot<T extends DTypeTag>(
  List<GpuArray<T>> arrays, {
  GpuArray<T>? out,
}) {
  if (arrays.length < 2) {
    throw ArgumentError.value(
      arrays.length,
      'arrays',
      'Must contain at least 2 arrays for multiDot.',
    );
  }
  for (var i = 0; i < arrays.length; i++) {
    if (arrays[i].isDisposed) {
      throw StateError('Cannot execute multiDot on a disposed GpuArray.');
    }
    if (arrays[i].device != arrays[0].device) {
      throw ArgumentError.value(
        arrays[i],
        'arrays[$i]',
        'Must reside on the same GpuDevice as arrays[0].',
      );
    }
    if (arrays[i].dtype != arrays[0].dtype) {
      throw ArgumentError.value(
        arrays[i],
        'arrays[$i]',
        'Must have the same DType as arrays[0].',
      );
    }
  }

  return ResourceScope.scope(() {
    var current = arrays[0];
    for (var i = 1; i < arrays.length; i++) {
      final isLast = i == arrays.length - 1;
      current = matmul(current, arrays[i], out: isLast ? out : null);
    }
    if (out == null) current.detachToParentScope();
    return current;
  });
}

/// Snake-case alias for [multiDot].
// ignore: non_constant_identifier_names
final multi_dot = multiDot;
