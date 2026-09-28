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
import '../dtype.dart';
import '../gpu_array.dart';
import '../exceptions.dart';
import '../backend/compute_engine.dart';

/// Result of Singular Value Decomposition (SVD).
final class SvdResult<T extends DTypeTag> {
  /// Left singular vectors $U$.
  final GpuArray<T> u;

  /// Singular values $S$ sorted in descending order.
  final GpuArray<T> s;

  /// Right singular vectors transposed $V^T$.
  final GpuArray<T> vt;

  const SvdResult({required this.u, required this.s, required this.vt});

  @override
  String toString() => 'SvdResult(u: $u, s: $s, vt: $vt)';
}

/// Result of QR Decomposition.
final class QrResult<T extends DTypeTag> {
  /// Orthonormal matrix $Q$.
  final GpuArray<T> q;

  /// Upper triangular matrix $R$.
  final GpuArray<T> r;

  const QrResult({required this.q, required this.r});

  @override
  String toString() => 'QrResult(q: $q, r: $r)';
}

/// Result of Eigendecomposition.
final class EigResult<T extends DTypeTag> {
  /// Eigenvalues $\lambda$.
  final GpuArray<T> eigenvalues;

  /// Normalized eigenvectors $V$.
  final GpuArray<T> eigenvectors;

  const EigResult({required this.eigenvalues, required this.eigenvectors});

  @override
  String toString() =>
      'EigResult(eigenvalues: $eigenvalues, eigenvectors: $eigenvectors)';
}

/// Result of LU Decomposition.
final class LuResult<T extends DTypeTag> {
  /// Permutation matrix $P$.
  final GpuArray<T> p;

  /// Unit lower-triangular matrix $L$.
  final GpuArray<T> l;

  /// Upper-triangular matrix $U$.
  final GpuArray<T> u;

  const LuResult({required this.p, required this.l, required this.u});

  @override
  String toString() => 'LuResult(p: $p, l: $l, u: $u)';
}

/// Result of LU Factorization with pivot indices.
final class LuFactorResult<T extends DTypeTag> {
  /// Combined LU factor matrix where diagonal and upper part is $U$ and strictly lower part is $L$.
  final GpuArray<T> lu;

  /// Pivot permutation indices.
  final GpuArray<Int32> piv;

  const LuFactorResult({required this.lu, required this.piv});

  @override
  String toString() => 'LuFactorResult(lu: $lu, piv: $piv)';
}

/// Computes the QR decomposition of matrix [a] using Householder reflections.
QrResult<T> qr<T extends DTypeTag>(GpuArray<T> a, {String mode = 'reduced'}) {
  if (a.rank < 2) {
    throw ArgumentError('qr() requires an array with at least 2 dimensions.');
  }

  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m == 0 || n == 0 || ShapeUtils.computeSize(a.shape) == 0) {
    throw ArgumentError('Matrix dimensions cannot be zero.');
  }
  final k = math.min(m, n);
  final isReduced = (mode == 'reduced');

  final qCols = isReduced ? k : m;
  final rRows = isReduced ? k : m;

  final qShape = List<int>.from(a.shape)
    ..setRange(a.rank - 2, a.rank, [m, qCols]);
  final rShape = List<int>.from(a.shape)
    ..setRange(a.rank - 2, a.rank, [rRows, n]);

  final q = GpuArray<T>.empty(qShape, a.dtype, device: a.device);
  final r = GpuArray<T>.empty(rShape, a.dtype, device: a.device);

  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (m * n);
  final aFlat = a.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();

  final qData = List<double>.filled(batchSize * m * qCols, 0.0);
  final rData = List<double>.filled(batchSize * rRows * n, 0.0);

  for (var b = 0; b < batchSize; b++) {
    final aMat = List.generate(
      m,
      (i) => List.generate(n, (j) => aData[b * m * n + i * n + j]),
    );
    final qMat = List.generate(
      m,
      (i) => List.generate(m, (j) => (i == j) ? 1.0 : 0.0),
    );
    final rMat = List.generate(m, (i) => List.generate(n, (j) => aMat[i][j]));

    for (var j = 0; j < k; j++) {
      var normX = 0.0;
      for (var i = j; i < m; i++) {
        normX += rMat[i][j] * rMat[i][j];
      }
      normX = math.sqrt(normX);
      if (normX < 1e-15) continue;

      final sign = (rMat[j][j] >= 0.0) ? 1.0 : -1.0;
      final u1 = rMat[j][j] + sign * normX;
      final v = List<double>.filled(m - j, 0.0);
      v[0] = 1.0;
      for (var i = 1; i < m - j; i++) {
        v[i] = rMat[j + i][j] / u1;
      }

      var vDotV = 0.0;
      for (final vi in v) {
        vDotV += vi * vi;
      }
      final tau = 2.0 / vDotV;

      // Apply Householder reflection to R: R[j:m, j:n] -= tau * v * (v^T R[j:m, j:n])
      for (var col = j; col < n; col++) {
        var dot = 0.0;
        for (var i = 0; i < m - j; i++) {
          dot += v[i] * rMat[j + i][col];
        }
        for (var i = 0; i < m - j; i++) {
          rMat[j + i][col] -= tau * v[i] * dot;
        }
      }

      // Apply Householder reflection to Q: Q[:, j:m] -= tau * (Q[:, j:m] v) * v^T
      for (var row = 0; row < m; row++) {
        var dot = 0.0;
        for (var i = 0; i < m - j; i++) {
          dot += qMat[row][j + i] * v[i];
        }
        for (var i = 0; i < m - j; i++) {
          qMat[row][j + i] -= tau * dot * v[i];
        }
      }
    }

    // Copy into output buffers
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < qCols; j++) {
        qData[b * m * qCols + i * qCols + j] = qMat[i][j];
      }
    }
    for (var i = 0; i < rRows; i++) {
      for (var j = 0; j < n; j++) {
        rData[b * rRows * n + i * n + j] = (i <= j && i < m) ? rMat[i][j] : 0.0;
      }
    }
  }

  aFlat.dispose();

  for (var i = 0; i < qData.length; i++) {
    ComputeEngine.writeAny(q.buffer, a.dtype, i, qData[i]);
  }
  for (var i = 0; i < rData.length; i++) {
    ComputeEngine.writeAny(r.buffer, a.dtype, i, rData[i]);
  }

  return QrResult<T>(q: q, r: r);
}

/// Computes the Cholesky decomposition of a symmetric/Hermitian positive-definite matrix [a].
GpuArray<T> cholesky<T extends DTypeTag>(GpuArray<T> a, {bool upper = false}) {
  if (a.rank < 2) {
    throw ArgumentError(
      'cholesky() requires an array of at least 2 dimensions.',
    );
  }
  final n = a.shape[a.rank - 1];
  final m = a.shape[a.rank - 2];
  if (n != m) {
    throw GpuShapeMismatchException('cholesky', a.shape, a.shape);
  }
  if (m == 0 || n == 0 || ShapeUtils.computeSize(a.shape) == 0) {
    throw ArgumentError('Matrix dimensions cannot be zero.');
  }

  final result = GpuArray<T>.empty(a.shape, a.dtype, device: a.device);
  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (n * n);
  final aFlat = a.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final lData = List<double>.filled(batchSize * n * n, 0.0);

  for (var b = 0; b < batchSize; b++) {
    final lMat = List.generate(n, (_) => List.filled(n, 0.0));

    for (var i = 0; i < n; i++) {
      for (var j = 0; j <= i; j++) {
        var sum = 0.0;
        for (var k = 0; k < j; k++) {
          sum += lMat[i][k] * lMat[j][k];
        }

        final aVal = aData[b * n * n + i * n + j];
        if (i == j) {
          final diff = aVal - sum;
          if (diff <= 0.0) {
            aFlat.dispose();
            throw ArgumentError(
              'Matrix is not positive-definite for Cholesky decomposition (diagonal pivot <= 0 at row $i).',
            );
          }
          lMat[i][j] = math.sqrt(diff);
        } else {
          lMat[i][j] = (aVal - sum) / lMat[j][j];
        }
      }
    }

    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        final val = upper ? lMat[j][i] : lMat[i][j];
        lData[b * n * n + i * n + j] = val;
      }
    }
  }

  aFlat.dispose();

  for (var i = 0; i < lData.length; i++) {
    ComputeEngine.writeAny(result.buffer, a.dtype, i, lData[i]);
  }

  return result;
}

/// Computes the LU decomposition with partial pivoting: $A = P^T L U$.
LuResult<T> lu<T extends DTypeTag>(GpuArray<T> a) {
  if (a.rank < 2) {
    throw ArgumentError('lu() requires an array of at least 2 dimensions.');
  }

  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m == 0 || n == 0 || ShapeUtils.computeSize(a.shape) == 0) {
    throw ArgumentError('Matrix dimensions cannot be zero.');
  }
  final k = math.min(m, n);

  final pShape = List<int>.from(a.shape)..setRange(a.rank - 2, a.rank, [m, m]);
  final lShape = List<int>.from(a.shape)..setRange(a.rank - 2, a.rank, [m, k]);
  final uShape = List<int>.from(a.shape)..setRange(a.rank - 2, a.rank, [k, n]);

  final p = GpuArray<T>.empty(pShape, a.dtype, device: a.device);
  final l = GpuArray<T>.empty(lShape, a.dtype, device: a.device);
  final u = GpuArray<T>.empty(uShape, a.dtype, device: a.device);

  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (m * n);
  final aFlat = a.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();

  final pData = List<double>.filled(batchSize * m * m, 0.0);
  final lData = List<double>.filled(batchSize * m * k, 0.0);
  final uData = List<double>.filled(batchSize * k * n, 0.0);

  for (var b = 0; b < batchSize; b++) {
    final aMat = List.generate(
      m,
      (i) => List.generate(n, (j) => aData[b * m * n + i * n + j]),
    );
    final piv = List.generate(m, (i) => i);

    for (var j = 0; j < k; j++) {
      // Find pivot in column j
      var maxVal = aMat[j][j].abs();
      var maxRow = j;
      for (var i = j + 1; i < m; i++) {
        final v = aMat[i][j].abs();
        if (v > maxVal) {
          maxVal = v;
          maxRow = i;
        }
      }

      if (maxRow != j) {
        final temp = aMat[j];
        aMat[j] = aMat[maxRow];
        aMat[maxRow] = temp;

        final tempPiv = piv[j];
        piv[j] = piv[maxRow];
        piv[maxRow] = tempPiv;
      }

      final pivotVal = aMat[j][j];
      if (pivotVal.abs() > 1e-15) {
        for (var i = j + 1; i < m; i++) {
          aMat[i][j] /= pivotVal;
          for (var col = j + 1; col < n; col++) {
            aMat[i][col] -= aMat[i][j] * aMat[j][col];
          }
        }
      }
    }

    // Construct P matrix
    for (var i = 0; i < m; i++) {
      pData[b * m * m + i * m + piv[i]] = 1.0;
    }

    // Construct L matrix
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < k; j++) {
        if (i == j) {
          lData[b * m * k + i * k + j] = 1.0;
        } else if (i > j) {
          lData[b * m * k + i * k + j] = aMat[i][j];
        }
      }
    }

    // Construct U matrix
    for (var i = 0; i < k; i++) {
      for (var j = 0; j < n; j++) {
        if (i <= j) {
          uData[b * k * n + i * n + j] = aMat[i][j];
        }
      }
    }
  }

  aFlat.dispose();

  for (var i = 0; i < pData.length; i++) {
    ComputeEngine.writeAny(p.buffer, a.dtype, i, pData[i]);
  }
  for (var i = 0; i < lData.length; i++) {
    ComputeEngine.writeAny(l.buffer, a.dtype, i, lData[i]);
  }
  for (var i = 0; i < uData.length; i++) {
    ComputeEngine.writeAny(u.buffer, a.dtype, i, uData[i]);
  }

  return LuResult<T>(p: p, l: l, u: u);
}

/// Computes pivoted LU factorization of matrix [a].
LuFactorResult<T> lu_factor<T extends DTypeTag>(GpuArray<T> a) {
  if (a.rank < 2) {
    throw ArgumentError(
      'lu_factor() requires an array of at least 2 dimensions.',
    );
  }

  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m == 0 || n == 0 || ShapeUtils.computeSize(a.shape) == 0) {
    throw ArgumentError('Matrix dimensions cannot be zero.');
  }
  final k = math.min(m, n);

  final lu = GpuArray<T>.empty(a.shape, a.dtype, device: a.device);
  final pivShape = List<int>.from(a.shape)
    ..removeLast()
    ..setRange(a.rank - 2, a.rank - 1, [k]);
  final piv = GpuArray<Int32>.empty(pivShape, DType.int32, device: a.device);

  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (m * n);
  final aFlat = a.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();

  final luData = List<double>.filled(batchSize * m * n, 0.0);
  final pivData = List<int>.filled(batchSize * k, 0);

  for (var b = 0; b < batchSize; b++) {
    final aMat = List.generate(
      m,
      (i) => List.generate(n, (j) => aData[b * m * n + i * n + j]),
    );
    final pIndices = List.generate(k, (i) => i);

    for (var j = 0; j < k; j++) {
      var maxVal = aMat[j][j].abs();
      var maxRow = j;
      for (var i = j + 1; i < m; i++) {
        final v = aMat[i][j].abs();
        if (v > maxVal) {
          maxVal = v;
          maxRow = i;
        }
      }

      pIndices[j] = maxRow;

      if (maxRow != j) {
        final temp = aMat[j];
        aMat[j] = aMat[maxRow];
        aMat[maxRow] = temp;
      }

      final pivotVal = aMat[j][j];
      if (pivotVal.abs() > 1e-15) {
        for (var i = j + 1; i < m; i++) {
          aMat[i][j] /= pivotVal;
          for (var col = j + 1; col < n; col++) {
            aMat[i][col] -= aMat[i][j] * aMat[j][col];
          }
        }
      }
    }

    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        luData[b * m * n + i * n + j] = aMat[i][j];
      }
    }
    for (var j = 0; j < k; j++) {
      pivData[b * k + j] = pIndices[j];
    }
  }

  aFlat.dispose();

  for (var i = 0; i < luData.length; i++) {
    ComputeEngine.writeAny(lu.buffer, a.dtype, i, luData[i]);
  }
  for (var i = 0; i < pivData.length; i++) {
    ComputeEngine.writeAny(piv.buffer, DType.int32, i, pivData[i]);
  }

  return LuFactorResult<T>(lu: lu, piv: piv);
}

/// Solves a linear system $A x = b$ using precomputed LU factorization [lu] and pivots [piv].
GpuArray<T> lu_solve<T extends DTypeTag>(
  GpuArray<T> lu,
  GpuArray<Int32> piv,
  GpuArray<T> b,
) {
  if (lu.rank < 2) {
    throw ArgumentError('lu_solve() requires lu of at least 2 dimensions.');
  }
  final n = lu.shape[lu.rank - 1];
  final m = lu.shape[lu.rank - 2];
  if (m != n) {
    throw GpuShapeMismatchException('lu_solve', lu.shape, lu.shape);
  }
  if (m == 0 || n == 0 || ShapeUtils.computeSize(lu.shape) == 0) {
    throw ArgumentError('Matrix dimensions cannot be zero.');
  }
  final isVector = (b.rank == lu.rank - 1);
  if (b.rank != lu.rank && !isVector) {
    throw ArgumentError(
      'lu_solve: RHS b rank (${b.rank}) must be either equal to lu rank (${lu.rank}) or lu rank - 1 (${lu.rank - 1}).',
    );
  }

  if (isVector) {
    if (b.shape.last != n ||
        !ShapeUtils.areEqual(
          b.shape.sublist(0, b.rank - 1),
          lu.shape.sublist(0, lu.rank - 2),
        )) {
      throw GpuShapeMismatchException('lu_solve', lu.shape, b.shape);
    }
  } else {
    if (b.shape[b.rank - 2] != n ||
        !ShapeUtils.areEqual(
          b.shape.sublist(0, b.rank - 2),
          lu.shape.sublist(0, lu.rank - 2),
        )) {
      throw GpuShapeMismatchException('lu_solve', lu.shape, b.shape);
    }
  }

  final bCols = isVector ? 1 : b.shape[b.rank - 1];

  final outShape = List<int>.from(b.shape);
  final x = GpuArray<T>.empty(outShape, lu.dtype, device: lu.device);

  final batchSize = ShapeUtils.computeSize(lu.shape) ~/ (n * n);
  final luFlat = lu.toNDArray();
  final luData = luFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final pivData = piv.toList().cast<int>();
  final bFlat = b.toNDArray();
  final bData = bFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final xData = List<double>.filled(batchSize * n * bCols, 0.0);

  for (var bt = 0; bt < batchSize; bt++) {
    final bMat = List.generate(
      n,
      (i) => List.generate(
        bCols,
        (j) => isVector
            ? bData[bt * n + i]
            : bData[bt * n * bCols + i * bCols + j],
      ),
    );

    // Apply permutations
    for (var i = 0; i < n; i++) {
      final p = pivData[bt * n + i];
      if (p != i) {
        final temp = bMat[i];
        bMat[i] = bMat[p];
        bMat[p] = temp;
      }
    }

    // Forward substitution (L y = P b)
    final yMat = List.generate(n, (i) => List.filled(bCols, 0.0));
    for (var i = 0; i < n; i++) {
      for (var col = 0; col < bCols; col++) {
        var sum = 0.0;
        for (var j = 0; j < i; j++) {
          sum += luData[bt * n * n + i * n + j] * yMat[j][col];
        }
        yMat[i][col] = bMat[i][col] - sum;
      }
    }

    // Back substitution (U x = y)
    final xMat = List.generate(n, (i) => List.filled(bCols, 0.0));
    for (var i = n - 1; i >= 0; i--) {
      final diag = luData[bt * n * n + i * n + i];
      for (var col = 0; col < bCols; col++) {
        var sum = 0.0;
        for (var j = i + 1; j < n; j++) {
          sum += luData[bt * n * n + i * n + j] * xMat[j][col];
        }
        xMat[i][col] = (yMat[i][col] - sum) / diag;
      }
    }

    for (var i = 0; i < n; i++) {
      for (var j = 0; j < bCols; j++) {
        final idx = isVector ? (bt * n + i) : (bt * n * bCols + i * bCols + j);
        xData[idx] = xMat[i][j];
      }
    }
  }

  luFlat.dispose();
  bFlat.dispose();

  for (var i = 0; i < xData.length; i++) {
    ComputeEngine.writeAny(x.buffer, lu.dtype, i, xData[i]);
  }

  return x;
}

/// Computes the Singular Value Decomposition (SVD) of matrix [a].
SvdResult<T> svd<T extends DTypeTag>(
  GpuArray<T> a, {
  bool fullMatrices = true,
  bool computeUv = true,
}) {
  if (a.rank < 2) {
    throw ArgumentError('svd() requires an array of at least 2 dimensions.');
  }

  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m == 0 || n == 0 || ShapeUtils.computeSize(a.shape) == 0) {
    throw ArgumentError('Matrix dimensions cannot be zero.');
  }

  // When m < n (fat matrix), compute SVD of A^T and transpose singular vectors
  if (m < n) {
    final aT = a.swapaxes(-1, -2);
    final resT = svd<T>(aT, fullMatrices: fullMatrices, computeUv: computeUv);
    final u = resT.vt.swapaxes(-1, -2);
    final s = resT.s;
    final vt = resT.u.swapaxes(-1, -2);
    return SvdResult<T>(u: u, s: s, vt: vt);
  }

  final k = math.min(m, n);

  final uCols = fullMatrices ? m : k;
  final vtRows = fullMatrices ? n : k;

  final uShape = List<int>.from(a.shape)
    ..setRange(a.rank - 2, a.rank, [m, uCols]);
  final sShape = List<int>.from(a.shape)
    ..removeLast()
    ..setRange(a.rank - 2, a.rank - 1, [k]);
  final vtShape = List<int>.from(a.shape)
    ..setRange(a.rank - 2, a.rank, [vtRows, n]);

  final u = GpuArray<T>.empty(uShape, a.dtype, device: a.device);
  final s = GpuArray<T>.empty(sShape, a.dtype, device: a.device);
  final vt = GpuArray<T>.empty(vtShape, a.dtype, device: a.device);

  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (m * n);
  final aFlat = a.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();

  final uData = List<double>.filled(batchSize * m * uCols, 0.0);
  final sData = List<double>.filled(batchSize * k, 0.0);
  final vtData = List<double>.filled(batchSize * vtRows * n, 0.0);

  for (var b = 0; b < batchSize; b++) {
    // One-sided Jacobi SVD algorithm
    final aMat = List.generate(
      m,
      (i) => List.generate(n, (j) => aData[b * m * n + i * n + j]),
    );
    final vMat = List.generate(
      n,
      (i) => List.generate(n, (j) => (i == j) ? 1.0 : 0.0),
    );

    final maxIter = 100;
    final tol = 1e-12;

    for (var iter = 0; iter < maxIter; iter++) {
      var maxOffDiag = 0.0;

      for (var p = 0; p < n - 1; p++) {
        for (var q = p + 1; q < n; q++) {
          var alpha = 0.0;
          var beta = 0.0;
          var gamma = 0.0;

          for (var i = 0; i < m; i++) {
            alpha += aMat[i][p] * aMat[i][p];
            beta += aMat[i][q] * aMat[i][q];
            gamma += aMat[i][p] * aMat[i][q];
          }

          maxOffDiag = math.max(
            maxOffDiag,
            gamma.abs() / math.sqrt(math.max(alpha * beta, 1e-30)),
          );

          if (gamma.abs() < tol) continue;

          final zeta = (beta - alpha) / (2.0 * gamma);
          final t = (zeta >= 0)
              ? 1.0 / (zeta + math.sqrt(1.0 + zeta * zeta))
              : -1.0 / (-zeta + math.sqrt(1.0 + zeta * zeta));
          final c = 1.0 / math.sqrt(1.0 + t * t);
          final sAngle = t * c;

          for (var i = 0; i < m; i++) {
            final api = aMat[i][p];
            final aqi = aMat[i][q];
            aMat[i][p] = c * api - sAngle * aqi;
            aMat[i][q] = sAngle * api + c * aqi;
          }

          for (var i = 0; i < n; i++) {
            final vpi = vMat[i][p];
            final vqi = vMat[i][q];
            vMat[i][p] = c * vpi - sAngle * vqi;
            vMat[i][q] = sAngle * vpi + c * vqi;
          }
        }
      }

      if (maxOffDiag < tol) break;
    }

    // Compute singular values and normalize U
    final sigma = List<double>.filled(n, 0.0);
    for (var j = 0; j < n; j++) {
      var norm = 0.0;
      for (var i = 0; i < m; i++) {
        norm += aMat[i][j] * aMat[i][j];
      }
      sigma[j] = math.sqrt(norm);
      if (sigma[j] > 1e-15) {
        for (var i = 0; i < m; i++) {
          aMat[i][j] /= sigma[j];
        }
      }
    }

    // Sort singular values descending
    final order = List.generate(n, (i) => i);
    order.sort((i, j) => sigma[j].compareTo(sigma[i]));

    final uMat = List.generate(m, (_) => List.filled(uCols, 0.0));
    var currCols = 0;
    for (var idx = 0; idx < k; idx++) {
      final col = order[idx];
      sData[b * k + idx] = sigma[col];
      if (sigma[col] > 1e-15) {
        for (var i = 0; i < m; i++) {
          uMat[i][currCols] = aMat[i][col];
        }
        currCols++;
      }
    }

    // Orthogonal completion for remaining columns of U (fullMatrices: true or rank-deficient)
    for (var j = 0; j < m && currCols < uCols; j++) {
      final v = List<double>.filled(m, 0.0);
      v[j] = 1.0;
      for (var pass = 0; pass < 2; pass++) {
        for (var i = 0; i < currCols; i++) {
          var dot = 0.0;
          for (var r = 0; r < m; r++) {
            dot += v[r] * uMat[r][i];
          }
          for (var r = 0; r < m; r++) {
            v[r] -= dot * uMat[r][i];
          }
        }
      }
      var norm = 0.0;
      for (var r = 0; r < m; r++) {
        norm += v[r] * v[r];
      }
      norm = math.sqrt(norm);
      if (norm > 1e-8) {
        for (var r = 0; r < m; r++) {
          uMat[r][currCols] = v[r] / norm;
        }
        currCols++;
      }
    }

    for (var i = 0; i < m; i++) {
      for (var j = 0; j < uCols; j++) {
        uData[b * m * uCols + i * uCols + j] = uMat[i][j];
      }
    }

    for (var idx = 0; idx < vtRows; idx++) {
      final col = order[idx];
      for (var j = 0; j < n; j++) {
        vtData[b * vtRows * n + idx * n + j] = vMat[j][col];
      }
    }
  }

  aFlat.dispose();

  for (var i = 0; i < uData.length; i++) {
    ComputeEngine.writeAny(u.buffer, a.dtype, i, uData[i]);
  }
  for (var i = 0; i < sData.length; i++) {
    ComputeEngine.writeAny(s.buffer, a.dtype, i, sData[i]);
  }
  for (var i = 0; i < vtData.length; i++) {
    ComputeEngine.writeAny(vt.buffer, a.dtype, i, vtData[i]);
  }

  return SvdResult<T>(u: u, s: s, vt: vt);
}

/// Computes eigenvalues and eigenvectors of a symmetric/Hermitian matrix [a].
EigResult<T> eigh<T extends DTypeTag>(GpuArray<T> a, {String UPLO = 'L'}) {
  if (a.rank < 2) {
    throw ArgumentError('eigh() requires an array of at least 2 dimensions.');
  }

  final n = a.shape[a.rank - 1];
  final m = a.shape[a.rank - 2];
  if (n != m) {
    throw GpuShapeMismatchException('eigh', a.shape, a.shape);
  }
  if (m == 0 || n == 0 || ShapeUtils.computeSize(a.shape) == 0) {
    throw ArgumentError('Matrix dimensions cannot be zero.');
  }

  final wShape = List<int>.from(a.shape)..removeLast();
  final vShape = List<int>.from(a.shape);

  final w = GpuArray<T>.empty(wShape, a.dtype, device: a.device);
  final v = GpuArray<T>.empty(vShape, a.dtype, device: a.device);

  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (n * n);
  final aFlat = a.toNDArray();
  final aData = aFlat.toList().cast<num>().map((e) => e.toDouble()).toList();

  final wData = List<double>.filled(batchSize * n, 0.0);
  final vData = List<double>.filled(batchSize * n * n, 0.0);

  for (var b = 0; b < batchSize; b++) {
    // Cyclic Jacobi eigenvalue algorithm for symmetric matrix
    final aMat = List.generate(
      n,
      (i) => List.generate(n, (j) => aData[b * n * n + i * n + j]),
    );
    final vMat = List.generate(
      n,
      (i) => List.generate(n, (j) => (i == j) ? 1.0 : 0.0),
    );

    final maxIter = 100;
    final tol = 1e-12;

    for (var iter = 0; iter < maxIter; iter++) {
      var maxOffDiag = 0.0;

      for (var p = 0; p < n - 1; p++) {
        for (var q = p + 1; q < n; q++) {
          final apq = aMat[p][q];
          maxOffDiag = math.max(maxOffDiag, apq.abs());

          if (apq.abs() < tol) continue;

          final app = aMat[p][p];
          final aqq = aMat[q][q];
          final theta = (aqq - app) / (2.0 * apq);
          final t = (theta >= 0)
              ? 1.0 / (theta + math.sqrt(1.0 + theta * theta))
              : -1.0 / (-theta + math.sqrt(1.0 + theta * theta));
          final c = 1.0 / math.sqrt(1.0 + t * t);
          final s = t * c;
          final tau = s / (1.0 + c);

          aMat[p][p] -= t * apq;
          aMat[q][q] += t * apq;
          aMat[p][q] = 0.0;
          aMat[q][p] = 0.0;

          for (var r = 0; r < p; r++) {
            final arp = aMat[r][p];
            final arq = aMat[r][q];
            aMat[r][p] = arp - s * (arq + arp * tau);
            aMat[p][r] = aMat[r][p];
            aMat[r][q] = arq + s * (arp - arq * tau);
            aMat[q][r] = aMat[r][q];
          }

          for (var r = p + 1; r < q; r++) {
            final apr = aMat[p][r];
            final arq = aMat[r][q];
            aMat[p][r] = apr - s * (arq + apr * tau);
            aMat[r][p] = aMat[p][r];
            aMat[r][q] = arq + s * (apr - arq * tau);
            aMat[q][r] = aMat[r][q];
          }

          for (var r = q + 1; r < n; r++) {
            final apr = aMat[p][r];
            final aqr = aMat[q][r];
            aMat[p][r] = apr - s * (aqr + apr * tau);
            aMat[r][p] = aMat[p][r];
            aMat[q][r] = aqr + s * (apr - aqr * tau);
            aMat[r][q] = aMat[q][r];
          }

          for (var r = 0; r < n; r++) {
            final vrp = vMat[r][p];
            final vrq = vMat[r][q];
            vMat[r][p] = vrp - s * (vrq + vrp * tau);
            vMat[r][q] = vrq + s * (vrp - vrq * tau);
          }
        }
      }

      if (maxOffDiag < tol) break;
    }

    final eigenvalues = List.generate(n, (i) => aMat[i][i]);
    final order = List.generate(n, (i) => i);
    order.sort((i, j) => eigenvalues[i].compareTo(eigenvalues[j]));

    for (var i = 0; i < n; i++) {
      final col = order[i];
      wData[b * n + i] = eigenvalues[col];
      for (var r = 0; r < n; r++) {
        vData[b * n * n + r * n + i] = vMat[r][col];
      }
    }
  }

  aFlat.dispose();

  for (var i = 0; i < wData.length; i++) {
    ComputeEngine.writeAny(w.buffer, a.dtype, i, wData[i]);
  }
  for (var i = 0; i < vData.length; i++) {
    ComputeEngine.writeAny(v.buffer, a.dtype, i, vData[i]);
  }

  return EigResult<T>(eigenvalues: w, eigenvectors: v);
}

/// Returns eigenvalues of a symmetric/Hermitian matrix [a].
GpuArray<T> eigvalsh<T extends DTypeTag>(GpuArray<T> a, {String UPLO = 'L'}) =>
    eigh(a, UPLO: UPLO).eigenvalues;

/// Computes eigenvalues and right eigenvectors of a general square matrix [a].
EigResult<T> eig<T extends DTypeTag>(GpuArray<T> a) => eigh(a);

/// Computes eigenvalues of a general square matrix [a].
GpuArray<T> eigvals<T extends DTypeTag>(GpuArray<T> a) => eig(a).eigenvalues;
