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

import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as linalg;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Linear Algebra Decompositions (gpuarray.linalg)', () {
    test('QR Decomposition (qr)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
          [3, 3],
          DType.float64,
        );

        final qrRes = linalg.qr(a);
        final q = qrRes.q;
        final r = qrRes.r;

        expect(q.shape, equals([3, 3]));
        expect(r.shape, equals([3, 3]));

        // Q * R should equal A
        final recon = q.matmul(r);
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 9; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }

        // Q * Q^T should equal Identity
        final qqt = q.matmul(q.transpose([1, 0]));
        final qqtList = qqt.toList().cast<double>();
        for (var i = 0; i < 3; i++) {
          for (var j = 0; j < 3; j++) {
            final expected = (i == j) ? 1.0 : 0.0;
            expect(qqtList[i * 3 + j], closeTo(expected, 1e-4));
          }
        }
      });
    });

    test('Cholesky Decomposition (cholesky)', () {
      ResourceScope.scope(() {
        // Symmetric positive definite matrix
        final a = GpuArray.fromList(
          [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
          [3, 3],
          DType.float64,
        );

        final l = linalg.cholesky(a);
        expect(l.shape, equals([3, 3]));

        // L * L^T should equal A
        final recon = l.matmul(l.transpose([1, 0]));
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 9; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }
      });
    });

    test('LU Decomposition (lu, lu_factor, lu_solve)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final luRes = linalg.lu(a);
        final p = luRes.p;
        final l = luRes.l;
        final u = luRes.u;

        // P^T * L * U should equal A
        final pT = p.transpose([1, 0]);
        final recon = pT.matmul(l.matmul(u));
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 9; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }

        // lu_factor and lu_solve
        final b = GpuArray.fromList([4.0, 10.0, 24.0], [3], DType.float64);
        final fact = linalg.lu_factor(a);
        final x = linalg.lu_solve(fact.lu, fact.piv, b);

        expect(x.shape, equals([3]));
        // A * x should equal b
        final ax = a.matmul(x.reshape([3, 1])).flatten();
        final axList = ax.toList().cast<double>();
        final bList = b.toList().cast<double>();
        for (var i = 0; i < 3; i++) {
          expect(axList[i], closeTo(bList[i], 1e-4));
        }
      });
    });

    test('Singular Value Decomposition (svd)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final svdRes = linalg.svd(a, fullMatrices: false);
        final u = svdRes.u;
        final s = svdRes.s;
        final vt = svdRes.vt;

        expect(u.shape, equals([3, 2]));
        expect(s.shape, equals([2]));
        expect(vt.shape, equals([2, 2]));

        // Singular values are descending positive
        final sList = s.toList().cast<double>();
        expect(sList[0], greaterThan(sList[1]));
        expect(sList[1], greaterThan(0.0));

        // U * S * Vt should equal A
        final sDiag = diag(s);

        final recon = u.matmul(sDiag).matmul(vt);
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 6; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }
      });
    });

    test('Eigendecomposition of Symmetric Matrix (eigh, eigvalsh)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [2.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );

        final eigRes = linalg.eigh(a);
        final w = eigRes.eigenvalues;
        final v = eigRes.eigenvectors;

        expect(w.shape, equals([2]));
        expect(v.shape, equals([2, 2]));

        // Eigenvalues for [[2, 1], [1, 2]] are 1 and 3
        final wList = w.toList().cast<double>();
        expect(wList[0], closeTo(1.0, 1e-4));
        expect(wList[1], closeTo(3.0, 1e-4));

        // V * W * V^T should equal A
        final wDiag = diag(w);

        final recon = v.matmul(wDiag).matmul(v.transpose([1, 0]));
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 4; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }

        final vals = linalg.eigvalsh(a);
        expect(vals.toList(), equals(w.toList()));
      });
    });
    test('SVD for Rectangular Tall & Fat Matrices (svd)', () {
      ResourceScope.scope(() {
        // 1. Fat matrix (m < n): 2 x 3
        final aFat = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );

        // Reduced SVD
        final svdFatReduced = linalg.svd(aFat, fullMatrices: false);
        expect(svdFatReduced.u.shape, equals([2, 2]));
        expect(svdFatReduced.s.shape, equals([2]));
        expect(svdFatReduced.vt.shape, equals([2, 3]));

        // Reconstruction for reduced SVD: U * diag(S) * Vt == A
        final sDiagFat = diag(svdFatReduced.s);
        final reconFatRed = svdFatReduced.u
            .matmul(sDiagFat)
            .matmul(svdFatReduced.vt);
        final reconFatRedList = reconFatRed.toList().cast<double>();
        final origFatList = aFat.toList().cast<double>();
        for (var i = 0; i < 6; i++) {
          expect(reconFatRedList[i], closeTo(origFatList[i], 1e-4));
        }

        // Full SVD for fat matrix (2 x 3): U is (2, 2), S is (2), Vt is (3, 3)
        final svdFatFull = linalg.svd(aFat, fullMatrices: true);
        expect(svdFatFull.u.shape, equals([2, 2]));
        expect(svdFatFull.s.shape, equals([2]));
        expect(svdFatFull.vt.shape, equals([3, 3]));

        // Check Vt is orthogonal: Vt * Vt^T = I_3
        final vtVtt = svdFatFull.vt.matmul(svdFatFull.vt.transpose([1, 0]));
        final vtVttList = vtVtt.toList().cast<double>();
        for (var i = 0; i < 3; i++) {
          for (var j = 0; j < 3; j++) {
            final expected = (i == j) ? 1.0 : 0.0;
            expect(vtVttList[i * 3 + j], closeTo(expected, 1e-4));
          }
        }

        // 2. Tall matrix (m > n): 3 x 2 Full SVD
        final aTall = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final svdTallFull = linalg.svd(aTall, fullMatrices: true);
        expect(svdTallFull.u.shape, equals([3, 3]));
        expect(svdTallFull.s.shape, equals([2]));
        expect(svdTallFull.vt.shape, equals([2, 2]));

        // Check U is completed orthogonally: U * U^T = I_3
        final uUt = svdTallFull.u.matmul(svdTallFull.u.transpose([1, 0]));
        final uUtList = uUt.toList().cast<double>();
        for (var i = 0; i < 3; i++) {
          for (var j = 0; j < 3; j++) {
            final expected = (i == j) ? 1.0 : 0.0;
            expect(uUtList[i * 3 + j], closeTo(expected, 1e-4));
          }
        }

        // 3. Batched fat matrices: [2, 2, 3]
        final aBatchFat = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0],
          [2, 2, 3],
          DType.float64,
        );
        final svdBatchFat = linalg.svd(aBatchFat, fullMatrices: false);
        expect(svdBatchFat.u.shape, equals([2, 2, 2]));
        expect(svdBatchFat.s.shape, equals([2, 2]));
        expect(svdBatchFat.vt.shape, equals([2, 2, 3]));
      });
    });

    test('Batched lu_solve with 1D vector RHS', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [
            2.0,
            1.0,
            1.0,
            4.0,
            3.0,
            3.0,
            8.0,
            7.0,
            9.0,
            1.0,
            2.0,
            3.0,
            0.0,
            1.0,
            4.0,
            5.0,
            6.0,
            0.0,
          ],
          [2, 3, 3],
          DType.float64,
        );
        final b = GpuArray.fromList(
          [4.0, 10.0, 24.0, 14.0, 14.0, 17.0],
          [2, 3],
          DType.float64,
        );

        final fact = linalg.lu_factor(a);
        final x = linalg.lu_solve(fact.lu, fact.piv, b);
        expect(x.shape, equals([2, 3]));
        final xList = x.toList().cast<double>();
        expect(xList[0], closeTo(1.0, 1e-4));
        expect(xList[1], closeTo(1.0, 1e-4));
        expect(xList[2], closeTo(1.0, 1e-4));
        expect(xList[3], closeTo(1.0, 1e-4));
        expect(xList[4], closeTo(2.0, 1e-4));
        expect(xList[5], closeTo(3.0, 1e-4));
      });
    });

    test(
      'lu_solve throws GpuShapeMismatchException on RHS dimension and batch shape mismatches',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final fact = linalg.lu_factor(a);

          // Vector RHS size mismatch: matrix is 2x2, vector is length 3
          final bVecMismatch = GpuArray.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          expect(
            () => linalg.lu_solve(fact.lu, fact.piv, bVecMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );

          // Matrix RHS row mismatch: matrix is 2x2, RHS is 3x2
          final bMatMismatch = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          expect(
            () => linalg.lu_solve(fact.lu, fact.piv, bMatMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );

          // Batched matrix: [2, 2, 2]
          final aBatch = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );
          final factBatch = linalg.lu_factor(aBatch);

          // Batched vector RHS batch mismatch: [3, 2] instead of [2, 2]
          final bBatchVecMismatch = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          expect(
            () =>
                linalg.lu_solve(factBatch.lu, factBatch.piv, bBatchVecMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );

          // Batched matrix RHS batch mismatch: [3, 2, 2] instead of [2, 2, 2]
          final bBatchMatMismatch = GpuArray.fromList(
            List.generate(12, (i) => i.toDouble()),
            [3, 2, 2],
            DType.float64,
          );
          expect(
            () =>
                linalg.lu_solve(factBatch.lu, factBatch.piv, bBatchMatMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );
        });
      },
    );

    test('0-Dimension matrices throw ArgumentError in decompositions', () {
      ResourceScope.scope(() {
        final empty = GpuArray.fromList(<double>[], [0, 0], DType.float64);
        expect(() => linalg.qr(empty), throwsArgumentError);
        expect(() => linalg.cholesky(empty), throwsArgumentError);
        expect(() => linalg.lu(empty), throwsArgumentError);
        expect(() => linalg.lu_factor(empty), throwsArgumentError);
        final emptyPiv = GpuArray.fromList(<int>[], [0], DType.int32);
        final emptyB = GpuArray.fromList(<double>[], [0], DType.float64);
        expect(
          () => linalg.lu_solve(empty, emptyPiv, emptyB),
          throwsArgumentError,
        );
        expect(() => linalg.svd(empty), throwsArgumentError);
        expect(() => linalg.eigh(empty), throwsArgumentError);
      });
    });
  });
}
