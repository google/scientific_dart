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
import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as linalg;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Linear Algebra Solvers & Metrics (gpuarray.linalg)', () {
    test('Linear system solver (solve) and matrix inverse (inv)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [3.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );

        final b = GpuArray.fromList([9.0, 8.0], [2], DType.float64);

        // Solve A x = b -> x = [2, 3]
        final x = linalg.solve(a, b);
        expect(x.shape, equals([2]));
        final xList = x.toList().cast<double>();
        expect(xList[0], closeTo(2.0, 1e-4));
        expect(xList[1], closeTo(3.0, 1e-4));

        // Matrix inverse A * A^-1 = I
        final aInv = linalg.inv(a);
        final identity = a.matmul(aInv);
        final idList = identity.toList().cast<double>();
        expect(idList[0], closeTo(1.0, 1e-4));
        expect(idList[1], closeTo(0.0, 1e-4));
        expect(idList[2], closeTo(0.0, 1e-4));
        expect(idList[3], closeTo(1.0, 1e-4));
      });
    });

    test('Pseudo-inverse (pinv)', () {
      ResourceScope.scope(() {
        // Non-square matrix (3x2)
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final aPinvarr = linalg.pinv(a);
        expect(aPinvarr.shape, equals([2, 3]));

        // Penrose condition 1: A * A^+ * A = A
        final recon = a.matmul(aPinvarr).matmul(a);
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();
        for (var i = 0; i < 6; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }
      });
    });

    test('Determinant and Slogdet (det, slogdet)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // det(A) = 1*4 - 2*3 = -2
        final d = linalg.det(a) as double;
        expect(d, closeTo(-2.0, 1e-4));

        final slog = linalg.slogdet(a);
        expect(slog.sign.item(), closeTo(-1.0, 1e-4));
        expect(slog.logabsdet.item(), closeTo(math.log(2.0), 1e-4));
      });
    });

    test('Matrix power and matrix rank (matrix_power, matrix_rank)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // A^2 = [[7, 10], [15, 22]]
        final a2 = linalg.matrix_power(a, 2);
        final a2List = a2.toList().cast<double>();
        expect(a2List[0], closeTo(7.0, 1e-4));
        expect(a2List[1], closeTo(10.0, 1e-4));
        expect(a2List[2], closeTo(15.0, 1e-4));
        expect(a2List[3], closeTo(22.0, 1e-4));

        // A^0 = Identity
        final a0 = linalg.matrix_power(a, 0);
        final a0List = a0.toList().cast<double>();
        expect(a0List, equals([1.0, 0.0, 0.0, 1.0]));

        // Matrix rank
        final rank = linalg.matrix_rank(a) as int;
        expect(rank, equals(2));

        // Rank deficient matrix
        final rankDef = GpuArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final rank1 = linalg.matrix_rank(rankDef) as int;
        expect(rank1, equals(1));
      });
    });

    test('Matrix norm and condition number (norm, cond)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // Frobenius norm: sqrt(1 + 4 + 9 + 16) = sqrt(30) = 5.4772
        final n = linalg.norm(a) as double;
        expect(n, closeTo(math.sqrt(30.0), 1e-4));

        // Condition number
        final c = linalg.cond(a) as double;
        expect(c, greaterThan(1.0));
      });
    });
    test('Optimized det and slogdet edge cases and batching', () {
      ResourceScope.scope(() {
        // 3x3 Matrix
        // [[6, 1, 1],
        //  [4, -2, 5],
        //  [2, 8, 7]]
        // det = 6*(-14 - 40) - 1*(28 - 10) + 1*(32 - (-4)) = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306
        final a3 = GpuArray.fromList(
          [6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0],
          [3, 3],
          DType.float64,
        );

        final d3 = linalg.det(a3) as double;
        expect(d3, closeTo(-306.0, 1e-4));

        final slog3 = linalg.slogdet(a3);
        expect(slog3.sign.item(), closeTo(-1.0, 1e-4));
        expect(slog3.logabsdet.item(), closeTo(math.log(306.0), 1e-4));

        // Singular matrix (det == 0)
        final singular = GpuArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final dSing = linalg.det(singular) as double;
        expect(dSing.abs(), closeTo(0.0, 1e-10));

        final slogSing = linalg.slogdet(singular);
        expect(slogSing.sign.item(), closeTo(0.0, 1e-4));
        expect(slogSing.logabsdet.item(), equals(double.negativeInfinity));

        // Batched determinant: 2 matrices of 2x2
        final aBatch = GpuArray.fromList(
          [
            1.0, 2.0, 3.0, 4.0, // det = -2
            5.0, 2.0, 1.0, 3.0, // det = 15 - 2 = 13
          ],
          [2, 2, 2],
          DType.float64,
        );

        final dBatch = linalg.det(aBatch) as GpuArray;
        expect(dBatch.shape, equals([2]));
        final dBatchList = dBatch.toList().cast<double>();
        expect(dBatchList[0], closeTo(-2.0, 1e-4));
        expect(dBatchList[1], closeTo(13.0, 1e-4));
      });
    });

    test('Batched linear solve with 1D vector RHS', () {
      ResourceScope.scope(() {
        // A0: [[2, 1, 1], [4, 3, 3], [8, 7, 9]], b0: [4, 10, 24] -> x0: [1, 1, 1]
        // A1: [[1, 2, 3], [0, 1, 4], [5, 6, 0]], b1: [14, 14, 17] -> x1: [1, 2, 3]
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

        final x = linalg.solve(a, b);
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
      'solve throws GpuShapeMismatchException on RHS dimension and batch shape mismatches',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );

          // Vector RHS size mismatch: matrix is 2x2, vector is length 3
          final bVecMismatch = GpuArray.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          expect(
            () => linalg.solve(a, bVecMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );

          // Matrix RHS row mismatch: matrix is 2x2, RHS is 3x2
          final bMatMismatch = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          expect(
            () => linalg.solve(a, bMatMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );

          // Batched matrix: [2, 2, 2]
          final aBatch = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );

          // Batched vector RHS batch mismatch: [3, 2] instead of [2, 2]
          final bBatchVecMismatch = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          expect(
            () => linalg.solve(aBatch, bBatchVecMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );

          // Batched matrix RHS batch mismatch: [3, 2, 2] instead of [2, 2, 2]
          final bBatchMatMismatch = GpuArray.fromList(
            List.generate(12, (i) => i.toDouble()),
            [3, 2, 2],
            DType.float64,
          );
          expect(
            () => linalg.solve(aBatch, bBatchMatMismatch),
            throwsA(isA<GpuShapeMismatchException>()),
          );
        });
      },
    );

    test('0-Dimension matrices throw ArgumentError in solvers', () {
      ResourceScope.scope(() {
        final empty = GpuArray.fromList(<double>[], [0, 0], DType.float64);
        final emptyVec = GpuArray.fromList(<double>[], [0], DType.float64);
        expect(() => linalg.solve(empty, emptyVec), throwsArgumentError);
        expect(() => linalg.inv(empty), throwsArgumentError);
        expect(() => linalg.pinv(empty), throwsArgumentError);
        expect(() => linalg.det(empty), throwsArgumentError);
        expect(() => linalg.slogdet(empty), throwsArgumentError);
        expect(() => linalg.matrix_power(empty, 2), throwsArgumentError);
        expect(() => linalg.matrix_rank(empty), throwsArgumentError);
        expect(() => linalg.cond(empty), throwsArgumentError);
      });
    });
  });
}
