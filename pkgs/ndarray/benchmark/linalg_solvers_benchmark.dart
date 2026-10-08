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
import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);

  await criterion(
    'NDArray Linear Algebra Solvers & Invariants Benchmark Suite',
    (c) {
      // Helper to generate well-conditioned invertible matrices
      NDArray<Float64> makeInvertible(int n) {
        final rand = math.Random(42);
        final a = NDArray.zeros([n, n], DType.float64);
        for (var i = 0; i < n; i++) {
          for (var j = 0; j < n; j++) {
            a.setCell([i, j], (rand.nextDouble() - 0.5) * 2.0);
            if (i == j) {
              a.setCell([i, j], a.getCell([i, j]) + n.toDouble());
            }
          }
        }
        return a;
      }

      c.group('1. Linear System Solvers (LAPACK dgesv & dgels)', () {
        for (final n in [50, 100, 200]) {
          final A = makeInvertible(n);
          final b = NDArray<AnySpec>.ones([n, 1], DType.float64);

          c.bench(
            'solve(A, b) [${n}x$n]',
            () {
              final x = solve(A, b);
              blackhole(x);
              x.dispose();
            },
            throughput: Throughput.elements(n * n * n), // O(N^3) FLOPs
          );
        }

        // Overdetermined Least Squares: 200 rows x 50 columns
        const m = 200;
        const k = 50;
        final rand = math.Random(42);
        final aRect = NDArray<AnySpec>.fromList(
          List.generate(m * k, (_) => rand.nextDouble()),
          [m, k],
          DType.float64,
        );
        final bRect = NDArray<AnySpec>.ones([m, 1], DType.float64);

        c.bench('lstsq(A, b) [200x50]', () {
          final res = lstsq(aRect, bRect);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(m * k * k));
      });

      c.group('2. Determinants, Invariants & Matrix Norms', () {
        final mat100 = makeInvertible(100);

        c.bench('det(A) [100x100]', () {
          final d = det(mat100);
          blackhole(d);
        }, throughput: Throughput.elements(100 * 100 * 100));

        c.bench('slogdet(A) [100x100]', () {
          final res = slogdet(mat100);
          blackhole(res);
        }, throughput: Throughput.elements(100 * 100 * 100));

        c.bench('pinv(A) (Moore-Penrose SVD) [100x100]', () {
          final res = pinv(mat100);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(100 * 100 * 100));

        c.bench('norm(A) (Frobenius matrix norm) [100x100]', () {
          final nVal = norm(mat100);
          blackhole(nVal);
          nVal.dispose();
        }, throughput: Throughput.elements(100 * 100));

        final mat50 = makeInvertible(50);
        c.bench('matrix_power(A, 5) [50x50]', () {
          final pMat = matrix_power(mat50, 5);
          blackhole(pMat);
          pMat.dispose();
        }, throughput: Throughput.elements(50 * 50 * 50 * 4));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/linalg_solvers',
    ),
  );
}
