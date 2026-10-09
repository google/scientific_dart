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
  const size = 100000;

  await criterion(
    'NDArray Eigenvalues, Matrix Chains, Partitioning & Search Benchmark Suite',
    (c) {
      final rand = math.Random(42);

      // Build a 100x100 symmetric positive-definite matrix for eigh / eigvalsh / cond
      const symDim = 100;
      final symData = List<double>.filled(symDim * symDim, 0.0);
      for (var i = 0; i < symDim; i++) {
        for (var j = i; j < symDim; j++) {
          final v = (rand.nextDouble() - 0.5) * 2.0;
          symData[i * symDim + j] = i == j ? v + symDim.toDouble() : v;
          symData[j * symDim + i] = symData[i * symDim + j];
        }
      }
      final symMat = NDArray<Float64>.fromList(symData, [
        symDim,
        symDim,
      ], DType.float64);

      // Build a 60x60 general square matrix for eig / eigvals
      const genDim = 60;
      final genMat = NDArray<Float64>.fromList(
        List.generate(genDim * genDim, (_) => rand.nextDouble() - 0.5),
        [genDim, genDim],
        DType.float64,
      );

      // Matrix chain for multi_dot: [100x10] * [10x500] * [500x20] * [20x100]
      final chainA = NDArray<Float64>.ones([100, 10], DType.float64);
      final chainB = NDArray<Float64>.ones([10, 500], DType.float64);
      final chainC = NDArray<Float64>.ones([500, 20], DType.float64);
      final chainD = NDArray<Float64>.ones([20, 100], DType.float64);

      final innerA = NDArray<Float64>.ones([200, 100], DType.float64);
      final innerB = NDArray<Float64>.ones([200, 100], DType.float64);
      final vdotA = linspace(0.0, 10.0, size, dtype: DType.float64);
      final vdotB = linspace(1.0, 11.0, size, dtype: DType.float64);

      c.group('1. Eigenvalues, Condition Numbers & Matrix Chains', () {
        c.bench('eigh(A) [100x100 symmetric]', () {
          final res = eigh(symMat);
          blackhole(res);
          res.eigenvalues.dispose();
          res.eigenvectors.dispose();
        });

        c.bench('eigvalsh(A) [100x100 symmetric]', () {
          final res = eigvalsh(symMat);
          blackhole(res);
          res.dispose();
        });

        c.bench('eig(A) [60x60 general]', () {
          final res = eig(genMat);
          blackhole(res);
          res.eigenvalues.dispose();
          res.eigenvectors.dispose();
        });

        c.bench('eigvals(A) [60x60 general]', () {
          final res = eigvals(genMat);
          blackhole(res);
          res.dispose();
        });

        c.bench('cond(A) [100x100]', () {
          final res = cond(symMat);
          blackhole(res);
          res.dispose();
        });

        c.bench('multi_dot([100x10, 10x500, 500x20, 20x100])', () {
          final res = multi_dot<Float64>([chainA, chainB, chainC, chainD]);
          blackhole(res);
          res.dispose();
        });

        c.bench('inner(A, B) [200x100, 200x100 -> 200x200]', () {
          final res = inner<Float64>(innerA, innerB);
          blackhole(res);
          res.dispose();
        });

        c.bench('vdot(a, b) [100k Float64]', () {
          final res = vdot<Float64>(vdotA, vdotB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Order Selection, Partitioning & Binary Search', () {
        final randVec = NDArray<Float64>.fromList(
          List.generate(size, (_) => rand.nextDouble() * 1000.0),
          [size],
          DType.float64,
        );

        c.bench('partition(arr, kth=50000) [100k Float64]', () {
          final res = partition(randVec, 50000);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('argpartition(arr, kth=50000) [100k Float64]', () {
          final res = argpartition<Float64>(randVec, 50000);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        final sortedTarget = linspace(0.0, 1000.0, size, dtype: DType.float64);
        c.bench('searchsorted(sorted, queries) [100k in 100k]', () {
          final res = searchsorted<Float64>(sortedTarget, randVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        final sparseArr = NDArray<Float64>.fromList(
          List.generate(size, (i) => i % 10 == 0 ? (i + 1).toDouble() : 0.0),
          [size],
          DType.float64,
        );

        c.bench('nonzero(sparse) [100k Float64, 10% nonzero]', () {
          final res = nonzero<Float64>(sparseArr);
          for (final idx in res) {
            blackhole(idx);
            idx.dispose();
          }
        }, throughput: Throughput.elements(size));

        c.bench('argwhere(sparse) [100k Float64, 10% nonzero]', () {
          final res = argwhere<Float64>(sparseArr);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('count_nonzero(sparse) [100k Float64]', () {
          final res = count_nonzero<Float64>(sparseArr);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('argmax(arr) [100k Float64]', () {
          final res = argmax<Float64>(randVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('argmin(arr) [100k Float64]', () {
          final res = argmin<Float64>(randVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/eigen_selection',
    ),
  );
}
