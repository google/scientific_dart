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

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);
  const size = 100000;
  const matrixRows = 1000;
  const matrixCols = 1000;

  await criterion(
    'NDArray Statistics & Reductions Benchmark Suite',
    (c) {
      final vec1d = linspace(0.0, 100.0, size, dtype: DType.float64);

      final mat2d = linspace(
        0.0,
        1000.0,
        matrixRows * matrixCols,
        dtype: DType.float64,
      ).reshape([matrixRows, matrixCols]);

      c.group('1. Basic Reductions & Moments', () {
        c.bench('mean() [1D flat 100k]', () {
          final res = mean(vec1d);
          blackhole(res);
        }, throughput: Throughput.elements(size));

        c.bench('mean(mat, axis=0) [1000x1000]', () {
          final res = mean(mat2d, axis: 0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(matrixRows * matrixCols));

        c.bench('mean(mat, axis=1) [1000x1000]', () {
          final res = mean(mat2d, axis: 1);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(matrixRows * matrixCols));

        c.bench('std() [1D flat 100k]', () {
          final res = std(vec1d);
          blackhole(res);
        }, throughput: Throughput.elements(size));

        c.bench('var_() [1D flat 100k]', () {
          final res = var_(vec1d);
          blackhole(res);
        }, throughput: Throughput.elements(size));

        final weights = linspace(1.0, 10.0, size, dtype: DType.float64);
        c.bench('average(vec, weights=w) [100k]', () {
          final res = average(vec1d, weights: weights);
          blackhole(res);
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Order Statistics (Median & Quantile)', () {
        final randVec = NDArray<Float64>.fromList(
          List.generate(size, (i) => ((i * 37) % 1000).toDouble()),
          [size],
          DType.float64,
        );

        c.bench('median() [100k items]', () {
          final res = median(randVec);
          blackhole(res);
        }, throughput: Throughput.elements(size));

        c.bench('quantile(p=0.75) [100k items]', () {
          final res = quantile(randVec, 0.75);
          blackhole(res);
        }, throughput: Throughput.elements(size));

        c.bench('ptp() (Peak-to-Peak) [100k items]', () {
          final res = ptp(randVec);
          blackhole(res);
        }, throughput: Throughput.elements(size));
      });

      c.group('3. Covariance & Correlation', () {
        const nVars = 50;
        const nObs = 500;
        final obsMat = linspace<Float64>(
          0.0,
          100.0,
          nVars * nObs,
          dtype: DType.float64,
        ).reshape([nVars, nObs]);

        c.bench('cov(X) [50 vars x 500 obs]', () {
          final res = cov(obsMat);
          blackhole(res);
          res.dispose();
        });

        c.bench('corrcoef(X) [50 vars x 500 obs]', () {
          final res = corrcoef(obsMat);
          blackhole(res);
          res.dispose();
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/stats_reductions',
    ),
  );
}
