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
  const dim = 500;

  await criterion(
    'NDArray NaN Reductions, Cumulative Scans, Floating-Point & Grids Benchmark Suite',
    (c) {
      // Array with 10% NaN values
      final nanVec = NDArray<Float64>.fromList(
        List.generate(
          size,
          (i) => i % 10 == 0 ? double.nan : (i % 100).toDouble() + 1.0,
        ),
        [size],
        DType.float64,
      );

      final cleanVec = linspace(1.00001, 1.00002, size, dtype: DType.float64);
      final cleanVecJitter = linspace(
        1.000010001,
        1.000020001,
        size,
        dtype: DType.float64,
      );
      final mat2d = linspace(
        0.0,
        100.0,
        dim * dim,
        dtype: DType.float64,
      ).reshape([dim, dim]);

      c.group('1. NaN-Resilient Statistical Reductions (10% NaN)', () {
        c.bench('nansum(arr) [100k Float64]', () {
          final res = nansum(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('nanmean(arr) [100k Float64]', () {
          final res = nanmean(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('nanstd(arr) [100k Float64]', () {
          final res = nanstd<Float64>(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('nanvar(arr) [100k Float64]', () {
          final res = nanvar<Float64>(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('nanmin(arr) [100k Float64]', () {
          final res = nanmin(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('nanmax(arr) [100k Float64]', () {
          final res = nanmax(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Cumulative Scans (cumsum & cumprod)', () {
        c.bench('cumsum(arr) [100k Float64]', () {
          final res = cumsum(cleanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('cumsum(mat, axis=0) [500x500 Float64]', () {
          final res = cumsum(mat2d, axis: 0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(dim * dim));

        c.bench('cumsum(mat, axis=1) [500x500 Float64]', () {
          final res = cumsum(mat2d, axis: 1);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(dim * dim));

        c.bench('cumprod(arr) [100k Float64]', () {
          final res = cumprod(cleanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('3. Floating-Point Inspection, Tolerances & Mesh Grids', () {
        c.bench('isnan(arr) [100k Float64]', () {
          final res = isnan<Float64>(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('isfinite(arr) [100k Float64]', () {
          final res = isfinite<Float64>(nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('isClose(a, b) [100k Float64]', () {
          final res = isClose<Float64, Float64>(cleanVec, cleanVecJitter);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('allClose(a, b) [100k Float64]', () {
          final res = allClose<Float64, Float64>(cleanVec, cleanVecJitter);
          blackhole(res);
        }, throughput: Throughput.elements(size));

        c.bench('copysign(a, b) [100k Float64]', () {
          final res = copysign<Float64>(cleanVec, nanVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('mgrid([0:500, 0:500]) [2x500x500 dense grid]', () {
          final res = mgrid([
            GridRange(0.0, 500.0, step: 1.0),
            GridRange(0.0, 500.0, step: 1.0),
          ]);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(2 * dim * dim));

        final rowVec = linspace(0.0, 10.0, dim, dtype: DType.float64);
        c.bench('broadcastTo(vec, [500, 500]) [zero-copy view]', () {
          final view = broadcastTo<Float64>(rowVec, [dim, dim]);
          blackhole(view.shape);
          view.dispose();
        }, throughput: Throughput.elements(dim * dim));

        c.bench('slidingWindowView(arr, [16]) [100k 1D window view]', () {
          final view = slidingWindowView<Float64>(cleanVec, [16]);
          blackhole(view.shape);
          view.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('tril(mat) [500x500]', () {
          final res = tril(mat2d);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(dim * dim));

        c.bench('triu(mat) [500x500]', () {
          final res = triu(mat2d);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(dim * dim));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/nan_cumulative_grids',
    ),
  );
}
