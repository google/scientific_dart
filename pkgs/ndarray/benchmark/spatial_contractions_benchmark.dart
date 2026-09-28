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

  await criterion(
    'NDArray Spatial Metrics, Distance & Tensor Contractions Benchmark Suite',
    (c) {
      const numPoints = 500;
      const pointDim = 20;

      final pointsA = linspace<double>(
        0.0,
        10.0,
        numPoints * pointDim,
        dtype: DType.float64,
      ).reshape([numPoints, pointDim]);

      final pointsB = linspace<double>(
        5.0,
        15.0,
        numPoints * pointDim,
        dtype: DType.float64,
      ).reshape([numPoints, pointDim]);

      c.group('1. Pairwise Spatial Distance Metrics', () {
        c.bench('cdist(A, B, euclidean) [500x20 vs 500x20 -> 500x500]', () {
          final res = cdist(pointsA, pointsB, metric: DistanceMetric.euclidean);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(numPoints * numPoints));

        c.bench('cdist(A, B, cosine) [500x20 vs 500x20 -> 500x500]', () {
          final res = cdist(pointsA, pointsB, metric: DistanceMetric.cosine);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(numPoints * numPoints));

        c.bench('pdist(A, euclidean) [500x20 -> 124,750 pairs]', () {
          final res = pdist(pointsA, metric: DistanceMetric.euclidean);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements((numPoints * (numPoints - 1)) ~/ 2));
      });

      c.group('2. Tensor Dot & Contractions', () {
        const matDim = 200;
        final matA = linspace<double>(
          0.0,
          10.0,
          matDim * matDim,
          dtype: DType.float64,
        ).reshape([matDim, matDim]);

        final matB = linspace<double>(
          5.0,
          15.0,
          matDim * matDim,
          dtype: DType.float64,
        ).reshape([matDim, matDim]);

        final einsumSpec = EinsumSubscripts.parse('ij,jk->ik');
        c.bench('einsum("ij,jk->ik", [A, B]) [200x200 @ 200x200]', () {
          final res = einsum(einsumSpec, [matA, matB]);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(matDim * matDim));

        const tensorDim = 40;
        final tA = linspace<double>(
          0.0,
          1.0,
          tensorDim * tensorDim * tensorDim,
          dtype: DType.float64,
        ).reshape([tensorDim, tensorDim, tensorDim]);
        final tB = linspace<double>(
          0.0,
          1.0,
          tensorDim * tensorDim * tensorDim,
          dtype: DType.float64,
        ).reshape([tensorDim, tensorDim, tensorDim]);

        c.bench('tensordot(tA, tB, axes=1) [40x40x40 @ 40x40x40]', () {
          final res = tensordot(tA, tB, axes: const TensordotAxes.count(1));
          blackhole(res);
          res.dispose();
        });

        const vLen = 1000;
        final vA = linspace<double>(0.0, 10.0, vLen, dtype: DType.float64);
        final vB = linspace<double>(5.0, 15.0, vLen, dtype: DType.float64);

        c.bench('outer(vA, vB) [1000 x 1000 -> 1M]', () {
          final res = outer(vA, vB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(vLen * vLen));

        const krA = 50;
        const krB = 10;
        final kA = linspace<double>(
          0.0,
          1.0,
          krA * krA,
          dtype: DType.float64,
        ).reshape([krA, krA]);
        final kB = linspace<double>(
          0.0,
          1.0,
          krB * krB,
          dtype: DType.float64,
        ).reshape([krB, krB]);

        c.bench('kron(kA, kB) [50x50 x 10x10 -> 500x500]', () {
          final res = kron(kA, kB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(500 * 500));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/spatial_contractions',
    ),
  );
}
