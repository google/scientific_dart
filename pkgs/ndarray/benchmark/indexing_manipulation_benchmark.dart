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
  const matrixDim = 500; // 500x500 matrix = 250,000 elements

  await criterion(
    'NDArray Indexing, Slicing & Manipulation Benchmark Suite',
    (c) {
      final a1d = linspace<DTypeTag>(0.0, 100.0, size, dtype: DType.float64);
      final mat2d = NDArray<DTypeTag>.arange(
        0.0,
        (matrixDim * matrixDim).toDouble(),
        dtype: DType.float64,
      ).reshape([matrixDim, matrixDim]);

      c.group('1. Views, Slicing & Reshaping (Zero-Copy Metadata)', () {
        c.bench('1D Strided Slice a[::2] [size=100,000]', () {
          final view = a1d.slice([const Slice(step: 2)]);
          blackhole(view.shape);
          view.dispose();
        }, throughput: Throughput.elements(size ~/ 2));

        c.bench(
          '2D Multi-Axis Strided Slice mat[10:490:2, 20:480:3] [500x500]',
          () {
            final view = mat2d.slice([
              const Slice(start: 10, stop: 490, step: 2),
              const Slice(start: 20, stop: 480, step: 3),
            ]);
            blackhole(view.shape);
            view.dispose();
          },
          throughput: Throughput.elements(240 * 153),
        );

        c.bench('2D Transpose (strides swap) [500x500]', () {
          final view = mat2d.transposed;
          blackhole(view.shape);
          view.dispose();
        }, throughput: Throughput.elements(matrixDim * matrixDim));

        c.bench('Reshape [500, 500] -> [250, 1000]', () {
          final view = mat2d.reshape([250, 1000]);
          blackhole(view.shape);
          view.dispose();
        }, throughput: Throughput.elements(matrixDim * matrixDim));
      });

      c.group('2. Advanced Indexing & Selection', () {
        final indices = NDArray<DTypeTag>.arange(
          0.0,
          matrixDim.toDouble(),
          dtype: DType.int32,
        ).reshape([matrixDim, 1]);

        c.bench('take_along_axis [500x500, axis=0]', () {
          final res = take_along_axis(mat2d, indices, 0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(matrixDim));

        final putValues = NDArray<DTypeTag>.ones([matrixDim, 1], DType.float64);
        c.bench('put_along_axis [500x500, axis=0]', () {
          put_along_axis(mat2d, indices, putValues, 0);
          blackhole(mat2d);
        }, throughput: Throughput.elements(matrixDim));

        c.bench('diag() Extraction [500x500]', () {
          final d = diag(mat2d);
          blackhole(d);
          d.dispose();
        }, throughput: Throughput.elements(matrixDim));

        final choices = [
          NDArray<DTypeTag>.zeros([10000], DType.float64),
          NDArray<DTypeTag>.ones([10000], DType.float64),
          linspace<DTypeTag>(0.0, 10.0, 10000, dtype: DType.float64),
        ];
        final selector = NDArray<DTypeTag>.fromList(
          List.generate(10000, (i) => i % 3),
          [10000],
          DType.int32,
        );

        c.bench('choose() Multi-Array Selection [size=10,000, 3 choices]', () {
          final res = choose(selector, choices);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(10000));
      });

      c.group('3. Array Assembly & Joining', () {
        final blockA = NDArray.ones([250, 500], DType.float64);
        final blockB = NDArray.zeros([250, 500], DType.float64);

        c.bench(
          'concatenate([A, B], axis=0) [250x500 + 250x500 -> 500x500]',
          () {
            final res = concatenate([blockA, blockB], axis: 0);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(500 * 500),
        );

        c.bench('stack([A, B], axis=0) [2 x 250x500 -> 2x250x500]', () {
          final res = stack([blockA, blockB], axis: 0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(250 * 500 * 2));

        final smallTile = NDArray<DTypeTag>.ones([50, 50], DType.float64);
        c.bench('tile([50, 50], [10, 10]) -> [500, 500]', () {
          final res = tile(smallTile, [10, 10]);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(500 * 500));

        final repVec = linspace<DTypeTag>(
          0.0,
          10.0,
          1000,
          dtype: DType.float64,
        );
        c.bench('repeat([1000], repeats=50) -> [50,000]', () {
          final res = repeat(repVec, [50]);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(50000));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/indexing_manipulation',
    ),
  );
}
