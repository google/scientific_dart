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
  const int size1d = 1000000;
  const int dim2d = 1000;

  await criterion(
    'NDArray (Dart) Ufunc Reductions & Masked Functions Benchmark',
    (c) {
      final a1d = NDArray.fromList(
        List.generate(size1d, (i) => i % 100 * 1.0),
        [size1d],
        DType.float64,
      );
      final b1d = NDArray.fromList(
        List.generate(size1d, (i) => (i % 50 + 1) * 1.0),
        [size1d],
        DType.float64,
      );
      final mask1d = NDArray.fromList(
        List.generate(size1d, (i) => i % 2 == 0 ? 1 : 0),
        [size1d],
        DType.uint8,
      );

      final a2d = NDArray.fromList(
        List.generate(dim2d * dim2d, (i) => (i % 100) * 1.0),
        [dim2d, dim2d],
        DType.float64,
      );
      final b2d = NDArray.fromList(
        List.generate(dim2d * dim2d, (i) => (i % 50 + 1) * 1.0),
        [dim2d, dim2d],
        DType.float64,
      );

      final aOuter = NDArray.fromList(List.generate(1000, (i) => i * 1.0), [
        1000,
      ], DType.float64);
      final bOuter = NDArray.fromList(
        List.generate(1000, (i) => (i + 1) * 1.0),
        [1000],
        DType.float64,
      );

      final indicesAt = NDArray.fromList(
        List.generate(10000, (i) => (i * 97) % size1d),
        [10000],
        DType.int64,
      );
      final valsAt = NDArray.fromList(List.generate(10000, (i) => i * 1.0), [
        10000,
      ], DType.float64);

      final indicesReduceat = NDArray.fromList(
        List.generate(1000, (i) => i * 1000),
        [1000],
        DType.int64,
      );

      final out1d = NDArray.zeros([size1d], DType.float64);

      c.group('1. Reductions', () {
        c.bench('reduce(add) [1D 1M global]', () {
          final res = a1d.reduce(op: BinaryOp.add);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size1d));

        c.bench('reduce(add) [2D 1000x1000 axis:0]', () {
          final res = a2d.reduce(op: BinaryOp.add, axis: 0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(dim2d * dim2d));

        final slice100k = a1d.slice([Slice(stop: 100000)]);
        c.bench('reduce(multiply) [1D 100K global]', () {
          final res = slice100k.reduce(op: BinaryOp.multiply);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(100000));
      });

      c.group('2. Accumulate', () {
        c.bench('accumulate(add) [1D 1M cumsum]', () {
          final res = a1d.accumulate(op: BinaryOp.add);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size1d));

        c.bench('accumulate(add) [2D 1000x1000 axis:0]', () {
          final res = a2d.accumulate(op: BinaryOp.add, axis: 0);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(dim2d * dim2d));
      });

      c.group('3. Outer Product', () {
        c.bench('outer(add) [1000 x 1000]', () {
          final res = aOuter.outer(bOuter, op: BinaryOp.add);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(1000 * 1000));

        c.bench('outer(multiply) [1000 x 1000]', () {
          final res = aOuter.outer(bOuter, op: BinaryOp.multiply);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(1000 * 1000));
      });

      c.group('4. Segment Reduction & Scatter (reduceat & at)', () {
        c.bench('reduceat(add) [1M array, 1000 segments]', () {
          final res = a1d.reduceat(indicesReduceat, op: BinaryOp.add);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size1d));

        c.bench('at(add) [1M array, 10K scatter updates]', () {
          a1d.at(indicesAt, valsAt, op: BinaryOp.add);
        }, throughput: Throughput.elements(10000));
      });

      c.group('5. Masked Functions (where=)', () {
        c.bench('add(where=mask) [1D 1M contiguous]', () {
          add(a1d, b1d, where: mask1d, out: out1d);
        }, throughput: Throughput.elements(size1d));

        c.bench('multiply(where=mask) [1D 1M contiguous]', () {
          multiply(a1d, b1d, where: mask1d, out: out1d);
        }, throughput: Throughput.elements(size1d));

        final aStrided = a2d.flatten().slice([Slice(step: 2)]);
        final bStrided = b2d.flatten().slice([Slice(step: 2)]);
        final outStrided = NDArray.zeros([aStrided.shape[0]], DType.float64);

        c.bench('add(where=mask) [1D strided view step=2]', () {
          add(aStrided, bStrided, out: outStrided);
        }, throughput: Throughput.elements(aStrided.shape[0]));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/ufunc_reductions',
    ),
  );
}
