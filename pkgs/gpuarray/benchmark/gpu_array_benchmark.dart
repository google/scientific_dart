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
import 'package:ndarray/ndarray.dart' as nd;
import 'package:gpuarray/gpuarray.dart';

void main() async {
  nd.setNumThreads(1);
  const size = 100000;
  const matSize = 128;

  await criterion(
    'GPU Array (gpuarray) Performance Benchmarks',
    (c) {
      final rand = math.Random(42);
      final rawList1 = List.generate(size, (_) => rand.nextDouble() * 100.0);
      final rawList2 = List.generate(size, (_) => rand.nextDouble() * 100.0);

      final ndArr1 = nd.NDArray<Float64>.fromList(rawList1, [
        size,
      ], nd.DType.float64);

      final gpuArr1 = GpuArray.fromList(rawList1, [size], DType.float64);
      final gpuArr2 = GpuArray.fromList(rawList2, [size], DType.float64);

      final matA = GpuArray.filled([matSize, matSize], 1.5, DType.float64);
      final matB = GpuArray.filled([matSize, matSize], 2.5, DType.float64);

      c.group('1. Arithmetic & Ufuncs [$size elements]', () {
        c.bench('GpuArray add (gpuArr1 + gpuArr2)', () {
          final res = gpuArr1 + gpuArr2;
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('GpuArray multiply (gpuArr1 * gpuArr2)', () {
          final res = gpuArr1 * gpuArr2;
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('GpuArray sqrt (gpuArr1.sqrt())', () {
          final res = gpuArr1.sqrt();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('GpuArray sin (gpuArr1.sin())', () {
          final res = gpuArr1.sin();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Reductions [$size elements]', () {
        c.bench('GpuArray sum()', () {
          final res = gpuArr1.sum();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('GpuArray mean()', () {
          final res = gpuArr1.mean();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('GpuArray min()', () {
          final res = gpuArr1.min();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('3. Linear Algebra (Matrix Multiplication)', () {
        c.bench('GpuArray matmul [$matSize x $matSize]', () {
          final res = matA.matmul(matB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(matSize * matSize * matSize));
      });

      c.group('4. Host <-> GPU Memory Transfers', () {
        c.bench('Host NDArray -> GPU (toGpu) [$size elements]', () {
          final res = ndArr1.toGpu();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('GPU -> Host NDArray (toNDArray) [$size elements]', () {
          final res = gpuArr1.toNDArray();
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report',
    ),
  );
}
