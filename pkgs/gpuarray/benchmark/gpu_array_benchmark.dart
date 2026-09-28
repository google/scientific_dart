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

import 'dart:io';
import 'dart:math' as math;

import 'package:criterion/criterion.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:ndarray/ndarray.dart' as nd;

Future<void> main([List<String> args = const []]) async {
  if (args.isNotEmpty) {
    stderr.writeln('Usage: dart benchmark/gpu_array_benchmark.dart');
    exitCode = 2;
    return;
  }

  nd.setNumThreads(1);
  const size = 100000;
  const matrixSize = 128;

  final random = math.Random(42);
  final rawList1 = List<double>.generate(
    size,
    (_) => random.nextDouble() * 100.0,
  );
  final rawList2 = List<double>.generate(
    size,
    (_) => random.nextDouble() * 100.0,
  );

  final hostArray1 = nd.NDArray<Float64>.fromList(rawList1, [
    size,
  ], nd.DType.float64);

  final gpuArray1 = GpuArray.fromList(rawList1, [size], DType.float64);
  final gpuArray2 = GpuArray.fromList(rawList2, [size], DType.float64);

  final matrixA = GpuArray.filled([matrixSize, matrixSize], 1.5, DType.float64);
  final matrixB = GpuArray.filled([matrixSize, matrixSize], 2.5, DType.float64);

  try {
    await criterion(
      'GPU Array (gpuarray) Performance Benchmarks',
      (c) {
        c.group('1. Arithmetic & Ufuncs [$size elements]', () {
          c.bench('GpuArray add (gpuArray1 + gpuArray2)', () {
            final result = gpuArray1 + gpuArray2;
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));

          c.bench('GpuArray multiply (gpuArray1 * gpuArray2)', () {
            final result = gpuArray1 * gpuArray2;
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));

          c.bench('GpuArray sqrt (gpuArray1.sqrt())', () {
            final result = gpuArray1.sqrt();
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));

          c.bench('GpuArray sin (gpuArray1.sin())', () {
            final result = gpuArray1.sin();
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));
        });

        c.group('2. Reductions [$size elements]', () {
          c.bench('GpuArray sum()', () {
            final result = gpuArray1.sum();
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));

          c.bench('GpuArray mean()', () {
            final result = gpuArray1.mean();
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));

          c.bench('GpuArray min()', () {
            final result = gpuArray1.min();
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));
        });

        c.group('3. Linear Algebra (Matrix Multiplication)', () {
          c.bench(
            'GpuArray matmul [$matrixSize x $matrixSize]',
            () {
              final result = matrixA.matmul(matrixB);
              blackhole(result);
              result.dispose();
            },
            throughput: Throughput.elements(
              matrixSize * matrixSize * matrixSize,
            ),
          );
        });

        c.group('4. Host <-> GPU Memory Transfers', () {
          c.bench('Host NDArray -> GPU (toGpu) [$size elements]', () {
            final result = hostArray1.toGpu();
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));

          c.bench('GPU -> Host NDArray (toNDArray) [$size elements]', () {
            final result = gpuArray1.toNDArray();
            blackhole(result);
            result.dispose();
          }, throughput: Throughput.elements(size));
        });
      },
      config: CriterionConfig(
        generateHtmlReport: true,
        exportJson: true,
        reportDir: 'benchmark/report',
      ),
    );
  } finally {
    hostArray1.dispose();
    gpuArray1.dispose();
    gpuArray2.dispose();
    matrixA.dispose();
    matrixB.dispose();
  }
}
