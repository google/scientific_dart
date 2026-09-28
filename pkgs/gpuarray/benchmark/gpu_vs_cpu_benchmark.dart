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
import 'dart:typed_data';

import 'package:gpuarray/gpuarray.dart';
import 'package:ndarray/ndarray.dart' as nd;

/// Performance metrics recorded for a single CPU vs. GPU benchmark workload.
final class BenchmarkResult {
  /// Human-readable name of the benchmark task.
  final String name;

  /// Workload shape or element count description.
  final String size;

  /// Average CPU execution time in milliseconds.
  final double cpuMs;

  /// Average GPU execution time in milliseconds.
  final double gpuMs;

  /// Relative speedup of GPU over CPU (`cpuMs / gpuMs`).
  final double speedup;

  /// Achieved CPU floating-point throughput in GFLOP/s, if applicable.
  final double? cpuGflops;

  /// Achieved GPU floating-point throughput in GFLOP/s, if applicable.
  final double? gpuGflops;

  /// Achieved CPU memory bandwidth in GB/s, if applicable.
  final double? cpuThroughputGb;

  /// Achieved GPU memory bandwidth in GB/s, if applicable.
  final double? gpuThroughputGb;

  /// Creates a [BenchmarkResult] summary entry.
  BenchmarkResult({
    required this.name,
    required this.size,
    required this.cpuMs,
    required this.gpuMs,
    required this.speedup,
    this.cpuGflops,
    this.gpuGflops,
    this.cpuThroughputGb,
    this.gpuThroughputGb,
  });
}

/// Orchestrates comparative CPU (`package:ndarray`) and GPU (`package:gpuarray`) benchmarks.
final class BenchmarkRunner {
  /// Target GPU hardware device under test.
  final GpuDevice gpuDevice;

  /// Collected benchmark results across all suites.
  final List<BenchmarkResult> results = [];

  /// Creates a [BenchmarkRunner] bound to [gpuDevice].
  BenchmarkRunner(this.gpuDevice);

  /// Initializes a WebGPU hardware device and creates a [BenchmarkRunner].
  static Future<BenchmarkRunner> create() async {
    final device = await createWebGpuDevice(
      name: 'WebGPU Physical Hardware Device',
    );
    return BenchmarkRunner(device);
  }

  double _measure(
    void Function() callback, {
    int warmup = 2,
    int iterations = 5,
  }) {
    for (var i = 0; i < warmup; i++) {
      callback();
    }
    final stopwatch = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      callback();
    }
    stopwatch.stop();
    return stopwatch.elapsedMicroseconds / (iterations * 1000.0);
  }

  /// Runs 2D matrix multiplication (GEMM) benchmarks across matrix sizes.
  void runGemmBenchmarks() {
    print(
      '\n===================================================================================',
    );
    print('  BENCHMARK 1: 2D Matrix Multiplication (GEMM)');
    print('  Hardware Kernel: Tiled 16x16 Shared-Memory WGSL Compute Pipeline');
    print('  Operation: C = A @ B (Float32)');
    print(
      '===================================================================================',
    );

    final sizes = [512, 1024, 2048];

    for (final n in sizes) {
      final totalElements = n * n;
      final rawDataA = Float32List(totalElements);
      final rawDataB = Float32List(totalElements);
      final random = math.Random(42);
      for (var i = 0; i < totalElements; i++) {
        rawDataA[i] = random.nextDouble();
        rawDataB[i] = random.nextDouble();
      }

      final flops = 2.0 * n * n * n;

      // 1. CPU (ndarray - OpenBLAS cblas_sgemm)
      final cpuA = nd.NDArray<nd.Float32>.fromList(rawDataA, [
        n,
        n,
      ], nd.DType.float32);
      final cpuB = nd.NDArray<nd.Float32>.fromList(rawDataB, [
        n,
        n,
      ], nd.DType.float32);

      final cpuMs = _measure(
        () {
          final product = nd.matmul(cpuA, cpuB);
          product.dispose();
        },
        warmup: 2,
        iterations: n >= 2048 ? 2 : 4,
      );

      final cpuGflops = flops / (cpuMs * 1e6);

      // 2. GPU (gpuarray - WebGPU Hardware Driver)
      final gpuA = GpuArray.fromList(
        rawDataA,
        [n, n],
        DType.float32,
        device: gpuDevice,
      );
      final gpuB = GpuArray.fromList(
        rawDataB,
        [n, n],
        DType.float32,
        device: gpuDevice,
      );

      final gpuMs = _measure(
        () {
          final result = gpuA.matmul(gpuB);
          final downloaded = result.toNDArray();
          downloaded.dispose();
          result.dispose();
        },
        warmup: 3,
        iterations: n >= 2048 ? 4 : 8,
      );

      final gpuGflops = flops / (gpuMs * 1e6);
      final speedup = cpuMs / gpuMs;

      results.add(
        BenchmarkResult(
          name: 'Matrix Multiplication (GEMM)',
          size: '${n}x$n',
          cpuMs: cpuMs,
          gpuMs: gpuMs,
          speedup: speedup,
          cpuGflops: cpuGflops,
          gpuGflops: gpuGflops,
        ),
      );

      print(
        '  [$n x $n] CPU: ${cpuMs.toStringAsFixed(2)} ms (${cpuGflops.toStringAsFixed(1)} GFLOP/s) | '
        'GPU: ${gpuMs.toStringAsFixed(2)} ms (${gpuGflops.toStringAsFixed(1)} GFLOP/s) | '
        'Speedup: ${speedup.toStringAsFixed(2)}x',
      );

      cpuA.dispose();
      cpuB.dispose();
      gpuA.dispose();
      gpuB.dispose();
    }
  }

  /// Runs JIT-fused elementwise pipeline benchmarks (`SiLU(2.5 * X + 1.2)`).
  void runElementwiseFusionBenchmarks() {
    print(
      '\n===================================================================================',
    );
    print(
      '  BENCHMARK 2: Deep Fused Elementwise Math Pipeline (JIT Kernel Fusion)',
    );
    print(
      '  Hardware Kernel: Single-Pass JIT Fused AST WGSL Shader (Zero Intermediate Allocations)',
    );
    print(
      '  Operation: Y = SiLU(2.5 * X + 1.2) = (2.5*X + 1.2) / (1 + exp(-(2.5*X + 1.2)))',
    );
    print(
      '===================================================================================',
    );

    final elementCounts = [1000000, 5000000, 20000000];

    // Compile JIT Fused Shader AST once (uses grid-stride loop for > 65535 workgroups)
    final xVariable = Expr.variable('x', bindingIndex: 0);
    final fusedAst = (xVariable * 2.5 + 1.2).silu();
    final fusedShader = WgslJitCompiler.instance.compile(
      fusedAst,
      kernelName: 'silu_fused_pipeline',
      strided: false,
    );

    for (final count in elementCounts) {
      final rawX = Float32List(count);
      final random = math.Random(123);
      for (var i = 0; i < count; i++) {
        rawX[i] = random.nextDouble() * 4.0 - 2.0;
      }

      // Memory moved: Read X (4B) + Write Y (4B) = 8 bytes per element
      final memoryBytes = count * 4.0 * 2.0;

      // 1. CPU (ndarray - sequential multi-step allocations)
      final cpuX = nd.NDArray<nd.Float32>.fromList(rawX, [
        count,
      ], nd.DType.float32);
      final cpuMs = _measure(
        () {
          final product = cpuX * 2.5;
          final scaled = product + 1.2;
          final negated = -scaled;
          final exponentiated = nd.exp(negated);
          final denominator = exponentiated + 1.0;
          final result = scaled / denominator;
          result.dispose();
          denominator.dispose();
          exponentiated.dispose();
          negated.dispose();
          scaled.dispose();
          product.dispose();
        },
        warmup: 2,
        iterations: 4,
      );

      final cpuThroughput = memoryBytes / (cpuMs * 1e6);

      // 2. GPU (gpuarray - JIT Fused single-pass WGSL compute shader in high-bandwidth VRAM)
      final gpuX = GpuArray.fromList(
        rawX,
        [count],
        DType.float32,
        device: gpuDevice,
      );
      final gpuDestination = GpuArray<Float32>.empty(
        [count],
        DType.float32,
        device: gpuDevice,
      );

      final gpuMs = _measure(
        () {
          gpuDevice.backend.dispatchComputePipeline(
            shaderModule: fusedShader,
            buffers: [gpuX.buffer, gpuDestination.buffer],
            uniforms: [count, 0, 0, 0],
            workgroupsX: math.min(65535, (count + 255) ~/ 256),
          );
          final downloaded = gpuDestination.toNDArray();
          downloaded.dispose();
        },
        warmup: 3,
        iterations: 8,
      );

      final gpuThroughput = memoryBytes / (gpuMs * 1e6);
      final speedup = cpuMs / gpuMs;

      results.add(
        BenchmarkResult(
          name: 'JIT Fused Pipeline (SiLU)',
          size: '${(count / 1e6).toStringAsFixed(0)}M elements',
          cpuMs: cpuMs,
          gpuMs: gpuMs,
          speedup: speedup,
          cpuThroughputGb: cpuThroughput,
          gpuThroughputGb: gpuThroughput,
        ),
      );

      print(
        '  [${(count / 1e6).toStringAsFixed(0)}M elements] CPU: ${cpuMs.toStringAsFixed(2)} ms (${cpuThroughput.toStringAsFixed(2)} GB/s) | '
        'GPU: ${gpuMs.toStringAsFixed(2)} ms (${gpuThroughput.toStringAsFixed(2)} GB/s) | '
        'Speedup: ${speedup.toStringAsFixed(2)}x',
      );

      cpuX.dispose();
      gpuX.dispose();
      gpuDestination.dispose();
    }
  }

  /// Runs large-scale vectorized binary multiplication (`A * B`) benchmarks.
  void runVectorizedBinaryBenchmarks() {
    print(
      '\n===================================================================================',
    );
    print('  BENCHMARK 3: Large-Scale Vectorized Binary Arithmetic (A * B)');
    print(
      '  Hardware Kernel: Parallel Elementwise Multiply WGSL Compute Pipeline',
    );
    print('  Operation: C = A * B (Float32)');
    print(
      '===================================================================================',
    );

    final elementCounts = [1000000, 5000000, 20000000];

    for (final count in elementCounts) {
      final rawA = Float32List(count);
      final rawB = Float32List(count);
      final random = math.Random(55);
      for (var i = 0; i < count; i++) {
        rawA[i] = random.nextDouble();
        rawB[i] = random.nextDouble();
      }

      // Memory moved: Read A (4B) + Read B (4B) + Write C (4B) = 12 bytes per element
      final memoryBytes = count * 4.0 * 3.0;

      // 1. CPU (ndarray)
      final cpuA = nd.NDArray<nd.Float32>.fromList(rawA, [
        count,
      ], nd.DType.float32);
      final cpuB = nd.NDArray<nd.Float32>.fromList(rawB, [
        count,
      ], nd.DType.float32);

      final cpuMs = _measure(
        () {
          final result = cpuA * cpuB;
          result.dispose();
        },
        warmup: 2,
        iterations: 4,
      );

      final cpuThroughput = memoryBytes / (cpuMs * 1e6);

      // 2. GPU (gpuarray)
      final gpuA = GpuArray.fromList(
        rawA,
        [count],
        DType.float32,
        device: gpuDevice,
      );
      final gpuB = GpuArray.fromList(
        rawB,
        [count],
        DType.float32,
        device: gpuDevice,
      );

      final gpuMs = _measure(
        () {
          final result = gpuA * gpuB;
          final downloaded = result.toNDArray();
          downloaded.dispose();
          result.dispose();
        },
        warmup: 3,
        iterations: 8,
      );

      final gpuThroughput = memoryBytes / (gpuMs * 1e6);
      final speedup = cpuMs / gpuMs;

      results.add(
        BenchmarkResult(
          name: 'Vectorized Binary Multiply (A * B)',
          size: '${(count / 1e6).toStringAsFixed(0)}M elements',
          cpuMs: cpuMs,
          gpuMs: gpuMs,
          speedup: speedup,
          cpuThroughputGb: cpuThroughput,
          gpuThroughputGb: gpuThroughput,
        ),
      );

      print(
        '  [${(count / 1e6).toStringAsFixed(0)}M elements] CPU: ${cpuMs.toStringAsFixed(2)} ms (${cpuThroughput.toStringAsFixed(2)} GB/s) | '
        'GPU: ${gpuMs.toStringAsFixed(2)} ms (${gpuThroughput.toStringAsFixed(2)} GB/s) | '
        'Speedup: ${speedup.toStringAsFixed(2)}x',
      );

      cpuA.dispose();
      cpuB.dispose();
      gpuA.dispose();
      gpuB.dispose();
    }
  }

  /// Prints a formatted markdown-style summary table of all benchmark results.
  void printSummaryTable() {
    print('\n');
    print(
      '=======================================================================================================',
    );
    print(
      '                                 SUMMARY PERFORMANCE BENCHMARK RESULTS',
    );
    print(
      '=======================================================================================================',
    );
    print(
      '| ${"Benchmark Task".padRight(35)} | ${"Workload / Shape".padRight(20)} | ${"CPU (ms)".padLeft(10)} | ${"GPU (ms)".padLeft(10)} | ${"Speedup".padLeft(9)} |',
    );
    print(
      '|-------------------------------------|----------------------|------------|------------|-----------|',
    );

    for (final entry in results) {
      final nameText = entry.name.length > 35
          ? '${entry.name.substring(0, 32)}...'
          : entry.name;
      final sizeText = entry.size.length > 20
          ? '${entry.size.substring(0, 17)}...'
          : entry.size;
      final cpuText = entry.cpuMs.toStringAsFixed(2);
      final gpuText = entry.gpuMs.toStringAsFixed(2);
      final speedupText = '${entry.speedup.toStringAsFixed(2)}x';

      print(
        '| ${nameText.padRight(35)} | ${sizeText.padRight(20)} | ${cpuText.padLeft(10)} | ${gpuText.padLeft(10)} | ${speedupText.padLeft(9)} |',
      );
    }
    print(
      '=======================================================================================================',
    );
  }
}

Future<void> main([List<String> args = const []]) async {
  if (args.isNotEmpty) {
    stderr.writeln('Usage: dart benchmark/gpu_vs_cpu_benchmark.dart');
    exitCode = 2;
    return;
  }

  print(
    '===================================================================================',
  );
  print(
    '  Scientific Dart: Hardware Acceleration Benchmark: package:gpuarray vs package:ndarray',
  );
  print(
    '===================================================================================',
  );

  final runner = await BenchmarkRunner.create();
  print(
    'Target Hardware Device: ${runner.gpuDevice.name} (${runner.gpuDevice.type.name})',
  );

  runner.runGemmBenchmarks();
  runner.runElementwiseFusionBenchmarks();
  runner.runVectorizedBinaryBenchmarks();
  runner.printSummaryTable();

  runner.gpuDevice.dispose();
}
