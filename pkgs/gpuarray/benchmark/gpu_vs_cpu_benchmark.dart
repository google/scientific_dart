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
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:gpuarray/gpuarray.dart';

class BenchmarkResult {
  final String name;
  final String size;
  final double cpuMs;
  final double gpuMs;
  final double speedup;
  final double? cpuGflops;
  final double? gpuGflops;
  final double? cpuThroughputGb;
  final double? gpuThroughputGb;

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

class BenchmarkRunner {
  final GpuDevice gpuDevice;
  final List<BenchmarkResult> results = [];

  BenchmarkRunner(this.gpuDevice);

  static Future<BenchmarkRunner> create() async {
    final dev = await createWebGpuDevice(
      name: 'WebGPU Physical Hardware Device',
    );
    return BenchmarkRunner(dev);
  }

  double _measure(void Function() fn, {int warmup = 2, int iterations = 5}) {
    for (var i = 0; i < warmup; i++) {
      fn();
    }
    final sw = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      fn();
    }
    sw.stop();
    return sw.elapsedMicroseconds / (iterations * 1000.0); // ms
  }

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
      final rng = math.Random(42);
      for (var i = 0; i < totalElements; i++) {
        rawDataA[i] = rng.nextDouble();
        rawDataB[i] = rng.nextDouble();
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
          nd.matmul(cpuA, cpuB);
        },
        warmup: 2,
        iterations: n >= 2048 ? 2 : 4,
      );

      final cpuGflops = (flops / (cpuMs * 1e6));

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
          final res = gpuA.matmul(gpuB);
          res.toNDArray();
        },
        warmup: 3,
        iterations: n >= 2048 ? 4 : 8,
      );

      final gpuGflops = (flops / (gpuMs * 1e6));
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

    // Compile JIT Fused Shader AST once
    final xVar = Expr.variable('x', bindingIndex: 0);
    final fusedAst = (xVar * 2.5 + 1.2).silu();
    final fusedShader = WgslJitCompiler.instance.compile(
      fusedAst,
      kernelName: 'silu_fused_pipeline',
      strided: false,
    );

    for (final count in elementCounts) {
      final rawX = Float32List(count);
      final rng = math.Random(123);
      for (var i = 0; i < count; i++) {
        rawX[i] = rng.nextDouble() * 4.0 - 2.0;
      }

      // Memory moved: Read X (4B) + Write Y (4B) = 8 bytes per element
      final memoryBytes = count * 4.0 * 2.0;

      // 1. CPU (ndarray - sequential multi-step allocations)
      final cpuX = nd.NDArray<nd.Float32>.fromList(rawX, [
        count,
      ], nd.DType.float32);
      final cpuMs = _measure(
        () {
          final scaled = cpuX * 2.5 + 1.2;
          final neg = -scaled;
          final expVal = nd.exp(neg);
          final denom = expVal + 1.0;
          final _ = scaled / denom;
        },
        warmup: 2,
        iterations: 4,
      );

      final cpuThroughput = (memoryBytes / (cpuMs * 1e6)); // GB/s

      // 2. GPU (gpuarray - JIT Fused single-pass WGSL compute shader in high-bandwidth VRAM)
      final gpuX = GpuArray.fromList(
        rawX,
        [count],
        DType.float32,
        device: gpuDevice,
      );
      final gpuDst = GpuArray<Float32>.empty(
        [count],
        DType.float32,
        device: gpuDevice,
      );

      final gpuMs = _measure(
        () {
          gpuDevice.backend.dispatchComputePipeline(
            shaderModule: fusedShader,
            buffers: [gpuX.buffer, gpuDst.buffer],
            uniforms: [count, 0, 0, 0],
            workgroupsX: math.min(65535, (count + 255) ~/ 256),
          );
          gpuDst.toNDArray();
        },
        warmup: 3,
        iterations: 8,
      );

      final gpuThroughput = (memoryBytes / (gpuMs * 1e6)); // GB/s
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
      gpuDst.dispose();
    }
  }

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
      final rng = math.Random(55);
      for (var i = 0; i < count; i++) {
        rawA[i] = rng.nextDouble();
        rawB[i] = rng.nextDouble();
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
          final _ = cpuA * cpuB;
        },
        warmup: 2,
        iterations: 4,
      );

      final cpuThroughput = (memoryBytes / (cpuMs * 1e6)); // GB/s

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
          final res = gpuA * gpuB;
          res.toNDArray();
        },
        warmup: 3,
        iterations: 8,
      );

      final gpuThroughput = (memoryBytes / (gpuMs * 1e6)); // GB/s
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

    for (final r in results) {
      final nameStr = r.name.length > 35
          ? '${r.name.substring(0, 32)}...'
          : r.name;
      final sizeStr = r.size.length > 20
          ? '${r.size.substring(0, 17)}...'
          : r.size;
      final cpuStr = r.cpuMs.toStringAsFixed(2);
      final gpuStr = r.gpuMs.toStringAsFixed(2);
      final speedupStr = '${r.speedup.toStringAsFixed(2)}x';

      print(
        '| ${nameStr.padRight(35)} | ${sizeStr.padRight(20)} | ${cpuStr.padLeft(10)} | ${gpuStr.padLeft(10)} | ${speedupStr.padLeft(9)} |',
      );
    }
    print(
      '=======================================================================================================',
    );
  }
}

void main() async {
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
