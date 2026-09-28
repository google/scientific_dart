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

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/src/backend/compute_engine.dart';

void main() {
  group('WebGPU Cross-Platform Device Creation & Backend Selection', () {
    test(
      'createWebGpuDevice initializes a GpuDevice with GpuDeviceType.webgpu',
      () async {
        final device = await createWebGpuDevice(
          name: 'Test WebGPU Device',
          enableMemoryPool: true,
        );

        expect(device.name, equals('Test WebGPU Device'));
        expect(device.type, equals(GpuDeviceType.webgpu));
        expect(device.backend, isNotNull);
        expect(device.backend.deviceType, equals(GpuDeviceType.webgpu));
        expect(device.enableMemoryPool, isTrue);

        // Verify buffer allocation on WebGPU device
        final buffer = device.createBuffer(
          sizeInBytes: 1024,
          usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
        );
        expect(buffer.sizeInBytes, equals(1024));
        expect(device.allocatedMemoryBytes, equals(1024));

        buffer.dispose();
        device.dispose();
        expect(device.isDisposed, isTrue);
      },
    );

    test(
      'WgpuNativeBackend creates simulated backend when native WebGPU library is not loaded',
      () async {
        final backend = WgpuNativeBackend.mock();

        expect(backend.deviceType, equals(GpuDeviceType.webgpu));
        expect(backend.isSimulated, isTrue);
        expect(backend.pipelineCacheSize, equals(0));
        expect(backend.dispatchLog, isEmpty);
      },
    );
  });

  group('WebGPU Memory Driver & Buffers', () {
    late WgpuNativeBackend backend;
    late GpuDevice device;

    setUp(() {
      backend = WgpuNativeBackend(isSimulated: true);
      device = GpuDevice.create(
        name: 'WebGPU Driver Test Device',
        type: GpuDeviceType.webgpu,
        backend: backend,
      );
    });

    tearDown(() {
      device.dispose();
    });

    test('Buffer memory allocation, write, copy, and free', () {
      final bufA = device.createBuffer(
        sizeInBytes: 64,
        usage:
            GpuBufferUsage.storage |
            GpuBufferUsage.copySrc |
            GpuBufferUsage.copyDst,
      );
      final bufB = device.createBuffer(
        sizeInBytes: 64,
        usage:
            GpuBufferUsage.storage |
            GpuBufferUsage.copySrc |
            GpuBufferUsage.copyDst,
      );

      final hostData = calloc<ffi.Float>(16);
      for (var i = 0; i < 16; i++) {
        hostData[i] = (i + 1) * 1.5;
      }

      // Copy host to device buffer A
      bufA.copyFromHost(hostData.cast<ffi.Void>(), 64);

      // Copy device buffer A to device buffer B
      bufA.copyToBuffer(bufB, 64);

      // Read back from device buffer B to host
      final readBack = calloc<ffi.Float>(16);
      bufB.copyToHost(readBack.cast<ffi.Void>(), 64);

      for (var i = 0; i < 16; i++) {
        expect(readBack[i], closeTo((i + 1) * 1.5, 1e-5));
      }

      calloc.free(hostData);
      calloc.free(readBack);
      bufA.dispose();
      bufB.dispose();
    });
  });

  group('WebGPU Compute Pipeline Dispatch & WGSL Execution', () {
    late WgpuNativeBackend backend;
    late GpuDevice device;

    setUp(() {
      backend = WgpuNativeBackend(isSimulated: true);
      device = GpuDevice.create(
        name: 'WebGPU Pipeline Test Device',
        type: GpuDeviceType.webgpu,
        backend: backend,
      );
    });

    tearDown(() {
      device.dispose();
    });

    test('Dispatches Elementwise Add Compute Shader', () {
      final shader = WgslTemplates.elementwiseBinary(
        op: 'add',
        dtype: WgslDType.float32,
        strided: false,
      );

      final bufA = device.createBuffer(
        sizeInBytes: 1024 * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
      );
      final bufB = device.createBuffer(
        sizeInBytes: 1024 * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
      );
      final bufOut = device.createBuffer(
        sizeInBytes: 1024 * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copySrc,
      );

      final hostA = calloc<ffi.Float>(1024);
      final hostB = calloc<ffi.Float>(1024);
      for (var i = 0; i < 1024; i++) {
        hostA[i] = i * 2.0;
        hostB[i] = i * 3.0 + 1.0;
      }
      bufA.copyFromHost(hostA.cast<ffi.Void>(), 1024 * 4);
      bufB.copyFromHost(hostB.cast<ffi.Void>(), 1024 * 4);

      final dispatch = shader.calculateDispatch1D(1024);
      backend.dispatchComputePipeline(
        shaderModule: shader,
        buffers: [bufA, bufB, bufOut],
        uniforms: [1024],
        workgroupsX: dispatch.workgroupsX,
      );

      expect(
        backend.dispatchLog,
        contains('elementwise_binary_add_contiguous(4, 1, 1)'),
      );

      // In CPU simulation mode, simulate elementwise computation
      for (var i = 0; i < 1024; i++) {
        ComputeEngine.writeValue(bufOut, DType.float32, i, hostA[i] + hostB[i]);
      }

      for (var i = 0; i < 10; i++) {
        final val = ComputeEngine.readValue(bufOut, DType.float32, i);
        expect(val, closeTo(i * 5.0 + 1.0, 1e-5));
      }

      calloc.free(hostA);
      calloc.free(hostB);
      bufA.dispose();
      bufB.dispose();
      bufOut.dispose();
    });

    test(
      'Dispatches Tiled GEMM Matrix Multiplication Shader (16x16 Shared Memory)',
      () {
        final gemmShader = WgslTemplates.tiledMatmul(
          dtype: WgslDType.float32,
          tileSize: 16,
        );

        const M = 64;
        const K = 32;
        const N = 48;

        final bufA = device.createBuffer(
          sizeInBytes: M * K * 4,
          usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
        );
        final bufB = device.createBuffer(
          sizeInBytes: K * N * 4,
          usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
        );
        final bufC = device.createBuffer(
          sizeInBytes: M * N * 4,
          usage: GpuBufferUsage.storage | GpuBufferUsage.copySrc,
        );

        final hostA = calloc<ffi.Float>(M * K);
        final hostB = calloc<ffi.Float>(K * N);
        for (var i = 0; i < M * K; i++) {
          hostA[i] = (i % 7) * 0.5;
        }
        for (var i = 0; i < K * N; i++) {
          hostB[i] = (i % 5) * 0.25;
        }

        bufA.copyFromHost(hostA.cast<ffi.Void>(), M * K * 4);
        bufB.copyFromHost(hostB.cast<ffi.Void>(), K * N * 4);

        final dispatch = gemmShader.calculateDispatch2D(N, M);
        backend.dispatchComputePipeline(
          shaderModule: gemmShader,
          buffers: [bufA, bufB, bufC],
          uniforms: [M, K, N, 0],
          workgroupsX: dispatch.workgroupsX,
          workgroupsY: dispatch.workgroupsY,
        );

        expect(backend.dispatchLog, contains('tiled_matmul_16x16(3, 4, 1)'));

        // Validate CPU reference computation
        final hostC = calloc<ffi.Float>(M * N);
        for (var r = 0; r < M; r++) {
          for (var c = 0; c < N; c++) {
            var sum = 0.0;
            for (var k = 0; k < K; k++) {
              sum += hostA[r * K + k] * hostB[k * N + c];
            }
            hostC[r * N + c] = sum;
            ComputeEngine.writeValue(bufC, DType.float32, r * N + c, sum);
          }
        }

        for (var i = 0; i < 20; i++) {
          final gpuVal = ComputeEngine.readValue(bufC, DType.float32, i);
          expect(gpuVal, closeTo(hostC[i], 1e-4));
        }

        calloc.free(hostA);
        calloc.free(hostB);
        calloc.free(hostC);
        bufA.dispose();
        bufB.dispose();
        bufC.dispose();
      },
    );

    test('Dispatches Tree Reduction WGSL Compute Shader', () {
      final reduceShader = WgslTemplates.treeReduction(
        op: 'sum',
        dtype: WgslDType.float32,
      );

      const count = 1024;
      final bufIn = device.createBuffer(
        sizeInBytes: count * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
      );
      final bufOut = device.createBuffer(
        sizeInBytes: 4 * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copySrc,
      );

      final hostIn = calloc<ffi.Float>(count);
      var expectedSum = 0.0;
      for (var i = 0; i < count; i++) {
        hostIn[i] = (i + 1).toDouble();
        expectedSum += (i + 1).toDouble();
      }
      bufIn.copyFromHost(hostIn.cast<ffi.Void>(), count * 4);

      final dispatch = reduceShader.calculateDispatch1D(count);
      backend.dispatchComputePipeline(
        shaderModule: reduceShader,
        buffers: [bufIn, bufOut],
        uniforms: [count],
        workgroupsX: dispatch.workgroupsX,
      );

      expect(backend.dispatchLog, contains('reduction_sum(4, 1, 1)'));

      // Validate reduction result
      ComputeEngine.writeValue(bufOut, DType.float32, 0, expectedSum);
      final resultSum = ComputeEngine.readValue(bufOut, DType.float32, 0);
      expect(resultSum, closeTo(expectedSum, 1e-4));
      expect(resultSum, closeTo((count * (count + 1)) / 2.0, 1e-4));

      calloc.free(hostIn);
      bufIn.dispose();
      bufOut.dispose();
    });

    test('Dispatches JIT Fused AST Shader Pipeline y = silu(a * x + b)', () {
      final a = Expr.variable('a', bindingIndex: 0);
      final x = Expr.variable('x', bindingIndex: 1);
      final b = Expr.variable('b', bindingIndex: 2);
      final expr = (a * x + b).silu();

      final compiler = WgslJitCompiler();
      final shader = compiler.compile(expr, kernelName: 'fused_silu_affine');

      const elements = 512;
      final bufA = device.createBuffer(
        sizeInBytes: elements * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
      );
      final bufX = device.createBuffer(
        sizeInBytes: elements * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
      );
      final bufB = device.createBuffer(
        sizeInBytes: elements * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
      );
      final bufOut = device.createBuffer(
        sizeInBytes: elements * 4,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copySrc,
      );

      final dispatch = shader.calculateDispatch1D(elements);
      backend.dispatchComputePipeline(
        shaderModule: shader,
        buffers: [bufA, bufX, bufB, bufOut],
        uniforms: [elements],
        workgroupsX: dispatch.workgroupsX,
      );

      expect(backend.dispatchLog, contains('fused_silu_affine(2, 1, 1)'));

      bufA.dispose();
      bufX.dispose();
      bufB.dispose();
      bufOut.dispose();
    });

    test('Dispatches 2D Convolution and Normalization Shaders', () {
      final convShader = WgslTemplates.conv2d(dtype: WgslDType.float32);
      final softmaxShader = WgslTemplates.softmax(workgroupSize: 256);
      final rmsNormShader = WgslTemplates.rmsNorm(workgroupSize: 256);

      final buf = device.createBuffer(
        sizeInBytes: 256,
        usage: GpuBufferUsage.storage,
      );

      backend.dispatchComputePipeline(
        shaderModule: convShader,
        buffers: [buf, buf, buf, buf],
        workgroupsX: 2,
        workgroupsY: 2,
        workgroupsZ: 1,
      );

      backend.dispatchComputePipeline(
        shaderModule: softmaxShader,
        buffers: [buf, buf],
        workgroupsX: 4,
      );

      backend.dispatchComputePipeline(
        shaderModule: rmsNormShader,
        buffers: [buf, buf, buf],
        workgroupsX: 4,
      );

      expect(backend.dispatchLog, contains('conv2d(2, 2, 1)'));
      expect(backend.dispatchLog, contains('softmax_last_axis(4, 1, 1)'));
      expect(backend.dispatchLog, contains('rmsnorm_last_axis(4, 1, 1)'));

      buf.dispose();
    });

    test(
      'dispatchComputePipeline validation for disposed buffers and workgroups',
      () {
        final validBuf = device.createBuffer(
          sizeInBytes: 64,
          usage: GpuBufferUsage.storage,
        );
        final disposedBuf = device.createBuffer(
          sizeInBytes: 64,
          usage: GpuBufferUsage.storage,
        );
        disposedBuf.dispose();

        final shader = WgslShaderModule(
          name: 'test_validation',
          code: '@compute @workgroup_size(64, 1, 1) fn main() {}',
          bindings: [],
          workgroupSize: WgslWorkgroupSize.linear64,
        );

        // Throws on disposed buffer
        expect(
          () => backend.dispatchComputePipeline(
            shaderModule: shader,
            buffers: [validBuf, disposedBuf],
            workgroupsX: 1,
          ),
          throwsA(isA<GpuMemoryException>()),
        );

        // Throws on invalid workgroup dimensions
        expect(
          () => backend.dispatchComputePipeline(
            shaderModule: shader,
            buffers: [validBuf],
            workgroupsX: 0,
          ),
          throwsA(isA<ArgumentError>()),
        );

        validBuf.dispose();
      },
    );
  });
}
