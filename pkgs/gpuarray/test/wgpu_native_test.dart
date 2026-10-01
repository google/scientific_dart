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
import 'package:gpuarray/src/backend/native/wgpu_bindings.dart';

void main() {
  group('WebGPU C-FFI Structs & Constants', () {
    test('Standard Buffer Usage Constants bitmask values', () {
      expect(WGPUBufferUsage.none, equals(0));
      expect(WGPUBufferUsage.mapRead, equals(1));
      expect(WGPUBufferUsage.mapWrite, equals(2));
      expect(WGPUBufferUsage.copySrc, equals(4));
      expect(WGPUBufferUsage.copyDst, equals(8));
      expect(WGPUBufferUsage.index, equals(16));
      expect(WGPUBufferUsage.vertex, equals(32));
      expect(WGPUBufferUsage.uniform, equals(64));
      expect(WGPUBufferUsage.storage, equals(128));
      expect(WGPUBufferUsage.indirect, equals(256));
      expect(WGPUBufferUsage.queryResolve, equals(512));

      // Global alias constants
      expect(WGPUBufferUsage_MapRead, equals(1));
      expect(WGPUBufferUsage_MapWrite, equals(2));
      expect(WGPUBufferUsage_CopySrc, equals(4));
      expect(WGPUBufferUsage_CopyDst, equals(8));
      expect(WGPUBufferUsage_Uniform, equals(64));
      expect(WGPUBufferUsage_Storage, equals(128));

      // Combined usage bitmasks
      final computeStorage =
          WGPUBufferUsage.storage |
          WGPUBufferUsage.copyDst |
          WGPUBufferUsage.copySrc;
      expect(computeStorage, equals(128 | 8 | 4));
      expect(
        computeStorage & WGPUBufferUsage.storage,
        equals(WGPUBufferUsage.storage),
      );
      expect(computeStorage & WGPUBufferUsage.uniform, equals(0));
    });

    test('Map Mode & SType constants', () {
      expect(WGPUMapMode.none, equals(0));
      expect(WGPUMapMode.read, equals(1));
      expect(WGPUMapMode.write, equals(2));

      expect(WGPUSType.invalid, equals(0));
      expect(WGPUSType.shaderModuleWGSLDescriptor, equals(6));
      expect(WGPUSType_ShaderModuleWGSLDescriptor, equals(6));
    });

    test('Backend Type & Power Preference constants', () {
      expect(WGPUPowerPreference.undefined, equals(0));
      expect(WGPUPowerPreference.lowPower, equals(1));
      expect(WGPUPowerPreference.highPerformance, equals(2));

      expect(WGPUBackendType.undefined, equals(0));
      expect(WGPUBackendType.nullBackend, equals(1));
      expect(WGPUBackendType.webGpu, equals(2));
      expect(WGPUBackendType.d3d11, equals(3));
      expect(WGPUBackendType.d3d12, equals(4));
      expect(WGPUBackendType.metal, equals(5));
      expect(WGPUBackendType.vulkan, equals(6));
      expect(WGPUBackendType.openGl, equals(7));
      expect(WGPUBackendType.openGlEs, equals(8));
    });

    test('FFI Struct Allocations & Sizes in Memory', () {
      using((arena) {
        // WGPUChainedStruct
        final chained = arena<WGPUChainedStruct>();
        chained.ref.next = ffi.nullptr;
        chained.ref.sType = WGPUSType.shaderModuleWGSLDescriptor;
        expect(chained.ref.sType, equals(6));
        expect(ffi.sizeOf<WGPUChainedStruct>(), greaterThan(0));

        // WGPUInstanceDescriptor
        final instDesc = arena<WGPUInstanceDescriptor>();
        instDesc.ref.nextInChain = chained;
        expect(instDesc.ref.nextInChain, equals(chained));
        expect(ffi.sizeOf<WGPUInstanceDescriptor>(), greaterThan(0));

        // WGPUBufferDescriptor
        final bufDesc = arena<WGPUBufferDescriptor>();
        bufDesc.ref.nextInChain = ffi.nullptr;
        bufDesc.ref.label.data = 'TestBuffer'.toNativeUtf8(allocator: arena);
        bufDesc.ref.label.length = 10;
        bufDesc.ref.usage = WGPUBufferUsage.storage | WGPUBufferUsage.copyDst;
        bufDesc.ref.size = 1024;
        bufDesc.ref.mappedAtCreation = 0;
        expect(bufDesc.ref.usage, equals(136));
        expect(bufDesc.ref.size, equals(1024));
        expect(bufDesc.ref.label.data.toDartString(), equals('TestBuffer'));

        // WGPUShaderSourceWGSL
        final wgslDesc = arena<WGPUShaderSourceWGSL>();
        wgslDesc.ref.chain.next = ffi.nullptr;
        wgslDesc.ref.chain.sType = WGPUSType.shaderSourceWGSL;
        wgslDesc.ref.code.data = '@compute @workgroup_size(64) fn main() {}'
            .toNativeUtf8(allocator: arena);
        wgslDesc.ref.code.length = 37;
        expect(wgslDesc.ref.chain.sType, equals(2));
        expect(wgslDesc.ref.code.data.toDartString(), contains('@compute'));

        // WGPUBindGroupEntry
        final bgEntry = arena<WGPUBindGroupEntry>();
        bgEntry.ref.nextInChain = ffi.nullptr;
        bgEntry.ref.binding = 0;
        bgEntry.ref.buffer = ffi.nullptr;
        bgEntry.ref.offset = 0;
        bgEntry.ref.size = 256;
        expect(bgEntry.ref.binding, equals(0));
        expect(bgEntry.ref.size, equals(256));

        // WGPUComputePipelineDescriptor
        final pipeDesc = arena<WGPUComputePipelineDescriptor>();
        pipeDesc.ref.nextInChain = ffi.nullptr;
        pipeDesc.ref.label.data = 'ComputePipeline'.toNativeUtf8(
          allocator: arena,
        );
        pipeDesc.ref.label.length = 15;
        pipeDesc.ref.computeEntryPoint.data = 'main'.toNativeUtf8(
          allocator: arena,
        );
        pipeDesc.ref.computeEntryPoint.length = 4;
        expect(
          pipeDesc.ref.label.data.toDartString(),
          equals('ComputePipeline'),
        );
        expect(
          pipeDesc.ref.computeEntryPoint.data.toDartString(),
          equals('main'),
        );
      });
    });
  });

  group('Native WebGPU Driver Backend (WgpuNativeBackend)', () {
    test('Initialization via createSync and async create', () async {
      final syncBackend = WgpuNativeBackend.createSync();
      expect(syncBackend.deviceType, equals(GpuDeviceType.webgpu));
      expect(syncBackend.isDisposed, isFalse);
      expect(syncBackend.activeAllocationCount, equals(0));
      syncBackend.dispose();
      expect(syncBackend.isDisposed, isTrue);

      final asyncBackend = await WgpuNativeBackend.create();
      expect(asyncBackend.deviceType, equals(GpuDeviceType.webgpu));
      expect(asyncBackend.isDisposed, isFalse);
      asyncBackend.dispose();
    });

    test('Buffer memory allocation, tracking and freeing', () {
      final backend = WgpuNativeBackend.createSync();

      // Allocating 0 bytes returns nullptr
      final nullPtr = backend.allocateBuffer(0);
      expect(nullPtr, equals(ffi.nullptr));
      expect(backend.activeAllocationCount, equals(0));

      // Allocating valid size
      final ptr1 = backend.allocateBuffer(512);
      expect(ptr1, isNot(equals(ffi.nullptr)));
      expect(backend.activeAllocationCount, equals(1));

      final ptr2 = backend.allocateBuffer(1024);
      expect(ptr2, isNot(equals(ffi.nullptr)));
      expect(backend.activeAllocationCount, equals(2));

      // Freeing buffers
      backend.freeBuffer(ptr1, 512);
      expect(backend.activeAllocationCount, equals(1));

      backend.freeBuffer(ptr2, 1024);
      expect(backend.activeAllocationCount, equals(0));

      backend.dispose();
      expect(backend.isDisposed, isTrue);

      // Allocating on disposed backend throws
      expect(
        () => backend.allocateBuffer(256),
        throwsA(isA<GpuDeviceDisposedException>()),
      );
    });

    test(
      'Host to Buffer and Buffer to Host memory copies (aligned and unaligned)',
      () {
        final backend = WgpuNativeBackend.createSync();
        final device = GpuDevice.create(
          backend: backend,
          type: GpuDeviceType.webgpu,
        );

        final buffer = GpuBuffer.allocate(
          sizeInBytes: 16,
          usage: GpuBufferUsage.storage,
          device: device,
        );

        using((arena) {
          final src = arena<ffi.Float>(4);
          src[0] = 1.0;
          src[1] = 2.5;
          src[2] = -4.0;
          src[3] = 8.25;

          // Copy host -> GPU buffer
          backend.copyHostToBuffer(src.cast<ffi.Uint8>(), buffer, 16);

          // Copy GPU buffer -> host dst
          final dst = arena<ffi.Float>(4);
          backend.copyBufferToHost(buffer, dst.cast<ffi.Uint8>(), 16);

          expect(dst[0], equals(1.0));
          expect(dst[1], equals(2.5));
          expect(dst[2], equals(-4.0));
          expect(dst[3], equals(8.25));
        });

        // Test unaligned byte lengths and unaligned offsets (COPY_BUFFER_ALIGNMENT = 4)
        final oddBuffer = GpuBuffer.allocate(
          sizeInBytes: 7,
          usage: GpuBufferUsage.storage,
          device: device,
        );
        expect(oddBuffer.sizeInBytes, equals(7));
        expect(oddBuffer.allocatedBytes, equals(8));

        oddBuffer.writeBytes([10, 20, 30, 40, 50, 60, 70]);
        expect(oddBuffer.readBytes(), equals([10, 20, 30, 40, 50, 60, 70]));

        // Partial unaligned write at offset 1 of length 3
        oddBuffer.writeBytes([99, 88, 77], offset: 1);
        expect(oddBuffer.readBytes(), equals([10, 99, 88, 77, 50, 60, 70]));
        expect(oddBuffer.readBytes(offset: 2, length: 3), equals([88, 77, 50]));

        oddBuffer.clear(offset: 1, length: 3);
        expect(oddBuffer.readBytes(), equals([10, 0, 0, 0, 50, 60, 70]));

        oddBuffer.dispose();
        buffer.dispose();
        device.dispose();
      },
    );

    test('Buffer to Buffer copies (aligned and unaligned)', () {
      final backend = WgpuNativeBackend.createSync();
      final device = GpuDevice.create(
        backend: backend,
        type: GpuDeviceType.webgpu,
      );

      final bufA = GpuBuffer.allocate(
        sizeInBytes: 16,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copySrc,
        device: device,
      );
      final bufB = GpuBuffer.allocate(
        sizeInBytes: 16,
        usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
        device: device,
      );

      using((arena) {
        final hostData = arena<ffi.Int32>(4);
        hostData[0] = 10;
        hostData[1] = 20;
        hostData[2] = 30;
        hostData[3] = 40;

        backend.copyHostToBuffer(hostData.cast<ffi.Uint8>(), bufA, 16);
        backend.copyBufferToBuffer(bufA, bufB, 16);

        final readback = arena<ffi.Int32>(4);
        backend.copyBufferToHost(bufB, readback.cast<ffi.Uint8>(), 16);

        expect(readback[0], equals(10));
        expect(readback[1], equals(20));
        expect(readback[2], equals(30));
        expect(readback[3], equals(40));
      });

      // Unaligned buffer-to-buffer copy (srcOffset: 1, dstOffset: 2, bytes: 5)
      bufA.writeBytes([1, 2, 3, 4, 5, 6, 7, 8]);
      bufB.clear();
      bufA.copyTo(bufB, srcOffset: 1, dstOffset: 2, length: 5);
      expect(bufB.readBytes(length: 8), equals([0, 0, 2, 3, 4, 5, 6, 0]));

      bufA.dispose();
      bufB.dispose();
      device.dispose();
    });

    test(
      'dispatchComputePipeline executes real WGSL on GPU including aliased buffers',
      () {
        final backend = WgpuNativeBackend.createSync();
        final device = GpuDevice.create(
          backend: backend,
          type: GpuDeviceType.webgpu,
        );

        final bufA = GpuBuffer.allocate(
          sizeInBytes: 16,
          usage: GpuBufferUsage.storage,
          device: device,
        );
        final bufB = GpuBuffer.allocate(
          sizeInBytes: 16,
          usage: GpuBufferUsage.storage,
          device: device,
        );
        final bufOut = GpuBuffer.allocate(
          sizeInBytes: 16,
          usage: GpuBufferUsage.storage,
          device: device,
        );

        // Initialize inputs
        using((arena) {
          final a = arena<ffi.Float>(4);
          final b = arena<ffi.Float>(4);
          for (var i = 0; i < 4; i++) {
            a[i] = (i + 1) * 2.0;
            b[i] = 10.0;
          }
          backend.copyHostToBuffer(a.cast<ffi.Uint8>(), bufA, 16);
          backend.copyHostToBuffer(b.cast<ffi.Uint8>(), bufB, 16);
        });

        const wgslCode = '''
@group(0) @binding(0) var<storage, read> inA: array<f32>;
@group(0) @binding(1) var<storage, read> inB: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 4u) {
        out[idx] = inA[idx] + inB[idx];
    }
}
''';

        final shaderModule = WgslShaderModule(
          name: 'vector_add',
          code: wgslCode,
          entryPoint: 'main',
        );

        expect(backend.dispatches, isEmpty);

        backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [bufA, bufB, bufOut],
          uniforms: [4],
          workgroupsX: 1,
          workgroupsY: 1,
          workgroupsZ: 1,
        );

        expect(backend.dispatches.length, equals(1));
        final record = backend.dispatches.first;
        expect(record.shaderModule.name, equals('vector_add'));
        expect(record.buffers.length, equals(3));
        expect(record.workgroupsX, equals(1));
        expect(record.uniforms, equals([4]));

        // Verify GPU execution result
        using((arena) {
          final result = arena<ffi.Float>(4);
          backend.copyBufferToHost(bufOut, result.cast<ffi.Uint8>(), 16);
          expect(result[0], equals(12.0));
          expect(result[1], equals(14.0));
          expect(result[2], equals(16.0));
          expect(result[3], equals(18.0));
        });

        // Test aliased bindings: [bufA, bufA, bufOut] (a + a -> out)
        backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [bufA, bufA, bufOut],
          workgroupsX: 1,
        );
        using((arena) {
          final result = arena<ffi.Float>(4);
          backend.copyBufferToHost(bufOut, result.cast<ffi.Uint8>(), 16);
          expect(result[0], equals(4.0));
          expect(result[1], equals(8.0));
          expect(result[2], equals(12.0));
          expect(result[3], equals(16.0));
        });

        // Test in-place aliased output binding: [bufA, bufB, bufA] (a + b -> a)
        backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [bufA, bufB, bufA],
          workgroupsX: 1,
        );
        using((arena) {
          final result = arena<ffi.Float>(4);
          backend.copyBufferToHost(bufA, result.cast<ffi.Uint8>(), 16);
          expect(result[0], equals(12.0));
          expect(result[1], equals(14.0));
          expect(result[2], equals(16.0));
          expect(result[3], equals(18.0));
        });

        // Clear dispatches
        backend.clearDispatches();
        expect(backend.dispatches, isEmpty);

        bufA.dispose();
        bufB.dispose();
        bufOut.dispose();
        device.dispose();
      },
    );

    test('dispatchComputePipeline validation errors', () {
      final backend = WgpuNativeBackend.createSync();
      final device = GpuDevice.create(
        backend: backend,
        type: GpuDeviceType.webgpu,
      );

      final buf = GpuBuffer.allocate(
        sizeInBytes: 16,
        usage: GpuBufferUsage.storage,
        device: device,
      );

      final shaderModule = WgslShaderModule(
        name: 'noop',
        code: '@compute @workgroup_size(1) fn main() {}',
      );

      // Invalid workgroups
      expect(
        () => backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [buf],
          workgroupsX: 0,
        ),
        throwsA(isA<ArgumentError>()),
      );

      buf.dispose();

      // Disposed buffer
      expect(
        () => backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [buf],
          workgroupsX: 1,
        ),
        throwsA(isA<GpuMemoryException>()),
      );

      backend.dispose();

      // Disposed backend
      expect(
        () => backend.dispatchComputePipeline(
          shaderModule: shaderModule,
          buffers: [],
          workgroupsX: 1,
        ),
        throwsA(isA<GpuDeviceDisposedException>()),
      );

      device.dispose();
    });

    test('Disposed GpuBuffer throws StateError on read/write/clear', () {
      final backend = WgpuNativeBackend.createSync();
      final device = GpuDevice.create(
        backend: backend,
        type: GpuDeviceType.webgpu,
      );

      final buf = GpuBuffer.allocate(
        sizeInBytes: 16,
        usage: GpuBufferUsage.storage,
        device: device,
      );

      expect(buf.nativeHandle, isNot(equals(ffi.nullptr)));

      buf.dispose();
      expect(() => buf.nativeHandle, throwsA(isA<StateError>()));
      expect(() => buf.readBytes(), throwsA(isA<StateError>()));
      expect(() => buf.writeBytes([1, 2, 3, 4]), throwsA(isA<StateError>()));
      expect(() => buf.clear(), throwsA(isA<StateError>()));
      device.dispose();
    });
  });
}
