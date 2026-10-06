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
import 'package:resource_scope/resource_scope.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/wgsl.dart';

class TrackingBackend extends GpuBackend {
  final WgpuNativeBackend _inner = WgpuNativeBackend.createSync();
  int allocCount = 0;
  int freeCount = 0;

  @override
  GpuDeviceType get deviceType => GpuDeviceType.webgpu;

  @override
  ffi.Pointer<ffi.Void> allocateBuffer(
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
  }) {
    allocCount++;
    return _inner.allocateBuffer(sizeInBytes, usage: usage);
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Void> handle, int sizeInBytes) {
    freeCount++;
    _inner.freeBuffer(handle, sizeInBytes);
  }

  @override
  void copyHostToBuffer(
    ffi.Pointer<ffi.Uint8> src,
    GpuBuffer dst,
    int bytes, {
    int offset = 0,
  }) {
    _inner.copyHostToBuffer(src, dst, bytes, offset: offset);
  }

  @override
  void copyBufferToHost(
    GpuBuffer src,
    ffi.Pointer<ffi.Uint8> dst,
    int bytes, {
    int offset = 0,
  }) {
    _inner.copyBufferToHost(src, dst, bytes, offset: offset);
  }

  @override
  void copyBufferToBuffer(
    GpuBuffer src,
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    _inner.copyBufferToBuffer(
      src,
      dst,
      bytes,
      srcOffset: srcOffset,
      dstOffset: dstOffset,
    );
  }

  @override
  void clearBuffer(GpuBuffer buffer, {int offset = 0, int? bytes}) {
    _inner.clearBuffer(buffer, offset: offset, bytes: bytes);
  }

  @override
  void dispatchComputePipeline({
    required WgslShaderModule shaderModule,
    required List<GpuBuffer> buffers,
    List<int>? uniforms,
    required int workgroupsX,
    int workgroupsY = 1,
    int workgroupsZ = 1,
  }) {
    _inner.dispatchComputePipeline(
      shaderModule: shaderModule,
      buffers: buffers,
      uniforms: uniforms,
      workgroupsX: workgroupsX,
      workgroupsY: workgroupsY,
      workgroupsZ: workgroupsZ,
    );
  }

  @override
  void dispose() {
    _inner.dispose();
  }
}

void main() {
  group('VRAM Block Memory Pool & Backend Drivers', () {
    test('Power-of-2 bucket sizing calculation', () {
      expect(GpuMemoryPool.computeBucketSize(10), equals(64));
      expect(GpuMemoryPool.computeBucketSize(64), equals(64));
      expect(GpuMemoryPool.computeBucketSize(65), equals(128));
      expect(GpuMemoryPool.computeBucketSize(1000), equals(1024));
      expect(GpuMemoryPool.computeBucketSize(1024), equals(1024));
      expect(GpuMemoryPool.computeBucketSize(1025), equals(2048));
    });

    test('GpuMemoryPool caches and reuses buffers in O(1)', () {
      final backend = TrackingBackend();
      final device = GpuDevice.create(
        name: 'Pooled Device',
        backend: backend,
        enableMemoryPool: true,
      );

      expect(device.memoryPool.hits, equals(0));
      expect(device.memoryPool.misses, equals(0));

      // 1st allocation: miss -> allocates 128 bytes bucket
      final buf1 = device.createBuffer(
        sizeInBytes: 100,
        usage: GpuBufferUsage.storage,
      );
      expect(device.memoryPool.misses, equals(1));
      expect(device.memoryPool.hits, equals(0));
      expect(backend.allocCount, equals(1));

      // Dispose buf1 -> recycled into pool
      buf1.dispose();
      expect(device.memoryPool.cachedBytes, equals(128));

      // 2nd allocation of same size bucket -> cache hit!
      final buf2 = device.createBuffer(
        sizeInBytes: 120,
        usage: GpuBufferUsage.storage,
      );
      expect(device.memoryPool.hits, equals(1));
      expect(device.memoryPool.misses, equals(1));
      expect(backend.allocCount, equals(1)); // No new native allocation!

      buf2.dispose();
      device.dispose();
    });

    test('GpuMemoryPool trim purges cached buffers to OS', () {
      final backend = TrackingBackend();
      final device = GpuDevice.create(backend: backend, enableMemoryPool: true);

      final buf = device.createBuffer(
        sizeInBytes: 500,
        usage: GpuBufferUsage.storage,
      );
      buf.dispose();
      expect(device.memoryPool.cachedBytes, equals(512));

      device.memoryPool.trim();
      expect(device.memoryPool.cachedBytes, equals(0));
      expect(backend.freeCount, equals(1));

      device.dispose();
    });

    test('ResourceScope seamlessly integrates with GpuMemoryPool', () {
      final device = GpuDevice.create(enableMemoryPool: true);

      final result = ResourceScope.returning(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
          device: device,
        );
        final b = GpuArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
          device: device,
        );
        final c = a + b;
        return c;
      });

      expect(result.toList(), equals([11.0, 22.0, 33.0, 44.0]));
      result.dispose();
      device.dispose();
    });

    test(
      'Device disposal frees active pooled buffers without leaking memory',
      () {
        final backend = TrackingBackend();
        final device = GpuDevice.create(
          backend: backend,
          enableMemoryPool: true,
        );

        final buf1 = device.createBuffer(
          sizeInBytes: 100,
          usage: GpuBufferUsage.storage,
        );
        final buf2 = device.createBuffer(
          sizeInBytes: 200,
          usage: GpuBufferUsage.storage,
        );
        expect(backend.allocCount, equals(2));
        expect(backend.freeCount, equals(0));

        device.dispose();

        expect(buf1.isDisposed, isTrue);
        expect(buf2.isDisposed, isTrue);
        expect(device.isDisposed, isTrue);
        expect(device.memoryPool.isDisposed, isTrue);
        expect(backend.freeCount, equals(2));

        // Subsequent dispose on buffer is safe
        buf1.dispose();
        expect(backend.freeCount, equals(2));
      },
    );

    test('GpuMemoryPool enforces maxCachedBytes limit', () {
      final backend = TrackingBackend();
      final device = GpuDevice.create(backend: backend, enableMemoryPool: true);
      final smallPool = GpuMemoryPool(device, maxCachedBytes: 128);

      final buf1 = smallPool.acquire(64);
      final buf2 = smallPool.acquire(64);
      final buf3 = smallPool.acquire(64);
      expect(backend.allocCount, equals(3));
      expect(backend.freeCount, equals(0));

      buf1.dispose();
      expect(smallPool.cachedBytes, equals(64));
      expect(backend.freeCount, equals(0));

      buf2.dispose();
      expect(smallPool.cachedBytes, equals(128));
      expect(backend.freeCount, equals(0));

      // buf3 exceeds maxCachedBytes limit (128 + 64 > 128) -> immediately freed
      buf3.dispose();
      expect(smallPool.cachedBytes, equals(128));
      expect(backend.freeCount, equals(1));

      smallPool.dispose();
      expect(backend.freeCount, equals(3));
      device.dispose();
    });

    test(
      'GpuMemoryPool zeroes recycled buffers on GPU and updates sizeInBytes on acquire',
      () {
        final device = GpuDevice.create(enableMemoryPool: true);

        // Allocate 80 bytes (bucket size 128), fill with non-zero bytes
        final buf1 = device.createBuffer(
          sizeInBytes: 80,
          usage: GpuBufferUsage.storage,
        );
        expect(buf1.sizeInBytes, equals(80));
        expect(buf1.allocatedBytes, equals(128));
        buf1.writeBytes(List<int>.filled(80, 0xAB));
        expect(buf1.readBytes(length: 4), equals([0xAB, 0xAB, 0xAB, 0xAB]));

        // Dispose to return to pool
        buf1.dispose();

        // Re-acquire from the same 128-byte bucket with a different requested size (96)
        final buf2 = device.createBuffer(
          sizeInBytes: 96,
          usage: GpuBufferUsage.storage,
        );
        expect(identical(buf1, buf2), isTrue);
        expect(buf2.sizeInBytes, equals(96));
        expect(buf2.allocatedBytes, equals(128));
        // Recycled buffer must be zero-cleared on the GPU
        expect(buf2.readBytes(), equals(List<int>.filled(96, 0)));

        buf2.dispose();
        device.dispose();
      },
    );

    test('Negative sizeInBytes throws ArgumentError', () {
      final device = GpuDevice.create(enableMemoryPool: true);

      expect(
        () => device.memoryPool.acquire(-1),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => GpuBuffer.allocate(
          sizeInBytes: -10,
          usage: GpuBufferUsage.storage,
          device: device,
        ),
        throwsA(isA<ArgumentError>()),
      );

      device.dispose();
    });

    test(
      'copyFromHost, copyToHost, and copyToBuffer throw GpuMemoryException on invalid bounds',
      () {
        final device = GpuDevice.create();
        final buf1 = device.createBuffer(
          sizeInBytes: 64,
          usage: GpuBufferUsage.storage,
        );
        final buf2 = device.createBuffer(
          sizeInBytes: 64,
          usage: GpuBufferUsage.storage,
        );
        final hostPtr = calloc<ffi.Uint8>(128);

        try {
          // Negative bytes
          expect(
            () => buf1.copyFromHost(hostPtr.cast<ffi.Void>(), -1),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToHost(hostPtr.cast<ffi.Void>(), -1),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToBuffer(buf2, -1),
            throwsA(isA<GpuMemoryException>()),
          );

          // Negative offsets
          expect(
            () => buf1.copyFromHost(hostPtr.cast<ffi.Void>(), 10, offset: -1),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToHost(hostPtr.cast<ffi.Void>(), 10, offset: -1),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToBuffer(buf2, 10, srcOffset: -1),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToBuffer(buf2, 10, dstOffset: -1),
            throwsA(isA<GpuMemoryException>()),
          );

          // Out-of-bounds offset + bytes > sizeInBytes
          expect(
            () => buf1.copyFromHost(hostPtr.cast<ffi.Void>(), 65),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyFromHost(hostPtr.cast<ffi.Void>(), 10, offset: 60),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToHost(hostPtr.cast<ffi.Void>(), 65),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToHost(hostPtr.cast<ffi.Void>(), 10, offset: 60),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToBuffer(buf2, 65),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToBuffer(buf2, 10, srcOffset: 60),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => buf1.copyToBuffer(buf2, 10, dstOffset: 60),
            throwsA(isA<GpuMemoryException>()),
          );

          // Direct driver backend validation
          expect(
            () => device.backend.copyHostToBuffer(hostPtr, buf1, -5),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () =>
                device.backend.copyHostToBuffer(hostPtr, buf1, 10, offset: -1),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => device.backend.copyHostToBuffer(hostPtr, buf1, 70),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => device.backend.copyBufferToHost(buf1, hostPtr, -5),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () =>
                device.backend.copyBufferToHost(buf1, hostPtr, 10, offset: -1),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => device.backend.copyBufferToHost(buf1, hostPtr, 70),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => device.backend.copyBufferToBuffer(buf1, buf2, -5),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => device.backend.copyBufferToBuffer(
              buf1,
              buf2,
              10,
              srcOffset: -1,
            ),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => device.backend.copyBufferToBuffer(
              buf1,
              buf2,
              10,
              dstOffset: -1,
            ),
            throwsA(isA<GpuMemoryException>()),
          );
          expect(
            () => device.backend.copyBufferToBuffer(buf1, buf2, 70),
            throwsA(isA<GpuMemoryException>()),
          );
        } finally {
          calloc.free(hostPtr);
          buf1.dispose();
          buf2.dispose();
          device.dispose();
        }
      },
    );

    test(
      'GpuMemoryPool acquire preserves and updates strongly-typed GpuBufferUsage',
      () {
        final device = GpuDevice.create(enableMemoryPool: true);

        final buf1 = device.memoryPool.acquire(
          100,
          usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
        );
        expect(buf1.usage.contains(GpuBufferUsage.storage), isTrue);
        expect(buf1.usage.contains(GpuBufferUsage.copyDst), isTrue);
        expect(buf1.usage.contains(GpuBufferUsage.uniform), isFalse);

        buf1.dispose();

        // Re-acquire same bucket with different usage
        final buf2 = device.memoryPool.acquire(
          100,
          usage: GpuBufferUsage.uniform,
        );
        expect(buf2.usage.contains(GpuBufferUsage.uniform), isTrue);
        expect(buf2.usage.contains(GpuBufferUsage.copyDst), isFalse);

        buf2.dispose();
        device.dispose();
      },
    );
  });
}
