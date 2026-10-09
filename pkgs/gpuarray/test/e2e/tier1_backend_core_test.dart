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
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/wgsl.dart';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void _expectCloseList(
  List<Object?> actual,
  List<Object?> expected, {
  double tolerance = 1e-4,
}) {
  expect(actual.length, equals(expected.length));
  for (var i = 0; i < actual.length; i++) {
    final actualItem = actual[i];
    final expectedItem = expected[i];
    if (actualItem is num && expectedItem is num) {
      expect(
        actualItem.toDouble(),
        closeTo(expectedItem.toDouble(), tolerance),
        reason: 'Mismatch at element $i: $actualItem vs $expectedItem',
      );
    } else if (actualItem is Complex && expectedItem is Complex) {
      expect(actualItem.real, closeTo(expectedItem.real, tolerance));
      expect(actualItem.imag, closeTo(expectedItem.imag, tolerance));
    } else {
      expect(actualItem, equals(expectedItem));
    }
  }
}

void main() {
  group('Tier 1 — Backend, Memory & Core Happy Path (F1–F10)', () {
    group('F1: WebGPU Native Backend & Device Creation', () {
      test(
        'F1.1: createWebGpuDevice creates an active WebGPU device',
        () async {
          final device = await createWebGpuDevice(
            name: 'E2E-WebGPU-Primary',
            enableMemoryPool: true,
          );
          try {
            expect(device.isDisposed, isFalse);
            expect(device.type, equals(GpuDeviceType.webgpu));
            expect(device.name, equals('E2E-WebGPU-Primary'));
            expect(device.enableMemoryPool, isTrue);
            expect(device.toString(), contains('E2E-WebGPU-Primary'));
            await device.synchronize();
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F1.2: WgpuNativeBackend.create initializes native backend',
        () async {
          final backend = await WgpuNativeBackend.create();
          final device = GpuDevice.create(
            name: 'Explicit-Backend-Device',
            type: GpuDeviceType.webgpu,
            backend: backend,
          );
          try {
            expect(backend.deviceType, equals(GpuDeviceType.webgpu));
            expect(backend.usesNativeFinalizer, isFalse);
            expect(device.backend, same(backend));
            expect(device.isDisposed, isFalse);
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F1.3: GpuDevice.create configures memory pool and JIT compiler',
        () async {
          final device = await createWebGpuDevice(
            name: 'Configured-Device',
            enableMemoryPool: true,
          );
          try {
            expect(device.memoryPool.device, same(device));
            expect(device.memoryPool.isDisposed, isFalse);
            expect(device.jitCompiler.cachedCount, greaterThanOrEqualTo(0));
          } finally {
            device.dispose();
          }
        },
      );

      test('F1.4: WgpuNativeBackend dispatches WGSL compute shader', () async {
        final device = await createWebGpuDevice(name: 'WGSL-Dispatch-Device');
        try {
          final firstInput = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
            device: device,
          );
          final secondInput = GpuArray.fromList(
            <double>[10.0, 20.0, 30.0, 40.0],
            [4],
            DType.float32,
            device: device,
          );
          final output = GpuArray.zeros([4], DType.float32, device: device);
          try {
            final shader = WgslTemplates.elementwiseBinary(op: 'add');
            device.backend.dispatchComputePipeline(
              shaderModule: shader,
              buffers: [firstInput.buffer, secondInput.buffer, output.buffer],
              uniforms: [4, 0, 0, 0],
              workgroupsX: 1,
            );
            await device.synchronize();
            _expectCloseList(output.toList(), <double>[11.0, 22.0, 33.0, 44.0]);
          } finally {
            firstInput.dispose();
            secondInput.dispose();
            output.dispose();
          }
        } finally {
          device.dispose();
        }
      });

      test(
        'F1.5: device.dispose shuts down active buffers and is idempotent',
        () async {
          final device = await createWebGpuDevice(name: 'Disposable-Device');
          final allocated = device.createBuffer(sizeInBytes: 64);
          expect(allocated.isDisposed, isFalse);
          expect(device.activeBufferCount, equals(1));
          device.dispose();
          expect(device.isDisposed, isTrue);
          expect(allocated.isDisposed, isTrue);
          expect(device.memoryPool.isDisposed, isTrue);
          device.dispose();
          expect(device.isDisposed, isTrue);
        },
      );
    });

    group('F2: GpuDevice.defaultDevice & Device Lifecycle', () {
      test('F2.1: GpuDevice.defaultDevice returns a live singleton device', () {
        final first = GpuDevice.defaultDevice;
        final second = GpuDevice.defaultDevice;
        expect(first.isDisposed, isFalse);
        expect(identical(first, second), isTrue);
      });

      test(
        'F2.2: GpuDevice.defaultDevice setter routes new allocations',
        () async {
          final original = GpuDevice.defaultDevice;
          final custom = await createWebGpuDevice(
            name: 'Custom-Default-Device',
          );
          try {
            GpuDevice.defaultDevice = custom;
            expect(GpuDevice.defaultDevice, same(custom));
            final tensor = GpuArray.fromList(
              <double>[5.0, 6.0],
              [2],
              DType.float32,
            );
            try {
              expect(tensor.device, same(custom));
              _expectCloseList(tensor.toList(), <double>[5.0, 6.0]);
            } finally {
              tensor.dispose();
            }
          } finally {
            GpuDevice.defaultDevice = original;
            custom.dispose();
          }
        },
      );

      test(
        'F2.3: activeBufferCount and allocatedMemoryBytes track live buffers',
        () async {
          final device = await createWebGpuDevice(
            name: 'Tracking-Device',
            enableMemoryPool: false,
          );
          try {
            expect(device.activeBufferCount, equals(0));
            expect(device.allocatedMemoryBytes, equals(0));
            final firstBuffer = device.createBuffer(sizeInBytes: 128);
            final secondBuffer = device.createBuffer(sizeInBytes: 256);
            expect(device.activeBufferCount, equals(2));
            expect(device.allocatedMemoryBytes, equals(384));
            firstBuffer.dispose();
            expect(device.activeBufferCount, equals(1));
            expect(device.allocatedMemoryBytes, equals(256));
            secondBuffer.dispose();
            expect(device.activeBufferCount, equals(0));
            expect(device.allocatedMemoryBytes, equals(0));
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F2.4: ResourceScope.scope automatically disposes scoped GpuDevice',
        () async {
          final backend = await WgpuNativeBackend.create();
          late final GpuDevice scopedDevice;
          ResourceScope.scope(() {
            scopedDevice = GpuDevice.create(
              name: 'Scoped-Device',
              type: GpuDeviceType.webgpu,
              backend: backend,
            );
            expect(scopedDevice.isDisposed, isFalse);
          });
          expect(scopedDevice.isDisposed, isTrue);
        },
      );

      test(
        'F2.5: detachFromScope and detachToParentScope manage device lifetime',
        () async {
          final backend = await WgpuNativeBackend.create();
          late final GpuDevice promotedDevice;
          ResourceScope.scope(() {
            promotedDevice = ResourceScope.scope(() {
              final inner = GpuDevice.create(
                name: 'Promoted-Device',
                type: GpuDeviceType.webgpu,
                backend: backend,
              );
              return inner.detachToParentScope();
            });
            expect(promotedDevice.isDisposed, isFalse);
            promotedDevice.detachFromScope();
          });
          expect(promotedDevice.isDisposed, isFalse);
          promotedDevice.dispose();
          expect(promotedDevice.isDisposed, isTrue);
        },
      );
    });

    group('F3: GpuBuffer Allocation & Host/Device Transfers', () {
      test(
        'F3.1: createBuffer and GpuBuffer.allocate configure buffer properties',
        () {
          ResourceScope.scope(() {
            final usage = GpuBufferUsage.storage | GpuBufferUsage.copySrc;
            final buffer = GpuBuffer.allocate(sizeInBytes: 64, usage: usage);
            expect(buffer.sizeInBytes, equals(64));
            expect(buffer.refCount, equals(1));
            expect(buffer.isDisposed, isFalse);
            expect(buffer.usage.contains(GpuBufferUsage.storage), isTrue);
            expect(buffer.usage.contains(GpuBufferUsage.copySrc), isTrue);
            expect((buffer.usage & GpuBufferUsage.storage).value, isNonZero);
          });
        },
      );

      test(
        'F3.2: copyFromHost and copyToHost round-trip aligned and offset slices',
        () {
          using((arena) {
            final hostIn = arena<ffi.Float>(8);
            final hostOut = arena<ffi.Float>(8);
            for (var i = 0; i < 8; i++) {
              hostIn[i] = (i + 1) * 2.5;
              hostOut[i] = 0.0;
            }
            final buffer = GpuBuffer.allocate(sizeInBytes: 32);
            try {
              buffer.copyFromHost(hostIn.cast<ffi.Void>(), 16, offset: 0);
              buffer.copyFromHost(
                (hostIn + 4).cast<ffi.Void>(),
                16,
                offset: 16,
              );
              buffer.copyToHost(hostOut.cast<ffi.Void>(), 32, offset: 0);
              for (var i = 0; i < 8; i++) {
                expect(hostOut[i], closeTo((i + 1) * 2.5, 1e-6));
              }
            } finally {
              buffer.dispose();
            }
          });
        },
      );

      test(
        'F3.3: createBufferWithData and readBufferIntoPointer transfer host data',
        () {
          final device = GpuDevice.defaultDevice;
          using((arena) {
            final sourcePtr = arena<ffi.Int32>(4);
            final targetPtr = arena<ffi.Int32>(4);
            sourcePtr[0] = 42;
            sourcePtr[1] = -17;
            sourcePtr[2] = 1024;
            sourcePtr[3] = 9999;
            final buffer = device.createBufferWithData(
              sourcePtr.cast<ffi.Void>(),
              16,
            );
            try {
              device.readBufferIntoPointer(
                buffer,
                targetPtr.cast<ffi.Void>(),
                16,
              );
              expect([
                targetPtr[0],
                targetPtr[1],
                targetPtr[2],
                targetPtr[3],
              ], equals([42, -17, 1024, 9999]));
            } finally {
              buffer.dispose();
            }
          });
        },
      );

      test('F3.4: copyToBuffer copies sub-ranges between GPU buffers', () {
        using((arena) {
          final sourcePtr = arena<ffi.Float>(6);
          final targetPtr = arena<ffi.Float>(6);
          for (var i = 0; i < 6; i++) {
            sourcePtr[i] = (i + 1) * 10.0;
            targetPtr[i] = 0.0;
          }
          final firstBuffer = GpuDevice.defaultDevice.createBufferWithData(
            sourcePtr.cast<ffi.Void>(),
            24,
          );
          final secondBuffer = GpuBuffer.allocate(sizeInBytes: 24);
          try {
            secondBuffer.copyFromHost(targetPtr.cast<ffi.Void>(), 24);
            firstBuffer.copyToBuffer(
              secondBuffer,
              16,
              srcOffset: 4,
              dstOffset: 8,
            );
            secondBuffer.copyToHost(targetPtr.cast<ffi.Void>(), 24);
            expect([
              targetPtr[0],
              targetPtr[1],
              targetPtr[2],
              targetPtr[3],
              targetPtr[4],
              targetPtr[5],
            ], equals([0.0, 0.0, 20.0, 30.0, 40.0, 50.0]));
          } finally {
            firstBuffer.dispose();
            secondBuffer.dispose();
          }
        });
      });

      test('F3.5: retain and release reference counting controls disposal', () {
        final buffer = GpuBuffer.allocate(sizeInBytes: 32);
        expect(buffer.refCount, equals(1));
        buffer.retain();
        expect(buffer.refCount, equals(2));
        buffer.release();
        expect(buffer.refCount, equals(1));
        expect(buffer.isDisposed, isFalse);
        buffer.release();
        expect(buffer.refCount, equals(0));
        expect(buffer.isDisposed, isTrue);
      });
    });

    group('F4: GpuMemoryPool Bucketed Caching', () {
      test('F4.1: computeBucketSize rounds up to powers of two >= 64', () {
        expect(GpuMemoryPool.minBucketSize, equals(64));
        expect(GpuMemoryPool.computeBucketSize(0), equals(64));
        expect(GpuMemoryPool.computeBucketSize(1), equals(64));
        expect(GpuMemoryPool.computeBucketSize(64), equals(64));
        expect(GpuMemoryPool.computeBucketSize(65), equals(128));
        expect(GpuMemoryPool.computeBucketSize(129), equals(256));
        expect(GpuMemoryPool.computeBucketSize(1000), equals(1024));
      });

      test(
        'F4.2: acquire and dispose recycle buffers and track hits/misses',
        () async {
          final device = await createWebGpuDevice(
            name: 'Pool-Reuse-Device',
            enableMemoryPool: true,
          );
          try {
            final pool = device.memoryPool;
            expect(pool.hits, equals(0));
            expect(pool.misses, equals(0));

            final first = device.createBuffer(sizeInBytes: 80);
            expect(pool.misses, equals(1));
            expect(pool.cachedBytes, equals(0));
            first.dispose();
            expect(pool.cachedBytes, equals(128));

            final second = device.createBuffer(sizeInBytes: 96);
            expect(pool.hits, equals(1));
            expect(pool.cachedBytes, equals(0));
            second.dispose();
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F4.3: recycled pooled buffers are zero-initialized on re-acquire',
        () async {
          final device = await createWebGpuDevice(
            name: 'Pool-Zeroing-Device',
            enableMemoryPool: true,
          );
          try {
            final first = GpuArray.fromList(
              <double>[99.0, 88.0, 77.0, 66.0],
              [4],
              DType.float32,
              device: device,
            );
            _expectCloseList(first.toList(), <double>[99.0, 88.0, 77.0, 66.0]);
            first.dispose();

            final recycled = device.createBuffer(sizeInBytes: 16);
            try {
              using((arena) {
                final hostOut = arena<ffi.Float>(4);
                recycled.copyToHost(hostOut.cast<ffi.Void>(), 16);
                expect([
                  hostOut[0],
                  hostOut[1],
                  hostOut[2],
                  hostOut[3],
                ], equals([0.0, 0.0, 0.0, 0.0]));
              });
            } finally {
              recycled.dispose();
            }
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F4.4: memoryPool.trim frees cached buckets and resets cachedBytes',
        () async {
          final device = await createWebGpuDevice(
            name: 'Pool-Trim-Device',
            enableMemoryPool: true,
          );
          try {
            final buffer = device.createBuffer(sizeInBytes: 200);
            buffer.dispose();
            expect(device.memoryPool.cachedBytes, equals(256));
            device.memoryPool.trim();
            expect(device.memoryPool.cachedBytes, equals(0));
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F4.5: memoryPool tracks multiple bucket sizes and disposes cleanly',
        () async {
          final device = await createWebGpuDevice(
            name: 'Pool-Limit-Device',
            enableMemoryPool: true,
          );
          try {
            final pool = device.memoryPool;
            final first = pool.acquire(64);
            final second = pool.acquire(128);
            first.dispose();
            expect(pool.cachedBytes, equals(64));
            second.dispose();
            expect(pool.cachedBytes, equals(192));
            pool.dispose();
            expect(pool.isDisposed, isTrue);
            expect(pool.cachedBytes, equals(0));
          } finally {
            device.dispose();
          }
        },
      );
    });

    group('F5: NDArray Interop (fromNDArray, toNDArray, toGpu)', () {
      test('F5.1: contiguous 2D Float32 and Float64 NDArray round-trip', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final hostF32 = nd.NDArray.fromList(
              <double>[1.5, -2.5, 3.5, 4.5, -5.5, 6.5],
              [2, 3],
              nd.DType.float32,
            );
            final gpuF32 = GpuArray.fromNDArray(hostF32);
            final roundTripF32 = gpuF32.toNDArray();
            expect(roundTripF32.shape, equals([2, 3]));
            expect(roundTripF32.dtype, equals(nd.DType.float32));
            _expectCloseList(roundTripF32.toList(), hostF32.toList());

            final hostF64 = nd.NDArray.fromList(
              <double>[10.125, -20.25, 30.5, 40.75],
              [2, 2],
              nd.DType.float64,
            );
            final gpuF64 = GpuArray.fromNDArray(hostF64);
            final roundTripF64 = gpuF64.toNDArray();
            expect(roundTripF64.shape, equals([2, 2]));
            _expectCloseList(
              roundTripF64.toList(),
              hostF64.toList(),
              tolerance: 1e-6,
            );
          });
        });
      });

      test('F5.2: NDArray.toGpu and GpuArray.toHostNDArray extensions', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final hostInt = nd.NDArray.fromList(
              <int>[-10, 0, 10, 20],
              [4],
              nd.DType.int32,
            );
            final gpuInt = hostInt.toGpu();
            final backInt = gpuInt.toHostNDArray();
            expect(backInt.toList(), equals(<int>[-10, 0, 10, 20]));
          });
        });
      });

      test('F5.3: non-contiguous strided NDArray view uploads accurately', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final baseHost = nd.NDArray.fromList(
              <double>[1, 2, 3, 4, 5, 6],
              [2, 3],
              nd.DType.float32,
            );
            final transposedHost = baseHost.transpose();
            final gpuFromView = GpuArray.fromNDArray(transposedHost);
            expect(gpuFromView.shape, equals([3, 2]));
            expect(gpuFromView.isContiguous, isTrue);
            _expectCloseList(gpuFromView.toList(), <double>[1, 4, 2, 5, 3, 6]);
          });
        });
      });

      test(
        'F5.4: non-contiguous GpuArray view downloads to contiguous NDArray',
        () {
          nd.NDArray.scope(() {
            ResourceScope.scope(() {
              final gpuBase = GpuArray.fromList(
                <double>[1, 2, 3, 4, 5, 6],
                [2, 3],
                DType.float32,
              );
              final gpuTransposed = gpuBase.transpose();
              final hostResult = gpuTransposed.toNDArray();
              expect(hostResult.shape, equals([3, 2]));
              expect(hostResult.isContiguous, isTrue);
              _expectCloseList(hostResult.toList(), <double>[1, 4, 2, 5, 3, 6]);
            });
          });
        },
      );

      test('F5.5: scalar, toList, and toNestedList match NDArray', () {
        ResourceScope.scope(() {
          final zeroDim = GpuArray.fromList(<double>[42.5], [], DType.float32);
          expect((zeroDim.scalar as num).toDouble(), closeTo(42.5, 1e-5));

          final matrix = GpuArray.fromList(
            <int>[1, 2, 3, 4, 5, 6],
            [2, 3],
            DType.int32,
          );
          expect(matrix.toList(), equals(<int>[1, 2, 3, 4, 5, 6]));
          expect(
            matrix.toNestedList(),
            equals(<List<int>>[
              [1, 2, 3],
              [4, 5, 6],
            ]),
          );
        });
      });
    });

    group('F6: 15 DTypes & Strong Generic Type System', () {
      test(
        'F6.1: floating-point DTypes (float64, float32, float16, bfloat16)',
        () {
          ResourceScope.scope(() {
            for (final dtype in [
              DType.float64,
              DType.float32,
              DType.float16,
              DType.bfloat16,
            ]) {
              final first = GpuArray.fromList(
                <double>[1.5, 2.5, -3.0, 4.0],
                [4],
                dtype,
              );
              final second = GpuArray.fromList(
                <double>[0.5, 1.5, 3.0, -1.0],
                [4],
                dtype,
              );
              final added = first + second;
              expect(added.dtype, equals(dtype));
              _expectCloseList(added.toList(), <double>[
                2.0,
                4.0,
                0.0,
                3.0,
              ], tolerance: 5e-2);
            }
          });
        },
      );

      test('F6.2: signed integer DTypes (int64, int32, int16, int8)', () {
        ResourceScope.scope(() {
          for (final dtype in [
            DType.int64,
            DType.int32,
            DType.int16,
            DType.int8,
          ]) {
            final first = GpuArray.fromList(<int>[-10, 5, 20, -3], [4], dtype);
            final second = GpuArray.fromList(<int>[4, -5, 10, 7], [4], dtype);
            final added = first + second;
            expect(added.dtype, equals(dtype));
            expect(added.toList(), equals(<int>[-6, 0, 30, 4]));
          }
        });
      });

      test(
        'F6.3: unsigned integer & boolean DTypes (uint64..uint8, boolean)',
        () {
          ResourceScope.scope(() {
            for (final dtype in [
              DType.uint64,
              DType.uint32,
              DType.uint16,
              DType.uint8,
            ]) {
              final first = GpuArray.fromList(<int>[1, 10, 20, 30], [4], dtype);
              final second = GpuArray.fromList(<int>[2, 5, 10, 15], [4], dtype);
              final added = first + second;
              expect(added.dtype, equals(dtype));
              expect(added.toList(), equals(<int>[3, 15, 30, 45]));
            }

            final flags = GpuArray.fromList(
              <bool>[true, false, true, false],
              [2, 2],
              DType.boolean,
            );
            expect(flags.dtype, equals(DType.boolean));
            expect(flags.toList(), equals(<bool>[true, false, true, false]));
          });
        },
      );

      test(
        'F6.4: complex DTypes (complex64, complex128) round-trip and add',
        () {
          ResourceScope.scope(() {
            for (final dtype in [DType.complex64, DType.complex128]) {
              final first = GpuArray.fromList(
                <Complex>[Complex(1.0, 2.0), Complex(3.0, -4.0)],
                [2],
                dtype,
              );
              final second = GpuArray.fromList(
                <Complex>[Complex(0.5, -1.0), Complex(-1.0, 2.0)],
                [2],
                dtype,
              );
              final sumResult = first + second;
              expect(sumResult.dtype, equals(dtype));
              _expectCloseList(sumResult.toList(), <Complex>[
                Complex(1.5, 1.0),
                Complex(2.0, -2.0),
              ]);
            }
          });
        },
      );

      test('F6.5: GpuArray.promoteDTypes and DType metadata properties', () {
        expect(DType.values.length, equals(15));
        expect(
          GpuArray.promoteDTypes(DType.float32, DType.float64),
          equals(DType.float64),
        );
        expect(
          GpuArray.promoteDTypes(DType.int16, DType.int32),
          equals(DType.int32),
        );
        expect(
          GpuArray.promoteDTypes(DType.uint8, DType.int8),
          equals(DType.int16),
        );
        expect(
          GpuArray.promoteDTypes(DType.float32, DType.complex64),
          equals(DType.complex64),
        );
        expect(
          GpuArray.promoteDTypes(DType.float64, DType.complex64),
          equals(DType.complex128),
        );
      });
    });

    group('F7: Array Creation, Copy, Views & astype', () {
      test('F7.1: zeros, ones, filled, empty, and fromList creation', () {
        ResourceScope.scope(() {
          final zerosArray = GpuArray.zeros([2, 3], DType.float32);
          final onesArray = GpuArray.ones([2, 3], DType.float32);
          final filledArray = GpuArray.filled([2, 3], 7.5, DType.float32);
          final emptyArray = GpuArray.empty([2, 3], DType.float32);

          expect(zerosArray.shape, equals([2, 3]));
          expect(emptyArray.size, equals(6));
          _expectCloseList(zerosArray.toList(), List<double>.filled(6, 0.0));
          _expectCloseList(onesArray.toList(), List<double>.filled(6, 1.0));
          _expectCloseList(filledArray.toList(), List<double>.filled(6, 7.5));
        });
      });

      test(
        'F7.2: copy() and copy(out:) isolate buffers and materialize views',
        () {
          ResourceScope.scope(() {
            final original = GpuArray.fromList(
              <double>[1, 2, 3, 4, 5, 6],
              [2, 3],
              DType.float32,
            );
            final transposed = original.transpose();
            final cloned = transposed.copy();
            expect(cloned.shape, equals([3, 2]));
            expect(cloned.isContiguous, isTrue);
            _expectCloseList(cloned.toList(), <double>[1, 4, 2, 5, 3, 6]);

            final outTarget = GpuArray.zeros([3, 2], DType.float32);
            final returned = transposed.copy(out: outTarget);
            expect(identical(returned, outTarget), isTrue);
            _expectCloseList(outTarget.toList(), <double>[1, 4, 2, 5, 3, 6]);
          });
        },
      );

      test(
        'F7.3: astype and astype(out:) cast across numeric and boolean DTypes',
        () {
          ResourceScope.scope(() {
            final floats = GpuArray.fromList(
              <double>[0.0, 1.9, -2.1, 4.0],
              [4],
              DType.float32,
            );
            final ints = floats.astype(DType.int32);
            expect(ints.dtype, equals(DType.int32));
            expect(ints.toList(), equals(<int>[0, 1, -2, 4]));

            final outBool = GpuArray.empty([4], DType.boolean);
            final bools = ints.astype(DType.boolean, out: outBool);
            expect(identical(bools, outBool), isTrue);
            expect(bools.toList(), equals(<bool>[false, true, true, true]));
          });
        },
      );

      test('F7.4: reshape, flatten, squeeze, and unsqueeze views', () {
        ResourceScope.scope(() {
          final baseTensor = GpuArray.fromList(
            <double>[1, 2, 3, 4, 5, 6],
            [2, 3],
            DType.float32,
          );
          final reshaped = baseTensor.reshape([3, 1, 2]);
          expect(reshaped.shape, equals([3, 1, 2]));

          final squeezed = reshaped.squeeze(axis: 1);
          expect(squeezed.shape, equals([3, 2]));

          final unsqueezed = squeezed.unsqueeze(0);
          expect(unsqueezed.shape, equals([1, 3, 2]));

          final flattened = baseTensor.flatten();
          expect(flattened.shape, equals([6]));
          _expectCloseList(flattened.toList(), <double>[1, 2, 3, 4, 5, 6]);
        });
      });

      test('F7.5: 2D and 3D transpose views match NDArray.transpose', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final data = List<double>.generate(12, (i) => i.toDouble());
            final gpuTensor = GpuArray.fromList(data, [2, 3, 2], DType.float32);
            final hostTensor = nd.NDArray.fromList(data, [
              2,
              3,
              2,
            ], nd.DType.float32);

            final gpuPermuted = gpuTensor.transpose([1, 0, 2]);
            final hostPermuted = hostTensor.transpose([1, 0, 2]);
            expect(gpuPermuted.shape, equals(hostPermuted.shape));
            _expectCloseList(gpuPermuted.toList(), hostPermuted.toList());
          });
        });
      });
    });

    group('F8: Elementwise Binary, Unary & Comparison Ufuncs', () {
      test('F8.1: binary arithmetic and broadcasting match NDArray oracle', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final colData = <double>[1.0, 2.0, 3.0];
            final rowData = <double>[10.0, 20.0, 30.0, 40.0];
            final gpuCol = GpuArray.fromList(colData, [3, 1], DType.float32);
            final gpuRow = GpuArray.fromList(rowData, [1, 4], DType.float32);
            final hostCol = nd.NDArray.fromList(colData, [
              3,
              1,
            ], nd.DType.float32);
            final hostRow = nd.NDArray.fromList(rowData, [
              1,
              4,
            ], nd.DType.float32);

            _expectCloseList(
              (gpuCol + gpuRow).toList(),
              (hostCol + hostRow).toList(),
            );
            _expectCloseList(
              (gpuRow - gpuCol).toList(),
              (hostRow - hostCol).toList(),
            );
            _expectCloseList(
              (gpuCol * gpuRow).toList(),
              (hostCol * hostRow).toList(),
            );
            _expectCloseList(
              (gpuRow / gpuCol).toList(),
              (hostRow / hostCol).toList(),
            );
            _expectCloseList(gpuCol.maximum(gpuRow).toList(), <double>[
              10.0,
              20.0,
              30.0,
              40.0,
              10.0,
              20.0,
              30.0,
              40.0,
              10.0,
              20.0,
              30.0,
              40.0,
            ]);
            _expectCloseList(gpuCol.minimum(gpuRow).toList(), <double>[
              1.0,
              1.0,
              1.0,
              1.0,
              2.0,
              2.0,
              2.0,
              2.0,
              3.0,
              3.0,
              3.0,
              3.0,
            ]);
          });
        });
      });

      test('F8.2: pow, remainder, scalar operands, and out: parameter', () {
        ResourceScope.scope(() {
          final baseVals = GpuArray.fromList(
            <double>[2.0, 3.0, 4.0, 9.0],
            [4],
            DType.float32,
          );
          final exponents = GpuArray.fromList(
            <double>[3.0, 2.0, 0.5, 1.5],
            [4],
            DType.float32,
          );
          _expectCloseList(baseVals.pow(exponents).toList(), <double>[
            8.0,
            9.0,
            2.0,
            27.0,
          ]);
          _expectCloseList(baseVals.remainder(2.5).toList(), <double>[
            2.0,
            0.5,
            1.5,
            1.5,
          ]);

          final outBuffer = GpuArray.zeros([4], DType.float32);
          final returned = baseVals.add(5.0, out: outBuffer);
          expect(identical(returned, outBuffer), isTrue);
          _expectCloseList(outBuffer.toList(), <double>[7.0, 8.0, 9.0, 14.0]);
        });
      });

      test('F8.3: unary math functions match NDArray oracle', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final raw = <double>[0.25, 1.0, 2.25, 4.0];
            final gpuX = GpuArray.fromList(raw, [4], DType.float32);
            final hostX = nd.NDArray.fromList(raw, [4], nd.DType.float32);

            _expectCloseList((-gpuX).toList(), (-hostX).toList());
            _expectCloseList((-gpuX).abs().toList(), nd.abs(-hostX).toList());
            _expectCloseList(gpuX.sqrt().toList(), nd.sqrt(hostX).toList());
            _expectCloseList(gpuX.exp().toList(), nd.exp(hostX).toList());
            _expectCloseList(gpuX.log().toList(), nd.log(hostX).toList());

            final roundingInput = GpuArray.fromList(
              <double>[-1.7, -0.2, 0.4, 2.8],
              [4],
              DType.float32,
            );
            _expectCloseList(roundingInput.floor().toList(), <double>[
              -2.0,
              -1.0,
              0.0,
              2.0,
            ]);
            _expectCloseList(roundingInput.ceil().toList(), <double>[
              -1.0,
              0.0,
              1.0,
              3.0,
            ]);
            _expectCloseList(roundingInput.round().toList(), <double>[
              -2.0,
              0.0,
              0.0,
              3.0,
            ]);
          });
        });
      });

      test('F8.4: trigonometric and hyperbolic unary ufuncs match NDArray', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final angles = <double>[-0.5, 0.0, 0.25, 0.75];
            final gpuA = GpuArray.fromList(angles, [4], DType.float32);
            final hostA = nd.NDArray.fromList(angles, [4], nd.DType.float32);

            _expectCloseList(gpuA.sin().toList(), nd.sin(hostA).toList());
            _expectCloseList(gpuA.cos().toList(), nd.cos(hostA).toList());
            _expectCloseList(gpuA.tan().toList(), nd.tan(hostA).toList());
            _expectCloseList(
              gpuA.asin().toList(),
              nd.asin(hostA).toList(),
              tolerance: 1e-3,
            );
            _expectCloseList(
              gpuA.acos().toList(),
              nd.acos(hostA).toList(),
              tolerance: 1e-3,
            );
            _expectCloseList(
              gpuA.atan().toList(),
              nd.atan(hostA).toList(),
              tolerance: 1e-3,
            );
            _expectCloseList(gpuA.sinh().toList(), nd.sinh(hostA).toList());
            _expectCloseList(gpuA.cosh().toList(), nd.cosh(hostA).toList());
            _expectCloseList(gpuA.tanh().toList(), nd.tanh(hostA).toList());
          });
        });
      });

      test(
        'F8.5: comparison ufuncs produce GpuArray<Boolean> matching NDArray',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.fromList(
              <double>[1.0, 5.0, 3.0, 4.0],
              [4],
              DType.float32,
            );
            final right = GpuArray.fromList(
              <double>[2.0, 5.0, 1.0, 6.0],
              [4],
              DType.float32,
            );

            expect(
              left.equal(right).toList(),
              equals([false, true, false, false]),
            );
            expect(
              left.notEqual(right).toList(),
              equals([true, false, true, true]),
            );
            expect(
              left.greater(right).toList(),
              equals([false, false, true, false]),
            );
            expect(
              left.greaterEqual(right).toList(),
              equals([false, true, true, false]),
            );
            expect(
              left.less(right).toList(),
              equals([true, false, false, true]),
            );
            expect(
              left.lessEqual(right).toList(),
              equals([true, true, false, true]),
            );
          });
        },
      );
    });

    group('F9: Reductions (sum, mean, prod, min, max)', () {
      test('F9.1: full-array reductions match NDArray oracle', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final values = <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            final gpuTensor = GpuArray.fromList(values, [2, 3], DType.float32);
            final hostTensor = nd.NDArray.fromList(values, [
              2,
              3,
            ], nd.DType.float32);

            expect(
              (gpuTensor.sum().scalar as num).toDouble(),
              closeTo((nd.sum(hostTensor).scalar as num).toDouble(), 1e-4),
            );
            expect(
              (gpuTensor.mean().scalar as num).toDouble(),
              closeTo((nd.mean(hostTensor).scalar as num).toDouble(), 1e-4),
            );
            expect(
              (gpuTensor.prod().scalar as num).toDouble(),
              closeTo((nd.prod(hostTensor).scalar as num).toDouble(), 1e-4),
            );
            expect(
              (gpuTensor.min().scalar as num).toDouble(),
              closeTo((nd.min(hostTensor).scalar as num).toDouble(), 1e-4),
            );
            expect(
              (gpuTensor.max().scalar as num).toDouble(),
              closeTo((nd.max(hostTensor).scalar as num).toDouble(), 1e-4),
            );
          });
        });
      });

      test('F9.2: axis reductions along positive and negative axes', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final values = <double>[1, 2, 3, 4, 5, 6];
            final gpuTensor = GpuArray.fromList(values, [2, 3], DType.float32);
            final hostTensor = nd.NDArray.fromList(values, [
              2,
              3,
            ], nd.DType.float32);

            _expectCloseList(
              gpuTensor.sum(axis: 0).toList(),
              nd.sum(hostTensor, axis: 0).toList(),
            );
            _expectCloseList(
              gpuTensor.sum(axis: 1).toList(),
              nd.sum(hostTensor, axis: 1).toList(),
            );
            _expectCloseList(
              gpuTensor.mean(axis: -1).toList(),
              nd.mean(hostTensor, axis: -1).toList(),
            );
            _expectCloseList(
              gpuTensor.min(axis: 0).toList(),
              nd.min(hostTensor, axis: 0).toList(),
            );
            _expectCloseList(
              gpuTensor.max(axis: 1).toList(),
              nd.max(hostTensor, axis: 1).toList(),
            );
          });
        });
      });

      test('F9.3: keepDims: true preserves rank across reductions', () {
        ResourceScope.scope(() {
          final tensor = GpuArray.fromList(
            <double>[1, 2, 3, 4, 5, 6],
            [2, 3],
            DType.float32,
          );
          final sumKeep = tensor.sum(axis: 1, keepDims: true);
          final maxKeep = tensor.max(keepDims: true);
          expect(sumKeep.shape, equals([2, 1]));
          _expectCloseList(sumKeep.toList(), <double>[6.0, 15.0]);
          expect(maxKeep.shape, equals([1, 1]));
          _expectCloseList(maxKeep.toList(), <double>[6.0]);
        });
      });

      test('F9.4: reductions with pre-allocated out: destination', () {
        ResourceScope.scope(() {
          final tensor = GpuArray.fromList(
            <double>[2, 4, 6, 8, 10, 12],
            [2, 3],
            DType.float32,
          );
          final outAxis = GpuArray.zeros([2], DType.float32);
          final result = tensor.sum(axis: 1, out: outAxis);
          expect(identical(result, outAxis), isTrue);
          _expectCloseList(outAxis.toList(), <double>[12.0, 30.0]);
        });
      });

      test('F9.5: reductions on transposed and sliced views match NDArray', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final values = List<double>.generate(12, (i) => (i + 1).toDouble());
            final gpuView = GpuArray.fromList(values, [
              3,
              4,
            ], DType.float32).transpose();
            final hostView = nd.NDArray.fromList(values, [
              3,
              4,
            ], nd.DType.float32).transpose();

            _expectCloseList(
              gpuView.sum(axis: 0).toList(),
              nd.sum(hostView, axis: 0).toList(),
            );
            _expectCloseList(
              gpuView.max(axis: 1).toList(),
              nd.max(hostView, axis: 1).toList(),
            );
          });
        });
      });
    });

    group('F10: Matrix Multiplication & Dot Products (matmul, dot, vdot)', () {
      test('F10.1: 1D inner product via matmul, dot, and complex vdot', () {
        ResourceScope.scope(() {
          final firstVec = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0],
            [3],
            DType.float32,
          );
          final secondVec = GpuArray.fromList(
            <double>[4.0, 5.0, 6.0],
            [3],
            DType.float32,
          );
          expect(
            (firstVec.matmul(secondVec).scalar as num).toDouble(),
            closeTo(32.0, 1e-4),
          );
          expect(
            (firstVec.dot(secondVec).scalar as num).toDouble(),
            closeTo(32.0, 1e-4),
          );

          final complexA = GpuArray.fromList(
            <Complex>[Complex(1.0, 2.0), Complex(3.0, -1.0)],
            [2],
            DType.complex128,
          );
          final complexB = GpuArray.fromList(
            <Complex>[Complex(2.0, 1.0), Complex(1.0, 4.0)],
            [2],
            DType.complex128,
          );
          final vdotResult =
              gpu_linalg.vdot(complexA, complexB).scalar as Complex;
          expect(vdotResult.real, closeTo(3.0, 1e-5));
          expect(vdotResult.imag, closeTo(10.0, 1e-5));
        });
      });

      test('F10.2: 2D rectangular GEMM matches NDArray.matmul', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final leftVals = List<double>.generate(
              6,
              (i) => (i + 1).toDouble(),
            );
            final rightVals = List<double>.generate(12, (i) => i * 0.5);
            final gpuLeft = GpuArray.fromList(leftVals, [2, 3], DType.float32);
            final gpuRight = GpuArray.fromList(rightVals, [
              3,
              4,
            ], DType.float32);
            final hostLeft = nd.NDArray.fromList(leftVals, [
              2,
              3,
            ], nd.DType.float32);
            final hostRight = nd.NDArray.fromList(rightVals, [
              3,
              4,
            ], nd.DType.float32);

            final gpuProduct = gpuLeft.matmul(gpuRight);
            final hostProduct = nd.matmul(hostLeft, hostRight);
            expect(gpuProduct.shape, equals([2, 4]));
            _expectCloseList(gpuProduct.toList(), hostProduct.toList());
          });
        });
      });

      test('F10.3: 3D batched matmul with broadcast batch dimension', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final leftVals = List<double>.generate(
              12,
              (i) => (i + 1).toDouble(),
            );
            final rightVals = List<double>.generate(6, (i) => i + 0.5);
            final gpuLeft = GpuArray.fromList(leftVals, [
              2,
              2,
              3,
            ], DType.float32);
            final gpuRight = GpuArray.fromList(rightVals, [
              1,
              3,
              2,
            ], DType.float32);
            final hostLeft = nd.NDArray.fromList(leftVals, [
              2,
              2,
              3,
            ], nd.DType.float32);
            final hostRight = nd.NDArray.fromList(rightVals, [
              1,
              3,
              2,
            ], nd.DType.float32);

            final gpuBatched = gpuLeft.matmul(gpuRight);
            final hostBatched = nd.matmul(hostLeft, hostRight);
            expect(gpuBatched.shape, equals([2, 2, 2]));
            _expectCloseList(gpuBatched.toList(), hostBatched.toList());
          });
        });
      });

      test('F10.4: matmul on transposed strided views matches NDArray', () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final leftVals = <double>[1, 2, 3, 4, 5, 6];
            final rightVals = <double>[7, 8, 9, 10, 11, 12];
            final gpuLeft = GpuArray.fromList(leftVals, [
              3,
              2,
            ], DType.float32).transpose();
            final gpuRight = GpuArray.fromList(rightVals, [
              2,
              3,
            ], DType.float32).transpose();
            final hostLeft = nd.NDArray.fromList(leftVals, [
              3,
              2,
            ], nd.DType.float32).transpose();
            final hostRight = nd.NDArray.fromList(rightVals, [
              2,
              3,
            ], nd.DType.float32).transpose();

            final gpuOut = gpuLeft.matmul(gpuRight);
            final hostOut = nd.matmul(hostLeft, hostRight);
            expect(gpuOut.shape, equals([2, 2]));
            _expectCloseList(gpuOut.toList(), hostOut.toList());
          });
        });
      });

      test(
        'F10.5: matmul and gpu_linalg.matmul write into pre-allocated out:',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.fromList(
              <double>[1, 2, 3, 4],
              [2, 2],
              DType.float32,
            );
            final right = GpuArray.fromList(
              <double>[5, 6, 7, 8],
              [2, 2],
              DType.float32,
            );
            final outMatrix = GpuArray.zeros([2, 2], DType.float32);
            final returned = gpu_linalg.matmul(left, right, out: outMatrix);
            expect(identical(returned, outMatrix), isTrue);
            _expectCloseList(outMatrix.toList(), <double>[
              19.0,
              22.0,
              43.0,
              50.0,
            ]);
          });
        },
      );
    });

    group(
      'R1 & R4 Core Overhaul: DTypeTag Preservation, Statistical Reductions & Ufuncs',
      () {
        test(
          'R1.1: GpuArray<Float32> binary and scalar ops preserve Float32 without casts',
          () {
            ResourceScope.scope(() {
              final a = GpuArray<Float32>.fromList(
                [4.0, 9.0, 16.0],
                [3],
                DType.float32,
              );
              final b = GpuArray<Float32>.fromList(
                [2.0, 3.0, 4.0],
                [3],
                DType.float32,
              );
              final GpuArray<Float32> sumArr = a + b;
              final GpuArray<Float32> scalarArr = a + 1.5;
              final GpuArray<Float32> powArr = b.pow(2.0);
              final GpuArray<Float32> maxArr = a.maximum(10.0);
              final GpuArray<Float32> minArr = a.minimum(10.0);
              expect(sumArr.dtype, equals(DType.float32));
              expect(scalarArr.dtype, equals(DType.float32));
              expect(powArr.dtype, equals(DType.float32));
              _expectCloseList(sumArr.toList(), <double>[6.0, 12.0, 20.0]);
              _expectCloseList(scalarArr.toList(), <double>[5.5, 10.5, 17.5]);
              _expectCloseList(powArr.toList(), <double>[4.0, 9.0, 16.0]);
              _expectCloseList(maxArr.toList(), <double>[10.0, 10.0, 16.0]);
              _expectCloseList(minArr.toList(), <double>[4.0, 9.0, 10.0]);
            });
          },
        );

        test(
          'R1.2 & R1.3: mean() preserves float dtypes and argmin/argmax/countNonzero return Int64',
          () {
            ResourceScope.scope(() {
              final f32 = GpuArray<Float32>.fromList(
                [1.0, 0.0, 3.0, 4.0],
                [4],
                DType.float32,
              );
              final GpuArray<Float32> m32 = f32.mean();
              final GpuArray<Int64> idxMin = f32.argmin();
              final GpuArray<Int64> idxMax = f32.argmax();
              final GpuArray<Int64> cnz = f32.countNonzero();
              expect(m32.dtype, equals(DType.float32));
              expect(m32.scalar, closeTo(2.0, 1e-5));
              expect(idxMin.dtype, equals(DType.int64));
              expect(idxMin.scalar, equals(1));
              expect(idxMax.dtype, equals(DType.int64));
              expect(idxMax.scalar, equals(3));
              expect(cnz.dtype, equals(DType.int64));
              expect(cnz.scalar, equals(3));
            });
          },
        );

        test(
          'R4.3 & R4.4: variance, std, ptp, NaN reductions, bitwise & complex ufuncs',
          () {
            ResourceScope.scope(() {
              final x = GpuArray<Float32>.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [4],
                DType.float32,
              );
              expect(variance(x).scalar, closeTo(1.25, 1e-5));
              expect(std(x).scalar, closeTo(1.118034, 1e-4));
              expect(ptp(x).scalar, closeTo(3.0, 1e-5));

              final withNan = GpuArray<Float32>.fromList(
                [2.0, double.nan, 4.0],
                [3],
                DType.float32,
              );
              expect(nansum(withNan).scalar, closeTo(6.0, 1e-5));
              expect(nanmean(withNan).scalar, closeTo(3.0, 1e-5));
              expect(nanmin(withNan).scalar, closeTo(2.0, 1e-5));
              expect(nanmax(withNan).scalar, closeTo(4.0, 1e-5));
              expect(isnan(withNan).toList(), equals([false, true, false]));
              _expectCloseList(nanToNum(withNan, nan: 0.0).toList(), <double>[
                2.0,
                0.0,
                4.0,
              ]);

              final bitsA = GpuArray<Int32>.fromList([6, 12], [2], DType.int32);
              final bitsB = GpuArray<Int32>.fromList([3, 5], [2], DType.int32);
              expect((bitsA & bitsB).toList(), equals([2, 4]));
              expect((bitsA | bitsB).toList(), equals([7, 13]));
              expect((bitsA ^ bitsB).toList(), equals([5, 9]));

              final c = GpuArray<Complex64>.fromList(
                [Complex(3.0, 4.0)],
                [1],
                DType.complex64,
              );
              expect(c.real().toList(), equals([3.0]));
              expect(c.imag().toList(), equals([4.0]));
              expect(c.conjugate().toList(), equals([Complex(3.0, -4.0)]));
            });
          },
        );
      },
    );
  });
}
