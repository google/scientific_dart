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
import 'package:ndarray/ndarray.dart' as nd;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

final Matcher _throwsShapeOrArgError = throwsA(
  anyOf(isA<ArgumentError>(), isA<GpuException>()),
);

void main() {
  group('Tier 2 — Backend, Memory & Core Boundary/Negative (F1–F10)', () {
    group('F1: WebGPU Native Backend & Device Creation Boundaries', () {
      test(
        'F1.B1: createBuffer on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-F1-1');
          device.dispose();
          expect(
            () => device.createBuffer(sizeInBytes: 32),
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );

      test(
        'F1.B2: createBufferWithData on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-F1-2');
          device.dispose();
          using((arena) {
            final hostPtr = arena<ffi.Float>(4);
            expect(
              () => device.createBufferWithData(hostPtr.cast<ffi.Void>(), 16),
              throwsA(isA<GpuDeviceDisposedException>()),
            );
          });
        },
      );

      test(
        'F1.B3: readBufferIntoPointer on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-F1-3');
          final liveBuffer = GpuBuffer.allocate(sizeInBytes: 16);
          device.dispose();
          try {
            using((arena) {
              final hostPtr = arena<ffi.Float>(4);
              expect(
                () => device.readBufferIntoPointer(
                  liveBuffer,
                  hostPtr.cast<ffi.Void>(),
                  16,
                ),
                throwsA(isA<GpuDeviceDisposedException>()),
              );
            });
          } finally {
            liveBuffer.dispose();
          }
        },
      );

      test(
        'F1.B4: synchronize on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-F1-4');
          device.dispose();
          expect(
            () => device.synchronize(),
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );

      test(
        'F1.B5: jitCompiler access on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-F1-5');
          device.dispose();
          expect(
            () => device.jitCompiler,
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );
    });

    group('F2: GpuDevice.defaultDevice & Lifecycle Boundaries', () {
      test(
        'F2.B1: setting defaultDevice to disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-Default');
          device.dispose();
          expect(
            () => GpuDevice.defaultDevice = device,
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );

      test(
        'F2.B2: detachFromScope on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-Detach');
          device.dispose();
          expect(
            () => device.detachFromScope(),
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );

      test(
        'F2.B3: detachToParentScope on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-Promote');
          device.dispose();
          expect(
            () => device.detachToParentScope(),
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );

      test(
        'F2.B4: defaultDevice re-initializes if previous defaultDevice was disposed',
        () async {
          final original = GpuDevice.defaultDevice;
          final temporary = await createWebGpuDevice(name: 'Temp-Default');
          try {
            GpuDevice.defaultDevice = temporary;
            temporary.dispose();
            final reinitialized = GpuDevice.defaultDevice;
            expect(reinitialized.isDisposed, isFalse);
          } finally {
            if (!original.isDisposed) {
              GpuDevice.defaultDevice = original;
            }
          }
        },
      );

      test(
        'F2.B5: exception inside ResourceScope.scope still disposes tracked resources',
        () async {
          final device = await createWebGpuDevice(
            name: 'Scope-Exception-Device',
            enableMemoryPool: false,
          );
          try {
            late final GpuBuffer leakedCandidate;
            expect(
              () => ResourceScope.scope(() {
                leakedCandidate = device.createBuffer(sizeInBytes: 64);
                throw StateError('Simulated failure inside scope');
              }),
              throwsStateError,
            );
            expect(leakedCandidate.isDisposed, isTrue);
            expect(device.activeBufferCount, equals(0));
          } finally {
            device.dispose();
          }
        },
      );
    });

    group('F3: GpuBuffer Allocation & Transfers Boundaries', () {
      test(
        'F3.B1: negative sizeInBytes in GpuBuffer.allocate throws RangeError',
        () {
          expect(() => GpuBuffer.allocate(sizeInBytes: -8), throwsRangeError);
        },
      );

      test(
        'F3.B2: GpuBuffer.allocate on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-Alloc');
          device.dispose();
          expect(
            () => GpuBuffer.allocate(sizeInBytes: 16, device: device),
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );

      test(
        'F3.B3: copyFromHost out of bounds throws GpuMemoryException or RangeError',
        () {
          using((arena) {
            final hostPtr = arena<ffi.Uint8>(32);
            final buffer = GpuBuffer.allocate(sizeInBytes: 16);
            try {
              expect(
                () => buffer.copyFromHost(
                  hostPtr.cast<ffi.Void>(),
                  24,
                  offset: 0,
                ),
                throwsA(anyOf(isA<GpuMemoryException>(), isA<RangeError>())),
              );
              expect(
                () => buffer.copyFromHost(
                  hostPtr.cast<ffi.Void>(),
                  8,
                  offset: 12,
                ),
                throwsA(anyOf(isA<GpuMemoryException>(), isA<RangeError>())),
              );
            } finally {
              buffer.dispose();
            }
          });
        },
      );

      test(
        'F3.B4: copyToHost and copyToBuffer out of bounds or disposed target throw',
        () {
          using((arena) {
            final hostPtr = arena<ffi.Uint8>(32);
            final srcBuffer = GpuBuffer.allocate(sizeInBytes: 16);
            final dstBuffer = GpuBuffer.allocate(sizeInBytes: 16);
            try {
              expect(
                () => srcBuffer.copyToHost(
                  hostPtr.cast<ffi.Void>(),
                  20,
                  offset: 0,
                ),
                throwsA(anyOf(isA<GpuMemoryException>(), isA<RangeError>())),
              );
              expect(
                () => srcBuffer.copyToBuffer(dstBuffer, 16, srcOffset: 4),
                throwsA(anyOf(isA<GpuMemoryException>(), isA<RangeError>())),
              );
              dstBuffer.dispose();
              expect(
                () => srcBuffer.copyToBuffer(dstBuffer, 8),
                throwsStateError,
              );
            } finally {
              srcBuffer.dispose();
            }
          });
        },
      );

      test('F3.B5: operations on disposed GpuBuffer throw StateError', () {
        using((arena) {
          final hostPtr = arena<ffi.Uint8>(8);
          final buffer = GpuBuffer.allocate(sizeInBytes: 8);
          buffer.dispose();
          expect(buffer.isDisposed, isTrue);
          buffer.dispose();
          expect(() => buffer.retain(), throwsStateError);
          expect(
            () => buffer.copyFromHost(hostPtr.cast<ffi.Void>(), 4),
            throwsStateError,
          );
          expect(
            () => buffer.copyToHost(hostPtr.cast<ffi.Void>(), 4),
            throwsStateError,
          );
          expect(() => buffer.detachFromScope(), throwsStateError);
          expect(() => buffer.detachToParentScope(), throwsStateError);
        });
      });
    });

    group('F4: GpuMemoryPool Boundaries', () {
      test('F4.B1: negative size in pool.acquire throws RangeError', () async {
        final device = await createWebGpuDevice(enableMemoryPool: true);
        try {
          expect(() => device.memoryPool.acquire(-1), throwsRangeError);
        } finally {
          device.dispose();
        }
      });

      test(
        'F4.B2: zero-byte pool.acquire succeeds with sizeInBytes == 0',
        () async {
          final device = await createWebGpuDevice(enableMemoryPool: true);
          try {
            final zeroBuffer = device.memoryPool.acquire(0);
            expect(zeroBuffer.sizeInBytes, equals(0));
            zeroBuffer.dispose();
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F4.B3: acquire on disposed pool throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(enableMemoryPool: true);
          try {
            final pool = device.memoryPool;
            pool.dispose();
            expect(
              () => pool.acquire(64),
              throwsA(isA<GpuDeviceDisposedException>()),
            );
          } finally {
            device.dispose();
          }
        },
      );

      test(
        'F4.B4: acquire on pool of disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(enableMemoryPool: true);
          final pool = device.memoryPool;
          device.dispose();
          expect(
            () => pool.acquire(64),
            throwsA(isA<GpuDeviceDisposedException>()),
          );
        },
      );

      test(
        'F4.B5: pool.dispose is idempotent and trim on empty pool is a no-op',
        () async {
          final device = await createWebGpuDevice(enableMemoryPool: true);
          try {
            device.memoryPool.trim();
            expect(device.memoryPool.cachedBytes, equals(0));
            device.memoryPool.dispose();
            device.memoryPool.dispose();
            expect(device.memoryPool.isDisposed, isTrue);
          } finally {
            device.dispose();
          }
        },
      );
    });

    group('F5: NDArray Interop Boundaries', () {
      test(
        'F5.B1: GpuArray.fromNDArray on disposed device throws GpuDeviceDisposedException',
        () async {
          final device = await createWebGpuDevice(name: 'Disposed-Interop');
          device.dispose();
          nd.NDArray.scope(() {
            final host = nd.NDArray.zeros([2, 2], nd.DType.float32);
            expect(
              () => GpuArray.fromNDArray(host, device: device),
              throwsA(isA<GpuDeviceDisposedException>()),
            );
          });
        },
      );

      test('F5.B2: toNDArray on disposed GpuArray throws StateError', () {
        final gpu = GpuArray.ones([2, 2], DType.float32);
        gpu.dispose();
        expect(() => gpu.toNDArray(), throwsStateError);
      });

      test(
        'F5.B3: toList and toNestedList on disposed GpuArray throw StateError',
        () {
          final gpu = GpuArray.ones([2, 2], DType.float32);
          gpu.dispose();
          expect(() => gpu.toList(), throwsStateError);
          expect(() => gpu.toNestedList(), throwsStateError);
        },
      );

      test(
        'F5.B4: scalar getter on multi-element or empty array throws StateError',
        () {
          ResourceScope.scope(() {
            final multi = GpuArray.ones([2, 2], DType.float32);
            expect(() => multi.scalar, throwsStateError);
            final empty = GpuArray.zeros([0], DType.float32);
            expect(() => empty.scalar, throwsStateError);
          });
        },
      );

      test(
        'F5.B5: empty 1D and 2D NDArray round-trips preserve shape and zero size',
        () {
          nd.NDArray.scope(() {
            ResourceScope.scope(() {
              final hostEmpty = nd.NDArray.zeros([0, 4], nd.DType.float32);
              final gpuEmpty = GpuArray.fromNDArray(hostEmpty);
              expect(gpuEmpty.shape, equals([0, 4]));
              expect(gpuEmpty.size, equals(0));
              final backEmpty = gpuEmpty.toNDArray();
              expect(backEmpty.shape, equals([0, 4]));
              expect(backEmpty.size, equals(0));
            });
          });
        },
      );
    });

    group('F6: 15 DTypes & Strong Generic Type System Boundaries', () {
      test(
        'F6.B1: fromList with incompatible element type throws ArgumentError',
        () {
          expect(
            () => GpuArray.fromList(<Object>['not_a_num'], [1], DType.float32),
            throwsArgumentError,
          );
        },
      );

      test('F6.B2: ordering comparisons on Complex DTypes throw error', () {
        ResourceScope.scope(() {
          final c1 = GpuArray.fromList(
            <Complex>[Complex(1.0, 2.0)],
            [1],
            DType.complex64,
          );
          final c2 = GpuArray.fromList(
            <Complex>[Complex(2.0, 1.0)],
            [1],
            DType.complex64,
          );
          expect(
            () => c1.greater(c2),
            throwsA(anyOf(isA<UnsupportedError>(), isA<ArgumentError>())),
          );
          expect(
            () => c1.less(c2),
            throwsA(anyOf(isA<UnsupportedError>(), isA<ArgumentError>())),
          );
        });
      });

      test(
        'F6.B3: sub-4-byte DTypes with odd element counts (1, 3, 5) round-trip',
        () {
          ResourceScope.scope(() {
            final oddU8 = GpuArray.fromList(
              <int>[7, 13, 255],
              [3],
              DType.uint8,
            );
            expect(oddU8.toList(), equals(<int>[7, 13, 255]));

            final oddI16 = GpuArray.fromList(
              <int>[-300, 0, 300, -1, 42],
              [5],
              DType.int16,
            );
            expect(oddI16.toList(), equals(<int>[-300, 0, 300, -1, 42]));

            final oddBool = GpuArray.fromList(<bool>[true], [1], DType.boolean);
            expect(oddBool.toList(), equals(<bool>[true]));
          });
        },
      );

      test(
        'F6.B4: IEEE-754 special values (NaN, Infinity, -Infinity) round-trip',
        () {
          ResourceScope.scope(() {
            final specials = GpuArray.fromList(
              <double>[
                double.infinity,
                double.negativeInfinity,
                double.nan,
                -0.0,
              ],
              [4],
              DType.float32,
            );
            final readBack = specials.toList().cast<double>();
            expect(readBack[0], equals(double.infinity));
            expect(readBack[1], equals(double.negativeInfinity));
            expect(readBack[2].isNaN, isTrue);
            expect(readBack[3], equals(0.0));
          });
        },
      );

      test(
        'F6.B5: binary operations across distinct GpuDevice instances throw',
        () async {
          final firstDevice = await createWebGpuDevice(name: 'Device-Alpha');
          final secondDevice = await createWebGpuDevice(name: 'Device-Beta');
          try {
            final onFirst = GpuArray.ones(
              [2],
              DType.float32,
              device: firstDevice,
            );
            final onSecond = GpuArray.ones(
              [2],
              DType.float32,
              device: secondDevice,
            );
            try {
              expect(() => onFirst + onSecond, _throwsShapeOrArgError);
            } finally {
              onFirst.dispose();
              onSecond.dispose();
            }
          } finally {
            firstDevice.dispose();
            secondDevice.dispose();
          }
        },
      );
    });

    group('F7: Array Creation, Copy, Views & astype Boundaries', () {
      test(
        'F7.B1: negative dimensions in shape throw ArgumentError or RangeError',
        () {
          expect(
            () => GpuArray.zeros([2, -1], DType.float32),
            throwsA(anyOf(isA<ArgumentError>(), isA<RangeError>())),
          );
          expect(
            () => GpuArray.empty([-4], DType.float32),
            throwsA(anyOf(isA<ArgumentError>(), isA<RangeError>())),
          );
        },
      );

      test('F7.B2: fromList with mismatched values length vs shape throws', () {
        expect(
          () =>
              GpuArray.fromList(<double>[1.0, 2.0, 3.0], [2, 2], DType.float32),
          _throwsShapeOrArgError,
        );
      });

      test(
        'F7.B3: reshape with incompatible element count or multiple -1 throws',
        () {
          ResourceScope.scope(() {
            final tensor = GpuArray.zeros([2, 3], DType.float32);
            expect(() => tensor.reshape([4, 2]), _throwsShapeOrArgError);
            expect(() => tensor.reshape([-1, -1]), _throwsShapeOrArgError);
          });
        },
      );

      test(
        'F7.B4: invalid transpose axes or squeeze on non-unit axis throws',
        () {
          ResourceScope.scope(() {
            final tensor = GpuArray.zeros([2, 3], DType.float32);
            expect(() => tensor.transpose([0]), _throwsShapeOrArgError);
            expect(() => tensor.transpose([0, 0]), _throwsShapeOrArgError);
            expect(() => tensor.transpose([0, 5]), _throwsShapeOrArgError);
            expect(() => tensor.squeeze(axis: 0), _throwsShapeOrArgError);
          });
        },
      );

      test(
        'F7.B5: copy and astype reject mismatched, broadcasted, or disposed out:',
        () {
          ResourceScope.scope(() {
            final src = GpuArray.ones([2, 2], DType.float32);
            final wrongShapeOut = GpuArray.zeros([4], DType.float32);
            expect(() => src.copy(out: wrongShapeOut), _throwsShapeOrArgError);
            expect(
              () => src.astype(DType.float32, out: wrongShapeOut),
              _throwsShapeOrArgError,
            );

            final broadcastOut = broadcastTo(
              GpuArray.ones([1, 2], DType.float32),
              [2, 2],
            );
            expect(() => src.copy(out: broadcastOut), _throwsShapeOrArgError);

            final disposedOut = GpuArray.zeros([2, 2], DType.float32)
              ..dispose();
            expect(() => src.copy(out: disposedOut), throwsStateError);
          });
        },
      );
    });

    group('F8: Elementwise Binary, Unary & Comparison Ufuncs Boundaries', () {
      test('F8.B1: non-broadcastable shapes in binary ops throw error', () {
        ResourceScope.scope(() {
          final left = GpuArray.ones([2, 3], DType.float32);
          final right = GpuArray.ones([4, 2], DType.float32);
          expect(() => left + right, _throwsShapeOrArgError);
          expect(() => left.equal(right), _throwsShapeOrArgError);
        });
      });

      test('F8.B2: binary and unary ops with wrong out: shape throw error', () {
        ResourceScope.scope(() {
          final input = GpuArray.ones([2, 2], DType.float32);
          final badOut = GpuArray.zeros([3, 3], DType.float32);
          expect(() => input.add(1.0, out: badOut), _throwsShapeOrArgError);
          expect(() => input.exp(out: badOut), _throwsShapeOrArgError);
        });
      });

      test(
        'F8.B3: binary and unary ops on disposed input or disposed out: throw StateError',
        () {
          ResourceScope.scope(() {
            final live = GpuArray.ones([2], DType.float32);
            final disposed = GpuArray.ones([2], DType.float32)..dispose();
            expect(() => disposed + live, throwsStateError);
            expect(() => disposed.sin(), throwsStateError);
            expect(() => live.sin(out: disposed), throwsStateError);
          });
        },
      );

      test('F8.B4: read-only broadcast view as out: throws error', () {
        ResourceScope.scope(() {
          final input = GpuArray.ones([2, 3], DType.float32);
          final bcastOut = broadcastTo(GpuArray.zeros([1, 3], DType.float32), [
            2,
            3,
          ]);
          expect(() => input.add(1.0, out: bcastOut), _throwsShapeOrArgError);
          expect(() => input.negate(out: bcastOut), _throwsShapeOrArgError);
        });
      });

      test(
        'F8.B5: in-place aliased out: (a.add(b, out: a)) succeeds cleanly',
        () {
          ResourceScope.scope(() {
            final target = GpuArray.fromList(
              <double>[1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float32,
            );
            final delta = GpuArray.fromList(
              <double>[10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float32,
            );
            final returned = target.add(delta, out: target);
            expect(identical(returned, target), isTrue);
            expect(
              target.toList().map((e) => (e as num).toDouble()).toList(),
              equals(<double>[11.0, 22.0, 33.0, 44.0]),
            );
          });
        },
      );
    });

    group('F9: Reductions Boundaries', () {
      test(
        'F9.B1: out-of-bounds reduction axis throws RangeError or ArgumentError',
        () {
          ResourceScope.scope(() {
            final matrix = GpuArray.ones([2, 3], DType.float32);
            expect(
              () => matrix.sum(axis: 2),
              throwsA(anyOf(isA<RangeError>(), isA<ArgumentError>())),
            );
            expect(
              () => matrix.mean(axis: -3),
              throwsA(anyOf(isA<RangeError>(), isA<ArgumentError>())),
            );
          });
        },
      );

      test('F9.B2: min and max on empty 0-element array throw error', () {
        ResourceScope.scope(() {
          final empty = GpuArray.zeros([0], DType.float32);
          expect(
            () => empty.min(),
            throwsA(anyOf(isA<StateError>(), isA<ArgumentError>())),
          );
          expect(
            () => empty.max(),
            throwsA(anyOf(isA<StateError>(), isA<ArgumentError>())),
          );
        });
      });

      test(
        'F9.B3: reductions with mismatched out: shape throw ArgumentError',
        () {
          ResourceScope.scope(() {
            final matrix = GpuArray.ones([2, 3], DType.float32);
            final wrongOut = GpuArray.zeros([3], DType.float32);
            expect(
              () => matrix.sum(axis: 1, out: wrongOut),
              _throwsShapeOrArgError,
            );
          });
        },
      );

      test(
        'F9.B4: reductions on disposed array or disposed out: throw StateError',
        () {
          ResourceScope.scope(() {
            final matrix = GpuArray.ones([2, 2], DType.float32);
            final disposedOut = GpuArray.zeros([], DType.float32)..dispose();
            expect(() => matrix.sum(out: disposedOut), throwsStateError);
            matrix.dispose();
            expect(() => matrix.sum(), throwsStateError);
          });
        },
      );

      test(
        'F9.B5: sum and prod on empty 0-element array return identity values',
        () {
          ResourceScope.scope(() {
            final empty = GpuArray.zeros([0], DType.float32);
            expect((empty.sum().scalar as num).toDouble(), closeTo(0.0, 1e-6));
            expect((empty.prod().scalar as num).toDouble(), closeTo(1.0, 1e-6));
          });
        },
      );
    });

    group('F10: Matrix Multiplication & Dot Products Boundaries', () {
      test('F10.B1: matmul on 0-D scalar array throws error', () {
        ResourceScope.scope(() {
          final scalarA = GpuArray.fromList(<double>[2.0], [], DType.float32);
          final scalarB = GpuArray.fromList(<double>[3.0], [], DType.float32);
          expect(() => scalarA.matmul(scalarB), _throwsShapeOrArgError);
        });
      });

      test(
        'F10.B2: 2D matmul with mismatched inner dimensions throws error',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.ones([2, 3], DType.float32);
            final right = GpuArray.ones([4, 2], DType.float32);
            expect(() => left.matmul(right), _throwsShapeOrArgError);
          });
        },
      );

      test(
        'F10.B3: 3D batched matmul with non-broadcastable batch dims throws',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.ones([2, 3, 4], DType.float32);
            final right = GpuArray.ones([3, 4, 2], DType.float32);
            expect(() => left.matmul(right), _throwsShapeOrArgError);
          });
        },
      );

      test('F10.B4: 1D dot and vdot with unequal vector lengths throw', () {
        ResourceScope.scope(() {
          final v3 = GpuArray.ones([3], DType.float32);
          final v4 = GpuArray.ones([4], DType.float32);
          expect(() => v3.dot(v4), _throwsShapeOrArgError);
          expect(() => gpu_linalg.vdot(v3, v4), _throwsShapeOrArgError);
        });
      });

      test(
        'F10.B5: matmul rejects wrong-shaped, broadcasted, or disposed out:',
        () {
          ResourceScope.scope(() {
            final left = GpuArray.ones([2, 2], DType.float32);
            final right = GpuArray.ones([2, 2], DType.float32);
            final wrongOut = GpuArray.zeros([3, 3], DType.float32);
            expect(
              () => left.matmul(right, out: wrongOut),
              _throwsShapeOrArgError,
            );
            final disposedOut = GpuArray.zeros([2, 2], DType.float32)
              ..dispose();
            expect(
              () => left.matmul(right, out: disposedOut),
              throwsStateError,
            );
          });
        },
      );

      test('F10.B6: R4 statistical reduction and ufunc boundary checks', () {
        ResourceScope.scope(() {
          final vec2 = GpuArray<Float32>.fromList(
            [1.0, 2.0],
            [2],
            DType.float32,
          );
          expect((variance(vec2, ddof: 2).scalar as num).isNaN, isTrue);
          expect((std(vec2, ddof: 2).scalar as num).isNaN, isTrue);
          expect(() => variance(vec2, axis: 3), throwsA(isA<RangeError>()));
          expect(() => vec2.clip(null, null), throwsArgumentError);
          expect(() => isClose(vec2, 'invalid'), throwsArgumentError);

          final empty = GpuArray<Float32>.zeros([0], DType.float32);
          expect(() => ptp(empty), throwsStateError);
          expect(() => nanmin(empty), throwsStateError);
          expect(() => nanmax(empty), throwsStateError);
        });
      });
    });
  });
}
