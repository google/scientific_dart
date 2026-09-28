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

import 'package:gpuarray/fft.dart' as gpu_fft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/nn.dart' as nn;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  late GpuDevice device;

  setUpAll(() {
    device = GpuDevice.cpu();
    device.detachFromScope();
  });

  tearDownAll(() {
    device.dispose();
  });

  void expectZeroLeaks(void Function() body) {
    final initialBuffers = device.activeBufferCount;
    final initialBytes = device.allocatedMemoryBytes;
    body();
    expect(
      device.activeBufferCount,
      equals(initialBuffers),
      reason: 'GpuDevice activeBufferCount leaked',
    );
    expect(
      device.allocatedMemoryBytes,
      equals(initialBytes),
      reason: 'GpuDevice allocatedMemoryBytes leaked',
    );
  }

  group('Lifetime & ResourceScope zero-leak invariants', () {
    test(
      'Core creation, views, arithmetic, and reductions free all buffers on scope exit',
      () {
        expectZeroLeaks(() {
          ResourceScope.scope(() {
            final a = GpuArray<Float64>.fromList(
              List<double>.generate(24, (i) => i * 0.5),
              [4, 6],
              DType.float64,
              device: device,
            );
            final b = GpuArray<Float64>.ones(
              [4, 6],
              DType.float64,
              device: device,
            );
            final c = (a + b) * a.sin() - b.cos();
            final s = c.sum(axis: 1);
            expect(s.shape, equals([4]));
          });
        });
      },
    );

    test(
      'ResourceScope.returning transfers kept GpuArray to outer scope and frees intermediates',
      () {
        expectZeroLeaks(() {
          ResourceScope.scope(() {
            final initialInsideOuter = device.activeBufferCount;
            final kept = ResourceScope.returning<GpuArray<DTypeTag>>(() {
              final a = GpuArray<Float32>.fromList(
                [1.0, 2.0, 3.0],
                [3],
                DType.float32,
                device: device,
              );
              final b = GpuArray<Float32>.fromList(
                [4.0, 5.0, 6.0],
                [3],
                DType.float32,
                device: device,
              );
              return a + b;
            });
            expect(kept.isDisposed, isFalse);
            expect(device.activeBufferCount, equals(initialInsideOuter + 1));
            expect(kept.toList(), equals([5.0, 7.0, 9.0]));
          });
        });
      },
    );

    test(
      'Exception unwinding inside ResourceScope.scope frees all allocated GPU and host buffers',
      () {
        expectZeroLeaks(() {
          expect(
            () => ResourceScope.scope(() {
              final a = GpuArray<Float32>.ones(
                [8, 8],
                DType.float32,
                device: device,
              );
              final b = GpuArray<Float32>.zeros(
                [3, 3],
                DType.float32,
                device: device,
              );
              // Triggers ArgumentError after allocating a and b
              a.add(b);
            }),
            throwsArgumentError,
          );
        });
      },
    );

    test(
      'Linear algebra (linalg) decompositions and solvers leak zero buffers',
      () {
        expectZeroLeaks(() {
          ResourceScope.scope(() {
            final a = GpuArray<Float64>.fromList(
              [
                [4.0, 1.0, 1.0],
                [1.0, 3.0, 0.5],
                [1.0, 0.5, 2.0],
              ],
              [3, 3],
              DType.float64,
              device: device,
            );
            final b = GpuArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
              device: device,
            );
            final l = gpu_linalg.cholesky(a);
            final qrRes = gpu_linalg.qr(a);
            final svdRes = gpu_linalg.svd(a);
            final eigRes = gpu_linalg.eigh(a);
            final x = gpu_linalg.solve(a, b);
            final invA = gpu_linalg.inv(a);
            final detA = gpu_linalg.det(a);
            final ein = gpu_linalg.einsum('ij,jk->ik', [a, invA]);
            expect(l.shape, equals([3, 3]));
            expect(qrRes.q.shape, equals([3, 3]));
            expect(svdRes.s.shape, equals([3]));
            expect(eigRes.eigenvalues.shape, equals([3]));
            expect(x.shape, equals([3]));
            expect(detA.ndim, equals(0));
            expect(ein.shape, equals([3, 3]));
          });
        });
      },
    );

    test(
      'FFT (1D, 2D, real, Hermitian, non-power-of-2) leaks zero buffers',
      () {
        expectZeroLeaks(() {
          ResourceScope.scope(() {
            final sig = GpuArray<Float64>.fromList(
              List<double>.generate(13, (i) => i / 12.0),
              [13],
              DType.float64,
              device: device,
            );
            final spec = gpu_fft.rfft(sig);
            final rec = gpu_fft.irfft(spec, n: 13);
            expect(rec.shape, equals([13]));

            final img = GpuArray<Float64>.ones(
              [4, 6],
              DType.float64,
              device: device,
            );
            final imgSpec = gpu_fft.fft2(img);
            final imgRec = gpu_fft.ifft2(imgSpec);
            expect(imgRec.shape, equals([4, 6]));
          });
        });
      },
    );

    test('Random generator distributions leak zero buffers', () {
      expectZeroLeaks(() {
        ResourceScope.scope(() {
          final rng = RandomState(12345);
          final u = rng.uniform(shape: [16], device: device);
          final n = rng.normal(shape: [16], device: device);
          final e = rng.exponential(shape: [16], device: device);
          final p = rng.permutation(10, device: device);
          expect(u.size + n.size + e.size + p.size, equals(58));
        });
      });
    });

    test(
      'Autograd forward/backward, NN training step, and Safetensors round-trip leak zero buffers',
      () {
        expectZeroLeaks(() {
          ResourceScope.scope(() {
            final model = nn.Sequential([
              nn.Linear(4, 8, device: device),
              nn.ReLU(),
              nn.Linear(8, 2, device: device),
            ]);
            final optimizer = nn.Adam(model.parameters, lr: 1e-2);
            final x = GpuArray<Float32>.ones(
              [3, 4],
              DType.float32,
              device: device,
            );
            final target = GpuArray<Float32>.zeros(
              [3, 2],
              DType.float32,
              device: device,
            );

            optimizer.zeroGrad();
            final pred = model.forward(x);
            final loss = nn.mseLoss(pred, target);
            loss.backward();
            optimizer.step();
            optimizer.dispose();

            final state = model.namedParameters();
            final bytes = saveSafetensors(state);
            final loaded = loadSafetensors(bytes, device: device);
            expect(loaded.keys, equals(state.keys));
          });
        });
      },
    );
  });
}
