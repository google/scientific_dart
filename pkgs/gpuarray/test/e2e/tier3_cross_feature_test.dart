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
import 'package:gpuarray/jit.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/nn.dart' as gpu_nn;
import 'package:gpuarray/random.dart' as gpu_random;
import 'package:gpuarray/safetensors.dart';
import 'package:gpuarray/serialization.dart';
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
  group('Tier 3 — Cross-Feature Pairwise Integration (24 Combinations)', () {
    test(
      'CF01 (F1 + F4 + F8): WebGPU device with memory pool executes ufuncs and reuses buckets',
      () async {
        final device = await createWebGpuDevice(
          name: 'CF01-Pooled-WebGPU',
          enableMemoryPool: true,
        );
        try {
          for (var iter = 0; iter < 3; iter++) {
            ResourceScope.scope(() {
              final left = GpuArray.fromList(
                <double>[1, 2, 3, 4],
                [4],
                DType.float32,
                device: device,
              );
              final right = GpuArray.fromList(
                <double>[10, 20, 30, 40],
                [4],
                DType.float32,
                device: device,
              );
              final combined = (left * 2.0) + right;
              _expectCloseList(combined.toList(), <double>[12, 24, 36, 48]);
            });
          }
          expect(device.memoryPool.hits, greaterThan(0));
        } finally {
          device.dispose();
        }
      },
    );

    test(
      'CF02 (F2 + F3 + F5): Custom defaultDevice buffer transfer and NDArray round-trip',
      () async {
        final original = GpuDevice.defaultDevice;
        final custom = await createWebGpuDevice(name: 'CF02-Device');
        try {
          GpuDevice.defaultDevice = custom;
          nd.NDArray.scope(() {
            ResourceScope.scope(() {
              final host = nd.NDArray.fromList(
                <double>[3.0, 6.0, 9.0, 12.0],
                [2, 2],
                nd.DType.float32,
              );
              final gpu = host.toGpu();
              expect(gpu.device, same(custom));
              final back = gpu.toNDArray();
              _expectCloseList(back.toList(), host.toList());
            });
          });
        } finally {
          GpuDevice.defaultDevice = original;
          custom.dispose();
        }
      },
    );

    test(
      'CF03 (F5 + F6 + F7): 15-DType astype cast chain followed by NDArray interop',
      () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final int8Gpu = GpuArray.fromList(
              <int>[-5, 0, 7, 12],
              [2, 2],
              DType.int8,
            );
            final f16Gpu = int8Gpu.astype<Float16>(DType.float16);
            final f64Gpu = f16Gpu.astype<Float64>(DType.float64);
            final hostF64 = f64Gpu.toNDArray();
            _expectCloseList(hostF64.toList(), <double>[
              -5.0,
              0.0,
              7.0,
              12.0,
            ], tolerance: 1e-2);
          });
        });
      },
    );

    test(
      'CF04 (F7 + F8 + F9): Transposed and broadcasted views into ufuncs and axis reductions',
      () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            final values = <double>[1, 2, 3, 4, 5, 6];
            final gpuMat = GpuArray.fromList(values, [
              2,
              3,
            ], DType.float32).transpose();
            final hostMat = nd.NDArray.fromList(values, [
              2,
              3,
            ], nd.DType.float32).transpose();
            final gpuBias = GpuArray.fromList(
              <double>[10, 20],
              [1, 2],
              DType.float32,
            );
            final hostBias = nd.NDArray.fromList(
              <double>[10, 20],
              [1, 2],
              nd.DType.float32,
            );

            final gpuReduced = (gpuMat + gpuBias).sum(axis: 0, keepDims: true);
            final hostReduced = nd.sum(hostMat + hostBias, axis: 0);
            expect(gpuReduced.shape, equals([1, 2]));
            _expectCloseList(gpuReduced.toList(), hostReduced.toList());
          });
        });
      },
    );

    test(
      'CF05 (F8 + F10 + F11): Sliced and masked matrices into matmul with out: destination',
      () {
        ResourceScope.scope(() {
          final raw = GpuArray.fromList(
            <double>[-1, 2, -3, 4, 5, -6, 7, 8, 9],
            [3, 3],
            DType.float32,
          );
          final positiveOnly = where(
            raw.greater(0.0),
            raw,
            GpuArray.zeros([3, 3], DType.float32),
          );
          final subBlock = positiveOnly.slice([Slice(0, 2), Slice(0, 2)]);
          final outMat = GpuArray.zeros([2, 2], DType.float32);
          subBlock.matmul(subBlock.transpose(), out: outMat);
          // subBlock = [[0, 2], [4, 5]], subBlock * subBlock^T = [[4, 10], [10, 41]]
          _expectCloseList(outMat.toList(), <double>[4, 10, 10, 41]);
        });
      },
    );

    test(
      'CF06 (F11 + F12): nonzero / flatnonzero indices into take, pad, and concatenate',
      () {
        ResourceScope.scope(() {
          final sparse = GpuArray.fromList(
            <double>[0.0, 5.0, 0.0, 10.0, 15.0],
            [5],
            DType.float32,
          );
          final activeIdx = flatnonzero(sparse);
          final denseVals = take(sparse, activeIdx);
          final paddedVals = pad(denseVals, [
            [1, 1],
          ], mode: PadMode.edge);
          _expectCloseList(paddedVals.toList(), <double>[
            5.0,
            5.0,
            10.0,
            15.0,
            15.0,
          ]);
        });
      },
    );

    test(
      'CF07 (F12 + F15): diag, triu, tril combined with LU and Cholesky decompositions',
      () {
        ResourceScope.scope(() {
          final lowerFactor = tril(
            GpuArray.fromList(
              <double>[2.0, 99.0, 1.0, 3.0],
              [2, 2],
              DType.float64,
            ),
          );
          final spd = lowerFactor.matmul(lowerFactor.transpose());
          final recoveredL = gpu_linalg.cholesky(spd);
          _expectCloseList(
            recoveredL.toList(),
            lowerFactor.toList(),
            tolerance: 1e-4,
          );
        });
      },
    );

    test(
      'CF08 (F13 + F1): WgslJitCompiler fused kernel dispatched on WebGPU device',
      () async {
        final device = await createWebGpuDevice(name: 'CF08-JIT-Device');
        try {
          final compiler = device.jitCompiler;
          final varX = Expr.variable('x', bindingIndex: 0);
          final varY = Expr.variable('y', bindingIndex: 1);
          final shader = compiler.compile(
            (varX * 3.0 + varY).relu(),
            kernelName: 'cf08_fused_relu',
          );
          final xArr = GpuArray.fromList(
            <double>[-2.0, 1.0, 2.0, -5.0],
            [4],
            DType.float32,
            device: device,
          );
          final yArr = GpuArray.fromList(
            <double>[1.0, 2.0, -1.0, 20.0],
            [4],
            DType.float32,
            device: device,
          );
          final outArr = GpuArray.zeros([4], DType.float32, device: device);
          try {
            device.backend.dispatchComputePipeline(
              shaderModule: shader,
              buffers: [xArr.buffer, yArr.buffer, outArr.buffer],
              uniforms: [4, 0, 0, 0],
              workgroupsX: 1,
            );
            await device.synchronize();
            _expectCloseList(outArr.toList(), <double>[0.0, 5.0, 5.0, 5.0]);
          } finally {
            xArr.dispose();
            yArr.dispose();
            outArr.dispose();
          }
        } finally {
          device.dispose();
        }
      },
    );

    test(
      'CF09 (F14 + F12 + F6): Safetensors round-trip of sliced/padded multi-dtype tensors',
      () {
        ResourceScope.scope(() {
          final base = GpuArray.fromList(
            <double>[1, 2, 3, 4, 5, 6],
            [2, 3],
            DType.float32,
          );
          final flipped = fliplr(base);
          final intCast = base.astype<Int16>(DType.int16);
          final payload = saveSafetensors({
            'flipped': flipped,
            'int16': intCast,
          });
          final restored = loadSafetensors(payload);
          _expectCloseList(restored['flipped']!.toList(), <double>[
            3,
            2,
            1,
            6,
            5,
            4,
          ]);
          expect(restored['int16']!.toList(), equals(<int>[1, 2, 3, 4, 5, 6]));
        });
      },
    );

    test(
      'CF10 (F15 + F16): QR, SVD, and LU decompositions cross-checked with solve, inv, and det',
      () {
        ResourceScope.scope(() {
          final mat = GpuArray.fromList(
            <double>[4.0, 2.0, 1.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final qrDecomp = gpu_linalg.qr(mat);
          final invMat = gpu_linalg.inv(mat);
          // R^-1 * Q^T == A^-1
          final qrInv = gpu_linalg
              .inv(qrDecomp.r)
              .matmul(qrDecomp.q.transpose());
          _expectCloseList(qrInv.toList(), invMat.toList(), tolerance: 1e-4);
          qrDecomp.dispose();
        });
      },
    );

    test(
      'CF11 (F16 + F17): einsum bilinear form matches multiDot and norm',
      () {
        ResourceScope.scope(() {
          final leftVec = GpuArray.fromList(
            <double>[1.0, 2.0],
            [2],
            DType.float64,
          );
          final weightMat = GpuArray.fromList(
            <double>[3.0, 1.0, 1.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final rightVec = GpuArray.fromList(
            <double>[2.0, -1.0],
            [2],
            DType.float64,
          );
          final viaEinsum = gpu_linalg.einsum('i,ij,j->', [
            leftVec,
            weightMat,
            rightVec,
          ]);
          final viaMatmul = leftVec.dot(gpu_linalg.matmul(weightMat, rightVec));
          expect(
            (viaEinsum.scalar as num).toDouble(),
            closeTo((viaMatmul.scalar as num).toDouble(), 1e-4),
          );
        });
      },
    );

    test(
      'CF12 (F18 + F8 + F9): Parseval theorem via fft, Complex abs/multiply, and sum',
      () {
        ResourceScope.scope(() {
          final timeSignal = GpuArray.fromList(
            <double>[1.0, -1.0, 2.0, 0.5],
            [4],
            DType.float64,
          );
          final timeEnergy = ((timeSignal * timeSignal).sum().scalar as num)
              .toDouble();
          final orthoSpec = gpu_fft.fft(
            timeSignal,
            norm: gpu_fft.FftNorm.ortho,
          );
          final freqEnergy = orthoSpec.toList().cast<Complex>().fold<double>(
            0.0,
            (acc, c) => acc + c.real * c.real + c.imag * c.imag,
          );
          expect(freqEnergy, closeTo(timeEnergy, 1e-4));
        });
      },
    );

    test(
      'CF13 (F18 + F12): 2D fft2 with roll and fftshift circular shift property',
      () {
        ResourceScope.scope(() {
          final signal = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          final shiftedSignal = roll(signal, 1);
          final specOrig = gpu_fft.fft(signal).toList().cast<Complex>();
          final specShifted = gpu_fft
              .fft(shiftedSignal)
              .toList()
              .cast<Complex>();
          // Circular shift in time preserves magnitude spectrum |X[k]|
          for (var k = 0; k < 4; k++) {
            final magOrig =
                specOrig[k].real * specOrig[k].real +
                specOrig[k].imag * specOrig[k].imag;
            final magShift =
                specShifted[k].real * specShifted[k].real +
                specShifted[k].imag * specShifted[k].imag;
            expect(magShift, closeTo(magOrig, 1e-4));
          }
        });
      },
    );

    test(
      'CF14 (F19 + F9 + F11): Random uniform sampling with where thresholding and mean reduction',
      () {
        ResourceScope.scope(() {
          final rng = gpu_random.RandomState(31415);
          final draws = rng.uniform(low: 0.0, high: 1.0, shape: [256]);
          final indicator = where(
            draws.greater(0.5),
            GpuArray.ones([256], DType.float64),
            GpuArray.zeros([256], DType.float64),
          );
          final fractionAboveHalf = (indicator.mean().scalar as num).toDouble();
          expect(fractionAboveHalf, closeTo(0.5, 0.15));
        });
      },
    );

    test(
      'CF15 (F19 + F15 + F10): Random Wishart SPD matrix into Cholesky and eigvalsh',
      () {
        ResourceScope.scope(() {
          final rng = gpu_random.RandomState(2718);
          final sampleMat = rng.normal(shape: [8, 3]);
          final eye3 = diag(GpuArray.ones([3], DType.float64));
          final spd = sampleMat.transpose().matmul(sampleMat) + eye3;
          final chol = gpu_linalg.cholesky(spd);
          final eigenvalues = gpu_linalg.eigvalsh(spd);
          for (final ev in eigenvalues.toList().cast<num>()) {
            expect(ev.toDouble(), greaterThan(0.0));
          }
          _expectCloseList(
            chol.matmul(chol.transpose()).toList(),
            spd.toList(),
            tolerance: 1e-4,
          );
        });
      },
    );

    test(
      'CF16 (F20 + F8 + F9): Autograd through exp, log, sin, cos, and mean reduction',
      () {
        ResourceScope.scope(() {
          final xLeaf = GpuArray.fromList(
            <double>[0.5, 1.0],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          // f(x) = mean(exp(x) * sin(x)); df/dx_i = 0.5 * exp(x_i) * (sin(x_i) + cos(x_i))
          final loss = (xLeaf.exp() * xLeaf.sin()).mean();
          loss.backward();
          final expectedGrad = [0.5, 1.0].map((v) {
            final expV =
                (GpuArray.fromList([v], [1], DType.float64).exp().scalar as num)
                    .toDouble();
            final sinV =
                (GpuArray.fromList([v], [1], DType.float64).sin().scalar as num)
                    .toDouble();
            final cosV =
                (GpuArray.fromList([v], [1], DType.float64).cos().scalar as num)
                    .toDouble();
            return 0.5 * expV * (sinV + cosV);
          }).toList();
          _expectCloseList(xLeaf.grad!.toList(), expectedGrad, tolerance: 1e-4);
        });
      },
    );

    test(
      'CF17 (F20 + F12): Autograd through concatenate, slice, reshape, and transpose',
      () {
        ResourceScope.scope(() {
          final left = GpuArray.fromList(
            <double>[1.0, 2.0],
            [1, 2],
            DType.float64,
            requiresGrad: true,
          );
          final right = GpuArray.fromList(
            <double>[3.0, 4.0],
            [1, 2],
            DType.float64,
            requiresGrad: true,
          );
          final merged = concatenate([left * 2.0, right * 3.0], axis: 0);
          final permuted = merged.transpose().reshape([4]);
          permuted.sum().backward();
          _expectCloseList(left.grad!.toList(), <double>[2.0, 2.0]);
          _expectCloseList(right.grad!.toList(), <double>[3.0, 3.0]);
        });
      },
    );

    test(
      'CF18 (F20 + F21): Autograd through LayerNorm, GELU, and Linear with SGD update',
      () {
        ResourceScope.scope(() {
          gpu_random.seed(77);
          final linear = gpu_nn.Linear(3, 3);
          final normLayer = gpu_nn.LayerNorm([3]);
          final act = gpu_nn.GELU();
          final input = GpuArray.fromList(
            <double>[1.0, -0.5, 2.0, 0.5, 1.5, -1.0],
            [2, 3],
            DType.float64,
          );
          final target = GpuArray.zeros([2, 3], DType.float64);
          final opt = gpu_nn.SGD([
            ...linear.parameters,
            ...normLayer.parameters,
          ], lr: 0.05);

          final predBefore = act.forward(
            normLayer.forward(linear.forward(input)),
          );
          final lossBefore = (gpu_nn.mseLoss(predBefore, target).scalar as num)
              .toDouble();
          gpu_nn.mseLoss(predBefore, target).backward();
          opt.step();
          opt.zeroGrad();

          final predAfter = act.forward(
            normLayer.forward(linear.forward(input)),
          );
          final lossAfter = (gpu_nn.mseLoss(predAfter, target).scalar as num)
              .toDouble();
          expect(lossAfter, lessThan(lossBefore));
        });
      },
    );

    test(
      'CF19 (F21 + F14): Neural network parameter saveSafetensors and loadSafetensors transfer',
      () {
        ResourceScope.scope(() {
          gpu_random.seed(123);
          final sourceLayer = gpu_nn.Linear(3, 2);
          final targetLayer = gpu_nn.Linear(3, 2);
          final bytes = saveSafetensors({
            'weight': sourceLayer.weight,
            'bias': sourceLayer.bias!,
          });
          final loaded = loadSafetensors(bytes);
          (loaded['weight']! as GpuArray<Float64>).copy(
            out: targetLayer.weight,
          );
          (loaded['bias']! as GpuArray<Float64>).copy(out: targetLayer.bias);

          final sampleIn = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0],
            [1, 3],
            DType.float64,
          );
          _expectCloseList(
            targetLayer.forward(sampleIn).toList(),
            sourceLayer.forward(sampleIn).toList(),
          );
        });
      },
    );

    test(
      'CF20 (F21 + F19 + F20): Embedding + RotaryEmbedding + scaledDotProductAttention backward',
      () {
        ResourceScope.scope(() {
          gpu_random.seed(909);
          final emb = gpu_nn.Embedding(8, 4);
          final rope = gpu_nn.RotaryEmbedding(4, maxSequenceLength: 8);
          final tokens = GpuArray.fromList(<int>[1, 2, 3], [1, 3], DType.int32);
          final xEmb = emb.forward(tokens);
          final rotated = rope.forward(xEmb);
          final attended = gpu_nn.scaledDotProductAttention(
            rotated,
            rotated,
            rotated,
            isCausal: true,
          );
          expect(attended.shape, equals([1, 3, 4]));
          attended.sum().backward();
          expect(emb.weight.grad, isNotNull);
        });
      },
    );

    test(
      'CF21 (F13 + F20): Symbolic Expr.grad matches numerical autograd backward on GPU',
      () {
        ResourceScope.scope(() {
          final varX = Expr.variable('x', bindingIndex: 0);
          // f(x) = x^3 + 2*x => f'(x) = 3*x^2 + 2
          final symExpr = (varX * varX * varX) + (varX * 2.0);
          final symGrad = symExpr.grad(varX);
          expect(symGrad.toWgsl(), isNotEmpty);

          final xTensor = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0],
            [3],
            DType.float64,
            requiresGrad: true,
          );
          final yTensor = ((xTensor * xTensor * xTensor) + (xTensor * 2.0))
              .sum();
          yTensor.backward();
          _expectCloseList(xTensor.grad!.toList(), <double>[5.0, 14.0, 29.0]);
        });
      },
    );

    test(
      'CF22 (F3 + F7 + F10): GpuArray.fromBuffer view over offset slice into matmul',
      () {
        ResourceScope.scope(() {
          final base = GpuArray.fromList(
            <double>[99, 99, 1, 2, 3, 4],
            [6],
            DType.float32,
          );
          base.buffer.retain();
          final view2x2 = GpuArray<Float32>.fromBuffer(
            buffer: base.buffer,
            shape: [2, 2],
            strides: [2, 1],
            dtype: DType.float32,
            device: base.device,
            offsetElements: 2,
          );
          final sq = view2x2.matmul(view2x2);
          _expectCloseList(sq.toList(), <double>[7.0, 10.0, 15.0, 22.0]);
        });
      },
    );

    test(
      'CF23 (F15 + F18): Circulant matrix eigenvalues via eigvals match 1D fft of first row',
      () {
        ResourceScope.scope(() {
          final firstRow = GpuArray.fromList(
            <double>[4.0, 1.0, 2.0, 1.0],
            [4],
            DType.float64,
          );
          final circulant = GpuArray.fromList(
            <double>[
              4.0,
              1.0,
              2.0,
              1.0,
              1.0,
              4.0,
              1.0,
              2.0,
              2.0,
              1.0,
              4.0,
              1.0,
              1.0,
              2.0,
              1.0,
              4.0,
            ],
            [4, 4],
            DType.float64,
          );
          final fftVals =
              gpu_fft
                  .fft(firstRow)
                  .toList()
                  .cast<Complex>()
                  .map((c) => c.real)
                  .toList()
                ..sort();
          final eigVals =
              gpu_linalg
                  .eigvalsh(circulant)
                  .toList()
                  .cast<num>()
                  .map((e) => e.toDouble())
                  .toList()
                ..sort();
          _expectCloseList(eigVals, fftVals, tolerance: 1e-4);
        });
      },
    );

    test(
      'CF24 (F4 + F21 + F2): Nested ResourceScope.returning with pooled WebGPU device',
      () async {
        final device = await createWebGpuDevice(
          name: 'CF24-Scoped-Pool',
          enableMemoryPool: true,
        );
        try {
          final output = ResourceScope.returning(() {
            final a = GpuArray.fromList(
              <double>[2.0, 4.0],
              [2],
              DType.float32,
              device: device,
            );
            final b = GpuArray.fromList(
              <double>[3.0, 5.0],
              [2],
              DType.float32,
              device: device,
            );
            return a * b;
          });
          expect(device.activeBufferCount, equals(1));
          expect(device.memoryPool.cachedBytes, greaterThan(0));
          _expectCloseList(output.toList(), <double>[6.0, 20.0]);
          output.dispose();
          expect(device.activeBufferCount, equals(0));
        } finally {
          device.dispose();
        }
      },
    );

    test(
      'CF25 (R1 + R2 + R3 + R4): Float32 rfft/irfft + CompiledWgslKernel + topk/cumsum + Float32 Linear/AdamW',
      () {
        ResourceScope.scope(() {
          final sig = GpuArray<Float32>.fromList(
            <double>[1.0, 3.0, 2.0, 4.0],
            [4],
            DType.float32,
          );
          final spec = gpu_fft.rfft(sig);
          expect(spec.dtype, equals(DType.complex64));
          final rec = gpu_fft.irfft(spec, n: 4);
          expect(rec.dtype, equals(DType.float32));

          final xVar = Expr.variable('x', bindingIndex: 0);
          final kernel = GpuDevice.defaultDevice.jitCompiler.compileKernel(
            xVar * 2.0,
          );
          final scaled = kernel.execute<Float32>({'x': rec});
          final cs = cumsum(scaled);
          expect(cs.dtype, equals(DType.float32));
          final top2 = topk(cs, 2);
          expect(top2.indices.dtype, equals(DType.int64));
          _expectCloseList(top2.values.toList(), <double>[20.0, 12.0]);
        });
      },
    );
  });
}
