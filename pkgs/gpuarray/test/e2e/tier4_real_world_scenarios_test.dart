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

import 'package:gpuarray/fft.dart' as gpu_fft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/jit.dart';
import 'package:gpuarray/linalg.dart' as gpu_linalg;
import 'package:gpuarray/nn.dart' as gpu_nn;
import 'package:gpuarray/random.dart' as gpu_random;
import 'package:gpuarray/safetensors.dart';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void _expectCloseList(
  List<Object?> actual,
  List<Object?> expected, {
  double tolerance = 1e-3,
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
  group('Tier 4 — Real-World Scientific & ML Application Scenarios (12 Scenarios)', () {
    test(
      'Scenario 01: Ordinary Least Squares (OLS) via QR, SVD Pseudoinverse & Normal Equations',
      () {
        nd.NDArray.scope(() {
          ResourceScope.scope(() {
            // True model: y = 2.5 * x1 - 1.5 * x2 + 0.5
            final designData = <double>[
              1.0,
              0.0,
              1.0,
              1.0,
              1.0,
              1.0,
              2.0,
              1.0,
              1.0,
              3.0,
              2.0,
              1.0,
              4.0,
              1.0,
              1.0,
              5.0,
              3.0,
              1.0,
            ];
            final targetData = <double>[3.0, 1.5, 4.0, 5.0, 9.0, 8.5];

            final xGpu = GpuArray.fromList(designData, [6, 3], DType.float64);
            final yGpu = GpuArray.fromList(targetData, [6], DType.float64);

            // Method 1: Normal equations solve(X^T X, X^T y)
            final xtx = xGpu.transpose().matmul(xGpu);
            final xty = gpu_linalg.matmul(xGpu.transpose(), yGpu);
            final betaNormal = gpu_linalg.solve(xtx, xty);

            // Method 2: Moore-Penrose pseudoinverse pinv(X) * y
            final betaPinv = gpu_linalg.matmul(gpu_linalg.pinv(xGpu), yGpu);

            // Method 3: QR decomposition R^-1 Q^T y
            final qrDecomp = gpu_linalg.qr(
              xGpu,
              mode: gpu_linalg.QrMode.reduced,
            );
            final betaQr = gpu_linalg.solve(
              qrDecomp.r,
              gpu_linalg.matmul(qrDecomp.q.transpose(), yGpu),
            );
            qrDecomp.dispose();

            final expectedBeta = <double>[2.5, -1.5, 0.5];
            _expectCloseList(
              betaNormal.toList(),
              expectedBeta,
              tolerance: 1e-4,
            );
            _expectCloseList(betaPinv.toList(), expectedBeta, tolerance: 1e-4);
            _expectCloseList(betaQr.toList(), expectedBeta, tolerance: 1e-4);
          });
        });
      },
    );

    test(
      'Scenario 02: Principal Component Analysis (PCA) via SVD and Covariance eigh',
      () {
        ResourceScope.scope(() {
          final rawSamples = <double>[
            2.5,
            2.4,
            0.5,
            0.7,
            2.2,
            2.9,
            1.9,
            2.2,
            3.1,
            3.0,
            2.3,
            2.7,
            2.0,
            1.6,
            1.0,
            1.1,
            1.5,
            1.6,
            1.1,
            0.9,
          ];
          final dataMat = GpuArray.fromList(rawSamples, [10, 2], DType.float64);
          final featureMeans = dataMat.mean(axis: 0, keepDims: true);
          final centered = dataMat - featureMeans;

          // Covariance matrix C = (X_c^T * X_c) / (n - 1)
          final cov = centered.transpose().matmul(centered) / 9.0;
          final eigResult = gpu_linalg.eigh(cov);
          final svdVals = gpu_linalg.svdValues(centered);

          // Singular values squared / (n - 1) must match sorted covariance eigenvalues
          final eigSortedDesc = List<double>.from(
            eigResult.eigenvalues.toList().cast<num>().map((e) => e.toDouble()),
          )..sort((a, b) => b.compareTo(a));
          final svdVar = svdVals
              .toList()
              .cast<num>()
              .map((s) => (s.toDouble() * s.toDouble()) / 9.0)
              .toList();
          _expectCloseList(svdVar, eigSortedDesc, tolerance: 1e-4);
          eigResult.dispose();
        });
      },
    );

    test(
      'Scenario 03: 2D Heat Diffusion PDE Stencil Solver with Conservation Verification',
      () {
        ResourceScope.scope(() {
          // Initialize 4x4 interior grid with a central heat pulse, padded with reflective boundaries
          var temperature = GpuArray.fromList(
            <double>[0, 0, 0, 0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 0, 0, 0],
            [4, 4],
            DType.float32,
          );
          final initialEnergy = (temperature.sum().scalar as num).toDouble();
          final initialPeak = (temperature.max().scalar as num).toDouble();

          const alpha = 0.1;
          for (var stepIndex = 0; stepIndex < 5; stepIndex++) {
            final padded = pad(temperature, [
              [1, 1],
              [1, 1],
            ], mode: PadMode.wrap);
            final center = padded.slice([Slice(1, 5), Slice(1, 5)]);
            final north = padded.slice([Slice(0, 4), Slice(1, 5)]);
            final south = padded.slice([Slice(2, 6), Slice(1, 5)]);
            final west = padded.slice([Slice(1, 5), Slice(0, 4)]);
            final east = padded.slice([Slice(1, 5), Slice(2, 6)]);

            final laplacian = (north + south + west + east) - (center * 4.0);
            temperature = center + (laplacian * alpha);
          }

          final finalEnergy = (temperature.sum().scalar as num).toDouble();
          final finalPeak = (temperature.max().scalar as num).toDouble();
          // Periodic boundary conditions conserve total thermal energy while peak diffuses
          expect(finalEnergy, closeTo(initialEnergy, 1e-3));
          expect(finalPeak, lessThan(initialPeak));
        });
      },
    );

    test(
      'Scenario 04: Spectral Signal Denoising via rfft, rfftfreq Masking & irfft',
      () {
        ResourceScope.scope(() {
          const sampleCount = 16;
          final cleanSignal = <double>[];
          final noisySignal = <double>[];
          for (var i = 0; i < sampleCount; i++) {
            final t = i / sampleCount;
            final lowFreq = math.sin(2 * math.pi * 1.0 * t);
            final highFreqNoise = 0.8 * math.cos(2 * math.pi * 6.0 * t);
            cleanSignal.add(lowFreq);
            noisySignal.add(lowFreq + highFreqNoise);
          }

          final gpuClean = GpuArray.fromList(cleanSignal, [
            sampleCount,
          ], DType.float64);
          final gpuNoisy = GpuArray.fromList(noisySignal, [
            sampleCount,
          ], DType.float64);

          final spectrum = gpu_fft.rfft(gpuNoisy);
          final frequencies = gpu_fft.rfftfreq(
            sampleCount,
            d: 1.0 / sampleCount,
          );
          // Zero out frequencies above 2.5 Hz
          final specList = spectrum.toList().cast<Complex>();
          final freqList = frequencies.toList().cast<num>();
          final filteredSpecList = <Complex>[
            for (var k = 0; k < specList.length; k++)
              freqList[k].abs() <= 2.5 ? specList[k] : Complex(0.0, 0.0),
          ];
          final filteredSpec = GpuArray.fromList(filteredSpecList, [
            specList.length,
          ], DType.complex128);
          final denoised = gpu_fft.irfft(filteredSpec, n: sampleCount);

          _expectCloseList(
            denoised.toList(),
            gpuClean.toList(),
            tolerance: 1e-4,
          );
        });
      },
    );

    test(
      'Scenario 05: 2D Frequency-Domain Image Filtering via fft2, fftshift, ifftshift & ifft2',
      () {
        ResourceScope.scope(() {
          final image = GpuArray.fromList(
            <double>[1, 2, 1, 2, 2, 4, 2, 4, 1, 2, 1, 2, 2, 4, 2, 4],
            [4, 4],
            DType.float64,
          );
          final spectrum2d = gpu_fft.fft2(image);
          final centeredSpec = gpu_fft.fftshift(spectrum2d);
          final uncenteredSpec = gpu_fft.ifftshift(centeredSpec);
          final spatialReconstructed = gpu_fft.ifft2(uncenteredSpec);
          final realPixels = spatialReconstructed
              .toList()
              .cast<Complex>()
              .map((c) => c.real)
              .toList();
          _expectCloseList(realPixels, image.toList(), tolerance: 1e-4);
        });
      },
    );

    test(
      'Scenario 06: Rigid-Body Mechanics & Quantum Pauli Kronecker Algebra (cross, einsum, kron, det)',
      () {
        ResourceScope.scope(() {
          // Angular momentum L = r x p; r . L == 0
          final position = GpuArray.fromList(
            <double>[2.0, -1.0, 3.0],
            [3],
            DType.float64,
          );
          final momentum = GpuArray.fromList(
            <double>[-1.0, 4.0, 2.0],
            [3],
            DType.float64,
          );
          final angularMomentum = gpu_linalg.cross(position, momentum);
          final orthogonality = position.dot(angularMomentum);
          expect((orthogonality.scalar as num).toDouble(), closeTo(0.0, 1e-5));

          // Pauli Z (x) Pauli X Kronecker product trace == 0, det == 1
          final pauliZ = GpuArray.fromList(
            <double>[1, 0, 0, -1],
            [2, 2],
            DType.float64,
          );
          final pauliX = GpuArray.fromList(
            <double>[0, 1, 1, 0],
            [2, 2],
            DType.float64,
          );
          final tensorProd = gpu_linalg.kron(pauliZ, pauliX);
          expect(tensorProd.shape, equals([4, 4]));
          final traceVal = gpu_linalg.einsum('ii->', [tensorProd]);
          expect((traceVal.scalar as num).toDouble(), closeTo(0.0, 1e-5));
          final detVal = gpu_linalg.det(tensorProd);
          expect((detVal.scalar as num).toDouble().abs(), closeTo(1.0, 1e-4));
        });
      },
    );

    test(
      'Scenario 07: End-to-End MLP Regression Training & safetensors Checkpoint Restore',
      () {
        ResourceScope.scope(() {
          gpu_random.seed(2026);
          final layer1 = gpu_nn.Linear(2, 4);
          final layer2 = gpu_nn.Linear(4, 1);
          final model = gpu_nn.Sequential([layer1, gpu_nn.ReLU(), layer2]);
          final optimizer = gpu_nn.Adam(model.parameters, lr: 0.05);

          final inputs = GpuArray.fromList(
            <double>[1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 0.0, 3.0],
            [4, 2],
            DType.float64,
          );
          final targets = GpuArray.fromList(
            <double>[3.0, 3.0, 3.0, 3.0],
            [4, 1],
            DType.float64,
          );

          final initialPred = model.forward(inputs);
          final initialLoss =
              (gpu_nn.mseLoss(initialPred, targets).scalar as num).toDouble();
          for (var epoch = 0; epoch < 12; epoch++) {
            optimizer.zeroGrad();
            final pred = model.forward(inputs);
            final loss = gpu_nn.mseLoss(pred, targets);
            loss.backward();
            optimizer.step();
          }
          final trainedPred = model.forward(inputs);
          final trainedLoss =
              (gpu_nn.mseLoss(trainedPred, targets).scalar as num).toDouble();
          expect(trainedLoss, lessThan(initialLoss));

          // Checkpoint via safetensors and restore into fresh layers
          final checkpoint = saveSafetensors({
            'l1.w': layer1.weight,
            'l1.b': layer1.bias!,
            'l2.w': layer2.weight,
            'l2.b': layer2.bias!,
          });
          final restoredMap = loadSafetensors(checkpoint);
          final restoredL1 = gpu_nn.Linear(2, 4);
          final restoredL2 = gpu_nn.Linear(4, 1);
          (restoredMap['l1.w']! as GpuArray<Float64>).copy(
            out: restoredL1.weight,
          );
          (restoredMap['l1.b']! as GpuArray<Float64>).copy(
            out: restoredL1.bias,
          );
          (restoredMap['l2.w']! as GpuArray<Float64>).copy(
            out: restoredL2.weight,
          );
          (restoredMap['l2.b']! as GpuArray<Float64>).copy(
            out: restoredL2.bias,
          );
          final restoredModel = gpu_nn.Sequential([
            restoredL1,
            gpu_nn.ReLU(),
            restoredL2,
          ]);

          noGrad(() {
            final origPred = model.forward(inputs);
            final restPred = restoredModel.forward(inputs);
            _expectCloseList(
              restPred.toList(),
              origPred.toList(),
              tolerance: 1e-6,
            );
          });
        });
      },
    );

    test(
      'Scenario 08: Spatial Conv2d Feature Extractor + LayerNorm + crossEntropy + AdamW Step',
      () {
        ResourceScope.scope(() {
          gpu_random.seed(808);
          final conv = gpu_nn.Conv2d(1, 2, 2, stride: 1, padding: 0);
          final normLayer = gpu_nn.LayerNorm([8]);
          final head = gpu_nn.Linear(8, 3);
          final optimizer = gpu_nn.AdamW(
            [...conv.parameters, ...normLayer.parameters, ...head.parameters],
            lr: 0.02,
            weightDecay: 0.01,
          );

          final batchImages = GpuArray.fromList(
            List<double>.generate(18, (i) => ((i % 5) - 2) * 0.25),
            [2, 1, 3, 3],
            DType.float64,
          );
          final labels = GpuArray.fromList(<int>[0, 2], [2], DType.int32);

          GpuArray<Float64> forwardPass() {
            final feat = gpu_nn.relu(conv.forward(batchImages));
            final flat = feat.reshape([2, 8]);
            final normed = normLayer.forward(flat);
            return head.forward(normed);
          }

          final lossBefore =
              (gpu_nn.crossEntropy(forwardPass(), labels).scalar as num)
                  .toDouble();
          for (var stepIndex = 0; stepIndex < 5; stepIndex++) {
            optimizer.zeroGrad();
            final logits = forwardPass();
            final loss = gpu_nn.crossEntropy(logits, labels);
            loss.backward();
            optimizer.step();
          }
          final lossAfter =
              (gpu_nn.crossEntropy(forwardPass(), labels).scalar as num)
                  .toDouble();
          expect(lossAfter, lessThan(lossBefore));
        });
      },
    );

    test(
      'Scenario 09: Transformer Encoder-Decoder Sequence Pipeline with RoPE, RMSNorm & SwiGLU',
      () {
        ResourceScope.scope(() {
          gpu_random.seed(909);
          final tokenEmbedding = gpu_nn.Embedding(12, 4);
          final rope = gpu_nn.RotaryEmbedding(4, maxSequenceLength: 8);
          final rmsNorm = gpu_nn.RMSNorm([4]);
          final swiglu = gpu_nn.SwiGLU(4, 8, outFeatures: 4);
          final encoderLayer = gpu_nn.TransformerEncoderLayer(
            4,
            2,
            dimFeedforward: 8,
            normFirst: true,
          );
          final decoderLayer = gpu_nn.TransformerDecoderLayer(
            4,
            2,
            dimFeedforward: 8,
            normFirst: true,
          );
          final lmHead = gpu_nn.Linear(4, 6);

          final srcTokens = GpuArray.fromList(
            <int>[1, 4, 7],
            [1, 3],
            DType.int32,
          );
          final tgtTokens = GpuArray.fromList(<int>[2, 5], [1, 2], DType.int32);

          final srcEmbedded = rope.forward(tokenEmbedding.forward(srcTokens));
          final memory = rmsNorm.forward(encoderLayer.forward(srcEmbedded));
          final tgtEmbedded = rope.forward(tokenEmbedding.forward(tgtTokens));
          final decoded = decoderLayer.forward(tgtEmbedded, memory: memory);
          final gated = swiglu.forward(decoded);
          final flatDecoded = gated.reshape([2, 4]);
          final vocabLogits = lmHead.forward(flatDecoded) as GpuArray<Float64>;
          expect(vocabLogits.shape, equals([2, 6]));

          final targetIds = GpuArray.fromList(<int>[3, 1], [2], DType.int32);
          final seqLoss = gpu_nn.crossEntropy(vocabLogits, targetIds);
          seqLoss.backward();
          expect(tokenEmbedding.weight.grad, isNotNull);
          expect(lmHead.weight.grad, isNotNull);
        });
      },
    );

    test(
      'Scenario 10: Correlated Monte Carlo Portfolio Simulation via Cholesky & Philox4x32 RNG',
      () {
        ResourceScope.scope(() {
          final rng = gpu_random.RandomState(4242);
          final targetCov = GpuArray.fromList(
            <double>[1.0, 0.6, 0.6, 1.0],
            [2, 2],
            DType.float64,
          );
          final cholLower = gpu_linalg.cholesky(targetCov);

          const numDraws = 512;
          final uncorrelated = rng.standardNormal(shape: [numDraws, 2]);
          // X = Z * L^T has covariance L * L^T = targetCov
          final correlated = uncorrelated.matmul(cholLower.transpose());
          final centered =
              correlated - correlated.mean(axis: 0, keepDims: true);
          final sampleCov =
              centered.transpose().matmul(centered) / (numDraws - 1).toDouble();

          final sampleCovList = sampleCov.toList().cast<num>();
          expect(sampleCovList[0].toDouble(), closeTo(1.0, 0.25));
          expect(sampleCovList[1].toDouble(), closeTo(0.6, 0.25));
          expect(sampleCovList[3].toDouble(), closeTo(1.0, 0.25));
        });
      },
    );

    test(
      'Scenario 11: Power Iteration Spectral Radius vs eigvalsh, svdvals & cond',
      () {
        ResourceScope.scope(() {
          final spd = GpuArray.fromList(
            <double>[5.0, 2.0, 2.0, 2.0],
            [2, 2],
            DType.float64,
          );
          // Analytical eigenvalues of [[5, 2], [2, 2]] are 6.0 and 1.0
          var vec = GpuArray.fromList(<double>[1.0, 1.0], [2], DType.float64);
          for (var iter = 0; iter < 15; iter++) {
            final nextVec = gpu_linalg.matmul(spd, vec);
            final length = (gpu_linalg.norm(nextVec).scalar as num).toDouble();
            vec = nextVec / length;
          }
          final rayleighQuotient =
              (vec.dot(gpu_linalg.matmul(spd, vec)).scalar as num).toDouble();
          final exactMaxEig = (gpu_linalg.eigvalsh(spd).max().scalar as num)
              .toDouble();
          final conditionNum = (gpu_linalg.cond(spd).scalar as num).toDouble();

          expect(rayleighQuotient, closeTo(6.0, 1e-4));
          expect(exactMaxEig, closeTo(6.0, 1e-4));
          expect(conditionNum, closeTo(6.0, 1e-4));
        });
      },
    );

    test(
      'Scenario 12: Zero-Leak Scoped Iterative Optimization on Pooled WebGPU Device',
      () async {
        final device = await createWebGpuDevice(
          name: 'Scenario12-Optimization-Device',
          enableMemoryPool: true,
        );
        try {
          // Optimize f(w) = 0.5 * ||w - target||^2 on device without leaking intermediate buffers
          final param = GpuArray.fromList(
            <double>[10.0, -10.0, 5.0, -5.0],
            [4],
            DType.float32,
            device: device,
          );
          final target = GpuArray.fromList(
            <double>[1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
            device: device,
          );
          try {
            final baselineBuffers = device.activeBufferCount;
            expect(baselineBuffers, equals(2));

            for (var stepIndex = 0; stepIndex < 20; stepIndex++) {
              ResourceScope.scope(() {
                final grad = param - target;
                final stepDelta = grad * 0.2;
                final updated = param - stepDelta;
                updated.copy(out: param);
              });
              expect(device.activeBufferCount, equals(baselineBuffers));
            }

            expect(device.memoryPool.hits, greaterThan(0));
            _expectCloseList(param.toList(), <double>[
              1.0,
              2.0,
              3.0,
              4.0,
            ], tolerance: 0.2);
          } finally {
            param.dispose();
            target.dispose();
          }
          expect(device.activeBufferCount, equals(0));
        } finally {
          device.dispose();
        }
      },
    );

    test(
      'Scenario 13: End-to-End Float32 Transformer Training, Fused AdamW, Grad Clipping & SafeTensors Checkpointing',
      () {
        ResourceScope.scope(() {
          gpu_random.seed(77);
          final model = gpu_nn.Sequential([
            gpu_nn.Linear(4, 8, dtype: DType.float32),
            gpu_nn.GELU(),
            gpu_nn.LayerNorm([8], dtype: DType.float32),
            gpu_nn.Linear(8, 2, dtype: DType.float32),
          ]);
          final opt = gpu_nn.AdamW(
            model.parameters,
            lr: 0.05,
            weightDecay: 0.01,
          );
          final x = GpuArray<Float32>.fromList(
            <double>[1.0, 0.5, -0.5, 2.0, -1.0, 1.5, 0.25, -0.75],
            [2, 4],
            DType.float32,
          );
          final y = GpuArray<Float32>.fromList(
            <double>[1.0, -1.0, -0.5, 0.5],
            [2, 2],
            DType.float32,
          );

          final initialLoss =
              (gpu_nn.mseLoss(model.forward(x), y).scalar as num).toDouble();
          for (var step = 0; step < 15; step++) {
            opt.zeroGrad();
            final pred = model.forward(x);
            final loss = gpu_nn.mseLoss(pred, y);
            loss.backward();
            gpu_nn.clipGradNorm(model.parameters, 5.0);
            opt.step();
          }
          final finalLoss = (gpu_nn.mseLoss(model.forward(x), y).scalar as num)
              .toDouble();
          expect(finalLoss, lessThan(initialLoss));

          final ckpt = model.saveToSafetensors();
          final restored = gpu_nn.Sequential([
            gpu_nn.Linear(4, 8, dtype: DType.float32),
            gpu_nn.GELU(),
            gpu_nn.LayerNorm([8], dtype: DType.float32),
            gpu_nn.Linear(8, 2, dtype: DType.float32),
          ]);
          restored.loadFromSafetensors(ckpt);
          _expectCloseList(
            restored.forward(x).toList(),
            model.forward(x).toList(),
            tolerance: 1e-5,
          );
        });
      },
    );

    test(
      'Scenario 14: Float32 Spectral Filtering, NaN Imputation & Top-K Peak Detection',
      () {
        ResourceScope.scope(() {
          final noisyWithNan = GpuArray<Float32>.fromList(
            <double>[1.0, 4.0, double.nan, 2.0, 5.0, 1.0, 3.0, 2.0],
            [8],
            DType.float32,
          );
          final meanVal = (nanmean(noisyWithNan).scalar as num).toDouble();
          final imputed = nanToNum(noisyWithNan, nan: meanVal);
          final spectrum = gpu_fft.rfft(imputed);
          expect(spectrum.dtype, equals(DType.complex64));
          final magnitudes = spectrum.abs().real();
          expect(magnitudes.dtype, equals(DType.float32));
          final topBins = topk(magnitudes, 2);
          expect(topBins.indices.dtype, equals(DType.int64));
          expect(
            topBins.indices.toList().first,
            equals(0),
          ); // DC component dominates
        });
      },
    );

    test(
      'Scenario 15: Custom Fused WGSL Kinematics Step via CompiledWgslKernel + Cumulative Scans',
      () {
        ResourceScope.scope(() {
          final vExpr = Expr.variable('v', bindingIndex: 0);
          final aExpr = Expr.variable('a', bindingIndex: 1);
          final dtExpr = Expr.scalar('dt');
          final stepKernel = GpuDevice.defaultDevice.jitCompiler.compileKernel(
            vExpr + (aExpr * dtExpr),
          );
          final v0 = GpuArray<Float32>.fromList(
            <double>[1.0, 1.0, 1.0, 1.0],
            [4],
            DType.float32,
          );
          final accel = GpuArray<Float32>.fromList(
            <double>[2.0, 4.0, 6.0, 8.0],
            [4],
            DType.float32,
          );
          final vNext = stepKernel.execute<Float32>(
            {'v': v0, 'a': accel},
            scalars: const {'dt': 0.5},
          );
          // vNext = [2.0, 3.0, 4.0, 5.0]
          final displacement = cumsum(vNext);
          final recoveredV = diff(displacement, prepend: 0.0);
          _expectCloseList(displacement.toList(), <double>[
            2.0,
            5.0,
            9.0,
            14.0,
          ]);
          _expectCloseList(recoveredV.toList(), vNext.toList());
        });
      },
    );
  });
}
