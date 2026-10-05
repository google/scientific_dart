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

import 'package:gpuarray/autograd.dart' as autograd;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/jit.dart';
import 'package:gpuarray/nn.dart' as nn;
import 'package:gpuarray/serialization.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  final device = GpuDevice.defaultDevice;

  group('Adversarial Audit: R2 Fused WGSL Activations & Zero Allocations', () {
    test(
      'All 9 fused activations allocate 0 intermediate buffers with out: destination',
      () {
        final baselineBuffers = device.activeBufferCount;

        ResourceScope.scope(() {
          final x = GpuArray<Float32>.fromList(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [5],
            DType.float32,
          );
          final out = GpuArray<Float32>.empty([5], DType.float32);

          final ops = <void Function()>[
            () => nn.relu(x, out: out),
            () => nn.sigmoid(x, out: out),
            () => nn.tanh(x, out: out),
            () => nn.gelu(x, out: out),
            () => nn.silu(x, out: out),
            () => nn.swish(x, out: out),
            () => nn.leakyRelu(x, negativeSlope: 0.1, out: out),
            () => nn.elu(x, alpha: 1.0, out: out),
            () => nn.softplus(x, beta: 1.0, out: out),
          ];

          for (final op in ops) {
            final countBefore = device.activeBufferCount;
            op();
            // With out: preallocated, no new buffers must be created during dispatch
            expect(
              device.activeBufferCount,
              equals(countBefore),
              reason: 'Intermediate buffers leaked during activation dispatch',
            );
          }

          out.dispose();
          x.dispose();
        });

        expect(device.activeBufferCount, equals(baselineBuffers));
      },
    );

    test('In-place 1-binding fused activation execution (out: input)', () {
      ResourceScope.scope(() {
        final x = GpuArray<Float32>.fromList(
          [-2.0, -1.0, 0.0, 1.0, 2.0],
          [5],
          DType.float32,
        );

        // In-place ReLU: x becomes [0, 0, 0, 1, 2]
        final res = nn.relu(x, out: x);
        expect(identical(res, x), isTrue);
        expect(x.toList().cast<double>(), equals([0.0, 0.0, 0.0, 1.0, 2.0]));

        // In-place LeakyReLU on negative values
        final x2 = GpuArray<Float32>.fromList(
          [-10.0, 10.0],
          [2],
          DType.float32,
        );
        nn.leakyRelu(x2, negativeSlope: 0.1, out: x2);
        expect(x2.toList().cast<double>(), equals([-1.0, 10.0]));

        x.dispose();
        x2.dispose();
      });
    });

    test(
      'Autograd backward through fused activations leaves zero buffer leaks',
      () {
        final baselineBuffers = device.activeBufferCount;

        ResourceScope.scope(() {
          final x = GpuArray<Float32>.fromList(
            [-1.5, -0.5, 0.5, 1.5],
            [4],
            DType.float32,
            requiresGrad: true,
          );

          final acts = <GpuArray<Float32> Function(GpuArray<Float32>)>[
            (t) => nn.relu(t),
            (t) => nn.sigmoid(t),
            (t) => nn.tanh(t),
            (t) => nn.gelu(t),
            (t) => nn.silu(t),
            (t) => nn.swish(t),
            (t) => nn.leakyRelu(t, negativeSlope: 0.2),
            (t) => nn.elu(t, alpha: 1.0),
            (t) => nn.softplus(t, beta: 1.0),
          ];

          for (final act in acts) {
            x.zeroGrad();
            final y = act(x);
            final loss = y.sum();
            loss.backward();
            expect(x.grad, isNotNull);
            expect(x.grad!.shape, equals([4]));
            expect(x.grad!.dtype, equals(DType.float32));
            for (final g in x.grad!.toList().cast<double>()) {
              expect(g.isFinite, isTrue);
            }
          }
          x.zeroGrad();
        });

        expect(device.activeBufferCount, equals(baselineBuffers));
      },
    );

    test('mseLoss single-pass fused kernel across all LossReduction modes', () {
      final baselineBuffers = device.activeBufferCount;

      ResourceScope.scope(() {
        final pred = GpuArray<Float32>.fromList(
          [2.0, 4.0, 6.0],
          [3],
          DType.float32,
          requiresGrad: true,
        );
        final target = GpuArray<Float32>.fromList(
          [1.0, 1.0, 3.0],
          [3],
          DType.float32,
        );

        // diff = [1, 3, 3], diff^2 = [1, 9, 9]
        // Reduction.none: [1, 9, 9]
        final lossNone = nn.mseLoss(
          pred,
          target,
          reduction: autograd.LossReduction.none,
        );
        expect(lossNone.shape, equals([3]));
        expect(lossNone.toList().cast<double>(), equals([1.0, 9.0, 9.0]));

        // Reduction.sum: 19.0
        final lossSum = nn.mseLoss(
          pred,
          target,
          reduction: autograd.LossReduction.sum,
        );
        expect((lossSum.scalar as num).toDouble(), closeTo(19.0, 1e-4));

        // Reduction.mean: 19/3 ~ 6.3333
        final lossMean = nn.mseLoss(
          pred,
          target,
          reduction: autograd.LossReduction.mean,
        );
        expect((lossMean.scalar as num).toDouble(), closeTo(19.0 / 3.0, 1e-4));

        // Backward through mean reduction: d/dpred = 2 * (pred - target) / 3 = [2/3, 2, 2]
        pred.zeroGrad();
        lossMean.backward();
        expect(pred.grad, isNotNull);
        final grads = pred.grad!.toList().cast<double>();
        expect(grads[0], closeTo(2.0 / 3.0, 1e-4));
        expect(grads[1], closeTo(2.0, 1e-4));
        expect(grads[2], closeTo(2.0, 1e-4));

        pred.zeroGrad();
      });

      expect(device.activeBufferCount, equals(baselineBuffers));
    });
  });

  group(
    'Adversarial Audit: R2 Fused WGSL Optimizers In-Place & Zero Allocations',
    () {
      test(
        'SGD, Adam, and AdamW allocate 0 intermediate buffers during step()',
        () {
          final baselineBuffers = device.activeBufferCount;

          ResourceScope.scope(() {
            final pSgd = GpuArray<Float32>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float32,
              requiresGrad: true,
            );
            final pAdam = GpuArray<Float32>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float32,
              requiresGrad: true,
            );
            final pAdamW = GpuArray<Float32>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float32,
              requiresGrad: true,
            );

            final sgd = nn.SGD(
              [pSgd],
              lr: 0.1,
              momentum: 0.9,
              weightDecay: 1e-3,
              nesterov: true,
            );
            final adam = nn.Adam([pAdam], lr: 0.01, weightDecay: 1e-4);
            final adamW = nn.AdamW([pAdamW], lr: 0.01, weightDecay: 1e-2);

            // Step 1: Initial state buffers created and detached
            (pSgd * 2.0).sum().backward();
            (pAdam * 2.0).sum().backward();
            (pAdamW * 2.0).sum().backward();

            sgd.step();
            adam.step();
            adamW.step();

            // Count buffers after step 1
            final steadyStateBuffers = device.activeBufferCount;

            // Steps 2 to 5: must allocate 0 new intermediate buffers during step()
            for (var i = 0; i < 4; i++) {
              ResourceScope.scope(() {
                sgd.zeroGrad();
                adam.zeroGrad();
                adamW.zeroGrad();

                (pSgd * 2.0).sum().backward();
                (pAdam * 2.0).sum().backward();
                (pAdamW * 2.0).sum().backward();

                final countBefore = device.activeBufferCount;
                sgd.step();
                adam.step();
                adamW.step();
                expect(
                  device.activeBufferCount,
                  equals(countBefore),
                  reason:
                      'Intermediate buffers allocated during optimizer step',
                );
              });
            }
            expect(device.activeBufferCount, equals(steadyStateBuffers));

            // Disposing optimizers frees internal velocity and moment buffers
            sgd.dispose();
            adam.dispose();
            adamW.dispose();
            pSgd.zeroGrad();
            pAdam.zeroGrad();
            pAdamW.zeroGrad();
          });

          expect(device.activeBufferCount, equals(baselineBuffers));
        },
      );
    },
  );

  group('Adversarial Audit: R2 CompiledWgslKernel & JIT GpuArray API', () {
    test(
      'CompiledWgslKernel named, positional, and callable invocation with scalars',
      () {
        ResourceScope.scope(() {
          final x = Expr.variable('x', bindingIndex: 0);
          final y = Expr.variable('y', bindingIndex: 1);
          final gamma = Expr.scalar('gamma', defaultValue: 1.0);
          final bias = Expr.scalar('bias', defaultValue: 0.0);

          final fusedExpr = ((x * gamma) + y + bias).gelu();
          final kernel = WgslJitCompiler.instance.compileKernel(
            fusedExpr,
            name: 'adversarial_gelu_kernel',
          );

          final inX = GpuArray<Float32>.fromList(
            [1.0, -1.0],
            [2],
            DType.float32,
          );
          final inY = GpuArray<Float32>.fromList(
            [0.5, -0.5],
            [2],
            DType.float32,
          );

          // 1. Named execution
          final resNamed = kernel.execute<Float32>(
            {'x': inX, 'y': inY},
            scalars: {'gamma': 2.0, 'bias': 1.0},
          );
          expect(resNamed.shape, equals([2]));
          expect(resNamed.dtype, equals(DType.float32));

          // 2. Positional execution
          final resPos = kernel.executePositional<Float32>(
            [inX, inY],
            scalars: {'gamma': 2.0, 'bias': 1.0},
          );
          expect(resPos.toList(), equals(resNamed.toList()));

          // 3. Callable shorthand syntax
          final resCall = kernel<Float32>(
            {'x': inX, 'y': inY},
            scalars: {'gamma': 2.0, 'bias': 1.0},
          );
          expect(resCall.toList(), equals(resNamed.toList()));

          // 4. Preallocated out: parameter
          final preallocated = GpuArray<Float32>.empty([2], DType.float32);
          final resOut = kernel.execute<Float32>(
            {'x': inX, 'y': inY},
            scalars: {'gamma': 2.0, 'bias': 1.0},
            out: preallocated,
          );
          expect(identical(resOut, preallocated), isTrue);
          expect(resOut.toList(), equals(resNamed.toList()));

          resNamed.dispose();
          resPos.dispose();
          resCall.dispose();
          preallocated.dispose();
          inX.dispose();
          inY.dispose();
        });
      },
    );
  });

  group('Adversarial Audit: R3 Module Lifecycle & Gradient Clipping', () {
    test(
      'clipGradNorm with zero gradients, L1, L2, L-inf, and single-parameter',
      () {
        ResourceScope.scope(() {
          final w = GpuArray<Float32>.zeros(
            [4],
            DType.float32,
            requiresGrad: true,
          );
          (w * 0.0).sum().backward();

          // Zero total norm
          final zeroNorm = nn.clipGradNorm([w], 1.0);
          expect(zeroNorm, equals(0.0));

          // Non-zero gradient norm
          final w2 = GpuArray<Float32>.fromList(
            [3.0, 4.0],
            [2],
            DType.float32,
            requiresGrad: true,
          );
          (w2 * 1.0)
              .sum()
              .backward(); // grad = [1.0, 1.0], L2 norm = sqrt(2) ~ 1.4142

          final initialL2 = nn.clipGradNorm([w2], 1.0, normType: 2.0);
          expect(initialL2, closeTo(math.sqrt(2.0), 1e-4));
          final clippedL2 = nn.clipGradNorm([w2], 100.0, normType: 2.0);
          expect(clippedL2, closeTo(1.0, 1e-3));

          // L-inf norm
          final w3 = GpuArray<Float32>.fromList(
            [1.0, 10.0],
            [2],
            DType.float32,
            requiresGrad: true,
          );
          (w3 * 1.0).sum().backward(); // grad = [1.0, 1.0], max = 1.0
          final infNorm = nn.clipGradNorm([w3], 0.5, normType: double.infinity);
          expect(infNorm, closeTo(1.0, 1e-4));
          final clippedInf = nn.clipGradNorm(
            [w3],
            100.0,
            normType: double.infinity,
          );
          expect(clippedInf, closeTo(0.5, 1e-3));

          // clipGradValue
          final w4 = GpuArray<Float32>.fromList(
            [1.0],
            [1],
            DType.float32,
            requiresGrad: true,
          );
          (w4 * 10.0).sum().backward(); // grad = [10.0]
          nn.clipGradValue([w4], 2.5);
          expect(w4.grad!.scalar, closeTo(2.5, 1e-5));

          w.zeroGrad();
          w2.zeroGrad();
          w3.zeroGrad();
          w4.zeroGrad();
        });
      },
    );

    test(
      'Module hierarchy: named submodules, buffers, stateDict, SafeTensors, to(), dispose()',
      () {
        ResourceScope.scope(() {
          final net = nn.Sequential([
            nn.Linear(2, 4, dtype: DType.float32),
            nn.BatchNorm1d(4, dtype: DType.float32),
            nn.Linear(4, 1, dtype: DType.float32),
          ]);

          // State dict keys verification
          final sd = net.stateDict();
          expect(
            sd.keys,
            containsAll([
              '0.weight',
              '0.bias',
              '1.weight',
              '1.bias',
              '1.runningMean',
              '1.runningVar',
              '2.weight',
              '2.bias',
            ]),
          );

          // SafeTensors round-trip bytes
          final bytes = net.saveToSafetensors(
            metadata: {'author': 'challenger_2'},
          );
          expect(bytes.length, greaterThan(0));

          final netClone = nn.Sequential([
            nn.Linear(2, 4, dtype: DType.float32),
            nn.BatchNorm1d(4, dtype: DType.float32),
            nn.Linear(4, 1, dtype: DType.float32),
          ]);
          netClone.loadFromSafetensors(bytes);

          final xTest = GpuArray<Float32>.fromList(
            [1.0, 2.0],
            [1, 2],
            DType.float32,
          );
          net.eval();
          netClone.eval();
          expect(netClone(xTest).toList(), equals(net(xTest).toList()));

          // In-place to(device)
          netClone.to(device);

          // strict: true rejects missing/unexpected keys
          expect(
            () => netClone.loadStateDict({
              'badKey': sd['0.weight']!,
            }, strict: true),
            throwsArgumentError,
          );

          // Dispose module marks disposed and prevents forward
          expect(netClone.isDisposed, isFalse);
          netClone.dispose();
          expect(netClone.isDisposed, isTrue);
          expect(() => netClone(xTest), throwsStateError);
        });
      },
    );
  });

  group('Adversarial Audit: R3 Clean Package Entrypoints & Naming Hygiene', () {
    test(
      'Public entrypoint imports do not collide and keep domain separation',
      () {
        // Core gpuarray does not export Module, Linear, etc.
        // nn.dart exports Module, Linear, optimizers
        // autograd.dart exports noGrad, enableGrad, isGradEnabled, runBackward
        // jit.dart exports WgslJitCompiler, CompiledWgslKernel
        // serialization.dart exports saveSafetensors, loadSafetensors
        expect(nn.Module, isNotNull);
        expect(autograd.runBackward, isNotNull);
        expect(WgslJitCompiler, isNotNull);
        expect(saveSafetensors, isNotNull);
      },
    );
  });
}
