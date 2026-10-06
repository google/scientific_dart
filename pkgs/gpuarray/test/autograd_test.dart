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

import 'package:gpuarray/nn.dart' as nn;
import 'package:gpuarray/src/autograd/autograd.dart';
import 'package:gpuarray/src/device.dart';
import 'package:gpuarray/src/dtype.dart';
import 'package:gpuarray/src/gpu_array.dart' hide ResourceScope, ScopedResource;
import 'package:gpuarray/src/operations/manipulation.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  group('GpuArray Automatic Differentiation (Autograd)', () {
    test('Scalar arithmetic and polynomial gradient', () {
      ResourceScope.scope(() {
        // f(x) = 3x^2 + 2x + 1, f'(x) = 6x + 2
        // at x = 4, f'(4) = 26
        final x = GpuArray.fromList(
          [4.0],
          [1],
          DType.float64,
          requiresGrad: true,
        );
        expect(x.isLeaf, isTrue);

        final y = (x * x * 3.0) + (x * 2.0) + 1.0;
        expect(y.requiresGrad, isTrue);
        expect(y.gradFn, isNotNull);

        y.backward();

        expect(x.grad, isNotNull);
        expect(x.grad!.toList().cast<double>().first, closeTo(26.0, 1e-4));
      });
    });

    test(
      'SigmoidBackward and TanhBackward declare forward input as DAG inputs',
      () {
        ResourceScope.scope(() {
          // Test Sigmoid at x = 0.0: sigma(0) = 0.5, sigma'(0) = 0.25
          final xSig = GpuArray.fromList(
            [0.0],
            [1],
            DType.float64,
            requiresGrad: true,
          );
          final ySig = nn.sigmoid(xSig);
          expect(ySig.requiresGrad, isTrue);
          expect(ySig.gradFn, isA<SigmoidBackward>());
          expect(
            ySig.gradFn!.inputs,
            equals([xSig]),
          ); // Must point to input x, not output y

          ySig.backward();
          expect(xSig.grad, isNotNull);
          expect(xSig.grad!.toList().cast<double>().first, closeTo(0.25, 1e-4));

          // Test Tanh at x = 0.0: tanh(0) = 0.0, tanh'(0) = 1.0
          final xTanh = GpuArray.fromList(
            [0.0],
            [1],
            DType.float64,
            requiresGrad: true,
          );
          final yTanh = nn.tanh(xTanh);
          expect(yTanh.requiresGrad, isTrue);
          expect(yTanh.gradFn, isA<TanhBackward>());
          expect(
            yTanh.gradFn!.inputs,
            equals([xTanh]),
          ); // Must point to input x

          yTanh.backward();
          expect(xTanh.grad, isNotNull);
          expect(xTanh.grad!.toList().cast<double>().first, closeTo(1.0, 1e-4));
        });
      },
    );

    test('View Tracking: Transpose and Swapaxes backward propagation', () {
      ResourceScope.scope(() {
        // X: [2, 3] -> X^T: [3, 2]
        final x = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
          requiresGrad: true,
        );

        final xT = x.transpose([1, 0]);
        expect(xT.requiresGrad, isTrue);
        expect(xT.shape, equals([3, 2]));
        expect(xT.gradFn, isA<TransposeBackward>());
        expect(xT.gradFn!.inputs, equals([x]));

        // Loss = sum(xT * weights)
        final w = GpuArray.fromList(
          [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
          [3, 2],
          DType.float64,
        );

        final loss = (xT * w).sum();
        loss.backward();

        expect(x.grad, isNotNull);
        expect(x.grad!.shape, equals([2, 3]));
        // w^T is [[10, 30, 50], [20, 40, 60]]
        final xGrad = x.grad!.toList().cast<double>();
        expect(xGrad, equals([10.0, 30.0, 50.0, 20.0, 40.0, 60.0]));

        // Test swapaxes
        final x2 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
          requiresGrad: true,
        );
        final xSwapped = x2.swapaxes(-1, -2);
        expect(xSwapped.requiresGrad, isTrue);
        expect(xSwapped.gradFn, isA<TransposeBackward>());
        final loss2 = xSwapped.sum();
        loss2.backward();
        expect(x2.grad!.toList().cast<double>(), equals([1.0, 1.0, 1.0, 1.0]));
      });
    });

    test(
      'View Tracking: Reshape, Squeeze, Unsqueeze, and Slice backward propagation',
      () {
        ResourceScope.scope(() {
          // Reshape [4] -> [2, 2]
          final x = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
            requiresGrad: true,
          );
          final reshaped = x.reshape([2, 2]);
          expect(reshaped.requiresGrad, isTrue);
          expect(reshaped.gradFn, isA<ReshapeBackward>());

          final loss1 = (reshaped * 2.0).sum();
          loss1.backward();
          expect(x.grad!.shape, equals([4]));
          expect(x.grad!.toList().cast<double>(), equals([2.0, 2.0, 2.0, 2.0]));

          // Squeeze & Unsqueeze
          final x2 = GpuArray.fromList(
            [5.0, 6.0],
            [1, 2, 1],
            DType.float64,
            requiresGrad: true,
          );
          final sq = x2.squeeze();
          expect(sq.shape, equals([2]));
          expect(sq.requiresGrad, isTrue);
          expect(sq.gradFn, isA<SqueezeBackward>());

          final unsq = sq.unsqueeze(0);
          expect(unsq.shape, equals([1, 2]));
          expect(unsq.requiresGrad, isTrue);
          expect(unsq.gradFn, isA<UnsqueezeBackward>());

          // Slice subview gradient
          final x3 = GpuArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [3, 2],
            DType.float64,
            requiresGrad: true,
          );

          // Slice row 1: [30.0, 40.0]
          final row1 = x3[1];
          expect(row1.shape, equals([2]));
          expect(row1.requiresGrad, isTrue);
          expect(row1.gradFn, isA<SliceBackward>());

          final loss3 = (row1 * 3.0).sum();
          loss3.backward();
          expect(x3.grad!.shape, equals([3, 2]));
          // Row 0 and 2 are 0.0, row 1 is [3.0, 3.0]
          expect(
            x3.grad!.toList().cast<double>(),
            equals([0.0, 0.0, 3.0, 3.0, 0.0, 0.0]),
          );
        });
      },
    );

    test('Vector and matrix multi-variable gradient (z = x @ y)', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
          requiresGrad: true,
        );

        final y = GpuArray.fromList(
          [5.0, 6.0, 7.0, 8.0],
          [2, 2],
          DType.float64,
          requiresGrad: true,
        );

        final z = x.matmul(y);
        final loss = z.sum();

        loss.backward();

        expect(x.grad, isNotNull);
        expect(y.grad, isNotNull);

        final xGrad = x.grad!.toList().cast<double>();
        expect(xGrad, equals([11.0, 15.0, 11.0, 15.0]));

        final yGrad = y.grad!.toList().cast<double>();
        expect(yGrad, equals([4.0, 4.0, 6.0, 6.0]));
      });
    });

    test('Batched Matmul with broadcast reduction in MatmulBackward', () {
      ResourceScope.scope(() {
        // A: [1, 2, 2], B: [2, 2, 2]
        final a = GpuArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [1, 2, 2],
          DType.float64,
          requiresGrad: true,
        );

        final b = GpuArray.fromList(
          [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [2, 2, 2],
          DType.float64,
          requiresGrad: true,
        );

        final c = a.matmul(b); // shape [2, 2, 2]
        expect(c.shape, equals([2, 2, 2]));

        final loss = c.sum();
        loss.backward();

        // a.grad should be reduced across batch dimension to [1, 2, 2]
        expect(a.grad, isNotNull);
        expect(a.grad!.shape, equals([1, 2, 2]));

        // b.grad should have shape [2, 2, 2]
        expect(b.grad, isNotNull);
        expect(b.grad!.shape, equals([2, 2, 2]));
      });
    });

    test('Broadcasting gradient accumulation (Linear layer: Y = XW + B)', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final w = GpuArray.fromList(
          [0.5, -0.5, 1.0, 2.0],
          [2, 2],
          DType.float64,
          requiresGrad: true,
        );

        final b = GpuArray.fromList(
          [0.1, 0.2],
          [2],
          DType.float64,
          requiresGrad: true,
        );

        final out = x.matmul(w) + b;
        final loss = out.mean();

        loss.backward();

        expect(w.grad, isNotNull);
        expect(b.grad, isNotNull);
        expect(b.grad!.shape, equals([2]));

        final bGrad = b.grad!.toList().cast<double>();
        expect(bGrad[0], closeTo(0.5, 1e-4));
        expect(bGrad[1], closeTo(0.5, 1e-4));
      });
    });

    test('Softmax and LogSoftmax backward', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList(
          [1.0, 2.0, 3.0],
          [1, 3],
          DType.float64,
          requiresGrad: true,
        );
        final sm = nn.softmax(x);
        final loss = sm
            .sum(); // sum of probabilities is 1.0, derivative wrt inputs should be 0.0
        loss.backward();

        expect(x.grad, isNotNull);
        final xGrad = x.grad!.toList().cast<double>();
        for (final g in xGrad) {
          expect(g, closeTo(0.0, 1e-4));
        }

        final x2 = GpuArray.fromList(
          [1.0, 2.0, 3.0],
          [1, 3],
          DType.float64,
          requiresGrad: true,
        );
        final lsm = nn.logSoftmax(x2);
        expect(lsm.requiresGrad, isTrue);
        expect(lsm.gradFn, isA<LogSoftmaxBackward>());
      });
    });

    test('noGrad, zeroGrad, detach, and non-scalar backward check', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList(
          [3.0],
          [1],
          DType.float64,
          requiresGrad: true,
        );
        final y1 = noGrad(() => x * 3.0 + 4.0);
        expect(y1.requiresGrad, isFalse);
        expect(y1.gradFn, isNull);
        final y2 = noGrad(() => enableGrad(() => x * 5.0));
        expect(y2.requiresGrad, isTrue);
        expect(y2.gradFn, isNotNull);

        final y = x * 2.0;
        y.backward();
        expect(x.grad, isNotNull);
        x.zeroGrad();
        expect(x.grad, isNull);

        final detached = x.detach();
        expect(detached.requiresGrad, isFalse);
        expect(detached.toList(), equals([3.0]));

        final nonScalar = GpuArray.fromList(
          [1.0, 2.0],
          [2],
          DType.float64,
          requiresGrad: true,
        );
        expect(() => (nonScalar * 2.0).backward(), throwsArgumentError);
      });
    });

    test(
      'ConcatenateBackward, Sqrt/Exp/LogBackward, and buffer leak check',
      () {
        final baselineBuffers = GpuDevice.defaultDevice.activeBufferCount;
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          final b = GpuArray.fromList(
            [3.0, 4.0],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          final concatenated = concatenate([a, b], axis: 0);
          expect(concatenated.gradFn, isA<ConcatenateBackward>());
          (concatenated * 2.0).sum().backward();
          expect(a.grad!.toList().cast<double>(), equals([2.0, 2.0]));
          expect(b.grad!.toList().cast<double>(), equals([2.0, 2.0]));

          final z = GpuArray.fromList(
            [4.0],
            [1],
            DType.float64,
            requiresGrad: true,
          );
          z.sqrt().exp().log().backward();
          expect(z.grad!.scalar, closeTo(0.25, 1e-5));
          a.zeroGrad();
          b.zeroGrad();
          z.zeroGrad();
        });
        expect(
          GpuDevice.defaultDevice.activeBufferCount,
          equals(baselineBuffers),
        );
      },
    );

    test(
      'retainGraph, released graph StateError, gradient shape check, and Sin/Cos/AbsBackward',
      () {
        ResourceScope.scope(() {
          final xLeaf = GpuArray.fromList(
            [3.0],
            [1],
            DType.float64,
            requiresGrad: true,
          );
          final loss = (xLeaf * xLeaf).sum();
          loss.backward(retainGraph: true);
          expect(xLeaf.grad!.toList().cast<double>(), equals([6.0]));

          xLeaf.zeroGrad();
          expect(xLeaf.grad, isNull);
          loss.backward();
          expect(xLeaf.grad!.toList().cast<double>(), equals([6.0]));
          expect(() => loss.backward(), throwsStateError);

          final vec = GpuArray.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          final out = vec * 2.0;
          final wrongGrad = GpuArray.ones([3], DType.float64);
          expect(() => out.backward(gradient: wrongGrad), throwsArgumentError);

          final angle = GpuArray.fromList(
            [0.0],
            [1],
            DType.float64,
            requiresGrad: true,
          );
          final seed = GpuArray.ones([1], DType.float64);
          expect(
            SinBackward(angle).backward(seed).first!.scalar,
            closeTo(1.0, 1e-5),
          );
          expect(
            CosBackward(angle).backward(seed).first!.scalar,
            closeTo(0.0, 1e-5),
          );
          expect(
            AbsBackward(vec)
                .backward(GpuArray.ones([2], DType.float64))
                .first!
                .toList()
                .cast<double>(),
            equals([1.0, 1.0]),
          );
        });
      },
    );
  });
}
