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
import 'package:gpuarray/src/dtype.dart';
import 'package:gpuarray/src/gpu_array.dart' hide ResourceScope, ScopedResource;
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  group('GpuArray Neural Network Primitives (gpuarray.nn)', () {
    test('Linear layer forward and parameter registration', () {
      ResourceScope.scope(() {
        final fc = nn.Linear(4, 2);
        expect(fc.parameters.length, equals(2)); // weight and bias
        expect(fc.namedParameters().keys, containsAll(['weight', 'bias']));

        final x = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
          [2, 4],
          DType.float64,
        );

        final out = fc(x);
        expect(out.shape, equals([2, 2]));
        expect(out.requiresGrad, isTrue);

        final loss = out.sum();
        loss.backward();

        // Check that gradients flow to weight and bias via TransposeBackward
        expect(fc.weight.grad, isNotNull);
        expect(fc.weight.grad!.shape, equals([2, 4]));
        expect(fc.bias!.grad, isNotNull);
        expect(fc.bias!.grad!.shape, equals([2]));
      });
    });

    test('Sequential MLP with ReLU and MSE Loss optimization (SGD / Adam)', () {
      ResourceScope.scope(() {
        // Target function: y = 2*x1 - 3*x2
        final model = nn.Sequential([
          nn.Linear(2, 4),
          nn.ReLU(),
          nn.Linear(4, 1),
        ]);

        final optimizer = nn.Adam(model.parameters, lr: 0.05);

        final xTrain = GpuArray.fromList(
          [1.0, 1.0, 2.0, 0.0, 0.0, 1.0, -1.0, 2.0],
          [4, 2],
          DType.float64,
        );

        final yTrain = GpuArray.fromList(
          [-1.0, 4.0, -3.0, -8.0],
          [4, 1],
          DType.float64,
        );

        // Train for 20 steps
        double? initialLoss;
        double? finalLoss;

        for (var epoch = 0; epoch < 20; epoch++) {
          optimizer.zeroGrad();
          final pred = model(xTrain);
          final loss = nn.mseLoss(pred, yTrain);

          final lossVal = (loss.scalar as num).toDouble();
          if (epoch == 0) initialLoss = lossVal;
          if (epoch == 19) finalLoss = lossVal;

          loss.backward();
          optimizer.step();
        }

        expect(initialLoss, isNotNull);
        expect(finalLoss, isNotNull);
        expect(finalLoss, lessThan(initialLoss!)); // Loss decreased
        optimizer.dispose();
      });
    });

    test('Embedding layer forward and backward gradient accumulation', () {
      ResourceScope.scope(() {
        final emb = nn.Embedding(5, 3);
        expect(emb.weight.shape, equals([5, 3]));

        // Indices: [1, 3, 1] (index 1 is repeated)
        final indices = GpuArray.fromList([1, 3, 1], [3], DType.int32);
        final out = emb(indices);
        expect(out.shape, equals([3, 3]));
        expect(out.requiresGrad, isTrue);

        final loss = out.sum();
        loss.backward();

        expect(emb.weight.grad, isNotNull);
        expect(emb.weight.grad!.shape, equals([5, 3]));

        final gradList = emb.weight.grad!.toList().cast<double>();
        // Row 0, 2, 4 should have 0.0 grad
        // Row 1 should have 2.0 grad (repeated twice)
        // Row 3 should have 1.0 grad
        expect(gradList.sublist(0, 3), equals([0.0, 0.0, 0.0]));
        expect(gradList.sublist(3, 6), equals([2.0, 2.0, 2.0]));
        expect(gradList.sublist(6, 9), equals([0.0, 0.0, 0.0]));
        expect(gradList.sublist(9, 12), equals([1.0, 1.0, 1.0]));
        expect(gradList.sublist(12, 15), equals([0.0, 0.0, 0.0]));
      });
    });

    test('crossEntropy and mseLoss with LossReduction.none, mean, and sum', () {
      ResourceScope.scope(() {
        // 2 samples, 3 classes
        final logits = GpuArray.fromList(
          [2.0, 1.0, 0.1, 0.5, 3.0, 0.2],
          [2, 3],
          DType.float64,
          requiresGrad: true,
        );

        final targets = GpuArray.fromList([0, 1], [2], DType.int32);
        final lossMean = nn.crossEntropy(
          logits,
          targets,
          reduction: nn.LossReduction.mean,
        );
        expect(lossMean.requiresGrad, isTrue);

        lossMean.backward();
        expect(logits.grad, isNotNull);
        expect(logits.grad!.shape, equals([2, 3]));

        final gList = logits.grad!.toList().cast<double>();
        final sumRow0 = gList[0] + gList[1] + gList[2];
        final sumRow1 = gList[3] + gList[4] + gList[5];
        expect(sumRow0, closeTo(0.0, 1e-4));
        expect(sumRow1, closeTo(0.0, 1e-4));

        logits.zeroGrad();
        final lossSum = nn.crossEntropy(
          logits,
          targets,
          reduction: nn.LossReduction.sum,
        );
        expect(
          lossSum.scalar,
          closeTo((lossMean.scalar as num).toDouble() * 2.0, 1e-5),
        );
        lossSum.backward();
        final gSumList = logits.grad!.toList().cast<double>();
        expect(gSumList[0], closeTo(gList[0] * 2.0, 1e-5));

        final lossNone = nn.crossEntropy(
          logits,
          targets,
          reduction: nn.LossReduction.none,
        );
        expect(lossNone.shape, equals([2]));

        // MSE Loss reductions
        final pred = GpuArray.fromList(
          [1.0, 3.0],
          [2],
          DType.float64,
          requiresGrad: true,
        );
        final target = GpuArray.fromList([2.0, 1.0], [2], DType.float64);
        final mseNone = nn.mseLoss(
          pred,
          target,
          reduction: nn.LossReduction.none,
        );
        expect(mseNone.toList().cast<double>(), equals([1.0, 4.0]));
        final mseSum = nn.mseLoss(
          pred,
          target,
          reduction: nn.LossReduction.sum,
        );
        expect(mseSum.scalar, closeTo(5.0, 1e-6));
      });
    });

    test('Conv2d layer forward and backward', () {
      ResourceScope.scope(() {
        final conv = nn.Conv2d(1, 2, 3, padding: 1);
        final x = GpuArray.ones(
          [1, 1, 4, 4],
          DType.float64,
          requiresGrad: true,
        );

        final out = conv(x);
        expect(out.shape, equals([1, 2, 4, 4]));
        expect(out.requiresGrad, isTrue);

        final loss = out.sum();
        loss.backward();

        expect(conv.weight.grad, isNotNull);
        expect(conv.weight.grad!.shape, equals([2, 1, 3, 3]));
        expect(conv.bias!.grad, isNotNull);
        expect(conv.bias!.grad!.shape, equals([2]));
        expect(x.grad, isNotNull);
        expect(x.grad!.shape, equals([1, 1, 4, 4]));
      });
    });

    test('LayerNorm, Dropout, and Module.train(mode:) propagation', () {
      ResourceScope.scope(() {
        final ln = nn.LayerNorm([4]);
        final x = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [1, 4],
          DType.float64,
          requiresGrad: true,
        );

        final out = ln(x);
        expect(out.shape, equals([1, 4]));

        final outList = out.toList().cast<double>();
        var sum = 0.0;
        for (final v in outList) {
          sum += v;
        }
        expect(sum / 4.0, closeTo(0.0, 1e-4)); // Normalized mean is 0

        final loss = out.sum();
        loss.backward();
        expect(ln.weight.grad, isNotNull);
        expect(ln.bias.grad, isNotNull);
        expect(x.grad, isNotNull);

        final drop = nn.Dropout(p: 0.5);
        final seq = nn.Sequential([ln, drop]);
        expect(seq.training, isTrue);
        expect(drop.training, isTrue);

        seq.train(mode: false);
        expect(seq.training, isFalse);
        expect(drop.training, isFalse);
        final dropEvalOut = drop(x);
        expect(dropEvalOut.toList(), equals(x.toList()));

        seq.train();
        expect(seq.training, isTrue);
        expect(drop.training, isTrue);
      });
    });

    test('Activations & Softmax with optional out: destination tensor', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList(
          [-2.0, 0.0, 2.0],
          [3],
          DType.float64,
          requiresGrad: true,
        );
        final outBuffer = GpuArray<Float64>.empty([3], DType.float64);

        final reluOut = nn.relu(x, out: outBuffer);
        expect(identical(reluOut, outBuffer), isTrue);
        expect(reluOut.toList(), equals([0.0, 0.0, 2.0]));
        expect(reluOut.requiresGrad, isTrue);
        reluOut.sum().backward();
        expect(x.grad!.toList().cast<double>(), equals([0.0, 0.0, 1.0]));

        final smOut = GpuArray<Float64>.empty([3], DType.float64);
        final sm = nn.softmax(x, out: smOut);
        expect(identical(sm, smOut), isTrue);
        final smList = sm.toList().cast<double>();
        var sum = 0.0;
        for (final v in smList) {
          sum += v;
        }
        expect(sum, closeTo(1.0, 1e-4));

        final lsmOut = GpuArray<Float64>.empty([3], DType.float64);
        final lsm = nn.logSoftmax(x, out: lsmOut);
        expect(identical(lsm, lsmOut), isTrue);

        final wrongShapeOut = GpuArray<Float64>.empty([2], DType.float64);
        expect(() => nn.relu(x, out: wrongShapeOut), throwsArgumentError);
      });
    });

    test(
      'Optimizer state retention, Nesterov SGD, and disposal StateError',
      () {
        ResourceScope.scope(() {
          final param = GpuArray.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          final opt = nn.AdamW([param], lr: 0.1);
          expect(opt.isDisposed, isFalse);

          for (var step = 0; step < 5; step++) {
            opt.zeroGrad();
            final loss = (param * 2.0).sum();
            loss.backward();
            opt.step();
          }

          expect(param.toList().cast<double>().first, lessThan(1.0));
          opt.dispose();
          expect(opt.isDisposed, isTrue);
          expect(() => opt.step(), throwsStateError);
          expect(() => opt.zeroGrad(), throwsStateError);

          // Test SGD with momentum and Nesterov
          final sgdParam = GpuArray.fromList(
            [2.0, -2.0],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          final sgd = nn.SGD(
            [sgdParam],
            lr: 0.1,
            momentum: 0.9,
            weightDecay: 0.01,
            nesterov: true,
          );
          for (var i = 0; i < 3; i++) {
            sgd.zeroGrad();
            (sgdParam * sgdParam).sum().backward();
            sgd.step();
          }
          expect(sgdParam.toList().cast<double>().first, lessThan(2.0));
          sgd.dispose();
        });
      },
    );

    test('Precondition validation on layers and optimizers', () {
      ResourceScope.scope(() {
        expect(() => nn.Linear(0, 2), throwsRangeError);
        expect(() => nn.Conv2d(1, 2, 0), throwsArgumentError);
        expect(() => nn.Dropout(p: 1.5), throwsArgumentError);
        expect(() => nn.Adam([], lr: -0.01), throwsArgumentError);
        expect(
          () => nn.SGD([], lr: 0.1, momentum: 0.0, nesterov: true),
          throwsArgumentError,
        );

        final emb = nn.Embedding(3, 2);
        final badIndices = GpuArray.fromList([0, 5], [2], DType.int32);
        expect(() => emb(badIndices), throwsRangeError);
      });
    });

    test(
      'BatchNorm1d (2D & 3D, train/eval), L1Loss, BCELoss, and Float32 Dropout',
      () {
        ResourceScope.scope(() {
          final bn = nn.BatchNorm1d(3);
          final x2d = GpuArray.fromList(
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
            [2, 3],
            DType.float64,
            requiresGrad: true,
          );
          final outTrain = bn(x2d);
          expect(outTrain.shape, equals([2, 3]));
          final colMeans = (outTrain.sum(axis: 0) * 0.5)
              .toList()
              .cast<double>();
          for (final m in colMeans) {
            expect(m, closeTo(0.0, 1e-4));
          }
          outTrain.sum().backward();
          expect(x2d.grad, isNotNull);
          expect(bn.weight!.grad, isNotNull);
          expect(bn.bias!.grad, isNotNull);

          bn.eval();
          final outEval = bn(x2d);
          expect(outEval.shape, equals([2, 3]));

          final x3d = GpuArray.ones([2, 3, 4], DType.float64);
          expect(bn(x3d).shape, equals([2, 3, 4]));

          // L1Loss & l1Loss
          final pred = GpuArray.fromList(
            [1.0, 4.0],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          final target = GpuArray.fromList([2.0, 1.0], [2], DType.float64);
          final l1 = const nn.L1Loss()(pred, target);
          expect((l1.scalar as num).toDouble(), closeTo(2.0, 1e-5));
          l1.backward();
          expect(pred.grad!.toList().cast<double>(), equals([-0.5, 0.5]));

          // BCELoss & binaryCrossEntropy
          final probs = GpuArray.fromList(
            [0.8, 0.2],
            [2],
            DType.float64,
            requiresGrad: true,
          );
          final bceTarget = GpuArray.fromList([1.0, 0.0], [2], DType.float64);
          final bce = const nn.BCELoss()(probs, bceTarget);
          expect((bce.scalar as num).toDouble(), closeTo(0.2231435, 1e-4));
          bce.backward();
          expect(probs.grad, isNotNull);

          // MSELoss & CrossEntropyLoss classes
          expect(
            (const nn.MSELoss()(pred, target).scalar as num).toDouble(),
            closeTo(5.0, 1e-5),
          );
          final ceCriterion = const nn.CrossEntropyLoss();
          final logits = GpuArray.ones([2, 3], DType.float64);
          final labels = GpuArray.fromList([0, 1], [2], DType.int32);
          expect(ceCriterion(logits, labels).rank, equals(0));

          // Float32 Dropout
          final f32Input = GpuArray.ones([8, 8], DType.float32);
          final drop32 = nn.Dropout(p: 0.25)(f32Input);
          expect(drop32.dtype, equals(DType.float32));
          expect(drop32.shape, equals([8, 8]));
        });
      },
    );
  });
}
