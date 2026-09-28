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

// ignore_for_file: non_constant_identifier_names
/// Example of implementing a Multi-Layer Perceptron (MLP) from scratch
/// using the `ndarray` package.
///
/// This example solves a non-linear classification problem (concentric circles)
/// and showcases memory efficiency by using [NDArray.scope] and [out] parameters
/// to achieve zero-allocation training loop.
library;

import 'dart:math' as math;

import 'package:ndarray/ndarray.dart';

/// Represents a dataset of inputs and labels.
final class Dataset {
  /// Input features of shape [numPoints, 2].
  final NDArray<Float64> x;

  /// Binary labels of shape [numPoints, 1].
  final NDArray<Float64> y;

  /// Creates a dataset with [x] and [y].
  Dataset(this.x, this.y);

  /// Disposes the underlying arrays.
  void dispose() {
    x.dispose();
    y.dispose();
  }
}

/// Generates synthetic concentric circles data.
///
/// Points inside radius are labeled 1, outside are labeled 0.
Dataset generateConcentricCircles(
  int numPoints, {
  double radius = 0.7,
  int? seed,
}) {
  return NDArray.scope(() {
    final x = NDArray<Float64>.create([numPoints, 2], DType.float64);
    uniform([numPoints, 2], dtype: DType.float64, seed: seed, out: x);

    final two = NDArray.scalar(2.0, dtype: DType.float64);
    final one = NDArray<Float64>.scalar(Float64(1.0), dtype: DType.float64);

    // Scale to [-1, 1)
    multiply(x, two, out: x);
    subtract(x, one, out: x);

    // x^2 + y^2
    final xSquared = NDArray<Float64>.create([numPoints, 2], DType.float64);
    multiply(x, x, out: xSquared);

    final d2 = NDArray<Float64>.create([numPoints], DType.float64);
    sum(xSquared, axis: 1, out: d2);

    final r2 = NDArray.scalar(radius * radius, dtype: DType.float64);
    final mask = NDArray<bool>.create([numPoints], DType.boolean);
    less(d2, r2, out: mask);

    final y = NDArray<Float64>.create([numPoints, 1], DType.float64);
    final zero = NDArray<Float64>.scalar(Float64(0.0), dtype: DType.float64);

    final mask2D = mask.reshape([numPoints, 1]);
    where(mask2D, one, zero, y);

    x.detachFromScope();
    y.detachFromScope();

    return Dataset(x, y);
  });
}

/// A simple 2-layer MLP (Multi-Layer Perceptron).
final class MLP {
  /// Dimension of the input features (must be 2 for this example).
  final int inputDim;

  /// Dimension of the hidden layer.
  final int hiddenDim;

  /// Dimension of the output (must be 1 for binary classification).
  final int outputDim;

  /// Batch size used for training.
  final int batchSize;

  /// Learning rate for gradient descent.
  final double learningRate;

  // Parameters
  /// Weights for the first layer, shape [inputDim, hiddenDim].
  late final NDArray<Float64> W1;

  /// Biases for the first layer, shape [1, hiddenDim].
  late final NDArray<Float64> b1;

  /// Weights for the second layer, shape [hiddenDim, outputDim].
  late final NDArray<Float64> W2;

  /// Biases for the second layer, shape [1, outputDim].
  late final NDArray<Float64> b2;

  // Parameter views
  /// Transpose view of W2, shape [outputDim, hiddenDim].
  late final NDArray<Float64> W2_T;

  // Activations (buffers)
  /// Pre-activation of first layer, shape [batchSize, hiddenDim].
  late final NDArray<Float64> Z1;

  /// Activation of first layer, shape [batchSize, hiddenDim].
  late final NDArray<Float64> A1;

  /// Pre-activation of second layer, shape [batchSize, outputDim].
  late final NDArray<Float64> Z2;

  /// Activation of second layer, shape [batchSize, outputDim].
  late final NDArray<Float64> A2;

  // Activation views
  /// Transpose view of A1, shape [hiddenDim, batchSize].
  late final NDArray<Float64> A1_T;

  // Gradients (buffers)
  /// Gradient of Z2, shape [batchSize, outputDim].
  late final NDArray<Float64> dZ2;

  /// Gradient of W2, shape [hiddenDim, outputDim].
  late final NDArray<Float64> dW2;

  /// Gradient of b2, shape [1, outputDim].
  late final NDArray<Float64> db2;

  /// Gradient of A1, shape [batchSize, hiddenDim].
  late final NDArray<Float64> dA1;

  /// Gradient of Z1, shape [batchSize, hiddenDim].
  late final NDArray<Float64> dZ1;

  /// Gradient of W1, shape [inputDim, hiddenDim].
  late final NDArray<Float64> dW1;

  /// Gradient of b1, shape [1, hiddenDim].
  late final NDArray<Float64> db1;

  // Views for sum reductions
  /// 1D view of db2 for sum reduction, shape [outputDim].
  late final NDArray<Float64> db2_1D;

  /// 1D view of db1 for sum reduction, shape [hiddenDim].
  late final NDArray<Float64> db1_1D;

  // Helper constants
  /// 0D array containing 1.0.
  late final NDArray<Float64> one;

  /// 0D array containing 0.0.
  late final NDArray<Float64> zero;

  /// Creates and initializes the MLP.
  MLP({
    required this.inputDim,
    required this.hiddenDim,
    required this.outputDim,
    required this.batchSize,
    required this.learningRate,
  }) {
    // Initialize parameters
    W1 = NDArray<Float64>.create([inputDim, hiddenDim], DType.float64);
    final limit1 = math.sqrt(6.0 / (inputDim + hiddenDim));
    uniform([inputDim, hiddenDim], dtype: DType.float64, out: W1);
    final twoLimit1 = NDArray.scalar(2.0 * limit1, dtype: DType.float64);
    final limit1Arr = NDArray.scalar(limit1, dtype: DType.float64);
    multiply(W1, twoLimit1, out: W1);
    subtract(W1, limit1Arr, out: W1);
    twoLimit1.dispose();
    limit1Arr.dispose();

    b1 = NDArray<Float64>.zeros([1, hiddenDim], DType.float64);

    W2 = NDArray<Float64>.create([hiddenDim, outputDim], DType.float64);
    final limit2 = math.sqrt(6.0 / (hiddenDim + outputDim));
    uniform([hiddenDim, outputDim], dtype: DType.float64, out: W2);
    final twoLimit2 = NDArray.scalar(2.0 * limit2, dtype: DType.float64);
    final limit2Arr = NDArray.scalar(limit2, dtype: DType.float64);
    multiply(W2, twoLimit2, out: W2);
    subtract(W2, limit2Arr, out: W2);
    twoLimit2.dispose();
    limit2Arr.dispose();

    b2 = NDArray<Float64>.zeros([1, outputDim], DType.float64);

    W2_T = W2.transposed;

    // Allocate activations
    Z1 = NDArray<Float64>.create([batchSize, hiddenDim], DType.float64);
    A1 = NDArray<Float64>.create([batchSize, hiddenDim], DType.float64);
    Z2 = NDArray<Float64>.create([batchSize, outputDim], DType.float64);
    A2 = NDArray<Float64>.create([batchSize, outputDim], DType.float64);

    A1_T = A1.transposed;

    // Allocate gradients
    dZ2 = NDArray<Float64>.create([batchSize, outputDim], DType.float64);
    dW2 = NDArray<Float64>.create([hiddenDim, outputDim], DType.float64);
    db2 = NDArray<Float64>.create([1, outputDim], DType.float64);
    dA1 = NDArray<Float64>.create([batchSize, hiddenDim], DType.float64);
    dZ1 = NDArray<Float64>.create([batchSize, hiddenDim], DType.float64);
    dW1 = NDArray<Float64>.create([inputDim, hiddenDim], DType.float64);
    db1 = NDArray<Float64>.create([1, hiddenDim], DType.float64);

    db2_1D = db2.reshape([outputDim]);
    db1_1D = db1.reshape([hiddenDim]);

    one = NDArray<Float64>.scalar(Float64(1.0), dtype: DType.float64);
    zero = NDArray<Float64>.scalar(Float64(0.0), dtype: DType.float64);
  }

  /// Disposes all allocated arrays and views.
  void dispose() {
    W1.dispose();
    b1.dispose();
    W2.dispose();
    b2.dispose();
    W2_T.dispose();

    Z1.dispose();
    A1.dispose();
    Z2.dispose();
    A2.dispose();
    A1_T.dispose();

    dZ2.dispose();
    dW2.dispose();
    db2.dispose();
    dA1.dispose();
    dZ1.dispose();
    dW1.dispose();
    db1.dispose();

    db2_1D.dispose();
    db1_1D.dispose();

    one.dispose();
    zero.dispose();
  }

  /// Performs the forward pass.
  ///
  /// Computes activations [Z1], [A1], [Z2], [A2] from input [X].
  void forward(NDArray<Float64> X) {
    matmul(X, W1, out: Z1);
    add(Z1, b1, out: Z1);
    _sigmoid(Z1, A1);

    matmul(A1, W2, out: Z2);
    add(Z2, b2, out: Z2);
    _sigmoid(Z2, A2);
  }

  /// Performs the backward pass.
  ///
  /// Computes gradients [dZ2], [dW2], [db2], [dA1], [dZ1], [dW1], [db1]
  /// using input [X], target [Y], and pre-transposed input [X_T].
  void backward(NDArray<Float64> X, NDArray<Float64> Y, NDArray<Float64> X_T) {
    subtract(A2, Y, out: dZ2);

    matmul(A1_T, dZ2, out: dW2);
    sum(dZ2, axis: 0, out: db2_1D);

    matmul(dZ2, W2_T, out: dA1);

    _sigmoidDeriv(A1, dZ1);
    multiply(dA1, dZ1, out: dZ1);

    matmul(X_T, dZ1, out: dW1);
    sum(dZ1, axis: 0, out: db1_1D);
  }

  /// Updates parameters W1, b1, W2, b2 using gradient descent.
  ///
  /// Scales gradients by `1 / batchSize` before update.
  void update() {
    final lrScaledVal = learningRate / batchSize;
    final lrScaled = NDArray.scalar(lrScaledVal, dtype: DType.float64);

    multiply(dW1, lrScaled, out: dW1);
    subtract(W1, dW1, out: W1);

    multiply(db1, lrScaled, out: db1);
    subtract(b1, db1, out: b1);

    multiply(dW2, lrScaled, out: dW2);
    subtract(W2, dW2, out: W2);

    multiply(db2, lrScaled, out: db2);
    subtract(b2, db2, out: b2);

    lrScaled.dispose();
  }

  void _sigmoid(NDArray<Float64> x, NDArray<Float64> out) {
    negative(x, out: out);
    exp(out, out: out);
    add(one, out, out: out);
    divide(one, out, out: out);
  }

  void _sigmoidDeriv(NDArray<Float64> a, NDArray<Float64> out) {
    subtract(one, a, out: out);
    multiply(a, out, out: out);
  }
}

/// Calculates Binary Cross Entropy loss.
double calculateLoss(NDArray<Float64> A2, NDArray<Float64> Y) {
  return NDArray.scope(() {
    final logA2 = NDArray<Float64>.create(A2.shape, DType.float64);
    clip(A2, min: 1e-15, max: 1.0 - 1e-15, out: logA2);
    log(logA2, out: logA2);

    final oneMinusA2 = NDArray<Float64>.create(A2.shape, DType.float64);
    final one = NDArray<Float64>.scalar(Float64(1.0), dtype: DType.float64);
    subtract(one, A2, out: oneMinusA2);
    clip(oneMinusA2, min: 1e-15, max: 1.0 - 1e-15, out: oneMinusA2);
    log(oneMinusA2, out: oneMinusA2);

    final term1 = NDArray<Float64>.create(A2.shape, DType.float64);
    multiply(Y, logA2, out: term1);

    final oneMinusY = NDArray<Float64>.create(Y.shape, DType.float64);
    subtract(one, Y, out: oneMinusY);

    final term2 = NDArray<Float64>.create(A2.shape, DType.float64);
    multiply(oneMinusY, oneMinusA2, out: term2);

    final sumTerms = NDArray<Float64>.create(A2.shape, DType.float64);
    add(term1, term2, out: sumTerms);

    final totalSum = sum(sumTerms);

    final lossVal = -totalSum.scalar / A2.shape[0];

    return lossVal;
  });
}

/// Calculates classification accuracy.
double calculateAccuracy(NDArray<Float64> A2, NDArray<Float64> Y) {
  return NDArray.scope(() {
    final threshold = NDArray.scalar(0.5, dtype: DType.float64);
    final predictions = NDArray<bool>.create(A2.shape, DType.boolean);
    greater(A2, threshold, out: predictions);

    final yBool = NDArray<bool>.create(Y.shape, DType.boolean);
    greater(Y, threshold, out: yBool);

    final correct = NDArray<bool>.create(A2.shape, DType.boolean);
    equal(predictions, yBool, out: correct);

    final correctDouble = NDArray<Float64>.create(A2.shape, DType.float64);
    final one = NDArray<Float64>.scalar(Float64(1.0), dtype: DType.float64);
    final zero = NDArray<Float64>.scalar(Float64(0.0), dtype: DType.float64);
    where(correct, one, zero, correctDouble);

    final correctSum = sum(correctDouble);
    final acc = correctSum.scalar / A2.shape[0];

    return acc;
  });
}

void main() {
  const numPoints = 200;
  const hiddenDim = 8;
  const epochs = 5000;
  const learningRate = 2.0;

  print('Generating synthetic concentric circles dataset...');
  final dataset = generateConcentricCircles(numPoints, seed: 42);
  print(
    'Dataset generated. X shape: ${dataset.x.shape}, Y shape: ${dataset.y.shape}',
  );

  // Split into train and test (50/50)
  const trainSize = numPoints ~/ 2;

  // We use views for train/test split
  final xTrain = dataset.x.slice([
    const Slice(start: 0, stop: trainSize),
    const Slice.all(),
  ]);
  final yTrain = dataset.y.slice([
    const Slice(start: 0, stop: trainSize),
    const Slice.all(),
  ]);
  final xTest = dataset.x.slice([
    const Slice(start: trainSize, stop: numPoints),
    const Slice.all(),
  ]);
  final yTest = dataset.y.slice([
    const Slice(start: trainSize, stop: numPoints),
    const Slice.all(),
  ]);

  final xTrain_T = xTrain.transposed;

  print('Initializing MLP...');
  final mlp = MLP(
    inputDim: 2,
    hiddenDim: hiddenDim,
    outputDim: 1,
    batchSize: trainSize,
    learningRate: learningRate,
  );

  print('Starting training loop...');
  print('Epoch\tLoss\tTrain Acc\tTest Acc');

  for (var epoch = 1; epoch <= epochs; epoch++) {
    // We run the training step inside a scope to ensure any temporary allocations are freed.
    // Since we use out: parameters, there should be almost zero allocations,
    // but scope is good for safety.
    NDArray.scope(() {
      mlp.forward(xTrain);
      mlp.backward(xTrain, yTrain, xTrain_T);
      mlp.update();
    });

    if (epoch % 100 == 0 || epoch == 1) {
      NDArray.scope(() {
        // Evaluate on train
        mlp.forward(xTrain);
        final trainLoss = calculateLoss(mlp.A2, yTrain);
        final trainAcc = calculateAccuracy(mlp.A2, yTrain);

        // Evaluate on test
        mlp.forward(xTest);
        final testAcc = calculateAccuracy(mlp.A2, yTest);

        print(
          '$epoch\t${trainLoss.toStringAsFixed(6)}\t${(trainAcc * 100).toStringAsFixed(2)}%\t\t${(testAcc * 100).toStringAsFixed(2)}%',
        );
      });
    }
  }

  // Final verification
  NDArray.scope(() {
    mlp.forward(xTest);
    final finalTestAcc = calculateAccuracy(mlp.A2, yTest);
    print('\nFinal Test Accuracy: ${(finalTestAcc * 100).toStringAsFixed(2)}%');
    if (finalTestAcc > 0.85) {
      print('🏆 Training SUCCESSFUL! Accuracy is above 85%.');
    } else {
      print('❌ Training FAILED. Accuracy is too low.');
    }
  });

  // Clean up
  mlp.dispose();
  xTrain.dispose();
  yTrain.dispose();
  xTest.dispose();
  yTest.dispose();
  xTrain_T.dispose();
  dataset.dispose();
}
