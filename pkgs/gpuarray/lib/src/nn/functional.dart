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
import 'dart:math' as math;

import '../autograd/autograd.dart';
import '../backend/compute_engine.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import '../random/random.dart' as random_ops;

export '../autograd/autograd.dart' show LossReduction;

/// Copies [computed] into [out] if provided, validating shape, dtype, and disposal state.
GpuArray<T> _finalizeOutput<T extends DTypeTag>(
  GpuArray<T> computed,
  GpuArray<T>? out,
) {
  if (out == null || identical(computed, out)) {
    return computed;
  }
  if (out.isDisposed) {
    computed.dispose();
    throw StateError('Cannot write into a disposed GpuArray out tensor.');
  }
  if (out.size > 1 && out.strides.contains(0)) {
    computed.dispose();
    throw ArgumentError.value(
      out,
      'out',
      'Must be writeable and not a broadcasted view.',
    );
  }
  if (!ShapeUtils.areEqual(out.shape, computed.shape)) {
    final computedShape = computed.shape;
    computed.dispose();
    throw ArgumentError.value(
      out.shape,
      'out',
      'Must match output shape $computedShape.',
    );
  }
  if (out.dtype != computed.dtype) {
    final computedDType = computed.dtype;
    computed.dispose();
    throw ArgumentError.value(
      out.dtype,
      'out',
      'Must match output dtype $computedDType.',
    );
  }
  if (!out.isContiguous || out.offsetElements != 0) {
    computed.dispose();
    throw ArgumentError.value(
      out,
      'out',
      'Must be a contiguous tensor with zero offset.',
    );
  }

  computed.buffer.copyToBuffer(out.buffer, out.byteSize);
  out.requiresGrad = computed.requiresGrad;
  final gradFunction = computed.gradFn;
  if (gradFunction is SigmoidBackward) {
    out.gradFn = SigmoidBackward(gradFunction.input, out);
  } else if (gradFunction is TanhBackward) {
    out.gradFn = TanhBackward(gradFunction.input, out);
  } else if (gradFunction is SoftmaxBackward) {
    out.gradFn = SoftmaxBackward(
      gradFunction.input,
      out,
      axis: gradFunction.axis,
    );
  } else if (gradFunction is LogSoftmaxBackward) {
    out.gradFn = LogSoftmaxBackward(
      gradFunction.input,
      out,
      axis: gradFunction.axis,
    );
  } else {
    out.gradFn = gradFunction;
  }
  computed.dispose();
  return out;
}

/// Applies the Rectified Linear Unit activation elementwise: $\text{ReLU}(x) = \max(0, x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> relu<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) {
  final computed = noGrad(() {
    final positiveMask = input.greater(0.0);
    final mask = positiveMask.astype(input.dtype);
    positiveMask.dispose();
    final result = (input * mask) as GpuArray<T>;
    mask.dispose();
    return result;
  });
  if (isGradEnabled && input.requiresGrad) {
    computed.requiresGrad = true;
    computed.gradFn = ReluBackward(input);
  }
  return _finalizeOutput(computed, out);
}

/// Applies the logistic Sigmoid activation elementwise: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> sigmoid<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) {
  final computed = noGrad(() {
    final negated = input.negate();
    final expNegated = negated.exp();
    negated.dispose();
    final denominator = expNegated + 1.0;
    expNegated.dispose();
    final result = denominator.pow(-1.0) as GpuArray<T>;
    denominator.dispose();
    return result;
  });
  if (isGradEnabled && input.requiresGrad) {
    computed.requiresGrad = true;
    computed.gradFn = SigmoidBackward(input, computed);
  }
  return _finalizeOutput(computed, out);
}

/// Applies the Hyperbolic Tangent activation elementwise: $\tanh(x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> tanh<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) {
  final computed = noGrad(() => input.tanh());
  if (isGradEnabled && input.requiresGrad) {
    computed.requiresGrad = true;
    computed.gradFn = TanhBackward(input, computed);
  }
  return _finalizeOutput(computed, out);
}

/// Applies the Gaussian Error Linear Unit (GELU) activation elementwise:
/// $\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right)\right)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> gelu<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) {
  final sqrt2OverPi = math.sqrt(2.0 / math.pi);
  final xSquared = input * input;
  final xCubed = xSquared * input;
  final scaledCubic = xCubed * 0.044715;
  final sumTerm = input + scaledCubic;
  final inner = (sumTerm * sqrt2OverPi) as GpuArray<T>;
  final tanhInner = tanh(inner);
  final onePlusTanh = tanhInner + 1.0;
  final halfInput = input * 0.5;
  final computed = (halfInput * onePlusTanh) as GpuArray<T>;
  return _finalizeOutput(computed, out);
}

/// Applies the Sigmoid Linear Unit (SiLU / Swish) activation elementwise:
/// $\text{SiLU}(x) = x \cdot \sigma(x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> silu<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) {
  final sigmoidVal = sigmoid(input);
  final computed = (input * sigmoidVal) as GpuArray<T>;
  return _finalizeOutput(computed, out);
}

/// Applies the Softmax function to an N-dimensional tensor along [axis].
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> softmax<T extends DTypeTag>(
  GpuArray<T> input, {
  int axis = -1,
  GpuArray<T>? out,
}) {
  final computed = noGrad(() {
    final maxVal = input.max(axis: axis, keepDims: true);
    final shifted = input - maxVal;
    maxVal.dispose();
    final expShifted = shifted.exp();
    shifted.dispose();
    final sumExp = expShifted.sum(axis: axis, keepDims: true);
    final result = (expShifted / sumExp) as GpuArray<T>;
    expShifted.dispose();
    sumExp.dispose();
    return result;
  });
  if (isGradEnabled && input.requiresGrad) {
    computed.requiresGrad = true;
    computed.gradFn = SoftmaxBackward(input, computed, axis: axis);
  }
  return _finalizeOutput(computed, out);
}

/// Applies the Log-Softmax function to an N-dimensional tensor along [axis].
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> logSoftmax<T extends DTypeTag>(
  GpuArray<T> input, {
  int axis = -1,
  GpuArray<T>? out,
}) {
  final computed = noGrad(() {
    final maxVal = input.max(axis: axis, keepDims: true);
    final shifted = input - maxVal;
    maxVal.dispose();
    final expShifted = shifted.exp();
    final sumExp = expShifted.sum(axis: axis, keepDims: true);
    expShifted.dispose();
    final logSumExp = sumExp.log();
    sumExp.dispose();
    final result = (shifted - logSumExp) as GpuArray<T>;
    shifted.dispose();
    logSumExp.dispose();
    return result;
  });
  if (isGradEnabled && input.requiresGrad) {
    computed.requiresGrad = true;
    computed.gradFn = LogSoftmaxBackward(input, computed, axis: axis);
  }
  return _finalizeOutput(computed, out);
}

/// Applies the Log-Softmax function to an N-dimensional tensor along [axis].
///
/// Alias for [logSoftmax] provided for PyTorch naming parity.
GpuArray<T> log_softmax<T extends DTypeTag>(
  GpuArray<T> input, {
  int axis = -1,
  GpuArray<T>? out,
}) => logSoftmax(input, axis: axis, out: out);

/// Measures the Mean Squared Error (squared L2 norm) between [input] and [target].
GpuArray<DTypeTag> mseLoss<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T> target, {
  LossReduction reduction = LossReduction.mean,
}) {
  final difference = input - target;
  final squared = difference * difference;
  switch (reduction) {
    case LossReduction.mean:
      return squared.mean();
    case LossReduction.sum:
      return squared.sum();
    case LossReduction.none:
      return squared;
  }
}

/// Measures the Mean Squared Error (squared L2 norm) between [input] and [target].
///
/// Alias for [mseLoss] provided for PyTorch naming parity.
GpuArray<DTypeTag> mse_loss<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T> target, {
  LossReduction reduction = LossReduction.mean,
}) => mseLoss(input, target, reduction: reduction);

/// Computes the categorical cross-entropy loss between [logits] (`[N, C]`) and
/// integer class [targets] (`[N]`).
///
/// Each target class index in [targets] must lie in `0 <= target < C`.
GpuArray<T> crossEntropy<T extends DTypeTag>(
  GpuArray<T> logits,
  GpuArray<DTypeTag> targets, {
  LossReduction reduction = LossReduction.mean,
}) {
  if (logits.rank < 2) {
    throw ArgumentError.value(
      logits.shape,
      'logits',
      'Must have rank at least 2 ([batchSize, numClasses]).',
    );
  }
  final numSamples = logits.shape[0];
  final numClasses = logits.shape[logits.rank - 1];
  if (targets.size != numSamples) {
    throw ArgumentError.value(
      targets.shape,
      'targets',
      'Must contain $numSamples target class indices.',
    );
  }

  final trackGrad = isGradEnabled && logits.requiresGrad;
  final GpuArray<T>? probabilities = trackGrad
      ? noGrad(() => softmax(logits, axis: -1))
      : null;
  final logProbabilities = noGrad(() => logSoftmax(logits, axis: -1));

  final GpuArray<T> lossArray;
  try {
    switch (reduction) {
      case LossReduction.mean:
      case LossReduction.sum:
        var totalLoss = 0.0;
        for (var i = 0; i < numSamples; i++) {
          final targetClass = ComputeEngine.readValue(
            targets.buffer,
            targets.dtype,
            i,
            offsetElements: targets.offsetElements,
          ).toInt();
          RangeError.checkValueInInterval(
            targetClass,
            0,
            numClasses - 1,
            'targets',
          );
          final logProb = ComputeEngine.readValue(
            logProbabilities.buffer,
            logProbabilities.dtype,
            i * numClasses + targetClass,
            offsetElements: logProbabilities.offsetElements,
          );
          totalLoss -= logProb;
        }
        final reducedValue = reduction == LossReduction.mean
            ? totalLoss / numSamples
            : totalLoss;
        lossArray = GpuArray.filled(
          [],
          reducedValue,
          logits.dtype,
          device: logits.device,
        );
      case LossReduction.none:
        lossArray = GpuArray.empty(
          [numSamples],
          logits.dtype,
          device: logits.device,
        );
        for (var i = 0; i < numSamples; i++) {
          final targetClass = ComputeEngine.readValue(
            targets.buffer,
            targets.dtype,
            i,
            offsetElements: targets.offsetElements,
          ).toInt();
          RangeError.checkValueInInterval(
            targetClass,
            0,
            numClasses - 1,
            'targets',
          );
          final logProb = ComputeEngine.readValue(
            logProbabilities.buffer,
            logProbabilities.dtype,
            i * numClasses + targetClass,
            offsetElements: logProbabilities.offsetElements,
          );
          ComputeEngine.writeValue(
            lossArray.buffer,
            lossArray.dtype,
            i,
            -logProb,
          );
        }
    }
  } finally {
    logProbabilities.dispose();
  }

  if (trackGrad && probabilities != null) {
    lossArray.requiresGrad = true;
    lossArray.gradFn = CrossEntropyBackward(
      logits,
      targets,
      probabilities,
      reduction: reduction,
    );
  }

  return lossArray;
}

/// Computes the categorical cross-entropy loss between [logits] and [targets].
///
/// Alias for [crossEntropy] provided for PyTorch naming parity.
GpuArray<T> cross_entropy<T extends DTypeTag>(
  GpuArray<T> logits,
  GpuArray<DTypeTag> targets, {
  LossReduction reduction = LossReduction.mean,
}) => crossEntropy(logits, targets, reduction: reduction);

/// Computes Scaled Dot-Product Attention (SDPA):
/// $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
///
/// Supports batched queries, keys, and values (e.g. `[B, H, N, D]` or `[N, D]`),
/// causal upper-triangular masking when [isCausal] is `true`, custom boolean or
/// additive float [attnMask], dropout probability [dropoutP], and custom [scale].
GpuArray<T> scaledDotProductAttention<T extends DTypeTag>(
  GpuArray<T> query,
  GpuArray<T> key,
  GpuArray<T> value, {
  GpuArray<DTypeTag>? attnMask,
  double dropoutP = 0.0,
  bool isCausal = false,
  double? scale,
}) {
  if (dropoutP < 0.0 || dropoutP >= 1.0) {
    throw ArgumentError.value(
      dropoutP,
      'dropoutP',
      'Must be in the half-open interval [0.0, 1.0).',
    );
  }
  final keyDimension = query.shape[query.rank - 1];
  final scaleFactor = scale ?? (1.0 / math.sqrt(keyDimension));

  final keyTransposed = key.swapaxes(-1, -2);
  var scores = (query.matmul(keyTransposed) * scaleFactor) as GpuArray<T>;

  final querySeqLength = query.shape[query.rank - 2];
  final keySeqLength = key.shape[key.rank - 2];

  if (isCausal) {
    final causalMask = GpuArray.empty(
      [querySeqLength, keySeqLength],
      scores.dtype,
      device: scores.device,
    );
    for (var row = 0; row < querySeqLength; row++) {
      for (var col = 0; col < keySeqLength; col++) {
        ComputeEngine.writeValue(
          causalMask.buffer,
          causalMask.dtype,
          row * keySeqLength + col,
          col > row ? -1e9 : 0.0,
        );
      }
    }
    scores = (scores + causalMask) as GpuArray<T>;
  }

  if (attnMask != null) {
    if (attnMask.dtype == DType.boolean) {
      final additiveMask = GpuArray.empty(
        attnMask.shape,
        scores.dtype,
        device: scores.device,
      );
      final totalMaskElements = attnMask.size;
      for (var i = 0; i < totalMaskElements; i++) {
        final rawValue = ComputeEngine.readAny(
          attnMask.buffer,
          attnMask.dtype,
          i,
          offsetElements: attnMask.offsetElements,
        );
        final keep = rawValue == true || (rawValue is num && rawValue != 0);
        ComputeEngine.writeValue(
          additiveMask.buffer,
          additiveMask.dtype,
          i,
          keep ? 0.0 : -1e9,
        );
      }
      scores = (scores + additiveMask) as GpuArray<T>;
    } else {
      scores = (scores + attnMask) as GpuArray<T>;
    }
  }

  var attentionWeights = softmax(scores, axis: -1);

  if (dropoutP > 0.0) {
    final randomValues = random_ops.rand(
      attentionWeights.shape,
      attentionWeights.device,
    );
    final keepBoolean = randomValues.greater(dropoutP);
    final keepMask = keepBoolean.astype(attentionWeights.dtype);
    randomValues.dispose();
    keepBoolean.dispose();
    final keepScale = 1.0 / (1.0 - dropoutP);
    attentionWeights = (attentionWeights * keepMask * keepScale) as GpuArray<T>;
  }

  return attentionWeights.matmul(value) as GpuArray<T>;
}

/// Computes Scaled Dot-Product Attention (SDPA).
///
/// Alias for [scaledDotProductAttention] provided for PyTorch naming parity.
GpuArray<T> scaled_dot_product_attention<T extends DTypeTag>(
  GpuArray<T> query,
  GpuArray<T> key,
  GpuArray<T> value, {
  GpuArray<DTypeTag>? attnMask,
  double dropoutP = 0.0,
  bool isCausal = false,
  double? scale,
}) => scaledDotProductAttention(
  query,
  key,
  value,
  attnMask: attnMask,
  dropoutP: dropoutP,
  isCausal: isCausal,
  scale: scale,
);
