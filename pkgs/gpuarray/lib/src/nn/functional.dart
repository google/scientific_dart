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
import '../autograd/autograd_wgsl.dart';
import '../autograd/loss_wgsl.dart';
import '../gpu_array.dart';
import '../random/random.dart' as random_ops;

export '../autograd/autograd.dart' show LossReduction;
export 'functional_attention.dart';

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
  if (!areShapesIdentical(out.shape, computed.shape)) {
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

/// Randomly zeroes elements of [input] with probability [p] during [training].
GpuArray<T> dropout<T extends DTypeTag>(
  GpuArray<T> input, {
  double p = 0.5,
  bool training = true,
}) {
  if (p < 0.0 || p >= 1.0 || p.isNaN) {
    throw ArgumentError.value(
      p,
      'p',
      'Must be in the half-open interval [0.0, 1.0).',
    );
  }
  if (!training || p == 0.0) return input;
  final randomValues = random_ops.rand(input.shape, input.device);
  final keepBoolean = randomValues.greater(p);
  final keepMask = keepBoolean.astype(input.dtype);
  randomValues.dispose();
  keepBoolean.dispose();
  final scale = 1.0 / (1.0 - p);
  return (input * keepMask * scale) as GpuArray<T>;
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
  if (!areShapesIdentical(input.shape, target.shape)) {
    throw ArgumentError.value(
      target.shape,
      'target',
      'Must match input shape ${input.shape}.',
    );
  }
  final difference = input - target;
  final squared = difference * difference;
  return switch (reduction) {
    LossReduction.mean => squared.mean(),
    LossReduction.sum => squared.sum(),
    LossReduction.none => squared,
  };
}

/// Measures the Mean Squared Error (squared L2 norm) between [input] and [target].
///
/// Alias for [mseLoss] provided for PyTorch naming parity.
GpuArray<DTypeTag> mse_loss<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T> target, {
  LossReduction reduction = LossReduction.mean,
}) => mseLoss(input, target, reduction: reduction);

/// Measures the Mean Absolute Error (L1 norm) between [input] and [target].
GpuArray<DTypeTag> l1Loss<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T> target, {
  LossReduction reduction = LossReduction.mean,
}) {
  if (!areShapesIdentical(input.shape, target.shape)) {
    throw ArgumentError.value(
      target.shape,
      'target',
      'Must match input shape ${input.shape}.',
    );
  }
  final computed = noGrad(() {
    final difference = input - target;
    final absDifference = difference.abs();
    difference.dispose();
    return switch (reduction) {
      LossReduction.mean => () {
        final result = absDifference.mean();
        absDifference.dispose();
        return result;
      }(),
      LossReduction.sum => () {
        final result = absDifference.sum();
        absDifference.dispose();
        return result;
      }(),
      LossReduction.none => absDifference,
    };
  });
  if (isGradEnabled && (input.requiresGrad || target.requiresGrad)) {
    computed.requiresGrad = true;
    computed.gradFn = L1LossBackward(input, target, reduction: reduction);
  }
  return computed;
}

/// Measures the Mean Absolute Error (L1 norm) between [input] and [target].
///
/// Alias for [l1Loss] provided for PyTorch naming parity.
GpuArray<DTypeTag> l1_loss<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T> target, {
  LossReduction reduction = LossReduction.mean,
}) => l1Loss(input, target, reduction: reduction);

/// Measures the Binary Cross-Entropy loss between target probabilities [target] and
/// predicted probabilities [input]:
/// $$\ell(x, y) = -\left(y \ln(x) + (1 - y) \ln(1 - x)\right)$$
GpuArray<DTypeTag> binaryCrossEntropy<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T> target, {
  LossReduction reduction = LossReduction.mean,
}) {
  if (!areShapesIdentical(input.shape, target.shape)) {
    throw ArgumentError.value(
      target.shape,
      'target',
      'Must match input shape ${input.shape}.',
    );
  }
  final computed = noGrad(() {
    final elementLosses = GpuArray.empty(
      input.shape,
      input.dtype,
      device: input.device,
    );
    dispatchBceForward(
      input: input,
      targetTensor: target,
      output: elementLosses,
    );
    return switch (reduction) {
      LossReduction.mean => () {
        final result = elementLosses.mean();
        elementLosses.dispose();
        return result;
      }(),
      LossReduction.sum => () {
        final result = elementLosses.sum();
        elementLosses.dispose();
        return result;
      }(),
      LossReduction.none => elementLosses,
    };
  });
  if (isGradEnabled && input.requiresGrad) {
    computed.requiresGrad = true;
    computed.gradFn = BinaryCrossEntropyBackward(
      input,
      target,
      reduction: reduction,
    );
  }
  return computed;
}

/// Measures the Binary Cross-Entropy loss between [input] and [target].
///
/// Alias for [binaryCrossEntropy] provided for PyTorch naming parity.
GpuArray<DTypeTag> binary_cross_entropy<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T> target, {
  LossReduction reduction = LossReduction.mean,
}) => binaryCrossEntropy(input, target, reduction: reduction);

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
    lossArray = noGrad(() {
      final sampleLosses = GpuArray.empty(
        [numSamples],
        logits.dtype,
        device: logits.device,
      );
      dispatchCrossEntropyForward(
        logProbabilities: logProbabilities,
        targets: targets,
        sampleLosses: sampleLosses,
        numSamples: numSamples,
        numClasses: numClasses,
      );
      return switch (reduction) {
        LossReduction.mean => () {
          final reduced = sampleLosses.mean().astype(logits.dtype);
          sampleLosses.dispose();
          return reduced;
        }(),
        LossReduction.sum => () {
          final reduced = sampleLosses.sum().astype(logits.dtype);
          sampleLosses.dispose();
          return reduced;
        }(),
        LossReduction.none => sampleLosses,
      };
    });
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
