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

import '../autograd/autograd.dart';
import '../autograd/autograd_wgsl.dart';
import '../autograd/loss_wgsl.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import '../random/random.dart' as random_ops;
import 'nn_wgsl.dart';

export '../autograd/autograd.dart' show LossReduction;
export 'functional_attention.dart';

void _validateOutTensor<T extends DTypeTag>(
  GpuArray<T> input,
  GpuArray<T>? out,
) {
  if (out == null) return;
  if (out.isDisposed) {
    throw StateError('Cannot write into a disposed GpuArray out tensor.');
  }
  if (out.size > 1 && out.strides.contains(0)) {
    throw ArgumentError.value(
      out,
      'out',
      'Must be writeable and not a broadcasted view.',
    );
  }
  if (!identical(out.device, input.device)) {
    throw ArgumentError.value(
      out.device,
      'out',
      'Must reside on the same GpuDevice as input.',
    );
  }
  if (!areShapesIdentical(out.shape, input.shape)) {
    throw ArgumentError.value(
      out.shape,
      'out',
      'Must match output shape ${input.shape}.',
    );
  }
  if (out.dtype != input.dtype) {
    throw ArgumentError.value(
      out.dtype,
      'out',
      'Must match output dtype ${input.dtype}.',
    );
  }
  if (!out.isContiguous || out.offsetElements != 0) {
    throw ArgumentError.value(
      out,
      'out',
      'Must be a contiguous tensor with zero offset.',
    );
  }
}

/// Copies [computed] into [out] if provided, validating shape, dtype, and disposal state.
GpuArray<T> _finalizeOutput<T extends DTypeTag>(
  GpuArray<T> computed,
  GpuArray<T>? out,
) {
  if (out == null || identical(computed, out)) {
    return computed;
  }
  try {
    _validateOutTensor(computed, out);
  } on Object {
    computed.dispose();
    rethrow;
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

GpuArray<T> _runUnaryActivation<T extends DTypeTag>({
  required GpuArray<T> input,
  required String op,
  required GradFn Function(GpuArray<T> savedInput, GpuArray<T> output)
  buildGradFn,
  double param0 = 0.0,
  double param1 = 0.0,
  GpuArray<T>? out,
}) {
  if (input.isDisposed) {
    throw StateError('Cannot operate on a disposed GpuArray.');
  }
  _validateOutTensor(input, out);

  final trackGrad = isGradEnabled && input.requiresGrad;
  final savedInput =
      (trackGrad && out != null && identical(input.buffer, out.buffer))
      ? noGrad(() => input.copy())
      : input;

  final target =
      out ?? GpuArray.empty(input.shape, input.dtype, device: input.device);
  dispatchUnaryActivationForward(
    input: savedInput,
    output: target,
    op: op,
    param0: param0,
    param1: param1,
  );

  if (trackGrad) {
    target.requiresGrad = true;
    target.gradFn = buildGradFn(savedInput, target);
  } else if (out != null) {
    target.requiresGrad = false;
    target.gradFn = null;
  }
  return target;
}

/// Applies the Rectified Linear Unit activation elementwise: $\text{ReLU}(x) = \max(0, x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> relu<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) =>
    _runUnaryActivation(
      input: input,
      op: 'relu',
      buildGradFn: (saved, _) => ReluBackward(saved),
      out: out,
    );

/// Applies the logistic Sigmoid activation elementwise: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> sigmoid<T extends DTypeTag>(
  GpuArray<T> input, {
  GpuArray<T>? out,
}) => _runUnaryActivation(
  input: input,
  op: 'sigmoid',
  buildGradFn: (saved, output) => SigmoidBackward(saved, output),
  out: out,
);

/// Applies the Hyperbolic Tangent activation elementwise: $\tanh(x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> tanh<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) =>
    _runUnaryActivation(
      input: input,
      op: 'tanh',
      buildGradFn: (saved, output) => TanhBackward(saved, output),
      out: out,
    );

/// Applies the Gaussian Error Linear Unit (GELU) activation elementwise:
/// $\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right)\right)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> gelu<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) =>
    _runUnaryActivation(
      input: input,
      op: 'gelu',
      buildGradFn: (saved, _) => GeluBackward(saved),
      out: out,
    );

/// Applies the Sigmoid Linear Unit (SiLU / Swish) activation elementwise:
/// $\text{SiLU}(x) = x \cdot \sigma(x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> silu<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) =>
    _runUnaryActivation(
      input: input,
      op: 'silu',
      buildGradFn: (saved, _) => SiluBackward(saved),
      out: out,
    );

/// Applies the Swish activation elementwise: $\text{Swish}(x) = x \cdot \sigma(x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> swish<T extends DTypeTag>(GpuArray<T> input, {GpuArray<T>? out}) =>
    silu(input, out: out);

/// Applies the Leaky Rectified Linear Unit activation elementwise:
/// $\text{LeakyReLU}(x) = \max(0, x) + \text{negativeSlope} \cdot \min(0, x)$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> leakyRelu<T extends DTypeTag>(
  GpuArray<T> input, {
  double negativeSlope = 0.01,
  GpuArray<T>? out,
}) {
  if (negativeSlope.isNaN) {
    throw ArgumentError.value(
      negativeSlope,
      'negativeSlope',
      'Must not be NaN.',
    );
  }
  return _runUnaryActivation(
    input: input,
    op: 'leaky_relu',
    param0: negativeSlope,
    buildGradFn: (saved, _) =>
        LeakyReluBackward(saved, negativeSlope: negativeSlope),
    out: out,
  );
}

/// Applies the Exponential Linear Unit (ELU) activation elementwise:
/// $\text{ELU}(x) = \max(0, x) + \min(0, \alpha (\exp(x) - 1))$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> elu<T extends DTypeTag>(
  GpuArray<T> input, {
  double alpha = 1.0,
  GpuArray<T>? out,
}) {
  if (alpha.isNaN) {
    throw ArgumentError.value(alpha, 'alpha', 'Must not be NaN.');
  }
  return _runUnaryActivation(
    input: input,
    op: 'elu',
    param0: alpha,
    buildGradFn: (saved, _) => EluBackward(saved, alpha: alpha),
    out: out,
  );
}

/// Applies the Softplus activation elementwise:
/// $\text{Softplus}(x) = \frac{1}{\beta} \ln(1 + \exp(\beta x))$, reverting to
/// linear $x$ when $\beta x > \text{threshold}$.
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> softplus<T extends DTypeTag>(
  GpuArray<T> input, {
  double beta = 1.0,
  double threshold = 20.0,
  GpuArray<T>? out,
}) {
  if (beta <= 0.0 || beta.isNaN) {
    throw ArgumentError.value(beta, 'beta', 'Must be positive.');
  }
  if (threshold.isNaN) {
    throw ArgumentError.value(threshold, 'threshold', 'Must not be NaN.');
  }
  return _runUnaryActivation(
    input: input,
    op: 'softplus',
    param0: beta,
    param1: threshold,
    buildGradFn: (saved, _) =>
        SoftplusBackward(saved, beta: beta, threshold: threshold),
    out: out,
  );
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
  final scaledMask = noGrad(() {
    final randomValues = random_ops.rand(input.shape, input.device);
    final keepBoolean = randomValues.greater(p);
    final keepMask = keepBoolean.astype(input.dtype);
    randomValues.dispose();
    keepBoolean.dispose();
    final scale = 1.0 / (1.0 - p);
    final mask = keepMask * scale;
    keepMask.dispose();
    return mask;
  });
  final result = input * scaledMask;
  if (!result.requiresGrad) {
    scaledMask.dispose();
  }
  return result;
}

/// Applies the Softmax function to an N-dimensional tensor along [axis].
///
/// If [out] is provided, the result is written directly into [out] and returned.
GpuArray<T> softmax<T extends DTypeTag>(
  GpuArray<T> input, {
  int axis = -1,
  GpuArray<T>? out,
}) {
  _validateOutTensor(input, out);
  final computed = noGrad(() {
    final maxVal = input.max(axis: axis, keepDims: true);
    final shifted = input - maxVal;
    maxVal.dispose();
    final expShifted = shifted.exp();
    shifted.dispose();
    final sumExp = expShifted.sum(axis: axis, keepDims: true);
    final rawResult = expShifted / sumExp;
    final result = rawResult.dtype == input.dtype && rawResult is GpuArray<T>
        ? rawResult
        : rawResult.astype<T>(input.dtype);
    if (!identical(result, rawResult)) {
      rawResult.dispose();
    }
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
  _validateOutTensor(input, out);
  final computed = noGrad(() {
    final maxVal = input.max(axis: axis, keepDims: true);
    final shifted = input - maxVal;
    maxVal.dispose();
    final expShifted = shifted.exp();
    final sumExp = expShifted.sum(axis: axis, keepDims: true);
    expShifted.dispose();
    final logSumExp = sumExp.log();
    sumExp.dispose();
    final result = shifted - logSumExp;
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

GpuArray<T> _castLossOutput<T extends DTypeTag>(
  GpuArray<DTypeTag> reduced,
  DType<T> targetDType,
) {
  if (reduced is GpuArray<T>) return reduced;
  final casted = reduced.astype(targetDType);
  reduced.dispose();
  return casted;
}

/// Measures the Mean Squared Error (squared L2 norm) between [input] and [target].
GpuArray<T> mseLoss<T extends SelfOf<DTypeTag>>(
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
    dispatchMseForward(
      input: input,
      targetTensor: target,
      output: elementLosses,
    );
    return switch (reduction) {
      LossReduction.mean => () {
        final result = elementLosses.mean();
        elementLosses.dispose();
        return _castLossOutput(result, input.dtype);
      }(),
      LossReduction.sum => () {
        final result = elementLosses.sum();
        elementLosses.dispose();
        return result;
      }(),
      LossReduction.none => elementLosses,
    };
  });
  if (isGradEnabled && (input.requiresGrad || target.requiresGrad)) {
    computed.requiresGrad = true;
    computed.gradFn = MseLossBackward(input, target, reduction: reduction);
  }
  return computed;
}

/// Measures the Mean Absolute Error (L1 norm) between [input] and [target].
GpuArray<T> l1Loss<T extends SelfOf<DTypeTag>>(
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
        return _castLossOutput(result, input.dtype);
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

/// Measures the Binary Cross-Entropy loss between target probabilities [target] and
/// predicted probabilities [input]:
/// $$\ell(x, y) = -\left(y \ln(x) + (1 - y) \ln(1 - x)\right)$$
GpuArray<T> binaryCrossEntropy<T extends SelfOf<DTypeTag>>(
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
        return _castLossOutput(result, input.dtype);
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
          final reduced = sampleLosses.mean();
          sampleLosses.dispose();
          return _castLossOutput(reduced, logits.dtype);
        }(),
        LossReduction.sum => () {
          final reduced = sampleLosses.sum();
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
