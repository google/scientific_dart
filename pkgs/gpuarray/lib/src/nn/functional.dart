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

import '../random/random.dart' as rng;
import '../dtype.dart';
// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../gpu_array.dart';
import '../autograd/autograd.dart';

/// Applies the Rectified Linear Unit function element-wise: $\text{ReLU}(x) = \max(0, x)$.
GpuArray relu(GpuArray input) {
  final mask = input.greater(0.0).astype(input.dtype);
  final out = input * mask;
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = ReluBackward(input);
  }
  return out;
}

/// Applies the Sigmoid element-wise: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
GpuArray sigmoid(GpuArray input) {
  final expNeg = input.negate().exp();
  final out = (expNeg + 1.0).pow(-1.0);
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = SigmoidBackward(input, out);
  }
  return out;
}

/// Applies the Hyperbolic Tangent element-wise: $\tanh(x)$.
GpuArray tanh(GpuArray input) {
  final out = input.tanh();
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = TanhBackward(input, out);
  }
  return out;
}

/// Applies Gaussian Error Linear Unit (GELU) activation:
/// $\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right)\right)$.
GpuArray gelu(GpuArray input) {
  final sqrt2OverPi = math.sqrt(2.0 / math.pi);
  final xCubed = input * input * input * 0.044715;
  final inner = (input + xCubed) * sqrt2OverPi;
  final tanhInner = inner.tanh();
  final onePlusTanh = tanhInner + 1.0;
  return input * onePlusTanh * 0.5;
}

/// Applies the Sigmoid Linear Unit (SiLU / Swish) activation: $\text{SiLU}(x) = x \cdot \sigma(x)$.
GpuArray silu(GpuArray input) {
  return input * sigmoid(input);
}

/// Applies Softmax function to an n-dimensional input tensor along [axis].
GpuArray softmax(GpuArray input, {int axis = -1}) {
  final maxVal = input.max(axis: axis, keepDims: true);
  final expX = (input - maxVal).exp();
  final sumExp = expX.sum(axis: axis, keepDims: true);
  final out = expX / sumExp;
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = SoftmaxBackward(input, out, axis: axis);
  }
  return out;
}

/// Applies Log-Softmax function to an n-dimensional input tensor along [axis].
GpuArray log_softmax(GpuArray input, {int axis = -1}) {
  final maxVal = input.max(axis: axis, keepDims: true);
  final expX = (input - maxVal).exp();
  final sumExp = expX.sum(axis: axis, keepDims: true);
  final out = (input - maxVal) - sumExp.log();
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = LogSoftmaxBackward(input, out, axis: axis);
  }
  return out;
}

/// Measures the Mean Squared Error (squared L2 norm) between [input] and [target].
GpuArray mse_loss(
  GpuArray input,
  GpuArray target, {
  String reduction = 'mean',
}) {
  final diff = input - target;
  final sq = diff * diff;
  if (reduction == 'mean') {
    return sq.mean();
  } else if (reduction == 'sum') {
    return sq.sum();
  }
  return sq;
}

/// Computes the cross entropy loss between input logits and target class indices.
GpuArray cross_entropy(
  GpuArray logits,
  GpuArray targets, {
  String reduction = 'mean',
}) {
  final probs = softmax(logits, axis: -1);
  final logProbs = log_softmax(logits, axis: -1);
  final numSamples = logits.shape[0];
  final numClasses = logits.shape[logits.rank - 1];

  final targetList = targets.toList().cast<int>();
  final logProbsND = logProbs.toNDArray();
  final logProbsList = logProbsND
      .toList()
      .cast<num>()
      .map((e) => e.toDouble())
      .toList();
  logProbsND.dispose();

  final losses = <double>[];
  for (var i = 0; i < numSamples; i++) {
    final targetClass = targetList[i];
    final logP = logProbsList[i * numClasses + targetClass];
    losses.add(-logP);
  }

  GpuArray lossArray;
  if (reduction == 'mean') {
    var sum = 0.0;
    for (final l in losses) {
      sum += l;
    }
    lossArray = GpuArray.fromList(
      [sum / numSamples],
      [],
      logits.dtype,
      device: logits.device,
    );
  } else if (reduction == 'sum') {
    var sum = 0.0;
    for (final l in losses) {
      sum += l;
    }
    lossArray = GpuArray.fromList(
      [sum],
      [],
      logits.dtype,
      device: logits.device,
    );
  } else {
    lossArray = GpuArray.fromList(
      losses,
      [numSamples],
      logits.dtype,
      device: logits.device,
    );
  }

  if (isGradEnabled && logits.requiresGrad) {
    lossArray.requiresGrad = true;
    lossArray.gradFn = CrossEntropyBackward(
      logits,
      targets,
      probs,
      reduction: reduction,
    );
  }

  return lossArray;
}

/// Scaled Dot-Product Attention:
/// \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
///
/// Supports batched queries, keys, and values (e.g. `[B, H, N, D]` or `[N, D]`).
/// Supports optional causal triangular masking when [isCausal] is true,
/// arbitrary custom [attnMask], [dropoutP], and explicit [scale].
GpuArray scaled_dot_product_attention(
  GpuArray query,
  GpuArray key,
  GpuArray value, {
  GpuArray? attnMask,
  double dropoutP = 0.0,
  bool isCausal = false,
  double? scale,
}) {
  final dK = query.shape[query.rank - 1];
  final scaleFactor = scale ?? (1.0 / math.sqrt(dK));

  final kT = key.swapaxes(-1, -2);
  var scores = query.matmul(kT) * scaleFactor;

  final qSeqLen = query.shape[query.rank - 2];
  final kSeqLen = key.shape[key.rank - 2];

  if (isCausal) {
    final causalList = List<double>.generate(qSeqLen * kSeqLen, (idx) {
      final i = idx ~/ kSeqLen;
      final j = idx % kSeqLen;
      return (j > i) ? -1e9 : 0.0;
    });
    final causalMask = GpuArray.fromList(
      causalList,
      [qSeqLen, kSeqLen],
      scores.dtype,
      device: scores.device,
    );
    scores = scores + causalMask;
  }

  if (attnMask != null) {
    if (attnMask.dtype == DType.boolean) {
      final maskND = attnMask.toNDArray();
      final maskList = maskND.toList().cast<dynamic>();
      maskND.dispose();
      final additiveList = List<double>.generate(maskList.length, (i) {
        final val = maskList[i];
        final keep = (val == true || (val is num && val != 0));
        return keep ? 0.0 : -1e9;
      });
      final additiveMask = GpuArray.fromList(
        additiveList,
        attnMask.shape,
        scores.dtype,
        device: scores.device,
      );
      scores = scores + additiveMask;
    } else {
      scores = scores + attnMask;
    }
  }

  var attnWeights = softmax(scores, axis: -1);

  if (dropoutP > 0.0) {
    final mask = rng
        .rand(attnWeights.shape, attnWeights.device)
        .greater(dropoutP)
        .astype(attnWeights.dtype);
    final pScale = 1.0 / (1.0 - dropoutP);
    attnWeights = attnWeights * mask * pScale;
  }

  return attnWeights.matmul(value);
}
