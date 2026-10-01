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
import '../device.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import '../operations/manipulation.dart' as manipulation;
import '../random/random.dart' as random_ops;
import '../slice.dart';
import 'functional.dart' as functional;
import 'layers_basic.dart';
import 'module.dart';
import 'nn_wgsl.dart';

/// During training, randomly zeroes elements of the input tensor with probability [p].
final class Dropout extends Module {
  /// Probability of an element to be zeroed during training.
  final double p;

  /// Creates a [Dropout] layer with drop probability [p] in `[0.0, 1.0)`.
  Dropout({this.p = 0.5}) {
    if (p < 0.0 || p >= 1.0 || p.isNaN) {
      throw ArgumentError.value(
        p,
        'p',
        'Must be in the half-open interval [0.0, 1.0).',
      );
    }
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) =>
      functional.dropout(input, p: p, training: isTraining);
}

/// Lookup table that stores embeddings of a fixed dictionary of size [numEmbeddings].
final class Embedding extends Module {
  /// Size of the dictionary of embeddings.
  final int numEmbeddings;

  /// Size of each embedding vector.
  final int embeddingDim;

  /// Learnable embedding table of shape `[numEmbeddings, embeddingDim]`.
  late final GpuArray<Float64> weight;

  /// Creates an [Embedding] table with [numEmbeddings] rows of dimension [embeddingDim].
  Embedding(this.numEmbeddings, this.embeddingDim, {GpuDevice? device}) {
    if (numEmbeddings <= 0) {
      throw ArgumentError.value(
        numEmbeddings,
        'numEmbeddings',
        'Must be positive.',
      );
    }
    if (embeddingDim <= 0) {
      throw ArgumentError.value(
        embeddingDim,
        'embeddingDim',
        'Must be positive.',
      );
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    final sampledWeight = random_ops.randn([
      numEmbeddings,
      embeddingDim,
    ], targetDevice)..requiresGrad = true;
    weight = registerParameter('weight', sampledWeight);
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    final outShape = [...input.shape, embeddingDim];
    final output = GpuArray.empty(outShape, weight.dtype, device: input.device);
    dispatchEmbeddingForward(
      weight: weight,
      indices: input,
      output: output,
      numEmbeddings: numEmbeddings,
      embeddingDim: embeddingDim,
    );
    if (isGradEnabled && weight.requiresGrad) {
      output.requiresGrad = true;
      output.gradFn = EmbeddingBackward(
        weight,
        input,
        numEmbeddings,
        embeddingDim,
      );
    }
    return output;
  }
}

/// Applies the Rectified Linear Unit (ReLU) activation as a [Module].
final class ReLU extends Module {
  /// Creates a [ReLU] activation module.
  ReLU();

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) =>
      functional.relu(input);
}

/// Applies the Gaussian Error Linear Unit (GELU) activation as a [Module].
final class GELU extends Module {
  /// Creates a [GELU] activation module.
  GELU();

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) =>
      functional.gelu(input);
}

/// Applies the logistic Sigmoid activation as a [Module].
final class Sigmoid extends Module {
  /// Creates a [Sigmoid] activation module.
  Sigmoid();

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) =>
      functional.sigmoid(input);
}

/// Applies the Hyperbolic Tangent (Tanh) activation as a [Module].
final class Tanh extends Module {
  /// Creates a [Tanh] activation module.
  Tanh();

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) =>
      functional.tanh(input);
}

/// Rotary Position Embedding (RoPE) for transformer query and key representations.
final class RotaryEmbedding extends Module {
  /// Feature dimension to rotate (must be even).
  final int dim;

  /// Maximum precomputed sequence length.
  final int maxSequenceLength;

  /// Geometric frequency base ($\theta$).
  final double base;

  /// Precomputed cosine table of shape `[maxSequenceLength, dim]`.
  late final GpuArray<Float64> cosCached;

  /// Precomputed sine table of shape `[maxSequenceLength, dim]`.
  late final GpuArray<Float64> sinCached;

  /// Creates a [RotaryEmbedding] module with even [dim].
  RotaryEmbedding(
    this.dim, {
    this.maxSequenceLength = 2048,
    this.base = 10000.0,
    GpuDevice? device,
  }) {
    if (dim <= 0 || dim % 2 != 0) {
      throw ArgumentError.value(dim, 'dim', 'Must be a positive even integer.');
    }
    if (maxSequenceLength <= 0) {
      throw ArgumentError.value(
        maxSequenceLength,
        'maxSequenceLength',
        'Must be positive.',
      );
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    cosCached = GpuArray.empty(
      [maxSequenceLength, dim],
      DType.float64,
      device: targetDevice,
    );
    sinCached = GpuArray.empty(
      [maxSequenceLength, dim],
      DType.float64,
      device: targetDevice,
    );
    dispatchRotaryEmbeddingCache(
      cosCached: cosCached,
      sinCached: sinCached,
      maxSequenceLength: maxSequenceLength,
      dim: dim,
      base: base,
    );
  }

  /// Rotates the trailing half dimensions of [x] (`[-x2, x1]`).
  static GpuArray<DTypeTag> rotateHalf(GpuArray<DTypeTag> x) {
    final dimension = x.shape[x.rank - 1];
    final halfDim = dimension ~/ 2;
    final rank = x.rank;

    final firstHalfSpecs = List<Object>.generate(rank, (d) {
      if (d == rank - 1) return Slice(0, halfDim);
      return const All();
    });
    final secondHalfSpecs = List<Object>.generate(rank, (d) {
      if (d == rank - 1) return Slice(halfDim, dimension);
      return const All();
    });

    final x1 = x.slice(firstHalfSpecs);
    final x2 = x.slice(secondHalfSpecs);
    final negX2 = x2.negate();

    return manipulation.concatenate([negX2, x1], axis: -1);
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input, {int offset = 0}) {
    RangeError.checkNotNegative(offset, 'offset');
    final sequenceLength = input.shape[input.rank - 2];
    final sliceSpecs = [Slice(offset, offset + sequenceLength), const All()];
    final cosSlice = cosCached.slice(sliceSpecs);
    final sinSlice = sinCached.slice(sliceSpecs);

    final xCos = input * cosSlice;
    final rotatedX = rotateHalf(input);
    final rotatedSin = rotatedX * sinSlice;
    return xCos + rotatedSin;
  }

  @override
  GpuArray<DTypeTag> call(GpuArray<DTypeTag> input, {int offset = 0}) =>
      forward(input, offset: offset);
}

/// Gated Linear Unit with SiLU activation (SwiGLU):
/// $$\text{SwiGLU}(x) = \left((x W_1) \odot \text{SiLU}(x W_2)\right) W_3$$
final class SwiGLU extends Module {
  /// Input feature dimension.
  final int inFeatures;

  /// Hidden gate/up-projection dimension.
  final int hiddenFeatures;

  /// Output feature dimension.
  final int outFeatures;

  /// Whether the linear projections learn additive biases.
  final bool hasBias;

  /// Gate projection layer ($W_1$).
  late final Linear w1;

  /// Up-projection layer ($W_2$).
  late final Linear w2;

  /// Down-projection layer ($W_3$).
  late final Linear w3;

  /// Creates a [SwiGLU] module.
  SwiGLU(
    this.inFeatures,
    this.hiddenFeatures, {
    int? outFeatures,
    this.hasBias = false,
    GpuDevice? device,
  }) : outFeatures = outFeatures ?? inFeatures {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    w1 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        device: targetDevice,
      ),
    );
    w2 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        device: targetDevice,
      ),
    );
    w3 = registerModule(
      Linear(
        hiddenFeatures,
        this.outFeatures,
        hasBias: hasBias,
        device: targetDevice,
      ),
    );
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    final gate = w1(input);
    final up = functional.silu(w2(input));
    final fused = gate * up;
    return w3(fused);
  }
}

/// Gated Linear Unit with GELU activation (GeGLU):
/// $$\text{GeGLU}(x) = \left((x W_1) \odot \text{GELU}(x W_2)\right) W_3$$
final class GeGLU extends Module {
  /// Input feature dimension.
  final int inFeatures;

  /// Hidden gate/up-projection dimension.
  final int hiddenFeatures;

  /// Output feature dimension.
  final int outFeatures;

  /// Whether the linear projections learn additive biases.
  final bool hasBias;

  /// Gate projection layer ($W_1$).
  late final Linear w1;

  /// Up-projection layer ($W_2$).
  late final Linear w2;

  /// Down-projection layer ($W_3$).
  late final Linear w3;

  /// Creates a [GeGLU] module.
  GeGLU(
    this.inFeatures,
    this.hiddenFeatures, {
    int? outFeatures,
    this.hasBias = false,
    GpuDevice? device,
  }) : outFeatures = outFeatures ?? inFeatures {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    w1 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        device: targetDevice,
      ),
    );
    w2 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        device: targetDevice,
      ),
    );
    w3 = registerModule(
      Linear(
        hiddenFeatures,
        this.outFeatures,
        hasBias: hasBias,
        device: targetDevice,
      ),
    );
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    final gate = w1(input);
    final up = functional.gelu(w2(input));
    final fused = gate * up;
    return w3(fused);
  }
}

/// Criterion that measures the Mean Squared Error (squared L2 norm) between predictions and targets.
final class MSELoss {
  /// Reduction mode applied to the output loss.
  final LossReduction reduction;

  /// Creates an [MSELoss] criterion with [reduction].
  const MSELoss({this.reduction = LossReduction.mean});

  /// Computes the MSE loss between [input] and [target].
  GpuArray<DTypeTag> forward<T extends DTypeTag>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => functional.mseLoss(input, target, reduction: reduction);

  /// Invokes [forward] on [input] and [target].
  GpuArray<DTypeTag> call<T extends DTypeTag>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => forward(input, target);
}

/// Criterion that measures the Mean Absolute Error (L1 norm) between predictions and targets.
final class L1Loss {
  /// Reduction mode applied to the output loss.
  final LossReduction reduction;

  /// Creates an [L1Loss] criterion with [reduction].
  const L1Loss({this.reduction = LossReduction.mean});

  /// Computes the L1 loss between [input] and [target].
  GpuArray<DTypeTag> forward<T extends DTypeTag>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => functional.l1Loss(input, target, reduction: reduction);

  /// Invokes [forward] on [input] and [target].
  GpuArray<DTypeTag> call<T extends DTypeTag>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => forward(input, target);
}

/// Criterion that measures the Binary Cross-Entropy loss between predicted and target probabilities.
final class BCELoss {
  /// Reduction mode applied to the output loss.
  final LossReduction reduction;

  /// Creates a [BCELoss] criterion with [reduction].
  const BCELoss({this.reduction = LossReduction.mean});

  /// Computes the binary cross-entropy loss between [input] and [target].
  GpuArray<DTypeTag> forward<T extends DTypeTag>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => functional.binaryCrossEntropy(input, target, reduction: reduction);

  /// Invokes [forward] on [input] and [target].
  GpuArray<DTypeTag> call<T extends DTypeTag>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => forward(input, target);
}

/// Criterion that computes the categorical cross-entropy loss between unnormalized logits and class targets.
final class CrossEntropyLoss {
  /// Reduction mode applied to the output loss.
  final LossReduction reduction;

  /// Creates a [CrossEntropyLoss] criterion with [reduction].
  const CrossEntropyLoss({this.reduction = LossReduction.mean});

  /// Computes the categorical cross-entropy loss between [logits] and [targets].
  GpuArray<T> forward<T extends DTypeTag>(
    GpuArray<T> logits,
    GpuArray<DTypeTag> targets,
  ) => functional.crossEntropy(logits, targets, reduction: reduction);

  /// Invokes [forward] on [logits] and [targets].
  GpuArray<T> call<T extends DTypeTag>(
    GpuArray<T> logits,
    GpuArray<DTypeTag> targets,
  ) => forward(logits, targets);
}
