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
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.dropout<T>(input, p: p, training: isTraining);
  }
}

/// Lookup table that stores embeddings of a fixed dictionary of size [numEmbeddings].
final class Embedding extends Module {
  /// Size of the dictionary of embeddings.
  final int numEmbeddings;

  /// Size of each embedding vector.
  final int embeddingDim;

  /// Learnable embedding table of shape `[numEmbeddings, embeddingDim]`.
  late final GpuArray<DTypeTag> weight;

  /// Creates an [Embedding] table with [numEmbeddings] rows of dimension [embeddingDim].
  Embedding(
    this.numEmbeddings,
    this.embeddingDim, {
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) {
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
    if (!dtype.isFloating) {
      throw ArgumentError.value(
        dtype,
        'dtype',
        'Must be a floating-point DType.',
      );
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    final rawWeight = random_ops.randn([
      numEmbeddings,
      embeddingDim,
    ], targetDevice);
    final GpuArray<DTypeTag> sampledWeight;
    if (dtype == DType.float64) {
      sampledWeight = rawWeight..requiresGrad = true;
    } else {
      sampledWeight = rawWeight.astype(dtype)..requiresGrad = true;
      rawWeight.dispose();
    }
    weight = registerParameter('weight', sampledWeight);
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<DTypeTag> input) {
    checkNotDisposed();
    final outShape = [...input.shape, embeddingDim];
    final targetDType = T == Float32
        ? DType.float32
        : (T == Float64 ? DType.float64 : weight.dtype);
    final output = GpuArray.empty(outShape, targetDType, device: input.device);
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
    return output as GpuArray<T>;
  }

  @override
  GpuArray<T> call<T extends DTypeTag>(GpuArray<DTypeTag> input) =>
      forward<T>(input);
}

/// Applies the Rectified Linear Unit (ReLU) activation as a [Module].
final class ReLU extends Module {
  /// Creates a [ReLU] activation module.
  ReLU();

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.relu<T>(input);
  }
}

/// Applies the Gaussian Error Linear Unit (GELU) activation as a [Module].
final class GELU extends Module {
  /// Creates a [GELU] activation module.
  GELU();

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.gelu<T>(input);
  }
}

/// Applies the logistic Sigmoid activation as a [Module].
final class Sigmoid extends Module {
  /// Creates a [Sigmoid] activation module.
  Sigmoid();

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.sigmoid<T>(input);
  }
}

/// Applies the Hyperbolic Tangent (Tanh) activation as a [Module].
final class Tanh extends Module {
  /// Creates a [Tanh] activation module.
  Tanh();

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.tanh<T>(input);
  }
}

/// Applies the Sigmoid Linear Unit (SiLU) activation as a [Module].
final class SiLU extends Module {
  /// Creates a [SiLU] activation module.
  SiLU();

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.silu<T>(input);
  }
}

/// Applies the Swish activation ($x \cdot \sigma(x)$) as a [Module].
final class Swish extends Module {
  /// Creates a [Swish] activation module.
  Swish();

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.swish<T>(input);
  }
}

/// Applies the Leaky Rectified Linear Unit (LeakyReLU) activation as a [Module].
final class LeakyReLU extends Module {
  /// Controls the angle of the negative slope.
  final double negativeSlope;

  /// Creates a [LeakyReLU] activation module with [negativeSlope].
  LeakyReLU({this.negativeSlope = 0.01}) {
    if (negativeSlope.isNaN) {
      throw ArgumentError.value(
        negativeSlope,
        'negativeSlope',
        'Must not be NaN.',
      );
    }
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.leakyRelu<T>(input, negativeSlope: negativeSlope);
  }
}

/// Applies the Exponential Linear Unit (ELU) activation as a [Module].
final class ELU extends Module {
  /// Scale factor $\alpha$ for negative inputs.
  final double alpha;

  /// Creates an [ELU] activation module with [alpha].
  ELU({this.alpha = 1.0}) {
    if (alpha.isNaN) {
      throw ArgumentError.value(alpha, 'alpha', 'Must not be NaN.');
    }
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.elu<T>(input, alpha: alpha);
  }
}

/// Applies the Softplus activation ($\frac{1}{\beta} \ln(1 + e^{\beta x})$) as a [Module].
final class Softplus extends Module {
  /// Inverse temperature scaling factor $\beta > 0$.
  final double beta;

  /// Numerical stability threshold above which Softplus reverts to linear $x$.
  final double threshold;

  /// Creates a [Softplus] activation module with [beta] and [threshold].
  Softplus({this.beta = 1.0, this.threshold = 20.0}) {
    if (beta <= 0.0 || beta.isNaN) {
      throw ArgumentError.value(beta, 'beta', 'Must be positive.');
    }
    if (threshold <= 0.0 || threshold.isNaN) {
      throw ArgumentError.value(threshold, 'threshold', 'Must be positive.');
    }
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.softplus<T>(input, beta: beta, threshold: threshold);
  }
}

/// Applies the Softmax normalization along [axis] as a [Module].
final class Softmax extends Module {
  /// Axis along which Softmax normalization is computed.
  final int axis;

  /// Creates a [Softmax] module along [axis].
  Softmax({this.axis = -1});

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.softmax<T>(input, axis: axis);
  }
}

/// Applies the Log-Softmax normalization along [axis] as a [Module].
final class LogSoftmax extends Module {
  /// Axis along which Log-Softmax normalization is computed.
  final int axis;

  /// Creates a [LogSoftmax] module along [axis].
  LogSoftmax({this.axis = -1});

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    return functional.logSoftmax<T>(input, axis: axis);
  }
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
  late final GpuArray<DTypeTag> cosCached;

  /// Precomputed sine table of shape `[maxSequenceLength, dim]`.
  late final GpuArray<DTypeTag> sinCached;

  /// Creates a [RotaryEmbedding] module with even [dim].
  RotaryEmbedding(
    this.dim, {
    this.maxSequenceLength = 2048,
    this.base = 10000.0,
    DType<DTypeTag> dtype = DType.float64,
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
    if (!dtype.isFloating) {
      throw ArgumentError.value(
        dtype,
        'dtype',
        'Must be a floating-point DType.',
      );
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    cosCached = registerBuffer(
      'cosCached',
      GpuArray.empty([maxSequenceLength, dim], dtype, device: targetDevice),
    );
    sinCached = registerBuffer(
      'sinCached',
      GpuArray.empty([maxSequenceLength, dim], dtype, device: targetDevice),
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
  static GpuArray<T> rotateHalf<T extends DTypeTag>(GpuArray<T> x) {
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

    return manipulation.concatenate([negX2, x1], axis: -1) as GpuArray<T>;
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input, {int offset = 0}) {
    checkNotDisposed();
    RangeError.checkNotNegative(offset, 'offset');
    final sequenceLength = input.shape[input.rank - 2];
    final sliceSpecs = [Slice(offset, offset + sequenceLength), const All()];
    final cosSlice = cosCached.slice(sliceSpecs);
    final sinSlice = sinCached.slice(sliceSpecs);

    final xCos = input * cosSlice;
    final rotatedX = rotateHalf<T>(input);
    final rotatedSin = rotatedX * sinSlice;
    return xCos + rotatedSin;
  }

  @override
  GpuArray<T> call<T extends DTypeTag>(GpuArray<T> input, {int offset = 0}) =>
      forward<T>(input, offset: offset);
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
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) : outFeatures = outFeatures ?? inFeatures {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    w1 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'w1',
    );
    w2 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'w2',
    );
    w3 = registerModule(
      Linear(
        hiddenFeatures,
        this.outFeatures,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'w3',
    );
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    final gate = w1<T>(input);
    final up = functional.silu<T>(w2<T>(input));
    final fused = gate * up;
    return w3<T>(fused);
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
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) : outFeatures = outFeatures ?? inFeatures {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    w1 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'w1',
    );
    w2 = registerModule(
      Linear(
        inFeatures,
        hiddenFeatures,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'w2',
    );
    w3 = registerModule(
      Linear(
        hiddenFeatures,
        this.outFeatures,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'w3',
    );
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    final gate = w1<T>(input);
    final up = functional.gelu<T>(w2<T>(input));
    final fused = gate * up;
    return w3<T>(fused);
  }
}

/// Criterion that measures the Mean Squared Error (squared L2 norm) between predictions and targets.
final class MSELoss {
  /// Reduction mode applied to the output loss.
  final LossReduction reduction;

  /// Creates an [MSELoss] criterion with [reduction].
  const MSELoss({this.reduction = LossReduction.mean});

  /// Computes the MSE loss between [input] and [target].
  GpuArray<T> forward<T extends SelfOf<DTypeTag>>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => functional.mseLoss<T>(input, target, reduction: reduction);

  /// Invokes [forward] on [input] and [target].
  GpuArray<T> call<T extends SelfOf<DTypeTag>>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => forward<T>(input, target);
}

/// Criterion that measures the Mean Absolute Error (L1 norm) between predictions and targets.
final class L1Loss {
  /// Reduction mode applied to the output loss.
  final LossReduction reduction;

  /// Creates an [L1Loss] criterion with [reduction].
  const L1Loss({this.reduction = LossReduction.mean});

  /// Computes the L1 loss between [input] and [target].
  GpuArray<T> forward<T extends SelfOf<DTypeTag>>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => functional.l1Loss<T>(input, target, reduction: reduction);

  /// Invokes [forward] on [input] and [target].
  GpuArray<T> call<T extends SelfOf<DTypeTag>>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => forward<T>(input, target);
}

/// Criterion that measures the Binary Cross-Entropy loss between predicted and target probabilities.
final class BCELoss {
  /// Reduction mode applied to the output loss.
  final LossReduction reduction;

  /// Creates a [BCELoss] criterion with [reduction].
  const BCELoss({this.reduction = LossReduction.mean});

  /// Computes the binary cross-entropy loss between [input] and [target].
  GpuArray<T> forward<T extends SelfOf<DTypeTag>>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => functional.binaryCrossEntropy<T>(input, target, reduction: reduction);

  /// Invokes [forward] on [input] and [target].
  GpuArray<T> call<T extends SelfOf<DTypeTag>>(
    GpuArray<T> input,
    GpuArray<T> target,
  ) => forward<T>(input, target);
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
  ) => functional.crossEntropy<T>(logits, targets, reduction: reduction);

  /// Invokes [forward] on [logits] and [targets].
  GpuArray<T> call<T extends DTypeTag>(
    GpuArray<T> logits,
    GpuArray<DTypeTag> targets,
  ) => forward<T>(logits, targets);
}
