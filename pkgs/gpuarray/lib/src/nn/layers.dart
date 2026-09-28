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

import 'dart:ffi' as ffi;
import 'dart:math' as math;

import '../autograd/autograd.dart';
import '../backend/compute_engine.dart';
import '../device.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import '../operations/manipulation.dart' as manipulation;
import '../random/random.dart' as random_ops;
import '../slice.dart';
import 'functional.dart' as functional;
import 'module.dart';

/// Applies an affine linear transformation to incoming data: $y = x A^T + b$.
final class Linear extends Module {
  /// Size of each input sample.
  final int inFeatures;

  /// Size of each output sample.
  final int outFeatures;

  /// Whether this layer learns an additive bias vector.
  final bool hasBias;

  /// Learnable weight matrix of shape `[outFeatures, inFeatures]`.
  late final GpuArray<Float64> weight;

  /// Optional learnable bias vector of shape `[outFeatures]`.
  late final GpuArray<Float64>? bias;

  /// Creates a [Linear] layer mapping [inFeatures] to [outFeatures].
  ///
  /// Both [inFeatures] and [outFeatures] must be positive integers.
  Linear(
    this.inFeatures,
    this.outFeatures, {
    this.hasBias = true,
    GpuDevice? device,
  }) {
    RangeError.checkValueInInterval(inFeatures, 1, 0x7fffffff, 'inFeatures');
    RangeError.checkValueInInterval(outFeatures, 1, 0x7fffffff, 'outFeatures');

    final targetDevice = device ?? GpuDevice.defaultDevice;
    final bound = 1.0 / math.sqrt(inFeatures);

    final sampledWeight = random_ops.uniform(
      low: -bound,
      high: bound,
      shape: [outFeatures, inFeatures],
      device: targetDevice,
    )..requiresGrad = true;
    weight = registerParameter('weight', sampledWeight);

    if (hasBias) {
      final sampledBias = random_ops.uniform(
        low: -bound,
        high: bound,
        shape: [outFeatures],
        device: targetDevice,
      )..requiresGrad = true;
      bias = registerParameter('bias', sampledBias);
    } else {
      bias = null;
    }
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    final weightTransposed = weight.swapaxes(-1, -2);
    final output = input.matmul(weightTransposed);
    if (bias != null) {
      return output + bias;
    }
    return output;
  }
}

/// Applies a 2D spatial convolution over a 4D input signal (`[N, C_in, H_in, W_in]`).
final class Conv2d extends Module {
  /// Number of channels in the input image.
  final int inChannels;

  /// Number of channels produced by the convolution.
  final int outChannels;

  /// Spatial height and width of the square convolution kernel.
  final int kernelSize;

  /// Stride of the convolution along height and width.
  final int stride;

  /// Zero-padding added to all four spatial borders of the input.
  final int padding;

  /// Whether this layer learns an additive per-channel bias vector.
  final bool hasBias;

  /// Learnable filter weights of shape `[outChannels, inChannels, kernelSize, kernelSize]`.
  late final GpuArray<Float64> weight;

  /// Optional learnable bias vector of shape `[outChannels]`.
  late final GpuArray<Float64>? bias;

  /// Creates a [Conv2d] layer.
  ///
  /// The [inChannels], [outChannels], [kernelSize], and [stride] must be positive,
  /// and [padding] must be non-negative.
  Conv2d(
    this.inChannels,
    this.outChannels,
    this.kernelSize, {
    this.stride = 1,
    this.padding = 0,
    this.hasBias = true,
    GpuDevice? device,
  }) {
    if (inChannels <= 0) {
      throw ArgumentError.value(inChannels, 'inChannels', 'Must be positive.');
    }
    if (outChannels <= 0) {
      throw ArgumentError.value(
        outChannels,
        'outChannels',
        'Must be positive.',
      );
    }
    if (kernelSize <= 0) {
      throw ArgumentError.value(kernelSize, 'kernelSize', 'Must be positive.');
    }
    if (stride <= 0) {
      throw ArgumentError.value(stride, 'stride', 'Must be positive.');
    }
    RangeError.checkNotNegative(padding, 'padding');

    final targetDevice = device ?? GpuDevice.defaultDevice;
    final bound = 1.0 / math.sqrt(inChannels * kernelSize * kernelSize);

    final sampledWeight = random_ops.uniform(
      low: -bound,
      high: bound,
      shape: [outChannels, inChannels, kernelSize, kernelSize],
      device: targetDevice,
    )..requiresGrad = true;
    weight = registerParameter('weight', sampledWeight);

    if (hasBias) {
      final sampledBias = random_ops.uniform(
        low: -bound,
        high: bound,
        shape: [outChannels],
        device: targetDevice,
      )..requiresGrad = true;
      bias = registerParameter('bias', sampledBias);
    } else {
      bias = null;
    }
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    if (input.rank != 4) {
      throw ArgumentError.value(
        input.shape,
        'input',
        'Must be a 4D tensor of shape [batchSize, inChannels, height, width].',
      );
    }
    final batchSize = input.shape[0];
    final inHeight = input.shape[2];
    final inWidth = input.shape[3];

    final outHeight = ((inHeight + 2 * padding - kernelSize) ~/ stride) + 1;
    final outWidth = ((inWidth + 2 * padding - kernelSize) ~/ stride) + 1;
    final patchSize = inChannels * kernelSize * kernelSize;

    final output = noGrad(() {
      final columns = extractIm2ColPatches(
        input,
        kernelSize: kernelSize,
        stride: stride,
        padding: padding,
        outHeight: outHeight,
        outWidth: outWidth,
      );
      final weightMatrix = weight.reshape([outChannels, patchSize]);
      final weightTransposed = weightMatrix.swapaxes(-1, -2);
      var outMatrix = columns.matmul(weightTransposed);
      columns.dispose();
      weightTransposed.dispose();
      weightMatrix.dispose();

      if (bias != null) {
        final withBias = outMatrix + bias;
        outMatrix.dispose();
        outMatrix = withBias;
      }

      final reshaped = outMatrix.reshape([
        batchSize,
        outHeight,
        outWidth,
        outChannels,
      ]);
      outMatrix.dispose();
      final permuted = reshaped.transpose([0, 3, 1, 2]);
      reshaped.dispose();
      final contiguousOut = permuted.copy();
      permuted.dispose();
      return contiguousOut;
    });

    if (isGradEnabled &&
        (input.requiresGrad ||
            weight.requiresGrad ||
            (bias != null && bias!.requiresGrad))) {
      output.requiresGrad = true;
      output.gradFn = Conv2dBackward(
        input: input,
        weight: weight,
        bias: bias,
        stride: stride,
        padding: padding,
        kernelSize: kernelSize,
      );
    }

    return output;
  }
}

/// Applies Layer Normalization over a mini-batch of inputs.
final class LayerNorm extends Module {
  /// Normalized trailing dimensions shape.
  final List<int> normalizedShape;

  /// Small constant added to the denominator for numerical stability.
  final double eps;

  /// Learnable elementwise affine scale parameter ($\gamma$).
  late final GpuArray<Float64> weight;

  /// Learnable elementwise affine shift parameter ($\beta$).
  late final GpuArray<Float64> bias;

  /// Creates a [LayerNorm] module for [normalizedShape].
  LayerNorm(List<int> normalizedShape, {this.eps = 1e-5, GpuDevice? device})
    : normalizedShape = List<int>.unmodifiable(normalizedShape) {
    if (this.normalizedShape.isEmpty) {
      throw ArgumentError.value(
        normalizedShape,
        'normalizedShape',
        'Must not be empty.',
      );
    }
    if (eps <= 0.0) {
      throw ArgumentError.value(eps, 'eps', 'Must be positive.');
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    weight = registerParameter(
      'weight',
      GpuArray.ones(
        this.normalizedShape,
        DType.float64,
        device: targetDevice,
        requiresGrad: true,
      ),
    );
    bias = registerParameter(
      'bias',
      GpuArray.zeros(
        this.normalizedShape,
        DType.float64,
        device: targetDevice,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    final mean = input.mean(axis: -1, keepDims: true);
    final centered = input - mean;
    final variance = (centered * centered).mean(axis: -1, keepDims: true);
    final normalized = centered / (variance + eps).sqrt();
    return normalized * weight + bias;
  }
}

/// During training, randomly zeroes elements of the input tensor with probability [p].
final class Dropout extends Module {
  /// Probability of an element to be zeroed during training.
  final double p;

  /// Creates a [Dropout] layer with drop probability [p] in `[0.0, 1.0)`.
  Dropout({this.p = 0.5}) {
    if (p < 0.0 || p >= 1.0) {
      throw ArgumentError.value(
        p,
        'p',
        'Must be in the half-open interval [0.0, 1.0).',
      );
    }
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    if (!isTraining || p == 0.0) return input;
    final randomValues = random_ops.rand(input.shape, input.device);
    final keepBoolean = randomValues.greater(p);
    final keepMask = keepBoolean.astype(input.dtype);
    randomValues.dispose();
    keepBoolean.dispose();
    final scale = 1.0 / (1.0 - p);
    return input * keepMask * scale;
  }
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
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> indices) {
    final indexCount = indices.size;
    final outShape = [...indices.shape, embeddingDim];
    final output = GpuArray.empty(
      outShape,
      weight.dtype,
      device: indices.device,
    );
    final contiguousWeight = weight.isContiguous ? weight : weight.copy();

    try {
      if (weight.dtype == DType.float64) {
        final weightPtr = contiguousWeight.buffer.pointer.cast<ffi.Double>();
        final outPtr = output.buffer.pointer.cast<ffi.Double>();
        final weightBase = contiguousWeight.offsetElements;

        for (var i = 0; i < indexCount; i++) {
          final tokenIndex = ComputeEngine.readValue(
            indices.buffer,
            indices.dtype,
            i,
            offsetElements: indices.offsetElements,
          ).toInt();
          RangeError.checkValueInInterval(
            tokenIndex,
            0,
            numEmbeddings - 1,
            'indices',
          );
          final srcRowOffset = weightBase + tokenIndex * embeddingDim;
          final dstRowOffset = i * embeddingDim;
          for (var d = 0; d < embeddingDim; d++) {
            outPtr[dstRowOffset + d] = weightPtr[srcRowOffset + d];
          }
        }
      } else {
        for (var i = 0; i < indexCount; i++) {
          final tokenIndex = ComputeEngine.readValue(
            indices.buffer,
            indices.dtype,
            i,
            offsetElements: indices.offsetElements,
          ).toInt();
          RangeError.checkValueInInterval(
            tokenIndex,
            0,
            numEmbeddings - 1,
            'indices',
          );
          final srcRowOffset = tokenIndex * embeddingDim;
          final dstRowOffset = i * embeddingDim;
          for (var d = 0; d < embeddingDim; d++) {
            final val = ComputeEngine.readValue(
              contiguousWeight.buffer,
              contiguousWeight.dtype,
              srcRowOffset + d,
              offsetElements: contiguousWeight.offsetElements,
            );
            ComputeEngine.writeValue(
              output.buffer,
              output.dtype,
              dstRowOffset + d,
              val,
            );
          }
        }
      }
    } finally {
      if (!identical(contiguousWeight, weight)) {
        contiguousWeight.dispose();
      }
    }

    if (isGradEnabled && weight.requiresGrad) {
      output.requiresGrad = true;
      output.gradFn = EmbeddingBackward(
        weight,
        indices,
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

/// Applies Multi-Head Attention over input sequences:
/// $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
/// where $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$.
final class MultiheadAttention extends Module {
  /// Total dimension of the model.
  final int embedDim;

  /// Number of parallel attention heads.
  final int numHeads;

  /// Dropout probability on attention weights.
  final double dropout;

  /// Whether projection layers learn an additive bias.
  final bool hasBias;

  /// Feature dimension of key inputs.
  final int kdim;

  /// Feature dimension of value inputs.
  final int vdim;

  /// Per-head dimension (`embedDim ~/ numHeads`).
  final int headDim;

  /// Query linear projection.
  late final Linear qProj;

  /// Key linear projection.
  late final Linear kProj;

  /// Value linear projection.
  late final Linear vProj;

  /// Output linear projection.
  late final Linear outProj;

  /// Creates a [MultiheadAttention] module.
  ///
  /// The [embedDim] must be positive and evenly divisible by [numHeads].
  MultiheadAttention(
    this.embedDim,
    this.numHeads, {
    this.dropout = 0.0,
    this.hasBias = true,
    int? kdim,
    int? vdim,
    GpuDevice? device,
  }) : kdim = kdim ?? embedDim,
       vdim = vdim ?? embedDim,
       headDim = embedDim ~/ numHeads {
    if (embedDim <= 0) {
      throw ArgumentError.value(embedDim, 'embedDim', 'Must be positive.');
    }
    if (numHeads <= 0) {
      throw ArgumentError.value(numHeads, 'numHeads', 'Must be positive.');
    }
    if (embedDim % numHeads != 0) {
      throw ArgumentError.value(
        embedDim,
        'embedDim',
        'Must be divisible by numHeads ($numHeads).',
      );
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    qProj = registerModule(
      Linear(embedDim, embedDim, hasBias: hasBias, device: targetDevice),
    );
    kProj = registerModule(
      Linear(this.kdim, embedDim, hasBias: hasBias, device: targetDevice),
    );
    vProj = registerModule(
      Linear(this.vdim, embedDim, hasBias: hasBias, device: targetDevice),
    );
    outProj = registerModule(
      Linear(embedDim, embedDim, hasBias: hasBias, device: targetDevice),
    );
  }

  @override
  GpuArray<DTypeTag> forward(
    GpuArray<DTypeTag> input, {
    GpuArray<DTypeTag>? key,
    GpuArray<DTypeTag>? value,
    GpuArray<DTypeTag>? attnMask,
    bool isCausal = false,
  }) {
    final query = input;
    final keyTensor = key ?? query;
    final valueTensor = value ?? query;

    final is2D = query.rank == 2;
    final qInput = is2D ? query.unsqueeze(0) : query;
    final kInput = (keyTensor.rank == 2) ? keyTensor.unsqueeze(0) : keyTensor;
    final vInput = (valueTensor.rank == 2)
        ? valueTensor.unsqueeze(0)
        : valueTensor;

    final batchSize = qInput.shape[0];
    final targetLength = qInput.shape[1];
    final sourceLength = kInput.shape[1];

    final qProjOut = qProj(qInput);
    final kProjOut = kProj(kInput);
    final vProjOut = vProj(vInput);

    final qHeads = qProjOut
        .reshape([batchSize, targetLength, numHeads, headDim])
        .swapaxes(1, 2);
    final kHeads = kProjOut
        .reshape([batchSize, sourceLength, numHeads, headDim])
        .swapaxes(1, 2);
    final vHeads = vProjOut
        .reshape([batchSize, sourceLength, numHeads, headDim])
        .swapaxes(1, 2);

    final attentionOut = functional.scaledDotProductAttention(
      qHeads,
      kHeads,
      vHeads,
      attnMask: attnMask,
      dropoutP: isTraining ? dropout : 0.0,
      isCausal: isCausal,
    );

    final merged = attentionOut.swapaxes(1, 2).reshape([
      batchSize,
      targetLength,
      embedDim,
    ]);

    final output = outProj(merged);
    return is2D ? output.squeeze(axis: 0) : output;
  }

  @override
  GpuArray<DTypeTag> call(
    GpuArray<DTypeTag> input, {
    GpuArray<DTypeTag>? key,
    GpuArray<DTypeTag>? value,
    GpuArray<DTypeTag>? attnMask,
    bool isCausal = false,
  }) => forward(
    input,
    key: key,
    value: value,
    attnMask: attnMask,
    isCausal: isCausal,
  );
}

/// Applies Root Mean Square Layer Normalization (RMSNorm) over a mini-batch of inputs:
/// $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
final class RMSNorm extends Module {
  /// Normalized trailing dimensions shape.
  final List<int> normalizedShape;

  /// Small constant added to the mean square for numerical stability.
  final double eps;

  /// Learnable elementwise scale parameter ($\gamma$).
  late final GpuArray<Float64> weight;

  /// Creates an [RMSNorm] module for [normalizedShape].
  RMSNorm(List<int> normalizedShape, {this.eps = 1e-6, GpuDevice? device})
    : normalizedShape = List<int>.unmodifiable(normalizedShape) {
    if (this.normalizedShape.isEmpty) {
      throw ArgumentError.value(
        normalizedShape,
        'normalizedShape',
        'Must not be empty.',
      );
    }
    if (eps <= 0.0) {
      throw ArgumentError.value(eps, 'eps', 'Must be positive.');
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    weight = registerParameter(
      'weight',
      GpuArray.ones(
        this.normalizedShape,
        DType.float64,
        device: targetDevice,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    final xSquared = input * input;
    final meanSquared = xSquared.mean(axis: -1, keepDims: true);
    final rms = (meanSquared + eps).sqrt();
    final normalized = input / rms;
    return normalized * weight;
  }
}

/// Rotary Position Embedding (RoPE) for transformer query and key representations.
final class RotaryEmbedding extends Module {
  /// Feature dimension to rotate (must be even).
  final int dim;

  /// Maximum precomputed sequence length.
  final int maxSeqLen;

  /// Geometric frequency base ($\theta$).
  final double base;

  /// Precomputed cosine table of shape `[maxSeqLen, dim]`.
  late final GpuArray<Float64> cosCached;

  /// Precomputed sine table of shape `[maxSeqLen, dim]`.
  late final GpuArray<Float64> sinCached;

  /// Creates a [RotaryEmbedding] module with even [dim].
  RotaryEmbedding(
    this.dim, {
    this.maxSeqLen = 2048,
    this.base = 10000.0,
    GpuDevice? device,
  }) {
    if (dim <= 0 || dim % 2 != 0) {
      throw ArgumentError.value(dim, 'dim', 'Must be a positive even integer.');
    }
    if (maxSeqLen <= 0) {
      throw ArgumentError.value(maxSeqLen, 'maxSeqLen', 'Must be positive.');
    }
    final targetDevice = device ?? GpuDevice.defaultDevice;
    final halfDim = dim ~/ 2;

    cosCached = GpuArray.empty(
      [maxSeqLen, dim],
      DType.float64,
      device: targetDevice,
    );
    sinCached = GpuArray.empty(
      [maxSeqLen, dim],
      DType.float64,
      device: targetDevice,
    );
    final cosPtr = cosCached.buffer.pointer.cast<ffi.Double>();
    final sinPtr = sinCached.buffer.pointer.cast<ffi.Double>();

    for (var i = 0; i < halfDim; i++) {
      final inverseFrequency = 1.0 / math.pow(base, (2.0 * i) / dim);
      for (var position = 0; position < maxSeqLen; position++) {
        final theta = position * inverseFrequency;
        final cosValue = math.cos(theta);
        final sinValue = math.sin(theta);
        final rowBase = position * dim;

        cosPtr[rowBase + i] = cosValue;
        cosPtr[rowBase + halfDim + i] = cosValue;
        sinPtr[rowBase + i] = sinValue;
        sinPtr[rowBase + halfDim + i] = sinValue;
      }
    }
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

/// Transformer Encoder Layer composed of multi-head self-attention and a
/// position-wise feed-forward network with residual connections and layer normalization.
final class TransformerEncoderLayer extends Module {
  /// Number of expected features in the input (`d_model`).
  final int dModel;

  /// Number of parallel attention heads (`nhead`).
  final int nhead;

  /// Dimension of the feed-forward network model.
  final int dimFeedforward;

  /// Dropout probability.
  final double dropout;

  /// Whether layer normalization is applied before (`true`, Pre-LN) or after (`false`, Post-LN) sublayers.
  final bool normFirst;

  /// Multi-head self-attention sublayer.
  late final MultiheadAttention selfAttn;

  /// First feed-forward linear projection.
  late final Linear linear1;

  /// Dropout applied to self-attention output.
  late final Dropout dropout1;

  /// Second feed-forward linear projection.
  late final Linear linear2;

  /// Dropout applied inside the feed-forward network.
  late final Dropout dropout2;

  /// First layer normalization sublayer.
  late final LayerNorm norm1;

  /// Second layer normalization sublayer.
  late final LayerNorm norm2;

  /// Activation module between [linear1] and [linear2].
  final Module activation;

  /// Creates a [TransformerEncoderLayer].
  TransformerEncoderLayer(
    this.dModel,
    this.nhead, {
    int? dimFeedforward,
    this.dropout = 0.1,
    Module? activation,
    this.normFirst = false,
    GpuDevice? device,
  }) : dimFeedforward = dimFeedforward ?? (4 * dModel),
       activation = activation ?? ReLU() {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    selfAttn = registerModule(
      MultiheadAttention(dModel, nhead, dropout: dropout, device: targetDevice),
    );
    linear1 = registerModule(
      Linear(dModel, this.dimFeedforward, device: targetDevice),
    );
    dropout1 = registerModule(Dropout(p: dropout));
    linear2 = registerModule(
      Linear(this.dimFeedforward, dModel, device: targetDevice),
    );
    dropout2 = registerModule(Dropout(p: dropout));
    norm1 = registerModule(LayerNorm([dModel], device: targetDevice));
    norm2 = registerModule(LayerNorm([dModel], device: targetDevice));
    registerModule(this.activation);
  }

  @override
  GpuArray<DTypeTag> forward(
    GpuArray<DTypeTag> input, {
    GpuArray<DTypeTag>? srcMask,
    bool isCausal = false,
  }) {
    if (normFirst) {
      var hidden = input;
      final selfAttnOut = selfAttn(
        norm1(hidden),
        attnMask: srcMask,
        isCausal: isCausal,
      );
      hidden = hidden + dropout1(selfAttnOut);
      final feedForwardOut = linear2(
        dropout2(activation(linear1(norm2(hidden)))),
      );
      return hidden + feedForwardOut;
    } else {
      var hidden = input;
      final selfAttnOut = selfAttn(
        hidden,
        attnMask: srcMask,
        isCausal: isCausal,
      );
      hidden = norm1(hidden + dropout1(selfAttnOut));
      final feedForwardOut = linear2(dropout2(activation(linear1(hidden))));
      return norm2(hidden + feedForwardOut);
    }
  }

  @override
  GpuArray<DTypeTag> call(
    GpuArray<DTypeTag> input, {
    GpuArray<DTypeTag>? srcMask,
    bool isCausal = false,
  }) => forward(input, srcMask: srcMask, isCausal: isCausal);
}

/// Transformer Decoder Layer composed of multi-head self-attention, encoder-decoder
/// cross-attention, and a position-wise feed-forward network.
final class TransformerDecoderLayer extends Module {
  /// Number of expected features in the target input (`d_model`).
  final int dModel;

  /// Number of parallel attention heads (`nhead`).
  final int nhead;

  /// Dimension of the feed-forward network model.
  final int dimFeedforward;

  /// Dropout probability.
  final double dropout;

  /// Whether layer normalization is applied before (`true`, Pre-LN) or after (`false`, Post-LN) sublayers.
  final bool normFirst;

  /// Masked multi-head self-attention sublayer.
  late final MultiheadAttention selfAttn;

  /// Encoder-decoder multi-head cross-attention sublayer.
  late final MultiheadAttention multiheadAttn;

  /// First feed-forward linear projection.
  late final Linear linear1;

  /// Dropout applied to self-attention output.
  late final Dropout dropout1;

  /// Second feed-forward linear projection.
  late final Linear linear2;

  /// Dropout applied to cross-attention output.
  late final Dropout dropout2;

  /// Dropout applied inside the feed-forward network.
  late final Dropout dropout3;

  /// First layer normalization sublayer.
  late final LayerNorm norm1;

  /// Second layer normalization sublayer.
  late final LayerNorm norm2;

  /// Third layer normalization sublayer.
  late final LayerNorm norm3;

  /// Activation module between [linear1] and [linear2].
  final Module activation;

  /// Creates a [TransformerDecoderLayer].
  TransformerDecoderLayer(
    this.dModel,
    this.nhead, {
    int? dimFeedforward,
    this.dropout = 0.1,
    Module? activation,
    this.normFirst = false,
    GpuDevice? device,
  }) : dimFeedforward = dimFeedforward ?? (4 * dModel),
       activation = activation ?? ReLU() {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    selfAttn = registerModule(
      MultiheadAttention(dModel, nhead, dropout: dropout, device: targetDevice),
    );
    multiheadAttn = registerModule(
      MultiheadAttention(dModel, nhead, dropout: dropout, device: targetDevice),
    );
    linear1 = registerModule(
      Linear(dModel, this.dimFeedforward, device: targetDevice),
    );
    dropout1 = registerModule(Dropout(p: dropout));
    linear2 = registerModule(
      Linear(this.dimFeedforward, dModel, device: targetDevice),
    );
    dropout2 = registerModule(Dropout(p: dropout));
    dropout3 = registerModule(Dropout(p: dropout));
    norm1 = registerModule(LayerNorm([dModel], device: targetDevice));
    norm2 = registerModule(LayerNorm([dModel], device: targetDevice));
    norm3 = registerModule(LayerNorm([dModel], device: targetDevice));
    registerModule(this.activation);
  }

  @override
  GpuArray<DTypeTag> forward(
    GpuArray<DTypeTag> input, {
    GpuArray<DTypeTag>? memory,
    GpuArray<DTypeTag>? tgtMask,
    GpuArray<DTypeTag>? memoryMask,
    bool tgtIsCausal = true,
  }) {
    if (normFirst) {
      var hidden = input;
      final selfAttnOut = selfAttn(
        norm1(hidden),
        attnMask: tgtMask,
        isCausal: tgtIsCausal,
      );
      hidden = hidden + dropout1(selfAttnOut);
      if (memory != null) {
        final crossAttnOut = multiheadAttn(
          norm2(hidden),
          key: memory,
          value: memory,
          attnMask: memoryMask,
        );
        hidden = hidden + dropout2(crossAttnOut);
      }
      final feedForwardOut = linear2(
        dropout3(activation(linear1(norm3(hidden)))),
      );
      return hidden + feedForwardOut;
    } else {
      var hidden = input;
      final selfAttnOut = selfAttn(
        hidden,
        attnMask: tgtMask,
        isCausal: tgtIsCausal,
      );
      hidden = norm1(hidden + dropout1(selfAttnOut));
      if (memory != null) {
        final crossAttnOut = multiheadAttn(
          hidden,
          key: memory,
          value: memory,
          attnMask: memoryMask,
        );
        hidden = norm2(hidden + dropout2(crossAttnOut));
      }
      final feedForwardOut = linear2(dropout3(activation(linear1(hidden))));
      return norm3(hidden + feedForwardOut);
    }
  }

  @override
  GpuArray<DTypeTag> call(
    GpuArray<DTypeTag> input, {
    GpuArray<DTypeTag>? memory,
    GpuArray<DTypeTag>? tgtMask,
    GpuArray<DTypeTag>? memoryMask,
    bool tgtIsCausal = true,
  }) => forward(
    input,
    memory: memory,
    tgtMask: tgtMask,
    memoryMask: memoryMask,
    tgtIsCausal: tgtIsCausal,
  );
}
