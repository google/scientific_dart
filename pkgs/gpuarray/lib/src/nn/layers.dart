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

import '../operations/manipulation.dart' as manip;
import '../slice.dart';
import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../device.dart';
import '../random/random.dart' as rng;
import '../autograd/autograd.dart';
import 'module.dart';
import 'functional.dart' as f;

/// Applies an affine linear transformation to the incoming data: $y = x A^T + b$.
class Linear extends Module {
  final int inFeatures;
  final int outFeatures;
  final bool hasBias;

  late final GpuArray weight;
  late final GpuArray? bias;

  Linear(
    this.inFeatures,
    this.outFeatures, {
    this.hasBias = true,
    GpuDevice? device,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final k = 1.0 / math.sqrt(inFeatures);

    final rawW = rng.uniform(
      low: -k,
      high: k,
      shape: [outFeatures, inFeatures],
      device: dev,
    );
    weight = registerParameter(
      'weight',
      GpuArray.fromList(
        rawW.toList(),
        [outFeatures, inFeatures],
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );

    if (hasBias) {
      final rawB = rng.uniform(
        low: -k,
        high: k,
        shape: [outFeatures],
        device: dev,
      );
      bias = registerParameter(
        'bias',
        GpuArray.fromList(
          rawB.toList(),
          [outFeatures],
          DType.float64,
          device: dev,
          requiresGrad: true,
        ),
      );
    } else {
      bias = null;
    }
  }

  @override
  GpuArray forward(GpuArray input) {
    final wT = weight.swapaxes(-1, -2);
    final output = input.matmul(wT);
    if (bias != null) {
      return output + bias;
    }
    return output;
  }
}

/// Applies a 2D convolution over an input signal composed of several input planes.
class Conv2d extends Module {
  final int inChannels;
  final int outChannels;
  final int kernelSize;
  final int stride;
  final int padding;
  final bool hasBias;

  late final GpuArray weight;
  late final GpuArray? bias;

  Conv2d(
    this.inChannels,
    this.outChannels,
    this.kernelSize, {
    this.stride = 1,
    this.padding = 0,
    this.hasBias = true,
    GpuDevice? device,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final k = 1.0 / math.sqrt(inChannels * kernelSize * kernelSize);

    final rawW = rng.uniform(
      low: -k,
      high: k,
      shape: [outChannels, inChannels, kernelSize, kernelSize],
      device: dev,
    );
    weight = registerParameter(
      'weight',
      GpuArray.fromList(
        rawW.toList(),
        [outChannels, inChannels, kernelSize, kernelSize],
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );

    if (hasBias) {
      final rawB = rng.uniform(
        low: -k,
        high: k,
        shape: [outChannels],
        device: dev,
      );
      bias = registerParameter(
        'bias',
        GpuArray.fromList(
          rawB.toList(),
          [outChannels],
          DType.float64,
          device: dev,
          requiresGrad: true,
        ),
      );
    } else {
      bias = null;
    }
  }

  @override
  GpuArray forward(GpuArray input) {
    // Input: [N, C_in, H, W]
    final batchSize = input.shape[0];
    final inC = input.shape[1];
    final inH = input.shape[2];
    final inW = input.shape[3];

    final outH = ((inH + 2 * padding - kernelSize) ~/ stride) + 1;
    final outW = ((inW + 2 * padding - kernelSize) ~/ stride) + 1;

    final inputND = input.toNDArray();
    final inputList = inputND
        .toList()
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    inputND.dispose();

    final weightND = weight.toNDArray();
    final weightList = weightND
        .toList()
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    weightND.dispose();

    List<double>? biasList;
    if (bias != null) {
      final biasND = bias!.toNDArray();
      biasList = biasND.toList().cast<num>().map((e) => e.toDouble()).toList();
      biasND.dispose();
    }

    final outSize = batchSize * outChannels * outH * outW;
    final outList = List<double>.filled(outSize, 0.0);

    for (var b = 0; b < batchSize; b++) {
      for (var oc = 0; oc < outChannels; oc++) {
        final bVal = biasList != null ? biasList[oc] : 0.0;
        for (var oh = 0; oh < outH; oh++) {
          for (var ow = 0; ow < outW; ow++) {
            var sum = bVal;
            final ihStart = oh * stride - padding;
            final iwStart = ow * stride - padding;

            for (var ic = 0; ic < inC; ic++) {
              for (var kh = 0; kh < kernelSize; kh++) {
                final ih = ihStart + kh;
                if (ih < 0 || ih >= inH) continue;
                for (var kw = 0; kw < kernelSize; kw++) {
                  final iw = iwStart + kw;
                  if (iw < 0 || iw >= inW) continue;

                  final inIdx = ((b * inC + ic) * inH + ih) * inW + iw;
                  final wIdx =
                      ((oc * inC + ic) * kernelSize + kh) * kernelSize + kw;
                  sum += inputList[inIdx] * weightList[wIdx];
                }
              }
            }

            final outIdx = ((b * outChannels + oc) * outH + oh) * outW + ow;
            outList[outIdx] = sum;
          }
        }
      }
    }

    final out = GpuArray.fromList(
      outList,
      [batchSize, outChannels, outH, outW],
      input.dtype,
      device: input.device,
    );

    if (isGradEnabled &&
        (input.requiresGrad ||
            weight.requiresGrad ||
            (bias != null && bias!.requiresGrad))) {
      out.requiresGrad = true;
      out.gradFn = Conv2dBackward(
        input: input,
        weight: weight,
        bias: bias,
        stride: stride,
        padding: padding,
        kernelSize: kernelSize,
      );
    }

    return out;
  }
}

/// Applies Layer Normalization over a mini-batch of inputs.
class LayerNorm extends Module {
  final List<int> normalizedShape;
  final double eps;

  late final GpuArray weight;
  late final GpuArray bias;

  LayerNorm(this.normalizedShape, {this.eps = 1e-5, GpuDevice? device}) {
    final dev = device ?? GpuDevice.defaultDevice;
    weight = registerParameter(
      'weight',
      GpuArray.ones(
        normalizedShape,
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );
    bias = registerParameter(
      'bias',
      GpuArray.zeros(
        normalizedShape,
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray forward(GpuArray input) {
    final mean = input.mean(axis: -1, keepDims: true);
    final variance = ((input - mean) * (input - mean)).mean(
      axis: -1,
      keepDims: true,
    );
    final normalized = (input - mean) / ((variance + eps).sqrt());
    return normalized * weight + bias;
  }
}

/// During training, randomly zeroes some of the elements of the input tensor with probability [p].
class Dropout extends Module {
  final double p;

  Dropout({this.p = 0.5});

  @override
  GpuArray forward(GpuArray input) {
    if (!isTraining || p == 0.0) return input;
    final mask = rng
        .rand(input.shape, input.device)
        .greater(p)
        .astype(input.dtype);
    final scale = 1.0 / (1.0 - p);
    return input * mask * scale;
  }
}

/// A simple lookup table that stores embeddings of a fixed dictionary and size.
class Embedding extends Module {
  final int numEmbeddings;
  final int embeddingDim;

  late final GpuArray weight;

  Embedding(this.numEmbeddings, this.embeddingDim, {GpuDevice? device}) {
    final dev = device ?? GpuDevice.defaultDevice;
    final rawW = rng.randn([numEmbeddings, embeddingDim], dev);
    weight = registerParameter(
      'weight',
      GpuArray.fromList(
        rawW.toList(),
        [numEmbeddings, embeddingDim],
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray forward(GpuArray indices) {
    final idxList = indices.toList().cast<int>();
    final outRows = <dynamic>[];
    for (final idx in idxList) {
      final row = weight[idx];
      outRows.addAll(row.toList() as Iterable<dynamic>);
    }
    final outShape = [...indices.shape, embeddingDim];
    final out = GpuArray.fromList(
      outRows,
      outShape,
      weight.dtype,
      device: indices.device,
    );
    if (isGradEnabled && weight.requiresGrad) {
      out.requiresGrad = true;
      out.gradFn = EmbeddingBackward(
        weight,
        indices,
        numEmbeddings,
        embeddingDim,
      );
    }
    return out;
  }
}

/// Applies ReLU activation as a module.
class ReLU extends Module {
  @override
  GpuArray forward(GpuArray input) => f.relu(input);
}

/// Applies GELU activation as a module.
class GELU extends Module {
  @override
  GpuArray forward(GpuArray input) => f.gelu(input);
}

/// Applies Sigmoid activation as a module.
class Sigmoid extends Module {
  @override
  GpuArray forward(GpuArray input) => f.sigmoid(input);
}

/// Applies Tanh activation as a module.
class Tanh extends Module {
  @override
  GpuArray forward(GpuArray input) => f.tanh(input);
}

/// Applies Multi-Head Attention over input sequences.
///
/// Multi-head attention allows the model to jointly attend to information
/// from different representation subspaces at different positions:
/// 2314300\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O2314300
/// where 2314300\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)2314300.
class MultiheadAttention extends Module {
  final int embedDim;
  final int numHeads;
  final double dropout;
  final bool hasBias;
  final int kdim;
  final int vdim;
  final int headDim;

  late final Linear qProj;
  late final Linear kProj;
  late final Linear vProj;
  late final Linear outProj;

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
    if (embedDim % numHeads != 0) {
      throw ArgumentError(
        'embedDim ($embedDim) must be divisible by numHeads ($numHeads)',
      );
    }
    final dev = device ?? GpuDevice.defaultDevice;
    qProj = registerModule(
      Linear(embedDim, embedDim, hasBias: hasBias, device: dev),
    );
    kProj = registerModule(
      Linear(this.kdim, embedDim, hasBias: hasBias, device: dev),
    );
    vProj = registerModule(
      Linear(this.vdim, embedDim, hasBias: hasBias, device: dev),
    );
    outProj = registerModule(
      Linear(embedDim, embedDim, hasBias: hasBias, device: dev),
    );
  }

  @override
  GpuArray forward(
    GpuArray input, {
    GpuArray? key,
    GpuArray? value,
    GpuArray? attnMask,
    bool isCausal = false,
  }) {
    final query = input;
    final k = key ?? query;
    final v = value ?? query;

    final is2D = query.rank == 2;
    final qInput = is2D ? query.unsqueeze(0) : query;
    final kInput = (k.rank == 2) ? k.unsqueeze(0) : k;
    final vInput = (v.rank == 2) ? v.unsqueeze(0) : v;

    final batchSize = qInput.shape[0];
    final tgtLen = qInput.shape[1];
    final srcLen = kInput.shape[1];

    // 1. Linear projections
    final qProjOut = qProj(qInput); // [B, tgtLen, embedDim]
    final kProjOut = kProj(kInput); // [B, srcLen, embedDim]
    final vProjOut = vProj(vInput); // [B, srcLen, embedDim]

    // 2. Split into multiple heads: [B, numHeads, seqLen, headDim]
    final qHeads = qProjOut
        .reshape([batchSize, tgtLen, numHeads, headDim])
        .swapaxes(1, 2);
    final kHeads = kProjOut
        .reshape([batchSize, srcLen, numHeads, headDim])
        .swapaxes(1, 2);
    final vHeads = vProjOut
        .reshape([batchSize, srcLen, numHeads, headDim])
        .swapaxes(1, 2);

    // 3. Scaled Dot-Product Attention
    final attnOut = f.scaled_dot_product_attention(
      qHeads,
      kHeads,
      vHeads,
      attnMask: attnMask,
      dropoutP: isTraining ? dropout : 0.0,
      isCausal: isCausal,
    ); // [B, numHeads, tgtLen, headDim]

    // 4. Merge heads: [B, tgtLen, embedDim]
    final merged = attnOut.swapaxes(1, 2).reshape([
      batchSize,
      tgtLen,
      embedDim,
    ]);

    // 5. Output projection
    final output = outProj(merged);

    return is2D ? output.squeeze(axis: 0) : output;
  }

  @override
  GpuArray call(
    GpuArray input, {
    GpuArray? key,
    GpuArray? value,
    GpuArray? attnMask,
    bool isCausal = false,
  }) => forward(
    input,
    key: key,
    value: value,
    attnMask: attnMask,
    isCausal: isCausal,
  );
}

/// Applies Root Mean Square Layer Normalization (RMSNorm) over a mini-batch of inputs.
///
/// 2314300	ext{RMSNorm}(x) = rac{x}{\sqrt{rac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma2314300
class RMSNorm extends Module {
  final List<int> normalizedShape;
  final double eps;

  late final GpuArray weight; // gamma

  RMSNorm(this.normalizedShape, {this.eps = 1e-6, GpuDevice? device}) {
    final dev = device ?? GpuDevice.defaultDevice;
    weight = registerParameter(
      'weight',
      GpuArray.ones(
        normalizedShape,
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray forward(GpuArray input) {
    final xSq = input * input;
    final meanSq = xSq.mean(axis: -1, keepDims: true);
    final rms = (meanSq + eps).sqrt();
    final normalized = input / rms;
    return normalized * weight;
  }
}

/// Rotary Position Embedding (RoPE).
///
/// Applies rotary position embeddings to query and key representations.
class RotaryEmbedding extends Module {
  final int dim;
  final int maxSeqLen;
  final double base;

  late final GpuArray cosCached;
  late final GpuArray sinCached;

  RotaryEmbedding(
    this.dim, {
    this.maxSeqLen = 2048,
    this.base = 10000.0,
    GpuDevice? device,
  }) {
    if (dim % 2 != 0) {
      throw ArgumentError('RotaryEmbedding dim ($dim) must be even.');
    }
    final dev = device ?? GpuDevice.defaultDevice;
    final halfDim = dim ~/ 2;

    // invFreq = 1.0 / (base ^ (2 * i / dim)) for i in 0..halfDim-1
    final invFreq = List<double>.generate(halfDim, (i) {
      return 1.0 / math.pow(base, (2.0 * i) / dim);
    });

    final cosList = List<double>.filled(maxSeqLen * dim, 0.0);
    final sinList = List<double>.filled(maxSeqLen * dim, 0.0);

    for (var pos = 0; pos < maxSeqLen; pos++) {
      for (var i = 0; i < halfDim; i++) {
        final theta = pos * invFreq[i];
        final cosVal = math.cos(theta);
        final sinVal = math.sin(theta);

        // First half and second half
        cosList[pos * dim + i] = cosVal;
        cosList[pos * dim + halfDim + i] = cosVal;

        sinList[pos * dim + i] = sinVal;
        sinList[pos * dim + halfDim + i] = sinVal;
      }
    }

    cosCached = GpuArray.fromList(
      cosList,
      [maxSeqLen, dim],
      DType.float64,
      device: dev,
    );
    sinCached = GpuArray.fromList(
      sinList,
      [maxSeqLen, dim],
      DType.float64,
      device: dev,
    );
  }

  /// Rotates the half dimensions of [x].
  static GpuArray rotateHalf(GpuArray x) {
    final dim = x.shape[x.rank - 1];
    final halfDim = dim ~/ 2;
    final rank = x.rank;

    final x1Specs = List<dynamic>.generate(rank, (d) {
      if (d == rank - 1) return Slice(0, halfDim);
      return const All();
    });
    final x2Specs = List<dynamic>.generate(rank, (d) {
      if (d == rank - 1) return Slice(halfDim, dim);
      return const All();
    });

    final x1 = x.slice(x1Specs);
    final x2 = x.slice(x2Specs);
    final negX2 = x2.negate();

    return manip.concatenate([negX2, x1], axis: -1);
  }

  @override
  GpuArray forward(GpuArray input, {int offset = 0}) {
    final x = input;
    final seqLen = x.shape[x.rank - 2];
    final cosSliceSpecs = [Slice(offset, offset + seqLen), const All()];
    final cos = cosCached.slice(cosSliceSpecs);
    final sin = sinCached.slice(cosSliceSpecs);

    final xCos = x * cos;
    final rotX = rotateHalf(x);
    final rotSin = rotX * sin;
    return xCos + rotSin;
  }

  @override
  GpuArray call(GpuArray input, {int offset = 0}) =>
      forward(input, offset: offset);
}

/// Gated Linear Unit with SiLU activation (SwiGLU).
///
/// 2314300	ext{SwiGLU}(x) = (x W_1) \odot 	ext{silu}(x W_2) W_32314300
class SwiGLU extends Module {
  final int inFeatures;
  final int hiddenFeatures;
  final int outFeatures;
  final bool hasBias;

  late final Linear w1;
  late final Linear w2;
  late final Linear w3;

  SwiGLU(
    this.inFeatures,
    this.hiddenFeatures, {
    int? outFeatures,
    this.hasBias = false,
    GpuDevice? device,
  }) : outFeatures = outFeatures ?? inFeatures {
    final dev = device ?? GpuDevice.defaultDevice;
    w1 = registerModule(
      Linear(inFeatures, hiddenFeatures, hasBias: hasBias, device: dev),
    );
    w2 = registerModule(
      Linear(inFeatures, hiddenFeatures, hasBias: hasBias, device: dev),
    );
    w3 = registerModule(
      Linear(hiddenFeatures, this.outFeatures, hasBias: hasBias, device: dev),
    );
  }

  @override
  GpuArray forward(GpuArray input) {
    final gate = w1(input);
    final up = f.silu(w2(input));
    final fused = gate * up;
    return w3(fused);
  }
}

/// Gated Linear Unit with GELU activation (GeGLU).
///
/// 2314300	ext{GeGLU}(x) = (x W_1) \odot 	ext{gelu}(x W_2) W_32314300
class GeGLU extends Module {
  final int inFeatures;
  final int hiddenFeatures;
  final int outFeatures;
  final bool hasBias;

  late final Linear w1;
  late final Linear w2;
  late final Linear w3;

  GeGLU(
    this.inFeatures,
    this.hiddenFeatures, {
    int? outFeatures,
    this.hasBias = false,
    GpuDevice? device,
  }) : outFeatures = outFeatures ?? inFeatures {
    final dev = device ?? GpuDevice.defaultDevice;
    w1 = registerModule(
      Linear(inFeatures, hiddenFeatures, hasBias: hasBias, device: dev),
    );
    w2 = registerModule(
      Linear(inFeatures, hiddenFeatures, hasBias: hasBias, device: dev),
    );
    w3 = registerModule(
      Linear(hiddenFeatures, this.outFeatures, hasBias: hasBias, device: dev),
    );
  }

  @override
  GpuArray forward(GpuArray input) {
    final gate = w1(input);
    final up = f.gelu(w2(input));
    final fused = gate * up;
    return w3(fused);
  }
}

/// Transformer Encoder Layer.
///
/// Composed of multi-head self-attention and position-wise feed-forward networks,
/// with residual connections and layer normalization.
class TransformerEncoderLayer extends Module {
  final int dModel;
  final int nhead;
  final int dimFeedforward;
  final double dropout;
  final bool normFirst;

  late final MultiheadAttention selfAttn;
  late final Linear linear1;
  late final Dropout dropout1;
  late final Linear linear2;
  late final Dropout dropout2;
  late final LayerNorm norm1;
  late final LayerNorm norm2;
  final Module activation;

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
    final dev = device ?? GpuDevice.defaultDevice;
    selfAttn = registerModule(
      MultiheadAttention(dModel, nhead, dropout: dropout, device: dev),
    );
    linear1 = registerModule(Linear(dModel, this.dimFeedforward, device: dev));
    dropout1 = registerModule(Dropout(p: dropout));
    linear2 = registerModule(Linear(this.dimFeedforward, dModel, device: dev));
    dropout2 = registerModule(Dropout(p: dropout));
    norm1 = registerModule(LayerNorm([dModel], device: dev));
    norm2 = registerModule(LayerNorm([dModel], device: dev));
    registerModule(this.activation);
  }

  @override
  GpuArray forward(GpuArray input, {GpuArray? srcMask, bool isCausal = false}) {
    final src = input;
    if (normFirst) {
      var x = src;
      final sa = selfAttn(norm1(x), attnMask: srcMask, isCausal: isCausal);
      x = x + dropout1(sa);
      final ffn = linear2(dropout2(activation(linear1(norm2(x)))));
      x = x + ffn;
      return x;
    } else {
      var x = src;
      final sa = selfAttn(x, attnMask: srcMask, isCausal: isCausal);
      x = norm1(x + dropout1(sa));
      final ffn = linear2(dropout2(activation(linear1(x))));
      x = norm2(x + ffn);
      return x;
    }
  }

  @override
  GpuArray call(GpuArray input, {GpuArray? srcMask, bool isCausal = false}) =>
      forward(input, srcMask: srcMask, isCausal: isCausal);
}

/// Transformer Decoder Layer.
///
/// Composed of multi-head self-attention, cross-attention (to encoder memory),
/// and position-wise feed-forward networks, with residual connections and layer normalization.
class TransformerDecoderLayer extends Module {
  final int dModel;
  final int nhead;
  final int dimFeedforward;
  final double dropout;
  final bool normFirst;

  late final MultiheadAttention selfAttn;
  late final MultiheadAttention multiheadAttn;
  late final Linear linear1;
  late final Dropout dropout1;
  late final Linear linear2;
  late final Dropout dropout2;
  late final Dropout dropout3;
  late final LayerNorm norm1;
  late final LayerNorm norm2;
  late final LayerNorm norm3;
  final Module activation;

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
    final dev = device ?? GpuDevice.defaultDevice;
    selfAttn = registerModule(
      MultiheadAttention(dModel, nhead, dropout: dropout, device: dev),
    );
    multiheadAttn = registerModule(
      MultiheadAttention(dModel, nhead, dropout: dropout, device: dev),
    );
    linear1 = registerModule(Linear(dModel, this.dimFeedforward, device: dev));
    dropout1 = registerModule(Dropout(p: dropout));
    linear2 = registerModule(Linear(this.dimFeedforward, dModel, device: dev));
    dropout2 = registerModule(Dropout(p: dropout));
    dropout3 = registerModule(Dropout(p: dropout));
    norm1 = registerModule(LayerNorm([dModel], device: dev));
    norm2 = registerModule(LayerNorm([dModel], device: dev));
    norm3 = registerModule(LayerNorm([dModel], device: dev));
    registerModule(this.activation);
  }

  @override
  GpuArray forward(
    GpuArray input, {
    GpuArray? memory,
    GpuArray? tgtMask,
    GpuArray? memoryMask,
    bool tgtIsCausal = true,
  }) {
    final tgt = input;
    if (normFirst) {
      var x = tgt;
      final sa = selfAttn(norm1(x), attnMask: tgtMask, isCausal: tgtIsCausal);
      x = x + dropout1(sa);
      if (memory != null) {
        final ca = multiheadAttn(
          norm2(x),
          key: memory,
          value: memory,
          attnMask: memoryMask,
        );
        x = x + dropout2(ca);
      }
      final ffn = linear2(dropout3(activation(linear1(norm3(x)))));
      x = x + ffn;
      return x;
    } else {
      var x = tgt;
      final sa = selfAttn(x, attnMask: tgtMask, isCausal: tgtIsCausal);
      x = norm1(x + dropout1(sa));
      if (memory != null) {
        final ca = multiheadAttn(
          x,
          key: memory,
          value: memory,
          attnMask: memoryMask,
        );
        x = norm2(x + dropout2(ca));
      }
      final ffn = linear2(dropout3(activation(linear1(x))));
      x = norm3(x + ffn);
      return x;
    }
  }

  @override
  GpuArray call(
    GpuArray input, {
    GpuArray? memory,
    GpuArray? tgtMask,
    GpuArray? memoryMask,
    bool tgtIsCausal = true,
  }) => forward(
    input,
    memory: memory,
    tgtMask: tgtMask,
    memoryMask: memoryMask,
    tgtIsCausal: tgtIsCausal,
  );
}
