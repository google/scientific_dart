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

import '../device.dart';
import '../dtype.dart';
import '../gpu_array.dart';
import 'functional.dart' as functional;
import 'layers_basic.dart';
import 'layers_losses.dart';
import 'module.dart';

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
    DType<DTypeTag> dtype = DType.float64,
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
      Linear(
        embedDim,
        embedDim,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'qProj',
    );
    kProj = registerModule(
      Linear(
        this.kdim,
        embedDim,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'kProj',
    );
    vProj = registerModule(
      Linear(
        this.vdim,
        embedDim,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'vProj',
    );
    outProj = registerModule(
      Linear(
        embedDim,
        embedDim,
        hasBias: hasBias,
        dtype: dtype,
        device: targetDevice,
      ),
      'outProj',
    );
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(
    GpuArray<T> input, {
    GpuArray<T>? key,
    GpuArray<T>? value,
    GpuArray<DTypeTag>? attnMask,
    bool isCausal = false,
  }) {
    checkNotDisposed();
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

    final qProjOut = qProj<T>(qInput);
    final kProjOut = kProj<T>(kInput);
    final vProjOut = vProj<T>(vInput);

    final qHeads = qProjOut
        .reshape([batchSize, targetLength, numHeads, headDim])
        .swapaxes(1, 2);
    final kHeads = kProjOut
        .reshape([batchSize, sourceLength, numHeads, headDim])
        .swapaxes(1, 2);
    final vHeads = vProjOut
        .reshape([batchSize, sourceLength, numHeads, headDim])
        .swapaxes(1, 2);

    final attentionOut = functional.scaledDotProductAttention<T>(
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

    final output = outProj<T>(merged);
    return is2D ? output.squeeze(axis: 0) : output;
  }

  @override
  GpuArray<T> call<T extends DTypeTag>(
    GpuArray<T> input, {
    GpuArray<T>? key,
    GpuArray<T>? value,
    GpuArray<DTypeTag>? attnMask,
    bool isCausal = false,
  }) => forward<T>(
    input,
    key: key,
    value: value,
    attnMask: attnMask,
    isCausal: isCausal,
  );
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
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) : dimFeedforward = dimFeedforward ?? (4 * dModel),
       activation = activation ?? ReLU() {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    selfAttn = registerModule(
      MultiheadAttention(
        dModel,
        nhead,
        dropout: dropout,
        dtype: dtype,
        device: targetDevice,
      ),
      'selfAttn',
    );
    linear1 = registerModule(
      Linear(dModel, this.dimFeedforward, dtype: dtype, device: targetDevice),
      'linear1',
    );
    dropout1 = registerModule(Dropout(p: dropout), 'dropout1');
    linear2 = registerModule(
      Linear(this.dimFeedforward, dModel, dtype: dtype, device: targetDevice),
      'linear2',
    );
    dropout2 = registerModule(Dropout(p: dropout), 'dropout2');
    norm1 = registerModule(
      LayerNorm([dModel], dtype: dtype, device: targetDevice),
      'norm1',
    );
    norm2 = registerModule(
      LayerNorm([dModel], dtype: dtype, device: targetDevice),
      'norm2',
    );
    registerModule(this.activation, 'activation');
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(
    GpuArray<T> input, {
    GpuArray<DTypeTag>? srcMask,
    bool isCausal = false,
  }) {
    checkNotDisposed();
    if (normFirst) {
      var hidden = input;
      final selfAttnOut = selfAttn<T>(
        norm1<T>(hidden),
        attnMask: srcMask,
        isCausal: isCausal,
      );
      hidden = hidden + dropout1<T>(selfAttnOut);
      final feedForwardOut = linear2<T>(
        dropout2<T>(activation<T>(linear1<T>(norm2<T>(hidden)))),
      );
      return hidden + feedForwardOut;
    } else {
      var hidden = input;
      final selfAttnOut = selfAttn<T>(
        hidden,
        attnMask: srcMask,
        isCausal: isCausal,
      );
      hidden = norm1<T>(hidden + dropout1<T>(selfAttnOut));
      final feedForwardOut = linear2<T>(
        dropout2<T>(activation<T>(linear1<T>(hidden))),
      );
      return norm2<T>(hidden + feedForwardOut);
    }
  }

  @override
  GpuArray<T> call<T extends DTypeTag>(
    GpuArray<T> input, {
    GpuArray<DTypeTag>? srcMask,
    bool isCausal = false,
  }) => forward<T>(input, srcMask: srcMask, isCausal: isCausal);
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
    DType<DTypeTag> dtype = DType.float64,
    GpuDevice? device,
  }) : dimFeedforward = dimFeedforward ?? (4 * dModel),
       activation = activation ?? ReLU() {
    final targetDevice = device ?? GpuDevice.defaultDevice;
    selfAttn = registerModule(
      MultiheadAttention(
        dModel,
        nhead,
        dropout: dropout,
        dtype: dtype,
        device: targetDevice,
      ),
      'selfAttn',
    );
    multiheadAttn = registerModule(
      MultiheadAttention(
        dModel,
        nhead,
        dropout: dropout,
        dtype: dtype,
        device: targetDevice,
      ),
      'multiheadAttn',
    );
    linear1 = registerModule(
      Linear(dModel, this.dimFeedforward, dtype: dtype, device: targetDevice),
      'linear1',
    );
    dropout1 = registerModule(Dropout(p: dropout), 'dropout1');
    linear2 = registerModule(
      Linear(this.dimFeedforward, dModel, dtype: dtype, device: targetDevice),
      'linear2',
    );
    dropout2 = registerModule(Dropout(p: dropout), 'dropout2');
    dropout3 = registerModule(Dropout(p: dropout), 'dropout3');
    norm1 = registerModule(
      LayerNorm([dModel], dtype: dtype, device: targetDevice),
      'norm1',
    );
    norm2 = registerModule(
      LayerNorm([dModel], dtype: dtype, device: targetDevice),
      'norm2',
    );
    norm3 = registerModule(
      LayerNorm([dModel], dtype: dtype, device: targetDevice),
      'norm3',
    );
    registerModule(this.activation, 'activation');
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(
    GpuArray<T> input, {
    GpuArray<T>? memory,
    GpuArray<DTypeTag>? tgtMask,
    GpuArray<DTypeTag>? memoryMask,
    bool tgtIsCausal = true,
  }) {
    checkNotDisposed();
    if (normFirst) {
      var hidden = input;
      final selfAttnOut = selfAttn<T>(
        norm1<T>(hidden),
        attnMask: tgtMask,
        isCausal: tgtIsCausal,
      );
      hidden = hidden + dropout1<T>(selfAttnOut);
      if (memory != null) {
        final crossAttnOut = multiheadAttn<T>(
          norm2<T>(hidden),
          key: memory,
          value: memory,
          attnMask: memoryMask,
        );
        hidden = hidden + dropout2<T>(crossAttnOut);
      }
      final feedForwardOut = linear2<T>(
        dropout3<T>(activation<T>(linear1<T>(norm3<T>(hidden)))),
      );
      return hidden + feedForwardOut;
    } else {
      var hidden = input;
      final selfAttnOut = selfAttn<T>(
        hidden,
        attnMask: tgtMask,
        isCausal: tgtIsCausal,
      );
      hidden = norm1<T>(hidden + dropout1<T>(selfAttnOut));
      if (memory != null) {
        final crossAttnOut = multiheadAttn<T>(
          hidden,
          key: memory,
          value: memory,
          attnMask: memoryMask,
        );
        hidden = norm2<T>(hidden + dropout2<T>(crossAttnOut));
      }
      final feedForwardOut = linear2<T>(
        dropout3<T>(activation<T>(linear1<T>(hidden))),
      );
      return norm3<T>(hidden + feedForwardOut);
    }
  }

  @override
  GpuArray<T> call<T extends DTypeTag>(
    GpuArray<T> input, {
    GpuArray<T>? memory,
    GpuArray<DTypeTag>? tgtMask,
    GpuArray<DTypeTag>? memoryMask,
    bool tgtIsCausal = true,
  }) => forward<T>(
    input,
    memory: memory,
    tgtMask: tgtMask,
    memoryMask: memoryMask,
    tgtIsCausal: tgtIsCausal,
  );
}
