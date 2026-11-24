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

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ndarray/ndarray.dart';

import 'safetensors.dart';
import 'wordpiece_tokenizer.dart';

/// Architecture hyperparameters for a BERT encoder model.
final class BertConfig {
  /// Vocabulary size (`word_embeddings` rows).
  final int vocabSize;

  /// Hidden representation width (`H`).
  final int hiddenSize;

  /// Number of stacked transformer encoder layers (`L`).
  final int numHiddenLayers;

  /// Number of parallel self-attention heads per layer.
  final int numAttentionHeads;

  /// Inner feed-forward projection width (typically `4 * hiddenSize`).
  final int intermediateSize;

  /// Maximum positional embedding length.
  final int maxPositionEmbeddings;

  /// Number of segment / token-type embeddings (typically `2`).
  final int typeVocabSize;

  /// Epsilon added to variance inside LayerNorm for numerical stability.
  final double layerNormEps;

  /// Creates a validated [BertConfig].
  BertConfig({
    required this.vocabSize,
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.intermediateSize,
    required this.maxPositionEmbeddings,
    this.typeVocabSize = 2,
    this.layerNormEps = 1e-12,
  }) {
    if (vocabSize <= 0) {
      throw ArgumentError.value(vocabSize, 'vocabSize', 'Must be positive');
    }
    if (hiddenSize <= 0) {
      throw ArgumentError.value(hiddenSize, 'hiddenSize', 'Must be positive');
    }
    if (numHiddenLayers <= 0) {
      throw ArgumentError.value(
        numHiddenLayers,
        'numHiddenLayers',
        'Must be positive',
      );
    }
    if (numAttentionHeads <= 0 || hiddenSize % numAttentionHeads != 0) {
      throw ArgumentError.value(
        numAttentionHeads,
        'numAttentionHeads',
        'Must be positive and evenly divide hiddenSize ($hiddenSize)',
      );
    }
    if (intermediateSize <= 0) {
      throw ArgumentError.value(
        intermediateSize,
        'intermediateSize',
        'Must be positive',
      );
    }
    if (maxPositionEmbeddings <= 0) {
      throw ArgumentError.value(
        maxPositionEmbeddings,
        'maxPositionEmbeddings',
        'Must be positive',
      );
    }
    if (typeVocabSize <= 0) {
      throw ArgumentError.value(
        typeVocabSize,
        'typeVocabSize',
        'Must be positive',
      );
    }
    if (layerNormEps <= 0 || !layerNormEps.isFinite) {
      throw ArgumentError.value(
        layerNormEps,
        'layerNormEps',
        'Must be a positive finite number',
      );
    }
  }

  /// Standard configuration for `sentence-transformers/all-MiniLM-L6-v2`
  /// (6 layers, 384 hidden, 12 attention heads, 1536 FFN, 30,522 vocab).
  factory BertConfig.allMiniLmL6V2() => BertConfig(
    vocabSize: 30522,
    hiddenSize: 384,
    numHiddenLayers: 6,
    numAttentionHeads: 12,
    intermediateSize: 1536,
    maxPositionEmbeddings: 512,
    typeVocabSize: 2,
    layerNormEps: 1e-12,
  );

  /// Parses a Hugging Face `config.json` map.
  factory BertConfig.fromJson(Map<String, dynamic> json) {
    int reqInt(String key, [int? fallback]) {
      final v = json[key];
      if (v is int) return v;
      if (fallback != null && v == null) return fallback;
      throw FormatException('config.json missing integer field "$key".');
    }

    final hiddenAct = json['hidden_act'];
    if (hiddenAct != null && hiddenAct != 'gelu') {
      throw FormatException(
        'Unsupported BERT hidden_act "$hiddenAct" (only exact "gelu" is supported).',
      );
    }
    final posType = json['position_embedding_type'];
    if (posType != null && posType != 'absolute') {
      throw FormatException(
        'Unsupported BERT position_embedding_type "$posType" (only "absolute" is supported).',
      );
    }

    final epsRaw = json['layer_norm_eps'];
    final eps = epsRaw is num ? epsRaw.toDouble() : 1e-12;
    return BertConfig(
      vocabSize: reqInt('vocab_size'),
      hiddenSize: reqInt('hidden_size'),
      numHiddenLayers: reqInt('num_hidden_layers'),
      numAttentionHeads: reqInt('num_attention_heads'),
      intermediateSize: reqInt('intermediate_size'),
      maxPositionEmbeddings: reqInt('max_position_embeddings'),
      typeVocabSize: reqInt('type_vocab_size', 2),
      layerNormEps: eps,
    );
  }

  /// Loads a [BertConfig] from a Hugging Face `config.json` [file].
  factory BertConfig.fromJsonFile(File file) {
    final Object? decoded;
    try {
      decoded = jsonDecode(file.readAsStringSync());
    } on FormatException catch (e) {
      throw FormatException('Invalid config.json: ${e.message}');
    }
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('config.json root must be a JSON object.');
    }
    return BertConfig.fromJson(decoded);
  }

  /// Dimension of each attention head (`hiddenSize ~/ numAttentionHeads`).
  int get headDim => hiddenSize ~/ numAttentionHeads;
}

/// Native `Float32` weight tensors for a single BERT encoder layer.
///
/// Linear projection matrices are stored pre-transposed as contiguous
/// `[inFeatures, outFeatures]` arrays so forward passes execute a single
/// contiguous `cblas_sgemm` without runtime transposes.
final class BertLayerWeights {
  /// Query projection weights transposed to `[H, H]`.
  final NDArray<Float32> queryWeightT;

  /// Query projection bias `[H]`.
  final NDArray<Float32> queryBias;

  /// Key projection weights transposed to `[H, H]`.
  final NDArray<Float32> keyWeightT;

  /// Key projection bias `[H]`.
  final NDArray<Float32> keyBias;

  /// Value projection weights transposed to `[H, H]`.
  final NDArray<Float32> valueWeightT;

  /// Value projection bias `[H]`.
  final NDArray<Float32> valueBias;

  /// Self-attention output projection weights transposed to `[H, H]`.
  final NDArray<Float32> attnOutputWeightT;

  /// Self-attention output projection bias `[H]`.
  final NDArray<Float32> attnOutputBias;

  /// Post-attention LayerNorm scale `[H]`.
  final NDArray<Float32> attnLayerNormWeight;

  /// Post-attention LayerNorm shift `[H]`.
  final NDArray<Float32> attnLayerNormBias;

  /// FFN intermediate projection weights transposed to `[H, intermediate]`.
  final NDArray<Float32> intermediateWeightT;

  /// FFN intermediate projection bias `[intermediate]`.
  final NDArray<Float32> intermediateBias;

  /// FFN output projection weights transposed to `[intermediate, H]`.
  final NDArray<Float32> outputWeightT;

  /// FFN output projection bias `[H]`.
  final NDArray<Float32> outputBias;

  /// Post-FFN LayerNorm scale `[H]`.
  final NDArray<Float32> outputLayerNormWeight;

  /// Post-FFN LayerNorm shift `[H]`.
  final NDArray<Float32> outputLayerNormBias;

  /// Creates a layer weight container.
  BertLayerWeights({
    required this.queryWeightT,
    required this.queryBias,
    required this.keyWeightT,
    required this.keyBias,
    required this.valueWeightT,
    required this.valueBias,
    required this.attnOutputWeightT,
    required this.attnOutputBias,
    required this.attnLayerNormWeight,
    required this.attnLayerNormBias,
    required this.intermediateWeightT,
    required this.intermediateBias,
    required this.outputWeightT,
    required this.outputBias,
    required this.outputLayerNormWeight,
    required this.outputLayerNormBias,
  });

  /// Disposes all native weight tensors owned by this layer.
  void dispose() {
    queryWeightT.dispose();
    queryBias.dispose();
    keyWeightT.dispose();
    keyBias.dispose();
    valueWeightT.dispose();
    valueBias.dispose();
    attnOutputWeightT.dispose();
    attnOutputBias.dispose();
    attnLayerNormWeight.dispose();
    attnLayerNormBias.dispose();
    intermediateWeightT.dispose();
    intermediateBias.dispose();
    outputWeightT.dispose();
    outputBias.dispose();
    outputLayerNormWeight.dispose();
    outputLayerNormBias.dispose();
  }
}

/// Complete set of native `Float32` BERT weights loaded from `.safetensors`.
final class BertWeights {
  /// Token embedding table `[vocabSize, H]`.
  final NDArray<Float32> wordEmbeddings;

  /// Positional embedding table `[maxPositionEmbeddings, H]`.
  final NDArray<Float32> positionEmbeddings;

  /// Token-type embedding table `[typeVocabSize, H]`.
  final NDArray<Float32> tokenTypeEmbeddings;

  /// Embedding LayerNorm scale `[H]`.
  final NDArray<Float32> embeddingLayerNormWeight;

  /// Embedding LayerNorm shift `[H]`.
  final NDArray<Float32> embeddingLayerNormBias;

  /// Per-layer transformer weights (length `numHiddenLayers`).
  final List<BertLayerWeights> layers;

  bool _isDisposed = false;

  /// Creates a [BertWeights] bundle.
  BertWeights({
    required this.wordEmbeddings,
    required this.positionEmbeddings,
    required this.tokenTypeEmbeddings,
    required this.embeddingLayerNormWeight,
    required this.embeddingLayerNormBias,
    required List<BertLayerWeights> layers,
  }) : layers = List<BertLayerWeights>.unmodifiable(layers);

  /// Whether [dispose] has been called.
  bool get isDisposed => _isDisposed;

  /// Loads BERT weights from a [SafetensorsFile] matching [config].
  ///
  /// Automatically handles both unprefixed (`embeddings.*`, `encoder.*`, as in
  /// `sentence-transformers/all-MiniLM-L6-v2`) and `bert.`-prefixed checkpoints,
  /// as well as both `LayerNorm.{weight,bias}` and `LayerNorm.{gamma,beta}`
  /// parameter names.
  ///
  /// All loaded arrays are detached from any enclosing [NDArray.scope] and
  /// owned by the returned [BertWeights]; call [dispose] when finished.
  factory BertWeights.fromSafetensors(
    SafetensorsFile safetensors,
    BertConfig config,
  ) {
    final prefix =
        safetensors.contains('bert.embeddings.word_embeddings.weight')
        ? 'bert.'
        : '';
    final owned = <NDArray<Float32>>[];

    NDArray<Float32> loadExact(String key, List<int> expectedShape) {
      final fullKey = '$prefix$key';
      if (!safetensors.contains(fullKey)) {
        throw FormatException(
          'Missing required tensor "$fullKey" in safetensors file.',
        );
      }
      final arr = safetensors.loadFloat32(fullKey);
      arr.detachFromScope();
      owned.add(arr);
      if (!_shapeEquals(arr.shape, expectedShape)) {
        throw FormatException(
          'Tensor "$fullKey" has shape ${arr.shape}, expected $expectedShape.',
        );
      }
      return arr;
    }

    NDArray<Float32> loadAny(List<String> candidates, List<int> expectedShape) {
      for (final key in candidates) {
        final fullKey = '$prefix$key';
        if (safetensors.contains(fullKey)) {
          return loadExact(key, expectedShape);
        }
      }
      throw FormatException(
        'Missing required tensor (tried ${candidates.map((c) => '"$prefix$c"').join(', ')}).',
      );
    }

    NDArray<Float32> loadTransposed(String key, List<int> storedShape) {
      final fullKey = '$prefix$key';
      if (!safetensors.contains(fullKey)) {
        throw FormatException(
          'Missing required tensor "$fullKey" in safetensors file.',
        );
      }
      final transposed = NDArray.scope(() {
        final raw = safetensors.loadFloat32(fullKey);
        if (!_shapeEquals(raw.shape, storedShape)) {
          throw FormatException(
            'Tensor "$fullKey" has shape ${raw.shape}, expected $storedShape.',
          );
        }
        return raw.transposed.copy().detachToParentScope();
      });
      transposed.detachFromScope();
      owned.add(transposed);
      return transposed;
    }

    final h = config.hiddenSize;
    final inter = config.intermediateSize;

    try {
      final wordEmb = loadExact('embeddings.word_embeddings.weight', [
        config.vocabSize,
        h,
      ]);
      final posEmb = loadExact('embeddings.position_embeddings.weight', [
        config.maxPositionEmbeddings,
        h,
      ]);
      final tokEmb = loadExact('embeddings.token_type_embeddings.weight', [
        config.typeVocabSize,
        h,
      ]);
      final embLnW = loadAny(
        ['embeddings.LayerNorm.weight', 'embeddings.LayerNorm.gamma'],
        [h],
      );
      final embLnB = loadAny(
        ['embeddings.LayerNorm.bias', 'embeddings.LayerNorm.beta'],
        [h],
      );

      final layers = <BertLayerWeights>[];
      for (var i = 0; i < config.numHiddenLayers; i++) {
        final p = 'encoder.layer.$i';
        layers.add(
          BertLayerWeights(
            queryWeightT: loadTransposed('$p.attention.self.query.weight', [
              h,
              h,
            ]),
            queryBias: loadExact('$p.attention.self.query.bias', [h]),
            keyWeightT: loadTransposed('$p.attention.self.key.weight', [h, h]),
            keyBias: loadExact('$p.attention.self.key.bias', [h]),
            valueWeightT: loadTransposed('$p.attention.self.value.weight', [
              h,
              h,
            ]),
            valueBias: loadExact('$p.attention.self.value.bias', [h]),
            attnOutputWeightT: loadTransposed(
              '$p.attention.output.dense.weight',
              [h, h],
            ),
            attnOutputBias: loadExact('$p.attention.output.dense.bias', [h]),
            attnLayerNormWeight: loadAny(
              [
                '$p.attention.output.LayerNorm.weight',
                '$p.attention.output.LayerNorm.gamma',
              ],
              [h],
            ),
            attnLayerNormBias: loadAny(
              [
                '$p.attention.output.LayerNorm.bias',
                '$p.attention.output.LayerNorm.beta',
              ],
              [h],
            ),
            intermediateWeightT: loadTransposed(
              '$p.intermediate.dense.weight',
              [inter, h],
            ),
            intermediateBias: loadExact('$p.intermediate.dense.bias', [inter]),
            outputWeightT: loadTransposed('$p.output.dense.weight', [h, inter]),
            outputBias: loadExact('$p.output.dense.bias', [h]),
            outputLayerNormWeight: loadAny(
              ['$p.output.LayerNorm.weight', '$p.output.LayerNorm.gamma'],
              [h],
            ),
            outputLayerNormBias: loadAny(
              ['$p.output.LayerNorm.bias', '$p.output.LayerNorm.beta'],
              [h],
            ),
          ),
        );
      }

      return BertWeights(
        wordEmbeddings: wordEmb,
        positionEmbeddings: posEmb,
        tokenTypeEmbeddings: tokEmb,
        embeddingLayerNormWeight: embLnW,
        embeddingLayerNormBias: embLnB,
        layers: layers,
      );
    } catch (_) {
      for (final arr in owned) {
        if (!arr.isDisposed) arr.dispose();
      }
      rethrow;
    }
  }

  /// Disposes all native weight tensors owned by this model.
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    wordEmbeddings.dispose();
    positionEmbeddings.dispose();
    tokenTypeEmbeddings.dispose();
    embeddingLayerNormWeight.dispose();
    embeddingLayerNormBias.dispose();
    for (final layer in layers) {
      layer.dispose();
    }
  }
}

/// Pure-Dart BERT transformer encoder running end-to-end in `NDArray<Float32>`.
///
/// Designed around `sentence-transformers/all-MiniLM-L6-v2` (mean pooling +
/// L2 normalization over token embeddings) while supporting any standard BERT
/// encoder configuration.
final class BertEncoder {
  /// Model hyperparameters.
  final BertConfig config;

  /// Native `Float32` weight tensors.
  final BertWeights weights;

  /// Creates an encoder backed by [config] and [weights].
  BertEncoder({required this.config, required this.weights});

  /// Convenience loader for a local Hugging Face model directory containing
  /// `config.json`, `model.safetensors`, and `tokenizer.json` (or `vocab.txt`).
  static ({BertEncoder encoder, WordPieceTokenizer tokenizer}) fromDirectory(
    Directory dir,
  ) {
    final configFile = File('${dir.path}/config.json');
    final weightsFile = File('${dir.path}/model.safetensors');
    final tokenizerJson = File('${dir.path}/tokenizer.json');
    final vocabTxt = File('${dir.path}/vocab.txt');

    if (!configFile.existsSync()) {
      throw FileSystemException('Missing config.json', configFile.path);
    }
    if (!weightsFile.existsSync()) {
      throw FileSystemException('Missing model.safetensors', weightsFile.path);
    }

    final config = BertConfig.fromJsonFile(configFile);
    final WordPieceTokenizer tokenizer;
    if (tokenizerJson.existsSync()) {
      tokenizer = WordPieceTokenizer.fromTokenizerJsonFile(tokenizerJson);
    } else if (vocabTxt.existsSync()) {
      tokenizer = WordPieceTokenizer.fromVocabTxtFile(vocabTxt);
    } else {
      throw FileSystemException(
        'Missing tokenizer.json or vocab.txt',
        dir.path,
      );
    }

    final st = SafetensorsFile.open(weightsFile);
    try {
      final weights = BertWeights.fromSafetensors(st, config);
      return (
        encoder: BertEncoder(config: config, weights: weights),
        tokenizer: tokenizer,
      );
    } finally {
      st.close();
    }
  }

  /// Disposes the underlying [weights].
  void dispose() => weights.dispose();

  /// Runs the full BERT encoder forward pass and returns contextualized token
  /// hidden states of shape `[batchSize, sequenceLength, hiddenSize]`.
  ///
  /// All intermediate activations are allocated inside per-layer [NDArray.scope]
  /// blocks so peak native memory stays $O(1)$ in layer depth.
  NDArray<Float32> forward(TokenizedBatch batch) {
    if (weights.isDisposed) {
      throw StateError('Cannot run forward() on a disposed BertEncoder.');
    }
    final b = batch.batchSize;
    final s = batch.sequenceLength;
    final h = config.hiddenSize;
    if (s > config.maxPositionEmbeddings) {
      throw ArgumentError.value(
        s,
        'batch.sequenceLength',
        'Must not exceed maxPositionEmbeddings (${config.maxPositionEmbeddings})',
      );
    }

    final flatIds = <int>[];
    final flatTypes = <int>[];
    final attnBiasData = Float32List(b * s);
    var offset = 0;
    for (var i = 0; i < b; i++) {
      final idsRow = batch.inputIds[i];
      final maskRow = batch.attentionMask[i];
      final typeRow = batch.tokenTypeIds[i];
      for (var j = 0; j < s; j++) {
        final id = idsRow[j];
        final ty = typeRow[j];
        if (id < 0 || id >= config.vocabSize) {
          throw RangeError.range(id, 0, config.vocabSize - 1, 'inputIds');
        }
        if (ty < 0 || ty >= config.typeVocabSize) {
          throw RangeError.range(
            ty,
            0,
            config.typeVocabSize - 1,
            'tokenTypeIds',
          );
        }
        flatIds.add(id);
        flatTypes.add(ty);
        attnBiasData[offset++] = maskRow[j] == 0 ? -10000.0 : 0.0;
      }
    }

    return NDArray.scope(() {
      final attnBias = NDArray<Float32>.fromTypedData(attnBiasData, [
        b,
        1,
        1,
        s,
      ], DType.float32);

      // Embedding lookup + LayerNorm inside its own scope so lookup tables
      // and intermediate sums are freed before layer 0 begins.
      var hidden = NDArray.scope(() {
        final word = weights.wordEmbeddings.take(flatIds, axis: 0).reshape([
          b,
          s,
          h,
        ]);
        final pos = weights.positionEmbeddings
            .slice([Slice(start: 0, stop: s), Slice.all()])
            .reshape([1, s, h]);
        final tok = weights.tokenTypeEmbeddings
            .take(flatTypes, axis: 0)
            .reshape([b, s, h]);
        final summed = (word + pos) + tok;
        return _layerNorm(
          summed,
          weights.embeddingLayerNormWeight,
          weights.embeddingLayerNormBias,
          config.layerNormEps,
        ).detachToParentScope();
      });

      final scale = 1.0 / math.sqrt(config.headDim);
      for (final layer in weights.layers) {
        final next = _encoderLayer(hidden, layer, attnBias, b, s, scale);
        hidden.dispose();
        hidden = next;
      }

      return hidden.detachToParentScope();
    });
  }

  /// Computes sentence embeddings of shape `[batchSize, hiddenSize]` using
  /// attention-mask-weighted mean pooling and optional L2 normalization
  /// (matching `sentence-transformers/all-MiniLM-L6-v2`).
  NDArray<Float32> embed(TokenizedBatch batch, {bool normalize = true}) {
    return NDArray.scope(() {
      final hidden = forward(batch);
      final b = batch.batchSize;
      final s = batch.sequenceLength;

      final maskData = Float32List(b * s);
      var k = 0;
      for (var i = 0; i < b; i++) {
        for (var j = 0; j < s; j++) {
          maskData[k++] = batch.attentionMask[i][j] == 0 ? 0.0 : 1.0;
        }
      }
      final mask = NDArray<Float32>.fromTypedData(maskData, [
        b,
        s,
        1,
      ], DType.float32);

      final maskedHidden = hidden * mask;
      final summed = sum(maskedHidden, axis: 1); // [B, H]
      final counts = clip(sum(mask, axis: 1), min: 1e-9); // [B, 1]
      var pooled = summed / counts; // [B, H]

      if (normalize) {
        final sq = pooled * pooled;
        final norms = sqrt(sum(sq, axis: -1, keepDims: true));
        final safeNorms = clip(norms, min: 1e-12);
        pooled = pooled / safeNorms;
      }
      return pooled.detachToParentScope();
    });
  }

  /// Tokenizes [texts] with [tokenizer] and returns sentence embeddings of
  /// shape `[texts.length, hiddenSize]`.
  NDArray<Float32> embedTexts(
    WordPieceTokenizer tokenizer,
    List<String> texts, {
    int maxLength = 256,
    bool normalize = true,
  }) {
    final batch = tokenizer.encodeBatch(texts, maxLength: maxLength);
    return embed(batch, normalize: normalize);
  }

  NDArray<Float32> _encoderLayer(
    NDArray<Float32> hidden,
    BertLayerWeights layer,
    NDArray<Float32> attnBias,
    int b,
    int s,
    double scale,
  ) {
    final h = config.hiddenSize;
    final numHeads = config.numAttentionHeads;
    final headDim = config.headDim;

    return NDArray.scope(() {
      // Multi-head self-attention.
      final q = _linear(
        hidden,
        layer.queryWeightT,
        layer.queryBias,
      ).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
      final k = _linear(
        hidden,
        layer.keyWeightT,
        layer.keyBias,
      ).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
      final v = _linear(
        hidden,
        layer.valueWeightT,
        layer.valueBias,
      ).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);

      final kT = k.transpose([0, 1, 3, 2]);
      final rawScores = matmul(q, kT) * scale;
      final maskedScores = rawScores + attnBias;
      final probs = _softmaxLastAxis(maskedScores);

      final context = matmul(
        probs,
        v,
      ).transpose([0, 2, 1, 3]).copy().reshape([b, s, h]);

      final attnProj = _linear(
        context,
        layer.attnOutputWeightT,
        layer.attnOutputBias,
      );
      final attnResidual = _layerNorm(
        hidden + attnProj,
        layer.attnLayerNormWeight,
        layer.attnLayerNormBias,
        config.layerNormEps,
      );

      // Feed-forward network with exact GELU activation.
      final inter = _linear(
        attnResidual,
        layer.intermediateWeightT,
        layer.intermediateBias,
      );
      final activated = _geluExact(inter);
      final ffnProj = _linear(activated, layer.outputWeightT, layer.outputBias);
      final out = _layerNorm(
        attnResidual + ffnProj,
        layer.outputLayerNormWeight,
        layer.outputLayerNormBias,
        config.layerNormEps,
      );

      return out.detachToParentScope();
    });
  }
}

/// Applies `y = x @ weightT + bias` to a 3D `[B, S, Din]` tensor using a
/// single flattened 2D `matmul` (`cblas_sgemm`).
NDArray<Float32> _linear(
  NDArray<Float32> x,
  NDArray<Float32> weightT,
  NDArray<Float32> bias,
) {
  final b = x.shape[0];
  final s = x.shape[1];
  final dIn = x.shape[2];
  final dOut = weightT.shape[1];
  final x2d = (x.isContiguous ? x : x.copy()).reshape([b * s, dIn]);
  final proj = matmul(x2d, weightT) + bias;
  return proj.reshape([b, s, dOut]);
}

/// Pure-`Float32` LayerNorm over the last axis (`-1`):
/// `y = ((x - mean) / sqrt(var + eps)) * weight + bias`.
///
/// Uses `sum(...) * (1 / H)` instead of `mean`/`variance` so computation stays
/// in `Float32` without promoting to `Float64`.
NDArray<Float32> _layerNorm(
  NDArray<Float32> x,
  NDArray<Float32> weight,
  NDArray<Float32> bias,
  double eps,
) {
  final invH = 1.0 / x.shape.last;
  return NDArray.scope(() {
    final mu = sum(x, axis: -1, keepDims: true) * invH;
    final diff = x - mu;
    final variance = sum(diff * diff, axis: -1, keepDims: true) * invH;
    final denom = sqrt(variance + eps);
    final normed = (diff / denom) * weight + bias;
    return normed.detachToParentScope();
  });
}

/// Numerically stable softmax along the last axis (`-1`).
NDArray<Float32> _softmaxLastAxis(NDArray<Float32> logits) {
  return NDArray.scope(() {
    final maxVal = max(logits, axis: -1, keepDims: true);
    final shifted = logits - maxVal;
    final exps = exp(shifted);
    final denom = sum(exps, axis: -1, keepDims: true);
    return (exps / denom).detachToParentScope();
  });
}

/// Exact Gaussian Error Linear Unit: `0.5 * x * (1 + erf(x / sqrt(2)))`.
NDArray<Float32> _geluExact(NDArray<Float32> x) {
  const invSqrt2 = 0.7071067811865475; // 1 / sqrt(2)
  return NDArray.scope(() {
    final erfTerm = erf(x * invSqrt2);
    final res = (x * 0.5) * (erfTerm + 1.0);
    return res.detachToParentScope();
  });
}

bool _shapeEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
