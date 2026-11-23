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
import 'package:ndarray_transformer/ndarray_transformer.dart';
import 'package:test/test.dart';

/// Deterministic synthetic BERT model in `.safetensors` bytes for repeatable
/// parity, lifecycle, and semantic-search verification.
({BertConfig config, Uint8List safetensorsBytes, WordPieceTokenizer tokenizer})
_makeTinyBert({bool bertPrefix = false, bool legacyLayerNormNames = false}) {
  final tokens = <String>[
    '[PAD]',
    '[UNK]',
    '[CLS]',
    '[SEP]',
    'dart',
    'array',
    'matrix',
    'fast',
    'bread',
    'oven',
    'neural',
    'search',
    'embed',
    '##ding',
    'query',
    'vector',
  ];
  final vocab = <String, int>{
    for (var i = 0; i < tokens.length; i++) tokens[i]: i,
  };
  final tokenizer = WordPieceTokenizer.fromVocab(vocab);
  final config = BertConfig(
    vocabSize: vocab.length,
    hiddenSize: 32,
    numHiddenLayers: 2,
    numAttentionHeads: 4,
    intermediateSize: 64,
    maxPositionEmbeddings: 16,
    typeVocabSize: 2,
    layerNormEps: 1e-12,
  );

  final rng = math.Random(12345);
  NDArray<Float32> rand(List<int> shape, double scale) {
    final n = shape.fold<int>(1, (a, b) => a * b);
    final data = Float32List(n);
    for (var i = 0; i < n; i++) {
      data[i] = (rng.nextDouble() * 2.0 - 1.0) * scale;
    }
    return NDArray<Float32>.fromTypedData(data, shape, DType.float32);
  }

  final pfx = bertPrefix ? 'bert.' : '';
  final lnW = legacyLayerNormNames ? 'gamma' : 'weight';
  final lnB = legacyLayerNormNames ? 'beta' : 'bias';

  final bytes = NDArray.scope(() {
    final h = config.hiddenSize;
    final inter = config.intermediateSize;
    final tensors = <String, NDArray<DTypeTag>>{
      '${pfx}embeddings.word_embeddings.weight': rand([
        config.vocabSize,
        h,
      ], 0.3),
      '${pfx}embeddings.position_embeddings.weight': rand([
        config.maxPositionEmbeddings,
        h,
      ], 0.1),
      '${pfx}embeddings.token_type_embeddings.weight': rand([
        config.typeVocabSize,
        h,
      ], 0.05),
      '${pfx}embeddings.LayerNorm.$lnW': rand([h], 0.2) + 1.0,
      '${pfx}embeddings.LayerNorm.$lnB': rand([h], 0.05),
    };
    for (var i = 0; i < config.numHiddenLayers; i++) {
      final p = '${pfx}encoder.layer.$i';
      tensors['$p.attention.self.query.weight'] = rand([h, h], 0.15);
      tensors['$p.attention.self.query.bias'] = rand([h], 0.05);
      tensors['$p.attention.self.key.weight'] = rand([h, h], 0.15);
      tensors['$p.attention.self.key.bias'] = rand([h], 0.05);
      tensors['$p.attention.self.value.weight'] = rand([h, h], 0.15);
      tensors['$p.attention.self.value.bias'] = rand([h], 0.05);
      tensors['$p.attention.output.dense.weight'] = rand([h, h], 0.15);
      tensors['$p.attention.output.dense.bias'] = rand([h], 0.05);
      tensors['$p.attention.output.LayerNorm.$lnW'] = rand([h], 0.2) + 1.0;
      tensors['$p.attention.output.LayerNorm.$lnB'] = rand([h], 0.05);
      tensors['$p.intermediate.dense.weight'] = rand([inter, h], 0.15);
      tensors['$p.intermediate.dense.bias'] = rand([inter], 0.05);
      tensors['$p.output.dense.weight'] = rand([h, inter], 0.15);
      tensors['$p.output.dense.bias'] = rand([h], 0.05);
      tensors['$p.output.LayerNorm.$lnW'] = rand([h], 0.2) + 1.0;
      tensors['$p.output.LayerNorm.$lnB'] = rand([h], 0.05);
    }
    return SafetensorsFile.encode(tensors, metadata: {'format': 'pt'});
  });

  return (config: config, safetensorsBytes: bytes, tokenizer: tokenizer);
}

/// Independent pure-NumPy BERT forward pass used to verify exact numerical
/// parity against [BertEncoder].
const String _pythonBertRef = r'''
import json, math, struct, sys
import numpy as np

st_path, req_path = sys.argv[1], sys.argv[2]
req = json.loads(open(req_path, "r", encoding="utf-8").read())

raw = open(st_path, "rb").read()
hlen = struct.unpack_from("<Q", raw, 0)[0]
header = json.loads(raw[8:8 + hlen].decode("utf-8"))
data_base = 8 + hlen

def load_f32(name):
    info = header[name]
    s, e = info["data_offsets"]
    return np.frombuffer(raw, dtype=np.float32, count=(e - s) // 4, offset=data_base + s).reshape(info["shape"]).copy()

def layer_norm(x, w, b, eps):
    mu = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True, dtype=np.float32)
    return (x - mu) / np.sqrt(var + np.float32(eps)) * w + b

def gelu(x):
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    erf_vec = np.vectorize(math.erf, otypes=[np.float32])
    return np.float32(0.5) * x * (np.float32(1.0) + erf_vec(x * inv_sqrt2))

def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)

cfg = req["config"]
input_ids = np.array(req["input_ids"], dtype=np.int64)
attention_mask = np.array(req["attention_mask"], dtype=np.float32)
token_type_ids = np.array(req["token_type_ids"], dtype=np.int64)
B, S = input_ids.shape
H = cfg["hidden_size"]
heads = cfg["num_attention_heads"]
head_dim = H // heads
eps = cfg["layer_norm_eps"]

word_emb = load_f32("embeddings.word_embeddings.weight")[input_ids]
pos_emb = load_f32("embeddings.position_embeddings.weight")[np.arange(S)][None, :, :]
tok_emb = load_f32("embeddings.token_type_embeddings.weight")[token_type_ids]
hidden = layer_norm(
    word_emb + pos_emb + tok_emb,
    load_f32("embeddings.LayerNorm.weight"),
    load_f32("embeddings.LayerNorm.bias"),
    eps,
)

attn_bias = np.where(attention_mask[:, None, None, :] == 0.0, np.float32(-10000.0), np.float32(0.0))

for i in range(cfg["num_hidden_layers"]):
    p = f"encoder.layer.{i}"
    q = (hidden @ load_f32(f"{p}.attention.self.query.weight").T + load_f32(f"{p}.attention.self.query.bias")).reshape(B, S, heads, head_dim).transpose(0, 2, 1, 3)
    k = (hidden @ load_f32(f"{p}.attention.self.key.weight").T + load_f32(f"{p}.attention.self.key.bias")).reshape(B, S, heads, head_dim).transpose(0, 2, 1, 3)
    v = (hidden @ load_f32(f"{p}.attention.self.value.weight").T + load_f32(f"{p}.attention.self.value.bias")).reshape(B, S, heads, head_dim).transpose(0, 2, 1, 3)
    scores = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / math.sqrt(head_dim)) + attn_bias
    probs = softmax(scores)
    ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(B, S, H)
    attn_out = ctx @ load_f32(f"{p}.attention.output.dense.weight").T + load_f32(f"{p}.attention.output.dense.bias")
    hidden = layer_norm(hidden + attn_out, load_f32(f"{p}.attention.output.LayerNorm.weight"), load_f32(f"{p}.attention.output.LayerNorm.bias"), eps)
    inter = gelu(hidden @ load_f32(f"{p}.intermediate.dense.weight").T + load_f32(f"{p}.intermediate.dense.bias"))
    ffn_out = inter @ load_f32(f"{p}.output.dense.weight").T + load_f32(f"{p}.output.dense.bias")
    hidden = layer_norm(hidden + ffn_out, load_f32(f"{p}.output.LayerNorm.weight"), load_f32(f"{p}.output.LayerNorm.bias"), eps)

mask_exp = attention_mask[:, :, None]
pooled = np.sum(hidden * mask_exp, axis=1) / np.clip(np.sum(mask_exp, axis=1), 1e-9, None)
norms = np.clip(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12, None)
normed = pooled / norms

print(json.dumps({
    "hidden": hidden.astype(np.float64).ravel().tolist(),
    "embeddings": normed.astype(np.float64).ravel().tolist(),
}))
''';

void main() {
  group('BertEncoder & SemanticSearchIndex', () {
    test(
      'matches independent Python/NumPy BERT forward pass and mean-pooled L2 embeddings',
      () {
        final probe = Process.runSync('python3', ['-c', 'import numpy']);
        if (probe.exitCode != 0) {
          markTestSkipped('python3 with numpy is not available on this host');
          return;
        }

        final tiny = _makeTinyBert();
        final tmp = Directory.systemTemp.createTempSync('bert_parity_');
        try {
          final stFile = File('${tmp.path}/model.safetensors')
            ..writeAsBytesSync(tiny.safetensorsBytes);
          final batch = tiny.tokenizer.encodeBatch([
            'dart array fast',
            'neural embedding search query vector',
            'bread oven',
          ]);
          final reqFile = File('${tmp.path}/req.json')
            ..writeAsStringSync(
              jsonEncode({
                'config': {
                  'hidden_size': tiny.config.hiddenSize,
                  'num_hidden_layers': tiny.config.numHiddenLayers,
                  'num_attention_heads': tiny.config.numAttentionHeads,
                  'layer_norm_eps': tiny.config.layerNormEps,
                },
                'input_ids': batch.inputIds,
                'attention_mask': batch.attentionMask,
                'token_type_ids': batch.tokenTypeIds,
              }),
            );

          final py = Process.runSync('python3', [
            '-c',
            _pythonBertRef,
            stFile.path,
            reqFile.path,
          ]);
          expect(
            py.exitCode,
            0,
            reason: 'Python reference failed: ${py.stderr}',
          );
          final ref =
              jsonDecode((py.stdout as String).trim()) as Map<String, dynamic>;
          final refHidden = (ref['hidden'] as List<dynamic>)
              .map((v) => (v as num).toDouble())
              .toList();
          final refEmb = (ref['embeddings'] as List<dynamic>)
              .map((v) => (v as num).toDouble())
              .toList();

          final st = SafetensorsFile.fromBytes(tiny.safetensorsBytes);
          final weights = BertWeights.fromSafetensors(st, tiny.config);
          final encoder = BertEncoder(config: tiny.config, weights: weights);
          try {
            NDArray.scope(() {
              final hidden = encoder.forward(batch);
              expect(hidden.shape, [
                batch.batchSize,
                batch.sequenceLength,
                tiny.config.hiddenSize,
              ]);
              var maxHiddenDiff = 0.0;
              for (var i = 0; i < hidden.size; i++) {
                final diff = (hidden.getCellFlat(i) - refHidden[i]).abs();
                if (diff > maxHiddenDiff) maxHiddenDiff = diff;
              }
              expect(
                maxHiddenDiff,
                lessThan(1e-5),
                reason: 'Max hidden-state diff vs NumPy: $maxHiddenDiff',
              );

              final emb = encoder.embed(batch, normalize: true);
              expect(emb.shape, [batch.batchSize, tiny.config.hiddenSize]);
              var maxEmbDiff = 0.0;
              for (var i = 0; i < emb.size; i++) {
                final diff = (emb.getCellFlat(i) - refEmb[i]).abs();
                if (diff > maxEmbDiff) maxEmbDiff = diff;
              }
              expect(
                maxEmbDiff,
                lessThan(1e-5),
                reason: 'Max pooled-embedding diff vs NumPy: $maxEmbDiff',
              );

              // Verify each output row is unit L2-norm.
              for (var row = 0; row < batch.batchSize; row++) {
                var normSq = 0.0;
                for (var col = 0; col < tiny.config.hiddenSize; col++) {
                  final v = emb.getCell([row, col]);
                  normSq += v * v;
                }
                expect(math.sqrt(normSq), closeTo(1.0, 1e-5));
              }
            });
          } finally {
            encoder.dispose();
          }
        } finally {
          tmp.deleteSync(recursive: true);
        }
      },
    );

    test(
      'supports bert.-prefixed checkpoints and gamma/beta LayerNorm names',
      () {
        final tiny = _makeTinyBert(
          bertPrefix: true,
          legacyLayerNormNames: true,
        );
        final st = SafetensorsFile.fromBytes(tiny.safetensorsBytes);
        final weights = BertWeights.fromSafetensors(st, tiny.config);
        final encoder = BertEncoder(config: tiny.config, weights: weights);
        try {
          NDArray.scope(() {
            final emb = encoder.embedTexts(tiny.tokenizer, [
              'dart array matrix',
            ]);
            expect(emb.shape, [1, tiny.config.hiddenSize]);
          });
        } finally {
          encoder.dispose();
        }
      },
    );

    test(
      'SemanticSearchIndex ranks exact match #1 with cosine score ~1.0 and leaks nothing',
      () {
        final tiny = _makeTinyBert();
        final st = SafetensorsFile.fromBytes(tiny.safetensorsBytes);
        final weights = BertWeights.fromSafetensors(st, tiny.config);
        final encoder = BertEncoder(config: tiny.config, weights: weights);
        final docs = [
          'dart array matrix fast',
          'bread oven',
          'neural embedding search query vector',
        ];
        final index = SemanticSearchIndex.fromDocuments(
          encoder: encoder,
          tokenizer: tiny.tokenizer,
          documents: docs,
        );
        try {
          final hits = index.searchText(
            encoder: encoder,
            tokenizer: tiny.tokenizer,
            query: 'neural embedding search query vector',
            topK: 2,
          );
          expect(hits.length, 2);
          expect(hits.first.index, 2);
          expect(hits.first.score, closeTo(1.0, 1e-5));
          expect(hits[0].score, greaterThan(hits[1].score));
        } finally {
          index.dispose();
          encoder.dispose();
        }
        if (NDArray.trackAllocations) {
          expect(NDArray.checkNoLeaks(), isTrue);
        }
      },
    );
  });
}
