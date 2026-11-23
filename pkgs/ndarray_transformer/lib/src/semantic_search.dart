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

import 'dart:math' as math;

import 'package:ndarray/ndarray.dart';

import 'bert_model.dart';
import 'wordpiece_tokenizer.dart';

/// A single ranked semantic-search match.
final class SearchHit {
  /// 0-based index of the document in the [SemanticSearchIndex].
  final int index;

  /// Original document text.
  final String text;

  /// Cosine similarity score in `[-1.0, 1.0]` (higher is more similar).
  final double score;

  /// Creates a search hit.
  const SearchHit({
    required this.index,
    required this.text,
    required this.score,
  });

  @override
  String toString() =>
      'SearchHit(#$index, score: ${score.toStringAsFixed(4)}, "$text")';
}

/// In-memory cosine-similarity semantic search index backed by a contiguous
/// `[numDocs, hiddenSize]` `NDArray<Float32>` embedding matrix.
final class SemanticSearchIndex {
  final List<String> _documents;
  final NDArray<Float32> _embeddings; // L2-normalized [N, H]
  bool _isDisposed = false;

  SemanticSearchIndex._(this._documents, this._embeddings);

  /// Builds an index by encoding [documents] with [encoder] and [tokenizer].
  ///
  /// The returned index owns its embedding matrix and must be disposed via
  /// [dispose] when no longer needed.
  factory SemanticSearchIndex.fromDocuments({
    required BertEncoder encoder,
    required WordPieceTokenizer tokenizer,
    required List<String> documents,
    int maxLength = 256,
  }) {
    if (documents.isEmpty) {
      throw ArgumentError.value(documents, 'documents', 'Must not be empty');
    }
    final embeddings = encoder.embedTexts(
      tokenizer,
      documents,
      maxLength: maxLength,
      normalize: true,
    );
    embeddings.detachFromScope();
    return SemanticSearchIndex._(
      List<String>.unmodifiable(documents),
      embeddings,
    );
  }

  /// Builds an index directly from precomputed 2D [embeddings] (`[N, H]`).
  ///
  /// Copies and L2-normalizes [embeddings] so the index owns its native buffer
  /// independently of caller lifecycle.
  factory SemanticSearchIndex.fromEmbeddings({
    required List<String> documents,
    required NDArray<Float32> embeddings,
  }) {
    if (documents.isEmpty) {
      throw ArgumentError.value(documents, 'documents', 'Must not be empty');
    }
    if (embeddings.rank != 2 || embeddings.shape[0] != documents.length) {
      throw ArgumentError.value(
        embeddings,
        'embeddings',
        'Must be a 2D array of shape [${documents.length}, hiddenSize]',
      );
    }
    final normalized = NDArray.scope(() {
      final norms = clip(
        sqrt(sum(embeddings * embeddings, axis: -1, keepDims: true)),
        min: 1e-12,
      );
      return (embeddings / norms).detachToParentScope();
    });
    normalized.detachFromScope();
    return SemanticSearchIndex._(
      List<String>.unmodifiable(documents),
      normalized,
    );
  }

  /// Indexed documents in insertion order.
  List<String> get documents => _documents;

  /// Number of indexed documents.
  int get length => _documents.length;

  /// Embedding dimensionality (`H`).
  int get embeddingDim => _embeddings.shape[1];

  /// Whether [dispose] has been called.
  bool get isDisposed => _isDisposed;

  /// Ranks the [topK] most similar documents to a single 1D (`[H]`) or 2D
  /// (`[1, H]`) [queryEmbedding] by cosine similarity.
  List<SearchHit> searchEmbedding(
    NDArray<Float32> queryEmbedding, {
    int topK = 5,
  }) {
    if (_isDisposed) {
      throw StateError('Cannot search a disposed SemanticSearchIndex.');
    }
    if (topK <= 0) {
      throw ArgumentError.value(topK, 'topK', 'Must be positive');
    }

    return NDArray.scope(() {
      final NDArray<Float32> q2d;
      if (queryEmbedding.rank == 1 && queryEmbedding.shape[0] == embeddingDim) {
        q2d = queryEmbedding.reshape([1, embeddingDim]);
      } else if (queryEmbedding.rank == 2 &&
          queryEmbedding.shape[0] == 1 &&
          queryEmbedding.shape[1] == embeddingDim) {
        q2d = queryEmbedding;
      } else {
        throw ArgumentError.value(
          queryEmbedding,
          'queryEmbedding',
          'Must have shape [$embeddingDim] or [1, $embeddingDim]',
        );
      }

      final qNorm = clip(
        sqrt(sum(q2d * q2d, axis: -1, keepDims: true)),
        min: 1e-12,
      );
      final qUnit = q2d / qNorm;
      final scores = matmul(qUnit, _embeddings.transposed).reshape([length]);
      final order = argsort(scores);
      final k = math.min(topK, length);
      final hits = <SearchHit>[];
      for (var rank = 0; rank < k; rank++) {
        final docIdx = order.getCellFlat(length - 1 - rank);
        hits.add(
          SearchHit(
            index: docIdx,
            text: _documents[docIdx],
            score: scores.getCellFlat(docIdx),
          ),
        );
      }
      return hits;
    });
  }

  /// Encodes [query] with [encoder] and [tokenizer] and returns the [topK]
  /// highest-scoring documents.
  List<SearchHit> searchText({
    required BertEncoder encoder,
    required WordPieceTokenizer tokenizer,
    required String query,
    int topK = 5,
    int maxLength = 256,
  }) {
    return NDArray.scope(() {
      final qEmb = encoder.embedTexts(
        tokenizer,
        [query],
        maxLength: maxLength,
        normalize: true,
      );
      return searchEmbedding(qEmb, topK: topK);
    });
  }

  /// Releases the underlying native embedding matrix.
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    _embeddings.dispose();
  }
}
