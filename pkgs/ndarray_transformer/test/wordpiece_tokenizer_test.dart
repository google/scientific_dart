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

import 'package:ndarray_transformer/ndarray_transformer.dart';
import 'package:test/test.dart';

void main() {
  group('WordPieceTokenizer', () {
    final vocab = <String, int>{
      '[PAD]': 0,
      '[UNK]': 1,
      '[CLS]': 2,
      '[SEP]': 3,
      'hello': 4,
      ',': 5,
      'world': 6,
      '!': 7,
      'play': 8,
      '##ing': 9,
      'cafe': 10,
      '中': 11,
      '文': 12,
      'dart': 13,
      'is': 14,
      'fast': 15,
    };

    test('tokenizes punctuation, accents, CJK, and ## subwords', () {
      final tok = WordPieceTokenizer.fromVocab(vocab);
      expect(tok.tokenize('Hello, World!'), ['hello', ',', 'world', '!']);
      expect(tok.tokenize('Café playing'), ['cafe', 'play', '##ing']);
      expect(tok.tokenize('中文!'), ['中', '文', '!']);
      expect(tok.tokenize('unseenword'), ['[UNK]']);
    });

    test('loads from Hugging Face tokenizer.json and vocab.txt', () {
      final jsonStr = jsonEncode({
        'normalizer': {'lowercase': true, 'strip_accents': true},
        'model': {
          'type': 'WordPiece',
          'unk_token': '[UNK]',
          'continuing_subword_prefix': '##',
          'max_input_chars_per_word': 100,
          'vocab': vocab,
        },
      });
      final fromJson = WordPieceTokenizer.fromTokenizerJson(jsonStr);
      expect(fromJson.vocabSize, vocab.length);
      expect(fromJson.encode('Dart is fast').inputIds, [2, 13, 14, 15, 3]);

      final vocabTxt = vocab.keys.join('\n');
      final fromTxt = WordPieceTokenizer.fromVocabTxt(vocabTxt);
      expect(fromTxt.encode('Dart is fast').inputIds, [2, 13, 14, 15, 3]);
    });

    test('encodes sentence pairs and pads batches with attention masks', () {
      final tok = WordPieceTokenizer.fromVocab(vocab);
      final pair = tok.encode('hello world', textPair: 'dart fast');
      expect(pair.inputIds, [2, 4, 6, 3, 13, 15, 3]);
      expect(pair.tokenTypeIds, [0, 0, 0, 0, 1, 1, 1]);
      expect(pair.attentionMask, [1, 1, 1, 1, 1, 1, 1]);

      final batch = tok.encodeBatch(['hello', 'dart is fast']);
      expect(batch.batchSize, 2);
      expect(batch.sequenceLength, 5);
      expect(batch.inputIds[0], [2, 4, 3, 0, 0]);
      expect(batch.attentionMask[0], [1, 1, 1, 0, 0]);
      expect(batch.inputIds[1], [2, 13, 14, 15, 3]);
      expect(batch.attentionMask[1], [1, 1, 1, 1, 1]);

      final truncated = tok.encode('hello world dart is fast', maxLength: 4);
      expect(truncated.inputIds, [2, 4, 6, 3]);
    });
  });
}
