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

/// Padded batch of tokenized BERT input sequences ready for [BertEncoder].
final class TokenizedBatch {
  /// Token IDs of shape `[batchSize, sequenceLength]`.
  final List<List<int>> inputIds;

  /// Attention mask (`1` for real tokens, `0` for padding) of shape
  /// `[batchSize, sequenceLength]`.
  final List<List<int>> attentionMask;

  /// Segment / token-type IDs (`0` for first segment, `1` for optional second
  /// segment) of shape `[batchSize, sequenceLength]`.
  final List<List<int>> tokenTypeIds;

  /// Creates a validated batch, defensively copying all inner rows.
  TokenizedBatch({
    required List<List<int>> inputIds,
    required List<List<int>> attentionMask,
    required List<List<int>> tokenTypeIds,
  }) : inputIds = List<List<int>>.unmodifiable(
         inputIds.map(List<int>.unmodifiable),
       ),
       attentionMask = List<List<int>>.unmodifiable(
         attentionMask.map(List<int>.unmodifiable),
       ),
       tokenTypeIds = List<List<int>>.unmodifiable(
         tokenTypeIds.map(List<int>.unmodifiable),
       ) {
    if (this.inputIds.isEmpty) {
      throw ArgumentError.value(inputIds, 'inputIds', 'Must not be empty');
    }
    final seqLen = this.inputIds.first.length;
    if (seqLen == 0) {
      throw ArgumentError.value(
        inputIds,
        'inputIds',
        'Must have non-zero sequence length',
      );
    }
    if (this.attentionMask.length != this.inputIds.length ||
        this.tokenTypeIds.length != this.inputIds.length) {
      throw ArgumentError.value(
        attentionMask,
        'attentionMask',
        'Must match batch size of inputIds (${this.inputIds.length})',
      );
    }
    for (var i = 0; i < this.inputIds.length; i++) {
      if (this.inputIds[i].length != seqLen ||
          this.attentionMask[i].length != seqLen ||
          this.tokenTypeIds[i].length != seqLen) {
        throw ArgumentError.value(
          inputIds,
          'inputIds',
          'Must be rectangular with uniform sequence length $seqLen',
        );
      }
    }
  }

  /// Number of sequences in this batch.
  int get batchSize => inputIds.length;

  /// Padded token count per sequence.
  int get sequenceLength => inputIds.first.length;
}

/// Pure-Dart BERT Basic + WordPiece subword tokenizer compatible with
/// Hugging Face `tokenizer.json` and `vocab.txt` files (including
/// `sentence-transformers/all-MiniLM-L6-v2`).
final class WordPieceTokenizer {
  final Map<String, int> _vocab;
  final Map<int, String> _idToToken;

  /// Whether input text is lowercased prior to tokenization.
  final bool lowercase;

  /// Whether Latin accents/diacritics are stripped prior to tokenization.
  final bool stripAccents;

  /// Prefix attached to WordPiece continuation subwords (typically `'##'`).
  final String continuingSubwordPrefix;

  /// Maximum character length of a single word before falling back to `[UNK]`.
  final int maxInputCharsPerWord;

  /// Token ID for `[PAD]`.
  final int padTokenId;

  /// Token ID for `[UNK]`.
  final int unkTokenId;

  /// Token ID for `[CLS]`.
  final int clsTokenId;

  /// Token ID for `[SEP]`.
  final int sepTokenId;

  /// Creates a tokenizer directly from a token-to-ID [vocab] map.
  ///
  /// Copies [vocab] defensively so subsequent caller mutations do not affect
  /// the tokenizer.
  WordPieceTokenizer.fromVocab(
    Map<String, int> vocab, {
    this.lowercase = true,
    this.stripAccents = true,
    this.continuingSubwordPrefix = '##',
    this.maxInputCharsPerWord = 100,
    String padToken = '[PAD]',
    String unkToken = '[UNK]',
    String clsToken = '[CLS]',
    String sepToken = '[SEP]',
  }) : _vocab = Map<String, int>.unmodifiable(vocab),
       _idToToken = Map<int, String>.unmodifiable({
         for (final e in vocab.entries) e.value: e.key,
       }),
       padTokenId = _requireToken(vocab, padToken),
       unkTokenId = _requireToken(vocab, unkToken),
       clsTokenId = _requireToken(vocab, clsToken),
       sepTokenId = _requireToken(vocab, sepToken);

  /// Loads a WordPiece tokenizer from a Hugging Face `tokenizer.json` string.
  ///
  /// Throws a [FormatException] if the JSON structure is missing a WordPiece
  /// vocabulary map.
  factory WordPieceTokenizer.fromTokenizerJson(String jsonSource) {
    final Object? decoded;
    try {
      decoded = jsonDecode(jsonSource);
    } on FormatException catch (e) {
      throw FormatException('Invalid tokenizer.json: ${e.message}');
    }
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('tokenizer.json root must be a JSON object.');
    }

    final model = decoded['model'];
    if (model is! Map<String, dynamic>) {
      throw const FormatException('tokenizer.json missing "model" object.');
    }
    final vocabRaw = model['vocab'];
    if (vocabRaw is! Map<String, dynamic>) {
      throw const FormatException(
        'tokenizer.json missing "model.vocab" dictionary.',
      );
    }
    final vocab = <String, int>{};
    for (final entry in vocabRaw.entries) {
      final id = entry.value;
      if (id is! int || id < 0) {
        throw FormatException(
          'Invalid token ID "$id" for token "${entry.key}".',
        );
      }
      vocab[entry.key] = id;
    }

    var lowercase = true;
    var stripAccents = true;
    final normalizer = decoded['normalizer'];
    if (normalizer is Map<String, dynamic>) {
      final lc = normalizer['lowercase'];
      if (lc is bool) lowercase = lc;
      final sa = normalizer['strip_accents'];
      if (sa is bool) {
        stripAccents = sa;
      } else if (sa == null) {
        stripAccents = lowercase;
      }
    }

    final prefix = (model['continuing_subword_prefix'] as String?) ?? '##';
    final maxChars = (model['max_input_chars_per_word'] as int?) ?? 100;
    final unkToken = (model['unk_token'] as String?) ?? '[UNK]';

    return WordPieceTokenizer.fromVocab(
      vocab,
      lowercase: lowercase,
      stripAccents: stripAccents,
      continuingSubwordPrefix: prefix,
      maxInputCharsPerWord: maxChars,
      unkToken: unkToken,
    );
  }

  /// Loads a WordPiece tokenizer from a Hugging Face `tokenizer.json` [file].
  factory WordPieceTokenizer.fromTokenizerJsonFile(File file) =>
      WordPieceTokenizer.fromTokenizerJson(file.readAsStringSync());

  /// Loads a WordPiece tokenizer from a BERT `vocab.txt` string (one token per
  /// line, with 0-based line indices as token IDs).
  factory WordPieceTokenizer.fromVocabTxt(
    String vocabTxt, {
    bool lowercase = true,
    bool stripAccents = true,
  }) {
    final vocab = <String, int>{};
    final lines = const LineSplitter().convert(vocabTxt);
    for (var i = 0; i < lines.length; i++) {
      final token = lines[i].trimRight();
      if (token.isNotEmpty) {
        vocab[token] = i;
      }
    }
    return WordPieceTokenizer.fromVocab(
      vocab,
      lowercase: lowercase,
      stripAccents: stripAccents,
    );
  }

  /// Loads a WordPiece tokenizer from a BERT `vocab.txt` [file].
  factory WordPieceTokenizer.fromVocabTxtFile(
    File file, {
    bool lowercase = true,
    bool stripAccents = true,
  }) => WordPieceTokenizer.fromVocabTxt(
    file.readAsStringSync(),
    lowercase: lowercase,
    stripAccents: stripAccents,
  );

  /// Total number of tokens in the vocabulary.
  int get vocabSize => _vocab.length;

  /// Looks up the ID for [token], or `null` if not in vocabulary.
  int? tokenToId(String token) => _vocab[token];

  /// Looks up the token string for [id], or `null` if unknown.
  String? idToToken(int id) => _idToToken[id];

  /// Tokenizes [text] into WordPiece subword strings without `[CLS]`/`[SEP]`.
  List<String> tokenize(String text) {
    final cleaned = _cleanText(text);
    final spaced = _tokenizeChineseChars(cleaned);
    final words = _splitWhitespace(spaced);
    final out = <String>[];
    for (final rawWord in words) {
      var word = rawWord;
      if (lowercase) {
        word = word.toLowerCase();
      }
      if (stripAccents) {
        word = _stripLatinAccents(word);
      }
      for (final piece in _splitOnPunctuation(word)) {
        out.addAll(_wordPiece(piece));
      }
    }
    return out;
  }

  /// Encodes [text] (and optional [textPair]) into BERT token IDs, attention
  /// mask, and token-type IDs, truncated to at most [maxLength] tokens
  /// including `[CLS]` and `[SEP]`.
  ///
  /// It is an error if [maxLength] is less than `3` (or `4` when [textPair] is
  /// provided).
  ({List<int> inputIds, List<int> attentionMask, List<int> tokenTypeIds})
  encode(String text, {String? textPair, int maxLength = 256}) {
    final minRequired = textPair == null ? 3 : 4;
    if (maxLength < minRequired) {
      throw ArgumentError.value(
        maxLength,
        'maxLength',
        'Must be at least $minRequired',
      );
    }

    final tokensA = tokenize(text);
    final tokensB = textPair == null ? null : tokenize(textPair);

    if (tokensB == null) {
      final maxA = maxLength - 2;
      final keptA = tokensA.length > maxA ? tokensA.sublist(0, maxA) : tokensA;
      final inputIds = <int>[
        clsTokenId,
        for (final t in keptA) _vocab[t] ?? unkTokenId,
        sepTokenId,
      ];
      return (
        inputIds: inputIds,
        attentionMask: List<int>.filled(inputIds.length, 1),
        tokenTypeIds: List<int>.filled(inputIds.length, 0),
      );
    }

    // Truncate longest-first to fit `[CLS] A [SEP] B [SEP]`.
    final keptA = List<String>.of(tokensA);
    final keptB = List<String>.of(tokensB);
    final budget = maxLength - 3;
    while (keptA.length + keptB.length > budget) {
      if (keptA.length >= keptB.length) {
        keptA.removeLast();
      } else {
        keptB.removeLast();
      }
    }
    final inputIds = <int>[
      clsTokenId,
      for (final t in keptA) _vocab[t] ?? unkTokenId,
      sepTokenId,
      for (final t in keptB) _vocab[t] ?? unkTokenId,
      sepTokenId,
    ];
    final tokenTypeIds = <int>[
      for (var i = 0; i < keptA.length + 2; i++) 0,
      for (var i = 0; i < keptB.length + 1; i++) 1,
    ];
    return (
      inputIds: inputIds,
      attentionMask: List<int>.filled(inputIds.length, 1),
      tokenTypeIds: tokenTypeIds,
    );
  }

  /// Encodes and right-pads a non-empty list of [texts] to the longest sequence
  /// in the batch (capped at [maxLength]).
  TokenizedBatch encodeBatch(List<String> texts, {int maxLength = 256}) {
    if (texts.isEmpty) {
      throw ArgumentError.value(texts, 'texts', 'Must not be empty');
    }
    final encoded = [for (final t in texts) encode(t, maxLength: maxLength)];
    var paddedLen = 0;
    for (final e in encoded) {
      if (e.inputIds.length > paddedLen) paddedLen = e.inputIds.length;
    }

    final batchIds = <List<int>>[];
    final batchMask = <List<int>>[];
    final batchTypes = <List<int>>[];
    for (final e in encoded) {
      final padCount = paddedLen - e.inputIds.length;
      batchIds.add([
        ...e.inputIds,
        for (var i = 0; i < padCount; i++) padTokenId,
      ]);
      batchMask.add([...e.attentionMask, for (var i = 0; i < padCount; i++) 0]);
      batchTypes.add([...e.tokenTypeIds, for (var i = 0; i < padCount; i++) 0]);
    }
    return TokenizedBatch(
      inputIds: batchIds,
      attentionMask: batchMask,
      tokenTypeIds: batchTypes,
    );
  }

  List<String> _wordPiece(String token) {
    if (token.isEmpty) return const [];
    final runes = token.runes.toList();
    if (runes.length > maxInputCharsPerWord) {
      return [_idToToken[unkTokenId] ?? '[UNK]'];
    }

    final subTokens = <String>[];
    var start = 0;
    while (start < runes.length) {
      var end = runes.length;
      String? matched;
      while (start < end) {
        var substr = String.fromCharCodes(runes, start, end);
        if (start > 0) {
          substr = '$continuingSubwordPrefix$substr';
        }
        if (_vocab.containsKey(substr)) {
          matched = substr;
          break;
        }
        end--;
      }
      if (matched == null) {
        return [_idToToken[unkTokenId] ?? '[UNK]'];
      }
      subTokens.add(matched);
      start = end;
    }
    return subTokens;
  }

  static int _requireToken(Map<String, int> vocab, String token) {
    final id = vocab[token];
    if (id == null) {
      throw ArgumentError.value(
        token,
        'vocab',
        'Must contain required special token "$token"',
      );
    }
    return id;
  }
}

String _cleanText(String text) {
  final sb = StringBuffer();
  for (final cp in text.runes) {
    if (cp == 0 || cp == 0xFFFD || _isControl(cp)) {
      continue;
    }
    if (_isWhitespace(cp)) {
      sb.write(' ');
    } else {
      sb.writeCharCode(cp);
    }
  }
  return sb.toString();
}

String _tokenizeChineseChars(String text) {
  final sb = StringBuffer();
  for (final cp in text.runes) {
    if (_isChineseChar(cp)) {
      sb
        ..write(' ')
        ..writeCharCode(cp)
        ..write(' ');
    } else {
      sb.writeCharCode(cp);
    }
  }
  return sb.toString();
}

List<String> _splitWhitespace(String text) {
  final trimmed = text.trim();
  if (trimmed.isEmpty) return const [];
  return trimmed.split(RegExp(r'\s+'));
}

List<String> _splitOnPunctuation(String text) {
  final out = <String>[];
  final current = StringBuffer();
  for (final cp in text.runes) {
    if (_isPunctuation(cp)) {
      if (current.isNotEmpty) {
        out.add(current.toString());
        current.clear();
      }
      out.add(String.fromCharCode(cp));
    } else {
      current.writeCharCode(cp);
    }
  }
  if (current.isNotEmpty) {
    out.add(current.toString());
  }
  return out;
}

bool _isWhitespace(int cp) =>
    cp == 0x20 ||
    cp == 0x09 ||
    cp == 0x0A ||
    cp == 0x0D ||
    cp == 0x00A0 ||
    cp == 0x1680 ||
    (cp >= 0x2000 && cp <= 0x200A) ||
    cp == 0x202F ||
    cp == 0x205F ||
    cp == 0x3000;

bool _isControl(int cp) {
  if (cp == 0x09 || cp == 0x0A || cp == 0x0D) return false;
  return (cp >= 0x00 && cp <= 0x1F) ||
      (cp >= 0x7F && cp <= 0x9F) ||
      (cp >= 0x200B && cp <= 0x200F) ||
      (cp >= 0x202A && cp <= 0x202E) ||
      cp == 0xFEFF;
}

bool _isPunctuation(int cp) {
  if ((cp >= 33 && cp <= 47) ||
      (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) ||
      (cp >= 123 && cp <= 126)) {
    return true;
  }
  // Latin-1 punctuation (`¡ § « ¶ · » ¿`).
  if (cp == 0x00A1 ||
      cp == 0x00A7 ||
      cp == 0x00AB ||
      cp == 0x00B6 ||
      cp == 0x00B7 ||
      cp == 0x00BB ||
      cp == 0x00BF) {
    return true;
  }
  // Common general/CJK punctuation blocks.
  return (cp >= 0x2000 && cp <= 0x206F) ||
      (cp >= 0x3000 && cp <= 0x303F) ||
      (cp >= 0xFF01 && cp <= 0xFF0F) ||
      (cp >= 0xFF1A && cp <= 0xFF20) ||
      (cp >= 0xFF3B && cp <= 0xFF40) ||
      (cp >= 0xFF5B && cp <= 0xFF65);
}

bool _isChineseChar(int cp) =>
    (cp >= 0x4E00 && cp <= 0x9FFF) ||
    (cp >= 0x3400 && cp <= 0x4DBF) ||
    (cp >= 0x20000 && cp <= 0x2A6DF) ||
    (cp >= 0x2A700 && cp <= 0x2B73F) ||
    (cp >= 0x2B740 && cp <= 0x2B81F) ||
    (cp >= 0x2B820 && cp <= 0x2CEAF) ||
    (cp >= 0xF900 && cp <= 0xFAFF) ||
    (cp >= 0x2F800 && cp <= 0x2FA1F);

/// Strips combining diacritical marks (`U+0300..U+036F`) and folds common
/// precomposed lowercase Latin-1 accented characters to ASCII base letters.
String _stripLatinAccents(String text) {
  final sb = StringBuffer();
  for (final cp in text.runes) {
    if (cp >= 0x0300 && cp <= 0x036F) continue;
    final folded = _latinFold[cp];
    if (folded != null) {
      sb.write(folded);
    } else {
      sb.writeCharCode(cp);
    }
  }
  return sb.toString();
}

const Map<int, String> _latinFold = <int, String>{
  0x00E0: 'a',
  0x00E1: 'a',
  0x00E2: 'a',
  0x00E3: 'a',
  0x00E4: 'a',
  0x00E5: 'a',
  0x00E7: 'c',
  0x00E8: 'e',
  0x00E9: 'e',
  0x00EA: 'e',
  0x00EB: 'e',
  0x00EC: 'i',
  0x00ED: 'i',
  0x00EE: 'i',
  0x00EF: 'i',
  0x00F1: 'n',
  0x00F2: 'o',
  0x00F3: 'o',
  0x00F4: 'o',
  0x00F5: 'o',
  0x00F6: 'o',
  0x00F9: 'u',
  0x00FA: 'u',
  0x00FB: 'u',
  0x00FC: 'u',
  0x00FD: 'y',
  0x00FF: 'y',
};
