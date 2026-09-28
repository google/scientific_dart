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

import 'color_theme.dart';
import 'line_state.dart';
import 'syntax_token.dart';
import 'syntax_tokenizer.dart';

/// Manages incremental re-tokenization with line-state caching
/// and early state convergence halting optimization.
class IncrementalTokenizer {
  final SyntaxTokenizer tokenizer;
  final StyleCache? styleCache;

  final List<LineState> _lineStates = [];
  final List<List<SyntaxToken>> _lineTokens = [];

  /// Number of lines tokenized in the most recent tokenization / re-tokenization operation.
  int lastTokenizedLineCount = 0;

  IncrementalTokenizer({required this.tokenizer, this.styleCache});

  List<LineState> get cachedLineStates => List.unmodifiable(_lineStates);
  List<List<SyntaxToken>> get cachedLineTokens =>
      List.unmodifiable(_lineTokens);

  /// Initializes or full re-tokenizes document lines.
  void setDocument(List<String> lines) {
    _lineStates.clear();
    _lineTokens.clear();
    lastTokenizedLineCount = 0;

    LineState state = const EmptyLineState();
    for (int i = 0; i < lines.length; i++) {
      final res = tokenizer.tokenizeLine(lines[i], state);
      final styledTokens = _applyStyle(res.tokens);
      _lineTokens.add(styledTokens);
      _lineStates.add(res.endState);
      state = res.endState;
      lastTokenizedLineCount++;
    }
  }

  /// Retokenizes starting from [startLineIndex] when lines in [documentLines] change.
  /// Employs early state convergence optimization: halts as soon as [endState] matches
  /// the cached state for line [i] and line structure is unchanged.
  void updateDocument(List<String> documentLines, {int startLineIndex = 0}) {
    if (documentLines.isEmpty) {
      _lineStates.clear();
      _lineTokens.clear();
      lastTokenizedLineCount = 0;
      return;
    }

    if (startLineIndex < 0) startLineIndex = 0;
    if (startLineIndex >= documentLines.length) return;

    lastTokenizedLineCount = 0;

    while (_lineStates.length < documentLines.length) {
      _lineStates.add(const EmptyLineState());
      _lineTokens.add([]);
    }
    if (_lineStates.length > documentLines.length) {
      _lineStates.removeRange(documentLines.length, _lineStates.length);
      _lineTokens.removeRange(documentLines.length, _lineTokens.length);
    }

    LineState state = startLineIndex > 0
        ? _lineStates[startLineIndex - 1]
        : const EmptyLineState();

    for (int i = startLineIndex; i < documentLines.length; i++) {
      final previousCachedState = _lineStates[i];
      final res = tokenizer.tokenizeLine(documentLines[i], state);
      final styledTokens = _applyStyle(res.tokens);

      _lineTokens[i] = styledTokens;
      _lineStates[i] = res.endState;
      state = res.endState;
      lastTokenizedLineCount++;

      // Early State Convergence Halting Optimization
      if (i > startLineIndex && res.endState == previousCachedState) {
        break;
      }
    }
  }

  List<SyntaxToken> getTokensForLine(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= _lineTokens.length) return const [];
    return _lineTokens[lineIndex];
  }

  LineState getLineState(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= _lineStates.length) {
      return const EmptyLineState();
    }
    return _lineStates[lineIndex];
  }

  List<SyntaxToken> _applyStyle(List<SyntaxToken> tokens) {
    if (styleCache == null) return tokens;
    return tokens.map((t) {
      final style = styleCache!.getStyle(t.scopes);
      return t.withStyle(style);
    }).toList();
  }
}
