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

import 'package:meta/meta.dart';
import '../../core/buffer/text_buffer.dart';
import '../../core/selection/text_position.dart';

/// Represents a pair of matching opening and closing bracket characters.
@immutable
final class BracketPair {
  /// The opening bracket character (e.g. `'('`, `'{'`, `'['`, `'<'`).
  final String open;

  /// The closing bracket character (e.g. `')'`, `'}'`, `']'`, `'>'`).
  final String close;

  /// Creates a bracket pair definition.
  const BracketPair(this.open, this.close);

  /// Standard bracket pairs for code editing: `()`, `{}`, `[]`.
  static const List<BracketPair> standard = [
    BracketPair('(', ')'),
    BracketPair('{', '}'),
    BracketPair('[', ']'),
  ];

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is BracketPair &&
          runtimeType == other.runtimeType &&
          open == other.open &&
          close == other.close;

  @override
  int get hashCode => Object.hash(open, close);

  @override
  String toString() => 'BracketPair($open, $close)';
}

/// Result of a matching bracket search.
@immutable
final class BracketMatchResult {
  /// The position of the bracket at or adjacent to the caret.
  final TextPosition source;

  /// The position of the corresponding matching bracket in the document.
  final TextPosition match;

  /// Whether [source] is the opening bracket (`true`) or closing bracket (`false`).
  final bool isOpening;

  /// The bracket pair definition.
  final BracketPair pair;

  /// Creates a bracket match result.
  const BracketMatchResult({
    required this.source,
    required this.match,
    required this.isOpening,
    required this.pair,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is BracketMatchResult &&
          runtimeType == other.runtimeType &&
          source == other.source &&
          match == other.match &&
          isOpening == other.isOpening &&
          pair == other.pair;

  @override
  int get hashCode => Object.hash(source, match, isOpening, pair);
}

/// Engine for finding matching opening and closing bracket pairs in a text buffer.
final class BracketMatcher {
  /// Registered bracket pairs.
  final List<BracketPair> pairs;

  /// Creates a [BracketMatcher] with configured [pairs] (defaults to [BracketPair.standard]).
  const BracketMatcher({this.pairs = BracketPair.standard});

  /// Finds the matching bracket for a character at or adjacent to [position].
  ///
  /// Scans forward for opening brackets and backward for closing brackets,
  /// properly accounting for nested bracket depth.
  ///
  /// Returns `null` if no bracket is found at [position] or if the bracket is unclosed.
  BracketMatchResult? findMatchingBracket(
    TextBuffer buffer,
    TextPosition position,
  ) {
    if (buffer.length == 0) return null;
    final totalLines = buffer.lineCount;
    if (position.line < 0 || position.line >= totalLines) return null;

    final lineStart = buffer.getLineOffset(position.line);
    final lineLength = buffer.getLineLength(position.line);
    final lineText = buffer.getTextInRange(lineStart, lineLength);
    final col = position.column;

    // Check character right under cursor or directly to the left of cursor
    for (final checkCol in [col, col - 1]) {
      if (checkCol < 0 || checkCol >= lineText.length) continue;
      final char = lineText[checkCol];

      for (final pair in pairs) {
        if (char == pair.open) {
          final matchPos = _findForwardMatch(
            buffer,
            position.line,
            checkCol,
            pair,
          );
          if (matchPos != null) {
            return BracketMatchResult(
              source: TextPosition(position.line, checkCol),
              match: matchPos,
              isOpening: true,
              pair: pair,
            );
          }
        } else if (char == pair.close) {
          final matchPos = _findBackwardMatch(
            buffer,
            position.line,
            checkCol,
            pair,
          );
          if (matchPos != null) {
            return BracketMatchResult(
              source: TextPosition(position.line, checkCol),
              match: matchPos,
              isOpening: false,
              pair: pair,
            );
          }
        }
      }
    }

    return null;
  }

  TextPosition? _findForwardMatch(
    TextBuffer buffer,
    int startLine,
    int startCol,
    BracketPair pair,
  ) {
    int depth = 0;
    final totalLines = buffer.lineCount;

    for (int l = startLine; l < totalLines; l++) {
      final lineOffset = buffer.getLineOffset(l);
      final lineLen = buffer.getLineLength(l);
      final text = buffer.getTextInRange(lineOffset, lineLen);
      final startC = (l == startLine) ? startCol : 0;

      for (int c = startC; c < text.length; c++) {
        final ch = text[c];
        if (ch == pair.open) {
          depth++;
        } else if (ch == pair.close) {
          depth--;
          if (depth == 0) {
            return TextPosition(l, c);
          }
        }
      }
    }
    return null;
  }

  TextPosition? _findBackwardMatch(
    TextBuffer buffer,
    int startLine,
    int startCol,
    BracketPair pair,
  ) {
    int depth = 0;

    for (int l = startLine; l >= 0; l--) {
      final lineOffset = buffer.getLineOffset(l);
      final lineLen = buffer.getLineLength(l);
      final text = buffer.getTextInRange(lineOffset, lineLen);
      final startC = (l == startLine) ? startCol : text.length - 1;

      for (int c = startC; c >= 0; c--) {
        final ch = text[c];
        if (ch == pair.close) {
          depth++;
        } else if (ch == pair.open) {
          depth--;
          if (depth == 0) {
            return TextPosition(l, c);
          }
        }
      }
    }
    return null;
  }
}
