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
import '../../core/selection/selection.dart';
import '../../core/selection/text_position.dart';

/// Specification for an auto-closing quote or bracket pair.
@immutable
final class AutoClosePair {
  /// The opening string (e.g. `'('`, `'{'`, `'['`, `'"'`, `'\''`).
  final String open;

  /// The closing string (e.g. `')'`, `'}'`, `']'`, `'"'`, `'\''`).
  final String close;

  /// Creates an auto-close pair definition.
  const AutoClosePair(this.open, this.close);

  /// Standard auto-close pairs for code editing.
  static const List<AutoClosePair> standard = [
    AutoClosePair('(', ')'),
    AutoClosePair('{', '}'),
    AutoClosePair('[', ']'),
    AutoClosePair('"', '"'),
    AutoClosePair("'", "'"),
    AutoClosePair('`', '`'),
  ];

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is AutoClosePair &&
          runtimeType == other.runtimeType &&
          open == other.open &&
          close == other.close;

  @override
  int get hashCode => Object.hash(open, close);
}

/// Action to perform when typing a character that matches an auto-closing pair.
sealed class AutoCloseAction {
  const AutoCloseAction();
}

/// Insert both open and close characters, leaving the cursor between them.
final class InsertPairAction extends AutoCloseAction {
  final String open;
  final String close;
  const InsertPairAction(this.open, this.close);
}

/// Skip over the existing closing character without inserting a duplicate.
final class SkipCloseAction extends AutoCloseAction {
  const SkipCloseAction();
}

/// Wrap the currently selected text with the open and close characters.
final class WrapSelectionAction extends AutoCloseAction {
  final String open;
  final String close;
  const WrapSelectionAction(this.open, this.close);
}

/// Engine managing auto-closing bracket and quote behaviors.
final class AutoCloseEngine {
  /// Configured auto-closing pairs.
  final List<AutoClosePair> pairs;

  /// Creates an [AutoCloseEngine] with [pairs] (defaults to [AutoClosePair.standard]).
  const AutoCloseEngine({this.pairs = AutoClosePair.standard});

  /// Evaluates the appropriate auto-close action when [typedChar] is input.
  ///
  /// Returns `null` if [typedChar] is not part of any registered auto-close pair.
  AutoCloseAction? handleType({
    required String typedChar,
    required TextBuffer buffer,
    required TextSelection selection,
  }) {
    if (typedChar.isEmpty) return null;

    // 1. If text is selected and typedChar is an open character, wrap selection
    if (!selection.isCollapsed) {
      for (final pair in pairs) {
        if (typedChar == pair.open) {
          return WrapSelectionAction(pair.open, pair.close);
        }
      }
    }

    // 2. Check if typing over an existing closing character
    if (selection.isCollapsed && buffer.length > 0) {
      final pos = selection.extent;
      if (pos.line >= 0 && pos.line < buffer.lineCount) {
        final lineOffset = buffer.getLineOffset(pos.line);
        final lineLen = buffer.getLineLength(pos.line);
        final lineText = buffer.getTextInRange(lineOffset, lineLen);

        if (pos.column < lineText.length) {
          final nextChar = lineText[pos.column];
          for (final pair in pairs) {
            if (typedChar == pair.close && nextChar == pair.close) {
              return const SkipCloseAction();
            }
          }
        }
      }
    }

    // 3. Check if inserting a new pair
    for (final pair in pairs) {
      if (typedChar == pair.open) {
        return InsertPairAction(pair.open, pair.close);
      }
    }

    return null;
  }

  /// Checks if pressing Backspace at [position] should delete both opening and closing pair characters.
  ///
  /// Returns the length to delete (2 if deleting both, 1 for regular backspace, 0 if at buffer start).
  int checkBackspacePairDeletion({
    required TextBuffer buffer,
    required TextPosition position,
  }) {
    if (buffer.length == 0 || position.column <= 0) return 0;
    if (position.line < 0 || position.line >= buffer.lineCount) return 1;

    final lineOffset = buffer.getLineOffset(position.line);
    final lineLen = buffer.getLineLength(position.line);
    final lineText = buffer.getTextInRange(lineOffset, lineLen);

    if (position.column - 1 >= 0 && position.column < lineText.length) {
      final prevChar = lineText[position.column - 1];
      final nextChar = lineText[position.column];

      for (final pair in pairs) {
        if (prevChar == pair.open && nextChar == pair.close) {
          return 2; // Delete both prevChar and nextChar
        }
      }
    }
    return 1;
  }
}
