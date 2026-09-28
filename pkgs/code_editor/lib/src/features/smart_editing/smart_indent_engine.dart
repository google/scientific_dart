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

import '../../core/buffer/text_buffer.dart';
import '../../core/selection/text_position.dart';

/// Result of a smart Enter key calculation.
final class SmartEnterResult {
  /// The text string to insert into the buffer (may include newlines and indentation).
  final String textToInsert;

  /// The cursor position relative to the insertion start offset.
  final int relativeCaretOffset;

  /// Creates a [SmartEnterResult].
  const SmartEnterResult({
    required this.textToInsert,
    required this.relativeCaretOffset,
  });
}

/// Smart indentation engine for auto-indenting on Enter and tab/outdent manipulation.
final class SmartIndentEngine {
  /// The indentation string (e.g. `'  '` for 2 spaces, `'    '` for 4 spaces).
  final String indentString;

  /// Number of spaces per indentation level.
  final int tabSize;

  /// Creates a [SmartIndentEngine] with configurable [tabSize] (defaults to 2 spaces).
  const SmartIndentEngine({this.tabSize = 2, this.indentString = '  '});

  /// Computes the leading whitespace string of [line].
  static String getLeadingWhitespace(String line) {
    int i = 0;
    while (i < line.length && (line[i] == ' ' || line[i] == '\t')) {
      i++;
    }
    return line.substring(0, i);
  }

  /// Calculates the text to insert and new cursor position when pressing Enter at [position].
  SmartEnterResult calculateEnter({
    required TextBuffer buffer,
    required TextPosition position,
  }) {
    if (buffer.length == 0 ||
        position.line < 0 ||
        position.line >= buffer.lineCount) {
      return SmartEnterResult(textToInsert: '\n', relativeCaretOffset: 1);
    }

    final lineOffset = buffer.getLineOffset(position.line);
    final lineLen = buffer.getLineLength(position.line);
    final lineText = buffer.getTextInRange(lineOffset, lineLen);
    final col = position.column.clamp(0, lineText.length);

    final beforeCursor = lineText.substring(0, col);
    final afterCursor = lineText.substring(col);

    final currentIndent = getLeadingWhitespace(beforeCursor);
    final trimmedBefore = beforeCursor.trimRight();
    final trimmedAfter = afterCursor.trimLeft();

    // Check if pressing Enter between `{}` or `()` or `[]`
    final isBetweenBrackets =
        (trimmedBefore.endsWith('{') && trimmedAfter.startsWith('}')) ||
        (trimmedBefore.endsWith('(') && trimmedAfter.startsWith(')')) ||
        (trimmedBefore.endsWith('[') && trimmedAfter.startsWith(']'));

    if (isBetweenBrackets) {
      final extraIndent = currentIndent + indentString;
      // Format: \n<extraIndent>\n<currentIndent>
      final text = '\n$extraIndent\n$currentIndent';
      final caretOffset = 1 + extraIndent.length;
      return SmartEnterResult(
        textToInsert: text,
        relativeCaretOffset: caretOffset,
      );
    }

    // Check if previous character opens a new block: `{`, `(`, `[`, `:`
    final shouldIncreaseIndent =
        trimmedBefore.endsWith('{') ||
        trimmedBefore.endsWith('(') ||
        trimmedBefore.endsWith('[') ||
        trimmedBefore.endsWith(':') ||
        trimmedBefore.endsWith('=>');

    final nextIndent = shouldIncreaseIndent
        ? currentIndent + indentString
        : currentIndent;

    final text = '\n$nextIndent';
    return SmartEnterResult(
      textToInsert: text,
      relativeCaretOffset: text.length,
    );
  }

  /// Calculates indented text for a range of selected lines when pressing Tab.
  List<String> indentLines(List<String> lines) {
    return lines
        .map((line) => line.isEmpty ? line : '$indentString$line')
        .toList();
  }

  /// Calculates outdented text for a range of selected lines when pressing Shift+Tab.
  List<String> outdentLines(List<String> lines) {
    return lines.map((line) {
      if (line.startsWith(indentString)) {
        return line.substring(indentString.length);
      } else if (line.startsWith('\t')) {
        return line.substring(1);
      } else if (line.startsWith(' ')) {
        int spaces = 0;
        while (spaces < line.length &&
            spaces < tabSize &&
            line[spaces] == ' ') {
          spaces++;
        }
        return line.substring(spaces);
      }
      return line;
    }).toList();
  }
}
