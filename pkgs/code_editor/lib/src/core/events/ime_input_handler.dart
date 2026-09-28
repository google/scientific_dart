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

import '../selection/selection_model.dart';

/// Range of character indices.
final class TextRange {
  /// Start index (inclusive).
  final int start;

  /// End index (exclusive).
  final int end;

  /// Creates a [TextRange] from [start] to [end].
  const TextRange({required this.start, required this.end})
    : assert(start <= end, 'start must be <= end');

  /// Invalid empty range.
  static const empty = TextRange(start: -1, end: -1);

  /// Whether range is valid.
  bool get isValid => start >= 0 && end >= start;

  /// Whether range is collapsed.
  bool get isCollapsed => start == end;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TextRange && other.start == start && other.end == end;
  }

  @override
  int get hashCode => Object.hash(start, end);

  @override
  String toString() => 'TextRange($start, $end)';
}

/// Representation of platform IME editing state.
final class TextEditingValue {
  /// The full flat text content.
  final String text;

  /// The active selection range within [text].
  final TextRange selection;

  /// The active composing range within [text], if any.
  final TextRange? composing;

  /// Creates a [TextEditingValue].
  const TextEditingValue({
    required this.text,
    required this.selection,
    this.composing,
  });

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TextEditingValue &&
        other.text == text &&
        other.selection == selection &&
        other.composing == composing;
  }

  @override
  int get hashCode => Object.hash(text, selection, composing);

  @override
  String toString() =>
      'TextEditingValue(text: "${text.length > 20 ? '${text.substring(0, 20)}...' : text}", selection: $selection, composing: $composing)';
}

/// Result of an IME edit operation.
final class ImeResult {
  /// The modified document lines.
  final List<String> lines;

  /// The resulting selection.
  final TextSelection selection;

  /// Active composition range, if any.
  final TextRange? composing;

  /// Creates an [ImeResult].
  const ImeResult({
    required this.lines,
    required this.selection,
    this.composing,
  });
}

/// Handler for IME text composition, commit text deltas, and platform synchronization.
final class ImeInputHandler {
  /// Handles commit text insertion into document lines at [selection].
  static ImeResult handleCommitText({
    required List<String> lines,
    required TextSelection selection,
    required String insertedText,
  }) {
    final mutableLines = List<String>.from(lines.isEmpty ? [''] : lines);
    final start = selection.start;
    final end = selection.end;

    final startLine = start.line.clamp(0, mutableLines.length - 1);
    final endLine = end.line.clamp(0, mutableLines.length - 1);

    final prefix = mutableLines[startLine].substring(
      0,
      start.column.clamp(0, mutableLines[startLine].length),
    );
    final suffix = mutableLines[endLine].substring(
      end.column.clamp(0, mutableLines[endLine].length),
    );

    final normalizedText = insertedText
        .replaceAll('\r\n', '\n')
        .replaceAll('\r', '\n');
    final insertedLines = normalizedText.split('\n');

    if (insertedLines.length == 1) {
      final newLine = prefix + insertedLines.first + suffix;
      mutableLines.replaceRange(startLine, endLine + 1, [newLine]);
      final newCol = prefix.length + insertedLines.first.length;
      final newPos = TextPosition(startLine, newCol);
      return ImeResult(
        lines: mutableLines,
        selection: TextSelection.collapsed(newPos),
      );
    } else {
      final replacement = <String>[];
      replacement.add(prefix + insertedLines.first);
      for (int i = 1; i < insertedLines.length - 1; i++) {
        replacement.add(insertedLines[i]);
      }
      replacement.add(insertedLines.last + suffix);

      mutableLines.replaceRange(startLine, endLine + 1, replacement);
      final newPosLine = startLine + insertedLines.length - 1;
      final newPosCol = insertedLines.last.length;
      return ImeResult(
        lines: mutableLines,
        selection: TextSelection.collapsed(TextPosition(newPosLine, newPosCol)),
      );
    }
  }

  /// Handles backward delete (Backspace).
  static ImeResult handleDeleteBackward({
    required List<String> lines,
    required TextSelection selection,
  }) {
    if (lines.isEmpty) {
      return ImeResult(
        lines: const [''],
        selection: TextSelection.collapsed(const TextPosition(0, 0)),
      );
    }

    if (!selection.isCollapsed) {
      return handleCommitText(
        lines: lines,
        selection: selection,
        insertedText: '',
      );
    }

    final pos = selection.start;
    final mutableLines = List<String>.from(lines);

    if (pos.column > 0) {
      final lineText = mutableLines[pos.line];
      final newLineText =
          lineText.substring(0, pos.column - 1) +
          lineText.substring(pos.column);
      mutableLines[pos.line] = newLineText;
      return ImeResult(
        lines: mutableLines,
        selection: TextSelection.collapsed(
          TextPosition(pos.line, pos.column - 1),
        ),
      );
    } else if (pos.line > 0) {
      final prevLineIndex = pos.line - 1;
      final prevLineText = mutableLines[prevLineIndex];
      final currentLineText = mutableLines[pos.line];
      final newCol = prevLineText.length;

      mutableLines[prevLineIndex] = prevLineText + currentLineText;
      mutableLines.removeAt(pos.line);

      return ImeResult(
        lines: mutableLines,
        selection: TextSelection.collapsed(TextPosition(prevLineIndex, newCol)),
      );
    }

    return ImeResult(lines: lines, selection: selection);
  }

  /// Handles forward delete (Delete key).
  static ImeResult handleDeleteForward({
    required List<String> lines,
    required TextSelection selection,
  }) {
    if (lines.isEmpty) {
      return ImeResult(
        lines: const [''],
        selection: TextSelection.collapsed(const TextPosition(0, 0)),
      );
    }

    if (!selection.isCollapsed) {
      return handleCommitText(
        lines: lines,
        selection: selection,
        insertedText: '',
      );
    }

    final pos = selection.start;
    final mutableLines = List<String>.from(lines);
    final lineText = mutableLines[pos.line];

    if (pos.column < lineText.length) {
      final newLineText =
          lineText.substring(0, pos.column) +
          lineText.substring(pos.column + 1);
      mutableLines[pos.line] = newLineText;
      return ImeResult(lines: mutableLines, selection: selection);
    } else if (pos.line < mutableLines.length - 1) {
      final nextLineText = mutableLines[pos.line + 1];
      mutableLines[pos.line] = lineText + nextLineText;
      mutableLines.removeAt(pos.line + 1);

      return ImeResult(lines: mutableLines, selection: selection);
    }

    return ImeResult(lines: lines, selection: selection);
  }

  /// Synchronizes document lines and selection to platform [TextEditingValue].
  static TextEditingValue syncToTextEditingValue({
    required List<String> lines,
    required TextSelection selection,
    TextRange? composing,
  }) {
    final fullText = lines.join('\n');
    final baseOffset = positionToOffset(lines, selection.base);
    final extentOffset = positionToOffset(lines, selection.extent);

    return TextEditingValue(
      text: fullText,
      selection: TextRange(start: baseOffset, end: extentOffset),
      composing: composing,
    );
  }

  /// Converts a document line & column [TextPosition] to flat string character offset.
  static int positionToOffset(List<String> lines, TextPosition position) {
    if (lines.isEmpty) return 0;
    int offset = 0;
    final targetLine = position.line.clamp(0, lines.length - 1);
    for (int i = 0; i < targetLine; i++) {
      offset += lines[i].length + 1; // +1 for '\n'
    }
    if (position.line >= lines.length) {
      offset += lines[targetLine].length;
    } else {
      offset += position.column.clamp(0, lines[targetLine].length);
    }
    return offset;
  }

  /// Converts a flat string character offset to document [TextPosition].
  static TextPosition offsetToPosition(List<String> lines, int offset) {
    if (lines.isEmpty || offset <= 0) return const TextPosition(0, 0);

    int remaining = offset;
    for (int i = 0; i < lines.length; i++) {
      final len = lines[i].length;
      if (remaining <= len) {
        return TextPosition(i, remaining);
      }
      remaining -= (len + 1); // +1 for '\n'
      if (remaining < 0) {
        return TextPosition(i, len);
      }
    }
    final lastLine = lines.length - 1;
    return TextPosition(lastLine, lines[lastLine].length);
  }
}
