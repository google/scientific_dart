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
import '../../core/selection/selection.dart';
import '../../core/selection/text_position.dart';

/// Result of a line manipulation operation.
final class LineOperationResult {
  /// The replacement start offset in the buffer.
  final int replaceStartOffset;

  /// The replacement length in the buffer.
  final int replaceLength;

  /// The new text to replace the range with.
  final String newText;

  /// The updated selection after the operation.
  final TextSelection newSelection;

  /// Creates a [LineOperationResult].
  const LineOperationResult({
    required this.replaceStartOffset,
    required this.replaceLength,
    required this.newText,
    required this.newSelection,
  });
}

/// Helper performing common IDE line operations (duplicate, move, delete, toggle comment).
final class LineOperations {
  /// Line comment prefix (default `'// '`).
  final String commentPrefix;

  /// Creates a [LineOperations] helper.
  const LineOperations({this.commentPrefix = '// '});

  /// Duplicates the line(s) spanned by [selection] directly below.
  LineOperationResult duplicateLinesDown({
    required TextBuffer buffer,
    required TextSelection selection,
  }) {
    final startLine = selection.start.line.clamp(0, buffer.lineCount - 1);
    final endLine = selection.end.line.clamp(0, buffer.lineCount - 1);

    final startOffset = buffer.getLineOffset(startLine);
    final endOffset = (endLine + 1 < buffer.lineCount)
        ? buffer.getLineOffset(endLine + 1)
        : buffer.length;

    final blockText = buffer.getTextInRange(
      startOffset,
      endOffset - startOffset,
    );
    final textToInsert = blockText.endsWith('\n') ? blockText : '\n$blockText';

    final insertOffset = endOffset;
    final lineCountDelta = endLine - startLine + 1;

    return LineOperationResult(
      replaceStartOffset: insertOffset,
      replaceLength: 0,
      newText: textToInsert,
      newSelection: TextSelection(
        base: TextPosition(
          selection.base.line + lineCountDelta,
          selection.base.column,
        ),
        extent: TextPosition(
          selection.extent.line + lineCountDelta,
          selection.extent.column,
        ),
      ),
    );
  }

  /// Moves the line(s) spanned by [selection] up by one line.
  LineOperationResult? moveLinesUp({
    required TextBuffer buffer,
    required TextSelection selection,
  }) {
    final startLine = selection.start.line.clamp(0, buffer.lineCount - 1);
    final endLine = selection.end.line.clamp(0, buffer.lineCount - 1);
    if (startLine == 0) return null; // Already at the top

    final targetLine = startLine - 1;
    final targetStartOffset = buffer.getLineOffset(targetLine);
    final blockEndOffset = (endLine + 1 < buffer.lineCount)
        ? buffer.getLineOffset(endLine + 1)
        : buffer.length;

    final prevLineLen = buffer.getLineOffset(startLine) - targetStartOffset;
    final prevLineText = buffer.getTextInRange(targetStartOffset, prevLineLen);
    final blockText = buffer.getTextInRange(
      buffer.getLineOffset(startLine),
      blockEndOffset - buffer.getLineOffset(startLine),
    );

    // Swap: blockText followed by prevLineText
    final newText = '$blockText$prevLineText';

    return LineOperationResult(
      replaceStartOffset: targetStartOffset,
      replaceLength: blockEndOffset - targetStartOffset,
      newText: newText,
      newSelection: TextSelection(
        base: TextPosition(selection.base.line - 1, selection.base.column),
        extent: TextPosition(
          selection.extent.line - 1,
          selection.extent.column,
        ),
      ),
    );
  }

  /// Moves the line(s) spanned by [selection] down by one line.
  LineOperationResult? moveLinesDown({
    required TextBuffer buffer,
    required TextSelection selection,
  }) {
    final startLine = selection.start.line.clamp(0, buffer.lineCount - 1);
    final endLine = selection.end.line.clamp(0, buffer.lineCount - 1);
    if (endLine >= buffer.lineCount - 1) return null; // Already at the bottom

    final nextLine = endLine + 1;
    final blockStartOffset = buffer.getLineOffset(startLine);
    final nextLineEndOffset = (nextLine + 1 < buffer.lineCount)
        ? buffer.getLineOffset(nextLine + 1)
        : buffer.length;

    final blockLen = buffer.getLineOffset(nextLine) - blockStartOffset;
    final blockText = buffer.getTextInRange(blockStartOffset, blockLen);
    final nextLineText = buffer.getTextInRange(
      buffer.getLineOffset(nextLine),
      nextLineEndOffset - buffer.getLineOffset(nextLine),
    );

    // Swap: nextLineText followed by blockText
    final newText = '$nextLineText$blockText';

    return LineOperationResult(
      replaceStartOffset: blockStartOffset,
      replaceLength: nextLineEndOffset - blockStartOffset,
      newText: newText,
      newSelection: TextSelection(
        base: TextPosition(selection.base.line + 1, selection.base.column),
        extent: TextPosition(
          selection.extent.line + 1,
          selection.extent.column,
        ),
      ),
    );
  }

  /// Deletes the full line(s) spanned by [selection].
  LineOperationResult deleteLines({
    required TextBuffer buffer,
    required TextSelection selection,
  }) {
    final startLine = selection.start.line.clamp(0, buffer.lineCount - 1);
    final endLine = selection.end.line.clamp(0, buffer.lineCount - 1);

    final startOffset = buffer.getLineOffset(startLine);
    final endOffset = (endLine + 1 < buffer.lineCount)
        ? buffer.getLineOffset(endLine + 1)
        : buffer.length;

    final nextLine = startLine.clamp(
      0,
      (buffer.lineCount - (endLine - startLine + 1)).clamp(
        0,
        buffer.lineCount - 1,
      ),
    );

    return LineOperationResult(
      replaceStartOffset: startOffset,
      replaceLength: endOffset - startOffset,
      newText: '',
      newSelection: TextSelection.collapsed(TextPosition(nextLine, 0)),
    );
  }

  /// Toggles line comments (`//`) on all lines spanned by [selection].
  LineOperationResult toggleLineComment({
    required TextBuffer buffer,
    required TextSelection selection,
  }) {
    final startLine = selection.start.line.clamp(0, buffer.lineCount - 1);
    final endLine = selection.end.line.clamp(0, buffer.lineCount - 1);

    final lines = <String>[];
    final hasTrailingNewlines = <bool>[];
    for (int l = startLine; l <= endLine; l++) {
      final start = buffer.getLineOffset(l);
      final len = buffer.getLineLength(l);
      var raw = buffer.getTextInRange(start, len);
      if (raw.endsWith('\r\n')) {
        raw = raw.substring(0, raw.length - 2);
        hasTrailingNewlines.add(true);
      } else if (raw.endsWith('\n')) {
        raw = raw.substring(0, raw.length - 1);
        hasTrailingNewlines.add(true);
      } else {
        hasTrailingNewlines.add(false);
      }
      lines.add(raw);
    }

    final prefix = commentPrefix.trimRight();
    final allCommented = lines.every((line) {
      final trimmed = line.trimLeft();
      return trimmed.isEmpty || trimmed.startsWith(prefix);
    });

    final modifiedLines = lines.map((line) {
      if (allCommented) {
        // Uncomment
        final leadingSpaces = line.length - line.trimLeft().length;
        final indent = line.substring(0, leadingSpaces);
        final rest = line.substring(leadingSpaces);
        if (rest.startsWith(commentPrefix)) {
          return '$indent${rest.substring(commentPrefix.length)}';
        } else if (rest.startsWith(prefix)) {
          return '$indent${rest.substring(prefix.length)}';
        }
        return line;
      } else {
        // Comment
        final leadingSpaces = line.length - line.trimLeft().length;
        final indent = line.substring(0, leadingSpaces);
        final rest = line.substring(leadingSpaces);
        return '$indent$commentPrefix$rest';
      }
    }).toList();

    final startOffset = buffer.getLineOffset(startLine);
    final endOffset = (endLine + 1 < buffer.lineCount)
        ? buffer.getLineOffset(endLine + 1)
        : buffer.length;

    final buf = StringBuffer();
    for (int i = 0; i < modifiedLines.length; i++) {
      buf.write(modifiedLines[i]);
      if (hasTrailingNewlines[i]) {
        buf.write('\n');
      }
    }
    final replacementText = buf.toString();

    return LineOperationResult(
      replaceStartOffset: startOffset,
      replaceLength: endOffset - startOffset,
      newText: replacementText,
      newSelection: selection,
    );
  }
}
