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

/// Slice/subrange of a single text line created by line wrapping.
final class TextLineSlice {
  /// The index of the document line this slice belongs to.
  final int lineIndex;

  /// The 0-based slice index within the wrapped line.
  final int sliceIndex;

  /// The starting character column (inclusive).
  final int startColumn;

  /// The ending character column (exclusive).
  final int endColumn;

  /// The text content of this slice.
  final String content;

  /// Creates a [TextLineSlice].
  const TextLineSlice({
    required this.lineIndex,
    required this.sliceIndex,
    required this.startColumn,
    required this.endColumn,
    required this.content,
  });

  /// Length of text in this slice.
  int get length => content.length;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TextLineSlice &&
        other.lineIndex == lineIndex &&
        other.sliceIndex == sliceIndex &&
        other.startColumn == startColumn &&
        other.endColumn == endColumn &&
        other.content == content;
  }

  @override
  int get hashCode =>
      Object.hash(lineIndex, sliceIndex, startColumn, endColumn, content);

  @override
  String toString() =>
      'TextLineSlice(line: $lineIndex, slice: $sliceIndex, range: $startColumn..$endColumn, text: "$content")';
}

/// Engine that performs soft line wrapping based on word boundaries and max width constraints.
final class LineWrappingEngine {
  /// Maximum number of character columns allowed per slice line.
  final int maxColumns;

  /// Whether soft wrapping is enabled.
  final bool enabled;

  /// Creates a [LineWrappingEngine].
  const LineWrappingEngine({this.maxColumns = 80, this.enabled = true});

  /// Wraps [lineText] into one or more [TextLineSlice] instances.
  List<TextLineSlice> wrapLine(String lineText, int lineIndex) {
    if (!enabled || maxColumns <= 0 || lineText.length <= maxColumns) {
      return [
        TextLineSlice(
          lineIndex: lineIndex,
          sliceIndex: 0,
          startColumn: 0,
          endColumn: lineText.length,
          content: lineText,
        ),
      ];
    }

    final slices = <TextLineSlice>[];
    int startCol = 0;
    int sliceIndex = 0;

    while (startCol < lineText.length) {
      final remaining = lineText.length - startCol;
      if (remaining <= maxColumns) {
        slices.add(
          TextLineSlice(
            lineIndex: lineIndex,
            sliceIndex: sliceIndex++,
            startColumn: startCol,
            endColumn: lineText.length,
            content: lineText.substring(startCol),
          ),
        );
        break;
      }

      final int candidateEnd = startCol + maxColumns;
      int breakCol = _findWordBoundary(lineText, startCol, candidateEnd);

      if (breakCol <= startCol) {
        // No word boundary found, hard break at maxColumns
        breakCol = candidateEnd;
      }

      slices.add(
        TextLineSlice(
          lineIndex: lineIndex,
          sliceIndex: sliceIndex++,
          startColumn: startCol,
          endColumn: breakCol,
          content: lineText.substring(startCol, breakCol),
        ),
      );

      startCol = breakCol;
    }

    return slices.isEmpty
        ? [
            TextLineSlice(
              lineIndex: lineIndex,
              sliceIndex: 0,
              startColumn: 0,
              endColumn: 0,
              content: '',
            ),
          ]
        : slices;
  }

  static int _findWordBoundary(String text, int start, int targetEnd) {
    for (int i = targetEnd; i > start; i--) {
      if (i < text.length && (text[i] == ' ' || text[i] == '\t')) {
        return i + 1 <= targetEnd ? i + 1 : targetEnd;
      }
      if (i - 1 >= start && (text[i - 1] == ' ' || text[i - 1] == '\t')) {
        return i <= targetEnd ? i : targetEnd;
      }
    }
    return start;
  }
}
