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
import 'selection.dart';
import 'text_position.dart';

export 'selection.dart';
export 'text_position.dart';

/// Model managing text selections and semantic selection boundary calculations.
final class SelectionModel {
  /// The primary selection.
  TextSelection primarySelection;
  List<TextSelection> _secondarySelections;

  /// Creates a [SelectionModel] with an initial [primarySelection].
  SelectionModel({
    this.primarySelection = const TextSelection(
      base: TextPosition(0, 0),
      extent: TextPosition(0, 0),
    ),
    List<TextSelection>? secondarySelections,
  }) : _secondarySelections = secondarySelections ?? const [];

  /// Gets all secondary selections (for multi-caret editing).
  List<TextSelection> get secondarySelections =>
      List.unmodifiable(_secondarySelections);

  /// Updates all secondary selections.
  set secondarySelections(List<TextSelection> selections) {
    _secondarySelections = List.from(selections);
  }

  /// Sets all primary and secondary selections.
  void setSelections(List<dynamic> selections) {
    if (selections.isEmpty) return;
    final converted = <TextSelection>[];
    for (final s in selections) {
      if (s is TextSelection) {
        converted.add(s);
      } else if (s != null) {
        try {
          final dynamic dyn = s;
          final TextPosition base =
              (dyn.base ?? dyn.anchor ?? const TextPosition(0, 0))
                  as TextPosition;
          final TextPosition extent =
              (dyn.extent ?? dyn.position ?? base) as TextPosition;
          converted.add(TextSelection(base: base, extent: extent));
        } catch (_) {}
      }
    }
    if (converted.isEmpty) return;
    primarySelection = converted.first;
    _secondarySelections = converted.length > 1
        ? List.from(converted.sublist(1))
        : const [];
  }

  /// Collapses selection to a single position.
  void collapseTo(TextPosition position) {
    primarySelection = TextSelection.collapsed(position);
    _secondarySelections = const [];
  }

  /// Gets all selections (primary selection followed by secondary selections).
  List<TextSelection> get selections => [
    primarySelection,
    ..._secondarySelections,
  ];

  /// Alias for primarySelection.
  TextSelection get primary => primarySelection;

  /// Sets single collapsed cursor at [position].
  void setSingleCursor(TextPosition position) => collapseTo(position);

  /// Shifts selection positions after inserting [length] characters at line/column.
  void transformOnInsert(int line, int column, int length) {
    TextPosition updatePos(TextPosition pos) {
      if (pos.line == line && pos.column >= column) {
        return TextPosition(pos.line, pos.column + length);
      }
      return pos;
    }

    primarySelection = TextSelection(
      base: updatePos(primarySelection.base),
      extent: updatePos(primarySelection.extent),
      affinity: primarySelection.affinity,
    );
    _secondarySelections = _secondarySelections
        .map(
          (s) => TextSelection(
            base: updatePos(s.base),
            extent: updatePos(s.extent),
            affinity: s.affinity,
          ),
        )
        .toList();
  }

  /// Shifts selection positions after deleting [length] characters at line/column.
  void transformOnDelete(int line, int column, int length) {
    TextPosition updatePos(TextPosition pos) {
      if (pos.line == line && pos.column >= column + length) {
        return TextPosition(pos.line, math.max(column, pos.column - length));
      } else if (pos.line == line && pos.column > column) {
        return TextPosition(pos.line, column);
      }
      return pos;
    }

    primarySelection = TextSelection(
      base: updatePos(primarySelection.base),
      extent: updatePos(primarySelection.extent),
      affinity: primarySelection.affinity,
    );
    _secondarySelections = _secondarySelections
        .map(
          (s) => TextSelection(
            base: updatePos(s.base),
            extent: updatePos(s.extent),
            affinity: s.affinity,
          ),
        )
        .toList();
  }

  /// Extends the current selection extent to [newExtent].
  void extendTo(TextPosition newExtent) {
    primarySelection = TextSelection(
      base: primarySelection.base,
      extent: newExtent,
    );
  }

  /// Selects the entire document given [lines].
  void selectAll(List<String> lines) {
    if (lines.isEmpty) {
      collapseTo(const TextPosition(0, 0));
      return;
    }
    final lastLine = lines.length - 1;
    final lastColumn = lines[lastLine].length;
    primarySelection = TextSelection(
      base: const TextPosition(0, 0),
      extent: TextPosition(lastLine, lastColumn),
    );
  }

  /// Selects the word at [position].
  void selectWordAt(List<String> lines, TextPosition position) {
    final wordRange = getWordBoundary(lines, position);
    primarySelection = wordRange;
  }

  /// Selects the entire line at [lineIndex].
  void selectLineAt(List<String> lines, int lineIndex) {
    if (lines.isEmpty) return;
    final safeLine = lineIndex.clamp(0, lines.length - 1);
    final lineText = lines[safeLine];
    primarySelection = TextSelection(
      base: TextPosition(safeLine, 0),
      extent: TextPosition(safeLine, lineText.length),
    );
  }

  /// Calculates the word boundary selection range surrounding [position].
  static TextSelection getWordBoundary(
    List<String> lines,
    TextPosition position,
  ) {
    if (lines.isEmpty) return TextSelection.collapsed(position);
    final line = position.line.clamp(0, lines.length - 1);
    final lineText = lines[line];
    final col = position.column.clamp(0, lineText.length);

    if (lineText.isEmpty) return TextSelection.collapsed(position);

    int startCol = col;
    int endCol = col;

    if (col < lineText.length && _isWordCharacter(lineText[col])) {
      while (startCol > 0 && _isWordCharacter(lineText[startCol - 1])) {
        startCol--;
      }
      while (endCol < lineText.length && _isWordCharacter(lineText[endCol])) {
        endCol++;
      }
    } else if (col > 0 && _isWordCharacter(lineText[col - 1])) {
      startCol = col - 1;
      while (startCol > 0 && _isWordCharacter(lineText[startCol - 1])) {
        startCol--;
      }
      endCol = col;
      while (endCol < lineText.length && _isWordCharacter(lineText[endCol])) {
        endCol++;
      }
    } else {
      // Non-word symbol or whitespace
      while (startCol > 0 &&
          !_isWordCharacter(lineText[startCol - 1]) &&
          !_isWhitespace(lineText[startCol - 1])) {
        startCol--;
      }
      while (endCol < lineText.length &&
          !_isWordCharacter(lineText[endCol]) &&
          !_isWhitespace(lineText[endCol])) {
        endCol++;
      }
      if (startCol == endCol && col < lineText.length) {
        endCol = math.min(col + 1, lineText.length);
      }
    }

    return TextSelection(
      base: TextPosition(line, startCol),
      extent: TextPosition(line, endCol),
    );
  }

  /// Finds the starting column index when moving left by word from [column].
  static int getWordStartColumn(String lineText, int column) {
    if (lineText.isEmpty || column <= 0) return 0;
    int i = column - 1;

    // Skip whitespace backward
    while (i > 0 && _isWhitespace(lineText[i])) {
      i--;
    }

    if (i >= 0 && _isWordCharacter(lineText[i])) {
      while (i > 0 && _isWordCharacter(lineText[i - 1])) {
        i--;
      }
    } else if (i >= 0 && !_isWhitespace(lineText[i])) {
      while (i > 0 &&
          !_isWordCharacter(lineText[i - 1]) &&
          !_isWhitespace(lineText[i - 1])) {
        i--;
      }
    }
    return math.max(0, i);
  }

  /// Finds the ending column index when moving right by word from [column].
  static int getWordEndColumn(String lineText, int column) {
    if (lineText.isEmpty || column >= lineText.length) return lineText.length;
    int i = column;

    // Skip whitespace forward
    while (i < lineText.length && _isWhitespace(lineText[i])) {
      i++;
    }

    if (i < lineText.length && _isWordCharacter(lineText[i])) {
      while (i < lineText.length && _isWordCharacter(lineText[i])) {
        i++;
      }
    } else if (i < lineText.length) {
      while (i < lineText.length &&
          !_isWordCharacter(lineText[i]) &&
          !_isWhitespace(lineText[i])) {
        i++;
      }
    }
    return math.min(lineText.length, i);
  }

  /// Calculates smart line start position with indentation jumping.
  static TextPosition getSmartLineStart(
    String lineText,
    TextPosition position,
  ) {
    int firstNonWhitespace = 0;
    while (firstNonWhitespace < lineText.length &&
        _isWhitespace(lineText[firstNonWhitespace])) {
      firstNonWhitespace++;
    }

    if (position.column == firstNonWhitespace) {
      return TextPosition(position.line, 0);
    } else {
      return TextPosition(position.line, firstNonWhitespace);
    }
  }

  /// Gets the line end position.
  static TextPosition getLineEnd(String lineText, int lineIndex) {
    return TextPosition(lineIndex, lineText.length);
  }

  /// Finds the start position of the paragraph surrounding [position].
  static TextPosition getParagraphStart(
    List<String> lines,
    TextPosition position,
  ) {
    if (lines.isEmpty) return const TextPosition(0, 0);
    int currentLine = position.line.clamp(0, lines.length - 1);
    if (lines[currentLine].trim().isEmpty) {
      return TextPosition(currentLine, 0);
    }
    while (currentLine > 0 && lines[currentLine - 1].trim().isNotEmpty) {
      currentLine--;
    }
    return TextPosition(currentLine, 0);
  }

  /// Finds the end position of the paragraph surrounding [position].
  static TextPosition getParagraphEnd(
    List<String> lines,
    TextPosition position,
  ) {
    if (lines.isEmpty) return const TextPosition(0, 0);
    int currentLine = position.line.clamp(0, lines.length - 1);
    if (lines[currentLine].trim().isEmpty) {
      return TextPosition(currentLine, lines[currentLine].length);
    }
    while (currentLine < lines.length - 1 &&
        lines[currentLine + 1].trim().isNotEmpty) {
      currentLine++;
    }
    return TextPosition(currentLine, lines[currentLine].length);
  }

  /// Moves cursor up by [pageSize] lines.
  static TextPosition movePageUp(TextPosition position, int pageSize) {
    final targetLine = math.max(0, position.line - pageSize);
    return TextPosition(targetLine, position.column);
  }

  /// Moves cursor down by [pageSize] lines given [totalLines].
  static TextPosition movePageDown(
    TextPosition position,
    int totalLines,
    int pageSize,
  ) {
    if (totalLines <= 0) return const TextPosition(0, 0);
    final targetLine = math.min(totalLines - 1, position.line + pageSize);
    return TextPosition(targetLine, position.column);
  }

  static bool _isWordCharacter(String char) {
    if (char.isEmpty) return false;
    final code = char.codeUnitAt(0);
    return (code >= 48 && code <= 57) || // 0-9
        (code >= 65 && code <= 90) || // A-Z
        (code >= 97 && code <= 122) || // a-z
        code == 95; // _
  }

  static bool _isWhitespace(String char) {
    return char == ' ' || char == '\t' || char == '\n' || char == '\r';
  }
}
