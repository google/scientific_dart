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

import 'dart:typed_data';

import '../../render/editor_renderer.dart';
import '../../render/render_viewport.dart';
import 'vt100_encoder.dart';

int _hexToInt(String hex) {
  var cleaned = hex.replaceAll('#', '').trim();
  if (cleaned.startsWith('rgba(') || cleaned.startsWith('rgb(')) {
    final numbers = RegExp(
      r'\d+',
    ).allMatches(cleaned).map((m) => int.parse(m.group(0)!)).toList();
    if (numbers.length >= 3) {
      return (numbers[0] << 16) | (numbers[1] << 8) | numbers[2];
    }
  }
  if (cleaned.length == 3 || cleaned.length == 4) {
    cleaned = cleaned.split('').map((c) => '$c$c').join();
  }
  if (cleaned.length == 8) {
    cleaned = cleaned.substring(0, 6);
  }
  if (cleaned.length != 6) {
    return 0xFFFFFF;
  }
  return int.tryParse(cleaned, radix: 16) ?? 0xFFFFFF;
}

String _intToHex(int val) {
  final hex = val.toRadixString(16).padLeft(6, '0').toUpperCase();
  return '#$hex';
}

/// Single cell in the terminal visual character matrix.
class TerminalCell {
  final String char;
  final String fgColor;
  final String bgColor;
  final bool bold;
  final bool italic;
  final bool underline;

  const TerminalCell({
    this.char = ' ',
    this.fgColor = '#CCCCCC',
    this.bgColor = '#1E1E1E',
    this.bold = false,
    this.italic = false,
    this.underline = false,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is TerminalCell &&
          runtimeType == other.runtimeType &&
          char == other.char &&
          fgColor == other.fgColor &&
          bgColor == other.bgColor &&
          bold == other.bold &&
          italic == other.italic &&
          underline == other.underline;

  @override
  int get hashCode =>
      Object.hash(char, fgColor, bgColor, bold, italic, underline);
}

/// Wasm-optimized 2D grid matrix of terminal cells using flat packed TypedArrays.
class TerminalMatrix {
  final int rows;
  final int cols;
  final Int32List charCodes;
  final Uint32List fgColors;
  final Uint32List bgColors;
  final Uint32List attributes;

  static final int defaultFgInt = _hexToInt('#CCCCCC');
  static final int defaultBgInt = _hexToInt('#1E1E1E');

  TerminalMatrix(this.rows, this.cols)
    : charCodes = Int32List(rows * cols),
      fgColors = Uint32List(rows * cols),
      bgColors = Uint32List(rows * cols),
      attributes = Uint32List(rows * cols) {
    clear();
  }

  void setCell(int row, int col, TerminalCell cell) {
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      final idx = row * cols + col;
      charCodes[idx] = cell.char.isEmpty ? 32 : cell.char.codeUnitAt(0);
      fgColors[idx] = _hexToInt(cell.fgColor);
      bgColors[idx] = _hexToInt(cell.bgColor);
      var attr = 0;
      if (cell.bold) attr |= 1;
      if (cell.italic) attr |= 2;
      if (cell.underline) attr |= 4;
      attributes[idx] = attr;
    }
  }

  TerminalCell getCell(int row, int col) {
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      final idx = row * cols + col;
      final code = charCodes[idx];
      final charStr = code == 0 ? ' ' : String.fromCharCode(code);
      final fg = _intToHex(fgColors[idx]);
      final bg = _intToHex(bgColors[idx]);
      final attr = attributes[idx];
      return TerminalCell(
        char: charStr,
        fgColor: fg,
        bgColor: bg,
        bold: (attr & 1) != 0,
        italic: (attr & 2) != 0,
        underline: (attr & 4) != 0,
      );
    }
    return const TerminalCell();
  }

  void clear() {
    charCodes.fillRange(0, charCodes.length, 32);
    fgColors.fillRange(0, fgColors.length, defaultFgInt);
    bgColors.fillRange(0, bgColors.length, defaultBgInt);
    attributes.fillRange(0, attributes.length, 0);
  }
}

/// Double-buffered ANSI Terminal renderer performing minimum delta stdout rendering.
class TerminalRenderer implements EditorRenderer {
  final int rows;
  final int cols;
  final List<RenderEventListener> _listeners = [];
  bool _isAttached = false;
  Object? _target;
  RenderViewport? _lastViewport;

  late TerminalMatrix _frontBuffer;
  late TerminalMatrix _backBuffer;

  TerminalRenderer({this.rows = 24, this.cols = 80}) {
    _frontBuffer = TerminalMatrix(rows, cols);
    _backBuffer = TerminalMatrix(rows, cols);
  }

  @override
  bool get isAttached => _isAttached;

  /// Attached stdout target surface.
  Object? get target => _target;

  @override
  RenderViewport? get lastViewport => _lastViewport;

  @override
  void attach(Object target) {
    _target = target;
    _isAttached = true;
  }

  @override
  void detach() {
    _target = null;
    _isAttached = false;
  }

  @override
  void addEventListener(RenderEventListener listener) {
    _listeners.add(listener);
  }

  @override
  void removeEventListener(RenderEventListener listener) {
    _listeners.remove(listener);
  }

  @override
  void render(RenderViewport viewport) {
    _lastViewport = viewport;
    renderDeltaToString(viewport);
  }

  /// Compute back-buffer and return minimal ANSI delta escape string to update terminal screen.
  String renderDeltaToString(RenderViewport viewport) {
    _lastViewport = viewport;
    _backBuffer.clear();

    final gutterWidthChars = (viewport.gutter.width / viewport.charWidth)
        .round()
        .clamp(1, cols - 1);

    // 1. Render Gutter to back buffer
    for (var r = 0; r < viewport.gutter.items.length && r < rows; r++) {
      final item = viewport.gutter.items[r];
      final lineText = '${item.lineNumberText.padLeft(gutterWidthChars - 1)} ';
      for (var c = 0; c < lineText.length && c < gutterWidthChars; c++) {
        _backBuffer.setCell(
          r,
          c,
          TerminalCell(
            char: lineText[c],
            fgColor: '#858585',
            bgColor: '#1E1E1E',
          ),
        );
      }
    }

    // 2. Render Text Lines & Tokens to back buffer
    for (var r = 0; r < viewport.lines.length && r < rows; r++) {
      final line = viewport.lines[r];
      var currentCol = gutterWidthChars;

      if (line.tokens.isEmpty) {
        for (
          var i = 0;
          i < line.text.length && currentCol < cols;
          i++, currentCol++
        ) {
          _backBuffer.setCell(
            r,
            currentCol,
            TerminalCell(
              char: line.text[i],
              fgColor: '#CCCCCC',
              bgColor: '#1E1E1E',
            ),
          );
        }
      } else {
        for (final token in line.tokens) {
          final fg = token.style.foreground ?? '#CCCCCC';
          final bg = token.style.background ?? '#1E1E1E';
          for (
            var i = 0;
            i < token.text.length && currentCol < cols;
            i++, currentCol++
          ) {
            _backBuffer.setCell(
              r,
              currentCol,
              TerminalCell(
                char: token.text[i],
                fgColor: fg,
                bgColor: bg,
                bold: token.style.bold,
                italic: token.style.italic,
                underline: token.style.underline,
              ),
            );
          }
        }
      }
    }

    // 3. Render Selections to back buffer
    for (final rect in viewport.selections) {
      final startRow = (rect.top / viewport.lineHeight).round().clamp(
        0,
        rows - 1,
      );
      final startCol = (rect.left / viewport.charWidth).round().clamp(
        0,
        cols - 1,
      );
      final widthCols = (rect.width / viewport.charWidth).round();

      for (var c = startCol; c < startCol + widthCols && c < cols; c++) {
        final existing = _backBuffer.getCell(startRow, c);
        _backBuffer.setCell(
          startRow,
          c,
          TerminalCell(
            char: existing.char,
            fgColor: '#FFFFFF',
            bgColor: '#264F78',
            bold: existing.bold,
            italic: existing.italic,
            underline: existing.underline,
          ),
        );
      }
    }

    // 4. Render Diagnostic Squiggles as colored terminal underlines
    for (final squiggle in viewport.squiggles) {
      final r = squiggle.line - viewport.firstVisibleLine;
      if (r >= 0 && r < rows) {
        final startCol = gutterWidthChars + squiggle.startColumn;
        final endCol = gutterWidthChars + squiggle.endColumn;
        for (var c = startCol; c < endCol && c < cols; c++) {
          if (c >= gutterWidthChars) {
            final existing = _backBuffer.getCell(r, c);
            _backBuffer.setCell(
              r,
              c,
              TerminalCell(
                char: existing.char,
                fgColor: squiggle.colorHex,
                bgColor: existing.bgColor,
                bold: existing.bold,
                italic: existing.italic,
                underline: true,
              ),
            );
          }
        }
      }
    }

    // 5. Render Carets to back buffer
    for (final caret in viewport.carets) {
      final row = (caret.y / viewport.lineHeight).round().clamp(0, rows - 1);
      final col = (caret.x / viewport.charWidth).round().clamp(0, cols - 1);
      final existing = _backBuffer.getCell(row, col);
      _backBuffer.setCell(
        row,
        col,
        TerminalCell(
          char: existing.char.isEmpty || existing.char == ' '
              ? '_'
              : existing.char,
          fgColor: '#000000',
          bgColor: caret.isPrimary ? '#FFFFFF' : '#AAAAAA',
          bold: true,
        ),
      );
    }

    // 6. Render Completion Dropdown UI overlay
    if (viewport.completionPopup != null &&
        viewport.completionPopup!.isVisible) {
      final popup = viewport.completionPopup!;
      final items = popup.filteredItems;
      final startRow = (popup.y / viewport.lineHeight).round().clamp(
        0,
        rows - 1,
      );
      final startCol = (popup.x / viewport.charWidth).round().clamp(
        0,
        cols - 1,
      );

      int maxLabelLen = 10;
      for (final item in items) {
        if (item.label.length > maxLabelLen) maxLabelLen = item.label.length;
      }
      final boxWidth = (maxLabelLen + 4).clamp(15, cols - startCol);

      for (var i = 0; i < items.length; i++) {
        final r = startRow + i;
        if (r >= rows) break;

        final item = items[i];
        final isSelected = i == popup.selectedIndex;
        final fg = isSelected ? '#FFFFFF' : '#CCCCCC';
        final bg = isSelected ? '#04395E' : '#252526';

        final contentText = item.label.padRight(boxWidth - 2);
        final lineStr = '│$contentText│';

        for (var c = 0; c < lineStr.length && (startCol + c) < cols; c++) {
          _backBuffer.setCell(
            r,
            startCol + c,
            TerminalCell(
              char: lineStr[c],
              fgColor: fg,
              bgColor: bg,
              bold: isSelected,
            ),
          );
        }
      }
    }

    // 7. Render Hover Tooltip UI overlay
    if (viewport.hoverTooltip != null && viewport.hoverTooltip!.isVisible) {
      final tooltip = viewport.hoverTooltip!;
      final startRow = (tooltip.y / viewport.lineHeight).round().clamp(
        0,
        rows - 1,
      );
      final startCol = (tooltip.x / viewport.charWidth).round().clamp(
        0,
        cols - 1,
      );
      final rawLines = tooltip.markdownContent.split('\n');
      final lines = [
        if (tooltip.signature != null && tooltip.signature!.isNotEmpty)
          tooltip.signature!,
        ...rawLines,
      ];

      int maxLineLen = 10;
      for (final line in lines) {
        if (line.length > maxLineLen) maxLineLen = line.length;
      }
      final boxWidth = (maxLineLen + 4).clamp(15, cols - startCol);

      for (var i = 0; i < lines.length; i++) {
        final r = startRow + i;
        if (r >= rows) break;

        final contentText = lines[i].padRight(boxWidth - 2);
        final lineStr = '│$contentText│';

        for (var c = 0; c < lineStr.length && (startCol + c) < cols; c++) {
          _backBuffer.setCell(
            r,
            startCol + c,
            TerminalCell(
              char: lineStr[c],
              fgColor: (i == 0 && tooltip.signature != null)
                  ? '#569CD6'
                  : '#D4D4D4',
              bgColor: '#252526',
              bold: i == 0 && tooltip.signature != null,
            ),
          );
        }
      }
    }

    // 8. Matrix Diff & ANSI sequence generation
    final output = StringBuffer();
    String? currentFg;
    String? currentBg;
    bool activeBold = false;
    bool activeItalic = false;
    bool activeUnderline = false;

    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final backCell = _backBuffer.getCell(r, c);
        final frontCell = _frontBuffer.getCell(r, c);

        if (backCell != frontCell) {
          // Position cursor (1-indexed ANSI coordinates)
          output.write(Vt100Encoder.moveCursor(r + 1, c + 1));

          if (backCell.fgColor != currentFg) {
            output.write(Vt100Encoder.hexToFg(backCell.fgColor));
            currentFg = backCell.fgColor;
          }
          if (backCell.bgColor != currentBg) {
            output.write(Vt100Encoder.hexToBg(backCell.bgColor));
            currentBg = backCell.bgColor;
          }

          if (backCell.bold != activeBold) {
            output.write(
              backCell.bold ? Vt100Encoder.bold : Vt100Encoder.resetBold,
            );
            activeBold = backCell.bold;
          }
          if (backCell.italic != activeItalic) {
            output.write(
              backCell.italic ? Vt100Encoder.italic : Vt100Encoder.resetItalic,
            );
            activeItalic = backCell.italic;
          }
          if (backCell.underline != activeUnderline) {
            output.write(
              backCell.underline
                  ? Vt100Encoder.underline
                  : Vt100Encoder.resetUnderline,
            );
            activeUnderline = backCell.underline;
          }

          output.write(backCell.char);

          // Update front buffer cell
          _frontBuffer.setCell(r, c, backCell);
        }
      }
    }

    output.write(Vt100Encoder.reset);
    return output.toString();
  }

  @override
  void dispose() {
    detach();
    _listeners.clear();
  }
}
