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

import 'package:code_editor/backends/terminal.dart';
import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('Terminal Cell & Matrix Tests', () {
    test('TerminalCell value equality', () {
      const c1 = TerminalCell(
        char: 'A',
        fgColor: '#FFFFFF',
        bgColor: '#000000',
      );
      const c2 = TerminalCell(
        char: 'A',
        fgColor: '#FFFFFF',
        bgColor: '#000000',
      );
      const c3 = TerminalCell(
        char: 'B',
        fgColor: '#FFFFFF',
        bgColor: '#000000',
      );

      expect(c1, equals(c2));
      expect(c1, isNot(equals(c3)));
    });

    test('TerminalMatrix grid bounds and flat packed typed arrays', () {
      final matrix = TerminalMatrix(10, 20);
      expect(matrix.rows, equals(10));
      expect(matrix.cols, equals(20));
      expect(matrix.charCodes.length, equals(200));
      expect(matrix.fgColors.length, equals(200));
      expect(matrix.bgColors.length, equals(200));
      expect(matrix.attributes.length, equals(200));

      const cell = TerminalCell(
        char: 'X',
        fgColor: '#FF0000',
        bgColor: '#00FF00',
        bold: true,
        italic: true,
        underline: true,
      );
      matrix.setCell(2, 5, cell);
      expect(matrix.getCell(2, 5), equals(cell));
      expect(matrix.charCodes[2 * 20 + 5], equals('X'.codeUnitAt(0)));
      expect(matrix.attributes[2 * 20 + 5], equals(7)); // 1 | 2 | 4 = 7

      matrix.clear();
      expect(matrix.getCell(2, 5), equals(const TerminalCell()));
    });
  });

  group('TerminalRenderer Tests', () {
    late TerminalRenderer renderer;

    setUp(() {
      renderer = TerminalRenderer(rows: 8, cols: 30);
    });

    test('renderDeltaToString generates minimal ANSI output string', () {
      const viewport = RenderViewport(
        width: 160,
        height: 100,
        scrollX: 0,
        scrollY: 0,
        firstVisibleLine: 0,
        lastVisibleLine: 1,
        lines: [
          RenderLine(
            lineIndex: 0,
            text: 'Hello',
            tokens: [
              RenderToken(
                text: 'Hello',
                startColumn: 0,
                endColumn: 5,
                style: RenderTokenStyle(foreground: '#569CD6'),
              ),
            ],
            top: 0,
            height: 20,
          ),
        ],
        selections: <SelectionRect>[],
        carets: [CaretPosition(x: 48, y: 0, height: 20, line: 0, column: 1)],
        gutter: RenderGutter(
          width: 24,
          items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
        ),
        lineHeight: 20,
        charWidth: 8,
      );

      final ansiOutput = renderer.renderDeltaToString(viewport);
      expect(
        ansiOutput,
        contains('\x1b[1;1H'),
      ); // Cursor position to line 1, col 1
      expect(ansiOutput, contains('1')); // Gutter line number
      expect(ansiOutput, contains('H'));
      expect(ansiOutput, contains('e'));
      expect(ansiOutput, contains('l'));
      expect(ansiOutput, contains('o'));
      expect(ansiOutput, contains(Vt100Encoder.reset));
    });

    test('renderDeltaToString outputs ANSI attribute reset sequences', () {
      const viewport = RenderViewport(
        width: 160,
        height: 100,
        scrollX: 0,
        scrollY: 0,
        firstVisibleLine: 0,
        lastVisibleLine: 1,
        lines: [
          RenderLine(
            lineIndex: 0,
            text: 'BoldPlain',
            tokens: [
              RenderToken(
                text: 'Bold',
                startColumn: 0,
                endColumn: 4,
                style: RenderTokenStyle(
                  bold: true,
                  italic: true,
                  underline: true,
                ),
              ),
              RenderToken(
                text: 'Plain',
                startColumn: 4,
                endColumn: 9,
                style: RenderTokenStyle(
                  bold: false,
                  italic: false,
                  underline: false,
                ),
              ),
            ],
            top: 0,
            height: 20,
          ),
        ],
        selections: <SelectionRect>[],
        carets: <CaretPosition>[],
        gutter: RenderGutter(
          width: 24,
          items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
        ),
        lineHeight: 20,
        charWidth: 8,
      );

      final ansiOutput = renderer.renderDeltaToString(viewport);
      expect(ansiOutput, contains(Vt100Encoder.bold));
      expect(ansiOutput, contains(Vt100Encoder.italic));
      expect(ansiOutput, contains(Vt100Encoder.underline));
      expect(ansiOutput, contains(Vt100Encoder.resetBold));
      expect(ansiOutput, contains(Vt100Encoder.resetItalic));
      expect(ansiOutput, contains(Vt100Encoder.resetUnderline));
    });

    test(
      'renderDeltaToString renders diagnostic squiggles as colored underlines',
      () {
        final viewport = RenderViewport(
          width: 160,
          height: 100,
          scrollX: 0,
          scrollY: 0,
          firstVisibleLine: 0,
          lastVisibleLine: 1,
          lines: const [
            RenderLine(
              lineIndex: 0,
              text: 'errorText',
              tokens: [
                RenderToken(
                  text: 'errorText',
                  startColumn: 0,
                  endColumn: 9,
                  style: RenderTokenStyle(foreground: '#CCCCCC'),
                ),
              ],
              top: 0,
              height: 20,
            ),
          ],
          selections: const <SelectionRect>[],
          carets: const <CaretPosition>[],
          gutter: const RenderGutter(
            width: 24,
            items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
          ),
          lineHeight: 20,
          charWidth: 8,
          squiggles: const [
            DiagnosticSquiggle(
              line: 0,
              startColumn: 0,
              endColumn: 5,
              message: 'Error here',
              severity: DiagnosticSeverity.error,
            ),
          ],
        );

        final ansiOutput = renderer.renderDeltaToString(viewport);
        expect(ansiOutput, contains(Vt100Encoder.underline));
        expect(ansiOutput, contains(Vt100Encoder.hexToFg('#FF0000')));
      },
    );

    test(
      'renderDeltaToString renders completion dropdown popup as inline character box overlay',
      () {
        final viewport = RenderViewport(
          width: 160,
          height: 100,
          scrollX: 0,
          scrollY: 0,
          firstVisibleLine: 0,
          lastVisibleLine: 2,
          lines: const [],
          selections: const [],
          carets: const [],
          gutter: const RenderGutter(
            width: 24,
            items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
          ),
          lineHeight: 20,
          charWidth: 8,
          completionPopup: const CompletionPopupModel(
            x: 24,
            y: 20,
            items: [
              CompletionDropdownItem(
                label: 'toString',
                insertText: 'toString()',
              ),
              CompletionDropdownItem(label: 'toInt', insertText: 'toInt()'),
            ],
            selectedIndex: 0,
          ),
        );

        final ansiOutput = renderer.renderDeltaToString(viewport);
        expect(ansiOutput, contains('│'));
        expect(ansiOutput, contains('t'));
        expect(ansiOutput, contains('S'));
        expect(ansiOutput, contains(Vt100Encoder.hexToBg('#04395E')));
      },
    );

    test(
      'renderDeltaToString renders hover tooltip popup as styled popover box overlay',
      () {
        final viewport = RenderViewport(
          width: 160,
          height: 100,
          scrollX: 0,
          scrollY: 0,
          firstVisibleLine: 0,
          lastVisibleLine: 2,
          lines: const [],
          selections: const [],
          carets: const [],
          gutter: const RenderGutter(
            width: 24,
            items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
          ),
          lineHeight: 20,
          charWidth: 8,
          hoverTooltip: const HoverTooltipModel(
            x: 24,
            y: 40,
            markdownContent: 'String toString()\nReturns text representation.',
          ),
        );

        final ansiOutput = renderer.renderDeltaToString(viewport);
        expect(ansiOutput, contains('│'));
        expect(ansiOutput, contains('S'));
        expect(ansiOutput, contains('t'));
        expect(ansiOutput, contains(Vt100Encoder.hexToFg('#D4D4D4')));
      },
    );
  });

  group('TerminalInputParser Tests', () {
    late TerminalInputParser parser;

    setUp(() {
      parser = TerminalInputParser();
    });

    test('parseInput decodes VT100 arrow key sequences', () {
      KeyCombination? capturedCombo;
      parser.addKeyListener((KeyCombination combo) {
        capturedCombo = combo;
      });

      parser.parseInput('\x1b[A');
      expect(capturedCombo, equals(const KeyCombination('ArrowUp')));

      parser.parseInput('\x1b[B');
      expect(capturedCombo, equals(const KeyCombination('ArrowDown')));
    });

    test('parseInput decodes SGR mouse reporting sequence', () {
      TerminalMouseEvent? capturedMouse;
      parser.addMouseListener((TerminalMouseEvent mouse) {
        capturedMouse = mouse;
      });

      // Press mouse button 0 at x=10, y=5
      parser.parseInput('\x1b[<0;10;5M');
      expect(capturedMouse, isNotNull);
      expect(capturedMouse!.button, equals(0));
      expect(capturedMouse!.x, equals(10));
      expect(capturedMouse!.y, equals(5));
      expect(capturedMouse!.isRelease, isFalse);

      // Release mouse button 0
      parser.parseInput('\x1b[<0;10;5m');
      expect(capturedMouse!.isRelease, isTrue);
    });

    test('parseInput decodes Ctrl key combinations', () {
      KeyCombination? capturedCombo;
      parser.addKeyListener((KeyCombination combo) {
        capturedCombo = combo;
      });

      // Ctrl+C is ASCII code 3
      parser.parseInput('\x03');
      expect(capturedCombo, equals(const KeyCombination('C', ctrl: true)));
    });

    test(
      'parseInput decodes Delete, Alt+Key, and Ctrl+Arrow sequences without phantom Escape',
      () {
        final capturedCombos = <KeyCombination>[];
        parser.addKeyListener((KeyCombination combo) {
          capturedCombos.add(combo);
        });

        // Delete key: \x1b[3~
        parser.parseInput('\x1b[3~');
        expect(capturedCombos, equals([const KeyCombination('Delete')]));
        capturedCombos.clear();

        // Alt+Key: \x1ba
        parser.parseInput('\x1ba');
        expect(capturedCombos, equals([const KeyCombination('a', alt: true)]));
        capturedCombos.clear();

        // Ctrl+RightArrow: \x1b[1;5C
        parser.parseInput('\x1b[1;5C');
        expect(
          capturedCombos,
          equals([const KeyCombination('ArrowRight', ctrl: true)]),
        );
        capturedCombos.clear();

        // Standalone Escape
        parser.parseInput('\x1b');
        expect(capturedCombos, equals([const KeyCombination('Escape')]));
      },
    );
  });
}
