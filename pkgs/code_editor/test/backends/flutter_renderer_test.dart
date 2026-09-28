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

import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('ParagraphCache Tests', () {
    test('Cache put, get, and eviction', () {
      final cache = ParagraphCache(maxCapacity: 2);
      const style = RenderTokenStyle(bold: true);

      const line1 = RenderLine(
        lineIndex: 0,
        text: 'First',
        tokens: [],
        top: 0,
        height: 20,
      );
      const line2 = RenderLine(
        lineIndex: 1,
        text: 'Second',
        tokens: [],
        top: 20,
        height: 20,
      );
      const line3 = RenderLine(
        lineIndex: 2,
        text: 'Third',
        tokens: [],
        top: 40,
        height: 20,
      );

      cache.put('First', style, line1);
      cache.put('Second', style, line2);
      expect(cache.size, equals(2));
      expect(cache.get('First', style), equals(line1));

      // Adding 3rd item triggers eviction of oldest item ('First')
      cache.put('Third', style, line3);
      expect(cache.size, equals(2));
      expect(cache.get('First', style), isNull);
      expect(cache.get('Second', style), equals(line2));
      expect(cache.get('Third', style), equals(line3));
    });
  });

  group('FlutterGestureAdapter Tests', () {
    test('positionFromOffset converts offset coordinates to line/column', () {
      final adapter = FlutterGestureAdapter(
        charWidth: 10.0,
        lineHeight: 20.0,
        gutterWidth: 40.0,
      );

      final (line1, col1) = adapter.positionFromOffset(40.0, 0.0);
      expect(line1, equals(0));
      expect(col1, equals(0));

      final (line2, col2) = adapter.positionFromOffset(90.0, 25.0);
      expect(line2, equals(1));
      expect(col2, equals(5));
    });
  });

  group('FlutterRenderer CanvasDrawCommand Tests', () {
    late FlutterRenderer renderer;

    setUp(() {
      renderer = FlutterRenderer();
    });

    test(
      'render records draw commands for line text, selection, carets, squiggles, and popups',
      () {
        final viewport = RenderViewport(
          width: 800,
          height: 600,
          scrollX: 0,
          scrollY: 0,
          firstVisibleLine: 0,
          lastVisibleLine: 2,
          lines: const [
            RenderLine(
              lineIndex: 0,
              text: 'void main() {',
              tokens: [
                RenderToken(
                  text: 'void',
                  startColumn: 0,
                  endColumn: 4,
                  style: RenderTokenStyle(foreground: '#569CD6'),
                ),
                RenderToken(
                  text: ' main() {',
                  startColumn: 4,
                  endColumn: 13,
                  style: RenderTokenStyle(foreground: '#DCDCAA'),
                ),
              ],
              top: 0,
              height: 20,
            ),
          ],
          selections: const [
            SelectionRect(left: 40, top: 0, width: 40, height: 20),
          ],
          carets: const [
            CaretPosition(x: 80, y: 0, height: 20, line: 0, column: 4),
          ],
          gutter: const RenderGutter(
            width: 40,
            items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
          ),
          lineHeight: 20,
          charWidth: 10,
          squiggles: const [
            DiagnosticSquiggle(
              line: 0,
              startColumn: 0,
              endColumn: 4,
              message: 'Undefined function',
              severity: DiagnosticSeverity.error,
            ),
          ],
          completionPopup: const CompletionPopupModel(
            x: 80,
            y: 20,
            items: [
              CompletionDropdownItem(label: 'print', insertText: 'print()'),
              CompletionDropdownItem(label: 'parse', insertText: 'parse()'),
            ],
            selectedIndex: 0,
          ),
          hoverTooltip: const HoverTooltipModel(
            x: 80,
            y: 60,
            markdownContent: 'void main()\nEntry point',
          ),
        );

        renderer.attach('canvas_target');
        renderer.render(viewport);

        final commands = renderer.recordedCommands;
        expect(commands, isNotEmpty);

        // Verify Gutter fillRect
        expect(
          commands.any(
            (c) => c.type == CanvasCommandType.fillRect && c.width == 40,
          ),
          isTrue,
        );

        // Verify Line Text tokens
        expect(
          commands.any(
            (c) => c.type == CanvasCommandType.drawText && c.text == 'void',
          ),
          isTrue,
        );

        // Verify Selection rect
        expect(
          commands.any(
            (c) =>
                c.type == CanvasCommandType.fillRect &&
                c.x == 40 &&
                c.width == 40,
          ),
          isTrue,
        );

        // Verify Diagnostic Squiggle
        expect(
          commands.any(
            (c) =>
                c.type == CanvasCommandType.drawSquiggle &&
                c.color == '#FF0000' &&
                c.text == 'Undefined function',
          ),
          isTrue,
        );

        // Verify Caret fillRect
        expect(
          commands.any(
            (c) =>
                c.type == CanvasCommandType.fillRect &&
                c.x == 80 &&
                c.width == 2.0,
          ),
          isTrue,
        );

        // Verify Completion Popup dropdown commands
        expect(
          commands.any(
            (c) => c.type == CanvasCommandType.drawText && c.text == 'print',
          ),
          isTrue,
        );
        expect(
          commands.any(
            (c) => c.type == CanvasCommandType.fillRect && c.color == '#04395E',
          ),
          isTrue,
        );

        // Verify Hover Tooltip popup commands
        expect(
          commands.any(
            (c) =>
                c.type == CanvasCommandType.drawText && c.text == 'void main()',
          ),
          isTrue,
        );
        expect(
          commands.any(
            (c) =>
                c.type == CanvasCommandType.drawText && c.text == 'Entry point',
          ),
          isTrue,
        );
      },
    );

    test('handleTapDown dispatches tap event with mapped line and column', () {
      renderer.attach('canvas_target');
      const viewport = RenderViewport(
        width: 800,
        height: 600,
        scrollX: 0,
        scrollY: 0,
        firstVisibleLine: 0,
        lastVisibleLine: 5,
        lines: [],
        selections: [],
        carets: [],
        gutter: RenderGutter(width: 40, items: []),
        lineHeight: 20,
        charWidth: 10,
      );

      renderer.render(viewport);

      Map<String, dynamic>? eventData;
      renderer.addEventListener((event, data) {
        if (event == 'tap') eventData = data as Map<String, dynamic>?;
      });

      renderer.handleTapDown(90.0, 45.0);
      expect(eventData, isNotNull);
      expect(eventData!['line'], equals(2));
      expect(eventData!['column'], equals(5));
    });
  });
}
