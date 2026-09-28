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

import 'package:code_editor/core.dart';
import 'package:code_editor/lsp.dart';
import 'package:code_editor/render.dart';
import 'package:code_editor/syntax.dart';
import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('Code Editor End-to-End Integration Tests', () {
    late PieceTreeTextBuffer buffer;

    setUp(() {
      buffer = PieceTreeTextBuffer('void main() {\n  int count = 42;\n}');
    });

    test(
      'Full pipeline: Buffer -> Edits -> Syntax -> LSP Popups/Squiggles -> Layout -> Renderers',
      () {
        // 1. Verify Initial Text Buffer
        expect(buffer.lineCount, equals(3));
        expect(buffer.getLine(1).trimRight(), equals('  int count = 42;'));

        // 2. Apply Invertible Operations
        final insertOp = InsertOperation(17, '\n  print("Count: \$count");');
        buffer.insert(17, '\n  print("Count: \$count");');
        expect(buffer.lineCount, equals(4));

        final inverseOp = insertOp.invert();
        expect(inverseOp, isA<DeleteOperation>());

        // 3. Syntax Tokenization & Scope Matching
        final theme = ColorTheme.darkPlus();
        final style = theme.resolveStyle([StyleScope('keyword.control')]);
        expect(style.foreground, equals(0xFF569CD6));

        // 4. Diagnostics & Squiggles Adapter
        final squiggles = LspDiagnosticAdapter.fromLspDiagnostics([
          {
            'range': {
              'start': {'line': 1, 'character': 6},
              'end': {'line': 1, 'character': 11},
            },
            'message': "Unused variable 'count'",
            'severity': 2, // Warning
            'source': 'analyzer',
          },
        ]);
        expect(squiggles.length, equals(1));
        expect(squiggles.first.colorHex, equals('#FFFF00'));

        // 5. LSP completion & hover models
        const completionPopup = CompletionPopupModel(
          x: 48,
          y: 40,
          items: [
            CompletionDropdownItem(
              label: 'count',
              insertText: 'count',
              detail: 'int',
            ),
            CompletionDropdownItem(label: 'toString', insertText: 'toString()'),
          ],
          selectedIndex: 0,
        );

        const hoverTooltip = HoverTooltipModel(
          x: 48,
          y: 60,
          markdownContent: 'int count = 42\nLocal variable.',
        );

        // 6. Calculate Virtual Layout & Viewport Payload
        const engine = LineWrappingEngine(maxColumns: 80);
        final calc = VirtualLayoutCalculator(lineWrappingEngine: engine);
        calc.computeLayout(buffer.lines);
        expect(calc.totalVirtualRows, equals(4));

        final renderViewport = RenderViewport(
          width: 800,
          height: 600,
          scrollX: 0,
          scrollY: 0,
          firstVisibleLine: 0,
          lastVisibleLine: 3,
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
            RenderLine(
              lineIndex: 1,
              text: '  int count = 42;',
              tokens: [
                RenderToken(
                  text: '  int ',
                  startColumn: 0,
                  endColumn: 6,
                  style: RenderTokenStyle(foreground: '#569CD6'),
                ),
                RenderToken(
                  text: 'count',
                  startColumn: 6,
                  endColumn: 11,
                  style: RenderTokenStyle(foreground: '#9CDCFE'),
                ),
              ],
              top: 20,
              height: 20,
            ),
          ],
          selections: const [
            SelectionRect(left: 0, top: 0, width: 32, height: 20),
          ],
          carets: const [
            CaretPosition(x: 32, y: 0, height: 20, line: 0, column: 4),
          ],
          gutter: const RenderGutter(
            width: 40,
            items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
          ),
          lineHeight: 20,
          charWidth: 8,
          squiggles: squiggles,
          completionPopup: completionPopup,
          hoverTooltip: hoverTooltip,
        );

        // 7. Test Multi-Backend UI Renderers

        // 7a. Flutter Canvas Renderer
        final flutterRenderer = FlutterRenderer();
        flutterRenderer.attach('flutter_canvas');
        flutterRenderer.render(renderViewport);
        expect(flutterRenderer.isAttached, isTrue);

        final commands = flutterRenderer.recordedCommands;
        expect(
          commands.any(
            (c) =>
                c.type == CanvasCommandType.drawSquiggle &&
                c.color == '#FFFF00',
          ),
          isTrue,
        );
        expect(
          commands.any(
            (c) => c.type == CanvasCommandType.drawText && c.text == 'count',
          ),
          isTrue,
        );
        expect(
          commands.any(
            (c) =>
                c.type == CanvasCommandType.drawText &&
                c.text == 'Local variable.',
          ),
          isTrue,
        );

        // 7b. ANSI Terminal Renderer
        final terminalRenderer = TerminalRenderer(rows: 15, cols: 50);
        terminalRenderer.attach('terminal_stdout');
        final ansiOutput = terminalRenderer.renderDeltaToString(renderViewport);
        expect(ansiOutput, contains(Vt100Encoder.reset));
        expect(ansiOutput, contains('c'));
        expect(ansiOutput, contains(Vt100Encoder.hexToFg('#FFFF00')));
        expect(ansiOutput, contains('│'));
      },
    );
  });
}
