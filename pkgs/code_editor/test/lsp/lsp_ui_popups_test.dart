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

import 'package:code_editor/lsp.dart';
import 'package:code_editor/render.dart';
import 'package:test/test.dart';

void main() {
  group('DiagnosticSquiggle & LspDiagnosticAdapter', () {
    test('DiagnosticSquiggle geometry and severity styling', () {
      const errorSquiggle = DiagnosticSquiggle(
        line: 2,
        startColumn: 4,
        endColumn: 12,
        severity: DiagnosticSeverity.error,
        message: 'Syntax error',
        source: 'dart',
      );

      expect(errorSquiggle.line, equals(2));
      expect(errorSquiggle.startColumn, equals(4));
      expect(errorSquiggle.endColumn, equals(12));
      expect(errorSquiggle.colorHex, equals('#FF0000'));

      const warningSquiggle = DiagnosticSquiggle(
        line: 2,
        startColumn: 4,
        endColumn: 12,
        severity: DiagnosticSeverity.warning,
        message: 'Unused variable',
      );
      expect(warningSquiggle.colorHex, equals('#FFFF00'));

      const infoSquiggle = DiagnosticSquiggle(
        line: 2,
        startColumn: 4,
        endColumn: 12,
        severity: DiagnosticSeverity.info,
        message: 'Consider refactoring',
      );
      expect(infoSquiggle.colorHex, equals('#0000FF'));

      const hintSquiggle = DiagnosticSquiggle(
        line: 2,
        startColumn: 4,
        endColumn: 12,
        severity: DiagnosticSeverity.hint,
        message: 'Type annotation',
      );
      expect(hintSquiggle.colorHex, equals('#808080'));
    });

    test(
      'LspDiagnosticAdapter converts raw LSP diagnostics to line squiggles',
      () {
        final rawDiagnostics = [
          {
            'range': {
              'start': {'line': 1, 'character': 5},
              'end': {'line': 3, 'character': 10},
            },
            'message': 'Multi-line issue',
            'severity': 1,
            'source': 'analyzer',
          },
        ];

        final squiggles = LspDiagnosticAdapter.fromLspDiagnostics(
          rawDiagnostics,
        );
        expect(squiggles.length, equals(3));

        // Line 1: startCol 5 to endCol 999
        expect(squiggles[0].line, equals(1));
        expect(squiggles[0].startColumn, equals(5));
        expect(squiggles[0].endColumn, equals(999));

        // Line 2: startCol 0 to endCol 999
        expect(squiggles[1].line, equals(2));
        expect(squiggles[1].startColumn, equals(0));
        expect(squiggles[1].endColumn, equals(999));

        // Line 3: startCol 0 to endCol 10
        expect(squiggles[2].line, equals(3));
        expect(squiggles[2].startColumn, equals(0));
        expect(squiggles[2].endColumn, equals(10));
      },
    );
  });

  group('CompletionPopupModel & CompletionDropdownItem', () {
    test('CompletionDropdownItem insertion text and attributes', () {
      const item1 = CompletionDropdownItem(
        label: 'main',
        kind: 'Function',
        detail: 'void main()',
        documentation: 'Entry point',
      );
      expect(item1.insertText, equals('main'));

      const item2 = CompletionDropdownItem(
        label: 'print',
        insertText: 'print(\$1);',
      );
      expect(item2.insertText, equals('print(\$1);'));
    });

    test('CompletionPopupModel positioning, selection, and copyWith', () {
      const items = [
        CompletionDropdownItem(label: 'abs', kind: 'Method'),
        CompletionDropdownItem(label: 'add', kind: 'Method'),
        CompletionDropdownItem(label: 'addAll', kind: 'Method'),
      ];

      const popup = CompletionPopupModel(
        x: 120.0,
        y: 240.0,
        items: items,
        selectedIndex: 1,
      );

      expect(popup.x, equals(120.0));
      expect(popup.y, equals(240.0));
      expect(popup.items.length, equals(3));
      expect(popup.selectedIndex, equals(1));
      expect(popup.selectedItem?.label, equals('add'));
      expect(popup.isVisible, isTrue);

      final updated = popup.copyWith(selectedIndex: 1, searchPrefix: 'add');
      expect(updated.selectedIndex, equals(1));
      expect(updated.selectedItem?.label, equals('addAll'));
      expect(updated.searchPrefix, equals('add'));
      expect(updated.x, equals(120.0));
    });
  });

  group('HoverTooltipModel', () {
    test('HoverTooltipModel positioning and content', () {
      const tooltip = HoverTooltipModel(
        x: 50.0,
        y: 100.0,
        markdownContent: '```dart\nvoid print(Object? object)\n```',
      );

      expect(tooltip.x, equals(50.0));
      expect(tooltip.y, equals(100.0));
      expect(tooltip.markdownContent, contains('void print'));

      final moved = tooltip.copyWith(x: 60.0, y: 110.0);
      expect(moved.x, equals(60.0));
      expect(moved.y, equals(110.0));
      expect(moved.markdownContent, equals(tooltip.markdownContent));
    });
  });

  group('RenderViewport integration with LSP UI models', () {
    test(
      'RenderViewport includes squiggles, completion popup, and hover tooltip',
      () {
        const squiggle = DiagnosticSquiggle(
          line: 0,
          startColumn: 0,
          endColumn: 5,
          severity: DiagnosticSeverity.error,
          message: 'Error message',
        );

        const popup = CompletionPopupModel(
          x: 10.0,
          y: 20.0,
          items: [CompletionDropdownItem(label: 'test')],
        );

        const tooltip = HoverTooltipModel(
          x: 10.0,
          y: 40.0,
          markdownContent: 'Hover content',
        );

        const viewport = RenderViewport(
          width: 800.0,
          height: 600.0,
          scrollX: 0.0,
          scrollY: 0.0,
          firstVisibleLine: 0,
          lastVisibleLine: 10,
          lines: [],
          selections: [],
          carets: [],
          gutter: RenderGutter(width: 40.0, items: []),
          lineHeight: 20.0,
          charWidth: 10.0,
          squiggles: [squiggle],
          completionPopup: popup,
          hoverTooltip: tooltip,
        );

        expect(viewport.squiggles.length, equals(1));
        expect(viewport.squiggles.first.message, equals('Error message'));
        expect(viewport.completionPopup, equals(popup));
        expect(viewport.hoverTooltip, equals(tooltip));
      },
    );
  });
}
