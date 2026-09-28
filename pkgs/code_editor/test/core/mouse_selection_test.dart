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

import 'package:test/test.dart';
import 'package:code_editor/code_editor.dart';

void main() {
  group('MouseSelectionHandler Tests', () {
    final lines = [
      'first line of document',
      'second line with words',
      'third line',
    ];

    test('Single click creates collapsed selection', () {
      const target = TextPosition(1, 7);
      final sel = MouseSelectionHandler.handleSingleClick(target);

      expect(sel.isCollapsed, isTrue);
      expect(sel.base, equals(target));
      expect(sel.extent, equals(target));
    });

    test('Drag update maintains base and updates extent', () {
      const initial = TextSelection(
        base: TextPosition(0, 5),
        extent: TextPosition(0, 5),
      );
      const dragTarget = TextPosition(1, 10);

      final sel = MouseSelectionHandler.handleDragUpdate(initial, dragTarget);

      expect(sel.isCollapsed, isFalse);
      expect(sel.base, equals(const TextPosition(0, 5)));
      expect(sel.extent, equals(const TextPosition(1, 10)));
    });

    test('Double click selects word at target position', () {
      const target = TextPosition(1, 13); // inside "with"
      final sel = MouseSelectionHandler.handleDoubleClick(lines, target);

      expect(sel.start, equals(const TextPosition(1, 12)));
      expect(sel.end, equals(const TextPosition(1, 16)));
    });

    test('Triple click selects full line', () {
      const target = TextPosition(1, 5);
      final sel = MouseSelectionHandler.handleTripleClick(lines, target);

      expect(sel.base, equals(const TextPosition(1, 0)));
      expect(sel.extent, equals(const TextPosition(2, 0)));
    });

    test('Triple click on last line selects to end of last line', () {
      const target = TextPosition(2, 2);
      final sel = MouseSelectionHandler.handleTripleClick(lines, target);

      expect(sel.base, equals(const TextPosition(2, 0)));
      expect(sel.extent, equals(const TextPosition(2, 10)));
    });
  });

  group('LineWrappingEngine & VirtualLayoutCalculator Tests', () {
    test('LineWrappingEngine splits text on word boundaries', () {
      const engine = LineWrappingEngine(maxColumns: 8, enabled: true);
      final slices = engine.wrapLine('hello world test', 0);

      // "hello " (6), "world " (6), "test" (4) -> 3 slices
      expect(slices.length, equals(3));
      expect(slices[0].content, equals('hello '));
      expect(slices[1].content, equals('world '));
      expect(slices[2].content, equals('test'));
    });

    test('LineWrappingEngine hard-wraps when no space boundary exists', () {
      const engine = LineWrappingEngine(maxColumns: 5, enabled: true);
      final slices = engine.wrapLine('abcdefghij', 0);

      expect(slices.length, equals(2));
      expect(slices[0].content, equals('abcde'));
      expect(slices[1].content, equals('fghij'));
    });

    test('VirtualLayoutCalculator computes row offsets and mappings', () {
      final docLines = [
        'short line',
        'a very long line that will wrap into multiple slices',
      ];

      const engine = LineWrappingEngine(maxColumns: 15, enabled: true);
      final calc = VirtualLayoutCalculator(lineWrappingEngine: engine);
      calc.computeLayout(docLines);

      expect(calc.totalVirtualRows, greaterThan(2));

      // Test position mappings
      const docPos = TextPosition(0, 3);
      final vPos = calc.documentToVirtualPosition(docPos);
      expect(vPos.virtualRow, equals(0));
      expect(vPos.virtualColumn, equals(3));

      final backDocPos = calc.virtualToDocumentPosition(vPos);
      expect(backDocPos, equals(docPos));
    });
  });

  group('GutterManager Tests', () {
    test('Calculates digits count and dynamic gutter width', () {
      const manager = GutterManager(
        charWidth: 10.0,
        paddingLeft: 10.0,
        paddingRight: 10.0,
      );

      expect(manager.getDigitsCount(5), equals(2));
      expect(manager.getDigitsCount(99), equals(2));
      expect(manager.getDigitsCount(100), equals(3));

      // 3 digits * 10 + 10 + 10 = 50
      expect(manager.calculateGutterWidth(100), equals(50.0));
    });

    test(
      'Generates correct GutterLineInfo for primary vs continuation slice',
      () {
        const manager = GutterManager();
        const firstRow = VirtualRowInfo(
          virtualRowIndex: 0,
          lineIndex: 0,
          sliceIndex: 0,
          slice: TextLineSlice(
            lineIndex: 0,
            sliceIndex: 0,
            startColumn: 0,
            endColumn: 10,
            content: 'hello ',
          ),
        );
        const contRow = VirtualRowInfo(
          virtualRowIndex: 1,
          lineIndex: 0,
          sliceIndex: 1,
          slice: TextLineSlice(
            lineIndex: 0,
            sliceIndex: 1,
            startColumn: 10,
            endColumn: 15,
            content: 'world',
          ),
        );

        final firstInfo = manager.getGutterLineInfo(firstRow);
        expect(firstInfo.isDisplayed, isTrue);
        expect(firstInfo.lineNumberText, equals('1'));

        final contInfo = manager.getGutterLineInfo(contRow);
        expect(contInfo.isDisplayed, isFalse);
        expect(contInfo.lineNumberText, equals(''));
      },
    );
  });
}
