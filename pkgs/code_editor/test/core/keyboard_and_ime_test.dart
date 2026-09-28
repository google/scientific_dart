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
  group('SelectionModel Tests', () {
    test('TextPosition comparison and equality', () {
      const p1 = TextPosition(0, 5);
      const p2 = TextPosition(0, 5);
      const p3 = TextPosition(0, 10);
      const p4 = TextPosition(1, 0);

      expect(p1, equals(p2));
      expect(p1 < p3, isTrue);
      expect(p3 < p4, isTrue);
      expect(p4 > p1, isTrue);
    });

    test('TextSelection properties', () {
      const base = TextPosition(1, 10);
      const extent = TextPosition(0, 5);
      const selection = TextSelection(base: base, extent: extent);

      expect(selection.isCollapsed, isFalse);
      expect(selection.isReversed, isTrue);
      expect(selection.start, equals(const TextPosition(0, 5)));
      expect(selection.end, equals(const TextPosition(1, 10)));
    });

    test('Word boundary calculation', () {
      final lines = ['hello world_test 1234'];
      const pos = TextPosition(0, 8); // inside "world_test"

      final wordSelection = SelectionModel.getWordBoundary(lines, pos);
      expect(wordSelection.start, equals(const TextPosition(0, 6)));
      expect(wordSelection.end, equals(const TextPosition(0, 16)));
    });

    test('Smart line start calculation', () {
      const lineText = '    final x = 42;';
      expect(
        SelectionModel.getSmartLineStart(lineText, const TextPosition(0, 10)),
        equals(const TextPosition(0, 4)),
      );
      expect(
        SelectionModel.getSmartLineStart(lineText, const TextPosition(0, 4)),
        equals(const TextPosition(0, 0)),
      );
    });

    test('Paragraph boundary calculation', () {
      final lines = ['Line 1', 'Line 2', '', 'Line 4', 'Line 5'];
      expect(
        SelectionModel.getParagraphStart(lines, const TextPosition(1, 2)),
        equals(const TextPosition(0, 0)),
      );
      expect(
        SelectionModel.getParagraphEnd(lines, const TextPosition(1, 2)),
        equals(const TextPosition(1, 6)),
      );
    });

    test('SelectionModel state operations', () {
      final model = SelectionModel();
      final lines = ['First line', 'Second line'];

      model.selectAll(lines);
      expect(model.primarySelection.start, equals(const TextPosition(0, 0)));
      expect(model.primarySelection.end, equals(const TextPosition(1, 11)));

      model.selectLineAt(lines, 0);
      expect(model.primarySelection.start, equals(const TextPosition(0, 0)));
      expect(model.primarySelection.end, equals(const TextPosition(0, 10)));
    });
  });

  group('KeyboardNavigationHandler Tests', () {
    final lines = ['    hello world', 'second line of text', 'third'];

    test('Arrow Left and Right navigation', () {
      const startSel = TextSelection(
        base: TextPosition(0, 5),
        extent: TextPosition(0, 5),
      );

      // Move left
      final leftSel = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.left,
        lines: lines,
      );
      expect(leftSel.start, equals(const TextPosition(0, 4)));

      // Move right
      final rightSel = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.right,
        lines: lines,
      );
      expect(rightSel.start, equals(const TextPosition(0, 6)));
    });

    test('Arrow Left across line boundaries', () {
      const lineStartSel = TextSelection(
        base: TextPosition(1, 0),
        extent: TextPosition(1, 0),
      );

      final navSel = KeyboardNavigationHandler.handleNavigation(
        currentSelection: lineStartSel,
        direction: NavigationDirection.left,
        lines: lines,
      );
      expect(navSel.start, equals(const TextPosition(0, 15)));
    });

    test('Shift + Arrow navigation for selection expansion', () {
      const startSel = TextSelection(
        base: TextPosition(0, 4),
        extent: TextPosition(0, 4),
      );

      final shiftRight = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.right,
        lines: lines,
        shift: true,
      );
      expect(shiftRight.base, equals(const TextPosition(0, 4)));
      expect(shiftRight.extent, equals(const TextPosition(0, 5)));
      expect(shiftRight.isCollapsed, isFalse);
    });

    test('Alt + Left/Right word jumping', () {
      const startSel = TextSelection(
        base: TextPosition(0, 9), // at space between "hello" and "world"
        extent: TextPosition(0, 9),
      );

      final altLeft = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.left,
        lines: lines,
        alt: true,
      );
      expect(
        altLeft.start,
        equals(const TextPosition(0, 4)),
      ); // start of "hello"

      final altRight = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.right,
        lines: lines,
        alt: true,
      );
      expect(
        altRight.start,
        equals(const TextPosition(0, 15)),
      ); // end of "world"
    });

    test('Cmd/Ctrl + Left/Right line boundaries', () {
      const startSel = TextSelection(
        base: TextPosition(0, 10),
        extent: TextPosition(0, 10),
      );

      final cmdLeft = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.left,
        lines: lines,
        ctrlOrCmd: true,
      );
      expect(
        cmdLeft.start,
        equals(const TextPosition(0, 4)),
      ); // smart indentation start

      final cmdRight = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.right,
        lines: lines,
        ctrlOrCmd: true,
      );
      expect(cmdRight.start, equals(const TextPosition(0, 15)));
    });

    test('Home and End keys', () {
      const startSel = TextSelection(
        base: TextPosition(0, 12),
        extent: TextPosition(0, 12),
      );

      final homeSel = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.home,
        lines: lines,
      );
      expect(homeSel.start, equals(const TextPosition(0, 4)));

      final endSel = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.end,
        lines: lines,
      );
      expect(endSel.start, equals(const TextPosition(0, 15)));
    });

    test('PageUp and PageDown navigation', () {
      const startSel = TextSelection(
        base: TextPosition(2, 2),
        extent: TextPosition(2, 2),
      );

      final pageUpSel = KeyboardNavigationHandler.handleNavigation(
        currentSelection: startSel,
        direction: NavigationDirection.pageUp,
        lines: lines,
        pageSize: 2,
      );
      expect(pageUpSel.start, equals(const TextPosition(0, 2)));

      final pageDownSel = KeyboardNavigationHandler.handleNavigation(
        currentSelection: const TextSelection(
          base: TextPosition(0, 2),
          extent: TextPosition(0, 2),
        ),
        direction: NavigationDirection.pageDown,
        lines: lines,
        pageSize: 2,
      );
      expect(pageDownSel.start, equals(const TextPosition(2, 2)));
    });
  });

  group('ImeInputHandler Tests', () {
    test('handleCommitText inserts text over collapsed selection', () {
      final lines = ['hello world'];
      const sel = TextSelection(
        base: TextPosition(0, 5),
        extent: TextPosition(0, 5),
      );

      final res = ImeInputHandler.handleCommitText(
        lines: lines,
        selection: sel,
        insertedText: ' brave',
      );

      expect(res.lines, equals(['hello brave world']));
      expect(res.selection.start, equals(const TextPosition(0, 11)));
    });

    test(
      'handleCommitText replaces non-collapsed selection with multiline text',
      () {
        final lines = ['hello world'];
        const sel = TextSelection(
          base: TextPosition(0, 5),
          extent: TextPosition(0, 11),
        );

        final res = ImeInputHandler.handleCommitText(
          lines: lines,
          selection: sel,
          insertedText: '\nbeautiful\nday',
        );

        expect(res.lines, equals(['hello', 'beautiful', 'day']));
        expect(res.selection.start, equals(const TextPosition(2, 3)));
      },
    );

    test('handleDeleteBackward removes char or merges lines', () {
      final lines = ['hello', 'world'];
      const selLineStart = TextSelection(
        base: TextPosition(1, 0),
        extent: TextPosition(1, 0),
      );

      final res = ImeInputHandler.handleDeleteBackward(
        lines: lines,
        selection: selLineStart,
      );

      expect(res.lines, equals(['helloworld']));
      expect(res.selection.start, equals(const TextPosition(0, 5)));
    });

    test('handleDeleteForward removes char or merges lines', () {
      final lines = ['hello', 'world'];
      const selLineEnd = TextSelection(
        base: TextPosition(0, 5),
        extent: TextPosition(0, 5),
      );

      final res = ImeInputHandler.handleDeleteForward(
        lines: lines,
        selection: selLineEnd,
      );

      expect(res.lines, equals(['helloworld']));
      expect(res.selection.start, equals(const TextPosition(0, 5)));
    });

    test('Offset and TextPosition conversions', () {
      final lines = ['abc', 'defg', 'hi'];

      expect(
        ImeInputHandler.positionToOffset(lines, const TextPosition(0, 2)),
        equals(2),
      );
      expect(
        ImeInputHandler.positionToOffset(lines, const TextPosition(1, 0)),
        equals(4),
      );
      expect(
        ImeInputHandler.positionToOffset(lines, const TextPosition(1, 4)),
        equals(8),
      );
      expect(
        ImeInputHandler.positionToOffset(lines, const TextPosition(2, 1)),
        equals(10),
      );

      expect(
        ImeInputHandler.offsetToPosition(lines, 2),
        equals(const TextPosition(0, 2)),
      );
      expect(
        ImeInputHandler.offsetToPosition(lines, 4),
        equals(const TextPosition(1, 0)),
      );
      expect(
        ImeInputHandler.offsetToPosition(lines, 10),
        equals(const TextPosition(2, 1)),
      );
    });

    test('syncToTextEditingValue produces valid TextEditingValue', () {
      final lines = ['hello', 'world'];
      const sel = TextSelection(
        base: TextPosition(0, 2),
        extent: TextPosition(1, 3),
      );

      final tev = ImeInputHandler.syncToTextEditingValue(
        lines: lines,
        selection: sel,
      );

      expect(tev.text, equals('hello\nworld'));
      expect(tev.selection, equals(const TextRange(start: 2, end: 9)));
    });
  });
}
