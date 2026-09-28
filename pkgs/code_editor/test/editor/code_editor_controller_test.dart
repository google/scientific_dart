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
  group('CodeEditorController Tests', () {
    test('initializes with initial text and tokenizes', () {
      final controller = CodeEditorController(initialText: 'final int x = 42;');
      expect(controller.text, 'final int x = 42;');
      expect(controller.lineCount, 1);
      expect(controller.lineTokens.isNotEmpty, isTrue);
      expect(
        controller.lineTokens.first.any((t) => t.type == TokenType.keyword),
        isTrue,
      );
    });

    test('inserts text and auto-closes brackets', () {
      final controller = CodeEditorController(initialText: 'var list = ');
      controller.selection = const TextSelection.collapsed(TextPosition(0, 11));

      // Type '('
      controller.insertText('(');
      expect(controller.text, 'var list = ()');
      expect(controller.selection.extent, const TextPosition(0, 12));

      // Type ')' (should step over without inserting duplicate)
      controller.insertText(')');
      expect(controller.text, 'var list = ()');
      expect(controller.selection.extent, const TextPosition(0, 13));
    });

    test('wraps selected text when typing opening quote or bracket', () {
      final controller = CodeEditorController(initialText: 'print(hello);');
      controller.selection = const TextSelection(
        base: TextPosition(0, 6),
        extent: TextPosition(0, 11),
      );

      controller.insertText("'");
      expect(controller.text, "print('hello');");
    });

    test('smart indent on Enter', () {
      final controller = CodeEditorController(initialText: 'void main() {');
      controller.selection = const TextSelection.collapsed(TextPosition(0, 13));

      controller.insertNewline();
      expect(controller.text, 'void main() {\n  ');
      expect(controller.selection.extent, const TextPosition(1, 2));
    });

    test('performs line operations: duplicate, move, comment', () {
      final controller = CodeEditorController(
        initialText: '  var a = 1;\n  var b = 2;',
      );
      controller.selection = const TextSelection.collapsed(TextPosition(0, 2));

      // Duplicate line down
      controller.duplicateLinesDown();
      expect(controller.lineCount, 3);

      // Toggle comment on line 1
      controller.toggleLineComment();
      expect(controller.buffer.getLine(1).contains('//'), isTrue);
    });

    test('finds and replaces text across document', () {
      final controller = CodeEditorController(
        initialText: 'cat and dog and cat',
      );
      controller.find(const SearchOptions(query: 'cat'));
      expect(controller.findReplace.matchCount, 2);

      controller.replaceAll('tiger');
      expect(controller.text, 'tiger and dog and tiger');
    });

    test('undoes and redoes multiple edit steps', () {
      final controller = CodeEditorController(initialText: 'hello');
      controller.selection = const TextSelection.collapsed(TextPosition(0, 5));

      controller.insertText(' world');
      expect(controller.text, 'hello world');

      final undone = controller.undo();
      expect(undone, isTrue);
      expect(controller.text, 'hello');

      final redone = controller.redo();
      expect(redone, isTrue);
      expect(controller.text, 'hello world');
    });

    test(
      'selectNextOccurrence (Ctrl+D) selects next occurrence and inserts text at all carets',
      () {
        final controller = CodeEditorController(
          initialText: 'var a = foo();\nvar b = foo();\n',
        );
        // Select first 'foo'
        controller.selection = const TextSelection(
          base: TextPosition(0, 8),
          extent: TextPosition(0, 11),
        );

        // Trigger Ctrl+D selectNextOccurrence
        controller.selectNextOccurrence();
        expect(controller.selectionModel.selections.length, 2);

        // Insert text across both occurrences
        controller.insertText('bar');
        expect(controller.text, 'var a = bar();\nvar b = bar();\n');
      },
    );

    test('formatDocument formats source code and supports undo', () {
      final controller = CodeEditorController(
        initialText: 'void test(){\nint x=1;\n}',
      );
      controller.formatDocument();
      expect(controller.text, 'void test() {\n  int x = 1;\n}');

      controller.undo();
      expect(controller.text, 'void test(){\nint x=1;\n}');
    });

    test('adds, sets, and clears diagnostic squiggles', () {
      final controller = CodeEditorController(initialText: 'var a = 1;');
      controller.addDiagnostic(
        const DiagnosticSquiggle(
          line: 0,
          startColumn: 0,
          endColumn: 3,
          severity: DiagnosticSeverity.error,
          message: 'Error message',
        ),
      );
      expect(controller.diagnostics.length, 1);
      expect(controller.diagnostics.first.color, SquiggleColor.red);

      controller.clearDiagnostics();
      expect(controller.diagnostics, isEmpty);
    });
  });
}
