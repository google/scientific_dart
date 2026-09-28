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
  group('AutoCloseEngine Tests', () {
    const engine = AutoCloseEngine();

    test(
      'returns InsertPairAction for opening bracket on collapsed cursor',
      () {
        final buffer = PieceTreeTextBuffer('final x = ');
        final action = engine.handleType(
          typedChar: '(',
          buffer: buffer,
          selection: const TextSelection.collapsed(TextPosition(0, 10)),
        );
        expect(action, isA<InsertPairAction>());
        final insert = action as InsertPairAction;
        expect(insert.open, '(');
        expect(insert.close, ')');
      },
    );

    test('returns WrapSelectionAction when text is selected', () {
      final buffer = PieceTreeTextBuffer('final x = 123;');
      final action = engine.handleType(
        typedChar: '"',
        buffer: buffer,
        selection: const TextSelection(
          base: TextPosition(0, 10),
          extent: TextPosition(0, 13),
        ),
      );
      expect(action, isA<WrapSelectionAction>());
      final wrap = action as WrapSelectionAction;
      expect(wrap.open, '"');
      expect(wrap.close, '"');
    });

    test(
      'returns SkipCloseAction when typing over existing closing bracket',
      () {
        final buffer = PieceTreeTextBuffer('final x = ()');
        final action = engine.handleType(
          typedChar: ')',
          buffer: buffer,
          selection: const TextSelection.collapsed(TextPosition(0, 11)),
        );
        expect(action, isA<SkipCloseAction>());
      },
    );

    test('deletes both opening and closing bracket on backspace', () {
      final buffer = PieceTreeTextBuffer('final x = ()');
      // Cursor between ( and ) at column 11
      final count = engine.checkBackspacePairDeletion(
        buffer: buffer,
        position: const TextPosition(0, 11),
      );
      expect(count, 2);
    });
  });
}
