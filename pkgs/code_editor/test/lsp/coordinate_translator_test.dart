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
import 'package:test/test.dart';

void main() {
  group('LspCoordinateTranslator', () {
    late PieceTreeTextBuffer buffer;

    setUp(() {
      buffer = PieceTreeTextBuffer('first line\nsecond line\nthird line');
    });

    test('bidirectional position conversion', () {
      const editorPos = TextPosition(1, 4);
      final lspPos = LspCoordinateTranslator.toLspPosition(buffer, editorPos);
      expect(lspPos, equals(const LspPosition(1, 4)));

      final convertedBack = LspCoordinateTranslator.toTextPosition(
        buffer,
        lspPos,
      );
      expect(convertedBack, equals(editorPos));
    });

    test('range to selection conversion', () {
      const selection = Selection(TextPosition(0, 2), TextPosition(1, 6));
      final range = LspCoordinateTranslator.toLspRange(buffer, selection);

      expect(range.start, equals(const LspPosition(0, 2)));
      expect(range.end, equals(const LspPosition(1, 6)));

      final selectionBack = LspCoordinateTranslator.toSelection(buffer, range);
      expect(selectionBack, equals(selection));
    });

    test('rune vs UTF-16 code unit translation with surrogate pairs', () {
      final textWithEmoji = 'A😀B'; // '😀' is 2 UTF-16 code units, 1 rune
      // Rune offsets: 0 ('A'), 1 ('😀'), 2 ('B')
      // Code unit offsets: 0 ('A'), 1 ('😀' high), 3 ('B')

      expect(
        LspCoordinateTranslator.runeOffsetToCodeUnitOffset(textWithEmoji, 0),
        equals(0),
      );
      expect(
        LspCoordinateTranslator.runeOffsetToCodeUnitOffset(textWithEmoji, 1),
        equals(1),
      );
      expect(
        LspCoordinateTranslator.runeOffsetToCodeUnitOffset(textWithEmoji, 2),
        equals(3),
      );

      expect(
        LspCoordinateTranslator.codeUnitOffsetToRuneOffset(textWithEmoji, 0),
        equals(0),
      );
      expect(
        LspCoordinateTranslator.codeUnitOffsetToRuneOffset(textWithEmoji, 1),
        equals(1),
      );
      expect(
        LspCoordinateTranslator.codeUnitOffsetToRuneOffset(textWithEmoji, 3),
        equals(2),
      );
    });
  });
}
