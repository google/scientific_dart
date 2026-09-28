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
  group('LineOperations Tests', () {
    const ops = LineOperations();

    test('duplicates lines down correctly', () {
      final buffer = PieceTreeTextBuffer('line 1\nline 2\nline 3');
      final res = ops.duplicateLinesDown(
        buffer: buffer,
        selection: const TextSelection(
          base: TextPosition(1, 0),
          extent: TextPosition(1, 6),
        ),
      );
      expect(res.newText, 'line 2\n');
      expect(res.newSelection.base.line, 2);
    });

    test('moves lines up and down correctly', () {
      final buffer = PieceTreeTextBuffer('line 1\nline 2\nline 3');
      // Move line 2 up
      final upRes = ops.moveLinesUp(
        buffer: buffer,
        selection: const TextSelection.collapsed(TextPosition(1, 2)),
      );
      expect(upRes, isNotNull);
      expect(upRes!.newText, 'line 2\nline 1\n');
      expect(upRes.newSelection.extent.line, 0);

      // Top line moving up returns null
      final topRes = ops.moveLinesUp(
        buffer: buffer,
        selection: const TextSelection.collapsed(TextPosition(0, 0)),
      );
      expect(topRes, isNull);
    });

    test('deletes full line correctly', () {
      final buffer = PieceTreeTextBuffer('line 1\nline 2\nline 3');
      final res = ops.deleteLines(
        buffer: buffer,
        selection: const TextSelection.collapsed(TextPosition(1, 3)),
      );
      expect(res.replaceLength, 'line 2\n'.length);
      expect(res.newText, '');
    });

    test('toggles line comments //', () {
      final buffer = PieceTreeTextBuffer('  var a = 1;\n  var b = 2;');
      final res = ops.toggleLineComment(
        buffer: buffer,
        selection: const TextSelection(
          base: TextPosition(0, 0),
          extent: TextPosition(1, 10),
        ),
      );
      expect(res.newText, '  // var a = 1;\n  // var b = 2;');
    });
  });
}
