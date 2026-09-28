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
  group('FindReplaceController Tests', () {
    test('finds plain text matches case-insensitively and navigates', () {
      final buffer = PieceTreeTextBuffer('hello world Hello Dart hello');
      final controller = FindReplaceController();

      controller.find(
        buffer,
        const SearchOptions(query: 'hello', matchCase: false),
      );
      expect(controller.matchCount, 3);
      expect(controller.activeMatchIndex, 0);

      final next = controller.findNext();
      expect(next, isNotNull);
      expect(controller.activeMatchIndex, 1);
      expect(next!.text, 'Hello');

      final prev = controller.findPrevious();
      expect(prev, isNotNull);
      expect(controller.activeMatchIndex, 0);
    });

    test('respects case-sensitivity and whole word search', () {
      final buffer = PieceTreeTextBuffer('foo fooBar FOO foo');
      final controller = FindReplaceController();

      controller.find(
        buffer,
        const SearchOptions(
          query: 'foo',
          matchCase: true,
          matchWholeWord: true,
        ),
      );
      expect(controller.matchCount, 2);
    });

    test('finds regex patterns correctly', () {
      final buffer = PieceTreeTextBuffer('val1: 100, val2: 200, val3: 300');
      final controller = FindReplaceController();

      controller.find(
        buffer,
        const SearchOptions(query: r'val\d+:\s+\d+', isRegex: true),
      );
      expect(controller.matchCount, 3);
    });
  });
}
