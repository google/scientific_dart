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
  group('SnippetEngine & Controller Snippet Tests', () {
    test('expands built-in for loop snippet with tab stops', () {
      final engine = SnippetEngine();
      final snippet = engine.findSnippet('for');
      expect(snippet, isNotNull);

      final buffer = PieceTreeTextBuffer('');
      final res = engine.expandSnippet(
        body: snippet!.body,
        insertOffset: 0,
        buffer: buffer,
      );

      expect(res.insertedText, 'for (var  = 0;  < ; ++) {\n  \n}');
      expect(res.tabStops.length, 5); // $1, $1, $2, $1, $0
      expect(res.tabStops.first.index, 1);
      expect(res.tabStops.last.index, 0);
    });

    test('controller Tab expands snippet prefix at cursor', () {
      final controller = CodeEditorController(initialText: 'for');
      controller.selection = const TextSelection.collapsed(TextPosition(0, 3));

      controller.tabPressed();
      expect(controller.text.startsWith('for (var '), isTrue);
    });
  });
}
