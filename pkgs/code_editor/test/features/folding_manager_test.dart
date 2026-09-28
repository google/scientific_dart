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
  group('FoldingManager Tests', () {
    test('scanCodeBlocks detects brace blocks and toggles fold state', () {
      final codeLines = [
        'void main() {',
        '  print(1);',
        '  print(2);',
        '}',
        'void other() {',
        '  print(3);',
        '}',
      ];

      final folding = FoldingManager();
      folding.scanCodeBlocks(codeLines);

      expect(folding.regions.length, 2);
      expect(folding.isFoldHeader(0), isTrue);
      expect(folding.isFoldHeader(4), isTrue);
      expect(folding.isLineHidden(1), isFalse);

      // Collapse region at line 0
      folding.toggleFold(0);
      expect(folding.getRegionAt(0)?.isCollapsed, isTrue);
      expect(folding.isLineHidden(1), isTrue);
      expect(folding.isLineHidden(2), isTrue);
      expect(folding.isLineHidden(3), isFalse); // closing brace line is visible
      expect(
        folding.getVisibleLineCount(codeLines.length),
        5,
      ); // 7 - 2 hidden lines

      // Toggle back to expand
      folding.toggleFold(0);
      expect(folding.getRegionAt(0)?.isCollapsed, isFalse);
      expect(folding.isLineHidden(1), isFalse);
      expect(folding.getVisibleLineCount(codeLines.length), 7);
    });
  });
}
