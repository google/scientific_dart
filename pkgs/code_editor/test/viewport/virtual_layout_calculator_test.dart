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
import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('VirtualLayoutCalculator', () {
    test('computes line wrapping virtual rows and character mappings', () {
      const engine = LineWrappingEngine(maxColumns: 10, enabled: true);
      final calc = VirtualLayoutCalculator(lineWrappingEngine: engine);

      final lines = ['short', 'a very long line of text that wraps'];

      calc.computeLayout(lines);

      expect(calc.totalVirtualRows, greaterThan(2));

      // Test line 0 mapping
      const pos0 = TextPosition(0, 3);
      final vPos0 = calc.documentToVirtualPosition(pos0);
      expect(vPos0.virtualRow, equals(0));
      expect(vPos0.virtualColumn, equals(3));

      final docPos0 = calc.virtualToDocumentPosition(vPos0);
      expect(docPos0, equals(pos0));

      // Test line 1 wrapped slice mapping
      final rowInfo1 = calc.getVirtualRowInfo(1);
      expect(rowInfo1.lineIndex, equals(1));
    });
  });
}
