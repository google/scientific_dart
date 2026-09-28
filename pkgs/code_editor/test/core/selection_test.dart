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
import 'package:test/test.dart';

void main() {
  group('Selection & SelectionModel', () {
    test('selection reversed and bounds properties', () {
      const forward = TextSelection(
        base: TextPosition(0, 2),
        extent: TextPosition(0, 5),
      );
      expect(forward.isReversed, isFalse);
      expect(forward.start, equals(const TextPosition(0, 2)));
      expect(forward.end, equals(const TextPosition(0, 5)));

      const reversed = TextSelection(
        base: TextPosition(0, 5),
        extent: TextPosition(0, 2),
      );
      expect(reversed.isReversed, isTrue);
      expect(reversed.start, equals(const TextPosition(0, 2)));
      expect(reversed.end, equals(const TextPosition(0, 5)));
    });

    test('multi-cursor selections', () {
      final model = SelectionModel(
        primarySelection: const TextSelection(
          base: TextPosition(0, 0),
          extent: TextPosition(0, 8),
        ),
        secondarySelections: const [
          TextSelection(base: TextPosition(1, 0), extent: TextPosition(1, 4)),
        ],
      );

      expect(model.selections.length, equals(2));
      expect(
        model.selections.first,
        equals(
          const TextSelection(
            base: TextPosition(0, 0),
            extent: TextPosition(0, 8),
          ),
        ),
      );
      expect(
        model.selections.last,
        equals(
          const TextSelection(
            base: TextPosition(1, 0),
            extent: TextPosition(1, 4),
          ),
        ),
      );
    });

    test('word selection bounds helper', () {
      final lines = ['hello_world foo_bar 123'];
      final model = SelectionModel();

      model.selectWordAt(lines, const TextPosition(0, 3));
      expect(
        model.primarySelection,
        equals(
          const TextSelection(
            base: TextPosition(0, 0),
            extent: TextPosition(0, 11),
          ),
        ),
      ); // hello_world

      model.selectWordAt(lines, const TextPosition(0, 14));
      expect(
        model.primarySelection,
        equals(
          const TextSelection(
            base: TextPosition(0, 12),
            extent: TextPosition(0, 19),
          ),
        ),
      ); // foo_bar
    });

    test('selection shift on insert', () {
      final model = SelectionModel();
      model.setSingleCursor(const TextPosition(0, 6)); // At 'W'

      // Transform selection before/during insert of 10 characters at line 0, column 6
      model.transformOnInsert(0, 6, 10);

      expect(model.primary.position, equals(const TextPosition(0, 16)));
    });

    test('selection shift on delete', () {
      final model = SelectionModel();
      model.setSingleCursor(const TextPosition(0, 16)); // At 'W'

      // Transform selection before delete of 10 characters at line 0, column 6
      model.transformOnDelete(0, 6, 10);

      expect(model.primary.position, equals(const TextPosition(0, 6)));
    });
  });
}
