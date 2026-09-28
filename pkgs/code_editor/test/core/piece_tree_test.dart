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

import 'dart:math';
import 'package:code_editor/core.dart';
import 'package:test/test.dart';

void main() {
  group('PieceTreeTextBuffer', () {
    test('initialization with empty content', () {
      final buffer = PieceTreeTextBuffer('');
      expect(buffer.length, equals(0));
      expect(buffer.lineCount, equals(1));
      expect(buffer.text, equals(''));
    });

    test('initialization with single line text', () {
      final buffer = PieceTreeTextBuffer('Hello, World!');
      expect(buffer.length, equals(13));
      expect(buffer.lineCount, equals(1));
      expect(buffer.text, equals('Hello, World!'));
      expect(buffer.getLine(0), equals('Hello, World!'));
    });

    test('initialization with multi-line text', () {
      final buffer = PieceTreeTextBuffer('Line 1\nLine 2\nLine 3');
      expect(buffer.lineCount, equals(3));
      expect(buffer.getLine(0), equals('Line 1\n'));
      expect(buffer.getLine(1), equals('Line 2\n'));
      expect(buffer.getLine(2), equals('Line 3'));
      expect(buffer.getLineOffset(0), equals(0));
      expect(buffer.getLineOffset(1), equals(7));
      expect(buffer.getLineOffset(2), equals(14));
    });

    test('insertions at beginning, middle, and end', () {
      final buffer = PieceTreeTextBuffer('World');
      buffer.insert(0, 'Hello ');
      expect(buffer.text, equals('Hello World'));

      buffer.insert(11, '!');
      expect(buffer.text, equals('Hello World!'));

      buffer.insert(6, 'Beautiful ');
      expect(buffer.text, equals('Hello Beautiful World!'));
    });

    test('deletions from beginning, middle, and end', () {
      final buffer = PieceTreeTextBuffer('Hello Beautiful World!');
      buffer.delete(6, 10); // delete "Beautiful "
      expect(buffer.text, equals('Hello World!'));

      buffer.delete(0, 6); // delete "Hello "
      expect(buffer.text, equals('World!'));

      buffer.delete(5, 1); // delete "!"
      expect(buffer.text, equals('World'));
    });

    test('line and column coordinate lookups', () {
      final buffer = PieceTreeTextBuffer('abc\ndefg\nhijk');
      expect(buffer.getLineAndColumnAt(0), equals((0, 0)));
      expect(buffer.getLineAndColumnAt(3), equals((0, 3)));
      expect(buffer.getLineAndColumnAt(4), equals((1, 0)));
      expect(buffer.getLineAndColumnAt(8), equals((1, 4)));
      expect(buffer.getLineAndColumnAt(9), equals((2, 0)));

      expect(buffer.getOffsetAt(0, 0), equals(0));
      expect(buffer.getOffsetAt(1, 0), equals(4));
      expect(buffer.getOffsetAt(1, 2), equals(6));
      expect(buffer.getOffsetAt(2, 4), equals(13));
    });

    test('surrogate pair protection', () {
      final emoji = '😀'; // \u{1F600} -> 2 UTF-16 code units (0xD83D, 0xDE00)
      final buffer = PieceTreeTextBuffer('Hello ${emoji}World');

      // Attempt to split in the middle of surrogate pair at index 7
      buffer.insert(7, '[SPLIT]');
      // The offset should shift after the low surrogate to preserve emoji integrity
      expect(buffer.text.contains(emoji), isTrue);
    });

    test('fuzzy randomized stress test against reference String', () {
      final random = Random(42);
      var reference = 'Initial content line 1\nInitial content line 2\n';
      final buffer = PieceTreeTextBuffer(reference);

      final charPool = 'abcdefghijklmnopqrstuvwxyz0123456789 \n';

      for (var i = 0; i < 2000; i++) {
        final action = random.nextInt(2); // 0: insert, 1: delete

        if (action == 0 || reference.isEmpty) {
          final offset = random.nextInt(reference.length + 1);
          final len = random.nextInt(10) + 1;
          final sb = StringBuffer();
          for (var j = 0; j < len; j++) {
            sb.write(charPool[random.nextInt(charPool.length)]);
          }
          final textToInsert = sb.toString();

          reference =
              reference.substring(0, offset) +
              textToInsert +
              reference.substring(offset);
          buffer.insert(offset, textToInsert);
        } else {
          final offset = random.nextInt(reference.length);
          final maxLen = min(15, reference.length - offset);
          if (maxLen > 0) {
            final deleteLen = random.nextInt(maxLen) + 1;

            reference =
                reference.substring(0, offset) +
                reference.substring(offset + deleteLen);
            buffer.delete(offset, deleteLen);
          }
        }

        expect(
          buffer.text,
          equals(reference),
          reason: 'Mismatch at iteration $i',
        );
        expect(buffer.length, equals(reference.length));
      }
    });
  });
}
