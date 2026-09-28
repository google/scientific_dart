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

import 'package:code_editor/backends/terminal.dart';
import 'package:test/test.dart';

void main() {
  group('Vt100Encoder Tests', () {
    test('trueColorFg formats 24-bit TrueColor escape sequence', () {
      final fg = Vt100Encoder.trueColorFg(255, 128, 64);
      expect(fg, equals('\x1b[38;2;255;128;64m'));
    });

    test('trueColorBg formats 24-bit TrueColor escape sequence', () {
      final bg = Vt100Encoder.trueColorBg(30, 30, 30);
      expect(bg, equals('\x1b[48;2;30;30;30m'));
    });

    test('hexToFg converts hex string to TrueColor escape sequence', () {
      final fgHex = Vt100Encoder.hexToFg('#FF8000');
      expect(fgHex, equals('\x1b[38;2;255;128;0m'));
    });

    test('hexToBg converts hex string to TrueColor escape sequence', () {
      final bgHex = Vt100Encoder.hexToBg('#1E1E1E');
      expect(bgHex, equals('\x1b[48;2;30;30;30m'));
    });

    test('parseHexColor handles 3-digit and 6-digit hex values', () {
      expect(Vt100Encoder.parseHexColor('#FFF'), equals((255, 255, 255)));
      expect(Vt100Encoder.parseHexColor('569CD6'), equals((86, 156, 214)));
    });

    test('moveCursor formats 1-indexed VT100 position sequence', () {
      expect(Vt100Encoder.moveCursor(10, 5), equals('\x1b[10;5H'));
      expect(Vt100Encoder.moveCursor(0, 0), equals('\x1b[1;1H'));
    });

    test('control sequences output correct ANSI escape codes', () {
      expect(Vt100Encoder.clearScreen, equals('\x1b[2J\x1b[H'));
      expect(Vt100Encoder.hideCursor, equals('\x1b[?25l'));
      expect(Vt100Encoder.showCursor, equals('\x1b[?25h'));
      expect(Vt100Encoder.reset, equals('\x1b[0m'));
      expect(Vt100Encoder.resetBold, equals('\x1b[22m'));
      expect(Vt100Encoder.resetItalic, equals('\x1b[23m'));
      expect(Vt100Encoder.resetUnderline, equals('\x1b[24m'));
    });
  });
}
