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

/// VT100 / ANSI escape sequence generator for 24-bit TrueColor terminals.
class Vt100Encoder {
  static const String reset = '\x1b[0m';
  static const String bold = '\x1b[1m';
  static const String resetBold = '\x1b[22m';
  static const String dim = '\x1b[2m';
  static const String italic = '\x1b[3m';
  static const String resetItalic = '\x1b[23m';
  static const String underline = '\x1b[4m';
  static const String resetUnderline = '\x1b[24m';
  static const String clearScreen = '\x1b[2J\x1b[H';
  static const String clearLine = '\x1b[2K';
  static const String hideCursor = '\x1b[?25l';
  static const String showCursor = '\x1b[?25h';
  static const String enableMouseSgr = '\x1b[?1000h\x1b[?1006h';
  static const String disableMouseSgr = '\x1b[?1000l\x1b[?1006l';
  static const String enableAltScreen = '\x1b[?1049h';
  static const String disableAltScreen = '\x1b[?1049l';

  /// Generate 24-bit TrueColor foreground escape sequence (`\x1b[38;2;R;G;Bm`).
  static String trueColorFg(int r, int g, int b) {
    return '\x1b[38;2;$r;$g;${b}m';
  }

  /// Generate 24-bit TrueColor background escape sequence (`\x1b[48;2;R;G;Bm`).
  static String trueColorBg(int r, int g, int b) {
    return '\x1b[48;2;$r;$g;${b}m';
  }

  /// Parse hex color string `#RRGGBB` or `RRGGBB` into TrueColor foreground ANSI escape string.
  static String hexToFg(String hex) {
    final (r, g, b) = parseHexColor(hex);
    return trueColorFg(r, g, b);
  }

  /// Parse hex color string `#RRGGBB` or `RRGGBB` into TrueColor background ANSI escape string.
  static String hexToBg(String hex) {
    final (r, g, b) = parseHexColor(hex);
    return trueColorBg(r, g, b);
  }

  /// Helper to convert hex color to (R, G, B) triple.
  static (int r, int g, int b) parseHexColor(String hex) {
    var cleaned = hex.replaceAll('#', '').trim();
    if (cleaned.startsWith('rgba(') || cleaned.startsWith('rgb(')) {
      final numbers = RegExp(
        r'\d+',
      ).allMatches(cleaned).map((m) => int.parse(m.group(0)!)).toList();
      if (numbers.length >= 3) {
        return (numbers[0], numbers[1], numbers[2]);
      }
    }
    if (cleaned.length == 3) {
      cleaned = cleaned.split('').map((c) => '$c$c').join();
    }
    if (cleaned.length != 6) {
      return (255, 255, 255);
    }
    final value = int.tryParse(cleaned, radix: 16) ?? 0xFFFFFF;
    final r = (value >> 16) & 0xFF;
    final g = (value >> 8) & 0xFF;
    final b = value & 0xFF;
    return (r, g, b);
  }

  /// Absolute cursor position sequence (`\x1b[<line>;<col>H`, 1-indexed).
  static String moveCursor(int line, int col) {
    final l = line < 1 ? 1 : line;
    final c = col < 1 ? 1 : col;
    return '\x1b[$l;${c}H';
  }
}
