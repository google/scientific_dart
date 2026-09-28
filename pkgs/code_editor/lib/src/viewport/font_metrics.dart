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

import 'package:characters/characters.dart';

class FontMetrics {
  final double fontSize;
  final double lineHeight;
  final double characterWidth;
  final double ascent;
  final double descent;
  final int tabSize;

  const FontMetrics({
    this.fontSize = 14.0,
    this.lineHeight = 20.0,
    this.characterWidth = 8.4,
    this.ascent = 11.2,
    this.descent = 2.8,
    this.tabSize = 4,
  });

  FontMetrics copyWith({
    double? fontSize,
    double? lineHeight,
    double? characterWidth,
    double? ascent,
    double? descent,
    int? tabSize,
  }) {
    return FontMetrics(
      fontSize: fontSize ?? this.fontSize,
      lineHeight: lineHeight ?? this.lineHeight,
      characterWidth: characterWidth ?? this.characterWidth,
      ascent: ascent ?? this.ascent,
      descent: descent ?? this.descent,
      tabSize: tabSize ?? this.tabSize,
    );
  }
}

class TextMeasurer {
  final FontMetrics metrics;

  const TextMeasurer(this.metrics);

  /// Calculates visual character count accounting for tab expansions.
  int visualLength(String text) {
    int len = 0;
    for (final char in text.characters) {
      if (char == '\t') {
        final remainder = len % metrics.tabSize;
        len += (metrics.tabSize - remainder);
      } else {
        len++;
      }
    }
    return len;
  }

  /// Measures pixel width of string.
  double measureWidth(String text) {
    return visualLength(text) * metrics.characterWidth;
  }

  /// Calculates pixel X coordinate of a character offset in a line.
  double offsetToX(String text, int offset) {
    if (offset <= 0) return 0.0;
    if (offset > text.length) offset = text.length;
    final substring = text.substring(0, offset);
    return measureWidth(substring);
  }

  /// Calculates character index closest to a pixel X offset in a line.
  int xToOffset(String text, double x) {
    if (x <= 0.0) return 0;
    double currentX = 0.0;
    int charIdx = 0;
    int colIdx = 0;

    for (final char in text.characters) {
      final charWidth = (char == '\t')
          ? (metrics.tabSize - (colIdx % metrics.tabSize)) *
                metrics.characterWidth
          : metrics.characterWidth;

      if (currentX + charWidth / 2.0 >= x) {
        return charIdx;
      }
      currentX += charWidth;
      charIdx += char.length;
      colIdx += (char == '\t')
          ? (metrics.tabSize - (colIdx % metrics.tabSize))
          : 1;
    }

    return text.length;
  }
}
