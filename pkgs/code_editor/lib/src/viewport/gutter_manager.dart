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

import 'dart:math' as math;
import 'virtual_layout_calculator.dart';

/// Metrics and info for rendering line numbers in the editor gutter.
final class GutterLineInfo {
  /// Virtual row index.
  final int virtualRowIndex;

  /// Document line index (0-based).
  final int lineIndex;

  /// 1-based display line number string (empty if wrapped continuation slice).
  final String lineNumberText;

  /// Whether this row displays a line number.
  final bool isDisplayed;

  /// Vertical Y pixel offset.
  final double yOffset;

  /// Creates a [GutterLineInfo].
  const GutterLineInfo({
    required this.virtualRowIndex,
    required this.lineIndex,
    required this.lineNumberText,
    required this.isDisplayed,
    required this.yOffset,
  });

  @override
  String toString() =>
      'GutterLineInfo(vRow: $virtualRowIndex, line: $lineIndex, text: "$lineNumberText", y: $yOffset)';
}

/// Manager responsible for gutter dimensions, dynamic line number padding, and layout alignment.
final class GutterManager {
  /// Font size for line numbers.
  final double fontSize;

  /// Width of a single character digit in pixels.
  final double charWidth;

  /// Height of a line in pixels.
  final double lineHeight;

  /// Left padding in pixels inside the gutter.
  final double paddingLeft;

  /// Right padding in pixels inside the gutter.
  final double paddingRight;

  /// Whether line numbers are enabled and visible.
  final bool showLineNumbers;

  /// Creates a [GutterManager].
  const GutterManager({
    this.fontSize = 13.0,
    this.charWidth = 8.0,
    this.lineHeight = 20.0,
    this.paddingLeft = 12.0,
    this.paddingRight = 12.0,
    this.showLineNumbers = true,
  });

  /// Calculates digits required to display [totalLines] (minimum 2 digits).
  int getDigitsCount(int totalLines) {
    if (totalLines <= 0) return 2;
    return math.max(2, totalLines.toString().length);
  }

  /// Calculates dynamic width of the gutter in pixels for [totalLines].
  double calculateGutterWidth(int totalLines) {
    if (!showLineNumbers) return 0.0;
    final digits = getDigitsCount(totalLines);
    return (digits * charWidth) + paddingLeft + paddingRight;
  }

  /// Calculates top Y pixel coordinate for virtual row [virtualRowIndex].
  double getLineYOffset(int virtualRowIndex) {
    return virtualRowIndex * lineHeight;
  }

  /// Generates gutter rendering info for [rowInfo].
  GutterLineInfo getGutterLineInfo(VirtualRowInfo rowInfo) {
    final yOffset = getLineYOffset(rowInfo.virtualRowIndex);
    if (!rowInfo.isFirstSlice || !showLineNumbers) {
      return GutterLineInfo(
        virtualRowIndex: rowInfo.virtualRowIndex,
        lineIndex: rowInfo.lineIndex,
        lineNumberText: '',
        isDisplayed: false,
        yOffset: yOffset,
      );
    }

    final displayNum = (rowInfo.lineIndex + 1).toString();
    return GutterLineInfo(
      virtualRowIndex: rowInfo.virtualRowIndex,
      lineIndex: rowInfo.lineIndex,
      lineNumberText: displayNum,
      isDisplayed: true,
      yOffset: yOffset,
    );
  }
}
