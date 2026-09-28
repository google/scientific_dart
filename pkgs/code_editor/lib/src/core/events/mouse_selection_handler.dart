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

import '../selection/selection_model.dart';

/// Handler for mouse selection events (click, drag, double-click, triple-click).
final class MouseSelectionHandler {
  /// Handles single click event to position cursor at [targetPosition].
  static TextSelection handleSingleClick(TextPosition targetPosition) {
    return TextSelection.collapsed(targetPosition);
  }

  /// Handles mouse drag update event, keeping [currentSelection.base] fixed and setting extent to [targetPosition].
  static TextSelection handleDragUpdate(
    TextSelection currentSelection,
    TextPosition targetPosition,
  ) {
    return currentSelection.copyWith(extent: targetPosition);
  }

  /// Handles double click event, selecting the word surrounding [targetPosition].
  static TextSelection handleDoubleClick(
    List<String> lines,
    TextPosition targetPosition,
  ) {
    return SelectionModel.getWordBoundary(lines, targetPosition);
  }

  /// Handles triple click event, selecting the entire line surrounding [targetPosition].
  static TextSelection handleTripleClick(
    List<String> lines,
    TextPosition targetPosition,
  ) {
    if (lines.isEmpty) {
      return TextSelection.collapsed(const TextPosition(0, 0));
    }

    final lineIndex = targetPosition.line.clamp(0, lines.length - 1);
    final isLastLine = lineIndex == lines.length - 1;

    final basePos = TextPosition(lineIndex, 0);
    final extentPos = isLastLine
        ? TextPosition(lineIndex, lines[lineIndex].length)
        : TextPosition(lineIndex + 1, 0);

    return TextSelection(base: basePos, extent: extentPos);
  }
}
