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
import '../selection/selection_model.dart';

/// Direction/action for keyboard navigation events.
enum NavigationDirection {
  /// Move caret/selection left.
  left,

  /// Move caret/selection right.
  right,

  /// Move caret/selection up.
  up,

  /// Move caret/selection down.
  down,

  /// Move caret/selection to line start (or document start with ctrl/cmd).
  home,

  /// Move caret/selection to line end (or document end with ctrl/cmd).
  end,

  /// Move caret/selection up by one viewport page.
  pageUp,

  /// Move caret/selection down by one viewport page.
  pageDown,
}

/// Handler for editor keyboard navigation logic.
final class KeyboardNavigationHandler {
  /// Computes the new selection resulting from a keyboard navigation action.
  static TextSelection handleNavigation({
    required TextSelection currentSelection,
    required NavigationDirection direction,
    required List<String> lines,
    int pageSize = 20,
    bool shift = false,
    bool alt = false,
    bool ctrlOrCmd = false,
  }) {
    if (lines.isEmpty) {
      return TextSelection.collapsed(const TextPosition(0, 0));
    }

    final activePosition = currentSelection.extent;
    TextPosition targetPosition;

    switch (direction) {
      case NavigationDirection.left:
        if (!shift && !currentSelection.isCollapsed && !ctrlOrCmd && !alt) {
          targetPosition = currentSelection.start;
        } else if (ctrlOrCmd) {
          final lineText = _getSafeLine(lines, activePosition.line);
          targetPosition = SelectionModel.getSmartLineStart(
            lineText,
            activePosition,
          );
        } else if (alt) {
          final lineText = _getSafeLine(lines, activePosition.line);
          final col = SelectionModel.getWordStartColumn(
            lineText,
            activePosition.column,
          );
          targetPosition = TextPosition(activePosition.line, col);
        } else {
          targetPosition = _moveLeft(lines, activePosition);
        }
        break;

      case NavigationDirection.right:
        if (!shift && !currentSelection.isCollapsed && !ctrlOrCmd && !alt) {
          targetPosition = currentSelection.end;
        } else if (ctrlOrCmd) {
          final lineText = _getSafeLine(lines, activePosition.line);
          targetPosition = SelectionModel.getLineEnd(
            lineText,
            activePosition.line,
          );
        } else if (alt) {
          final lineText = _getSafeLine(lines, activePosition.line);
          final col = SelectionModel.getWordEndColumn(
            lineText,
            activePosition.column,
          );
          targetPosition = TextPosition(activePosition.line, col);
        } else {
          targetPosition = _moveRight(lines, activePosition);
        }
        break;

      case NavigationDirection.up:
        if (ctrlOrCmd) {
          targetPosition = const TextPosition(0, 0);
        } else {
          final targetLine = math.max(0, activePosition.line - 1);
          final lineText = _getSafeLine(lines, targetLine);
          final targetCol = activePosition.column.clamp(0, lineText.length);
          targetPosition = TextPosition(targetLine, targetCol);
        }
        break;

      case NavigationDirection.down:
        if (ctrlOrCmd) {
          final lastLine = lines.length - 1;
          targetPosition = TextPosition(lastLine, lines[lastLine].length);
        } else {
          final targetLine = math.min(
            lines.length - 1,
            activePosition.line + 1,
          );
          final lineText = _getSafeLine(lines, targetLine);
          final targetCol = activePosition.column.clamp(0, lineText.length);
          targetPosition = TextPosition(targetLine, targetCol);
        }
        break;

      case NavigationDirection.home:
        if (ctrlOrCmd) {
          targetPosition = const TextPosition(0, 0);
        } else {
          final lineText = _getSafeLine(lines, activePosition.line);
          targetPosition = SelectionModel.getSmartLineStart(
            lineText,
            activePosition,
          );
        }
        break;

      case NavigationDirection.end:
        if (ctrlOrCmd) {
          final lastLine = lines.length - 1;
          targetPosition = TextPosition(lastLine, lines[lastLine].length);
        } else {
          final lineText = _getSafeLine(lines, activePosition.line);
          targetPosition = TextPosition(activePosition.line, lineText.length);
        }
        break;

      case NavigationDirection.pageUp:
        targetPosition = SelectionModel.movePageUp(activePosition, pageSize);
        final lineText = _getSafeLine(lines, targetPosition.line);
        targetPosition = TextPosition(
          targetPosition.line,
          targetPosition.column.clamp(0, lineText.length),
        );
        break;

      case NavigationDirection.pageDown:
        targetPosition = SelectionModel.movePageDown(
          activePosition,
          lines.length,
          pageSize,
        );
        final lineText = _getSafeLine(lines, targetPosition.line);
        targetPosition = TextPosition(
          targetPosition.line,
          targetPosition.column.clamp(0, lineText.length),
        );
        break;
    }

    if (shift) {
      return currentSelection.copyWith(extent: targetPosition);
    } else {
      return TextSelection.collapsed(targetPosition);
    }
  }

  static TextPosition _moveLeft(List<String> lines, TextPosition position) {
    if (position.column > 0) {
      return TextPosition(position.line, position.column - 1);
    } else if (position.line > 0) {
      final prevLine = position.line - 1;
      return TextPosition(prevLine, lines[prevLine].length);
    }
    return position;
  }

  static TextPosition _moveRight(List<String> lines, TextPosition position) {
    final lineText = _getSafeLine(lines, position.line);
    if (position.column < lineText.length) {
      return TextPosition(position.line, position.column + 1);
    } else if (position.line < lines.length - 1) {
      return TextPosition(position.line + 1, 0);
    }
    return position;
  }

  static String _getSafeLine(List<String> lines, int lineIndex) {
    if (lines.isEmpty) return '';
    final safeIndex = lineIndex.clamp(0, lines.length - 1);
    return lines[safeIndex];
  }
}
