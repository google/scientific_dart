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

import 'package:meta/meta.dart';

import '../lsp/completion_popup_model.dart';
import '../lsp/hover_tooltip_model.dart';
import '../lsp/lsp_diagnostic_adapter.dart';

/// Style properties for a rendered token.
@immutable
class RenderTokenStyle {
  final String? foreground;
  final String? background;
  final bool bold;
  final bool italic;
  final bool underline;

  const RenderTokenStyle({
    this.foreground,
    this.background,
    this.bold = false,
    this.italic = false,
    this.underline = false,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is RenderTokenStyle &&
          runtimeType == other.runtimeType &&
          foreground == other.foreground &&
          background == other.background &&
          bold == other.bold &&
          italic == other.italic &&
          underline == other.underline;

  @override
  int get hashCode =>
      Object.hash(foreground, background, bold, italic, underline);
}

/// A token representing a styled segment of text within a line.
@immutable
class RenderToken {
  final String text;
  final int startColumn;
  final int endColumn;
  final RenderTokenStyle style;
  final String? scope;

  const RenderToken({
    required this.text,
    required this.startColumn,
    required this.endColumn,
    this.style = const RenderTokenStyle(),
    this.scope,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is RenderToken &&
          runtimeType == other.runtimeType &&
          text == other.text &&
          startColumn == other.startColumn &&
          endColumn == other.endColumn &&
          style == other.style &&
          scope == other.scope;

  @override
  int get hashCode => Object.hash(text, startColumn, endColumn, style, scope);
}

/// Visual fold indicator states in the gutter.
enum FoldIndicatorState { none, expanded, collapsed }

/// Diagnostic level indicators in the gutter.
enum GutterDiagnosticSeverity { none, hint, info, warning, error }

/// Git diff indicators in the gutter.
enum GitDiffStatus { none, added, modified, deleted }

/// An item in the gutter for a specific line.
@immutable
class GutterItem {
  final int lineIndex;
  final String lineNumberText;
  final FoldIndicatorState foldState;
  final GutterDiagnosticSeverity diagnosticSeverity;
  final GitDiffStatus gitDiffStatus;

  const GutterItem({
    required this.lineIndex,
    required this.lineNumberText,
    this.foldState = FoldIndicatorState.none,
    this.diagnosticSeverity = GutterDiagnosticSeverity.none,
    this.gitDiffStatus = GitDiffStatus.none,
  });
}

/// Metadata model for the editor gutter.
@immutable
class RenderGutter {
  final double width;
  final List<GutterItem> items;

  const RenderGutter({required this.width, required this.items});
}

/// Render model for a single line of text in the viewport.
@immutable
class RenderLine {
  final int lineIndex;
  final String text;
  final List<RenderToken> tokens;
  final double top;
  final double height;

  const RenderLine({
    required this.lineIndex,
    required this.text,
    required this.tokens,
    required this.top,
    required this.height,
  });
}

/// A visual selection rectangle in screen/canvas coordinates.
@immutable
class SelectionRect {
  final double left;
  final double top;
  final double width;
  final double height;

  const SelectionRect({
    required this.left,
    required this.top,
    required this.width,
    required this.height,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SelectionRect &&
          runtimeType == other.runtimeType &&
          left == other.left &&
          top == other.top &&
          width == other.width &&
          height == other.height;

  @override
  int get hashCode => Object.hash(left, top, width, height);
}

/// Caret cursor position payload.
@immutable
class CaretPosition {
  final double x;
  final double y;
  final double height;
  final bool isPrimary;
  final int line;
  final int column;

  const CaretPosition({
    required this.x,
    required this.y,
    required this.height,
    this.isPrimary = true,
    required this.line,
    required this.column,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is CaretPosition &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          height == other.height &&
          isPrimary == other.isPrimary &&
          line == other.line &&
          column == other.column;

  @override
  int get hashCode => Object.hash(x, y, height, isPrimary, line, column);
}

/// Complete immutable viewport snapshot passed to UI renderers.
@immutable
class RenderViewport {
  final double width;
  final double height;
  final double scrollX;
  final double scrollY;
  final int firstVisibleLine;
  final int lastVisibleLine;
  final List<RenderLine> lines;
  final List<SelectionRect> selections;
  final List<CaretPosition> carets;
  final RenderGutter gutter;
  final double lineHeight;
  final double charWidth;
  final List<DiagnosticSquiggle> squiggles;
  final CompletionPopupModel? completionPopup;
  final HoverTooltipModel? hoverTooltip;

  const RenderViewport({
    required this.width,
    required this.height,
    required this.scrollX,
    required this.scrollY,
    required this.firstVisibleLine,
    required this.lastVisibleLine,
    required this.lines,
    required this.selections,
    required this.carets,
    required this.gutter,
    required this.lineHeight,
    required this.charWidth,
    this.squiggles = const [],
    this.completionPopup,
    this.hoverTooltip,
  });
}
