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
import '../../render/editor_renderer.dart';
import '../../render/render_viewport.dart';

/// Canvas draw command types captured by the Flutter renderer abstraction.
enum CanvasCommandType { drawRect, drawText, drawLine, fillRect, drawSquiggle }

/// Represents an abstract draw command for a Flutter Canvas or custom painter backend.
class CanvasDrawCommand {
  final CanvasCommandType type;
  final double x;
  final double y;
  final double width;
  final double height;
  final String? text;
  final String? color;
  final RenderTokenStyle? style;
  final int? selectedIndex;

  CanvasDrawCommand({
    required this.type,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    this.text,
    this.color,
    this.style,
    this.selectedIndex,
  });
}

/// Abstract Canvas interface simulating Flutter Canvas drawing operations.
abstract class EditorCanvas {
  void drawRect(double x, double y, double width, double height, String color);
  void drawText(double x, double y, String text, RenderTokenStyle style);
  void drawLine(
    double x1,
    double y1,
    double x2,
    double y2,
    String color,
    double strokeWidth,
  );
  void clear();
}

/// In-memory ParagraphBuilder / TextPainter layout cache for high-performance text rendering.
class ParagraphCache {
  final int maxCapacity;
  final Map<String, RenderLine> _cache = {};

  ParagraphCache({this.maxCapacity = 1000});

  int get size => _cache.length;

  static String buildKey(String text, RenderTokenStyle style) {
    return '$text|fg=${style.foreground}|bg=${style.background}|b=${style.bold}|i=${style.italic}|u=${style.underline}';
  }

  RenderLine? get(String text, RenderTokenStyle style) {
    final key = buildKey(text, style);
    return _cache[key];
  }

  void put(String text, RenderTokenStyle style, RenderLine line) {
    if (_cache.length >= maxCapacity) {
      // Simple LRU eviction: remove first key
      _cache.remove(_cache.keys.first);
    }
    final key = buildKey(text, style);
    _cache[key] = line;
  }

  void clear() {
    _cache.clear();
  }
}

/// Gesture adapter for translating Flutter touch/mouse input events to editor coordinates.
class FlutterGestureAdapter {
  final double charWidth;
  final double lineHeight;
  final double gutterWidth;

  FlutterGestureAdapter({
    required this.charWidth,
    required this.lineHeight,
    required this.gutterWidth,
  });

  /// Map canvas (x, y) coordinates to editor (line, column) positions.
  (int line, int column) positionFromOffset(
    double x,
    double y, {
    int firstVisibleLine = 0,
  }) {
    final effectiveX = (x - gutterWidth).clamp(0.0, double.infinity);
    final effectiveY = y.clamp(0.0, double.infinity);

    final lineIndex = (effectiveY / lineHeight).floor() + firstVisibleLine;
    final columnIndex = (effectiveX / charWidth).round();

    return (lineIndex, columnIndex);
  }
}

/// Flutter Canvas Painter & Backend Renderer implementing EditorRenderer.
class FlutterRenderer implements EditorRenderer {
  final ParagraphCache paragraphCache;
  final List<CanvasDrawCommand> recordedCommands = [];
  final List<RenderEventListener> _listeners = [];
  bool _isAttached = false;
  Object? _target;
  RenderViewport? _lastViewport;
  FlutterGestureAdapter? gestureAdapter;

  FlutterRenderer({ParagraphCache? cache})
    : paragraphCache = cache ?? ParagraphCache();

  @override
  bool get isAttached => _isAttached;

  /// Attached canvas target surface.
  Object? get target => _target;

  @override
  RenderViewport? get lastViewport => _lastViewport;

  @override
  void attach(Object target) {
    _target = target;
    _isAttached = true;
  }

  @override
  void detach() {
    _target = null;
    _isAttached = false;
  }

  @override
  void addEventListener(RenderEventListener listener) {
    _listeners.add(listener);
  }

  @override
  void removeEventListener(RenderEventListener listener) {
    _listeners.remove(listener);
  }

  @override
  void render(RenderViewport viewport) {
    _lastViewport = viewport;
    recordedCommands.clear();
    gestureAdapter = FlutterGestureAdapter(
      charWidth: viewport.charWidth,
      lineHeight: viewport.lineHeight,
      gutterWidth: viewport.gutter.width,
    );

    // 1. Draw Gutter background & items
    recordedCommands.add(
      CanvasDrawCommand(
        type: CanvasCommandType.fillRect,
        x: 0,
        y: 0,
        width: viewport.gutter.width,
        height: viewport.height,
        color: '#1e1e1e',
      ),
    );

    for (final item in viewport.gutter.items) {
      final y = item.lineIndex * viewport.lineHeight;
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.drawText,
          x: 8.0,
          y: y,
          width: viewport.gutter.width - 16.0,
          height: viewport.lineHeight,
          text: item.lineNumberText,
          color: '#858585',
        ),
      );
    }

    // 2. Draw Selection rects
    for (final rect in viewport.selections) {
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.fillRect,
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height,
          color: 'rgba(0, 120, 215, 0.4)',
        ),
      );
    }

    // 3. Draw Lines & Tokens
    for (final line in viewport.lines) {
      double currentX = viewport.gutter.width - viewport.scrollX;
      if (line.tokens.isEmpty) {
        final cached = paragraphCache.get(line.text, const RenderTokenStyle());
        if (cached == null) {
          paragraphCache.put(line.text, const RenderTokenStyle(), line);
        }
        recordedCommands.add(
          CanvasDrawCommand(
            type: CanvasCommandType.drawText,
            x: currentX,
            y: line.top,
            width: viewport.width - currentX,
            height: line.height,
            text: line.text,
            style: const RenderTokenStyle(),
          ),
        );
      } else {
        for (final token in line.tokens) {
          final cached = paragraphCache.get(token.text, token.style);
          if (cached == null) {
            paragraphCache.put(token.text, token.style, line);
          }
          final tokenWidth = token.text.length * viewport.charWidth;
          recordedCommands.add(
            CanvasDrawCommand(
              type: CanvasCommandType.drawText,
              x: currentX,
              y: line.top,
              width: tokenWidth,
              height: line.height,
              text: token.text,
              style: token.style,
            ),
          );
          currentX += tokenWidth;
        }
      }
    }

    // 4. Draw Diagnostic Squiggles
    for (final squiggle in viewport.squiggles) {
      final startX =
          viewport.gutter.width +
          (squiggle.startColumn * viewport.charWidth) -
          viewport.scrollX;
      final endX =
          viewport.gutter.width +
          (squiggle.endColumn * viewport.charWidth) -
          viewport.scrollX;
      final y =
          (squiggle.line * viewport.lineHeight) +
          viewport.lineHeight -
          2.0 -
          viewport.scrollY;
      final rawWidth = endX - startX;
      final width = rawWidth > 0 ? rawWidth : viewport.charWidth;
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.drawSquiggle,
          x: startX,
          y: y,
          width: width,
          height: 3.0,
          color: squiggle.colorHex,
          text: squiggle.message,
        ),
      );
    }

    // 5. Draw Carets
    for (final caret in viewport.carets) {
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.fillRect,
          x: caret.x,
          y: caret.y,
          width: 2.0,
          height: caret.height,
          color: caret.isPrimary ? '#ffffff' : '#aaaaaa',
        ),
      );
    }

    // 6. Draw Completion Dropdown Popup
    if (viewport.completionPopup != null &&
        viewport.completionPopup!.isVisible) {
      final popup = viewport.completionPopup!;
      final items = popup.filteredItems;
      const popupWidth = 220.0;
      final popupHeight = (items.length * viewport.lineHeight).clamp(
        viewport.lineHeight,
        300.0,
      );
      final posX = popup.x
          .clamp(0.0, math.max(0.0, viewport.width - popupWidth))
          .toDouble();
      final posY = popup.y
          .clamp(0.0, math.max(0.0, viewport.height - popupHeight))
          .toDouble();

      // Background Box
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.fillRect,
          x: posX,
          y: posY,
          width: popupWidth,
          height: popupHeight,
          color: '#252526',
        ),
      );

      // Border
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.drawRect,
          x: posX,
          y: posY,
          width: popupWidth,
          height: popupHeight,
          color: '#454545',
        ),
      );

      // Item list & selected item highlight
      for (var i = 0; i < items.length; i++) {
        final item = items[i];
        final itemY = posY + (i * viewport.lineHeight);
        if (itemY + viewport.lineHeight > posY + popupHeight) break;

        final isSelected = i == popup.selectedIndex;

        if (isSelected) {
          recordedCommands.add(
            CanvasDrawCommand(
              type: CanvasCommandType.fillRect,
              x: posX,
              y: itemY,
              width: popupWidth,
              height: viewport.lineHeight,
              color: '#04395E',
              selectedIndex: i,
            ),
          );
        }

        recordedCommands.add(
          CanvasDrawCommand(
            type: CanvasCommandType.drawText,
            x: posX + 8.0,
            y: itemY,
            width: popupWidth - 16.0,
            height: viewport.lineHeight,
            text: item.label,
            color: isSelected ? '#FFFFFF' : '#CCCCCC',
            selectedIndex: isSelected ? i : null,
          ),
        );
      }
    }

    // 7. Draw Hover Tooltip Popup
    if (viewport.hoverTooltip != null && viewport.hoverTooltip!.isVisible) {
      final tooltip = viewport.hoverTooltip!;
      final rawLines = tooltip.markdownContent.split('\n');
      final lines = [
        if (tooltip.signature != null && tooltip.signature!.isNotEmpty)
          tooltip.signature!,
        ...rawLines,
      ];

      const tooltipWidth = 260.0;
      final tooltipHeight = lines.length * viewport.lineHeight + 8.0;
      final posX = tooltip.x
          .clamp(0.0, math.max(0.0, viewport.width - tooltipWidth))
          .toDouble();
      final posY = tooltip.y
          .clamp(0.0, math.max(0.0, viewport.height - tooltipHeight))
          .toDouble();

      // Card Background Box
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.fillRect,
          x: posX,
          y: posY,
          width: tooltipWidth,
          height: tooltipHeight,
          color: '#252526',
        ),
      );

      // Border
      recordedCommands.add(
        CanvasDrawCommand(
          type: CanvasCommandType.drawRect,
          x: posX,
          y: posY,
          width: tooltipWidth,
          height: tooltipHeight,
          color: '#454545',
        ),
      );

      // Text / markdown content lines
      for (var i = 0; i < lines.length; i++) {
        final lineY = posY + 4.0 + (i * viewport.lineHeight);
        recordedCommands.add(
          CanvasDrawCommand(
            type: CanvasCommandType.drawText,
            x: posX + 8.0,
            y: lineY,
            width: tooltipWidth - 16.0,
            height: viewport.lineHeight,
            text: lines[i],
            color: (i == 0 && tooltip.signature != null)
                ? '#569CD6'
                : '#CCCCCC',
          ),
        );
      }
    }
  }

  /// Handle gesture tap at (x, y) canvas coordinates.
  void handleTapDown(double x, double y) {
    if (gestureAdapter != null && _lastViewport != null) {
      final (line, col) = gestureAdapter!.positionFromOffset(
        x,
        y,
        firstVisibleLine: _lastViewport!.firstVisibleLine,
      );
      for (final listener in List<RenderEventListener>.from(_listeners)) {
        listener('tap', {'line': line, 'column': col, 'x': x, 'y': y});
      }
    }
  }

  @override
  void dispose() {
    detach();
    paragraphCache.clear();
    recordedCommands.clear();
    _listeners.clear();
  }
}
