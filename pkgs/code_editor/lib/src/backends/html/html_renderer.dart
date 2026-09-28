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

import '../../render/editor_renderer.dart';
import '../../render/render_viewport.dart';
import 'dom_node_pool.dart';

/// Web HTML/DOM virtual list renderer with node recycling, XSS prevention,
/// selection rects, and blinking caret overlays.
class HtmlRenderer implements EditorRenderer {
  final DomNodePool nodePool;
  final List<RenderEventListener> _listeners = [];
  bool _isAttached = false;
  Object? _container;
  RenderViewport? _lastViewport;

  HtmlRenderer({DomNodePool? nodePool})
    : nodePool = nodePool ?? DomNodePool(extraBuffer: 10);

  @override
  bool get isAttached => _isAttached;

  /// Attached target container.
  Object? get container => _container;

  @override
  RenderViewport? get lastViewport => _lastViewport;

  @override
  void attach(Object target) {
    _container = target;
    _isAttached = true;
  }

  @override
  void detach() {
    _container = null;
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

  final List<DomElementNode> _renderedLineNodes = [];

  /// Acquired DOM line elements from the last render pass.
  List<DomElementNode> get renderedLineNodes => _renderedLineNodes;

  @override
  void render(RenderViewport viewport) {
    _lastViewport = viewport;
    _renderedLineNodes.clear();
    final visibleLineCount = viewport.lines.length;
    nodePool.preparePool(visibleLineCount);
    nodePool.recycleAll();

    for (final line in viewport.lines) {
      final lineNode = nodePool.acquireLineNode();
      lineNode.style['position'] = 'absolute';
      lineNode.style['top'] = '${line.top}px';
      lineNode.style['height'] = '${line.height}px';
      lineNode.style['white-space'] = 'pre';

      if (line.tokens.isEmpty) {
        lineNode.textContent = line.text;
      } else {
        for (final token in line.tokens) {
          final spanNode = nodePool.acquireSpanNode();
          spanNode.className = 'token';
          if (token.style.foreground != null) {
            spanNode.style['color'] = token.style.foreground!;
          }
          if (token.style.background != null) {
            spanNode.style['background-color'] = token.style.background!;
          }
          if (token.style.bold) {
            spanNode.style['font-weight'] = 'bold';
          }
          if (token.style.italic) {
            spanNode.style['font-style'] = 'italic';
          }
          // Strict textContent assignment avoids XSS injection
          spanNode.textContent = token.text;
          lineNode.children.add(spanNode);
        }
      }
      _renderedLineNodes.add(lineNode);
    }
  }

  /// Generate an HTML DOM tree string representation of the current viewport.
  /// Useful for web rendering, SSR, and automated testing.
  String renderToHtmlString(RenderViewport viewport) {
    render(viewport);

    final rootNode = DomElementNode('div');
    rootNode.className = 'code-editor-viewport';
    rootNode.style['position'] = 'relative';
    rootNode.style['width'] = '${viewport.width}px';
    rootNode.style['height'] = '${viewport.height}px';
    rootNode.style['overflow'] = 'hidden';
    rootNode.style['font-family'] = 'monospace';

    // 1. Render Gutter Container
    final gutterNode = DomElementNode('div');
    gutterNode.className = 'editor-gutter';
    gutterNode.style['position'] = 'absolute';
    gutterNode.style['left'] = '0px';
    gutterNode.style['top'] = '0px';
    gutterNode.style['width'] = '${viewport.gutter.width}px';
    gutterNode.style['height'] = '100%';
    gutterNode.style['user-select'] = 'none';
    gutterNode.style['background-color'] = '#181825';
    gutterNode.style['border-right'] = '1px solid #313244';
    gutterNode.style['color'] = '#6c7086';
    gutterNode.style['text-align'] = 'right';
    gutterNode.style['padding-right'] = '8px';
    gutterNode.style['box-sizing'] = 'border-box';

    for (final item in viewport.gutter.items) {
      final itemNode = DomElementNode('div');
      itemNode.className = 'gutter-item';
      itemNode.style['height'] = '${viewport.lineHeight}px';
      itemNode.textContent = item.lineNumberText;
      gutterNode.children.add(itemNode);
    }
    rootNode.children.add(gutterNode);

    // 2. Render Lines Layer
    final linesContainer = DomElementNode('div');
    linesContainer.className = 'editor-lines-layer';
    linesContainer.style['position'] = 'absolute';
    linesContainer.style['left'] = '${viewport.gutter.width + 8}px';
    linesContainer.style['top'] = '0px';
    linesContainer.style['width'] =
        '${viewport.width - viewport.gutter.width}px';

    linesContainer.children.addAll(_renderedLineNodes);
    rootNode.children.add(linesContainer);

    // 3. Render Selection Overlays
    final selectionsContainer = DomElementNode('div');
    selectionsContainer.className = 'editor-selections-layer';
    selectionsContainer.style['position'] = 'absolute';
    selectionsContainer.style['pointer-events'] = 'none';

    for (final rect in viewport.selections) {
      final selNode = DomElementNode('div');
      selNode.className = 'selection-rect';
      selNode.style['position'] = 'absolute';
      selNode.style['left'] = '${rect.left}px';
      selNode.style['top'] = '${rect.top}px';
      selNode.style['width'] = '${rect.width}px';
      selNode.style['height'] = '${rect.height}px';
      selNode.style['background-color'] = 'rgba(0, 120, 215, 0.4)';
      selectionsContainer.children.add(selNode);
    }
    rootNode.children.add(selectionsContainer);

    // 4. Render Carets Overlay
    final caretsContainer = DomElementNode('div');
    caretsContainer.className = 'editor-carets-layer';
    caretsContainer.style['position'] = 'absolute';

    for (final caret in viewport.carets) {
      final caretNode = DomElementNode('div');
      caretNode.className = 'editor-caret';
      caretNode.style['position'] = 'absolute';
      caretNode.style['left'] = '${caret.x}px';
      caretNode.style['top'] = '${caret.y}px';
      caretNode.style['width'] = '2px';
      caretNode.style['height'] = '${caret.height}px';
      caretNode.style['background-color'] = caret.isPrimary
          ? '#ffffff'
          : '#aaaaaa';
      caretNode.style['animation'] = 'editor-blink 1s step-end infinite';
      caretsContainer.children.add(caretNode);
    }
    rootNode.children.add(caretsContainer);

    return rootNode.toHtmlString();
  }

  /// Dispatch an event to registered listeners.
  void notifyEvent(String eventType, Map<String, dynamic> data) {
    for (final listener in List<RenderEventListener>.from(_listeners)) {
      listener(eventType, data);
    }
  }

  @override
  void dispose() {
    detach();
    nodePool.clear();
    _listeners.clear();
  }
}
