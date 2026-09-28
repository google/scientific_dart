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

/// Virtualized DOM element model for cross-platform compatibility and Wasm optimization.
class DomElementNode {
  final String tagName;
  final Map<String, String> style = {};
  final List<DomElementNode> children = [];
  String textContent = '';
  String className = '';
  Object? nativeHandle;

  DomElementNode(this.tagName);

  void reset() {
    style.clear();
    children.clear();
    textContent = '';
    className = '';
  }

  String toHtmlString() {
    final styleStr = style.entries
        .map((e) => '${_escapeHtml(e.key)}:${_escapeHtml(e.value)}')
        .join(';');
    final styleAttr = style.isEmpty ? '' : ' style="$styleStr"';
    final classAttr = className.isEmpty
        ? ''
        : ' class="${_escapeHtml(className)}"';
    final escapedText = _escapeHtml(textContent);

    if (children.isEmpty) {
      return '<$tagName$classAttr$styleAttr>$escapedText</$tagName>';
    }

    final childrenHtml = children.map((c) => c.toHtmlString()).join();
    return '<$tagName$classAttr$styleAttr>$escapedText$childrenHtml</$tagName>';
  }

  static String _escapeHtml(String input) {
    return input
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
  }
}

/// Recycling pool for virtualized DOM line and token elements.
/// Keeps visibleLineCount + extraBuffer nodes allocated to eliminate DOM layout thrashing.
class DomNodePool {
  final int extraBuffer;
  final List<DomElementNode> _linePool = [];
  final Set<DomElementNode> _activeLines = {};
  final List<DomElementNode> _spanPool = [];
  final Set<DomElementNode> _activeSpans = {};

  DomNodePool({this.extraBuffer = 10});

  int get totalAllocated => _linePool.length + _spanPool.length;
  int get activeCount => _activeLines.length + _activeSpans.length;

  /// Ensure the pool contains sufficient nodes for the visible viewport line count.
  void preparePool(int visibleLineCount) {
    final targetCapacity = visibleLineCount + extraBuffer;
    while (_linePool.length < targetCapacity) {
      _linePool.add(DomElementNode('div'));
    }
  }

  /// Acquire a recycled DOM line element from the pool.
  DomElementNode acquireLineNode() {
    for (final node in _linePool) {
      if (!_activeLines.contains(node)) {
        node.reset();
        node.className = 'editor-line';
        _activeLines.add(node);
        return node;
      }
    }
    // If pool exhausted, create a new line node
    final newNode = DomElementNode('div');
    newNode.className = 'editor-line';
    _linePool.add(newNode);
    _activeLines.add(newNode);
    return newNode;
  }

  /// Acquire a recycled DOM token span element from the pool.
  DomElementNode acquireSpanNode() {
    for (final node in _spanPool) {
      if (!_activeSpans.contains(node)) {
        node.reset();
        node.className = 'token';
        _activeSpans.add(node);
        return node;
      }
    }
    // If pool exhausted, create a new span node
    final newNode = DomElementNode('span');
    newNode.className = 'token';
    _spanPool.add(newNode);
    _activeSpans.add(newNode);
    return newNode;
  }

  /// Recycle active line nodes and span nodes that are no longer visible.
  void recycleAll() {
    for (final node in _activeLines) {
      node.reset();
    }
    _activeLines.clear();

    for (final node in _activeSpans) {
      node.reset();
    }
    _activeSpans.clear();
  }

  /// Clear all nodes in the pool.
  void clear() {
    _linePool.clear();
    _activeLines.clear();
    _spanPool.clear();
    _activeSpans.clear();
  }
}
