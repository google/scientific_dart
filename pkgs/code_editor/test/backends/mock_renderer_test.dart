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

import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

class MockEditorRenderer implements EditorRenderer {
  bool _isAttached = false;
  RenderViewport? _lastViewport;
  int renderCount = 0;
  final List<RenderEventListener> _listeners = [];

  @override
  bool get isAttached => _isAttached;

  @override
  RenderViewport? get lastViewport => _lastViewport;

  @override
  void attach(Object target) {
    _isAttached = true;
  }

  @override
  void detach() {
    _isAttached = false;
  }

  @override
  void render(RenderViewport viewport) {
    _lastViewport = viewport;
    renderCount++;
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
  void dispose() {
    detach();
    _listeners.clear();
  }
}

void main() {
  group('MockEditorRenderer Tests', () {
    late MockEditorRenderer renderer;

    setUp(() {
      renderer = MockEditorRenderer();
    });

    test('attach and detach updates isAttached flag', () {
      expect(renderer.isAttached, isFalse);
      renderer.attach('mock_target');
      expect(renderer.isAttached, isTrue);
      renderer.detach();
      expect(renderer.isAttached, isFalse);
    });

    test('render captures viewport snapshot and increments render count', () {
      renderer.attach('mock_target');
      const viewport = RenderViewport(
        width: 800,
        height: 600,
        scrollX: 0,
        scrollY: 0,
        firstVisibleLine: 0,
        lastVisibleLine: 2,
        lines: [
          RenderLine(
            lineIndex: 0,
            text: 'void main() {',
            tokens: [
              RenderToken(text: 'void', startColumn: 0, endColumn: 4),
              RenderToken(text: ' main() {', startColumn: 4, endColumn: 13),
            ],
            top: 0,
            height: 20,
          ),
        ],
        selections: [SelectionRect(left: 0, top: 0, width: 32, height: 20)],
        carets: [CaretPosition(x: 32, y: 0, height: 20, line: 0, column: 4)],
        gutter: RenderGutter(
          width: 40,
          items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
        ),
        lineHeight: 20,
        charWidth: 8,
      );

      renderer.render(viewport);
      expect(renderer.renderCount, equals(1));
      expect(renderer.lastViewport, equals(viewport));
      expect(renderer.lastViewport!.lines.length, equals(1));
      expect(renderer.lastViewport!.carets.first.column, equals(4));
    });
  });
}
