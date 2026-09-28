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

import 'package:code_editor/backends/html.dart';
import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('DomNodePool & DomElementNode Tests', () {
    test(
      'DomElementNode escapes text and attributes against XSS injection',
      () {
        final node = DomElementNode('div');
        node.className = 'test-class" onclick="alert(1)';
        node.style['color'] = 'red"; font-size: 12px';
        node.textContent = '<script>alert("xss")</script>';

        final html = node.toHtmlString();
        expect(
          html,
          contains('class="test-class&quot; onclick=&quot;alert(1)"'),
        );
        expect(html, contains('style="color:red&quot;; font-size: 12px"'));
        expect(
          html,
          contains('&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'),
        );
      },
    );

    test('DomNodePool acquires and recycles line div and token span nodes', () {
      final pool = DomNodePool(extraBuffer: 2);
      pool.preparePool(5);

      final line1 = pool.acquireLineNode();
      expect(line1.tagName, equals('div'));
      expect(line1.className, equals('editor-line'));

      final span1 = pool.acquireSpanNode();
      expect(span1.tagName, equals('span'));
      expect(span1.className, equals('token'));

      expect(pool.activeCount, equals(2));

      pool.recycleAll();
      expect(pool.activeCount, equals(0));

      final lineReused = pool.acquireLineNode();
      expect(identical(lineReused, line1), isTrue);

      final spanReused = pool.acquireSpanNode();
      expect(identical(spanReused, span1), isTrue);
    });
  });

  group('HtmlRenderer Tests', () {
    late HtmlRenderer renderer;

    setUp(() {
      renderer = HtmlRenderer();
    });

    test(
      'render builds line divs and token spans using nodePool recycling',
      () {
        const viewport = RenderViewport(
          width: 200,
          height: 100,
          scrollX: 0,
          scrollY: 0,
          firstVisibleLine: 0,
          lastVisibleLine: 1,
          lines: [
            RenderLine(
              lineIndex: 0,
              text: 'const x = 1;',
              tokens: [
                RenderToken(
                  text: 'const',
                  startColumn: 0,
                  endColumn: 5,
                  style: RenderTokenStyle(foreground: '#569CD6', bold: true),
                ),
                RenderToken(
                  text: ' x = 1;',
                  startColumn: 5,
                  endColumn: 12,
                  style: RenderTokenStyle(foreground: '#9CDCFFE'),
                ),
              ],
              top: 0,
              height: 20,
            ),
          ],
          selections: <SelectionRect>[],
          carets: <CaretPosition>[],
          gutter: RenderGutter(
            width: 30,
            items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
          ),
          lineHeight: 20,
          charWidth: 8,
        );

        renderer.render(viewport);
        expect(renderer.renderedLineNodes.length, equals(1));
        expect(
          renderer.nodePool.activeCount,
          equals(3),
        ); // 1 line div + 2 token spans

        final htmlStr = renderer.renderToHtmlString(viewport);
        expect(htmlStr, contains('<div class="code-editor-viewport"'));
        expect(htmlStr, contains('font-weight:bold'));
        expect(htmlStr, contains('const'));
        expect(htmlStr, contains(' x = 1;'));
      },
    );
  });
}
