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

import 'package:test/test.dart';
import 'package:symbolic_dart/symbolic_dart.dart';
import 'package:notebook/src/kernel_helper.dart';

void main() {
  group('Notebook Symbolic Integration Tests', () {
    test('prettyFormat formats Expr into KaTeX LaTeX HTML', () {
      final x = Symbol('x');
      final f = sin(x ^ 2);
      final formatted = prettyFormat(f);

      expect(formatted, contains('class="math-latex"'));
      expect(formatted, contains(r'\sin'));
      expect(formatted, contains(r'x^2'));
    });

    test('display captures Expr and produces HTML output item', () {
      clearCapturedOutput();
      final x = Symbol('x');
      final f = (x ^ 3) + (Integer(5) * x);
      display(f);

      final outputs = capturedOutputs;
      expect(outputs, hasLength(1));
      expect(outputs.first.mimeType, equals('text/html'));
      expect(outputs.first.data, contains('class="math-latex"'));
      expect(outputs.first.data, contains(r'x^3'));
    });

    test('plotSymbolic compiles and returns renderable Plot widget', () {
      final x = Symbol('x');
      final f = sin(x);
      final plot = plotSymbolic(
        f,
        x,
        from: 0,
        to: 3.14159,
        points: 50,
        title: 'Sine Wave',
      );

      expect(plot, isA<Plot>());
      expect(plot.title, equals('Sine Wave'));
      expect(plot.x, isNotNull);
      expect(plot.x!.shape, equals([50]));
      expect(plot.y.shape, equals([50]));

      final html = plot.toHtml();
      expect(html, contains('<svg'));
      expect(html, contains('Sine Wave'));
    });

    test('plotSymbolic2D compiles and returns renderable Heatmap widget', () {
      final x = Symbol('x');
      final y = Symbol('y');
      final f = (x ^ 2) + (y ^ 2);
      final heatmap = plotSymbolic2D(
        f,
        x,
        y,
        xFrom: -2,
        xTo: 2,
        yFrom: -2,
        yTo: 2,
        points: 20,
        title: 'Paraboloid',
      );

      expect(heatmap, isA<Heatmap>());
      final html = heatmap.toHtml();
      expect(html, contains('heatmap-container'));
      expect(html, contains('Paraboloid'));
    });

    test(
      'prettyFormat formats SymbolicMatrix into KaTeX LaTeX bmatrix HTML',
      () {
        final x = Symbol('x');
        final mat = SymbolicMatrix.fromList([
          [x ^ 2, sin(x)],
          [cos(x), Integer(5)],
        ]);
        final formatted = prettyFormat(mat);

        expect(formatted, contains('class="math-latex"'));
        expect(formatted, contains(r'\begin{bmatrix}'));
        expect(formatted, contains(r'x^2'));
        expect(formatted, contains(r'&'));
        expect(formatted, contains(r'\\'));
        expect(formatted, contains(r'\end{bmatrix}'));
      },
    );

    test(
      'prettyFormat formats FlintRationalPoly into KaTeX LaTeX polynomial HTML',
      () {
        final p = FlintRationalPoly.fromIntCoefficients([-4, 0, 1]); // x^2 - 4
        final formatted = prettyFormat(p);

        expect(formatted, contains('class="math-latex"'));
        expect(formatted, contains(r'x^{2} - 4'));
      },
    );

    test('PolyFactorization toLatex formats factored polynomial', () {
      final p1 = FlintRationalPoly.fromIntCoefficients([-4, 0, 1]); // x^2 - 4
      final fac = p1.factor();
      final latex = fac.toLatex();

      expect(latex, contains(r'\left('));
      expect(latex, contains(r'\right)'));
    });
  });
}
