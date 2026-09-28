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

void main() {
  group('Expr constants and elementary constructors', () {
    test('zero, one, minusOne, i, pi, e', () {
      expect(Expr.zero.isZero, isTrue);
      expect(Expr.one.asDouble, closeTo(1.0, 1e-12));
      expect(Expr.minusOne.isNegative, isTrue);
      expect(Expr.i.isComplex, isTrue);
      expect(Expr.pi.asDouble, closeTo(3.141592653589793, 1e-12));
      expect(Expr.e.asDouble, closeTo(2.718281828459045, 1e-12));
      expect(Expr.eulerGamma.asDouble, closeTo(0.5772156649, 1e-6));
      expect(Expr.catalan.asDouble, closeTo(0.9159655941, 1e-6));
      expect(Expr.goldenRatio.asDouble, closeTo(1.6180339887, 1e-6));
      expect(Expr.infinity.toString(), contains('oo'));
      expect(
        Expr.modInverse(3, 7).asDouble,
        equals(5.0),
      ); // 3 * 5 = 15 = 1 mod 7
    });

    test('rational constructors', () {
      final half = Rational(1, 2);
      expect(half.asDouble, closeTo(0.5, 1e-12));

      expect(() => Rational(1, 0), throwsArgumentError);
    });

    test('symbol constructors and parsing', () {
      final x = Symbol('x');
      expect(x.toString(), 'x');

      final parsed = Expr.parse('sin(x) + 2*x');
      expect(parsed.hasSymbol(x), isTrue);
    });
  });

  group('Expr arithmetic operators', () {
    test('addition, subtraction, multiplication, division', () {
      final x = Symbol('x');
      final expr = (x * 2) + 5 - x;
      final evaluated = expr.subs({x: 3.0}).asDouble;
      // 3*2 + 5 - 3 = 8
      expect(evaluated, closeTo(8.0, 1e-12));
    });

    test('exponentiation using ^ operator', () {
      final x = Symbol('x');
      final square = x ^ 2;
      expect(square.subs({x: 4.0}).asDouble, closeTo(16.0, 1e-12));
    });
  });

  group('Calculus and symbolic manipulation', () {
    test('differentiation (diff)', () {
      final x = Symbol('x');
      // f(x) = x^3 + 2*x^2 + 5*x
      final f = (x ^ 3) + ((x ^ 2) * 2) + (x * 5);
      final df = f.diff(x);
      // df/dx at x=2: 3(4) + 4(2) + 5 = 12 + 8 + 5 = 25
      expect(df.subs({x: 2.0}).asDouble, closeTo(25.0, 1e-12));
    });

    test('differentiation of elementary functions', () {
      final x = Symbol('x');
      final f = sin(x);
      final df = f.diff(x);
      // derivative of sin(x) is cos(x)
      expect(df.subs({x: 0.0}).asDouble, closeTo(1.0, 1e-12));
    });

    test('polynomial expansion', () {
      final x = Symbol('x');
      final f = (x + 2) ^ 2;
      final expanded = f.expand();
      // (x+2)^2 = x^2 + 4x + 4
      expect(expanded.subs({x: 10.0}).asDouble, closeTo(144.0, 1e-12));
    });

    test('args and freeSymbols properties', () {
      final x = Symbol('x');
      final y = Symbol('y');
      final f = x + y + 5;
      expect(f.freeSymbols.length, 2);
    });

    test('toLatex, toCCode, toJSCode, toMathML format printers', () {
      final x = Symbol('x');
      final f = sin(x ^ 2);
      expect(f.toLatex(), isNotEmpty);
      expect(f.toCCode(), isNotEmpty);
      expect(f.toJSCode(), isNotEmpty);
      expect(f.toMathML(), isNotEmpty);
    });

    test('trig, hyperbolic, and special functions chaining', () {
      expect(Integer(9).sqrt().asDouble, closeTo(3.0, 1e-12));
      expect(Integer(27).cbrt().asDouble, closeTo(3.0, 1e-12));
      expect(kroneckerDelta(1, 1).asDouble, equals(1.0));
      expect(kroneckerDelta(1, 2).asDouble, equals(0.0));
      expect(gcd(12, 18).asDouble, equals(6.0));
      expect(lcm(12, 18).asDouble, equals(36.0));
      expect(Real(3.7).floor().asDouble, equals(3.0));
      expect(Real(3.7).ceil().asDouble, equals(4.0));
    });

    test('asNumerDenom rational separation', () {
      final x = Symbol('x');
      final frac = (x ^ 2) / (x + 1);
      final parts = frac.asNumerDenom();
      expect(parts.numerator.eq(x ^ 2), isTrue);
      expect(parts.denominator.eq(x + 1), isTrue);
    });

    test('exact polynomial solver solvePoly', () {
      final x = Symbol('x');
      // x^2 - 9 = 0
      final roots = Expr.solvePoly((x ^ 2) - 9, x);
      expect(roots.length, equals(2));
      final rootVals = roots.map((r) => r.asDouble).toSet();
      expect(rootVals, equals({-3.0, 3.0}));
    });

    test('exact linear system solver solveLinearSystem', () {
      final x = Symbol('x');
      final y = Symbol('y');
      // 2*x + y - 5 = 0
      // x + 3*y - 5 = 0
      // Solution: x = 2, y = 1
      final sol = Expr.solveLinearSystem(
        [(Integer(2) * x) + y - 5, x + (Integer(3) * y) - 5],
        [x, y],
      );
      expect(sol.length, equals(2));
      expect(sol[0].asDouble, closeTo(2.0, 1e-12));
      expect(sol[1].asDouble, closeTo(1.0, 1e-12));
    });

    test('all remaining top-level trig, hyperbolic, and special functions', () {
      expect(tan(0).asDouble, closeTo(0.0, 1e-12));
      expect(asin(0).asDouble, closeTo(0.0, 1e-12));
      expect(acos(1).asDouble, closeTo(0.0, 1e-12));
      expect(atan(0).asDouble, closeTo(0.0, 1e-12));
      expect(atan2(0, 1).asDouble, closeTo(0.0, 1e-12));
      expect(csc(Expr.pi / 2).asDouble, closeTo(1.0, 1e-12));
      expect(sec(0).asDouble, closeTo(1.0, 1e-12));
      expect(cot(Expr.pi / 2).asDouble, closeTo(0.0, 1e-12));

      expect(sinh(0).asDouble, closeTo(0.0, 1e-12));
      expect(cosh(0).asDouble, closeTo(1.0, 1e-12));
      expect(tanh(0).asDouble, closeTo(0.0, 1e-12));
      expect(asinh(0).asDouble, closeTo(0.0, 1e-12));
      expect(acosh(1).asDouble, closeTo(0.0, 1e-12));
      expect(atanh(0).asDouble, closeTo(0.0, 1e-12));

      expect(floor(3.8).asDouble, equals(3.0));
      expect(ceil(3.2).asDouble, equals(4.0));
      expect(erf(0).asDouble, closeTo(0.0, 1e-12));
      expect(erfc(0).asDouble, closeTo(1.0, 1e-12));
      expect(gamma(1).asDouble, closeTo(1.0, 1e-12));
      expect(lambertw(0).asDouble, closeTo(0.0, 1e-12));
      expect(zeta(2).asDouble, closeTo(1.644934, 1e-5)); // pi^2/6
    });

    test('special constants and BigInt / String constructors', () {
      expect(Expr.negInfinity.toString(), contains('-oo'));
      expect(Expr.complexInfinity.toString(), contains('zoo'));
      expect(Expr.nan.toString(), contains('nan'));

      final bigIntExpr = Expr.fromObject(BigInt.from(123456789));
      expect(bigIntExpr.asDouble, closeTo(123456789.0, 1e-6));

      final strExpr = Expr.fromObject('y');
      expect(strExpr.toString(), equals('y'));

      expect(() => Expr.parse('invalid ((( formula'), throwsFormatException);
      expect(() => Expr.fromObject(Object()), throwsArgumentError);
    });

    test('unary negation and binary arithmetic operators', () {
      final x = Symbol('x');
      final neg = -x;
      expect(neg.subs({x: 5.0}).asDouble, closeTo(-5.0, 1e-12));

      final a = Integer(10);
      final b = Integer(2);
      expect((a + b).asDouble, equals(12.0));
      expect((a - b).asDouble, equals(8.0));
      expect((a * b).asDouble, equals(20.0));
      expect((a / b).asDouble, equals(5.0));
    });

    test(
      'Symbol extension type, SymbolicNumExtension, exp, higher-order diff, series, gradient, hessian',
      () {
        final Symbol x = Symbol('x');
        final Symbol y = Symbol('y');
        expect(Symbol.fromExpr(x), equals(x));
        expect(() => Symbol.fromExpr(x + 1), throwsArgumentError);

        // SymbolicNumExtension: natural .toExpr and Expr operator syntax
        final poly = 2.toExpr * (x ^ 2) + 3.toExpr * x - 5 + (10.toExpr / x);
        // At x = 2: 2(4) + 3(2) - 5 + 5 = 8 + 6 - 5 + 5 = 14
        expect(poly.subs({x: 2.0}).asDouble, closeTo(14.0, 1e-12));

        // Instance .exp() method
        expect(x.exp().subs({x: 0.0}).asDouble, closeTo(1.0, 1e-12));

        // Higher-order derivatives
        final cubic = (x ^ 3) + ((x ^ 2) * 2) + (x * 5);
        expect(cubic.diff(x, 0), equals(cubic));
        expect(cubic.diff(x, 1).subs({x: 2.0}).asDouble, closeTo(25.0, 1e-12));
        expect(
          cubic.diff(x, 2).subs({x: 2.0}).asDouble,
          closeTo(16.0, 1e-12),
        ); // 6x + 4 at x=2 -> 16
        expect(cubic.diff(x, 3).asDouble, closeTo(6.0, 1e-12));
        expect(cubic.diff(x, 4).isZero, isTrue);
        expect(() => cubic.diff(x, -1), throwsArgumentError);

        // Taylor series expansion
        // sin(x) around 0 up to order 6 -> x - x^3/6 + x^5/120
        // At x = 0.5: 0.5 - 0.125/6 + 0.03125/120 = 0.47942708333333334
        final sinTaylor = sin(x).series(x, at: 0, order: 6);
        expect(
          sinTaylor.subs({x: 0.5}).asDouble,
          closeTo(0.47942708333333334, 1e-12),
        );

        // Gradient and Hessian
        // f(x, y) = x^2 * y + 3 * y^2
        final f = ((x ^ 2) * y) + (3.toExpr * (y ^ 2));
        final grad = f.gradient([x, y]);
        expect(grad.shape, equals((rows: 2, cols: 1)));
        // df/dx = 2xy (at 2, 3 -> 12), df/dy = x^2 + 6y (at 2, 3 -> 4 + 18 = 22)
        final gradAt = grad.subs({x: 2.0, y: 3.0});
        expect(gradAt.getCell(0, 0).asDouble, closeTo(12.0, 1e-12));
        expect(gradAt.getCell(1, 0).asDouble, closeTo(22.0, 1e-12));

        final hess = f.hessian([x, y]);
        expect(hess.shape, equals((rows: 2, cols: 2)));
        // d2f/dx2 = 2y (6), d2f/dxdy = 2x (4), d2f/dy2 = 6 (6)
        final hessAt = hess.subs({x: 2.0, y: 3.0});
        expect(hessAt.getCell(0, 0).asDouble, closeTo(6.0, 1e-12));
        expect(hessAt.getCell(0, 1).asDouble, closeTo(4.0, 1e-12));
        expect(hessAt.getCell(1, 0).asDouble, closeTo(4.0, 1e-12));
        expect(hessAt.getCell(1, 1).asDouble, closeTo(6.0, 1e-12));
      },
    );
  });
}
