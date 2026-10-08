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

import "package:ndarray/ndarray.dart";
import "package:test/test.dart";

void main() {
  group("Standard Polynomial Operations - polyval, polyfit, roots", () {
    test("polyval evaluates 1D polynomials correctly", () {
      NDArray.scope(() {
        // p(x) = 3x^2 + 2x + 1
        final c = NDArray.fromList([3.0, 2.0, 1.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = polyval(c, x);
        expect(y.shape, equals([3]));
        expect(y.getCell([0]), closeTo(1.0, 1e-6));
        expect(y.getCell([1]), closeTo(2.75, 1e-6));
        expect(y.getCell([2]), closeTo(6.0, 1e-6));
      });
    });

    test(
      "polyval with single coefficient and int64 promotion / mixed dtypes",
      () {
        NDArray.scope(() {
          final cInt = NDArray.fromList([5], [1], DType.int64);
          final x = NDArray.fromList([1, 2, 3], [3], DType.int64);
          final y = polyval(cInt, x);
          expect(y.dtype, equals(DType.float64));
          expect(y.shape, equals([3]));
          expect(y.getCell([0]), equals(5.0));
          expect(y.getCell([1]), equals(5.0));
          expect(y.getCell([2]), equals(5.0));

          final cF64 = NDArray.fromList([5.0], [1], DType.float64);
          expect(
            () => polyval<AnySpec, DTypeTag>(cF64, x),
            throwsArgumentError,
          );
        });
      },
    );

    test("polyval with out parameter", () {
      NDArray.scope(() {
        final c = NDArray.fromList(
          [1.0, -3.0, 2.0],
          [3],
          DType.float64,
        ); // x^2 - 3x + 2
        final x = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
        final out = NDArray.zeros([3], DType.float64);
        final res = polyval(c, x, out: out);
        expect(identical(res, out), isTrue);
        expect(out.getCell([0]), closeTo(2.0, 1e-6));
        expect(out.getCell([1]), closeTo(0.0, 1e-6));
        expect(out.getCell([2]), closeTo(0.0, 1e-6));
      });
    });

    test("polyfit fits linear and quadratic data accurately", () {
      NDArray.scope(() {
        final x = NDArray.fromList([0.0, 1.0, 2.0, 3.0], [4], DType.float64);
        final y = NDArray.fromList(
          [1.0, 3.0, 5.0, 7.0],
          [4],
          DType.float64,
        ); // y = 2x + 1
        final p = polyfit(x, y, 1);
        expect(p.shape, equals([2]));
        expect(p.getCell([0]), closeTo(2.0, 1e-5));
        expect(p.getCell([1]), closeTo(1.0, 1e-5));
      });
    });

    test("polyfit with weights", () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final y = NDArray.fromList([3.0, 5.0, 7.0, 9.0], [4], DType.float64);
        final w = NDArray.fromList([1.0, 1.0, 1.0, 1.0], [4], DType.float64);
        final p = polyfit(x, y, 1, w: w);
        expect(p.getCell([0]), closeTo(2.0, 1e-5));
        expect(p.getCell([1]), closeTo(1.0, 1e-5));
      });
    });

    test("quadratic polyfit", () {
      NDArray.scope(() {
        final x = NDArray.fromList([-1.0, 0.0, 1.0, 2.0], [4], DType.float64);
        final y = NDArray.fromList([6.0, 2.0, 0.0, 0.0], [4], DType.float64);
        final p = polyfit(x, y, 2);
        expect(p.shape, equals([3]));
        expect(p.getCell([0]), closeTo(1.0, 1e-5));
        expect(p.getCell([1]), closeTo(-3.0, 1e-5));
        expect(p.getCell([2]), closeTo(2.0, 1e-5));
      });
    });

    test("roots finds real polynomial roots", () {
      NDArray.scope(() {
        // 2x^2 - 4x - 6 = 0 => roots -1 and 3
        final p = NDArray.fromList([2.0, -4.0, -6.0], [3], DType.float64);
        final r = roots(p);
        expect(r.shape, equals([2]));
        final r1 = r.getCell([0]).real;
        final r2 = r.getCell([1]).real;
        final vals = [r1, r2]..sort();
        expect(vals[0], closeTo(-1.0, 1e-5));
        expect(vals[1], closeTo(3.0, 1e-5));
      });
    });

    test("roots finds linear root", () {
      NDArray.scope(() {
        final p = NDArray.fromList([3.0, -6.0], [2], DType.float64);
        final r = roots(p);
        expect(r.shape, equals([1]));
        expect(r.getCell([0]).real, closeTo(2.0, 1e-5));
      });
    });

    test("roots finds complex roots for x^2 + 1", () {
      NDArray.scope(() {
        final p = NDArray.fromList([1.0, 0.0, 1.0], [3], DType.float64);
        final r = roots(p);
        expect(r.shape, equals([2]));
        final i1 = r.getCell([0]).imag.abs();
        final i2 = r.getCell([1]).imag.abs();
        expect(i1, closeTo(1.0, 1e-5));
        expect(i2, closeTo(1.0, 1e-5));
      });
    });

    test("error conditions for polyval, polyfit, roots", () {
      NDArray.scope(() {
        final empty = NDArray.zeros([0], DType.float64);
        final mat = NDArray.zeros([2, 2], DType.float64);
        final vec = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        expect(() => polyval(empty, vec), throwsArgumentError);
        expect(() => polyval(mat, vec), throwsArgumentError);
        expect(() => polyfit(vec, vec, -1), throwsArgumentError);
        expect(
          () => polyfit(vec, NDArray.fromList([1.0], [1], DType.float64), 1),
          throwsArgumentError,
        );
        expect(() => roots(mat), throwsArgumentError);
      });
    });
  });

  group("Orthogonal Polynomial Bases - Chebyshev, Legendre, Hermite", () {
    test("chebval evaluation and chebroots", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = chebval(x, c);
        expect(y.getCell([0]), closeTo(-2.0, 1e-5));
        expect(y.getCell([1]), closeTo(0.5, 1e-5));
        expect(y.getCell([2]), closeTo(6.0, 1e-5));

        final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final r = chebroots(c2);
        final r1 = r.getCell([0]).real.abs();
        final r2 = r.getCell([1]).real.abs();
        final rVals = [r1, r2]..sort();
        expect(rVals[0], closeTo(0.707106, 1e-4));
      });
    });

    test("legval evaluation and legroots", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = legval(x, c);
        expect(y.getCell([0]), closeTo(-0.5, 1e-5));
        expect(y.getCell([1]), closeTo(1.625, 1e-5));
        expect(y.getCell([2]), closeTo(6.0, 1e-5));

        final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final r = legroots(c2);
        final r1 = r.getCell([0]).real.abs();
        final r2 = r.getCell([1]).real.abs();
        final rVals = [r1, r2]..sort();
        expect(rVals[0], closeTo(0.57735, 1e-4));
      });
    });

    test("hermval evaluation and hermroots", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = hermval(x, c);
        expect(y.getCell([0]), closeTo(-5.0, 1e-5));
        expect(y.getCell([1]), closeTo(0.0, 1e-5));
        expect(y.getCell([2]), closeTo(11.0, 1e-5));

        final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final r = hermroots(c2);
        final r1 = r.getCell([0]).real.abs();
        final r2 = r.getCell([1]).real.abs();
        final rVals = [r1, r2]..sort();
        expect(rVals[0], closeTo(0.707106, 1e-4));
      });
    });

    test("lagval evaluation and lagroots", () {
      NDArray.scope(() {
        // L_0 = 1, L_1 = 1-x, L_2 = 1 - 2x + 0.5x^2
        // c = [1, 2, 3] => p(x) = 1*L0 + 2*L1 + 3*L2
        // p(0) = 1 + 2 + 3 = 6.0
        // p(0.5) = 1 + 1 + 3(0.125) = 2.375
        // p(1.0) = 1 + 0 + 3(-0.5) = -0.5
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = lagval(x, c);
        expect(y.getCell([0]), closeTo(6.0, 1e-5));
        expect(y.getCell([1]), closeTo(2.375, 1e-5));
        expect(y.getCell([2]), closeTo(-0.5, 1e-5));

        // Roots of L_2(x) = 0.5*(x^2 - 4x + 2) => 2 +/- sqrt(2) approx 0.585786, 3.414214
        final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final r = lagroots(c2);
        expect(r.shape, equals([2]));
        final r1 = r.getCell([0]).real;
        final r2 = r.getCell([1]).real;
        final rVals = [r1, r2]..sort();
        expect(rVals[0], closeTo(0.585786, 1e-4));
        expect(rVals[1], closeTo(3.414214, 1e-4));
      });
    });

    test("orthogonal polynomials with Float32 dtype", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float32);

        final yCheb = chebval(x, c);
        expect(yCheb.dtype, equals(DType.float32));
        expect(yCheb.getCell([0]), closeTo(-2.0, 1e-4));
        expect(yCheb.getCell([1]), closeTo(0.5, 1e-4));
        expect(yCheb.getCell([2]), closeTo(6.0, 1e-4));

        final yLeg = legval(x, c);
        expect(yLeg.dtype, equals(DType.float32));
        expect(yLeg.getCell([0]), closeTo(-0.5, 1e-4));
        expect(yLeg.getCell([1]), closeTo(1.625, 1e-4));
        expect(yLeg.getCell([2]), closeTo(6.0, 1e-4));

        final yHerm = hermval(x, c);
        expect(yHerm.dtype, equals(DType.float32));
        expect(yHerm.getCell([0]), closeTo(-5.0, 1e-4));
        expect(yHerm.getCell([1]), closeTo(0.0, 1e-4));
        expect(yHerm.getCell([2]), closeTo(11.0, 1e-4));

        final yLag = lagval(x, c);
        expect(yLag.dtype, equals(DType.float32));
        expect(yLag.getCell([0]), closeTo(6.0, 1e-4));
        expect(yLag.getCell([1]), closeTo(2.375, 1e-4));
        expect(yLag.getCell([2]), closeTo(-0.5, 1e-4));
      });
    });

    test("orthogonal polynomials with Complex dtypes", () {
      NDArray.scope(() {
        final c = NDArray.fromList(
          [Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0)],
          [3],
          DType.complex128,
        );
        final x = NDArray.fromList(
          [Complex(0.0, 1.0), Complex(1.0, -1.0)],
          [2],
          DType.complex128,
        );

        final yCheb = chebval(x, c);
        expect(yCheb.dtype, equals(DType.complex128));
        expect(yCheb.shape, equals([2]));
        // T_0(i) = 1, T_1(i) = i, T_2(i) = 2(i)^2 - 1 = -3
        // chebval(i, [1, 2, 3]) = 1(1) + 2(i) + 3(-3) = -8 + 2i
        expect(yCheb.getCell([0]).real, closeTo(-8.0, 1e-5));
        expect(yCheb.getCell([0]).imag, closeTo(2.0, 1e-5));

        final yPoly = polyval(c, x);
        expect(yPoly.dtype, equals(DType.complex128));
        // polyval([1, 2, 3], i) = 1*(i)^2 + 2*(i) + 3 = -1 + 2i + 3 = 2 + 2i
        expect(yPoly.getCell([0]).real, closeTo(2.0, 1e-5));
        expect(yPoly.getCell([0]).imag, closeTo(2.0, 1e-5));
      });
    });

    test("strided and multidimensional orthogonal evaluations", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        // 2x2 multidimensional array
        final x2d = NDArray.fromList(
          [0.0, 0.5, 1.0, 0.0],
          [2, 2],
          DType.float64,
        );
        final yCheb2d = chebval(x2d, c);
        expect(yCheb2d.shape, equals([2, 2]));
        expect(yCheb2d.getCell([0, 0]), closeTo(-2.0, 1e-5));
        expect(yCheb2d.getCell([0, 1]), closeTo(0.5, 1e-5));
        expect(yCheb2d.getCell([1, 0]), closeTo(6.0, 1e-5));
        expect(yCheb2d.getCell([1, 1]), closeTo(-2.0, 1e-5));

        // Strided (non-contiguous) input
        final xFull = NDArray.fromList(
          [0.0, 99.0, 0.5, 99.0, 1.0],
          [5],
          DType.float64,
        );
        final xSlice = xFull.slice([
          const Slice(start: 0, stop: 5, step: 2),
        ]); // [0.0, 0.5, 1.0], stride = 2
        expect(xSlice.isContiguous, isFalse);

        final yStrided = chebval(xSlice, c);
        expect(yStrided.getCell([0]), closeTo(-2.0, 1e-5));
        expect(yStrided.getCell([1]), closeTo(0.5, 1e-5));
        expect(yStrided.getCell([2]), closeTo(6.0, 1e-5));

        // Out buffer with strided array
        final outBuf = NDArray.zeros([3], DType.float64);
        final res = chebval(xSlice, c, out: outBuf);
        expect(identical(res, outBuf), isTrue);
        expect(outBuf.getCell([0]), closeTo(-2.0, 1e-5));
        expect(outBuf.getCell([1]), closeTo(0.5, 1e-5));
        expect(outBuf.getCell([2]), closeTo(6.0, 1e-5));
      });
    });

    test("degree 0 and degree 1 edge cases", () {
      NDArray.scope(() {
        // Degree 0 (single coefficient)
        final c0 = NDArray.fromList([42.0], [1], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        expect(chebval(x, c0).getCell([0]), equals(42.0));
        expect(chebval(x, c0).getCell([1]), equals(42.0));
        expect(legval(x, c0).getCell([0]), equals(42.0));
        expect(hermval(x, c0).getCell([0]), equals(42.0));
        expect(lagval(x, c0).getCell([0]), equals(42.0));

        // Degree 1 (two coefficients c0 + c1*Basis_1(x))
        final c1 = NDArray.fromList([3.0, 5.0], [2], DType.float64);
        // cheb: 3 + 5*x
        expect(chebval(x, c1).getCell([1]), closeTo(5.5, 1e-5));
        // leg: 3 + 5*x
        expect(legval(x, c1).getCell([1]), closeTo(5.5, 1e-5));
        // herm: 3 + 5*(2x) = 3 + 10x
        expect(hermval(x, c1).getCell([1]), closeTo(8.0, 1e-5));
        // lag: 3 + 5*(1-x) = 3 + 2.5 = 5.5
        expect(lagval(x, c1).getCell([1]), closeTo(5.5, 1e-5));
      });
    });
  });
}
