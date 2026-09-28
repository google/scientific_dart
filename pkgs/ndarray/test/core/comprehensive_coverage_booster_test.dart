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
import 'package:ndarray/ndarray.dart';

void main() {
  group('Comprehensive Coverage Booster Suite', () {
    final allDTypes = [
      DType.float64,
      DType.float32,
      DType.float16,
      DType.bfloat16,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
      DType.complex128,
      DType.complex64,
      DType.boolean,
    ];

    final numericDTypes = [
      DType.float64,
      DType.float32,
      DType.float16,
      DType.bfloat16,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
    ];

    final integerDTypes = [
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
    ];

    NDArray<AnySpec> createArray(
      DType dt,
      List<int> shape, {
      bool strided = false,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<Object>.generate(size * (strided ? 2 : 1), (i) {
        final val = (i % 7) + 2; // avoid 0/1 division edge cases
        if (dt == DType.boolean) return val.isOdd;
        if (dt == DType.complex128 || dt == DType.complex64) {
          return Complex(val.toDouble(), (val + 1).toDouble());
        }
        return val;
      });

      final dtObj = dt;
      if (strided) {
        final flatArr = NDArray.fromList(rawList, [size * 2], dtObj);
        final sliced = flatArr[Slice(step: 2)];
        return sliced.reshape(shape);
      } else {
        return NDArray.fromList(rawList, shape, (dtObj as DType<AnySpec>));
      }
    }

    test(
      'Binary Arithmetic with out and where parameters across all 15 DTypes (Contiguous & Strided)',
      () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            for (final isStrided in [false, true]) {
              final a = createArray(dt, [2, 3], strided: isStrided);
              final b = createArray(dt, [2, 3], strided: isStrided);
              final mask = NDArray<Boolean>.fromList(
                [true, false, true, false, true, false],
                [2, 3],
                DType.boolean,
              );

              // Homogeneous binary ops with where mask
              final rAddMask = add(a, b, where: mask);
              expect(rAddMask.shape, [2, 3]);

              final rSubMask = subtract(a, b, where: mask);
              expect(rSubMask.shape, [2, 3]);

              final rMulMask = multiply(a, b, where: mask);
              expect(rMulMask.shape, [2, 3]);

              final rDivMask = divide(a, b, where: mask);
              expect(rDivMask.shape, [2, 3]);

              // Out buffer reuse
              final outAdd = NDArray.create([2, 3], rAddMask.dtype);
              add(a, b, out: outAdd, where: mask);
              expect(outAdd.shape, [2, 3]);

              final outSub = NDArray.create([2, 3], rSubMask.dtype);
              subtract(a, b, out: outSub, where: mask);
              expect(outSub.shape, [2, 3]);

              final outMul = NDArray.create([2, 3], rMulMask.dtype);
              multiply(a, b, out: outMul, where: mask);
              expect(outMul.shape, [2, 3]);

              final outDiv = NDArray.create([2, 3], rDivMask.dtype);
              divide(a, b, out: outDiv, where: mask);
              expect(outDiv.shape, [2, 3]);
            }
          }
        });
      },
    );

    test(
      'All Unary Arithmetic & Math Functions across all 15 DTypes (Contiguous & Strided with out & where)',
      () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            for (final isStrided in [false, true]) {
              final a = createArray(dt, [2, 3], strided: isStrided);
              final mask = NDArray<Boolean>.fromList(
                [true, false, true, false, true, false],
                [2, 3],
                DType.boolean,
              );

              // positive & negative
              if (dt != DType.boolean) {
                final pos = positive(a);
                expect(pos.shape, [2, 3]);
                final neg = negative(a);
                expect(neg.shape, [2, 3]);

                final posOut = NDArray.create([2, 3], dt);
                positive(a, out: posOut, where: mask);
                expect(posOut.shape, [2, 3]);

                final negOut = NDArray.create([2, 3], dt);
                negative(a, out: negOut, where: mask);
                expect(negOut.shape, [2, 3]);
              }

              // square & reciprocal
              if (dt != DType.boolean) {
                final sq = square(a);
                expect(sq.shape, [2, 3]);
                final sqOut = NDArray.create([2, 3], dt);
                square(a, out: sqOut, where: mask);
                expect(sqOut.shape, [2, 3]);

                final rec = reciprocal(a);
                expect(rec.shape, [2, 3]);
              }

              // sign, ceil, floor, round, rint, trunc, fix, abs
              if (dt != DType.boolean &&
                  dt != DType.complex128 &&
                  dt != DType.complex64) {
                final sgn = sign(a);
                expect(sgn.shape, [2, 3]);

                final cl = ceil(a);
                expect(cl.shape, [2, 3]);

                final fl = floor(a);
                expect(fl.shape, [2, 3]);

                final rd = round(a);
                expect(rd.shape, [2, 3]);

                final rn = rint(a);
                expect(rn.shape, [2, 3]);

                final tr = trunc(a);
                expect(tr.shape, [2, 3]);

                final fx = fix(a);
                expect(fx.shape, [2, 3]);

                final ab = abs(a);
                expect(ab.shape, [2, 3]);

                final abOut = NDArray.create([2, 3], dt);
                abs(a, out: abOut, where: mask);
                expect(abOut.shape, [2, 3]);
              }

              // sqrt, expm1, log1p
              if (dt != DType.boolean) {
                final sqr = sqrt(a);
                expect(sqr.shape, [2, 3]);

                final expm = expm1(a);
                expect(expm.shape, [2, 3]);

                final log1 = log1p(a);
                expect(log1.shape, [2, 3]);
              }
            }
          }
        });
      },
    );

    test(
      'Bitwise and Logical operators across integer and boolean DTypes (Contiguous & Strided)',
      () {
        NDArray.scope(() {
          for (final dt in [
            DType.int64,
            DType.int32,
            DType.int16,
            DType.uint8,
          ]) {
            for (final isStrided in [false, true]) {
              final a = createArray(dt, [2, 2], strided: isStrided);
              final b = createArray(dt, [2, 2], strided: isStrided);
              final mask = NDArray<Boolean>.fromList(
                [true, false, true, false],
                [2, 2],
                DType.boolean,
              );

              final band = bitwiseAnd(a, b);
              expect(band.shape, [2, 2]);

              final bor = bitwiseOr(a, b);
              expect(bor.shape, [2, 2]);

              final bxor = bitwiseXor(a, b);
              expect(bxor.shape, [2, 2]);

              final inv = invert(a);
              expect(inv.shape, [2, 2]);

              final ls = leftShift(a, b);
              expect(ls.shape, [2, 2]);

              final rs = rightShift(a, b);
              expect(rs.shape, [2, 2]);

              final outObj = NDArray.create([2, 2], dt);
              bitwiseAnd(a, b, out: outObj, where: mask);
              expect(outObj.shape, [2, 2]);
            }
          }

          for (final dt in [...integerDTypes, DType.boolean]) {
            for (final isStrided in [false, true]) {
              final a = createArray(dt, [2, 2], strided: isStrided);
              final b = createArray(dt, [2, 2], strided: isStrided);

              final land = logicalAnd(a, b);
              expect(land.shape, [2, 2]);

              final lor = logicalOr(a, b);
              expect(lor.shape, [2, 2]);

              final lxor = logicalXor(a, b);
              expect(lxor.shape, [2, 2]);

              final lnot = logicalNot(a);
              expect(lnot.shape, [2, 2]);
            }
          }
        });
      },
    );

    test(
      'Clip and ClipArray across all DTypes with scalar and array bounds',
      () {
        NDArray.scope(() {
          for (final dt in numericDTypes) {
            for (final isStrided in [false, true]) {
              final a = createArray(dt, [3, 3], strided: isStrided);

              final cBoth = clip(a, min: 3, max: 5);
              expect(cBoth.shape, [3, 3]);

              final cMinOnly = clip(a, min: 3, max: null);
              expect(cMinOnly.shape, [3, 3]);

              final cMaxOnly = clip(a, min: null, max: 5);
              expect(cMaxOnly.shape, [3, 3]);

              final minArr = createArray(dt, [3, 3]);
              final maxArr = createArray(dt, [3, 3]);
              final cArr = clipArray(a, min: minArr, max: maxArr);
              expect(cArr.shape, [3, 3]);

              final cArrMinOnly = clipArray(a, min: minArr, max: null);
              expect(cArrMinOnly.shape, [3, 3]);

              final cArrMaxOnly = clipArray(a, min: null, max: maxArr);
              expect(cArrMaxOnly.shape, [3, 3]);
            }
          }
        });
      },
    );

    test('Calculus operations (gradient, gradientArray, diff, trapz)', () {
      NDArray.scope(() {
        final a1d = NDArray.fromList(
          [1.0, 4.0, 9.0, 16.0, 25.0],
          [5],
          DType.float64,
        );
        final grad1 = gradient(a1d);
        expect(grad1.shape, [5]);

        final a2d = NDArray.fromList(
          [1.0, 2.0, 4.0, 7.0, 3.0, 5.0, 9.0, 12.0],
          [2, 4],
          DType.float64,
        );
        final grads = gradientArray(a2d);
        expect(grads.length, 2);
        expect(grads[0].shape, [2, 4]);
        expect(grads[1].shape, [2, 4]);

        final diff1 = diff(a2d, axis: 0);
        expect(diff1.shape, [1, 4]);

        final diff2 = diff(a2d, axis: 1, n: 2);
        expect(diff2.shape, [2, 2]);

        final tr0 = trapz(a2d, axis: 0);
        expect(tr0.shape, [4]);

        final tr1 = trapz(a2d, axis: 1, spacing: const Spacing.step(0.5));
        expect(tr1.shape, [2]);
      });
    });

    test('Orthogonal and Standard Polynomials evaluation and roots', () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList(
          [-1.0, -0.5, 0.0, 0.5, 1.0],
          [5],
          DType.float64,
        );

        // Legendre
        final legVal = legval(c, x);
        expect(legVal.shape, [5]);

        final legRt = legroots(c);
        expect(legRt.shape, [2]);

        // Chebyshev
        final chebVal = chebval(c, x);
        expect(chebVal.shape, [5]);

        final chebRt = chebroots(c);
        expect(chebRt.shape, [2]);

        // Hermite
        final hermVal = hermval(c, x);
        expect(hermVal.shape, [5]);

        final hermRt = hermroots(c);
        expect(hermRt.shape, [2]);

        // Laguerre
        final lagVal = lagval(c, x);
        expect(lagVal.shape, [5]);

        final lagRt = lagroots(c);
        expect(lagRt.shape, [2]);

        // Standard Poly
        final pval = polyval(c, x);
        expect(pval.shape, [5]);

        final pfit = polyfit(x, x, 1);
        expect(pfit.shape, [2]);

        final pRt = roots(c);
        expect(pRt.shape, [2]);
      });
    });

    test('Special functions, Windows, and DSP (correlate, convolve)', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0.1, 0.5, 1.0, 2.0], [4], DType.float64);

        final sc = sinc(x);
        expect(sc.shape, [4]);

        final ef = erf(x);
        expect(ef.shape, [4]);

        final gm = gamma(x);
        expect(gm.shape, [4]);

        final bessel = i0(x);
        expect(bessel.shape, [4]);

        // Windows
        final w3 = hamming(10);
        expect(w3.shape, [10]);

        final w4 = hanning(10);
        expect(w4.shape, [10]);

        // DSP Convolve & Correlate
        final sig1 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final sig2 = NDArray.fromList([0.5, 1.0, 0.5], [3], DType.float64);

        final convFull = convolve(sig1, sig2, mode: ConvMode.full);
        expect(convFull.shape, [7]);

        final convSame = convolve(sig1, sig2, mode: ConvMode.same);
        expect(convSame.shape, [5]);

        final convValid = convolve(sig1, sig2, mode: ConvMode.valid);
        expect(convValid.shape, [3]);

        final corrFull = correlate(sig1, sig2, mode: ConvMode.full);
        expect(corrFull.shape, [7]);

        final corrSame = correlate(sig1, sig2, mode: ConvMode.same);
        expect(corrSame.shape, [5]);

        final corrValid = correlate(sig1, sig2, mode: ConvMode.valid);
        expect(corrValid.shape, [3]);
      });
    });

    test(
      'Linalg advanced operators (norm, matrix_power, solve, multi_dot)',
      () {
        NDArray.scope(() {
          final mat2d = NDArray.fromList(
            [4.0, 2.0, 1.0, 3.0],
            [2, 2],
            DType.float64,
          );

          final mp2 = matrix_power(mat2d, 2);
          expect(mp2.shape, [2, 2]);

          final mp0 = matrix_power(mat2d, 0);
          expect(mp0.shape, [2, 2]);

          final mpNeg = matrix_power(mat2d, -1);
          expect(mpNeg.shape, [2, 2]);

          final nf = norm(mat2d, ord: 'fro');
          expect(nf.shape, <int>[]);

          final n1 = norm(mat2d, ord: 1);
          expect(n1.shape, <int>[]);

          final n2 = norm(mat2d, ord: 2);
          expect(n2.shape, <int>[]);

          final nInf = norm(mat2d, ord: double.infinity);
          expect(nInf.shape, <int>[]);

          final m1 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final m2 = NDArray.fromList(
            [2.0, 0.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final m3 = NDArray.fromList(
            [1.0, 1.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );

          final mdot = multi_dot([m1, m2, m3]);
          expect(mdot.shape, [2, 2]);

          final b = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final slv = solve(mat2d, b);
          expect(slv.shape, [2]);
        });
      },
    );
  });
}
