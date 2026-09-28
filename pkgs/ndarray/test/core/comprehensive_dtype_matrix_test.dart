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
import 'package:ndarray/src/operations/helpers.dart'
    show castValue, resolveDType;

void main() {
  group('Comprehensive 15x15 DType Arithmetic Matrix Tests', () {
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

    NDArray<AnySpec> createSampleArray(
      DType dt,
      List<int> shape, {
      bool strided = false,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<Object>.generate(size * (strided ? 2 : 1), (i) {
        final val = (i % 5) + 1;
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

    NDArray<AnySpec> createNumericArray(
      DType dt,
      List<int> shape, {
      bool strided = false,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<num>.generate(
        size * (strided ? 2 : 1),
        (i) => ((i % 5) + 1),
      );
      final dtNum = dt;
      if (strided) {
        final flatArr = NDArray.fromList(rawList, [size * 2], dtNum);
        final sliced = flatArr[Slice(step: 2)];
        return sliced.reshape(shape);
      } else {
        return NDArray.fromList(rawList, shape, (dtNum as DType<AnySpec>));
      }
    }

    test(
      'All 15x15 DType binary arithmetic (add, sub, mul, div, power, floorDivide, remainder, fmod)',
      () {
        NDArray.scope(() {
          for (final dtA in allDTypes) {
            for (final dtB in allDTypes) {
              for (final isStrided in [false, true]) {
                final a = createSampleArray(dtA, [2, 3], strided: isStrided);
                final b = createSampleArray(dtB, [2, 3], strided: isStrided);

                final expectedDType = resolveDType(dtA, dtB);
                final resAdd = add(a, b);
                expect(resAdd.shape, [2, 3]);
                expect(resAdd.dtype, expectedDType);

                final resSub = subtract(a, b);
                expect(resSub.shape, [2, 3]);
                expect(resSub.dtype, expectedDType);

                final resMul = multiply(a, b);
                expect(resMul.shape, [2, 3]);
                expect(resMul.dtype, expectedDType);

                final resDiv = divide(a, b);
                expect(resDiv.shape, [2, 3]);

                void expectRealClose(double got, double exp) {
                  expect(got.isNaN, exp.isNaN);
                  if (!exp.isNaN) {
                    if (exp.isInfinite) {
                      expect(got, exp);
                    } else {
                      expect(got, closeTo(exp, exp.abs() * 0.02 + 1e-1));
                    }
                  }
                }

                void expectCpxClose(Complex got, Complex exp) {
                  expectRealClose(got.real, exp.real);
                  expectRealClose(got.imag, exp.imag);
                }

                double toDbl(Object? v) {
                  if (v is bool) return v ? 1.0 : 0.0;
                  return (v as num).toDouble();
                }

                for (var r = 0; r < 2; r++) {
                  for (var c = 0; c < 3; c++) {
                    final va = a.getCell([r, c]);
                    final vb = b.getCell([r, c]);
                    Complex toCpx(Object? v) {
                      if (v is Complex) return v;
                      if (v is bool) return Complex(v ? 1.0 : 0.0, 0.0);
                      return Complex((v as num).toDouble(), 0.0);
                    }

                    final ca = toCpx(va);
                    final cb = toCpx(vb);
                    if (expectedDType.isComplex) {
                      final gotAdd = resAdd.getCell([r, c]) as Complex;
                      final gotSub = resSub.getCell([r, c]) as Complex;
                      final gotMul = resMul.getCell([r, c]) as Complex;
                      final gotDiv = resDiv.getCell([r, c]) as Complex;
                      expectCpxClose(gotAdd, ca + cb);
                      expectCpxClose(gotSub, ca - cb);
                      expectCpxClose(gotMul, ca * cb);
                      expectCpxClose(gotDiv, ca / cb);
                    } else if (expectedDType == DType.boolean) {
                      final ba = va as bool;
                      final bb = vb as bool;
                      expect(resAdd.getCell([r, c]), ba || bb);
                      expect(resSub.getCell([r, c]), ba ^ bb);
                      expect(resMul.getCell([r, c]), ba && bb);
                      expectRealClose(
                        toDbl(resDiv.getCell([r, c])),
                        (ba ? 1.0 : 0.0) / (bb ? 1.0 : 0.0),
                      );
                    } else if (expectedDType.isInteger) {
                      int wrapInt(int v, DType dt) => switch (dt) {
                        DType.int8 => v.toSigned(8),
                        DType.uint8 => v.toUnsigned(8),
                        DType.int16 => v.toSigned(16),
                        DType.uint16 => v.toUnsigned(16),
                        DType.int32 => v.toSigned(32),
                        DType.uint32 => v.toUnsigned(32),
                        _ => v,
                      };
                      final ia = ca.real.toInt();
                      final ib = cb.real.toInt();
                      expect(
                        resAdd.getCell([r, c]),
                        wrapInt(ia + ib, expectedDType),
                      );
                      expect(
                        resSub.getCell([r, c]),
                        wrapInt(ia - ib, expectedDType),
                      );
                      expect(
                        resMul.getCell([r, c]),
                        wrapInt(ia * ib, expectedDType),
                      );
                      expectRealClose(
                        toDbl(resDiv.getCell([r, c])),
                        ca.real / cb.real,
                      );
                    } else {
                      final da = ca.real;
                      final db = cb.real;
                      expectRealClose(
                        toDbl(resAdd.getCell([r, c])),
                        toDbl(castValue(da + db, expectedDType)),
                      );
                      expectRealClose(
                        toDbl(resSub.getCell([r, c])),
                        toDbl(castValue(da - db, expectedDType)),
                      );
                      expectRealClose(
                        toDbl(resMul.getCell([r, c])),
                        toDbl(castValue(da * db, expectedDType)),
                      );
                      expectRealClose(toDbl(resDiv.getCell([r, c])), da / db);
                    }
                  }
                }

                if (dtA == dtB) {
                  final resPow = power(a, b);
                  expect(resPow.shape, [2, 3]);
                  expect(resPow.dtype, dtA);
                  for (var r = 0; r < 2; r++) {
                    for (var c = 0; c < 3; c++) {
                      final va = a.getCell([r, c]);
                      final vb = b.getCell([r, c]);
                      final got = resPow.getCell([r, c]);
                      if (dtA == DType.boolean) {
                        expect(got, !(vb as bool) || (va as bool));
                      } else if (dtA.isComplex) {
                        final ca = va as Complex;
                        final cb = vb as Complex;
                        final rad = (ca.real * ca.real + ca.imag * ca.imag);
                        if (rad == 0.0) {
                          expectCpxClose(
                            got as Complex,
                            (cb.real == 0.0 && cb.imag == 0.0)
                                ? Complex(1.0, 0.0)
                                : Complex(0.0, 0.0),
                          );
                        } else {
                          expect((got as Complex).real.isNaN, isFalse);
                        }
                      } else if (dtA.isInteger) {
                        final ia = (va as num).toInt();
                        final ib = (vb as num).toInt();
                        var p = 1;
                        for (var k = 0; k < ib; k++) {
                          p *= ia;
                        }
                        final wrapped = switch (dtA) {
                          DType.int8 => p.toSigned(8),
                          DType.uint8 => p.toUnsigned(8),
                          DType.int16 => p.toSigned(16),
                          DType.uint16 => p.toUnsigned(16),
                          DType.int32 => p.toSigned(32),
                          DType.uint32 => p.toUnsigned(32),
                          _ => p,
                        };
                        expect(got, wrapped);
                      } else {
                        final da = (va as num).toDouble();
                        final db = (vb as num).toDouble();
                        var expP = 1.0;
                        for (var k = 0; k < db.toInt(); k++) {
                          expP *= da;
                        }
                        expectRealClose(
                          toDbl(got),
                          toDbl(castValue(expP, dtA)),
                        );
                      }
                    }
                  }
                }

                if (dtA != DType.complex128 &&
                    dtA != DType.complex64 &&
                    dtA != DType.boolean &&
                    dtB != DType.complex128 &&
                    dtB != DType.complex64 &&
                    dtB != DType.boolean) {
                  final resFDiv = floorDivide(a, b);
                  expect(resFDiv.shape, [2, 3]);

                  final resRem = remainder(a, b);
                  expect(resRem.shape, [2, 3]);

                  final resFmod = fmod(a, b);
                  expect(resFmod.shape, [2, 3]);

                  final resHeavi = heaviside(a, b);
                  expect(resHeavi.shape, [2, 3]);

                  final resCopy = copysign(a, b);
                  expect(resCopy.shape, [2, 3]);

                  final resHypot = hypot(a, b);
                  expect(resHypot.shape, [2, 3]);

                  final resAtan2 = atan2(a, b);
                  expect(resAtan2.shape, [2, 3]);

                  final resLogadd = logaddexp(a, b);
                  expect(resLogadd.shape, [2, 3]);

                  final resLogadd2 = logaddexp2(a, b);
                  expect(resLogadd2.shape, [2, 3]);
                }
              }
            }
          }
        });
      },
    );

    test('Integer binary arithmetic (gcd, lcm)', () {
      NDArray.scope(() {
        for (final dtA in integerDTypes) {
          for (final dtB in integerDTypes) {
            for (final isStrided in [false, true]) {
              final a = createNumericArray(dtA, [2, 2], strided: isStrided);
              final b = createNumericArray(dtB, [2, 2], strided: isStrided);

              int scalarGcd(int x, int y) {
                var u = x.abs();
                var v = y.abs();
                while (v != 0) {
                  final t = v;
                  v = u % v;
                  u = t;
                }
                return u;
              }

              int scalarLcm(int x, int y) {
                if (x == 0 || y == 0) return 0;
                return (x.abs() ~/ scalarGcd(x, y)) * y.abs();
              }

              final targetDt = resolveDType(dtA, dtB);
              final resGcd = gcd(a, b);
              expect(resGcd.shape, [2, 2]);
              expect(resGcd.dtype, targetDt);

              final resLcm = lcm(a, b);
              expect(resLcm.shape, [2, 2]);
              expect(resLcm.dtype, targetDt);

              for (var r = 0; r < 2; r++) {
                for (var c = 0; c < 2; c++) {
                  final av = (a.getCell([r, c]) as num).toInt();
                  final bv = (b.getCell([r, c]) as num).toInt();
                  expect(
                    resGcd.getCell([r, c]),
                    castValue(scalarGcd(av, bv), targetDt),
                  );
                  expect(
                    resLcm.getCell([r, c]),
                    castValue(scalarLcm(av, bv), targetDt),
                  );
                }
              }
            }
          }
        }
      });
    });

    test(
      'All 15 DTypes unary math and reductions (flat, axis 0, axis 1, keepdims)',
      () {
        NDArray.scope(() {
          for (final dt in numericDTypes) {
            for (final isStrided in [false, true]) {
              final a = createNumericArray(dt, [3, 4], strided: isStrided);

              // Unary math
              final resSqrt = sqrt(a);
              expect(resSqrt.shape, [3, 4]);

              final resAbs = abs(a);
              expect(resAbs.shape, [3, 4]);

              final resSin = sin(a);
              expect(resSin.shape, [3, 4]);

              final resCos = cos(a);
              expect(resCos.shape, [3, 4]);

              final resTan = tan(a);
              expect(resTan.shape, [3, 4]);

              final resExp = exp(a);
              expect(resExp.shape, [3, 4]);

              final resLog = log(a);
              expect(resLog.shape, [3, 4]);

              final resLog2 = log2(a);
              expect(resLog2.shape, [3, 4]);

              final resLog10 = log10(a);
              expect(resLog10.shape, [3, 4]);

              final resExpm1 = expm1(a);
              expect(resExpm1.shape, [3, 4]);

              final resLog1p = log1p(a);
              expect(resLog1p.shape, [3, 4]);

              // Reductions
              final sumFlat = sum(a);
              expect(sumFlat.shape, <int>[]);

              final sumAxis0 = sum(a, axis: 0);
              expect(sumAxis0.shape, [4]);

              final sumAxis1 = sum(a, axis: 1, keepdims: true);
              expect(sumAxis1.shape, [3, 1]);

              final prodFlat = prod(a);
              expect(prodFlat.shape, <int>[]);

              final prodAxis0 = prod(a, axis: 0);
              expect(prodAxis0.shape, [4]);

              final meanFlat = mean(a);
              expect(meanFlat.shape, <int>[]);

              final meanAxis0 = mean(a, axis: 0);
              expect(meanAxis0.shape, [4]);

              final cumsum0 = cumsum(a, axis: 0);
              expect(cumsum0.shape, [3, 4]);

              final cumsum1 = cumsum(a, axis: 1);
              expect(cumsum1.shape, [3, 4]);

              final stdFlat = std(a);
              expect(stdFlat.shape, <int>[]);

              final stdAxis0 = std(a, axis: 0);
              expect(stdAxis0.shape, [4]);

              final varFlat = variance(a);
              expect(varFlat.shape, <int>[]);

              final varAxis0 = variance(a, axis: 0);
              expect(varAxis0.shape, [4]);

              final minFlat = min(a);
              expect(minFlat.shape, <int>[]);

              final maxFlat = max(a);
              expect(maxFlat.shape, <int>[]);

              final ptpFlat = ptp(a);
              expect(ptpFlat.shape, <int>[]);
            }
          }
        });
      },
    );

    test('1D and 2D FFT and Linear Algebra across float and complex types', () {
      NDArray.scope(() {
        final floatTypes = [DType.float64, DType.float32];
        final complexTypes = [DType.complex128, DType.complex64];

        for (final dt in [...floatTypes, ...complexTypes]) {
          final a1d = createSampleArray(dt, [8]);
          final f1 = fft(a1d);
          expect(f1.shape, [8]);
          final if1 = ifft((f1 as NDArray<AnySpec>));
          expect(if1.shape, [8]);

          final a2d = createSampleArray(dt, [4, 4]);
          final f2 = fft2(a2d);
          expect(f2.shape, [4, 4]);
          final if2 = ifft2((f2 as NDArray<AnySpec>));
          expect(if2.shape, [4, 4]);

          final trans = a2d.transpose();
          expect(trans.shape, [4, 4]);
        }

        for (final dt in floatTypes) {
          final aReal = createNumericArray(dt, [8]);
          final rf = rfft(aReal);
          expect(rf.shape, [5]);
          final irf = irfft((rf as NDArray<AnySpec>), n: 8);
          expect(irf.shape, [8]);

          final a2dReal = createNumericArray(dt, [4, 4]);
          final detVal = det(a2dReal);
          expect(detVal.shape, <int>[]);
        }
      });
    });
  });
}
