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
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group(
    'Comprehensive Advanced Linear Algebra, Tensor Contractions & Financial Suite',
    () {
      // ------------------------------------------------------------------------
      // Group 1: Quantitative Financial Operations (fv, pv, npv, irr)
      // ------------------------------------------------------------------------
      group('1. Financial Operations (financial.dart)', () {
        group('Future Value (fv)', () {
          test('calculates fv with zero interest rate (fv_zero branch)', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.0, dtype: DType.float64);
              final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
              final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
              final pvVal = NDArray<Float64>.scalar(
                -1000.0,
                dtype: DType.float64,
              );

              final res = fv(rate, nper, pmt, pvVal);
              // fv_zero = - (pv + pmt * nper) = - (-1000 + (-100 * 10)) = 2000.0
              expect(res.scalar, closeTo(2000.0, 1e-9));
            });
          });

          test('calculates fv with non-zero rate and end/begin payments', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
              final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
              final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
              final pvVal = NDArray<Float64>.scalar(
                -1000.0,
                dtype: DType.float64,
              );

              final resEnd = fv(rate, nper, pmt, pvVal, when: PaymentDue.end);
              expect(resEnd.scalar, closeTo(2886.68388, 1e-4));

              final resBegin = fv(
                rate,
                nper,
                pmt,
                pvVal,
                when: PaymentDue.begin,
              );
              expect(resBegin.scalar, closeTo(2949.57334, 1e-4));
            });
          });

          test('supports PaymentDue enum for when parameter', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
              final nper = NDArray<Float64>.scalar(5.0, dtype: DType.float64);
              final pmt = NDArray<Float64>.scalar(-50.0, dtype: DType.float64);
              final pvVal = NDArray<Float64>.scalar(
                -500.0,
                dtype: DType.float64,
              );

              final resEnd = fv(rate, nper, pmt, pvVal, when: PaymentDue.end);
              final resBegin = fv(
                rate,
                nper,
                pmt,
                pvVal,
                when: PaymentDue.begin,
              );
              expect(resEnd.scalar, greaterThan(0.0));
              expect(resEnd.scalar, lessThan(resBegin.scalar));
            });
          });

          test('supports out parameter and multidimensional broadcasting', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.fromList(
                [0.05, 0.06],
                [2, 1],
                DType.float64,
              );
              final nper = NDArray<Float64>.fromList(
                [5.0, 10.0, 15.0],
                [1, 3],
                DType.float64,
              );
              final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
              final pvVal = NDArray<Float64>.scalar(0.0, dtype: DType.float64);

              final out = NDArray<Float64>.zeros([2, 3], DType.float64);
              final res = fv(rate, nper, pmt, pvVal, out: out);
              expect(identical(res, out), isTrue);
              expect(res.shape, [2, 3]);
              expect(res.getCell([0, 0]), closeTo(552.563125, 1e-4));
            });
          });

          test('throws on disposed inputs', () {
            final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
            final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
            final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
            final pvVal = NDArray<Float64>.scalar(
              -1000.0,
              dtype: DType.float64,
            );

            rate.dispose();
            expect(() => fv(rate, nper, pmt, pvVal), throwsStateError);
            nper.dispose();
            pmt.dispose();
            pvVal.dispose();
          });
        });

        group('Present Value (pv)', () {
          test('calculates pv with zero interest rate (pv_zero branch)', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.0, dtype: DType.float64);
              final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
              final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
              final fvVal = NDArray<Float64>.scalar(
                2000.0,
                dtype: DType.float64,
              );

              final res = pv(rate, nper, pmt, fvVal);
              // pv_zero = - (fv + pmt * nper) = - (2000 + (-100 * 10)) = -1000.0
              expect(res.scalar, closeTo(-1000.0, 1e-9));
            });
          });

          test('calculates pv with non-zero rate and end/begin payments', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
              final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
              final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
              final fvVal = NDArray<Float64>.scalar(
                2886.68388,
                dtype: DType.float64,
              );

              final resEnd = pv(rate, nper, pmt, fvVal, when: PaymentDue.end);
              expect(resEnd.scalar, closeTo(-1000.0, 1e-4));

              final resBegin = pv(
                rate,
                nper,
                pmt,
                fvVal,
                when: PaymentDue.begin,
              );
              expect(resBegin.scalar, closeTo(-961.3913, 1e-3));
            });
          });

          test('supports PaymentDue enum for when and out buffer', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
              final nper = NDArray<Float64>.scalar(5.0, dtype: DType.float64);
              final pmt = NDArray<Float64>.scalar(-50.0, dtype: DType.float64);
              final fvVal = NDArray<Float64>.scalar(
                1000.0,
                dtype: DType.float64,
              );

              final out = NDArray<Float64>.zeros([], DType.float64);
              final res = pv(
                rate,
                nper,
                pmt,
                fvVal,
                when: PaymentDue.begin,
                out: out,
              );
              expect(identical(res, out), isTrue);
            });
          });

          test('throws on disposed inputs', () {
            final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
            final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
            final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
            final fvVal = NDArray<Float64>.scalar(1000.0, dtype: DType.float64);

            fvVal.dispose();
            expect(() => pv(rate, nper, pmt, fvVal), throwsStateError);
            rate.dispose();
            nper.dispose();
            pmt.dispose();
          });
        });

        group('Net Present Value (npv)', () {
          test('calculates npv for 1D cash flows', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.281, dtype: DType.float64);
              final values = NDArray<Float64>.fromList(
                [-100.0, 39.0, 59.0, 55.0, 20.0],
                [5],
                DType.float64,
              );

              final res = npv(rate, values);
              expect(res.shape, <int>[]);
              expect(res.scalar, closeTo(-0.008, 1e-2));
            });
          });

          test(
            'calculates npv with multidimensional broadcasting and out parameter',
            () {
              NDArray.scope(() {
                final rate = NDArray<Float64>.fromList(
                  [0.05, 0.10],
                  [2],
                  DType.float64,
                );
                final values = NDArray<Float64>.fromList(
                  [-1000.0, 300.0, 400.0, 500.0, -2000.0, 600.0, 800.0, 1000.0],
                  [2, 4],
                  DType.float64,
                );

                final out = NDArray<Float64>.zeros([2, 2], DType.float64);
                final res = npv(rate, values, out: out);
                expect(identical(res, out), isTrue);
                expect(res.shape, [2, 2]);
              });
            },
          );

          test('throws for 0D values and disposed arrays', () {
            NDArray.scope(() {
              final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
              final scalarValues = NDArray<Float64>.scalar(
                100.0,
                dtype: DType.float64,
              );
              expect(() => npv(rate, scalarValues), throwsArgumentError);
            });

            final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
            final values = NDArray<Float64>.fromList(
              [100.0, 200.0],
              [2],
              DType.float64,
            );
            values.dispose();
            expect(() => npv(rate, values), throwsStateError);
            rate.dispose();
          });
        });

        group('Internal Rate of Return (irr)', () {
          test('calculates irr for standard cash flows', () {
            NDArray.scope(() {
              final values = NDArray<Float64>.fromList(
                [-100.0, 39.0, 59.0, 55.0, 20.0],
                [5],
                DType.float64,
              );

              final res = irr(values);
              expect(res.shape, <int>[]);
              expect(res.scalar, closeTo(0.28095, 1e-4));
            });
          });

          test('handles leading zeros by stripping to slice view', () {
            NDArray.scope(() {
              final values = NDArray<Float64>.fromList(
                [0.0, 0.0, -100.0, 39.0, 59.0, 55.0, 20.0],
                [7],
                DType.float64,
              );

              final res = irr(values);
              expect(res.scalar, closeTo(0.28095, 1e-4));
            });
          });

          test(
            'returns NaN or throws NoRealSolutionException for invalid cash flow signs',
            () {
              NDArray.scope(() {
                // All zeros
                final allZeros = NDArray<Float64>.fromList(
                  [0.0, 0.0, 0.0],
                  [3],
                  DType.float64,
                );
                final resZeros = irr(allZeros);
                expect(resZeros.scalar.isNaN, isTrue);
                expect(
                  () => irr(allZeros, raiseExceptions: true),
                  throwsA(isA<NoRealSolutionException>()),
                );

                // All positive
                final allPos = NDArray<Float64>.fromList(
                  [10.0, 20.0, 30.0],
                  [3],
                  DType.float64,
                );
                final resPos = irr(allPos);
                expect(resPos.scalar.isNaN, isTrue);
                expect(
                  () => irr(allPos, raiseExceptions: true),
                  throwsA(isA<NoRealSolutionException>()),
                );

                // All negative
                final allNeg = NDArray<Float64>.fromList(
                  [-10.0, -20.0, -30.0],
                  [3],
                  DType.float64,
                );
                final resNeg = irr(allNeg);
                expect(resNeg.scalar.isNaN, isTrue);
                expect(
                  () => irr(allNeg, raiseExceptions: true),
                  throwsA(isA<NoRealSolutionException>()),
                );

                // Single non-zero
                final singleNonZero = NDArray<Float64>.fromList(
                  [0.0, 100.0],
                  [2],
                  DType.float64,
                );
                expect(irr(singleNonZero).scalar.isNaN, isTrue);
              });
            },
          );

          test(
            'multi-root cash flows exercise _irrDefaultSelection branches',
            () {
              NDArray.scope(() {
                // Multi-sign changes leading to multiple roots
                final values = NDArray<Float64>.fromList(
                  [-100.0, 230.0, -132.0],
                  [3],
                  DType.float64,
                );
                final res = irr(values);
                expect(res.scalar, isNotNull);
                expect(res.scalar.isNaN, isFalse);
              });
            },
          );

          test(
            'supports out parameter and validates 1D rank and disposed checks',
            () {
              NDArray.scope(() {
                final values = NDArray<Float64>.fromList(
                  [-50.0, 60.0],
                  [2],
                  DType.float64,
                );
                final out = NDArray<Float64>.create([], DType.float64);
                final res = irr(values, out: out);
                expect(identical(res, out), isTrue);
                expect(res.scalar, closeTo(0.2, 1e-6));

                final matrix2D = NDArray<Float64>.zeros([2, 2], DType.float64);
                expect(() => irr(matrix2D), throwsArgumentError);
              });

              final values = NDArray<Float64>.fromList(
                [-50.0, 60.0],
                [2],
                DType.float64,
              );
              values.dispose();
              expect(() => irr(values), throwsStateError);
            },
          );
        });
      });

      // ------------------------------------------------------------------------
      // Group 2: Matrix Powers & Advanced Solvers (linalg.dart)
      // ------------------------------------------------------------------------
      group('2. Matrix Powers, Decompositions & Solvers', () {
        group('matrix_power', () {
          test('matrix_power n = 0 returns identity matrix', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [2.0, 3.0, 4.0, 5.0],
                [2, 2],
                DType.float64,
              );
              final res = matrix_power(a, 0);
              expect(res.shape, [2, 2]);
              expect(res.getCell([0, 0]), 1.0);
              expect(res.getCell([0, 1]), 0.0);
              expect(res.getCell([1, 0]), 0.0);
              expect(res.getCell([1, 1]), 1.0);

              final out = NDArray<Float64>.zeros([2, 2], DType.float64);
              final resOut = matrix_power(a, 0, out: out);
              expect(identical(resOut, out), isTrue);
              expect(out.getCell([0, 0]), 1.0);
            });
          });

          test('matrix_power n = 1 returns copy of matrix', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [2.0, 3.0, 4.0, 5.0],
                [2, 2],
                DType.float64,
              );
              final res = matrix_power(a, 1);
              expect(res.toList(), [2.0, 3.0, 4.0, 5.0]);

              final out = NDArray<Float64>.zeros([2, 2], DType.float64);
              matrix_power(a, 1, out: out);
              expect(out.toList(), [2.0, 3.0, 4.0, 5.0]);
            });
          });

          test(
            'matrix_power n > 1 computes powers via binary exponentiation',
            () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [2, 2],
                  DType.float64,
                );
                // A^2 = [[7, 10], [15, 22]]
                final a2 = matrix_power(a, 2);
                expect(a2.toList(), [7.0, 10.0, 15.0, 22.0]);

                // A^3 = A^2 * A = [[37, 54], [81, 118]]
                final a3 = matrix_power(a, 3);
                expect(a3.toList(), [37.0, 54.0, 81.0, 118.0]);

                // A^5
                final a5 = matrix_power(a, 5);
                expect(a5.shape, [2, 2]);
                expect(a5.getCell([0, 0]), 1069.0);
              });
            },
          );

          test(
            'matrix_power n < 0 computes negative power via matrix inversion',
            () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [4.0, 7.0, 2.0, 6.0],
                  [2, 2],
                  DType.float64,
                );
                // A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
                final aInv = matrix_power(a, -1);
                expect(aInv.getCell([0, 0]), closeTo(0.6, 1e-9));
                expect(aInv.getCell([0, 1]), closeTo(-0.7, 1e-9));
                expect(aInv.getCell([1, 0]), closeTo(-0.2, 1e-9));
                expect(aInv.getCell([1, 1]), closeTo(0.4, 1e-9));

                // A^-2 = (A^-1)^2
                final aInv2 = matrix_power(a, -2);
                final directInv2 = matmul(aInv, aInv);
                for (var i = 0; i < 2; i++) {
                  for (var j = 0; j < 2; j++) {
                    expect(
                      aInv2.getCell([i, j]),
                      closeTo(directInv2.getCell([i, j]), 1e-9),
                    );
                  }
                }
              });
            },
          );

          test('matrix_power complex matrices and error checks', () {
            NDArray.scope(() {
              final ac = NDArray.fromList(
                [
                  Complex(0.0, 1.0),
                  Complex(1.0, 0.0),
                  Complex(1.0, 0.0),
                  Complex(0.0, -1.0),
                ],
                [2, 2],
                DType.complex128,
              );
              final ac2 = matrix_power(ac, 2);
              expect(ac2.dtype, DType.complex128);

              final intMat = NDArray.fromList(
                [1, 2, 3, 4],
                [2, 2],
                DType.int32,
              );
              final intPow2 = matrix_power(intMat, 2);
              expect(intPow2.toList(), [7, 10, 15, 22]);

              // Negative power on integer matrix throws ArgumentError
              expect(() => matrix_power(intMat, -1), throwsArgumentError);

              // Non-square matrix throws ArgumentError
              final nonSquare = NDArray.zeros([2, 3], DType.float64);
              expect(() => matrix_power(nonSquare, 2), throwsArgumentError);

              // 1D array throws ArgumentError
              final vec1D = NDArray.zeros([4], DType.float64);
              expect(() => matrix_power(vec1D, 2), throwsArgumentError);
            });
          });
        });

        group('pinv (Moore-Penrose Pseudo-Inverse)', () {
          test(
            'pinv for tall (M > N), square (M == N), and wide (M < N) matrices',
            () {
              NDArray.scope(() {
                // Square 2x2
                final sq = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [2, 2],
                  DType.float64,
                );
                final sqPinv = pinv(sq);
                expect(sqPinv.shape, [2, 2]);
                final ident = matmul(sq, sqPinv);
                expect(ident.getCell([0, 0]), closeTo(1.0, 1e-10));
                expect(ident.getCell([1, 1]), closeTo(1.0, 1e-10));
                expect(ident.getCell([0, 1]), closeTo(0.0, 1e-10));
                expect(ident.getCell([1, 0]), closeTo(0.0, 1e-10));

                // Tall 3x2
                final tall = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                  [3, 2],
                  DType.float64,
                );
                final tallPinv = pinv(tall);
                expect(tallPinv.shape, [2, 3]);
                // Identity: A * A^+ * A = A
                final recTall = matmul(tall, matmul(tallPinv, tall));
                for (var i = 0; i < 3; i++) {
                  for (var j = 0; j < 2; j++) {
                    expect(
                      recTall.getCell([i, j]),
                      closeTo(tall.getCell([i, j]), 1e-9),
                    );
                  }
                }

                // Wide 2x3
                final wide = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                  [2, 3],
                  DType.float64,
                );
                final widePinv = pinv(wide);
                expect(widePinv.shape, [3, 2]);
                final recWide = matmul(wide, matmul(widePinv, wide));
                for (var i = 0; i < 2; i++) {
                  for (var j = 0; j < 3; j++) {
                    expect(
                      recWide.getCell([i, j]),
                      closeTo(wide.getCell([i, j]), 1e-9),
                    );
                  }
                }
              });
            },
          );

          test(
            'pinv with Float32, Complex128, custom rcond, and zero-dimension matrices',
            () {
              NDArray.scope(() {
                final f32 = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [2, 2],
                  DType.float32,
                );
                final f32Pinv = pinv(f32, rcond: 1e-5);
                expect(f32Pinv.dtype, DType.float32);

                final c128 = NDArray.fromList(
                  [
                    Complex(1.0, 1.0),
                    Complex(2.0, 0.0),
                    Complex(0.0, 1.0),
                    Complex(3.0, -2.0),
                  ],
                  [2, 2],
                  DType.complex128,
                );
                final c128Pinv = pinv(c128);
                expect(c128Pinv.dtype, DType.complex128);

                // Zero-dimension matrix
                final zeroMat = NDArray<Float64>.zeros([0, 3], DType.float64);
                final zeroPinv = pinv(zeroMat);
                expect(zeroPinv.shape, [3, 0]);

                // Non-2D throws ArgumentError
                final vec1D = NDArray.zeros([5], DType.float64);
                expect(() => pinv(vec1D), throwsArgumentError);
              });
            },
          );
        });

        group('lstsq (Least Squares Solver)', () {
          test(
            'lstsq solves overdetermined system (M > N) with 1D and 2D RHS',
            () {
              NDArray.scope(() {
                // A: 4x2, B: 4 (1D)
                final a = NDArray.fromList(
                  [0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0],
                  [4, 2],
                  DType.float64,
                );
                final b = NDArray.fromList(
                  [-1.0, 0.2, 0.9, 2.1],
                  [4],
                  DType.float64,
                );

                final res = lstsq(a, b);
                expect(res.x.shape, [2]);
                expect(res.x.getCell([0]), closeTo(1.0, 1e-2));
                expect(res.x.getCell([1]), closeTo(-0.95, 1e-2));
                expect(res.rank, 2);
                expect(res.s.shape, [2]);
                expect(res.residuals.shape, [1]);
                expect(res.residuals.getCell([0]), greaterThan(0.0));

                // B: 4x2 (2D multiple RHS)
                final b2D = NDArray.fromList(
                  [-1.0, 1.0, 0.2, 2.0, 0.9, 3.0, 2.1, 4.0],
                  [4, 2],
                  DType.float64,
                );
                final res2D = lstsq(a, b2D);
                expect(res2D.x.shape, [2, 2]);
                expect(res2D.residuals.shape, [2]);
              });
            },
          );

          test(
            'lstsq with Float32, Complex128, Complex64, and zero-dimension systems',
            () {
              NDArray.scope(() {
                final aF32 = NDArray.fromList(
                  [1.0, 1.0, 1.0, -1.0],
                  [2, 2],
                  DType.float32,
                );
                final bF32 = NDArray.fromList([2.0, 0.0], [2], DType.float32);
                final resF32 = lstsq(aF32, bF32);
                expect(resF32.x.dtype, DType.float32);
                expect(resF32.x.getCell([0]), closeTo(1.0, 1e-4));
                expect(resF32.x.getCell([1]), closeTo(1.0, 1e-4));

                final aC128 = NDArray.fromList(
                  [
                    Complex(1.0, 0.0),
                    Complex(0.0, 1.0),
                    Complex(0.0, -1.0),
                    Complex(1.0, 0.0),
                  ],
                  [2, 2],
                  DType.complex128,
                );
                final bC128 = NDArray.fromList(
                  [Complex(1.0, 1.0), Complex(1.0, -1.0)],
                  [2],
                  DType.complex128,
                );
                final resC128 = lstsq(aC128, bC128);
                expect(resC128.x.dtype, DType.complex128);

                final aC64 = NDArray.fromList(
                  [
                    Complex(2.0, 0.0),
                    Complex(0.0, 0.0),
                    Complex(0.0, 0.0),
                    Complex(2.0, 0.0),
                  ],
                  [2, 2],
                  DType.complex64,
                );
                final bC64 = NDArray.fromList(
                  [Complex(4.0, 2.0), Complex(6.0, -2.0)],
                  [2],
                  DType.complex64,
                );
                final resC64 = lstsq(aC64, bC64);
                expect(resC64.x.dtype, DType.complex64);

                // Zero dimension
                final aZero = NDArray<Float64>.zeros([0, 2], DType.float64);
                final bZero = NDArray<Float64>.zeros([0], DType.float64);
                final resZero = lstsq(aZero, bZero);
                expect(resZero.x.shape, [2]);
                expect(resZero.rank, 0);

                // Dispose extension check
                final standaloneRes = lstsq(aF32, bF32);
                standaloneRes.dispose();
              });
            },
          );
        });

        group('cholesky Decomposition', () {
          test('cholesky 2D, 3D batch, and 4D batch decomposition', () {
            NDArray.scope(() {
              // 2D Positive Definite
              final a = NDArray.fromList(
                [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
                [3, 3],
                DType.float64,
              );
              final l = cholesky(a);
              expect(l.shape, [3, 3]);
              final lLt = matmul(l, l.transpose());
              for (var i = 0; i < 3; i++) {
                for (var j = 0; j < 3; j++) {
                  expect(
                    lLt.getCell([i, j]),
                    closeTo(a.getCell([i, j]), 1e-10),
                  );
                }
              }

              // 3D Batch Cholesky
              final a3D = NDArray.fromList(
                [
                  // Batch 0
                  4.0, 2.0, 2.0, 3.0,
                  // Batch 1
                  9.0, 3.0, 3.0, 2.0,
                ],
                [2, 2, 2],
                DType.float64,
              );
              final l3D = cholesky(a3D);
              expect(l3D.shape, [2, 2, 2]);

              // 4D Batch Cholesky with non-contiguous strided slice view
              final a4D = NDArray<Float64>.zeros([2, 2, 2, 2], DType.float64);
              for (var b1 = 0; b1 < 2; b1++) {
                for (var b2 = 0; b2 < 2; b2++) {
                  a4D.setCell([b1, b2, 0, 0], 4.0);
                  a4D.setCell([b1, b2, 0, 1], 1.0);
                  a4D.setCell([b1, b2, 1, 0], 1.0);
                  a4D.setCell([b1, b2, 1, 1], 4.0);
                }
              }
              final l4D = cholesky(a4D);
              expect(l4D.shape, [2, 2, 2, 2]);

              // Non-positive definite matrix throws NonPositiveDefiniteException
              final notPD = NDArray.fromList(
                [-1.0, 0.0, 0.0, -1.0],
                [2, 2],
                DType.float64,
              );
              expect(
                () => cholesky(notPD),
                throwsA(isA<NonPositiveDefiniteException>()),
              );

              // Complex Hermitian Positive Definite Cholesky
              final cpd = NDArray.fromList(
                [
                  Complex(2.0, 0.0),
                  Complex(0.0, 1.0),
                  Complex(0.0, -1.0),
                  Complex(2.0, 0.0),
                ],
                [2, 2],
                DType.complex128,
              );
              final lC = cholesky(cpd);
              expect(lC.dtype, DType.complex128);
            });
          });
        });

        group('qr and svd Decompositions', () {
          test('qr on tall, square, wide, and batch 3D/4D matrices', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
                [3, 3],
                DType.float64,
              );
              final res = qr(a);
              expect(res.q.shape, [3, 3]);
              expect(res.r.shape, [3, 3]);

              // Orthogonality Q^T Q = I
              final qTq = matmul(res.q.transpose(), res.q);
              expect(qTq.getCell([0, 0]), closeTo(1.0, 1e-10));
              expect(qTq.getCell([1, 1]), closeTo(1.0, 1e-10));
              expect(qTq.getCell([2, 2]), closeTo(1.0, 1e-10));
              expect(qTq.getCell([0, 1]), closeTo(0.0, 1e-10));

              // Reconstruction Q R = A
              final qrProd = matmul(res.q, res.r);
              for (var i = 0; i < 3; i++) {
                for (var j = 0; j < 3; j++) {
                  expect(
                    qrProd.getCell([i, j]),
                    closeTo(a.getCell([i, j]), 1e-9),
                  );
                }
              }

              // Batch 3D QR
              final batch3D = NDArray<Float64>.zeros([2, 3, 2], DType.float64);
              for (var b = 0; b < 2; b++) {
                for (var i = 0; i < 3; i++) {
                  for (var j = 0; j < 2; j++) {
                    batch3D.setCell([b, i, j], (i + j + b + 1).toDouble());
                  }
                }
              }
              final res3D = qr(batch3D);
              expect(res3D.q.shape, [2, 3, 2]);
              expect(res3D.r.shape, [2, 2, 2]);
            });
          });

          test('svd on tall, wide, and batch matrices', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [3, 2],
                DType.float64,
              );
              final res = svd(a);
              expect(res.u.shape, [3, 3]);
              expect(res.s.shape, [2]);
              expect(res.vh.shape, [2, 2]);

              // Singular values are in descending order
              expect(
                res.s.getCell([0]),
                greaterThanOrEqualTo(res.s.getCell([1])),
              );

              // Wide 2x3 SVD
              final wide = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2, 3],
                DType.float64,
              );
              final resWide = svd(wide);
              expect(resWide.u.shape, [2, 2]);
              expect(resWide.s.shape, [2]);
              expect(resWide.vh.shape, [3, 3]);

              // Batch 3D SVD
              final batchA = NDArray<Float64>.zeros([2, 3, 2], DType.float64);
              for (var b = 0; b < 2; b++) {
                for (var i = 0; i < 3; i++) {
                  for (var j = 0; j < 2; j++) {
                    batchA.setCell([b, i, j], (i * 2 + j + 1).toDouble());
                  }
                }
              }
              final resBatch = svd(batchA);
              expect(resBatch.u.shape, [2, 3, 3]);
              expect(resBatch.s.shape, [2, 2]);
              expect(resBatch.vh.shape, [2, 2, 2]);
            });
          });
        });

        group('eig, eigvals, eigh, eigvalsh, schur, and hessenberg', () {
          test('eig and eigvals for real 2D matrix', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 2.0, 1.0],
                [2, 2],
                DType.float64,
              );
              final res = eig(a);
              expect(res.eigenvalues.shape, [2]);
              expect(res.eigenvectors.shape, [2, 2]);

              final valsOnly = eigvals(a);
              expect(valsOnly.shape, [2]);
            });
          });

          test('eigh and eigvalsh for symmetric matrix', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [2.0, -1.0, -1.0, 2.0],
                [2, 2],
                DType.float64,
              );
              final res = eigh(a);
              expect(res.eigenvalues.shape, [2]);
              expect(res.eigenvectors.shape, [2, 2]);

              final vals = eigvalsh(a, uplo: MatrixTriangle.upper);
              expect(vals.shape, [2]);
              expect(vals.getCell([0]), closeTo(1.0, 1e-9));
              expect(vals.getCell([1]), closeTo(3.0, 1e-9));
            });
          });

          test('schur and hessenberg decompositions', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              final sRes = schur(a, output: SchurForm.real);
              expect(sRes.t.shape, [2, 2]);
              expect(sRes.z.shape, [2, 2]);

              final hRes = hessenberg(a);
              expect(hRes.h.shape, [2, 2]);
              expect(hRes.q.shape, [2, 2]);
            });
          });
        });
      });

      // ------------------------------------------------------------------------
      // Group 3: Matrix Inversion, Batch Solvers, Norms & Multi-Dot
      // ------------------------------------------------------------------------
      group('3. Inversion, Batch Solvers, Norms & Multi-Dot (linalg.dart)', () {
        group('inv and solve with 3D/4D Batches and Strided Views', () {
          test('inv 2D, 3D, and 4D stacked batch inversion', () {
            NDArray.scope(() {
              // 2D Inversion
              final a2D = NDArray.fromList(
                [4.0, 7.0, 2.0, 6.0],
                [2, 2],
                DType.float64,
              );
              final inv2D = inv(a2D);
              expect(inv2D.shape, [2, 2]);
              final eye2D = matmul(a2D, inv2D);
              expect(eye2D.getCell([0, 0]), closeTo(1.0, 1e-10));
              expect(eye2D.getCell([1, 1]), closeTo(1.0, 1e-10));
              expect(eye2D.getCell([0, 1]), closeTo(0.0, 1e-10));

              // 3D Stacked Batch Inversion
              final a3D = NDArray<Float64>.zeros([3, 2, 2], DType.float64);
              for (var b = 0; b < 3; b++) {
                a3D.setCell([b, 0, 0], (b + 1) * 2.0);
                a3D.setCell([b, 0, 1], 1.0);
                a3D.setCell([b, 1, 0], 1.0);
                a3D.setCell([b, 1, 1], 3.0);
              }
              final inv3D = inv(a3D);
              expect(inv3D.shape, [3, 2, 2]);
              final eye3D = matmul(a3D, inv3D);
              for (var b = 0; b < 3; b++) {
                expect(eye3D.getCell([b, 0, 0]), closeTo(1.0, 1e-9));
                expect(eye3D.getCell([b, 1, 1]), closeTo(1.0, 1e-9));
              }

              // 4D Stacked Batch Inversion
              final a4D = NDArray<Float64>.zeros([2, 2, 2, 2], DType.float64);
              for (var i = 0; i < 2; i++) {
                for (var j = 0; j < 2; j++) {
                  a4D.setCell([i, j, 0, 0], 3.0);
                  a4D.setCell([i, j, 0, 1], 1.0);
                  a4D.setCell([i, j, 1, 0], 1.0);
                  a4D.setCell([i, j, 1, 1], 2.0);
                }
              }
              final inv4D = inv(a4D);
              expect(inv4D.shape, [2, 2, 2, 2]);
            });
          });

          test('det and slogdet on 2D and 3D batch arrays', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              final d = det(a);
              expect(d.shape, <int>[]);
              expect(d.scalar, closeTo(-2.0, 1e-9));

              final sld = slogdet(a);
              expect(sld.sign.shape, <int>[]);
              expect(sld.logabsdet.shape, <int>[]);
              expect(sld.sign.scalar, closeTo(-1.0, 1e-9));
              expect(sld.logabsdet.scalar, closeTo(math.log(2.0), 1e-9));

              // 3D batch det
              final a3D = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
              a3D.setCell([0, 0, 0], 2.0);
              a3D.setCell([0, 1, 1], 3.0);
              a3D.setCell([1, 0, 0], 4.0);
              a3D.setCell([1, 1, 1], 5.0);
              final d3D = det(a3D);
              expect(d3D.shape, [2]);
              expect(d3D.getCell([0]), closeTo(6.0, 1e-9));
              expect(d3D.getCell([1]), closeTo(20.0, 1e-9));
            });
          });

          test('solve on 2D, 3D, and 4D batch linear systems', () {
            NDArray.scope(() {
              // 2D solve: A x = b
              final a = NDArray.fromList(
                [3.0, 1.0, 1.0, 2.0],
                [2, 2],
                DType.float64,
              );
              final b = NDArray.fromList([9.0, 8.0], [2], DType.float64);
              final x = solve(a, b);
              expect(x.shape, [2]);
              expect(x.getCell([0]), closeTo(2.0, 1e-9));
              expect(x.getCell([1]), closeTo(3.0, 1e-9));

              // 3D batch solve: A: [2, 2, 2], b: [2, 2, 1]
              final a3D = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
              final b3D = NDArray<Float64>.zeros([2, 2, 1], DType.float64);

              a3D.setCell([0, 0, 0], 3.0);
              a3D.setCell([0, 0, 1], 1.0);
              a3D.setCell([0, 1, 0], 1.0);
              a3D.setCell([0, 1, 1], 2.0);
              b3D.setCell([0, 0, 0], 9.0);
              b3D.setCell([0, 1, 0], 8.0);

              a3D.setCell([1, 0, 0], 2.0);
              a3D.setCell([1, 0, 1], 0.0);
              a3D.setCell([1, 1, 0], 0.0);
              a3D.setCell([1, 1, 1], 4.0);
              b3D.setCell([1, 0, 0], 6.0);
              b3D.setCell([1, 1, 0], 8.0);

              final x3D = solve(a3D, b3D);
              expect(x3D.shape, [2, 2, 1]);
              expect(x3D.getCell([0, 0, 0]), closeTo(2.0, 1e-9));
              expect(x3D.getCell([0, 1, 0]), closeTo(3.0, 1e-9));
              expect(x3D.getCell([1, 0, 0]), closeTo(3.0, 1e-9));
              expect(x3D.getCell([1, 1, 0]), closeTo(2.0, 1e-9));
            });
          });
        });

        group('norm across matrix and vector orders', () {
          test('vector norms across all orders and NormKind enum', () {
            NDArray.scope(() {
              final v = NDArray.fromList([3.0, -4.0], [2], DType.float64);

              // 2-norm (default)
              expect(norm(v).scalar, closeTo(5.0, 1e-9));
              expect(norm(v, ord: 2).scalar, closeTo(5.0, 1e-9));
              expect(norm(v, ord: NormKind.l2).scalar, closeTo(5.0, 1e-9));

              // 1-norm
              expect(norm(v, ord: 1).scalar, closeTo(7.0, 1e-9));
              expect(norm(v, ord: NormKind.l1).scalar, closeTo(7.0, 1e-9));

              // Infinity norms
              expect(norm(v, ord: double.infinity).scalar, closeTo(4.0, 1e-9));
              expect(
                norm(v, ord: NormKind.infinity).scalar,
                closeTo(4.0, 1e-9),
              );
              expect(
                norm(v, ord: double.negativeInfinity).scalar,
                closeTo(3.0, 1e-9),
              );
              expect(
                norm(v, ord: NormKind.negInfinity).scalar,
                closeTo(3.0, 1e-9),
              );

              // 0-norm (count of non-zero elements)
              expect(norm(v, ord: 0).scalar, closeTo(2.0, 1e-9));

              // Lp norm (p = 3.0)
              final l3 = math
                  .pow(math.pow(3.0, 3) + math.pow(4.0, 3), 1.0 / 3.0)
                  .toDouble();
              expect(norm(v, ord: 3.0).scalar, closeTo(l3, 1e-9));

              // Invalid vector norm orders
              expect(
                () => norm(v, ord: NormKind.frobenius),
                throwsArgumentError,
              );
              expect(() => norm(v, ord: NormKind.nuclear), throwsArgumentError);
            });
          });

          test('matrix norms across all matrix orders', () {
            NDArray.scope(() {
              final m = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );

              // Frobenius norm (default)
              final frob = math.sqrt(1 + 4 + 9 + 16);
              expect(norm(m).scalar, closeTo(frob, 1e-9));
              expect(
                norm(m, ord: NormKind.frobenius).scalar,
                closeTo(frob, 1e-9),
              );

              // 1-norm (max column sum: max(1+3, 2+4) = 6)
              expect(norm(m, ord: 1).scalar, closeTo(6.0, 1e-9));

              // -1-norm (min column sum: min(1+3, 2+4) = 4)
              expect(norm(m, ord: -1).scalar, closeTo(4.0, 1e-9));

              // Infinity norm (max row sum: max(1+2, 3+4) = 7)
              expect(norm(m, ord: double.infinity).scalar, closeTo(7.0, 1e-9));

              // -Infinity norm (min row sum: min(1+2, 3+4) = 3)
              expect(
                norm(m, ord: double.negativeInfinity).scalar,
                closeTo(3.0, 1e-9),
              );

              // 2-norm (spectral norm = largest singular value)
              final svdRes = svd(m);
              final maxSingular = svdRes.s.getCell([0]);
              expect(norm(m, ord: 2).scalar, closeTo(maxSingular, 1e-9));

              // -2-norm (smallest singular value)
              final minSingular = svdRes.s.getCell([1]);
              expect(norm(m, ord: -2).scalar, closeTo(minSingular, 1e-9));

              // Nuclear norm (sum of singular values)
              expect(
                norm(m, ord: NormKind.nuclear).scalar,
                closeTo(maxSingular + minSingular, 1e-9),
              );
            });
          });

          test('norm along specific axes and keepdims', () {
            NDArray.scope(() {
              final a3D = NDArray<Float64>.zeros([2, 3, 4], DType.float64);
              a3D.fill(1.0);

              // Vector norm along axis 1
              final normAx1 = norm(a3D, axis: 1);
              expect(normAx1.shape, [2, 4]);
              expect(normAx1.getCell([0, 0]), closeTo(math.sqrt(3.0), 1e-9));

              // Keepdims
              final normKeep = norm(a3D, axis: 1, keepdims: true);
              expect(normKeep.shape, [2, 1, 4]);

              // Matrix norm along axes [1, 2]
              final normMat = norm(a3D, axis: [1, 2]);
              expect(normMat.shape, [2]);
              expect(normMat.getCell([0]), closeTo(math.sqrt(12.0), 1e-9));
            });
          });
        });

        group('multi_dot and chain parenthesization', () {
          test(
            'multi_dot computes optimal chain product of 3, 4, and 5 matrices',
            () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                  [2, 3],
                  DType.float64,
                );
                final b = NDArray.fromList(
                  [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                  [3, 2],
                  DType.float64,
                );
                final c = NDArray.fromList(
                  [2.0, 3.0, 4.0, 5.0],
                  [2, 2],
                  DType.float64,
                );

                final res3 = multi_dot([a, b, c]);
                expect(res3.shape, [2, 2]);
                final expected3 = matmul(matmul(a, b), c);
                expect(res3.toList(), expected3.toList());

                // 4 matrices
                final d = NDArray.fromList(
                  [1.0, -1.0, 0.0, 2.0],
                  [2, 2],
                  DType.float64,
                );
                final res4 = multi_dot([a, b, c, d]);
                expect(res4.shape, [2, 2]);
                final expected4 = matmul(expected3, d);
                expect(res4.toList(), expected4.toList());
              });
            },
          );

          test('multi_dot with 1D vector edge projections', () {
            NDArray.scope(() {
              final v1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
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
              final v2 = NDArray.fromList([3.0, 1.0], [2], DType.float64);

              // 1D at start: (1D x 2D x 2D -> 1D)
              final resStart = multi_dot([v1, m1, m2]);
              expect(resStart.shape, [2]);

              // 1D at end: (2D x 2D x 1D -> 1D)
              final resEnd = multi_dot([m1, m2, v2]);
              expect(resEnd.shape, [2]);

              // 1D at both ends: (1D x 2D x 2D x 1D -> scalar 0D)
              final resBoth = multi_dot([v1, m1, m2, v2]);
              expect(resBoth.shape, <int>[]);
            });
          });
        });

        group('outer and cross products', () {
          test('outer product across all 15 DTypes', () {
            NDArray.scope(() {
              final dtypes = <DType<AnySpec>>[
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

              for (final dt in dtypes) {
                final a = dt.isComplex
                    ? NDArray.fromList(
                        [Complex(1.0, 2.0), Complex(3.0, 4.0)],
                        [2],
                        dt,
                      )
                    : dt == DType.boolean
                    ? NDArray.fromList([true, false], [2], dt)
                    : NDArray.fromList([1.0, 2.0], [2], dt);
                final b = dt.isComplex
                    ? NDArray.fromList(
                        [Complex(2.0, 0.0), Complex(0.0, 1.0)],
                        [2],
                        dt,
                      )
                    : dt == DType.boolean
                    ? NDArray.fromList([false, true], [2], dt)
                    : NDArray.fromList([3.0, 4.0], [2], dt);

                final res = outer(a, b);
                expect(res.shape, [2, 2], reason: 'Failed for DType $dt');
              }
            });
          });

          test(
            'cross product for 2D and 3D vectors along axes and across dtypes',
            () {
              NDArray.scope(() {
                // 2D vectors: (a_x b_y - a_y b_x)
                final a2 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
                final b2 = NDArray.fromList([3.0, 4.0], [2], DType.float64);
                final res2D = cross(a2, b2);
                // 1*4 - 2*3 = -2
                expect(res2D.shape, <int>[]);
                expect(res2D.scalar, closeTo(-2.0, 1e-9));

                // 3D vectors
                final a3 = NDArray.fromList(
                  [1.0, 0.0, 0.0],
                  [3],
                  DType.float64,
                );
                final b3 = NDArray.fromList(
                  [0.0, 1.0, 0.0],
                  [3],
                  DType.float64,
                );
                final res3D = cross(a3, b3);
                expect(res3D.shape, [3]);
                expect(res3D.toList(), [0.0, 0.0, 1.0]);

                // Multidimensional cross with axis specification
                final m1 = NDArray.fromList(
                  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [2, 3],
                  DType.float64,
                );
                final m2 = NDArray.fromList(
                  [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  [2, 3],
                  DType.float64,
                );
                final resMulti = cross(m1, m2, axis: -1);
                expect(resMulti.shape, [2, 3]);
                expect(resMulti.getCell([0, 2]), 1.0);
                expect(resMulti.getCell([1, 0]), 1.0);
              });
            },
          );
        });
      });

      // ------------------------------------------------------------------------
      // Group 4: Tensor Contractions & Products (tensor_contractions.dart)
      // ------------------------------------------------------------------------
      group(
        '4. Tensor Contractions, Products & Einsum (tensor_contractions.dart)',
        () {
          group('tensordot and TensordotAxes', () {
            test('tensordot with count, pair, and explicit axes', () {
              NDArray.scope(() {
                final a = NDArray<Float64>.arange(
                  0.0,
                  60.0,
                  dtype: DType.float64,
                ).reshape([3, 4, 5]);
                final b = NDArray<Float64>.arange(
                  0.0,
                  24.0,
                  dtype: DType.float64,
                ).reshape([4, 3, 2]);

                // Explicit axes
                final c1 = tensordot(
                  a,
                  b,
                  axes: TensordotAxes.explicit([1, 0], [0, 1]),
                );
                expect(c1.shape, [5, 2]);

                // Pair axes
                final a2 = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [2, 2],
                  DType.float64,
                );
                final b2 = NDArray.fromList(
                  [5.0, 6.0, 7.0, 8.0],
                  [2, 2],
                  DType.float64,
                );
                final cPair = tensordot(a2, b2, axes: (1, 0));
                expect(cPair.shape, [2, 2]);

                // Count = 0 (Tensor outer product)
                final cOuter = tensordot(a2, b2, axes: 0);
                expect(cOuter.shape, [2, 2, 2, 2]);
              });
            });

            test('TensordotAxes.from parsing and error handling', () {
              final t1 = TensordotAxes.from(2);
              expect(t1.count, 2);

              final t2 = TensordotAxes.from((1, 0));
              expect(t2.explicitAxesA, [1]);
              expect(t2.explicitAxesB, [0]);

              final t3 = TensordotAxes.from(([0, 1], [1, 0]));
              expect(t3.explicitAxesA, [0, 1]);

              expect(() => TensordotAxes.from('invalid'), throwsArgumentError);
              expect(() => TensordotAxes.from([1, 2, 3]), throwsArgumentError);
            });
          });

          group('vdot and inner', () {
            test('vdot handles complex conjugate vector dot products', () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [Complex(1.0, 2.0), Complex(3.0, 4.0)],
                  [2],
                  DType.complex128,
                );
                final b = NDArray.fromList(
                  [Complex(2.0, 1.0), Complex(1.0, -1.0)],
                  [2],
                  DType.complex128,
                );

                final res = vdot(a, b);
                expect(res.shape, <int>[]);
                final c = res.scalar;
                expect(c.real, closeTo(3.0, 1e-9));
                expect(c.imag, closeTo(-10.0, 1e-9));

                // Multidimensional vdot auto-flattens
                final a2D = a.reshape([2, 1]);
                final b2D = b.reshape([1, 2]);
                final res2D = vdot(a2D, b2D);
                expect(res2D.scalar, res.scalar);
              });
            });

            test('inner product for 1D, 2D, and scalar tensors', () {
              NDArray.scope(() {
                final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
                final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
                final res1D = inner(a, b);
                expect(res1D.scalar, closeTo(32.0, 1e-9));

                final m1 = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [2, 2],
                  DType.float64,
                );
                final m2 = NDArray.fromList(
                  [5.0, 6.0, 7.0, 8.0],
                  [2, 2],
                  DType.float64,
                );
                final res2D = inner(m1, m2);
                expect(res2D.shape, [2, 2]);
                expect(res2D.getCell([0, 0]), 17.0); // 1*5 + 2*6 = 17
                expect(res2D.getCell([0, 1]), 23.0); // 1*7 + 2*8 = 23
              });
            });
          });

          group('kron across supported dtypes and multi-D shapes', () {
            test('kron across 15 dtypes and 2D/3D shapes', () {
              NDArray.scope(() {
                final supportedDTypes = <DType<AnySpec>>[
                  DType.float64,
                  DType.float32,
                  DType.int64,
                  DType.int32,
                  DType.int16,
                  DType.uint8,
                  DType.complex128,
                  DType.complex64,
                  DType.boolean,
                ];

                for (final dt in supportedDTypes) {
                  final a = dt.isComplex
                      ? NDArray.fromList(
                          [Complex(1.0, 0.0), Complex(0.0, 1.0)],
                          [2],
                          dt,
                        )
                      : dt == DType.boolean
                      ? NDArray.fromList([true, false], [2], dt)
                      : NDArray.fromList([1.0, 2.0], [2], dt);
                  final b = dt.isComplex
                      ? NDArray.fromList(
                          [Complex(2.0, 0.0), Complex(3.0, 0.0)],
                          [2],
                          dt,
                        )
                      : dt == DType.boolean
                      ? NDArray.fromList([false, true], [2], dt)
                      : NDArray.fromList([3.0, 4.0], [2], dt);

                  final res = kron(a, b);
                  expect(res.shape, [
                    4,
                  ], reason: 'Kronecker failed for DType $dt');
                }

                // 3D Kronecker product
                final a3D = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
                a3D.fill(2.0);
                final b3D = NDArray<Float64>.zeros([2, 3, 2], DType.float64);
                b3D.fill(3.0);
                final res3D = kron(a3D, b3D);
                expect(res3D.shape, [4, 6, 4]);
                expect(res3D.getCell([0, 0, 0]), 6.0);
              });
            });
          });

          group('einsum advanced patterns', () {
            test(
              'einsum matrix multiplication, diagonal, trace, transpose, and ellipsis',
              () {
                NDArray.scope(() {
                  final a = NDArray.fromList(
                    [1.0, 2.0, 3.0, 4.0],
                    [2, 2],
                    DType.float64,
                  );
                  final b = NDArray.fromList(
                    [2.0, 0.0, 1.0, 3.0],
                    [2, 2],
                    DType.float64,
                  );

                  // Matrix multiplication 'ij,jk->ik'
                  final cMatmul = einsum(EinsumSubscripts.parse('ij,jk->ik'), [
                    a,
                    b,
                  ]);
                  expect(cMatmul.shape, [2, 2]);
                  expect(cMatmul.toList(), [4.0, 6.0, 10.0, 12.0]);

                  // Vector dot product 'i,i->'
                  final v1 = NDArray.fromList(
                    [1.0, 2.0, 3.0],
                    [3],
                    DType.float64,
                  );
                  final v2 = NDArray.fromList(
                    [4.0, 5.0, 6.0],
                    [3],
                    DType.float64,
                  );
                  final dotRes = einsum(EinsumSubscripts.parse('i,i->'), [
                    v1,
                    v2,
                  ]);
                  expect(dotRes.scalar, 32.0);

                  // Diagonal extraction 'ii->i'
                  final diagRes = einsum(EinsumSubscripts.parse('ii->i'), [a]);
                  expect(diagRes.shape, [2]);
                  expect(diagRes.toList(), [1.0, 4.0]);

                  // Trace 'ii->'
                  final traceRes = einsum(EinsumSubscripts.parse('ii->'), [a]);
                  expect(traceRes.scalar, 5.0);

                  // Matrix Transpose 'ij->ji'
                  final transRes = einsum(EinsumSubscripts.parse('ij->ji'), [
                    a,
                  ]);
                  expect(transRes.toList(), [1.0, 3.0, 2.0, 4.0]);

                  // Ellipsis batch matrix multiplication '...ij,...jk->...ik'
                  final aBatch = NDArray<Float64>.zeros([
                    2,
                    2,
                    2,
                  ], DType.float64);
                  final bBatch = NDArray<Float64>.zeros([
                    2,
                    2,
                    2,
                  ], DType.float64);
                  aBatch.fill(1.0);
                  bBatch.fill(2.0);
                  final batchRes = einsum(
                    EinsumSubscripts.parse('...ij,...jk->...ik'),
                    [aBatch, bBatch],
                  );
                  expect(batchRes.shape, [2, 2, 2]);
                  expect(batchRes.getCell([0, 0, 0]), 4.0);
                });
              },
            );

            test(
              'einsum error handling on invalid subscripts and shape mismatches',
              () {
                NDArray.scope(() {
                  final a = NDArray.zeros([2, 3], DType.float64);
                  final b = NDArray.zeros([4, 2], DType.float64);

                  expect(
                    () => einsum(EinsumSubscripts.parse('ij,jk->ik'), [a, b]),
                    throwsArgumentError,
                  );
                  expect(
                    () => einsum(EinsumSubscripts.parse('ij,jk->ik'), [a]),
                    throwsArgumentError,
                  );
                  expect(
                    () => EinsumSubscripts.parse('ij->ik->il'),
                    throwsArgumentError,
                  );
                });
              },
            );
          });
        },
      );
    },
  );
}
