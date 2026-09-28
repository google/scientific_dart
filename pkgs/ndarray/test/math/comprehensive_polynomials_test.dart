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
  group('Comprehensive Polynomials: Standard Polynomials', () {
    group('polyval evaluation', () {
      test('Degree 0 (constant) polynomial across scalar, 1D, 2D, 3D x', () {
        NDArray.scope(() {
          final c = NDArray.fromList([7.5], [1], DType.float64);

          // 0D scalar
          final x0 = NDArray.scalar(3.0, dtype: DType.float64);
          final y0 = polyval(c, x0);
          expect(y0.shape, equals([]));
          expect(y0.scalar, equals(7.5));

          // 1D array
          final x1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final y1 = polyval(c, x1);
          expect(y1.shape, equals([3]));
          expect(y1.toList(), equals([7.5, 7.5, 7.5]));

          // 2D grid
          final x2 = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [2, 3],
            DType.float64,
          );
          final y2 = polyval(c, x2);
          expect(y2.shape, equals([2, 3]));
          expect(y2.toList(), equals([7.5, 7.5, 7.5, 7.5, 7.5, 7.5]));

          // 3D tensor
          final x3 = NDArray.zeros([2, 2, 2], DType.float64);
          final y3 = polyval(c, x3);
          expect(y3.shape, equals([2, 2, 2]));
          for (final val in y3.toList()) {
            expect(val, equals(7.5));
          }
        });
      });

      test('Linear and Quadratic evaluation', () {
        NDArray.scope(() {
          final cLin = NDArray.fromList([2.0, 3.0], [2], DType.float64);
          final xLin = NDArray.fromList(
            [-1.0, 0.0, 1.5, 4.0],
            [4],
            DType.float64,
          );
          final yLin = polyval(cLin, xLin);
          expect(yLin.shape, equals([4]));
          expect(yLin.getCell([0]), closeTo(1.0, 1e-9));
          expect(yLin.getCell([1]), closeTo(3.0, 1e-9));
          expect(yLin.getCell([2]), closeTo(6.0, 1e-9));
          expect(yLin.getCell([3]), closeTo(11.0, 1e-9));

          final cQuad = NDArray.fromList([3.0, -4.0, 5.0], [3], DType.float64);
          final xQuad = NDArray.fromList(
            [-2.0, 0.0, 1.0, 2.0],
            [4],
            DType.float64,
          );
          final yQuad = polyval(cQuad, xQuad);
          expect(yQuad.getCell([0]), closeTo(25.0, 1e-9));
          expect(yQuad.getCell([1]), closeTo(5.0, 1e-9));
          expect(yQuad.getCell([2]), closeTo(4.0, 1e-9));
          expect(yQuad.getCell([3]), closeTo(9.0, 1e-9));
        });
      });

      test('High degree polynomials (deg 5 and deg 7)', () {
        NDArray.scope(() {
          final c5 = NDArray.fromList(
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
            [6],
            DType.float64,
          );
          final x = NDArray.fromList([0.0, 1.0, 2.0, -1.0], [4], DType.float64);
          final y = polyval(c5, x);
          expect(y.getCell([0]), closeTo(-6.0, 1e-9));
          expect(y.getCell([1]), closeTo(-3.0, 1e-9));
          expect(y.getCell([2]), closeTo(12.0, 1e-9));
          expect(y.getCell([3]), closeTo(-21.0, 1e-9));
        });
      });

      test(
        'DTypes coverage: Float32, Complex128, Complex64, Mixed integer dtypes',
        () {
          NDArray.scope(() {
            final c32 = NDArray.fromList([2.0, 1.0], [2], DType.float32);
            final x32 = NDArray.fromList([3.0, 4.0], [2], DType.float32);
            final y32 = polyval(c32, x32);
            expect(y32.dtype, equals(DType.float32));
            expect(y32.getCell([0]), closeTo(7.0, 1e-5));
            expect(y32.getCell([1]), closeTo(9.0, 1e-5));

            final cCpx128 = NDArray.fromList(
              [Complex(1.0, 1.0), Complex(2.0, -1.0), Complex(3.0, 0.0)],
              [3],
              DType.complex128,
            );
            final xCpx128 = NDArray.fromList(
              [Complex(0.0, 1.0), Complex(1.0, 0.0)],
              [2],
              DType.complex128,
            );
            final yCpx128 = polyval(cCpx128, xCpx128);
            expect(yCpx128.dtype, equals(DType.complex128));
            expect(yCpx128.getCell([0]).real, closeTo(3.0, 1e-9));
            expect(yCpx128.getCell([0]).imag, closeTo(1.0, 1e-9));
            expect(yCpx128.getCell([1]).real, closeTo(6.0, 1e-9));
            expect(yCpx128.getCell([1]).imag, closeTo(0.0, 1e-9));

            final cCpx64 = NDArray.fromList(
              [Complex(1.0, 0.0), Complex(0.0, 1.0)],
              [2],
              DType.complex64,
            );
            final xCpx64 = NDArray.fromList(
              [Complex(2.0, 1.0)],
              [1],
              DType.complex64,
            );
            final yCpx64 = polyval(cCpx64, xCpx64);
            expect(yCpx64.dtype, equals(DType.complex64));
            expect(yCpx64.getCell([0]).real, closeTo(2.0, 1e-5));
            expect(yCpx64.getCell([0]).imag, closeTo(2.0, 1e-5));

            final cInt = NDArray.fromList([3, 1], [2], DType.int32);
            final xF64 = NDArray.fromList([2.5, 4.0], [2], DType.float64);
            final yMixed = polyval(cInt, xF64);
            expect(yMixed.dtype, equals(DType.float64));
            expect(yMixed.getCell([0]), closeTo(8.5, 1e-9));
            expect(yMixed.getCell([1]), closeTo(13.0, 1e-9));
          });
        },
      );

      test('Strided and non-contiguous evaluations for all dtypes', () {
        NDArray.scope(() {
          final cFull = NDArray.fromList(
            [1.0, 999.0, 2.0, 999.0, 3.0],
            [5],
            DType.float64,
          );
          final cSlice = cFull.slice([const Slice(start: 0, stop: 5, step: 2)]);
          expect(cSlice.isContiguous, isFalse);

          final xFull = NDArray.fromList(
            [0.0, 888.0, 1.0, 888.0, 2.0],
            [5],
            DType.float64,
          );
          final xSlice = xFull.slice([const Slice(start: 0, stop: 5, step: 2)]);
          expect(xSlice.isContiguous, isFalse);

          final yStrided = polyval(cSlice, xSlice);
          expect(yStrided.getCell([0]), closeTo(3.0, 1e-9));
          expect(yStrided.getCell([1]), closeTo(6.0, 1e-9));
          expect(yStrided.getCell([2]), closeTo(11.0, 1e-9));

          // Strided Float32
          final c32Full = NDArray.fromList(
            [1.0, 99.0, 2.0, 99.0],
            [4],
            DType.float32,
          );
          final c32Slice = c32Full.slice([
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          final x32Full = NDArray.fromList(
            [3.0, 88.0, 4.0, 88.0],
            [4],
            DType.float32,
          );
          final x32Slice = x32Full.slice([
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          final y32Strided = polyval(c32Slice, x32Slice);
          expect(y32Strided.dtype, equals(DType.float32));
          expect(y32Strided.getCell([0]), closeTo(5.0, 1e-5)); // 1*3 + 2 = 5
          expect(y32Strided.getCell([1]), closeTo(6.0, 1e-5)); // 1*4 + 2 = 6

          // Strided Complex128
          final cCpxFull = NDArray.fromList(
            [Complex(1, 0), Complex(99, 99), Complex(0, 1), Complex(99, 99)],
            [4],
            DType.complex128,
          );
          final cCpxSlice = cCpxFull.slice([
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          final xCpxFull = NDArray.fromList(
            [Complex(2, 0), Complex(88, 88), Complex(0, 2), Complex(88, 88)],
            [4],
            DType.complex128,
          );
          final xCpxSlice = xCpxFull.slice([
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          final yCpxStrided = polyval(cCpxSlice, xCpxSlice);
          expect(yCpxStrided.dtype, equals(DType.complex128));
          expect(
            yCpxStrided.getCell([0]),
            equals(Complex(2.0, 1.0)),
          ); // 1*(2) + i = 2 + i

          // Strided Complex64
          final c64Full = NDArray.fromList(
            [Complex(1, 0), Complex(99, 99), Complex(0, 1), Complex(99, 99)],
            [4],
            DType.complex64,
          );
          final c64Slice = c64Full.slice([
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          final x64Full = NDArray.fromList(
            [Complex(2, 0), Complex(88, 88), Complex(0, 2), Complex(88, 88)],
            [4],
            DType.complex64,
          );
          final x64Slice = x64Full.slice([
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          final y64Strided = polyval(c64Slice, x64Slice);
          expect(y64Strided.dtype, equals(DType.complex64));
          expect(y64Strided.getCell([0]).real, closeTo(2.0, 1e-5));
          expect(y64Strided.getCell([0]).imag, closeTo(1.0, 1e-5));
        });
      });

      test('Out buffer reuse and validation', () {
        NDArray.scope(() {
          final c = NDArray.fromList([2.0, 1.0], [2], DType.float64);
          final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final out = NDArray<double>.zeros([3], DType.float64);

          final res = polyval(c, x, out: out);
          expect(identical(res, out), isTrue);
          expect(out.toList(), equals([3.0, 5.0, 7.0]));

          // Float32 out buffer
          final c32 = NDArray.fromList([2.0, 1.0], [2], DType.float32);
          final x32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
          final out32 = NDArray<double>.zeros([2], DType.float32);
          final res32 = polyval(c32, x32, out: out32);
          expect(identical(res32, out32), isTrue);

          // Complex128 out buffer
          final cCpx = NDArray.fromList(
            [Complex(1, 0), Complex(0, 1)],
            [2],
            DType.complex128,
          );
          final xCpx = NDArray.fromList([Complex(2, 0)], [1], DType.complex128);
          final outCpx = NDArray<Complex>.zeros([1], DType.complex128);
          final resCpx = polyval(cCpx, xCpx, out: outCpx);
          expect(identical(resCpx, outCpx), isTrue);

          // Complex64 out buffer
          final cCpx64 = NDArray.fromList(
            [Complex(1, 0), Complex(0, 1)],
            [2],
            DType.complex64,
          );
          final xCpx64 = NDArray.fromList(
            [Complex(2, 0)],
            [1],
            DType.complex64,
          );
          final outCpx64 = NDArray<Complex>.zeros([1], DType.complex64);
          final resCpx64 = polyval(cCpx64, xCpx64, out: outCpx64);
          expect(identical(resCpx64, outCpx64), isTrue);

          final outBadShape = NDArray<double>.zeros([4], DType.float64);
          expect(() => polyval(c, x, out: outBadShape), throwsArgumentError);

          final outBadDtype = NDArray<double>.zeros([3], DType.float32);
          expect(() => polyval(c, x, out: outBadDtype), throwsArgumentError);
        });
      });

      test('Error handling and preconditions for polyval', () {
        NDArray.scope(() {
          final validC = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final validX = NDArray.fromList([1.0, 2.0], [2], DType.float64);

          final dispC = NDArray.fromList([1.0], [1], DType.float64)..dispose();
          expect(() => polyval(dispC, validX), throwsStateError);

          final dispX = NDArray.fromList([1.0], [1], DType.float64)..dispose();
          expect(() => polyval(validC, dispX), throwsStateError);

          final dispOut = NDArray<double>.zeros([2], DType.float64)..dispose();
          expect(() => polyval(validC, validX, out: dispOut), throwsStateError);

          final c2D = NDArray.zeros([2, 2], DType.float64);
          expect(() => polyval(c2D, validX), throwsArgumentError);

          final cEmpty = NDArray.zeros([0], DType.float64);
          expect(() => polyval(cEmpty, validX), throwsArgumentError);
        });
      });
    });

    group('polyfit least-squares fitting', () {
      test('Exact degree 1 and degree 2 polynomial fits', () {
        NDArray.scope(() {
          final xLin = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0],
            [4],
            DType.float64,
          );
          final yLin = NDArray.fromList(
            [-2.0, 1.0, 4.0, 7.0],
            [4],
            DType.float64,
          );
          final pLin = polyfit(xLin, yLin, 1);
          expect(pLin.shape, equals([2]));
          expect(pLin.getCell([0]), closeTo(3.0, 1e-5));
          expect(pLin.getCell([1]), closeTo(-2.0, 1e-5));

          final xQuad = NDArray.fromList(
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [5],
            DType.float64,
          );
          final yQuad = NDArray.fromList(
            [10.0, 3.0, 0.0, 1.0, 6.0],
            [5],
            DType.float64,
          );
          final pQuad = polyfit(xQuad, yQuad, 2);
          expect(pQuad.shape, equals([3]));
          expect(pQuad.getCell([0]), closeTo(2.0, 1e-5));
          expect(pQuad.getCell([1]), closeTo(-5.0, 1e-5));
          expect(pQuad.getCell([2]), closeTo(3.0, 1e-5));
        });
      });

      test('Cubic polynomial fit', () {
        NDArray.scope(() {
          final x = NDArray.fromList(
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [6],
            DType.float64,
          );
          final y = NDArray.fromList(
            [-22.0, -8.0, -4.0, -4.0, -2.0, 8.0],
            [6],
            DType.float64,
          );
          final p = polyfit(x, y, 3);
          expect(p.shape, equals([4]));
          expect(p.getCell([0]), closeTo(1.0, 1e-4));
          expect(p.getCell([1]), closeTo(-2.0, 1e-4));
          expect(p.getCell([2]), closeTo(1.0, 1e-4));
          expect(p.getCell([3]), closeTo(-4.0, 1e-4));
        });
      });

      test('Weighted polynomial fit', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final y = NDArray.fromList([2.1, 3.9, 6.1, 7.9], [4], DType.float64);
          final w = NDArray.fromList([1.0, 2.0, 2.0, 1.0], [4], DType.float64);

          final p = polyfit(x, y, 1, w: w);
          expect(p.shape, equals([2]));
          expect(p.getCell([0]), closeTo(1.94, 0.1));
        });
      });

      test('Explicit rcond parameter and rank deficiency fallback', () {
        NDArray.scope(() {
          final x = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );
          final y = NDArray.fromList(
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [5],
            DType.float64,
          );

          final pRcond = polyfit(x, y, 1, rcond: 1e-7);
          expect(pRcond.getCell([0]), closeTo(2.0, 1e-5));
          expect(pRcond.getCell([1]), closeTo(0.0, 1e-5));

          final pDeg0 = polyfit(x, y, 0);
          expect(pDeg0.shape, equals([1]));
          expect(pDeg0.getCell([0]), closeTo(6.0, 1e-5));
        });
      });

      test('DTypes coverage: Float32, Complex128, Complex64 with weights and rcond', () {
        NDArray.scope(() {
          // Float32
          final x32 = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float32);
          final y32 = NDArray.fromList([1.0, 3.0, 5.0], [3], DType.float32);
          final w32 = NDArray.fromList([1.0, 1.0, 1.0], [3], DType.float32);
          final p32 = polyfit(x32, y32, 1, w: w32, rcond: 1e-5);
          expect(p32.dtype, equals(DType.float32));
          expect(p32.getCell([0]), closeTo(2.0, 1e-4));
          expect(p32.getCell([1]), closeTo(1.0, 1e-4));

          // Complex128
          final xCpx = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)],
            [3],
            DType.complex128,
          );
          final yCpx = NDArray.fromList(
            [Complex(0.0, 1.0), Complex(1.0, 2.0), Complex(2.0, 3.0)],
            [3],
            DType.complex128,
          );
          final wCpx = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0)],
            [3],
            DType.complex128,
          );
          final pCpx = polyfit(xCpx, yCpx, 1, w: wCpx, rcond: 1e-5);
          expect(pCpx.dtype, equals(DType.complex128));
          expect(pCpx.getCell([0]).real, closeTo(1.0, 1e-5));
          expect(pCpx.getCell([0]).imag, closeTo(1.0, 1e-5));
          expect(pCpx.getCell([1]).real, closeTo(0.0, 1e-5));
          expect(pCpx.getCell([1]).imag, closeTo(1.0, 1e-5));

          // Complex64
          final xCpx64 = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)],
            [3],
            DType.complex64,
          );
          final yCpx64 = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(3.0, 0.0), Complex(5.0, 0.0)],
            [3],
            DType.complex64,
          );
          final wCpx64 = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0)],
            [3],
            DType.complex64,
          );
          final pCpx64 = polyfit(xCpx64, yCpx64, 1, w: wCpx64, rcond: 1e-5);
          expect(pCpx64.dtype, equals(DType.complex64));
          expect(pCpx64.getCell([0]).real, closeTo(2.0, 1e-4));
          expect(pCpx64.getCell([1]).real, closeTo(1.0, 1e-4));
        });
      });

      test('Strided inputs for polyfit across dtypes', () {
        NDArray.scope(() {
          final xFull = NDArray.fromList(
            [0.0, 99.0, 1.0, 99.0, 2.0, 99.0, 3.0],
            [7],
            DType.float64,
          );
          final yFull = NDArray.fromList(
            [1.0, 99.0, 3.0, 99.0, 5.0, 99.0, 7.0],
            [7],
            DType.float64,
          );
          final wFull = NDArray.fromList(
            [1.0, 99.0, 1.0, 99.0, 1.0, 99.0, 1.0],
            [7],
            DType.float64,
          );
          final xSlice = xFull.slice([const Slice(start: 0, stop: 7, step: 2)]);
          final ySlice = yFull.slice([const Slice(start: 0, stop: 7, step: 2)]);
          final wSlice = wFull.slice([const Slice(start: 0, stop: 7, step: 2)]);
          expect(xSlice.isContiguous, isFalse);
          expect(ySlice.isContiguous, isFalse);
          expect(wSlice.isContiguous, isFalse);

          final p = polyfit(xSlice, ySlice, 1, w: wSlice);
          expect(p.getCell([0]), closeTo(2.0, 1e-5));
          expect(p.getCell([1]), closeTo(1.0, 1e-5));

          // Strided Float32
          final x32Full = NDArray.fromList(
            [0.0, 99.0, 1.0, 99.0, 2.0, 99.0],
            [6],
            DType.float32,
          );
          final y32Full = NDArray.fromList(
            [1.0, 99.0, 3.0, 99.0, 5.0, 99.0],
            [6],
            DType.float32,
          );
          final x32Slice = x32Full.slice([
            const Slice(start: 0, stop: 6, step: 2),
          ]);
          final y32Slice = y32Full.slice([
            const Slice(start: 0, stop: 6, step: 2),
          ]);
          final p32 = polyfit(x32Slice, y32Slice, 1);
          expect(p32.getCell([0]), closeTo(2.0, 1e-4));
          expect(p32.getCell([1]), closeTo(1.0, 1e-4));
        });
      });

      test('Out buffer reuse and validation for polyfit', () {
        NDArray.scope(() {
          final x = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
          final y = NDArray.fromList([1.0, 3.0, 5.0], [3], DType.float64);
          final out = NDArray<double>.zeros([2], DType.float64);

          final res = polyfit(x, y, 1, out: out);
          expect(identical(res, out), isTrue);
          expect(out.getCell([0]), closeTo(2.0, 1e-5));
          expect(out.getCell([1]), closeTo(1.0, 1e-5));

          final badOut = NDArray<double>.zeros([3], DType.float64);
          expect(() => polyfit(x, y, 1, out: badOut), throwsArgumentError);
        });
      });

      test('Error handling and preconditions for polyfit', () {
        NDArray.scope(() {
          final x = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
          final y = NDArray.fromList([1.0, 3.0, 5.0], [3], DType.float64);
          final dispX = NDArray.fromList([0.0, 1.0], [2], DType.float64)
            ..dispose();
          expect(() => polyfit(dispX, y, 1), throwsStateError);

          final dispY = NDArray.fromList([0.0, 1.0], [2], DType.float64)
            ..dispose();
          expect(() => polyfit(x, dispY, 1), throwsStateError);

          final dispW = NDArray.fromList([1.0, 1.0, 1.0], [3], DType.float64)
            ..dispose();
          expect(() => polyfit(x, y, 1, w: dispW), throwsStateError);

          final dispOut = NDArray<double>.zeros([2], DType.float64)..dispose();
          expect(() => polyfit(x, y, 1, out: dispOut), throwsStateError);

          final x2d = NDArray.zeros([2, 2], DType.float64);
          expect(() => polyfit(x2d, y, 1), throwsArgumentError);
          expect(() => polyfit(x, x2d, 1), throwsArgumentError);

          final yShort = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => polyfit(x, yShort, 1), throwsArgumentError);

          expect(() => polyfit(x, y, -1), throwsArgumentError);
          expect(() => polyfit(x, y, 3), throwsArgumentError);

          final wShort = NDArray.fromList([1.0, 1.0], [2], DType.float64);
          expect(() => polyfit(x, y, 1, w: wShort), throwsArgumentError);
        });
      });
    });

    group('roots polynomial root finding', () {
      test('Degree 0 and empty/all-zeros cases', () {
        NDArray.scope(() {
          final emptyP = NDArray.zeros([0], DType.float64);
          final r0 = roots(emptyP);
          expect(r0.shape, equals([0]));

          final zeroP = NDArray.fromList([0.0, 0.0, 0.0], [3], DType.float64);
          final rZero = roots(zeroP);
          expect(rZero.shape, equals([0]));

          final constP = NDArray.fromList([5.0], [1], DType.float64);
          final rConst = roots(constP);
          expect(rConst.shape, equals([0]));
        });
      });

      test('Leading zeros stripping and linear root', () {
        NDArray.scope(() {
          final p = NDArray.fromList(
            [0.0, 0.0, 3.0, -12.0],
            [4],
            DType.float64,
          );
          final r = roots(p);
          expect(r.shape, equals([1]));
          expect(r.getCell([0]).real, closeTo(4.0, 1e-5));
          expect(r.getCell([0]).imag, closeTo(0.0, 1e-5));
        });
      });

      test(
        'Quadratic roots: real distinct, repeated, and complex conjugate',
        () {
          NDArray.scope(() {
            final pDistinct = NDArray.fromList(
              [1.0, -5.0, 6.0],
              [3],
              DType.float64,
            );
            final rDistinct = roots(pDistinct);
            expect(rDistinct.shape, equals([2]));
            final rVals = [
              rDistinct.getCell([0]).real,
              rDistinct.getCell([1]).real,
            ]..sort();
            expect(rVals[0], closeTo(2.0, 1e-5));
            expect(rVals[1], closeTo(3.0, 1e-5));

            final pRepeated = NDArray.fromList(
              [1.0, -6.0, 9.0],
              [3],
              DType.float64,
            );
            final rRepeated = roots(pRepeated);
            expect(rRepeated.shape, equals([2]));
            expect(rRepeated.getCell([0]).real, closeTo(3.0, 1e-4));
            expect(rRepeated.getCell([1]).real, closeTo(3.0, 1e-4));

            final pComplex = NDArray.fromList(
              [1.0, 0.0, 9.0],
              [3],
              DType.float64,
            );
            final rComplex = roots(pComplex);
            expect(rComplex.shape, equals([2]));
            final imagVals = [
              rComplex.getCell([0]).imag,
              rComplex.getCell([1]).imag,
            ]..sort();
            expect(imagVals[0], closeTo(-3.0, 1e-5));
            expect(imagVals[1], closeTo(3.0, 1e-5));
          });
        },
      );

      test('Cubic and Quartic roots', () {
        NDArray.scope(() {
          final pCubic = NDArray.fromList(
            [1.0, -6.0, 11.0, -6.0],
            [4],
            DType.float64,
          );
          final rCubic = roots(pCubic);
          expect(rCubic.shape, equals([3]));
          final cVals = [
            rCubic.getCell([0]).real,
            rCubic.getCell([1]).real,
            rCubic.getCell([2]).real,
          ]..sort();
          expect(cVals[0], closeTo(1.0, 1e-4));
          expect(cVals[1], closeTo(2.0, 1e-4));
          expect(cVals[2], closeTo(3.0, 1e-4));

          final pQuartic = NDArray.fromList(
            [1.0, 0.0, 0.0, 0.0, -1.0],
            [5],
            DType.float64,
          );
          final rQuartic = roots(pQuartic);
          expect(rQuartic.shape, equals([4]));
          final magnitudes = List.generate(4, (i) {
            final c = rQuartic.getCell([i]);
            return math.sqrt(c.real * c.real + c.imag * c.imag);
          });
          for (final m in magnitudes) {
            expect(m, closeTo(1.0, 1e-4));
          }
        });
      });

      test('Complex coefficient polynomials (Complex128, Complex64)', () {
        NDArray.scope(() {
          final pCpx = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(-1.0, -1.0), Complex(0.0, 1.0)],
            [3],
            DType.complex128,
          );
          final rCpx = roots(pCpx);
          expect(rCpx.shape, equals([2]));
          final c0 = rCpx.getCell([0]);
          final c1 = rCpx.getCell([1]);

          final isRoot1 = (c0.real - 1.0).abs() < 1e-4 && c0.imag.abs() < 1e-4;
          final isRootI = c0.real.abs() < 1e-4 && (c0.imag - 1.0).abs() < 1e-4;
          expect(isRoot1 || isRootI, isTrue);

          final otherIsRoot1 =
              (c1.real - 1.0).abs() < 1e-4 && c1.imag.abs() < 1e-4;
          final otherIsRootI =
              c1.real.abs() < 1e-4 && (c1.imag - 1.0).abs() < 1e-4;
          expect(otherIsRoot1 || otherIsRootI, isTrue);

          // Complex64
          final p64 = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(-3.0, 0.0), Complex(2.0, 0.0)],
            [3],
            DType.complex64,
          );
          final r64 = roots(p64);
          expect(r64.shape, equals([2]));
        });
      });

      test('Out buffer reuse and validation for roots', () {
        NDArray.scope(() {
          final p = NDArray.fromList([1.0, -3.0, 2.0], [3], DType.float64);
          final out = NDArray<Complex>.zeros([2], DType.complex128);

          final res = roots(p, out: out);
          expect(identical(res, out), isTrue);
          final vals = [
            out.getCell([0]).real,
            out.getCell([1]).real,
          ]..sort();
          expect(vals[0], closeTo(1.0, 1e-5));
          expect(vals[1], closeTo(2.0, 1e-5));

          final pConst = NDArray.fromList([5.0], [1], DType.float64);
          final emptyOut = NDArray<Complex>.zeros([0], DType.complex128);
          final emptyRes = roots(pConst, out: emptyOut);
          expect(identical(emptyRes, emptyOut), isTrue);

          final badOut = NDArray<Complex>.zeros([3], DType.complex128);
          expect(() => roots(p, out: badOut), throwsArgumentError);
        });
      });

      test('Error handling for roots', () {
        NDArray.scope(() {
          final p = NDArray.fromList([1.0, 2.0], [2], DType.float64);

          final dispP = NDArray.fromList([1.0, 2.0], [2], DType.float64)
            ..dispose();
          expect(() => roots(dispP), throwsStateError);

          final dispOut = NDArray<Complex>.zeros([1], DType.complex128)
            ..dispose();
          expect(() => roots(p, out: dispOut), throwsStateError);

          final p2D = NDArray.zeros([2, 2], DType.float64);
          expect(() => roots(p2D), throwsArgumentError);
        });
      });
    });
  });

  group('Comprehensive Polynomials: Orthogonal Polynomial Series', () {
    group('Chebyshev polynomials (chebval, chebroots)', () {
      test('Standard basis identities T_0, T_1, T_2, T_3', () {
        NDArray.scope(() {
          final x = NDArray.fromList(
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [5],
            DType.float64,
          );

          final c0 = NDArray.fromList([1.0], [1], DType.float64);
          expect(chebval(c0, x).toList(), equals([1.0, 1.0, 1.0, 1.0, 1.0]));

          final c1 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          final y1 = chebval(c1, x);
          for (var i = 0; i < 5; i++) {
            expect(y1.getCell([i]), closeTo(x.getCell([i]), 1e-9));
          }

          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final y2 = chebval(c2, x);
          expect(y2.getCell([0]), closeTo(1.0, 1e-9));
          expect(y2.getCell([1]), closeTo(-0.5, 1e-9));
          expect(y2.getCell([2]), closeTo(-1.0, 1e-9));
          expect(y2.getCell([3]), closeTo(-0.5, 1e-9));
          expect(y2.getCell([4]), closeTo(1.0, 1e-9));

          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final y3 = chebval(c3, x);
          expect(y3.getCell([3]), closeTo(-1.0, 1e-9));
        });
      });

      test('Flexible argument ordering (c, x) and (x, c)', () {
        NDArray.scope(() {
          final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final x2d = NDArray.fromList(
            [0.2, 0.8, -0.5, 0.0],
            [2, 2],
            DType.float64,
          );

          final y1 = chebval(c, x2d);
          final y2 = chebval(x2d, c);
          expect(y1.shape, equals([2, 2]));
          expect(y2.shape, equals([2, 2]));
          expect(y1.toList(), equals(y2.toList()));
        });
      });

      test('Strided non-contiguous chebval across all dtypes', () {
        NDArray.scope(() {
          final cFull = NDArray.fromList(
            [1.0, 99.0, 2.0, 99.0, 3.0],
            [5],
            DType.float64,
          );
          final cSlice = cFull.slice([const Slice(start: 0, stop: 5, step: 2)]);
          final xFull = NDArray.fromList(
            [0.0, 88.0, 0.5, 88.0, 1.0],
            [5],
            DType.float64,
          );
          final xSlice = xFull.slice([const Slice(start: 0, stop: 5, step: 2)]);

          final yStrided = chebval(cSlice, xSlice);
          expect(yStrided.getCell([0]), closeTo(-2.0, 1e-5));
          expect(yStrided.getCell([1]), closeTo(0.5, 1e-5));
          expect(yStrided.getCell([2]), closeTo(6.0, 1e-5));

          // Strided Float32
          final c32 = NDArray.fromList([1.0, 99.0, 2.0], [3], DType.float32);
          final c32Slice = c32.slice([const Slice(start: 0, stop: 3, step: 2)]);
          final x32 = NDArray.fromList([0.0, 88.0, 1.0], [3], DType.float32);
          final x32Slice = x32.slice([const Slice(start: 0, stop: 3, step: 2)]);
          final y32 = chebval(c32Slice, x32Slice);
          expect(y32.dtype, equals(DType.float32));

          // Strided Complex128
          final cCpx = NDArray.fromList(
            [Complex(1, 0), Complex(99, 99), Complex(2, 0)],
            [3],
            DType.complex128,
          );
          final cCpxSlice = cCpx.slice([
            const Slice(start: 0, stop: 3, step: 2),
          ]);
          final xCpx = NDArray.fromList(
            [Complex(0, 1), Complex(88, 88), Complex(1, 0)],
            [3],
            DType.complex128,
          );
          final xCpxSlice = xCpx.slice([
            const Slice(start: 0, stop: 3, step: 2),
          ]);
          final yCpx = chebval(cCpxSlice, xCpxSlice);
          expect(yCpx.dtype, equals(DType.complex128));

          // Strided Complex64
          final cCpx64 = NDArray.fromList(
            [Complex(1, 0), Complex(99, 99), Complex(2, 0)],
            [3],
            DType.complex64,
          );
          final cCpx64Slice = cCpx64.slice([
            const Slice(start: 0, stop: 3, step: 2),
          ]);
          final xCpx64 = NDArray.fromList(
            [Complex(0, 1), Complex(88, 88), Complex(1, 0)],
            [3],
            DType.complex64,
          );
          final xCpx64Slice = xCpx64.slice([
            const Slice(start: 0, stop: 3, step: 2),
          ]);
          final yCpx64 = chebval(cCpx64Slice, xCpx64Slice);
          expect(yCpx64.dtype, equals(DType.complex64));
        });
      });

      test('chebroots analytic and eigenvalue companion matrix', () {
        NDArray.scope(() {
          expect(
            chebroots(NDArray.fromList([0.0, 0.0], [2], DType.float64)).shape,
            equals([0]),
          );
          expect(
            chebroots(NDArray.fromList([4.0], [1], DType.float64)).shape,
            equals([0]),
          );

          final cDeg1 = NDArray.fromList([2.0, -4.0], [2], DType.float64);
          final r1 = chebroots(cDeg1);
          expect(r1.shape, equals([1]));
          expect(r1.getCell([0]).real, closeTo(0.5, 1e-5));

          final cDeg2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = chebroots(cDeg2);
          expect(r2.shape, equals([2]));
          final r2Vals = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Vals[0], closeTo(-0.70710678, 1e-4));
          expect(r2Vals[1], closeTo(0.70710678, 1e-4));

          final cDeg3 = NDArray.fromList(
            [0.0, 0.0, 0.0, 1.0],
            [4],
            DType.float64,
          );
          final r3 = chebroots(cDeg3);
          expect(r3.shape, equals([3]));
          final r3Vals = [
            r3.getCell([0]).real,
            r3.getCell([1]).real,
            r3.getCell([2]).real,
          ]..sort();
          expect(r3Vals[0], closeTo(-0.8660254, 1e-4));
          expect(r3Vals[1], closeTo(0.0, 1e-4));
          expect(r3Vals[2], closeTo(0.8660254, 1e-4));

          // Complex chebroots
          final cCpx = NDArray.fromList(
            [Complex(0, 0), Complex(0, 0), Complex(1, 0)],
            [3],
            DType.complex128,
          );
          final rCpx = chebroots(cCpx);
          expect(rCpx.shape, equals([2]));
        });
      });
    });

    group('Legendre polynomials (legval, legroots)', () {
      test('Standard basis identities P_0, P_1, P_2, P_3', () {
        NDArray.scope(() {
          final x = NDArray.fromList([-1.0, 0.0, 1.0], [3], DType.float64);

          final c0 = NDArray.fromList([1.0], [1], DType.float64);
          expect(legval(c0, x).toList(), equals([1.0, 1.0, 1.0]));

          final c1 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          expect(legval(c1, x).toList(), equals([-1.0, 0.0, 1.0]));

          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final y2 = legval(c2, x);
          expect(y2.getCell([0]), closeTo(1.0, 1e-9));
          expect(y2.getCell([1]), closeTo(-0.5, 1e-9));
          expect(y2.getCell([2]), closeTo(1.0, 1e-9));

          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final y3 = legval(c3, x);
          expect(y3.getCell([0]), closeTo(-1.0, 1e-9));
          expect(y3.getCell([1]), closeTo(0.0, 1e-9));
          expect(y3.getCell([2]), closeTo(1.0, 1e-9));
        });
      });

      test('Strided non-contiguous legval', () {
        NDArray.scope(() {
          final cFull = NDArray.fromList(
            [1.0, 99.0, 2.0, 99.0, 3.0],
            [5],
            DType.float64,
          );
          final cSlice = cFull.slice([const Slice(start: 0, stop: 5, step: 2)]);
          final xFull = NDArray.fromList(
            [0.0, 88.0, 0.5, 88.0, 1.0],
            [5],
            DType.float64,
          );
          final xSlice = xFull.slice([const Slice(start: 0, stop: 5, step: 2)]);

          final yStrided = legval(cSlice, xSlice);
          expect(yStrided.getCell([0]), closeTo(-0.5, 1e-5));
          expect(yStrided.getCell([1]), closeTo(1.625, 1e-5));
          expect(yStrided.getCell([2]), closeTo(6.0, 1e-5));
        });
      });

      test('legroots analytic and companion matrix roots', () {
        NDArray.scope(() {
          expect(
            legroots(NDArray.fromList([3.0], [1], DType.float64)).shape,
            equals([0]),
          );

          final c1 = NDArray.fromList([3.0, 6.0], [2], DType.float64);
          final r1 = legroots(c1);
          expect(r1.getCell([0]).real, closeTo(-0.5, 1e-5));

          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = legroots(c2);
          final r2Vals = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Vals[0], closeTo(-0.57735027, 1e-4));
          expect(r2Vals[1], closeTo(0.57735027, 1e-4));

          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final r3 = legroots(c3);
          final r3Vals = [
            r3.getCell([0]).real,
            r3.getCell([1]).real,
            r3.getCell([2]).real,
          ]..sort();
          expect(r3Vals[0], closeTo(-0.77459667, 1e-4));
          expect(r3Vals[1], closeTo(0.0, 1e-4));
          expect(r3Vals[2], closeTo(0.77459667, 1e-4));
        });
      });
    });

    group('Hermite polynomials (hermval, hermroots)', () {
      test('Standard basis identities H_0, H_1, H_2, H_3', () {
        NDArray.scope(() {
          final x = NDArray.fromList([-1.0, 0.0, 1.0], [3], DType.float64);

          final c0 = NDArray.fromList([1.0], [1], DType.float64);
          expect(hermval(c0, x).toList(), equals([1.0, 1.0, 1.0]));

          final c1 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          expect(hermval(c1, x).toList(), equals([-2.0, 0.0, 2.0]));

          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final y2 = hermval(c2, x);
          expect(y2.getCell([0]), closeTo(2.0, 1e-9));
          expect(y2.getCell([1]), closeTo(-2.0, 1e-9));
          expect(y2.getCell([2]), closeTo(2.0, 1e-9));

          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final y3 = hermval(c3, x);
          expect(y3.getCell([0]), closeTo(4.0, 1e-9));
          expect(y3.getCell([1]), closeTo(0.0, 1e-9));
          expect(y3.getCell([2]), closeTo(-4.0, 1e-9));
        });
      });

      test('Strided non-contiguous hermval', () {
        NDArray.scope(() {
          final cFull = NDArray.fromList(
            [1.0, 99.0, 2.0, 99.0, 3.0],
            [5],
            DType.float64,
          );
          final cSlice = cFull.slice([const Slice(start: 0, stop: 5, step: 2)]);
          final xFull = NDArray.fromList(
            [0.0, 88.0, 0.5, 88.0, 1.0],
            [5],
            DType.float64,
          );
          final xSlice = xFull.slice([const Slice(start: 0, stop: 5, step: 2)]);

          final yStrided = hermval(cSlice, xSlice);
          expect(yStrided.getCell([0]), closeTo(-5.0, 1e-5));
          expect(yStrided.getCell([1]), closeTo(0.0, 1e-5));
          expect(yStrided.getCell([2]), closeTo(11.0, 1e-5));
        });
      });

      test('hermroots analytic and companion matrix roots', () {
        NDArray.scope(() {
          expect(
            hermroots(NDArray.fromList([10.0], [1], DType.float64)).shape,
            equals([0]),
          );

          final c1 = NDArray.fromList([4.0, 2.0], [2], DType.float64);
          final r1 = hermroots(c1);
          expect(r1.getCell([0]).real, closeTo(-1.0, 1e-5));

          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = hermroots(c2);
          final r2Vals = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Vals[0], closeTo(-0.70710678, 1e-4));
          expect(r2Vals[1], closeTo(0.70710678, 1e-4));

          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final r3 = hermroots(c3);
          final r3Vals = [
            r3.getCell([0]).real,
            r3.getCell([1]).real,
            r3.getCell([2]).real,
          ]..sort();
          expect(r3Vals[0], closeTo(-1.22474487, 1e-4));
          expect(r3Vals[1], closeTo(0.0, 1e-4));
          expect(r3Vals[2], closeTo(1.22474487, 1e-4));
        });
      });
    });

    group('Laguerre polynomials (lagval, lagroots)', () {
      test('Standard basis identities L_0, L_1, L_2, L_3', () {
        NDArray.scope(() {
          final x = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);

          final c0 = NDArray.fromList([1.0], [1], DType.float64);
          expect(lagval(c0, x).toList(), equals([1.0, 1.0, 1.0]));

          final c1 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          expect(lagval(c1, x).toList(), equals([1.0, 0.0, -1.0]));

          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final y2 = lagval(c2, x);
          expect(y2.getCell([0]), closeTo(1.0, 1e-9));
          expect(y2.getCell([1]), closeTo(-0.5, 1e-9));
          expect(y2.getCell([2]), closeTo(-1.0, 1e-9));

          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final y3 = lagval(c3, x);
          expect(y3.getCell([0]), closeTo(1.0, 1e-9));
          expect(y3.getCell([1]), closeTo(-2.0 / 3.0, 1e-9));
        });
      });

      test('Strided non-contiguous lagval', () {
        NDArray.scope(() {
          final cFull = NDArray.fromList(
            [1.0, 99.0, 2.0, 99.0, 3.0],
            [5],
            DType.float64,
          );
          final cSlice = cFull.slice([const Slice(start: 0, stop: 5, step: 2)]);
          final xFull = NDArray.fromList(
            [0.0, 88.0, 0.5, 88.0, 1.0],
            [5],
            DType.float64,
          );
          final xSlice = xFull.slice([const Slice(start: 0, stop: 5, step: 2)]);

          final yStrided = lagval(cSlice, xSlice);
          expect(yStrided.getCell([0]), closeTo(6.0, 1e-5));
          expect(yStrided.getCell([1]), closeTo(2.375, 1e-5));
          expect(yStrided.getCell([2]), closeTo(-0.5, 1e-5));
        });
      });

      test('lagroots analytic and companion matrix roots', () {
        NDArray.scope(() {
          expect(
            lagroots(NDArray.fromList([8.0], [1], DType.float64)).shape,
            equals([0]),
          );

          final c1 = NDArray.fromList([2.0, 4.0], [2], DType.float64);
          final r1 = lagroots(c1);
          expect(r1.getCell([0]).real, closeTo(1.5, 1e-5));

          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = lagroots(c2);
          final r2Vals = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Vals[0], closeTo(0.58578644, 1e-4));
          expect(r2Vals[1], closeTo(3.41421356, 1e-4));
        });
      });
    });

    group('Preconditions & Error Handling for Orthogonal Polynomials', () {
      test('Disposed and invalid shape errors', () {
        NDArray.scope(() {
          final c = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);

          final dispC = NDArray.fromList([1.0], [1], DType.float64)..dispose();
          expect(() => chebval(dispC, x), throwsStateError);
          expect(() => legval(dispC, x), throwsStateError);
          expect(() => hermval(dispC, x), throwsStateError);
          expect(() => lagval(dispC, x), throwsStateError);
          expect(() => chebroots(dispC), throwsStateError);
          expect(() => legroots(dispC), throwsStateError);
          expect(() => hermroots(dispC), throwsStateError);
          expect(() => lagroots(dispC), throwsStateError);

          final dispX = NDArray.fromList([1.0], [1], DType.float64)..dispose();
          expect(() => chebval(c, dispX), throwsStateError);
          expect(() => legval(c, dispX), throwsStateError);
          expect(() => hermval(c, dispX), throwsStateError);
          expect(() => lagval(c, dispX), throwsStateError);

          final dispOut = NDArray<double>.zeros([2], DType.float64)..dispose();
          expect(() => chebval(c, x, out: dispOut), throwsStateError);

          final c2D = NDArray.zeros([2, 2], DType.float64);
          final x2D = NDArray.zeros([2, 2], DType.float64);
          expect(() => chebval(c2D, x2D), throwsArgumentError);
          expect(() => legval(c2D, x2D), throwsArgumentError);
          expect(() => hermval(c2D, x2D), throwsArgumentError);
          expect(() => lagval(c2D, x2D), throwsArgumentError);
          expect(() => chebroots(c2D), throwsArgumentError);
          expect(() => legroots(c2D), throwsArgumentError);
          expect(() => hermroots(c2D), throwsArgumentError);
          expect(() => lagroots(c2D), throwsArgumentError);

          final emptyC = NDArray.zeros([0], DType.float64);
          expect(() => chebval(emptyC, x), throwsArgumentError);
          expect(() => legval(emptyC, x), throwsArgumentError);
          expect(() => hermval(emptyC, x), throwsArgumentError);
          expect(() => lagval(emptyC, x), throwsArgumentError);
        });
      });
    });
  });
}
