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
  group('1. Orthogonal Polynomials (Chebyshev, Legendre, Hermite, Laguerre)', () {
    group('Chebyshev Series (chebval & chebroots)', () {
      test('chebval 1D, 2D, 3D on float64, float32, complex128, complex64', () {
        NDArray.scope(() {
          // T_0=1, T_1=x, T_2=2x^2-1, T_3=4x^3-3x
          // c = [2, 0, 3] -> 2*T_0 + 0*T_1 + 3*T_2 = 2 + 3(2x^2 - 1) = 6x^2 - 1
          final cF64 = NDArray.fromList([2.0, 0.0, 3.0], [3], DType.float64);
          final x1D = NDArray.fromList(
            [0.0, 1.0, -1.0, 2.0],
            [4],
            DType.float64,
          );
          final y1D = chebval(x1D, cF64);
          expect(y1D.shape, equals([4]));
          expect(y1D.getCell([0]), closeTo(-1.0, 1e-10)); // 6(0)-1 = -1
          expect(y1D.getCell([1]), closeTo(5.0, 1e-10)); // 6(1)-1 = 5
          expect(y1D.getCell([2]), closeTo(5.0, 1e-10)); // 6(1)-1 = 5
          expect(y1D.getCell([3]), closeTo(23.0, 1e-10)); // 6(4)-1 = 23

          // 2D grid evaluation
          final x2D = NDArray.fromList(
            [0.0, 1.0, -1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final y2D = chebval(x2D, cF64);
          expect(y2D.shape, equals([2, 2]));
          expect(y2D.getCell([0, 0]), closeTo(-1.0, 1e-10));
          expect(y2D.getCell([0, 1]), closeTo(5.0, 1e-10));
          expect(y2D.getCell([1, 0]), closeTo(5.0, 1e-10));
          expect(y2D.getCell([1, 1]), closeTo(23.0, 1e-10));

          // 2D x evaluation consistency
          final yRev = chebval(x2D, cF64);
          expect(yRev.toList(), equals(y2D.toList()));

          // 3D grid evaluation
          final x3D = NDArray.zeros([2, 2, 2], DType.float64);
          x3D[[1, 1, 1]] = 2.0;
          final y3D = chebval(x3D, cF64);
          expect(y3D.shape, equals([2, 2, 2]));
          expect(y3D.getCell([0, 0, 0]), closeTo(-1.0, 1e-10));
          expect(y3D.getCell([1, 1, 1]), closeTo(23.0, 1e-10));

          // Float32 evaluation
          final cF32 = NDArray.fromList(
            [1.0, 2.0],
            [2],
            DType.float32,
          ); // 1 + 2x
          final xF32 = NDArray.fromList([3.0, -2.0], [2], DType.float32);
          final yF32 = chebval(xF32, cF32);
          expect(yF32.dtype, equals(DType.float32));
          expect(yF32.getCell([0]), closeTo(7.0, 1e-5));
          expect(yF32.getCell([1]), closeTo(-3.0, 1e-5));

          // Complex128 evaluation: c = [1+0i, 0+1i] -> 1 + i*x
          final cC128 = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(0.0, 1.0)],
            [2],
            DType.complex128,
          );
          final xC128 = NDArray.fromList(
            [Complex(2.0, 0.0), Complex(0.0, 1.0)],
            [2],
            DType.complex128,
          );
          final yC128 = chebval(xC128, cC128);
          expect(yC128.dtype, equals(DType.complex128));
          expect(yC128.getCell([0]).real, closeTo(1.0, 1e-10));
          expect(
            yC128.getCell([0]).imag,
            closeTo(2.0, 1e-10),
          ); // 1 + i*2 = 1+2i
          expect(yC128.getCell([1]).real, closeTo(0.0, 1e-10)); // 1 + i*i = 0
          expect(yC128.getCell([1]).imag, closeTo(0.0, 1e-10));

          // Complex64 evaluation
          final cC64 = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(1.0, 0.0)],
            [2],
            DType.complex64,
          );
          final xC64 = NDArray.fromList(
            [Complex(3.0, 1.0)],
            [1],
            DType.complex64,
          );
          final yC64 = chebval(xC64, cC64);
          expect(yC64.dtype, equals(DType.complex64));
          expect(yC64.getCell([0]).real, closeTo(4.0, 1e-4));
          expect(yC64.getCell([0]).imag, closeTo(1.0, 1e-4));

          // Int32 evaluation promotes to float64; mixed dtypes throw ArgumentError
          final xInt = NDArray.fromList([0, 1, 2], [3], DType.int32);
          final cInt = NDArray.fromList([1, 2, 2], [3], DType.int32);
          final yIntCheb = chebval(xInt, cInt);
          expect(yIntCheb.dtype, equals(DType.float64));
          expect(yIntCheb.getCell([0]), closeTo(-1.0, 1e-10));
          expect(yIntCheb.getCell([1]), closeTo(5.0, 1e-10));
          expect(
            () => chebval<AnySpec, DTypeTag>(xInt, cF64),
            throwsArgumentError,
          );

          // Custom out buffer
          final outBuf = NDArray.zeros([4], DType.float64);
          final resOut = chebval(x1D, cF64, out: outBuf);
          expect(identical(resOut, outBuf), isTrue);
          expect(outBuf.getCell([0]), closeTo(-1.0, 1e-10));

          // Non-contiguous strided evaluation
          final matFull = NDArray.fromList(
            [0.0, 10.0, 1.0, 20.0, -1.0, 30.0],
            [3, 2],
            DType.float64,
          );
          final col0View = matFull.slice([
            Slice.all(),
            Index(0),
          ]); // [0.0, 1.0, -1.0] strided
          expect(col0View.isContiguous, isFalse);
          final yStrided = chebval(col0View, cF64);
          expect(yStrided.getCell([0]), closeTo(-1.0, 1e-10));
          expect(yStrided.getCell([1]), closeTo(5.0, 1e-10));
          expect(yStrided.getCell([2]), closeTo(5.0, 1e-10));
        });
      });

      test('chebroots for degree 0, 1, 2, 3, 4 and trailing zeros', () {
        NDArray.scope(() {
          // Degree 0 (constant only) -> empty roots
          final c0 = NDArray.fromList([5.0], [1], DType.float64);
          final r0 = chebroots(c0);
          expect(r0.shape, equals([0]));

          // Degree 1 (linear): c = [3.0, 2.0] -> 3*T_0 + 2*T_1 = 3 + 2x = 0 -> x = -1.5
          final c1 = NDArray.fromList([3.0, 2.0], [2], DType.float64);
          final r1 = chebroots(c1);
          expect(r1.shape, equals([1]));
          expect(r1.getCell([0]).real, closeTo(-1.5, 1e-10));
          expect(r1.getCell([0]).imag, closeTo(0.0, 1e-10));

          // Degree 2: T_2(x) = 2x^2 - 1 = 0 -> x = +/- 1/sqrt(2) ~ +/- 0.707106
          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = chebroots(c2);
          expect(r2.shape, equals([2]));
          final r2Sorted = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Sorted[0], closeTo(-1.0 / math.sqrt(2.0), 1e-5));
          expect(r2Sorted[1], closeTo(1.0 / math.sqrt(2.0), 1e-5));

          // Degree 3: T_3(x) = 4x^3 - 3x = 0 -> roots: 0, +/- sqrt(3)/2 ~ +/- 0.866025
          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final r3 = chebroots(c3);
          expect(r3.shape, equals([3]));
          final r3Sorted = [
            r3.getCell([0]).real,
            r3.getCell([1]).real,
            r3.getCell([2]).real,
          ]..sort();
          expect(r3Sorted[0], closeTo(-math.sqrt(3.0) / 2.0, 1e-5));
          expect(r3Sorted[1], closeTo(0.0, 1e-5));
          expect(r3Sorted[2], closeTo(math.sqrt(3.0) / 2.0, 1e-5));

          // Degree 4: T_4(x) = 8x^4 - 8x^2 + 1 = 0 -> roots: +/- cos(pi/8), +/- cos(3pi/8)
          final c4 = NDArray.fromList(
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [5],
            DType.float64,
          );
          final r4 = chebroots(c4);
          expect(r4.shape, equals([4]));
          final r4Sorted = [
            for (var i = 0; i < 4; i++) r4.getCell([i]).real,
          ]..sort();
          expect(r4Sorted[0], closeTo(-math.cos(math.pi / 8.0), 1e-4));
          expect(r4Sorted[3], closeTo(math.cos(math.pi / 8.0), 1e-4));

          // Trailing zeros in c
          final cTrail = NDArray.fromList(
            [3.0, 2.0, 0.0, 0.0],
            [4],
            DType.float64,
          );
          final rTrail = chebroots(cTrail);
          expect(rTrail.shape, equals([1]));
          expect(rTrail.getCell([0]).real, closeTo(-1.5, 1e-10));

          // Float32 coefficients
          final cF32 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float32);
          final rF32 = chebroots(cF32);
          expect(rF32.shape, equals([2]));

          // Complex coefficients
          final cComp = NDArray.fromList(
            [Complex(2.0, 0.0), Complex(0.0, 2.0)],
            [2],
            DType.complex128,
          ); // 2 + 2i*x = 0 -> x = i
          final rComp = chebroots(cComp);
          expect(rComp.shape, equals([1]));
          expect(rComp.getCell([0]).real, closeTo(0.0, 1e-10));
          expect(rComp.getCell([0]).imag, closeTo(1.0, 1e-10));
        });
      });
    });

    group('Legendre Series (legval & legroots)', () {
      test(
        'legval 1D, 2D, 3D across float64, float32, complex128, complex64',
        () {
          NDArray.scope(() {
            // P_0=1, P_1=x, P_2=0.5*(3x^2-1), P_3=0.5*(5x^3-3x)
            // c = [1, 2, 3] -> 1*P_0 + 2*P_1 + 3*P_2 = 1 + 2x + 1.5(3x^2-1) = 4.5x^2 + 2x - 0.5
            final cF64 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            final x1D = NDArray.fromList(
              [0.0, 1.0, -1.0, 2.0],
              [4],
              DType.float64,
            );
            final y1D = legval(x1D, cF64);
            expect(y1D.shape, equals([4]));
            expect(
              y1D.getCell([0]),
              closeTo(-0.5, 1e-10),
            ); // 4.5(0)+2(0)-0.5 = -0.5
            expect(
              y1D.getCell([1]),
              closeTo(6.0, 1e-10),
            ); // 4.5(1)+2(1)-0.5 = 6.0
            expect(
              y1D.getCell([2]),
              closeTo(2.0, 1e-10),
            ); // 4.5(1)-2(1)-0.5 = 2.0
            expect(
              y1D.getCell([3]),
              closeTo(21.5, 1e-10),
            ); // 4.5(4)+2(2)-0.5 = 18+4-0.5 = 21.5

            // 2D grid
            final x2D = NDArray.fromList(
              [0.0, 1.0, -1.0, 2.0],
              [2, 2],
              DType.float64,
            );
            final y2D = legval(x2D, cF64);
            expect(y2D.shape, equals([2, 2]));
            expect(y2D.getCell([0, 0]), closeTo(-0.5, 1e-10));
            expect(y2D.getCell([1, 1]), closeTo(21.5, 1e-10));

            // 2D x evaluation consistency
            final yRev = legval(x2D, cF64);
            expect(yRev.toList(), equals(y2D.toList()));

            // Float32
            final cF32 = NDArray.fromList(
              [0.0, 0.0, 1.0],
              [3],
              DType.float32,
            ); // P_2(x) = 0.5*(3x^2-1)
            final xF32 = NDArray.fromList([1.0, 0.0], [2], DType.float32);
            final yF32 = legval(xF32, cF32);
            expect(yF32.dtype, equals(DType.float32));
            expect(yF32.getCell([0]), closeTo(1.0, 1e-5));
            expect(yF32.getCell([1]), closeTo(-0.5, 1e-5));

            // Complex128
            final cC128 = NDArray.fromList(
              [Complex(1.0, 0.0), Complex(0.0, 1.0)],
              [2],
              DType.complex128,
            ); // 1 + i*x
            final xC128 = NDArray.fromList(
              [Complex(0.0, 2.0)],
              [1],
              DType.complex128,
            );
            final yC128 = legval(xC128, cC128);
            expect(
              yC128.getCell([0]).real,
              closeTo(-1.0, 1e-10),
            ); // 1 + i(2i) = 1 - 2 = -1
            expect(yC128.getCell([0]).imag, closeTo(0.0, 1e-10));

            // Complex64
            final cC64 = NDArray.fromList(
              [Complex(2.0, 1.0), Complex(1.0, 0.0)],
              [2],
              DType.complex64,
            );
            final xC64 = NDArray.fromList(
              [Complex(1.0, 0.0)],
              [1],
              DType.complex64,
            );
            final yC64 = legval(xC64, cC64);
            expect(yC64.dtype, equals(DType.complex64));
            expect(yC64.getCell([0]).real, closeTo(3.0, 1e-4));
            expect(yC64.getCell([0]).imag, closeTo(1.0, 1e-4));

            // Non-contiguous strided
            final mat = NDArray.fromList(
              [0.0, 100.0, 1.0, 200.0],
              [2, 2],
              DType.float64,
            );
            final sliceCol = mat.slice([Slice.all(), Index(0)]);
            final yStrided = legval(sliceCol, cF64);
            expect(yStrided.getCell([0]), closeTo(-0.5, 1e-10));
            expect(yStrided.getCell([1]), closeTo(6.0, 1e-10));
          });
        },
      );

      test('legroots for degree 0, 1, 2, 3, 4', () {
        NDArray.scope(() {
          // Degree 0 -> empty
          final c0 = NDArray.fromList([42.0], [1], DType.float64);
          expect(legroots(c0).shape, equals([0]));

          // Degree 1: c = [4.0, 2.0] -> 4 + 2x = 0 -> x = -2.0
          final c1 = NDArray.fromList([4.0, 2.0], [2], DType.float64);
          final r1 = legroots(c1);
          expect(r1.shape, equals([1]));
          expect(r1.getCell([0]).real, closeTo(-2.0, 1e-10));

          // Degree 2: P_2(x) = 0.5*(3x^2 - 1) = 0 -> x = +/- 1/sqrt(3) ~ +/- 0.577350
          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = legroots(c2);
          expect(r2.shape, equals([2]));
          final r2Sorted = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Sorted[0], closeTo(-1.0 / math.sqrt(3.0), 1e-5));
          expect(r2Sorted[1], closeTo(1.0 / math.sqrt(3.0), 1e-5));

          // Degree 3: P_3(x) = 0.5*(5x^3 - 3x) = 0 -> roots 0, +/- sqrt(3/5) ~ +/- 0.7745966
          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final r3 = legroots(c3);
          expect(r3.shape, equals([3]));
          final r3Sorted = [
            r3.getCell([0]).real,
            r3.getCell([1]).real,
            r3.getCell([2]).real,
          ]..sort();
          expect(r3Sorted[0], closeTo(-math.sqrt(3.0 / 5.0), 1e-5));
          expect(r3Sorted[1], closeTo(0.0, 1e-5));
          expect(r3Sorted[2], closeTo(math.sqrt(3.0 / 5.0), 1e-5));

          // Complex coefficients
          final cComp = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(0.0, 1.0)],
            [2],
            DType.complex128,
          ); // 1 + ix = 0 -> x = i
          final rComp = legroots(cComp);
          expect(rComp.getCell([0]).real, closeTo(0.0, 1e-10));
          expect(rComp.getCell([0]).imag, closeTo(1.0, 1e-10));
        });
      });
    });

    group('Hermite Series (hermval & hermroots)', () {
      test(
        'hermval 1D, 2D, 3D across float64, float32, complex128, complex64',
        () {
          NDArray.scope(() {
            // H_0=1, H_1=2x, H_2=4x^2-2, H_3=8x^3-12x
            // c = [0, 0, 1] -> H_2(x) = 4x^2 - 2
            final cF64 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
            final x1D = NDArray.fromList(
              [0.0, 1.0, -1.0, 2.0],
              [4],
              DType.float64,
            );
            final y1D = hermval(x1D, cF64);
            expect(y1D.shape, equals([4]));
            expect(y1D.getCell([0]), closeTo(-2.0, 1e-10)); // 4(0)-2 = -2
            expect(y1D.getCell([1]), closeTo(2.0, 1e-10)); // 4(1)-2 = 2
            expect(y1D.getCell([2]), closeTo(2.0, 1e-10)); // 4(1)-2 = 2
            expect(y1D.getCell([3]), closeTo(14.0, 1e-10)); // 4(4)-2 = 14

            // 2D grid
            final x2D = NDArray.fromList(
              [0.0, 1.0, -1.0, 2.0],
              [2, 2],
              DType.float64,
            );
            final y2D = hermval(x2D, cF64);
            expect(y2D.getCell([0, 0]), closeTo(-2.0, 1e-10));
            expect(y2D.getCell([1, 1]), closeTo(14.0, 1e-10));

            // 2D x evaluation consistency
            final yRev = hermval(x2D, cF64);
            expect(yRev.toList(), equals(y2D.toList()));

            // Float32
            final cF32 = NDArray.fromList(
              [1.0, 1.0],
              [2],
              DType.float32,
            ); // H_0 + H_1 = 1 + 2x
            final xF32 = NDArray.fromList([2.0, -1.0], [2], DType.float32);
            final yF32 = hermval(xF32, cF32);
            expect(yF32.dtype, equals(DType.float32));
            expect(yF32.getCell([0]), closeTo(5.0, 1e-5));
            expect(yF32.getCell([1]), closeTo(-1.0, 1e-5));

            // Complex128
            final cC128 = NDArray.fromList(
              [Complex(0.0, 0.0), Complex(1.0, 0.0)],
              [2],
              DType.complex128,
            ); // H_1 = 2x
            final xC128 = NDArray.fromList(
              [Complex(1.0, 2.0)],
              [1],
              DType.complex128,
            );
            final yC128 = hermval(xC128, cC128);
            expect(yC128.getCell([0]).real, closeTo(2.0, 1e-10));
            expect(yC128.getCell([0]).imag, closeTo(4.0, 1e-10));

            // Complex64
            final cC64 = NDArray.fromList(
              [Complex(1.0, 0.0), Complex(1.0, 0.0)],
              [2],
              DType.complex64,
            );
            final xC64 = NDArray.fromList(
              [Complex(1.0, 0.0)],
              [1],
              DType.complex64,
            );
            final yC64 = hermval(xC64, cC64);
            expect(yC64.dtype, equals(DType.complex64));
            expect(yC64.getCell([0]).real, closeTo(3.0, 1e-4));

            // Non-contiguous strided
            final mat = NDArray.fromList(
              [0.0, 9.0, 2.0, 8.0],
              [2, 2],
              DType.float64,
            );
            final col0 = mat.slice([Slice.all(), Index(0)]);
            final yStrided = hermval(col0, cF64);
            expect(yStrided.getCell([0]), closeTo(-2.0, 1e-10));
            expect(yStrided.getCell([1]), closeTo(14.0, 1e-10));
          });
        },
      );

      test('hermroots for degree 0, 1, 2, 3, 4', () {
        NDArray.scope(() {
          // Degree 0 -> empty
          final c0 = NDArray.fromList([10.0], [1], DType.float64);
          expect(hermroots(c0).shape, equals([0]));

          // Degree 1: c = [4.0, 1.0] -> 4*H_0 + 1*H_1 = 4 + 2x = 0 -> x = -2.0
          final c1 = NDArray.fromList([4.0, 1.0], [2], DType.float64);
          final r1 = hermroots(c1);
          expect(r1.shape, equals([1]));
          expect(r1.getCell([0]).real, closeTo(-2.0, 1e-10));

          // Degree 2: H_2(x) = 4x^2 - 2 = 0 -> x = +/- 1/sqrt(2) ~ +/- 0.707106
          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = hermroots(c2);
          expect(r2.shape, equals([2]));
          final r2Sorted = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Sorted[0], closeTo(-1.0 / math.sqrt(2.0), 1e-5));
          expect(r2Sorted[1], closeTo(1.0 / math.sqrt(2.0), 1e-5));

          // Degree 3: H_3(x) = 8x^3 - 12x = 0 -> roots 0, +/- sqrt(1.5) ~ +/- 1.2247448
          final c3 = NDArray.fromList([0.0, 0.0, 0.0, 1.0], [4], DType.float64);
          final r3 = hermroots(c3);
          expect(r3.shape, equals([3]));
          final r3Sorted = [
            r3.getCell([0]).real,
            r3.getCell([1]).real,
            r3.getCell([2]).real,
          ]..sort();
          expect(r3Sorted[0], closeTo(-math.sqrt(1.5), 1e-5));
          expect(r3Sorted[1], closeTo(0.0, 1e-5));
          expect(r3Sorted[2], closeTo(math.sqrt(1.5), 1e-5));
        });
      });
    });

    group('Laguerre Series (lagval & lagroots)', () {
      test(
        'lagval 1D, 2D, 3D across float64, float32, complex128, complex64',
        () {
          NDArray.scope(() {
            // L_0=1, L_1=1-x, L_2=0.5*(x^2-4x+2)
            // c = [0, 0, 1] -> L_2(x) = 0.5*(x^2 - 4x + 2)
            final cF64 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
            final x1D = NDArray.fromList([0.0, 2.0, 4.0], [3], DType.float64);
            final y1D = lagval(x1D, cF64);
            expect(y1D.shape, equals([3]));
            expect(y1D.getCell([0]), closeTo(1.0, 1e-10)); // 0.5(2) = 1
            expect(y1D.getCell([1]), closeTo(-1.0, 1e-10)); // 0.5(4-8+2) = -1
            expect(y1D.getCell([2]), closeTo(1.0, 1e-10)); // 0.5(16-16+2) = 1

            // 2D grid
            final x2D = NDArray.fromList(
              [0.0, 2.0, 4.0, 0.0],
              [2, 2],
              DType.float64,
            );
            final y2D = lagval(x2D, cF64);
            expect(y2D.getCell([0, 0]), closeTo(1.0, 1e-10));
            expect(y2D.getCell([0, 1]), closeTo(-1.0, 1e-10));

            // 2D x evaluation consistency
            final yRev = lagval(x2D, cF64);
            expect(yRev.toList(), equals(y2D.toList()));

            // Float32
            final cF32 = NDArray.fromList(
              [1.0, 1.0],
              [2],
              DType.float32,
            ); // L_0 + L_1 = 1 + (1 - x) = 2 - x
            final xF32 = NDArray.fromList([0.0, 5.0], [2], DType.float32);
            final yF32 = lagval(xF32, cF32);
            expect(yF32.dtype, equals(DType.float32));
            expect(yF32.getCell([0]), closeTo(2.0, 1e-5));
            expect(yF32.getCell([1]), closeTo(-3.0, 1e-5));

            // Complex128
            final cC128 = NDArray.fromList(
              [Complex(1.0, 0.0), Complex(1.0, 0.0)],
              [2],
              DType.complex128,
            ); // 2 - x
            final xC128 = NDArray.fromList(
              [Complex(0.0, 2.0)],
              [1],
              DType.complex128,
            );
            final yC128 = lagval(xC128, cC128);
            expect(yC128.getCell([0]).real, closeTo(2.0, 1e-10));
            expect(yC128.getCell([0]).imag, closeTo(-2.0, 1e-10));

            // Complex64
            final cC64 = NDArray.fromList(
              [Complex(0.0, 0.0), Complex(1.0, 0.0)],
              [2],
              DType.complex64,
            ); // 1 - x
            final xC64 = NDArray.fromList(
              [Complex(1.0, 0.0)],
              [1],
              DType.complex64,
            );
            final yC64 = lagval(xC64, cC64);
            expect(yC64.dtype, equals(DType.complex64));
            expect(yC64.getCell([0]).real, closeTo(0.0, 1e-4));

            // Non-contiguous strided
            final mat = NDArray.fromList(
              [0.0, -1.0, 2.0, -2.0],
              [2, 2],
              DType.float64,
            );
            final col0 = mat.slice([Slice.all(), Index(0)]);
            final yStrided = lagval(col0, cF64);
            expect(yStrided.getCell([0]), closeTo(1.0, 1e-10));
            expect(yStrided.getCell([1]), closeTo(-1.0, 1e-10));
          });
        },
      );

      test('lagroots for degree 0, 1, 2, 3', () {
        NDArray.scope(() {
          // Degree 0 -> empty
          final c0 = NDArray.fromList([7.0], [1], DType.float64);
          expect(lagroots(c0).shape, equals([0]));

          // Degree 1: c = [2.0, 1.0] -> 2*L_0 + 1*L_1 = 2 + (1 - x) = 3 - x = 0 -> x = 3.0
          final c1 = NDArray.fromList([2.0, 1.0], [2], DType.float64);
          final r1 = lagroots(c1);
          expect(r1.shape, equals([1]));
          expect(r1.getCell([0]).real, closeTo(3.0, 1e-10));

          // Degree 2: L_2(x) = 0.5*(x^2 - 4x + 2) = 0 -> roots 2 +/- sqrt(2) ~ 0.585786, 3.4142135
          final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
          final r2 = lagroots(c2);
          expect(r2.shape, equals([2]));
          final r2Sorted = [
            r2.getCell([0]).real,
            r2.getCell([1]).real,
          ]..sort();
          expect(r2Sorted[0], closeTo(2.0 - math.sqrt(2.0), 1e-5));
          expect(r2Sorted[1], closeTo(2.0 + math.sqrt(2.0), 1e-5));
        });
      });
    });

    group('Orthogonal Series Error Paths & Pre-conditions', () {
      test('chebval, legval, hermval, lagval input validation', () {
        NDArray.scope(() {
          final cEmpty = NDArray.zeros([0], DType.float64);
          final c2D = NDArray.zeros([2, 2], DType.float64);
          final x2D = NDArray.zeros([2, 2], DType.float64);
          final x = NDArray.zeros([3], DType.float64);

          expect(() => chebval(x, cEmpty), throwsArgumentError);
          expect(() => legval(x, cEmpty), throwsArgumentError);
          expect(() => hermval(x, cEmpty), throwsArgumentError);
          expect(() => lagval(x, cEmpty), throwsArgumentError);

          // If both are 2D, neither is a 1D coefficient vector
          expect(() => chebval(x2D, c2D), throwsArgumentError);
          expect(() => legval(x2D, c2D), throwsArgumentError);
          expect(() => hermval(x2D, c2D), throwsArgumentError);
          expect(() => lagval(x2D, c2D), throwsArgumentError);

          expect(() => chebroots(c2D), throwsArgumentError);
          expect(() => legroots(c2D), throwsArgumentError);
          expect(() => hermroots(c2D), throwsArgumentError);
          expect(() => lagroots(c2D), throwsArgumentError);

          final cValid = NDArray.zeros([3], DType.float64);
          final badOutShape = NDArray.zeros([4], DType.float64);
          expect(
            () => chebval(x, cValid, out: badOutShape),
            throwsArgumentError,
          );
          expect(
            () => legval(x, cValid, out: badOutShape),
            throwsArgumentError,
          );
          expect(
            () => hermval(x, cValid, out: badOutShape),
            throwsArgumentError,
          );
          expect(
            () => lagval(x, cValid, out: badOutShape),
            throwsArgumentError,
          );

          // Disposed arrays
          final cDisp = NDArray.zeros([2], DType.float64)..dispose();
          expect(() => chebval(x, cDisp), throwsStateError);
          expect(() => legval(x, cDisp), throwsStateError);
          expect(() => hermval(x, cDisp), throwsStateError);
          expect(() => lagval(x, cDisp), throwsStateError);

          expect(() => chebroots(cDisp), throwsStateError);
          expect(() => legroots(cDisp), throwsStateError);
          expect(() => hermroots(cDisp), throwsStateError);
          expect(() => lagroots(cDisp), throwsStateError);
        });
      });
    });
  });

  group('2. Standard Polynomial Operations (polyval, polyfit, roots)', () {
    test(
      'polyval Horner across float64, float32, complex128, complex64, integers and strided slices',
      () {
        NDArray.scope(() {
          // p(x) = 3x^3 - 2x^2 + 4x - 5
          final c = NDArray.fromList(
            [3.0, -2.0, 4.0, -5.0],
            [4],
            DType.float64,
          );
          final x = NDArray.fromList([0.0, 1.0, 2.0, -1.0], [4], DType.float64);
          final y = polyval(c, x);
          expect(y.getCell([0]), closeTo(-5.0, 1e-10)); // 0 - 5 = -5
          expect(y.getCell([1]), closeTo(0.0, 1e-10)); // 3 - 2 + 4 - 5 = 0
          expect(
            y.getCell([2]),
            closeTo(19.0, 1e-10),
          ); // 3(8) - 2(4) + 4(2) - 5 = 24 - 8 + 8 - 5 = 19
          expect(
            y.getCell([3]),
            closeTo(-14.0, 1e-10),
          ); // 3(-1) - 2(1) + 4(-1) - 5 = -3 - 2 - 4 - 5 = -14

          // Multi-dimensional 2D and 3D x
          final x2D = NDArray.fromList(
            [0.0, 1.0, 2.0, -1.0],
            [2, 2],
            DType.float64,
          );
          final y2D = polyval(c, x2D);
          expect(y2D.shape, equals([2, 2]));
          expect(y2D.getCell([0, 0]), closeTo(-5.0, 1e-10));
          expect(y2D.getCell([1, 0]), closeTo(19.0, 1e-10));

          // Float32
          final c32 = NDArray.fromList(
            [2.0, 3.0],
            [2],
            DType.float32,
          ); // 2x + 3
          final x32 = NDArray.fromList([4.0, 5.0], [2], DType.float32);
          final y32 = polyval(c32, x32);
          expect(y32.dtype, equals(DType.float32));
          expect(y32.getCell([0]), closeTo(11.0, 1e-5));
          expect(y32.getCell([1]), closeTo(13.0, 1e-5));

          // Complex128: c = [1+0i, 0+1i, 2+0i] -> x^2 + i*x + 2
          final cComp = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(0.0, 1.0), Complex(2.0, 0.0)],
            [3],
            DType.complex128,
          );
          final xComp = NDArray.fromList(
            [Complex(0.0, 1.0)],
            [1],
            DType.complex128,
          ); // x = i -> i^2 + i*i + 2 = -1 - 1 + 2 = 0
          final yComp = polyval(cComp, xComp);
          expect(yComp.getCell([0]).real, closeTo(0.0, 1e-10));
          expect(yComp.getCell([0]).imag, closeTo(0.0, 1e-10));

          // Complex64
          final cC64 = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(1.0, 0.0)],
            [2],
            DType.complex64,
          );
          final xC64 = NDArray.fromList(
            [Complex(2.0, 3.0)],
            [1],
            DType.complex64,
          );
          final yC64 = polyval(cC64, xC64);
          expect(yC64.dtype, equals(DType.complex64));
          expect(yC64.getCell([0]).real, closeTo(3.0, 1e-4));
          expect(yC64.getCell([0]).imag, closeTo(3.0, 1e-4));

          // Integer inputs upcasted to Float64
          final cInt = NDArray.fromList(
            [1, 2, 3],
            [3],
            DType.int32,
          ); // x^2 + 2x + 3
          final xInt = NDArray.fromList([0, 1, 2], [3], DType.int32);
          final yInt = polyval(cInt, xInt);
          expect(yInt.dtype, equals(DType.float64));
          expect(yInt.getCell([0]), closeTo(3.0, 1e-10));
          expect(yInt.getCell([1]), closeTo(6.0, 1e-10));
          expect(yInt.getCell([2]), closeTo(11.0, 1e-10));

          // Strided non-contiguous views
          final mat = NDArray.fromList(
            [0.0, 99.0, 1.0, 88.0, 2.0, 77.0],
            [3, 2],
            DType.float64,
          );
          final col0 = mat.slice([Slice.all(), Index(0)]); // [0.0, 1.0, 2.0]
          final yStrided = polyval(cInt.astype(DType.float64), col0);
          expect(yStrided.getCell([0]), closeTo(3.0, 1e-10));
          expect(yStrided.getCell([1]), closeTo(6.0, 1e-10));
          expect(yStrided.getCell([2]), closeTo(11.0, 1e-10));
        });
      },
    );

    test(
      'polyfit least-squares polynomial fit across degrees, weights and rcond',
      () {
        NDArray.scope(() {
          // Degree 0 (constant fit)
          final x0 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final y0 = NDArray.fromList([5.0, 5.0, 5.0, 5.0], [4], DType.float64);
          final c0 = polyfit(x0, y0, 0);
          expect(c0.shape, equals([1]));
          expect(c0.getCell([0]), closeTo(5.0, 1e-5));

          // Degree 1 (linear fit: y = 2x + 3)
          final x1 = NDArray.fromList([0.0, 1.0, 2.0, 3.0], [4], DType.float64);
          final y1 = NDArray.fromList([3.0, 5.0, 7.0, 9.0], [4], DType.float64);
          final c1 = polyfit(x1, y1, 1);
          expect(c1.shape, equals([2]));
          expect(c1.getCell([0]), closeTo(2.0, 1e-5));
          expect(c1.getCell([1]), closeTo(3.0, 1e-5));

          // Degree 2 (quadratic fit: y = x^2 - 4x + 3)
          final x2 = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );
          final y2 = NDArray.fromList(
            [3.0, 0.0, -1.0, 0.0, 3.0],
            [5],
            DType.float64,
          );
          final c2 = polyfit(x2, y2, 2);
          expect(c2.shape, equals([3]));
          expect(c2.getCell([0]), closeTo(1.0, 1e-5));
          expect(c2.getCell([1]), closeTo(-4.0, 1e-5));
          expect(c2.getCell([2]), closeTo(3.0, 1e-5));

          // Degree 3 (cubic fit: y = 2x^3 - x^2 + 3x - 1)
          final x3 = NDArray.fromList(
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [5],
            DType.float64,
          );
          final y3 = NDArray.fromList(
            [-7.0, -1.0, 3.0, 17.0, 53.0],
            [5],
            DType.float64,
          );
          final c3 = polyfit(x3, y3, 3);
          expect(c3.shape, equals([4]));
          expect(c3.getCell([0]), closeTo(2.0, 1e-4));
          expect(c3.getCell([1]), closeTo(-1.0, 1e-4));
          expect(c3.getCell([2]), closeTo(3.0, 1e-4));
          expect(c3.getCell([3]), closeTo(-1.0, 1e-4));

          // Weighted fit
          final w = NDArray.fromList(
            [1.0, 1.0, 2.0, 1.0, 1.0],
            [5],
            DType.float64,
          );
          final cWeighted = polyfit(x2, y2, 2, w: w);
          expect(cWeighted.getCell([0]), closeTo(1.0, 1e-4));

          // Explicit rcond
          final cRcond = polyfit(x2, y2, 2, rcond: 1e-12);
          expect(cRcond.getCell([0]), closeTo(1.0, 1e-4));

          // Float32 polyfit
          final xF32 = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float32);
          final yF32 = NDArray.fromList(
            [1.0, 3.0, 5.0],
            [3],
            DType.float32,
          ); // y = 2x + 1
          final cF32 = polyfit(xF32, yF32, 1);
          expect(cF32.dtype, equals(DType.float32));
          expect(cF32.getCell([0]), closeTo(2.0, 1e-4));
          expect(cF32.getCell([1]), closeTo(1.0, 1e-4));

          // Complex128 polyfit
          final xComp = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)],
            [3],
            DType.complex128,
          );
          final yComp = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 1.0), Complex(3.0, 1.0)],
            [3],
            DType.complex128,
          ); // y = x + (1+i)
          final cComp = polyfit(xComp, yComp, 1);
          expect(cComp.dtype, equals(DType.complex128));
          expect(cComp.getCell([0]).real, closeTo(1.0, 1e-5));
          expect(cComp.getCell([0]).imag, closeTo(0.0, 1e-5));
          expect(cComp.getCell([1]).real, closeTo(1.0, 1e-5));
          expect(cComp.getCell([1]).imag, closeTo(1.0, 1e-5));
        });
      },
    );

    test(
      'roots for linear, quadratic, cubic, quartic and degenerate polynomials',
      () {
        NDArray.scope(() {
          // Degree 0 (constant only) -> empty roots
          final pConst = NDArray.fromList([99.0], [1], DType.float64);
          expect(roots(pConst).shape, equals([0]));

          // Leading zeros
          final pLead = NDArray.fromList(
            [0.0, 0.0, 2.0, -6.0],
            [4],
            DType.float64,
          ); // 2x - 6 = 0 -> x = 3
          final rLead = roots(pLead);
          expect(rLead.shape, equals([1]));
          expect(rLead.getCell([0]).real, closeTo(3.0, 1e-10));

          // Linear: 3x + 12 = 0 -> root -4
          final pLin = NDArray.fromList([3.0, 12.0], [2], DType.float64);
          final rLin = roots(pLin);
          expect(rLin.shape, equals([1]));
          expect(rLin.getCell([0]).real, closeTo(-4.0, 1e-10));

          // Quadratic: x^2 - 7x + 12 = 0 -> roots 3, 4
          final pQuad = NDArray.fromList([1.0, -7.0, 12.0], [3], DType.float64);
          final rQuad = roots(pQuad);
          expect(rQuad.shape, equals([2]));
          final rQuadSorted = [
            rQuad.getCell([0]).real,
            rQuad.getCell([1]).real,
          ]..sort();
          expect(rQuadSorted[0], closeTo(3.0, 1e-5));
          expect(rQuadSorted[1], closeTo(4.0, 1e-5));

          // Quadratic with complex conjugate roots: x^2 + 4 = 0 -> roots +/- 2i
          final pCompRoots = NDArray.fromList(
            [1.0, 0.0, 4.0],
            [3],
            DType.float64,
          );
          final rCompQuad = roots(pCompRoots);
          expect(rCompQuad.shape, equals([2]));
          final imagParts = [
            rCompQuad.getCell([0]).imag,
            rCompQuad.getCell([1]).imag,
          ]..sort();
          expect(imagParts[0], closeTo(-2.0, 1e-5));
          expect(imagParts[1], closeTo(2.0, 1e-5));

          // Cubic: (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6 = 0 -> roots 1, 2, 3
          final pCubic = NDArray.fromList(
            [1.0, -6.0, 11.0, -6.0],
            [4],
            DType.float64,
          );
          final rCubic = roots(pCubic);
          expect(rCubic.shape, equals([3]));
          final rCubicSorted = [
            rCubic.getCell([0]).real,
            rCubic.getCell([1]).real,
            rCubic.getCell([2]).real,
          ]..sort();
          expect(rCubicSorted[0], closeTo(1.0, 1e-5));
          expect(rCubicSorted[1], closeTo(2.0, 1e-5));
          expect(rCubicSorted[2], closeTo(3.0, 1e-5));

          // Float32 input
          final pF32 = NDArray.fromList(
            [1.0, -3.0, 2.0],
            [3],
            DType.float32,
          ); // roots 1, 2
          final rF32 = roots(pF32);
          expect(rF32.shape, equals([2]));

          // Complex128 polynomial: x^2 - (1+i)x + i = (x - 1)(x - i) = 0 -> roots 1, i
          final pComp = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(-1.0, -1.0), Complex(0.0, 1.0)],
            [3],
            DType.complex128,
          );
          final rComp = roots(pComp);
          expect(rComp.shape, equals([2]));
        });
      },
    );

    test('Standard polynomial error paths', () {
      NDArray.scope(() {
        final cEmpty = NDArray.zeros([0], DType.float64);
        final c2D = NDArray.zeros([2, 2], DType.float64);
        final x = NDArray.zeros([3], DType.float64);
        final y = NDArray.zeros([3], DType.float64);

        expect(() => polyval(cEmpty, x), throwsArgumentError);
        expect(() => polyval(c2D, x), throwsArgumentError);
        expect(
          () => polyval(
            NDArray.zeros([2], DType.float64),
            x,
            out: NDArray.zeros([4], DType.float64),
          ),
          throwsArgumentError,
        );

        expect(() => polyfit(c2D, y, 1), throwsArgumentError);
        expect(() => polyfit(x, c2D, 1), throwsArgumentError);
        expect(
          () => polyfit(x, NDArray.zeros([4], DType.float64), 1),
          throwsArgumentError,
        );
        expect(() => polyfit(x, y, -1), throwsArgumentError);
        expect(() => polyfit(x, y, 5), throwsArgumentError); // m <= deg
        expect(
          () => polyfit(x, y, 1, w: NDArray.zeros([4], DType.float64)),
          throwsArgumentError,
        );

        expect(() => roots(c2D), throwsArgumentError);

        final dispArr = NDArray.zeros([3], DType.float64)..dispose();
        expect(() => polyval(dispArr, x), throwsStateError);
        expect(() => polyfit(dispArr, y, 1), throwsStateError);
        expect(() => roots(dispArr), throwsStateError);
      });
    });
  });

  group(
    '3. Fast Fourier Transforms (FFT, IFFT, RFFT, IRFFT, FFT2, IFFT2, FFTN, IFFTN, Shifts, Frequencies)',
    () {
      test('1D FFT and IFFT across all 15 DTypes and odd/even lengths', () {
        NDArray.scope(() {
          // Test all numeric, boolean, and complex DTypes
          for (final dt in DType.specs) {
            final len = 8;
            final a = NDArray.zeros([len], dt);
            // Set a pulse at index 0
            if (dt == DType.boolean) {
              a.setCellFlat(0, true);
            } else if (dt.isComplex) {
              a.setCellFlat(0, Complex(1.0, 0.0));
            } else if (dt.isInteger) {
              a.setCellFlat(0, 1);
            } else {
              a.setCellFlat(0, 1.0);
            }

            final res = fft(a);
            final expectedDType = (dt == DType.float32 || dt == DType.complex64)
                ? DType.complex64
                : DType.complex128;
            expect(res.dtype, equals(expectedDType));
            expect(res.shape, equals([len]));

            // FFT of unit impulse is all 1s
            for (var i = 0; i < len; i++) {
              final c = res.getCell([i]);
              expect(c.real, closeTo(1.0, 1e-4));
              expect(c.imag, closeTo(0.0, 1e-4));
            }

            // IFFT round-trip
            final inv = ifft((res as NDArray<AnySpec>));
            expect(inv.shape, equals([len]));
            final c0 = inv.getCell([0]);
            expect(c0.real, closeTo(1.0, 1e-4));
            expect(c0.imag, closeTo(0.0, 1e-4));
            for (var i = 1; i < len; i++) {
              final ci = inv.getCell([i]);
              expect(ci.real, closeTo(0.0, 1e-4));
              expect(ci.imag, closeTo(0.0, 1e-4));
            }
          }

          // Odd prime lengths: 3, 5, 7, 11, 13, 17, 31
          for (final n in [3, 5, 7, 11, 13, 17, 31]) {
            final x = NDArray.fromList(List.generate(n, (i) => i.toDouble()), [
              n,
            ], DType.float64);
            final f = fft(x);
            expect(f.shape, equals([n]));
            final back = ifft(f);
            expect(back.shape, equals([n]));
            for (var i = 0; i < n; i++) {
              expect(back.getCell([i]).real, closeTo(i.toDouble(), 1e-8));
              expect(back.getCell([i]).imag, closeTo(0.0, 1e-8));
            }
          }
        });
      });

      test(
        '1D FFT along custom axis and with custom target length n (padding & cropping)',
        () {
          NDArray.scope(() {
            final mat = NDArray.fromList(
              [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
              [2, 4],
              DType.float64,
            );

            // FFT along axis 1 (default)
            final resAxis1 = fft(mat, axis: 1);
            expect(resAxis1.shape, equals([2, 4]));
            expect(resAxis1.getCell([0, 0]).real, closeTo(1.0, 1e-10));
            expect(resAxis1.getCell([1, 0]).real, closeTo(2.0, 1e-10));

            // FFT along axis 0
            final resAxis0 = fft(mat, axis: 0);
            expect(resAxis0.shape, equals([2, 4]));
            expect(
              resAxis0.getCell([0, 0]).real,
              closeTo(3.0, 1e-10),
            ); // 1 + 2 = 3
            expect(
              resAxis0.getCell([1, 0]).real,
              closeTo(-1.0, 1e-10),
            ); // 1 - 2 = -1

            // Negative axis -2
            final resNegAxis = fft(mat, axis: -2);
            expect(resNegAxis.shape, equals([2, 4]));

            // Target length n padding
            final padded = fft(mat, n: 8, axis: 1);
            expect(padded.shape, equals([2, 8]));

            // Target length n cropping
            final cropped = fft(mat, n: 2, axis: 1);
            expect(cropped.shape, equals([2, 2]));
          });
        },
      );

      test('rfft and irfft with even and odd lengths and batch axes', () {
        NDArray.scope(() {
          // Even length: N=6 -> rfft length 4
          final sigEven = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );
          final rEven = rfft(sigEven);
          expect(rEven.shape, equals([4]));
          expect(
            rEven.getCell([0]).real,
            closeTo(21.0, 1e-10),
          ); // DC = sum = 21

          final invEven = irfft(rEven, n: 6);
          expect(invEven.shape, equals([6]));
          for (var i = 0; i < 6; i++) {
            expect(invEven.getCell([i]), closeTo(sigEven.getCell([i]), 1e-10));
          }

          // Odd length: N=7 -> rfft length 4
          final sigOdd = NDArray.fromList(
            [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
            [7],
            DType.float64,
          );
          final rOdd = rfft(sigOdd);
          expect(rOdd.shape, equals([4]));
          final invOdd = irfft(rOdd, n: 7);
          expect(invOdd.shape, equals([7]));
          for (var i = 0; i < 7; i++) {
            expect(invOdd.getCell([i]), closeTo(sigOdd.getCell([i]), 1e-10));
          }

          // Float32 rfft / irfft
          final sigF32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
          );
          final rF32 = rfft(sigF32);
          expect(rF32.dtype, equals(DType.complex64));
          final invF32 = irfft(rF32, n: 4);
          expect(invF32.dtype, equals(DType.float32));
          expect(invF32.getCell([0]), closeTo(1.0, 1e-4));
        });
      });

      test(
        '2D and N-D FFT: fft2, ifft2, fftn, ifftn with custom axes, shapes and clearFFTPlanCache',
        () {
          NDArray.scope(() {
            // 2D FFT
            final img = NDArray.zeros([4, 4], DType.float64);
            img[[0, 0]] = 16.0;
            final f2D = fft2(img);
            expect(f2D.shape, equals([4, 4]));
            for (var i = 0; i < 4; i++) {
              for (var j = 0; j < 4; j++) {
                expect(f2D.getCell([i, j]).real, closeTo(16.0, 1e-10));
              }
            }
            final if2D = ifft2(f2D);
            expect(if2D.getCell([0, 0]).real, closeTo(16.0, 1e-10));
            expect(if2D.getCell([0, 1]).real, closeTo(0.0, 1e-10));

            // 2D FFT with shape parameter s
            final f2DShape = fft2(img, s: [8, 8]);
            expect(f2DShape.shape, equals([8, 8]));

            // 3D FFT (fftn / ifftn) with custom axes
            final tensor3D = NDArray.zeros([2, 3, 4], DType.float64);
            tensor3D[[0, 0, 0]] = 24.0;
            final f3D = fftn(tensor3D, axes: [0, 1, 2]);
            expect(f3D.shape, equals([2, 3, 4]));
            final if3D = ifftn(f3D, axes: [0, 1, 2]);
            expect(if3D.getCell([0, 0, 0]).real, closeTo(24.0, 1e-10));
            expect(if3D.getCell([1, 2, 3]).real, closeTo(0.0, 1e-10));

            // 3D FFT with shape parameter s
            final f3DShape = fftn(tensor3D, s: [4, 4, 4]);
            expect(f3DShape.shape, equals([4, 4, 4]));

            // 4D FFT
            final tensor4D = NDArray.zeros([2, 2, 2, 2], DType.float64);
            tensor4D[[0, 0, 0, 0]] = 16.0;
            final f4D = fftn(tensor4D, axes: [0, 1, 2, 3]);
            expect(f4D.shape, equals([2, 2, 2, 2]));
            final if4D = ifftn(f4D, axes: [0, 1, 2, 3]);
            expect(if4D.getCell([0, 0, 0, 0]).real, closeTo(16.0, 1e-10));

            // Clear plan cache
            clearFFTPlanCache();
          });
        },
      );

      test('fftfreq and rfftfreq with various lengths and spacings', () {
        NDArray.scope(() {
          final f8 = fftfreq(8);
          expect(f8.shape, equals([8]));
          expect(
            f8.toList(),
            equals([0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]),
          );

          final f7 = fftfreq(7, d: 2.0);
          expect(f7.shape, equals([7]));
          // values = [0, 1, 2, 3, -3, -2, -1] / (7*2) = [0/14, 1/14, 2/14, 3/14, -3/14, -2/14, -1/14]
          expect(f7.getCell([0]), closeTo(0.0, 1e-10));
          expect(f7.getCell([1]), closeTo(1.0 / 14.0, 1e-10));
          expect(f7.getCell([4]), closeTo(-3.0 / 14.0, 1e-10));

          final rf7 = rfftfreq(7, d: 2.0);
          expect(rf7.shape, equals([4]));
          expect(rf7.getCell([0]), closeTo(0.0, 1e-10));
          expect(rf7.getCell([3]), closeTo(3.0 / 14.0, 1e-10));
        });
      });

      test('fftshift and ifftshift along 1D, 2D, 3D, and 4D axes', () {
        NDArray.scope(() {
          // 1D even
          final a1Even = NDArray.fromList([0, 1, 2, 3, 4, 5], [6], DType.int32);
          final s1Even = fftshift(a1Even);
          expect(s1Even.toList(), equals([3, 4, 5, 0, 1, 2]));
          expect(ifftshift(s1Even).toList(), equals([0, 1, 2, 3, 4, 5]));

          // 1D odd
          final a1Odd = NDArray.fromList([0, 1, 2, 3, 4], [5], DType.int32);
          final s1Odd = fftshift(a1Odd);
          expect(s1Odd.toList(), equals([3, 4, 0, 1, 2]));
          expect(ifftshift(s1Odd).toList(), equals([0, 1, 2, 3, 4]));

          // 2D with single axis
          final a2 = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
          final sAxis0 = fftshift(a2, axes: 0);
          expect(sAxis0.toList(), equals([4, 5, 6, 1, 2, 3]));
          expect(ifftshift(sAxis0, axes: 0).toList(), equals(a2.toList()));

          final sAxis1 = fftshift(a2, axes: 1);
          expect(sAxis1.toList(), equals([3, 1, 2, 6, 4, 5]));
          expect(ifftshift(sAxis1, axes: 1).toList(), equals(a2.toList()));

          // Negative axis
          final sNegAxis = fftshift(a2, axes: -1);
          expect(sNegAxis.toList(), equals([3, 1, 2, 6, 4, 5]));

          // 3D
          final a3 = NDArray.arange(
            0.0,
            24.0,
            dtype: DType.float64,
          ).reshape([2, 3, 4]);
          final s3 = fftshift(a3, axes: [0, 1, 2]);
          expect(ifftshift(s3, axes: [0, 1, 2]).toList(), equals(a3.toList()));
        });
      });

      test('FFT error paths & validations', () {
        NDArray.scope(() {
          final a0D = NDArray.scalar(1.0, dtype: DType.float64);
          final a1D = NDArray.zeros([4], DType.float64);

          expect(() => fft(a0D), throwsArgumentError);
          expect(() => ifft(a0D), throwsArgumentError);
          expect(() => rfft(a0D), throwsArgumentError);
          expect(
            () => irfft(NDArray.zeros([0], DType.complex128)),
            throwsArgumentError,
          );

          expect(() => fft(a1D, axis: 5), throwsRangeError);
          expect(() => fft(a1D, n: 0), throwsArgumentError);
          expect(() => fft(a1D, n: -2), throwsArgumentError);

          expect(
            () => fft(a1D, out: NDArray.zeros([3], DType.complex128)),
            throwsArgumentError,
          );
          expect(() => fftshift(a1D, axes: 5), throwsRangeError);
          expect(() => fftshift(a1D, axes: [0, 0]), throwsArgumentError);
          expect(() => fftshift(a1D, axes: 'invalid'), throwsArgumentError);

          final dispArr = NDArray.zeros([4], DType.float64)..dispose();
          expect(() => fft(dispArr), throwsStateError);
          expect(() => ifft(dispArr), throwsStateError);
          expect(() => fftshift(dispArr), throwsStateError);
          expect(() => ifftshift(dispArr), throwsStateError);
        });
      });
    },
  );

  group('4. Core NDArray Constructors, Advanced Indexing, Slicing & Scopes', () {
    test(
      'Constructors and Spacers (linspace, logspace, geomspace, eye, arange, full, view)',
      () {
        NDArray.scope(() {
          // linspace
          final lin = linspace(0.0, 1.0, 5, dtype: DType.float64);
          expect(lin.shape, equals([5]));
          expect(lin.toList(), equals([0.0, 0.25, 0.5, 0.75, 1.0]));

          final linNoEnd = linspace(
            0.0,
            1.0,
            4,
            endpoint: false,
            dtype: DType.float64,
          );
          expect(linNoEnd.toList(), equals([0.0, 0.25, 0.5, 0.75]));

          final (samples: linS, step: linStep) = linspaceWithStep(
            0.0,
            10.0,
            5,
            dtype: DType.float64,
          );
          expect(linStep, equals(2.5));
          expect(linS.toList(), equals([0.0, 2.5, 5.0, 7.5, 10.0]));

          // linspaceGrid
          final startGrid = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stopGrid = NDArray.fromList([1.0, 12.0], [2], DType.float64);
          final grid = linspaceGrid(startGrid, stopGrid, 3);
          expect(grid.shape, equals([3, 2]));
          expect(grid.getCell([0, 0]), equals(0.0));
          expect(grid.getCell([0, 1]), equals(10.0));
          expect(grid.getCell([2, 0]), equals(1.0));
          expect(grid.getCell([2, 1]), equals(12.0));

          // logspace: 10^0 to 10^3 in 4 steps -> [1, 10, 100, 1000]
          final log = logspace(0.0, 3.0, 4, dtype: DType.float64);
          expect(log.shape, equals([4]));
          expect(log.getCell([0]), closeTo(1.0, 1e-10));
          expect(log.getCell([1]), closeTo(10.0, 1e-10));
          expect(log.getCell([2]), closeTo(100.0, 1e-10));
          expect(log.getCell([3]), closeTo(1000.0, 1e-10));

          // geomspace: 1 to 1000 in 4 steps -> [1, 10, 100, 1000]
          final geom = geomspace(1.0, 1000.0, 4, dtype: DType.float64);
          expect(geom.shape, equals([4]));
          expect(geom.getCell([0]), closeTo(1.0, 1e-10));
          expect(geom.getCell([1]), closeTo(10.0, 1e-10));
          expect(geom.getCell([2]), closeTo(100.0, 1e-10));
          expect(geom.getCell([3]), closeTo(1000.0, 1e-10));

          // eye
          final eyeMat = NDArray.eye(3, DType.float64);
          expect(eyeMat.shape, equals([3, 3]));
          expect(eyeMat.getCell([0, 0]), equals(1.0));
          expect(eyeMat.getCell([1, 1]), equals(1.0));
          expect(eyeMat.getCell([2, 2]), equals(1.0));
          expect(eyeMat.getCell([0, 1]), equals(0.0));

          final eyeBool = NDArray.eye(2, DType.boolean);
          expect(eyeBool.getCell([0, 0]), isTrue);
          expect(eyeBool.getCell([0, 1]), isFalse);

          // arange with non-unit and negative steps
          final ar1 = NDArray.arange(
            10.0,
            0.0,
            step: -2.0,
            dtype: DType.float64,
          );
          expect(ar1.toList(), equals([10.0, 8.0, 6.0, 4.0, 2.0]));

          // full
          final fullArr = NDArray.full([2, 3], 7.5, dtype: DType.float64);
          expect(fullArr.toList(), equals([7.5, 7.5, 7.5, 7.5, 7.5, 7.5]));

          // view
          final parent = NDArray.arange(0.0, 10.0, dtype: DType.float64);
          final customView = NDArray.view(
            parent,
            shape: [3],
            strides: [2],
            offsetElements: 1,
          );
          expect(customView.toList(), equals([1.0, 3.0, 5.0]));
        });
      },
    );

    test(
      'Advanced Indexing: take_along_axis, put_along_axis, choose, select, where',
      () {
        NDArray.scope(() {
          // take_along_axis
          final a = NDArray.fromList(
            [10, 20, 30, 40, 50, 60],
            [2, 3],
            DType.int32,
          );
          final idx = NDArray.fromList([2, 0, 1, 2], [2, 2], DType.int32);
          final taken = take_along_axis(a, idx, 1);
          expect(taken.shape, equals([2, 2]));
          expect(taken.toList(), equals([30, 10, 50, 60]));

          // put_along_axis
          final dest = NDArray.fromList(
            [10, 20, 30, 40, 50, 60],
            [2, 3],
            DType.int32,
          );
          final vals = NDArray.fromList([99, 88, 77, 66], [2, 2], DType.int32);
          put_along_axis(dest, idx, vals, 1);
          expect(dest.toList(), equals([88, 20, 99, 40, 77, 66]));

          // choose
          final choices = [
            NDArray.fromList([10, 20, 30], [3], DType.int32),
            NDArray.fromList([100, 200, 300], [3], DType.int32),
            NDArray.fromList([1000, 2000, 3000], [3], DType.int32),
          ];
          final selector = NDArray.fromList([0, 1, 2], [3], DType.int32);
          final chosen = choose(selector, choices);
          expect(chosen.toList(), equals([10, 200, 3000]));

          // select
          final c1 = NDArray.fromList([true, false, false], [3], DType.boolean);
          final c2 = NDArray.fromList([false, true, false], [3], DType.boolean);
          final ch1 = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final ch2 = NDArray.fromList([10, 20, 30], [3], DType.int32);
          final selRes = select([c1, c2], [ch1, ch2], defaultValue: 99);
          expect(selRes.toList(), equals([1, 20, 99]));

          // where
          final cond = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final wX = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final wY = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [4],
            DType.float64,
          );
          final wRes = where(cond, wX, wY) as NDArray<Float64>;
          expect(wRes.toList(), equals([1.0, 20.0, 3.0, 40.0]));

          final coords = where(cond) as List<NDArray<AnySpec>>;
          expect(coords.length, equals(1));
          expect(coords[0].toList(), equals([0, 2]));
        });
      },
    );

    test(
      'Reshaping, Slicing with Negative Steps, Transposition & Operators',
      () {
        NDArray.scope(() {
          final a = NDArray.arange(
            0.0,
            12.0,
            dtype: DType.float64,
          ).reshape([3, 4]);
          expect(a.shape, equals([3, 4]));

          // Negative step slice
          final sRev = a.slice([Slice(step: -1), Slice(step: -1)]);
          expect(sRev.shape, equals([3, 4]));
          expect(sRev.getCell([0, 0]), equals(11.0));
          expect(sRev.getCell([2, 3]), equals(0.0));

          // expandDims and squeeze
          final exp = a.expandDims(1);
          expect(exp.shape, equals([3, 1, 4]));
          final sq = exp.squeeze();
          expect(sq.shape, equals([3, 4]));

          // swapaxes
          final swapped = a.swapaxes(0, 1);
          expect(swapped.shape, equals([4, 3]));
          expect(swapped.getCell([0, 1]), equals(4.0));

          // moveaxis
          final tensor = NDArray.zeros([2, 3, 4], DType.float64);
          final moved = tensor.moveaxis(0, 2);
          expect(moved.shape, equals([3, 4, 2]));

          // transpose
          final transposed = a.transpose([1, 0]);
          expect(transposed.shape, equals([4, 3]));

          // Arithmetic & bitwise operators
          final a1 = NDArray.fromList([10, 20, 30], [3], DType.int32);
          final a2 = NDArray.fromList([1, 2, 3], [3], DType.int32);

          final addRes = a1 + a2;
          expect(addRes.toList(), equals([11, 22, 33]));

          final subRes = a1 - a2;
          expect(subRes.toList(), equals([9, 18, 27]));

          final mulRes = a1 * a2;
          expect(mulRes.toList(), equals([10, 40, 90]));

          final divRes = a1 / a2;
          expect(divRes.toList(), equals([10.0, 10.0, 10.0]));

          final fdivRes = a1 ~/ a2;
          expect(fdivRes.toList(), equals([10, 10, 10]));

          final remRes = a1 % a2;
          expect(remRes.toList(), equals([0, 0, 0]));

          final andRes = a1 & a2;
          expect(andRes.shape, equals([3]));

          final orRes = a1 | a2;
          expect(orRes.shape, equals([3]));

          final xorRes = a1 ^ a2;
          expect(xorRes.shape, equals([3]));

          final notRes = ~a1;
          expect(notRes.shape, equals([3]));

          final shlRes = a1 << 1;
          expect(shlRes.toList(), equals([20, 40, 60]));

          final shrRes = a1 >> 1;
          expect(shrRes.toList(), equals([5, 10, 15]));
        });
      },
    );

    test('Complex Class methods, equality and properties', () {
      final c1 = Complex(3.0, 4.0);
      expect(c1.abs, closeTo(5.0, 1e-10));
      expect(c1.arg, closeTo(math.atan2(4.0, 3.0), 1e-10));
      expect(c1.toString(), contains('3.0 + 4.0i'));

      final c2 = Complex(1.0, -2.0);
      final sum = c1 + c2;
      expect(sum.real, equals(4.0));
      expect(sum.imag, equals(2.0));

      final diff = c1 - c2;
      expect(diff.real, equals(2.0));
      expect(diff.imag, equals(6.0));

      final prod = c1 * c2;
      // (3+4i)*(1-2i) = 3 - 6i + 4i + 8 = 11 - 2i
      expect(prod.real, equals(11.0));
      expect(prod.imag, equals(-2.0));

      final quot = c1 / c2;
      // (3+4i)/(1-2i) = (3+4i)(1+2i)/5 = (3 + 6i + 4i - 8)/5 = (-5 + 10i)/5 = -1 + 2i
      expect(quot.real, closeTo(-1.0, 1e-10));
      expect(quot.imag, closeTo(2.0, 1e-10));

      final pwNum = c1.pow(2);
      expect(
        pwNum.real,
        closeTo(-7.0, 1e-10),
      ); // (3+4i)^2 = 9 - 16 + 24i = -7 + 24i
      expect(pwNum.imag, closeTo(24.0, 1e-10));

      final pwComp = c1.pow(Complex(0.0, 0.0));
      expect(pwComp.real, equals(1.0));
      expect(pwComp.imag, equals(0.0));

      final cLog = c1.log();
      expect(cLog.real, closeTo(math.log(5.0), 1e-10));
      expect(cLog.imag, closeTo(math.atan2(4.0, 3.0), 1e-10));

      // Equality and hashcode
      expect(c1 == Complex(3.0, 4.0), isTrue);
      expect(c1 == c2, isFalse);
      expect(c1.hashCode, equals(Complex(3.0, 4.0).hashCode));
    });

    test('NDArray scope, detachToParentScope and memory management', () {
      NDArray<Float64>? detached;
      NDArray.scope(() {
        final inner = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        detached = inner.detachToParentScope();
      });

      // Detached array should still be valid outside inner scope
      expect(detached!.isDisposed, isFalse);
      expect(detached!.toList(), equals([1.0, 2.0, 3.0]));
      detached!.dispose();
      expect(detached!.isDisposed, isTrue);
    });
  });
}
