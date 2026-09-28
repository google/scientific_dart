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
  group('Comprehensive Math Operators Tests', () {
    group('Bitwise Operations', () {
      test(
        'bitwiseAnd, bitwiseOr, bitwiseXor across int64, int32, int16, uint8',
        () {
          NDArray.scope(() {
            // Int64
            final a64 = NDArray<Int64>.fromList(
              [0x12, 0x34, 0x56],
              [3],
              DType.int64,
            );
            final b64 = NDArray<Int64>.fromList(
              [0x0F, 0xF0, 0xFF],
              [3],
              DType.int64,
            );
            expect(
              bitwiseAnd(a64, b64).toList(),
              equals([0x12 & 0x0F, 0x34 & 0xF0, 0x56 & 0xFF]),
            );
            expect(
              bitwiseOr(a64, b64).toList(),
              equals([0x12 | 0x0F, 0x34 | 0xF0, 0x56 | 0xFF]),
            );
            expect(
              bitwiseXor(a64, b64).toList(),
              equals([0x12 ^ 0x0F, 0x34 ^ 0xF0, 0x56 ^ 0xFF]),
            );

            // Int32
            final a32 = NDArray<Int32>.fromList([10, 20, 30], [3], DType.int32);
            final b32 = NDArray<Int32>.fromList([3, 7, 15], [3], DType.int32);
            expect(
              bitwiseAnd(a32, b32).toList(),
              equals([10 & 3, 20 & 7, 30 & 15]),
            );
            expect(
              bitwiseOr(a32, b32).toList(),
              equals([10 | 3, 20 | 7, 30 | 15]),
            );
            expect(
              bitwiseXor(a32, b32).toList(),
              equals([10 ^ 3, 20 ^ 7, 30 ^ 15]),
            );

            // Int16
            final a16 = NDArray<Int16>.fromList(
              [100, 200, 300],
              [3],
              DType.int16,
            );
            final b16 = NDArray<Int16>.fromList(
              [50, 100, 150],
              [3],
              DType.int16,
            );
            expect(
              bitwiseAnd(a16, b16).toList(),
              equals([100 & 50, 200 & 100, 300 & 150]),
            );
            expect(
              bitwiseOr(a16, b16).toList(),
              equals([100 | 50, 200 | 100, 300 | 150]),
            );
            expect(
              bitwiseXor(a16, b16).toList(),
              equals([100 ^ 50, 200 ^ 100, 300 ^ 150]),
            );

            // Uint8
            final a8 = NDArray<Uint8>.fromList([0xAA, 0x55], [2], DType.uint8);
            final b8 = NDArray<Uint8>.fromList([0x0F, 0xF0], [2], DType.uint8);
            expect(
              bitwiseAnd(a8, b8).toList(),
              equals([0xAA & 0x0F, 0x55 & 0xF0]),
            );
            expect(
              bitwiseOr(a8, b8).toList(),
              equals([0xAA | 0x0F, 0x55 | 0xF0]),
            );
            expect(
              bitwiseXor(a8, b8).toList(),
              equals([0xAA ^ 0x0F, 0x55 ^ 0xF0]),
            );
          });
        },
      );

      test('invert (~x) on integer arrays', () {
        NDArray.scope(() {
          final i32 = NDArray<Int32>.fromList(
            [0, -1, 1, 100],
            [4],
            DType.int32,
          );
          expect(invert(i32).toList(), equals([~0, ~(-1), ~1, ~100]));

          final i64 = NDArray<Int64>.fromList([0, 42], [2], DType.int64);
          expect(invert(i64).toList(), equals([~0, ~42]));

          final u8 = NDArray<Uint8>.fromList([0, 255, 0x0F], [3], DType.uint8);
          expect(invert(u8).toList(), equals([255, 0, 0xF0]));
        });
      });

      test('leftShift and rightShift', () {
        NDArray.scope(() {
          final a = NDArray<Int32>.fromList([1, 2, 4, 8], [4], DType.int32);
          final shift = NDArray<Int32>.fromList([1, 2, 1, 3], [4], DType.int32);

          expect(leftShift(a, shift).toList(), equals([2, 8, 8, 64]));
          expect(rightShift(a, shift).toList(), equals([0, 0, 2, 1]));

          // Scalar shift broadcasting
          final scalarShift = NDArray<Int32>.fromList([2], [1], DType.int32);
          expect(leftShift(a, scalarShift).toList(), equals([4, 8, 16, 32]));
          expect(rightShift(a, scalarShift).toList(), equals([0, 0, 1, 2]));
        });
      });

      test('Bitwise broadcasting, striding, and where masking', () {
        NDArray.scope(() {
          final mat = NDArray<Int64>.fromList(
            [1, 2, 3, 4, 5, 6],
            [2, 3],
            DType.int64,
          );
          final vec = NDArray<Int64>.fromList([10, 20, 30], [3], DType.int64);
          final res = bitwiseOr(mat, vec);
          expect(res.shape, equals([2, 3]));
          expect(
            res.toList(),
            equals([1 | 10, 2 | 20, 3 | 30, 4 | 10, 5 | 20, 6 | 30]),
          );

          // Strided view
          final transposed = mat.transpose(); // [3, 2]
          final invTrans = invert(transposed);
          expect(invTrans.shape, equals([3, 2]));
          expect(invTrans.getCell([0, 0]), equals(~1));
          expect(invTrans.getCell([0, 1]), equals(~4));

          // Where mask
          final mask = NDArray<Boolean>.fromList(
            [true, false, true, false, true, false],
            [2, 3],
            DType.boolean,
          );
          final out = NDArray<Int64>.zeros([2, 3], DType.int64);
          bitwiseOr(mat, vec, where: mask, out: out);
          expect(out.getCell([0, 0]), equals(1 | 10));
          expect(out.getCell([0, 1]), equals(0)); // untouched
          expect(out.getCell([0, 2]), equals(3 | 30));
        });
      });

      test('Bitwise type error validation (non-integer types)', () {
        NDArray.scope(() {
          final f64 = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final f64_2 = NDArray<Float64>.fromList(
            [3.0, 4.0],
            [2],
            DType.float64,
          );

          expect(() => bitwiseAnd(f64, f64_2), throwsArgumentError);
          expect(() => bitwiseOr(f64, f64_2), throwsArgumentError);
          expect(() => bitwiseXor(f64, f64_2), throwsArgumentError);
          expect(() => invert(f64), throwsArgumentError);
          expect(() => leftShift(f64, f64_2), throwsArgumentError);
          expect(() => rightShift(f64, f64_2), throwsArgumentError);
        });
      });
    });

    group('Logical Operations & Comparisons', () {
      test(
        'logicalNot, logicalAnd, logicalOr, logicalXor on boolean arrays',
        () {
          NDArray.scope(() {
            final a = NDArray<Boolean>.fromList(
              [true, true, false, false],
              [4],
              DType.boolean,
            );
            final b = NDArray<Boolean>.fromList(
              [true, false, true, false],
              [4],
              DType.boolean,
            );

            expect(logicalNot(a).toList(), equals([false, false, true, true]));
            expect(
              logicalAnd(a, b).toList(),
              equals([true, false, false, false]),
            );
            expect(logicalOr(a, b).toList(), equals([true, true, true, false]));
            expect(
              logicalXor(a, b).toList(),
              equals([false, true, true, false]),
            );
          });
        },
      );

      test('Truthiness coercion from numeric types (float, int, complex)', () {
        NDArray.scope(() {
          final f64 = NDArray<Float64>.fromList(
            [0.0, 1.5, -2.0, 0.0],
            [4],
            DType.float64,
          );
          expect(logicalNot(f64).toList(), equals([true, false, false, true]));

          final i32 = NDArray<Int32>.fromList([0, 10, 0, -5], [4], DType.int32);
          expect(logicalNot(i32).toList(), equals([true, false, true, false]));

          final c128 = NDArray<Complex128>.fromList(
            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 2.0)],
            [3],
            DType.complex128,
          );
          expect(logicalNot(c128).toList(), equals([true, false, false]));

          final f32 = NDArray<Float32>.fromList([0.0, 2.0], [2], DType.float32);
          final i64 = NDArray<Int64>.fromList([1, 0], [2], DType.int64);
          expect(logicalAnd(f32, i64).toList(), equals([false, false]));
          expect(logicalOr(f32, i64).toList(), equals([true, true]));
          expect(logicalXor(f32, i64).toList(), equals([true, true]));
        });
      });

      test(
        'Comparisons: equal, notEqual, greater, greaterEqual, less, lessEqual',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final b = NDArray<Float64>.fromList(
              [2.0, 2.0, 2.0, 2.0],
              [4],
              DType.float64,
            );

            expect(equal(a, b).toList(), equals([false, true, false, false]));
            expect(notEqual(a, b).toList(), equals([true, false, true, true]));
            expect(greater(a, b).toList(), equals([false, false, true, true]));
            expect(
              greaterEqual(a, b).toList(),
              equals([false, true, true, true]),
            );
            expect(less(a, b).toList(), equals([true, false, false, false]));
            expect(
              lessEqual(a, b).toList(),
              equals([true, true, false, false]),
            );
          });
        },
      );

      test('Comparisons with complex numbers', () {
        NDArray.scope(() {
          final c1 = NDArray<Complex128>.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            DType.complex128,
          );
          final c2 = NDArray<Complex128>.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 5.0)],
            [2],
            DType.complex128,
          );

          expect(equal(c1, c2).toList(), equals([true, false]));
          expect(notEqual(c1, c2).toList(), equals([false, true]));

          // Inequality comparisons throw UnsupportedError on complex
          expect(() => greater(c1, c2), throwsUnsupportedError);
          expect(() => greaterEqual(c1, c2), throwsUnsupportedError);
          expect(() => less(c1, c2), throwsUnsupportedError);
          expect(() => lessEqual(c1, c2), throwsUnsupportedError);
        });
      });

      test('Logical operations broadcasting, where mask, and out buffer', () {
        NDArray.scope(() {
          final mat = NDArray<Boolean>.fromList(
            [true, false, true, false],
            [2, 2],
            DType.boolean,
          );
          final row = NDArray<Boolean>.fromList(
            [true, true],
            [2],
            DType.boolean,
          );

          final res = logicalAnd(mat, row);
          expect(res.shape, equals([2, 2]));
          expect(res.toList(), equals([true, false, true, false]));

          final out = NDArray<Boolean>.zeros([2, 2], DType.boolean);
          final whereMask = NDArray<Boolean>.fromList(
            [true, true, false, false],
            [2, 2],
            DType.boolean,
          );
          logicalOr(mat, row, where: whereMask, out: out);
          expect(out.getCell([0, 0]), isTrue);
          expect(out.getCell([0, 1]), isTrue);
          expect(out.getCell([1, 0]), isFalse); // untouched
        });
      });
    });

    group('Windowing Functions (hanning, hamming)', () {
      test('hanning window lengths: M <= 0, M = 1, M = 5, M = 64', () {
        NDArray.scope(() {
          final w0 = hanning(0);
          expect(w0.shape, equals([0]));

          final w1 = hanning(1);
          expect(w1.shape, equals([1]));
          expect(w1.getCell([0]), equals(1.0));

          final w5 = hanning(5);
          expect(w5.shape, equals([5]));
          expect(w5.getCell([0]), closeTo(0.0, 1e-6));
          expect(w5.getCell([4]), closeTo(0.0, 1e-6));
          // Symmetry
          expect(w5.getCell([1]), closeTo(w5.getCell([3]), 1e-6));
          // Peak at center
          expect(w5.getCell([2]), closeTo(1.0, 1e-6));

          final w64 = hanning(64);
          expect(w64.shape, equals([64]));
          for (var i = 0; i < 32; i++) {
            expect(w64.getCell([i]), closeTo(w64.getCell([63 - i]), 1e-6));
          }
        });
      });

      test('hamming window lengths and symmetry', () {
        NDArray.scope(() {
          final w0 = hamming(0);
          expect(w0.shape, equals([0]));

          final w1 = hamming(1);
          expect(w1.shape, equals([1]));
          expect(w1.getCell([0]), equals(1.0));

          final w5 = hamming(5);
          expect(w5.shape, equals([5]));
          // Hamming endpoints are ~0.08
          expect(w5.getCell([0]), closeTo(0.08, 1e-2));
          expect(w5.getCell([4]), closeTo(0.08, 1e-2));
          // Symmetry
          expect(w5.getCell([1]), closeTo(w5.getCell([3]), 1e-6));
          expect(w5.getCell([2]), closeTo(1.0, 1e-6));
        });
      });

      test(
        'Windows custom dtype (Float32, Float64, Int32) and out buffer recycling',
        () {
          NDArray.scope(() {
            final h32 = hanning(8, dtype: DType.float32);
            expect(h32.dtype, equals(DType.float32));
            expect(h32.shape, equals([8]));

            final out = NDArray<Float64>.zeros([8], DType.float64);
            final res = hanning(8, out: out);
            expect(identical(res, out), isTrue);

            final outHamming = NDArray<Float32>.zeros([8], DType.float32);
            final resHamming = hamming(
              8,
              dtype: DType.float32,
              out: outHamming,
            );
            expect(identical(resHamming, outHamming), isTrue);

            // Incompatible out shape error
            final wrongShape = NDArray<Float64>.zeros([4], DType.float64);
            expect(() => hanning(8, out: wrongShape), throwsArgumentError);
          });
        },
      );
    });

    group('Special Math Functions (i0, gamma, erf, sinc, log1p, expm1)', () {
      test('i0 (Bessel I0) across real and complex domains', () {
        NDArray.scope(() {
          final realArr = NDArray<Float64>.fromList(
            [0.0, 1.0, 3.0, 5.0],
            [4],
            DType.float64,
          );
          final resI0 = i0(realArr);
          expect(resI0.getCell([0]), closeTo(1.0, 1e-6)); // I0(0) = 1
          expect(
            resI0.getCell([1]),
            closeTo(1.2660658777520084, 1e-5),
          ); // I0(1)
          expect(resI0.getCell([3]), closeTo(27.23987182289191, 1e-4)); // I0(5)

          // Integer promotion
          final intArr = NDArray<Int32>.fromList([0, 1, 2], [3], DType.int32);
          final resIntI0 = i0(intArr);
          expect(resIntI0.dtype, equals(DType.float64));
          expect(resIntI0.getCell([0]), closeTo(1.0, 1e-6));

          // Complex domain
          final cArr = NDArray<Complex128>.fromList(
            [Complex(0.0, 0.0), Complex(1.0, 1.0), Complex(20.0, 0.0)],
            [3],
            DType.complex128,
          );
          final resCI0 = i0(cArr);
          expect(resCI0.getCell([0]).real, closeTo(1.0, 1e-6));
          expect(resCI0.getCell([0]).imag, closeTo(0.0, 1e-6));
        });
      });

      test('gamma function known integer and half-integer values', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
            [6],
            DType.float64,
          );
          final res = gamma(a);
          expect(res.getCell([0]), closeTo(1.0, 1e-6)); // gamma(1) = 0! = 1
          expect(res.getCell([1]), closeTo(1.0, 1e-6)); // gamma(2) = 1! = 1
          expect(res.getCell([2]), closeTo(2.0, 1e-6)); // gamma(3) = 2! = 2
          expect(res.getCell([3]), closeTo(6.0, 1e-6)); // gamma(4) = 3! = 6
          expect(res.getCell([4]), closeTo(24.0, 1e-6)); // gamma(5) = 4! = 24
          expect(
            res.getCell([5]),
            closeTo(math.sqrt(math.pi), 1e-6),
          ); // gamma(0.5) = sqrt(pi)

          // Complex inputs throw UnsupportedError
          final cArr = NDArray<Complex128>.fromList(
            [Complex(1.0, 1.0)],
            [1],
            DType.complex128,
          );
          expect(() => gamma(cArr), throwsUnsupportedError);
        });
      });

      test('erf error function known values', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [0.0, 1.0, 2.0, 3.0],
            [4],
            DType.float64,
          );
          final res = erf(a);
          expect(res.getCell([0]), closeTo(0.0, 1e-6)); // erf(0) = 0
          expect(res.getCell([1]), closeTo(0.8427007929497149, 1e-5)); // erf(1)
          expect(res.getCell([2]), closeTo(0.9953222650189527, 1e-5)); // erf(2)
          expect(res.getCell([3]), closeTo(0.9999779095030014, 1e-5)); // erf(3)

          // Complex inputs throw UnsupportedError
          final cArr = NDArray<Complex128>.fromList(
            [Complex(1.0, 0.0)],
            [1],
            DType.complex128,
          );
          expect(() => erf(cArr), throwsUnsupportedError);
        });
      });

      test('sinc, log1p, and expm1 functions', () {
        NDArray.scope(() {
          // sinc(x) = sin(pi * x) / (pi * x)
          final x = NDArray<Float64>.fromList(
            [0.0, 0.5, 1.0, -0.5],
            [4],
            DType.float64,
          );
          final resSinc = sinc(x);
          expect(resSinc.getCell([0]), closeTo(1.0, 1e-6)); // sinc(0) = 1
          expect(
            resSinc.getCell([1]),
            closeTo(2.0 / math.pi, 1e-6),
          ); // sinc(0.5) = 2/pi
          expect(resSinc.getCell([2]), closeTo(0.0, 1e-6)); // sinc(1) = 0
          expect(
            resSinc.getCell([3]),
            closeTo(2.0 / math.pi, 1e-6),
          ); // sinc(-0.5) = 2/pi

          // log1p and expm1 precision near zero
          final eps = NDArray<Float64>.fromList(
            [0.0, 1e-12],
            [2],
            DType.float64,
          );
          final resLog1p = log1p(eps);
          expect(resLog1p.getCell([0]), equals(0.0));
          expect(resLog1p.getCell([1]), closeTo(1e-12, 1e-18));

          final resExpm1 = expm1(eps);
          expect(resExpm1.getCell([0]), equals(0.0));
          expect(resExpm1.getCell([1]), closeTo(1e-12, 1e-18));
        });
      });

      test('Special functions with striding, where mask, and out buffers', () {
        NDArray.scope(() {
          final mat = NDArray<Float64>.fromList(
            [0.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final transposed = mat.transpose();
          final resErf = erf(transposed);
          expect(resErf.shape, equals([2, 2]));
          expect(resErf.getCell([0, 0]), closeTo(0.0, 1e-6));
          expect(
            resErf.getCell([0, 1]),
            closeTo(0.9953222650189527, 1e-5),
          ); // val 2.0

          final out = NDArray<Float64>.zeros([2, 2], DType.float64);
          final mask = NDArray<Boolean>.fromList(
            [true, false, false, true],
            [2, 2],
            DType.boolean,
          );
          erf(mat, where: mask, out: out);
          expect(out.getCell([0, 0]), closeTo(0.0, 1e-6));
          expect(out.getCell([0, 1]), equals(0.0)); // untouched
          expect(
            out.getCell([1, 1]),
            closeTo(0.9999779095030014, 1e-5),
          ); // val 3.0
        });
      });
    });

    group('Clip Operations & Edge Cases', () {
      test('clip with scalar bounds across float and integer types', () {
        NDArray.scope(() {
          // Float64
          final f64 = NDArray<Float64>.fromList(
            [1.0, 5.0, 10.0, 15.0],
            [4],
            DType.float64,
          );
          expect(
            clip(f64, min: 3.0, max: 12.0).toList(),
            equals([3.0, 5.0, 10.0, 12.0]),
          );
          expect(clip(f64, min: 4.0).toList(), equals([4.0, 5.0, 10.0, 15.0]));
          expect(clip(f64, max: 8.0).toList(), equals([1.0, 5.0, 8.0, 8.0]));
          expect(clip(f64).toList(), equals([1.0, 5.0, 10.0, 15.0]));

          // Int32
          final i32 = NDArray<Int32>.fromList(
            [-10, 0, 50, 100],
            [4],
            DType.int32,
          );
          expect(clip(i32, min: 0, max: 60).toList(), equals([0, 0, 50, 60]));

          // Uint8
          final u8 = NDArray<Uint8>.fromList(
            [5, 50, 150, 250],
            [4],
            DType.uint8,
          );
          expect(
            clip(u8, min: 20, max: 200).toList(),
            equals([20, 50, 150, 200]),
          );

          // Complex type throws UnsupportedError
          final cArr = NDArray<Complex128>.fromList(
            [Complex(1.0, 2.0)],
            [1],
            DType.complex128,
          );
          expect(() => clip(cArr, min: 0.0), throwsUnsupportedError);
        });
      });

      test('clipArray with broadcasted array bounds', () {
        NDArray.scope(() {
          final mat = NDArray<Float64>.fromList(
            [1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            [2, 3],
            DType.float64,
          );
          final minArr = NDArray<Float64>.fromList(
            [3.0, 3.0, 3.0],
            [3],
            DType.float64,
          );
          final maxArr = NDArray<Float64>.fromList(
            [12.0, 12.0, 12.0],
            [3],
            DType.float64,
          );

          final clipped = clipArray(mat, min: minArr, max: maxArr);
          expect(clipped.shape, equals([2, 3]));
          expect(clipped.toList(), equals([3.0, 5.0, 10.0, 12.0, 12.0, 12.0]));

          // Null min or null max
          final minOnly = clipArray(mat, min: minArr);
          expect(minOnly.toList(), equals([3.0, 5.0, 10.0, 15.0, 20.0, 25.0]));

          final maxOnly = clipArray(mat, max: maxArr);
          expect(maxOnly.toList(), equals([1.0, 5.0, 10.0, 12.0, 12.0, 12.0]));
        });
      });

      test('clipArray with strided inputs, where mask, and out buffer', () {
        NDArray.scope(() {
          final mat = NDArray<Int64>.fromList(
            [10, 20, 30, 40],
            [2, 2],
            DType.int64,
          );
          final transposed = mat.transpose();
          final minArr = NDArray<Int64>.fromList([15, 15], [2], DType.int64);
          final maxArr = NDArray<Int64>.fromList([35, 35], [2], DType.int64);

          final out = NDArray<Int64>.zeros([2, 2], DType.int64);
          final mask = NDArray<Boolean>.fromList(
            [true, false, true, true],
            [2, 2],
            DType.boolean,
          );

          clipArray(
            transposed,
            min: minArr,
            max: maxArr,
            where: mask,
            out: out,
          );
          expect(out.getCell([0, 0]), equals(15)); // 10 clipped to 15
          expect(out.getCell([0, 1]), equals(0)); // untouched by where
          expect(out.getCell([1, 0]), equals(20)); // 20 in bounds
          expect(out.getCell([1, 1]), equals(35)); // 40 clipped to 35
        });
      });

      test('clip and clipArray error cases', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final boolBounds = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );
          final cBounds = NDArray<Complex128>.fromList(
            [Complex(1.0, 0.0), Complex(2.0, 0.0)],
            [2],
            DType.complex128,
          );

          // Boolean bounds throw ArgumentError
          expect(() => clipArray(a, min: boolBounds), throwsArgumentError);
          // Complex bounds throw ArgumentError
          expect(() => clipArray(a, min: cBounds), throwsArgumentError);
        });
      });
    });
  });
}
