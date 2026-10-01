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
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Vectorized Bitwise Operations Tests', () {
    test(
      'bitwiseAnd, bitwiseOr, bitwiseXor basic behavior (int32 and int64)',
      () {
        NDArray.scope(() {
          final a32 = NDArray.fromList([5, 12, 3], [3], DType.int32);
          final b32 = NDArray.fromList([3, 4, 5], [3], DType.int32);

          final a64 = NDArray.fromList([5, 12, 3], [3], DType.int64);
          final b64 = NDArray.fromList([3, 4, 5], [3], DType.int64);

          // bitwiseAnd: 5 & 3 = 1, 12 & 4 = 4, 3 & 5 = 1
          expect(bitwiseAnd(a32, b32).toList(), [1, 4, 1]);
          expect(bitwiseAnd(a64, b64).toList(), [1, 4, 1]);

          // bitwiseOr: 5 | 3 = 7, 12 | 4 = 12, 3 | 5 = 7
          expect(bitwiseOr(a32, b32).toList(), [7, 12, 7]);
          expect(bitwiseOr(a64, b64).toList(), [7, 12, 7]);

          // bitwiseXor: 5 ^ 3 = 6, 12 ^ 4 = 8, 3 ^ 5 = 6
          expect(bitwiseXor(a32, b32).toList(), [6, 8, 6]);
          expect(bitwiseXor(a64, b64).toList(), [6, 8, 6]);
        });
      },
    );

    test('bitwise operations with uint8 and int16 dtypes', () {
      NDArray.scope(() {
        final a8 = NDArray.fromList([15, 240], [2], DType.uint8);
        final b8 = NDArray.fromList([240, 15], [2], DType.uint8);

        final a16 = NDArray.fromList([15, 240], [2], DType.int16);
        final b16 = NDArray.fromList([240, 15], [2], DType.int16);

        expect(bitwiseAnd(a8, b8).toList(), [0, 0]);
        expect(bitwiseAnd(a16, b16).toList(), [0, 0]);

        expect(bitwiseOr(a8, b8).toList(), [255, 255]);
        expect(bitwiseOr(a16, b16).toList(), [255, 255]);

        expect(bitwiseXor(a8, b8).toList(), [255, 255]);
        expect(bitwiseXor(a16, b16).toList(), [255, 255]);
      });
    });

    test('leftShift and rightShift basic and edge case safe shifting', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 4, 8], [4], DType.int32);
        final shift = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);

        // leftShift: 1<<1=2, 2<<2=8, 4<<3=32, 8<<4=128
        expect(leftShift(a, shift).toList(), [2, 8, 32, 128]);

        // rightShift: 2>>1=1, 8>>2=2, 32>>3=4, 128>>4=8
        final shifted = leftShift(a, shift);
        expect(rightShift(shifted, shift).toList(), [1, 2, 4, 8]);

        // Safe shifting behavior (prevents undefined C behavior)
        final badShift = NDArray.fromList([-1, 32, 100, 0], [4], DType.int32);
        expect(leftShift(a, badShift).toList(), [0, 0, 0, 8]);
        expect(rightShift(a, badShift).toList(), [0, 0, 0, 8]);
      });
    });

    test('invert (bitwise NOT) for all four integer dtypes', () {
      NDArray.scope(() {
        final a32 = NDArray.fromList([0, -1, 5], [3], DType.int32);
        final a64 = NDArray.fromList([0, -1, 5], [3], DType.int64);
        final a8 = NDArray.fromList([0, 255, 5], [3], DType.uint8);
        final a16 = NDArray.fromList([0, -1, 5], [3], DType.int16);

        expect(invert(a32).toList(), [-1, 0, -6]);
        expect(invert(a64).toList(), [-1, 0, -6]);
        expect(invert(a8).toList(), [255, 0, 250]);
        expect(invert(a16).toList(), [-1, 0, -6]);
      });
    });

    test('broadcasting with bitwise operations (contiguous & strided)', () {
      NDArray.scope(() {
        // Broadcast scalar-like array [2] (shape [1]) against [3, 4, 5] (shape [3])
        final a = NDArray.fromList([3, 4, 5], [3], DType.int32);
        final b = NDArray.fromList([2], [1], DType.int32);

        expect(bitwiseAnd(a, b).toList(), [2, 0, 0]); // 3&2=2, 4&2=0, 5&2=0
        expect(bitwiseOr(a, b).toList(), [3, 6, 7]); // 3|2=3, 4|2=6, 5|2=7

        // Strided sliced broadcasting test
        final stridedA = a.slice([Slice(start: 0, stop: 3, step: 2)]); // [3, 5]
        expect(bitwiseOr(stridedA, b).toList(), [3, 7]);
      });
    });

    test('mixed integer type rejection and explicit astype', () {
      NDArray.scope(() {
        final a32 = NDArray.fromList([5, 12], [2], DType.int32);
        final b64 = NDArray.fromList([3, 4], [2], DType.int64);

        expect(() => bitwiseAnd<DTypeTag>(a32, b64), throwsArgumentError);
        final res = bitwiseAnd(a32.astype(DType.int64), b64);
        expect(res.dtype, DType.int64);
        expect(res.toList(), [1, 4]);
      });
    });

    test('named recycler out parameter buffer reuse', () {
      NDArray.scope(() {
        final a = NDArray.fromList([5, 12], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);
        final out = NDArray.create([2], DType.int32);

        final res = bitwiseAnd(a, b, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [1, 4]);
      });
    });

    test('non-integer input throws ArgumentError', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3, 4], [2], DType.int32);

        expect(() => bitwiseAnd(a, b), throwsArgumentError);
        expect(() => invert(a), throwsArgumentError);
      });
    });

    test('recycler out buffer shape incompatibility throws ArgumentError', () {
      NDArray.scope(() {
        final a = NDArray.fromList([5, 12], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);
        final wrongOut = NDArray.create([3], DType.int32);

        expect(() => bitwiseAnd(a, b, out: wrongOut), throwsArgumentError);
      });
    });
  });

  group('Top-Level Comparison ufuncs with Recycling Tests', () {
    test(
      'equal() and notEqual() broadcasted comparison',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([1.0, 9.9, 3.0], [3], DType.float64);

        // equal
        final eq = equal(a, b);
        expect(eq.shape, [3]);
        expect(eq.dtype, DType.boolean);
        expect(eq.toList(), [true, false, true]);

        // notEqual
        final neq = notEqual(a, b);
        expect(neq.shape, [3]);
        expect(neq.dtype, DType.boolean);
        expect(neq.toList(), [false, true, false]);
      }),
    );

    test(
      'equal() with named recycler out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final b = NDArray.fromList([10.0, 99.0], [2], DType.float64);
        final out = NDArray<Boolean>.create([2], DType.boolean);

        final res = equal(a, b, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [true, false]);
      }),
    );

    test(
      'inequalities (greater, greaterEqual, less, lessEqual) contiguous & strided',
      () => NDArray.scope(() {
        final a = NDArray.fromList([3.0, 1.0, 5.0], [3], DType.float64);
        final b = NDArray.fromList([2.0, 1.0, 9.0], [3], DType.float64);

        // greater
        final gt = greater(a, b);
        expect(gt.toList(), [true, false, false]);

        // greaterEqual
        final gte = greaterEqual(a, b);
        expect(gte.toList(), [true, true, false]);

        // less
        final lt = less(a, b);
        expect(lt.toList(), [false, false, true]);

        // lessEqual
        final lte = lessEqual(a, b);
        expect(lte.toList(), [false, true, true]);
      }),
    );

    test(
      'complex numbers inequality throws UnsupportedError',
      () => NDArray.scope(() {
        final a = NDArray.create([2], DType.complex128);
        final b = NDArray.create([2], DType.complex128);

        expect(() => greater(a, b), throwsUnsupportedError);
        expect(() => greaterEqual(a, b), throwsUnsupportedError);
        expect(() => less(a, b), throwsUnsupportedError);
        expect(() => lessEqual(a, b), throwsUnsupportedError);
      }),
    );

    test(
      'recycler out buffer shape incompatibility throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final wrongShape = NDArray<Boolean>.create([3], DType.boolean);

        expect(() => equal(a, b, out: wrongShape), throwsArgumentError);
      }),
    );
  });

  group('NumPy Compatibility Universal Functions (ufuncs) tests', () {
    group('square ufunc', () {
      test('contiguous basic types', () {
        final a = NDArray.fromList(Float64List.fromList([2.0, -3.0, 4.0]), [
          3,
        ], DType.float64);
        final res = square(a);
        expect(res.toList(), [4.0, 9.0, 16.0]);
        expect(res.dtype, DType.float64);

        final aInt = NDArray.fromList(Int32List.fromList([2, -5, 10]), [
          3,
        ], DType.int32);
        final resInt = square(aInt);
        expect(resInt.toList(), [4, 25, 100]);
        expect(resInt.dtype, DType.int32);
      });

      test('complex types', () {
        final a = NDArray.create([2], DType.complex128);
        a[[0]] = Complex(3.0, 4.0); // (3+4i)^2 = 9 - 16 + 24i = -7 + 24i
        a[[1]] = Complex(0.0, -2.0); // (0-2i)^2 = -4

        final res = square(a);
        final resList = res.toList();
        expect(resList[0].real, closeTo(-7.0, 1e-10));
        expect(resList[0].imag, closeTo(24.0, 1e-10));
        expect(resList[1].real, closeTo(-4.0, 1e-10));
        expect(resList[1].imag, closeTo(0.0, 1e-10));
      });

      test('strided non-contiguous views', () {
        // Shape [3, 2]
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );

        // Slice column 0: [1.0, 3.0, 5.0] (strided)
        final col0 = parent.slice([Slice(start: 0, stop: 3), Index(0)]);
        expect(col0.isContiguous, false);

        final res = square(col0);
        expect(res.toList(), [1.0, 9.0, 25.0]);
      });

      test('boolean array and fallbacks', () {
        final aBool = NDArray.fromList([true, false], [2], DType.boolean);
        final resBool = square(aBool);
        expect(resBool.toList(), [true, false]);

        // Fallback types (uint8)
        final aUint = NDArray.fromList(Uint8List.fromList([3, 5]), [
          2,
        ], DType.uint8);
        final resUint = square(aUint);
        expect(resUint.toList(), [9, 25]);
      });

      test('out recycler buffer', () {
        final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final outRecycler = NDArray.zeros([2], DType.float64);
        final res = square(a, out: outRecycler);
        expect(identical(res, outRecycler), true);
        expect(outRecycler.toList(), [4.0, 9.0]);
      });
    });

    group('power float ufunc FFI speedups', () {
      test('basic float power', () {
        final x1 = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([3.0, 2.0]), [
          2,
        ], DType.float64);
        final res = power(x1, x2);
        expect(res.toList(), [8.0, 9.0]);
      });

      test('broadcasting strided power', () {
        final x1 = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([3.0]), [
          1,
        ], DType.float64);
        final res = power(x1, x2);
        expect(res.toList(), [8.0, 27.0]);
      });
    });

    group('floorDivide, remainder/mod, and divmod ufuncs', () {
      test(
        'positive and negative floor division combinations matching Python/NumPy',
        () {
          final x = NDArray.fromList(
            Float64List.fromList([5.0, -5.0, -5.0, 5.0]),
            [4],
            DType.float64,
          );
          final y = NDArray.fromList(
            Float64List.fromList([2.0, 2.0, -2.0, -2.0]),
            [4],
            DType.float64,
          );

          final q = floorDivide(x, y);
          // 5 // 2 = 2
          // -5 // 2 = -3
          // -5 // -2 = 2
          // 5 // -2 = -3
          expect(q.toList(), [2.0, -3.0, 2.0, -3.0]);
        },
      );

      test('remainder and mod matching Python/NumPy signed floor modulo', () {
        final x = NDArray.fromList(
          Float64List.fromList([5.0, -5.0, -5.0, 5.0]),
          [4],
          DType.float64,
        );
        final y = NDArray.fromList(
          Float64List.fromList([2.0, 2.0, -2.0, -2.0]),
          [4],
          DType.float64,
        );

        final r = remainder(x, y);
        final m = mod(x, y);
        // 5 % 2 = 1
        // -5 % 2 = 1 (since -5 = 2 * -3 + 1)
        // -5 % -2 = -1 (since -5 = -2 * 2 - 1)
        // 5 % -2 = -1 (since 5 = -2 * -3 - 1)
        expect(r.toList(), [1.0, 1.0, -1.0, -1.0]);
        expect(m.toList(), [1.0, 1.0, -1.0, -1.0]);
      });

      test('divmod combined tuple', () {
        final x = NDArray.fromList(Int32List.fromList([5, -5]), [
          2,
        ], DType.int32);
        final y = NDArray.fromList(Int32List.fromList([-2, -2]), [
          2,
        ], DType.int32);

        final res = divmod(x, y);
        final q = res.quotient;
        final r = res.remainder;

        expect(q.toList(), [-3, 2]);
        expect(r.toList(), [-1, -1]);
      });

      test('integer division by zero checks', () {
        final x = NDArray.fromList(Int32List.fromList([5]), [1], DType.int32);
        final y = NDArray.fromList(Int32List.fromList([0]), [1], DType.int32);

        expect(() => floorDivide(x, y), throwsA(isA<UnsupportedError>()));
        expect(() => remainder(x, y), throwsA(isA<UnsupportedError>()));
        expect(() => divmod(x, y), throwsA(isA<UnsupportedError>()));
      });
    });

    group('floating-point classifications (isnan, isinf, isfinite)', () {
      test('real types', () {
        final a = NDArray.fromList(
          Float64List.fromList([
            1.0,
            double.nan,
            double.infinity,
            double.negativeInfinity,
          ]),
          [4],
          DType.float64,
        );

        final nanMask = isnan(a);
        expect(nanMask.toList(), [false, true, false, false]);

        final infMask = isinf(a);
        expect(infMask.toList(), [false, false, true, true]);

        final finMask = isfinite(a);
        expect(finMask.toList(), [true, false, false, false]);
      });

      test('complex types', () {
        final a = NDArray.create([3], DType.complex128);
        a[[0]] = Complex(double.nan, 1.0);
        a[[1]] = Complex(1.0, double.infinity);
        a[[2]] = Complex(1.0, 2.0);

        final nanMask = isnan(a);
        expect(nanMask.toList(), [true, false, false]);

        final infMask = isinf(a);
        expect(infMask.toList(), [false, true, false]);

        final finMask = isfinite(a);
        expect(finMask.toList(), [false, false, true]);
      });

      test('integer fallbacks always clean', () {
        final a = NDArray.fromList(Int32List.fromList([1, 2]), [
          2,
        ], DType.int32);
        expect(isnan(a).toList().every((x) => !x), true);
        expect(isinf(a).toList().every((x) => !x), true);
        expect(isfinite(a).toList().every((x) => x), true);
      });
    });

    group('copysign ufunc', () {
      test('basic copysign', () {
        final x1 = NDArray.fromList(Float64List.fromList([1.0, -1.0, 2.0]), [
          3,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([-3.0, 5.0, -0.0]), [
          3,
        ], DType.float64);

        final res = copysign(x1, x2);
        final resList = res.toList();
        expect(resList[0], -1.0);
        expect(resList[1], 1.0);
        // sign of -0.0 should be negative!
        expect(resList[2].isNegative, true);
        expect(resList[2].abs(), 2.0);
      });

      test('broadcasting copysign', () {
        final x1 = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([-5.0]), [
          1,
        ], DType.float64);

        final res = copysign(x1, x2);
        expect(res.toList(), [-1.0, -2.0]);
      });
    });
  });

  group('Universal Functions (ufuncs) & Broadcasting Tests', () {
    group('Mixed DType Arithmetic Upcasting', () {
      test(
        'Add Complex and Float64 arrays',
        () => NDArray.scope(() {
          final a = NDArray.create([2], DType.complex128);
          a[[0]] = Complex(1.0, 2.0);
          a[[1]] = Complex(3.0, 4.0);

          final b = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
            2,
          ], DType.float64);

          expect(() => add<DTypeTag>(a, b), throwsArgumentError);
          final c = add(a, b.astype(DType.complex128));
          expect(c.dtype, DType.complex128);
          expect(c.shape, [2]);
          expect(c.toList(), [Complex(11.0, 2.0), Complex(23.0, 4.0)]);
        }),
      );

      test(
        'Subtract Float64 and Complex arrays',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
            2,
          ], DType.float64);

          final b = NDArray.create([2], DType.complex128);
          b[[0]] = Complex(1.0, 2.0);
          b[[1]] = Complex(3.0, 4.0);

          expect(() => subtract<DTypeTag>(a, b), throwsArgumentError);
          final c = subtract(a.astype(DType.complex128), b);
          expect(c.dtype, DType.complex128);
          expect(c.toList(), [Complex(9.0, -2.0), Complex(17.0, -4.0)]);
        }),
      );

      test(
        'Multiply Int32 and Complex arrays',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([2, 3]), [
            2,
          ], DType.int32);

          final b = NDArray.create([2], DType.complex128);
          b[[0]] = Complex(4.0, 5.0);
          b[[1]] = Complex(1.0, -2.0);

          expect(() => multiply<DTypeTag>(a, b), throwsArgumentError);
          final c = multiply(a.astype(DType.complex128), b);
          expect(c.dtype, DType.complex128);
          expect(c.toList(), [Complex(8.0, 10.0), Complex(3.0, -6.0)]);
        }),
      );

      test(
        'True Division always returns floats or complex',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([5, 10]), [
            2,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([2, 4]), [
            2,
          ], DType.int32);

          // int / int -> float64
          final c = divide(a, b);
          expect(c.dtype, DType.float64);
          expect(c.toList(), [2.5, 2.5]);
        }),
      );
    });

    group('New Math Ufuncs', () {
      test(
        'tan',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([0.0, math.pi / 4]), [
            2,
          ], DType.float64);
          final b = tan(a);
          expect(b.shape, [2]);
          final bList = b.toList();
          expect(bList[0], closeTo(0.0, 1e-10));
          expect(bList[1], closeTo(1.0, 1e-10));

          final c = NDArray.create([1], DType.complex128);
          c[[0]] = Complex(0.0, 0.0);
          final resC = tan(c);
          expect(resC.dtype, DType.complex128);
          expect(resC.toList(), [Complex(0.0, 0.0)]);
        }),
      );

      test(
        'atan2 with broadcasting',
        () => NDArray.scope(() {
          final y = NDArray.fromList(Float64List.fromList([1.0]), [
            1,
          ], DType.float64);
          final x = NDArray.fromList(
            Float64List.fromList([1.0, math.sqrt(3)]),
            [2],
            DType.float64,
          );

          final b = atan2(y, x); // y broadcasts to [2]
          expect(b.shape, [2]);
          final bList = b.toList();
          expect(bList[0], closeTo(math.pi / 4, 1e-10)); // atan2(1, 1) = pi/4
          expect(
            bList[1],
            closeTo(math.pi / 6, 1e-10),
          ); // atan2(1, sqrt(3)) = pi/6
        }),
      );

      test(
        'Hyperbolic trig (sinh, cosh, tanh)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([0.0]), [
            1,
          ], DType.float64);
          final resSinh = sinh(a);
          expect(resSinh.toList(), [0.0]);

          final resCosh = cosh(a);
          expect(resCosh.toList(), [1.0]);

          final resTanh = tanh(a);
          expect(resTanh.toList(), [0.0]);
        }),
      );

      test(
        'abs (Complex magnitude)',
        () => NDArray.scope(() {
          final a = NDArray.create([2], DType.complex128);
          a[[0]] = Complex(3.0, 4.0); // mag = 5.0
          a[[1]] = Complex(-5.0, 12.0); // mag = 13.0

          final b = abs(a);
          expect(b.dtype, DType.float64); // should return double real array
          expect(b.shape, [2]);
          expect(b.toList(), [5.0, 13.0]);
        }),
      );

      test(
        'Rounding (ceil, floor, round)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-1.5, 1.2, 2.7]), [
            3,
          ], DType.float64);
          final resCeil = ceil(a);
          expect(resCeil.toList(), [-1.0, 2.0, 3.0]);

          final resFloor = floor(a);
          expect(resFloor.toList(), [-2.0, 1.0, 2.0]);

          final resRound = round(a);
          expect(resRound.toList(), [
            -2.0,
            1.0,
            3.0,
          ]); // Dart round() for -1.5 is -2
        }),
      );

      test(
        'clip with scalar bounds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-5.0, 1.5, 10.0]), [
            3,
          ], DType.float64);
          final b = clip(a, min: -1.0, max: 5.0);
          expect(b.toList(), [-1.0, 1.5, 5.0]);

          // Verify Complex clip throws UnsupportedError
          final c = NDArray.create([2], DType.complex128);
          expect(() => clip(c, min: -1.0, max: 5.0), throwsUnsupportedError);
        }),
      );

      test(
        'clip with array bounds (exact matching shape)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 5.0, 10.0]), [
            3,
          ], DType.float64);
          final minBounds = NDArray.fromList(
            Float64List.fromList([2.0, 2.0, 2.0]),
            [3],
            DType.float64,
          );
          final maxBounds = NDArray.fromList(
            Float64List.fromList([8.0, 8.0, 8.0]),
            [3],
            DType.float64,
          );

          final b = clipArray(a, min: minBounds, max: maxBounds);
          expect(b.toList(), [2.0, 5.0, 8.0]);
        }),
      );

      test(
        'clip with broadcasted array bounds',
        () => NDArray.scope(() {
          // Input is 2x3 matrix
          final a = NDArray.fromList(
            Float64List.fromList([
              1.0, 2.0, 3.0, // row 0
              4.0, 5.0, 6.0, // row 1
            ]),
            [2, 3],
            DType.float64,
          );

          // min bounds is a 1D row vector [2.5, 2.5, 2.5] (shape [3]) -> broadcasts to [2, 3]
          final minBounds = NDArray.fromList(
            Float64List.fromList([2.5, 2.5, 2.5]),
            [3],
            DType.float64,
          );

          // max bounds is a 2D column vector [[4.5], [4.5]] (shape [2, 1]) -> broadcasts to [2, 3]
          final maxBounds = NDArray.fromList(Float64List.fromList([4.5, 4.5]), [
            2,
            1,
          ], DType.float64);

          // Clip bounds for row 0: min=2.5, max=4.5
          // row 0 values: [1.0, 2.0, 3.0] -> clipped: [2.5, 2.5, 3.0]
          // Clip bounds for row 1: min=2.5, max=4.5
          // row 1 values: [4.0, 5.0, 6.0] -> clipped: [4.0, 4.5, 4.5]
          final b = clipArray(a, min: minBounds, max: maxBounds);
          expect(b.shape, [2, 3]);
          expect(b.toList(), [2.5, 2.5, 3.0, 4.0, 4.5, 4.5]);
        }),
      );

      test(
        'clip with mixed scalar and array bounds on integer DType',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 5, 10]), [
            3,
          ], DType.int32);

          // min is scalar, max is array
          final maxBounds = NDArray.fromList(Int32List.fromList([3, 7, 12]), [
            3,
          ], DType.int32);
          final minBounds = NDArray.fromList(Int32List.fromList([2]), [
            1,
          ], DType.int32);

          final b = clipArray(a, min: minBounds, max: maxBounds);
          expect(b.dtype, DType.int32);
          expect(b.toList(), [2, 5, 10]);
        }),
      );

      test(
        'clip throws ArgumentError on incompatible shape broadcasting',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final incompatible = NDArray.fromList(
            Float64List.fromList([1.0, 2.0]),
            [2],
            DType.float64,
          );
          final maxBounds = NDArray.fromList(Float64List.fromList([5.0]), [
            1,
          ], DType.float64);

          expect(
            () => clipArray(a, min: incompatible, max: maxBounds),
            throwsArgumentError,
          );
        }),
      );
      test(
        'clip with one-sided (nullable) scalar bounds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-5.0, 1.5, 10.0]), [
            3,
          ], DType.float64);

          // Clip with min only
          final b1 = clip(a, min: -1.0);
          expect(b1.toList(), [-1.0, 1.5, 10.0]);

          // Clip with max only
          final b2 = clip(a, max: 5.0);
          expect(b2.toList(), [-5.0, 1.5, 5.0]);

          // Clip with neither (should return exact copy)
          final b3 = clip(a);
          expect(b3.toList(), [-5.0, 1.5, 10.0]);
        }),
      );

      test(
        'clipArray with one-sided (nullable) array bounds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 5.0, 10.0]), [
            3,
          ], DType.float64);
          final minBounds = NDArray.fromList(
            Float64List.fromList([2.0, 2.0, 2.0]),
            [3],
            DType.float64,
          );
          final maxBounds = NDArray.fromList(
            Float64List.fromList([8.0, 8.0, 8.0]),
            [3],
            DType.float64,
          );

          // Clip with min only
          final b1 = clipArray(a, min: minBounds);
          expect(b1.toList(), [2.0, 5.0, 10.0]);

          // Clip with max only
          final b2 = clipArray(a, max: maxBounds);
          expect(b2.toList(), [1.0, 5.0, 8.0]);

          // Clip with neither (should return exact copy)
          final b3 = clipArray(a);
          expect(b3.toList(), [1.0, 5.0, 10.0]);
        }),
      );
    });

    group('Comparison Operator Broadcasting', () {
      test(
        'Array and Scalar comparison',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final mask = a > 2.0;
          expect(mask.dtype, DType.boolean);
          expect(mask.shape, [2, 2]);
          expect(mask.toList(), [false, false, true, true]);
        }),
      );

      test(
        'Compatible shapes array comparison broadcasting',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 10.0, 4.0, 2.0]),
            [2, 2],
            DType.float64,
          );

          final vec = NDArray.fromList(Float64List.fromList([3.0]), [
            1,
          ], DType.float64); // shape [1] broadcasts to [2, 2]

          final mask = mat < vec;
          expect(mask.shape, [2, 2]);
          expect(mask.toList(), [
            true,
            false,
            false,
            true,
          ]); // [1<3(true), 10<3(false), 4<3(false), 2<3(true)]
        }),
      );

      test(
        'Complex equality and inequality exceptions',
        () => NDArray.scope(() {
          final c1 = NDArray.create([1], DType.complex128);
          c1[[0]] = Complex(1, 2);

          final c2 = NDArray.create([1], DType.complex128);
          c2[[0]] = Complex(1, 2);

          // Equality is supported for complex!
          final eqRes = c1.eq(c2);
          expect(eqRes.toList(), [true]);

          // Inequalities throw UnsupportedError
          expect(() => c1 > c2, throwsUnsupportedError);
          expect(() => c1 <= 2.0, throwsUnsupportedError);
        }),
      );
    });

    group('Logical Operations', () {
      test(
        'logicalNot on ints and doubles',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([0.0, 2.5, -1.1]), [
            3,
          ], DType.float64);
          final resNot = logicalNot(a);
          expect(resNot.dtype, DType.boolean);
          expect(
            resNot.toList(),
            [true, false, false],
          ); // 0.0 is false (so not is true), others are true (so not is false)
        }),
      );

      test(
        'logicalAnd, logicalOr, logicalXor with broadcasting',
        () => NDArray.scope(() {
          // shape [2, 1]
          final m1 = NDArray.fromList(Int32List.fromList([0, 1]), [
            2,
            1,
          ], DType.int32);
          // shape [1, 2]
          final m2 = NDArray.fromList(Int32List.fromList([0, 1]), [
            1,
            2,
          ], DType.int32);

          // Broadcasted shape [2, 2]
          // m1 expanded: [[0, 0], [1, 1]]
          // m2 expanded: [[0, 1], [0, 1]]

          final andRes = logicalAnd(m1, m2);
          expect(andRes.dtype, DType.boolean);
          expect(andRes.shape, [2, 2]);
          expect(andRes.toList(), [
            false, false, // false&&false, false&&true
            false, true, // true&&false, true&&true
          ]);

          final orRes = logicalOr(m1, m2);
          expect(orRes.dtype, DType.boolean);
          expect(orRes.toList(), [
            false, true, // false||false, false||true
            true, true, // true||false, true||true
          ]);

          final xorRes = logicalXor(m1, m2);
          expect(xorRes.dtype, DType.boolean);
          expect(xorRes.toList(), [
            false, true, // false^false, false^true
            true, false, // true^false, true^true
          ]);
        }),
      );
    });
  });

  group('NDArray Logical Reductions (all, any) Tests', () {
    test(
      'all() basic global reduction checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, true, true], [3], DType.boolean);
        final b = NDArray.fromList([true, false, true], [3], DType.boolean);
        final c = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final d = NDArray.fromList([1, 0, 3], [3], DType.int32);

        expect(all(a).scalar, true);
        expect(all(b).scalar, false);
        expect(all(c).scalar, true);
        expect(all(d).scalar, false);
      }),
    );

    test(
      'any() basic global reduction checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList([false, false, false], [3], DType.boolean);
        final b = NDArray.fromList([false, true, false], [3], DType.boolean);
        final c = NDArray.fromList([0, 0, 0], [3], DType.int32);
        final d = NDArray.fromList([0, 5, 0], [3], DType.int32);

        expect(any(a).scalar, false);
        expect(any(b).scalar, true);
        expect(any(c).scalar, false);
        expect(any(d).scalar, true);
      }),
    );

    test(
      'all() axis reductions on 2D matrices',
      () => NDArray.scope(() {
        final mat = NDArray.fromList(
          [true, true, false, true, false, false],
          [2, 3],
          DType.boolean,
        );

        // all along rows (axis: 0) -> shape [3]
        final res0 = all(mat, axis: 0);
        expect(res0.shape, [3]);
        expect(res0.toList(), [true, false, false]);

        // all along columns (axis: 1) -> shape [2]
        final res1 = all(mat, axis: 1);
        expect(res1.shape, [2]);
        expect(res1.toList(), [false, false]);

        // negative axis support
        final resNeg = all(mat, axis: -1);
        expect(resNeg.shape, [2]);
        expect(resNeg.toList(), [false, false]);
      }),
    );

    test(
      'any() axis reductions on 2D matrices',
      () => NDArray.scope(() {
        final mat = NDArray.fromList(
          [true, false, false, false, false, false],
          [2, 3],
          DType.boolean,
        );

        // any along rows (axis: 0) -> shape [3]
        final res0 = any(mat, axis: 0);
        expect(res0.shape, [3]);
        expect(res0.toList(), [true, false, false]);

        // any along columns (axis: 1) -> shape [2]
        final res1 = any(mat, axis: 1);
        expect(res1.shape, [2]);
        expect(res1.toList(), [true, false]);
      }),
    );

    test(
      'logicalAnd, logicalOr, and logicalXor successfully combine boolean mask arrays',
      () => NDArray.scope(() {
        final mask1 = NDArray.fromList([true, false, true], [3], DType.boolean);
        final mask2 = NDArray.fromList([true, true, false], [3], DType.boolean);

        final resAnd = logicalAnd(mask1, mask2);
        expect(resAnd.dtype, DType.boolean);
        expect(resAnd.toList(), [true, false, false]);

        final resOr = logicalOr(mask1, mask2);
        expect(resOr.dtype, DType.boolean);
        expect(resOr.toList(), [true, true, true]);

        final resXor = logicalXor(mask1, mask2);
        expect(resXor.dtype, DType.boolean);
        expect(resXor.toList(), [false, true, true]);
      }),
    );

    test(
      'logicalAnd cross-type rejection and explicit astype on float/boolean, complex/boolean, and int/boolean',
      () => NDArray.scope(() {
        final mask = NDArray.fromList([true, true, false], [3], DType.boolean);

        // Float64 / boolean
        final f64 = NDArray.fromList([0.0, 2.5, 0.0], [3], DType.float64);
        expect(() => logicalAnd(f64, mask), throwsArgumentError);
        expect(() => logicalAnd(mask, f64), throwsArgumentError);
        final resF1 = logicalAnd(f64.astype(DType.boolean), mask);
        expect(resF1.dtype, DType.boolean);
        expect(resF1.toList(), [false, true, false]);
        final resF2 = logicalAnd(mask, f64.astype(DType.boolean));
        expect(resF2.dtype, DType.boolean);
        expect(resF2.toList(), [false, true, false]);

        // Complex128 / boolean
        final c128 = NDArray.create([3], DType.complex128);
        c128[[0]] = Complex(0.0, 0.0); // false
        c128[[1]] = Complex(1.0, -1.0); // true
        c128[[2]] = Complex(0.0, 0.0);
        expect(() => logicalAnd(c128, mask), throwsArgumentError);
        expect(() => logicalAnd(mask, c128), throwsArgumentError);
        final resC1 = logicalAnd(c128.astype(DType.boolean), mask);
        expect(resC1.dtype, DType.boolean);
        expect(resC1.toList(), [false, true, false]);
        final resC2 = logicalAnd(mask, c128.astype(DType.boolean));
        expect(resC2.dtype, DType.boolean);
        expect(resC2.toList(), [false, true, false]);

        // Int32 / boolean
        final i32 = NDArray.fromList([0, 5, 0], [3], DType.int32);
        expect(() => logicalAnd(i32, mask), throwsArgumentError);
        expect(() => logicalAnd(mask, i32), throwsArgumentError);
        final resI1 = logicalAnd(i32.astype(DType.boolean), mask);
        expect(resI1.dtype, DType.boolean);
        expect(resI1.toList(), [false, true, false]);
        final resI2 = logicalAnd(mask, i32.astype(DType.boolean));
        expect(resI2.dtype, DType.boolean);
        expect(resI2.toList(), [false, true, false]);
      }),
    );

    test(
      'Disposed array checks throw StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, false], [2], DType.boolean);
        a.dispose();

        expect(() => all(a), throwsStateError);
        expect(() => any(a), throwsStateError);
      }),
    );

    test(
      'Axis out of bounds throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, false], [2], DType.boolean);
        expect(() => all(a, axis: 5), throwsArgumentError);
        expect(() => any(a, axis: -5), throwsArgumentError);
      }),
    );
    test(
      '_resolveDType() complex64 and float64 cross-promotion coverage',
      () => NDArray.scope(() {
        final a = NDArray.fromList([Complex(1.0, 1.0)], [1], DType.complex64);
        final b = NDArray.fromList([2.0], [1], DType.float64);

        expect(() => add<DTypeTag>(a, b), throwsArgumentError);
        expect(() => addAs(a, b, DType.complex128), throwsArgumentError);
        final c = add(a.astype(DType.complex128), b.astype(DType.complex128));
        expect(c.dtype, DType.complex128);
        expect(c.getCell([0]).real, 3.0);
        expect(c.getCell([0]).imag, 1.0);
      }),
    );

    test(
      'ufuncs in-place out buffer shape and dtype validation checks',
      () => NDArray.scope(() {
        final a = NDArray.ones([3], DType.float64);
        final b = NDArray.ones([3], DType.float64);
        final incompatibleOut = NDArray.ones([4], DType.float64);
        final incompatibleDTypeOut = NDArray.ones([3], DType.int32);

        // 1. add() contiguous shape mismatch
        expect(() => add(a, b, out: incompatibleOut), throwsArgumentError);
        expect(() => add(a, b, out: incompatibleDTypeOut), throwsArgumentError);

        // 2. add() broadcast shape mismatch
        final broadcastA = NDArray.ones([1, 3], DType.float64);
        expect(
          () => add(broadcastA, b, out: incompatibleOut),
          throwsArgumentError,
        );

        // 3. sqrt() shape mismatch
        expect(() => sqrt(a, out: incompatibleOut), throwsArgumentError);

        // 4. sin() shape mismatch
        expect(() => sin(a, out: incompatibleOut), throwsArgumentError);
      }),
    );

    test(
      '_resolveDType cross-promotion additions coverage',
      () => NDArray.scope(() {
        final f64 = NDArray.fromList([1.0], [1], DType.float64);
        final f32 = NDArray.fromList([2.0], [1], DType.float32);
        final i64 = NDArray.fromList([3], [1], DType.int64);
        final i32 = NDArray.fromList([4], [1], DType.int32);

        // Mixed dtypes are rejected on add() and addAs(); explicit astype works
        expect(() => add<DTypeTag>(f64, f32), throwsArgumentError);
        expect(() => addAs(f64, f32, DType.float64), throwsArgumentError);
        final r1 = add(f64, f32.astype(DType.float64));
        expect(r1.dtype, DType.float64);

        expect(() => add<DTypeTag>(f32, i64), throwsArgumentError);
        final r2 = add(f32.astype(DType.float64), i64.astype(DType.float64));
        expect(r2.dtype, DType.float64);

        expect(() => add<DTypeTag>(f32, i32), throwsArgumentError);
        final r2b = add(f32.astype(DType.float64), i32.astype(DType.float64));
        expect(r2b.dtype, DType.float64);

        expect(() => add<DTypeTag>(i64, i32), throwsArgumentError);
        final r3 = add(i64, i32.astype(DType.int64));
        expect(r3.dtype, DType.int64);

        expect(() => add<DTypeTag>(i32, f64), throwsArgumentError);
        final r4 = add(i32.astype(DType.float64), f64);
        expect(r4.dtype, DType.float64);
        expect(r4.toList(), [5.0]);
      }),
    );

    test(
      'prod() contiguous FFI leaf paths coverage',
      () => NDArray.scope(() {
        final f64 = NDArray.fromList([2.0, 3.0, 4.0], [3], DType.float64);
        final f32 = NDArray.fromList([5.0, 2.0, 3.0], [3], DType.float32);

        // 1. float64 contiguous FFI prod()
        final r1 = prod(f64);
        expect(r1.scalar, closeTo(24.0, 1e-9));

        // 2. float32 contiguous FFI prod()
        final r2 = prod(f32);
        expect(r2.scalar, closeTo(30.0, 1e-9));
      }),
    );

    test(
      'prod() and sum() on non-contiguous strided views along axes',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final viewT = parent.transposed; // non-contiguous view of shape [2, 3]
        expect(viewT.isContiguous, false);

        // Global reduction
        expect(prod(viewT).scalar, 720.0); // 1*2*3*4*5*6 = 720
        expect(sum(viewT).scalar, 21.0);

        // Reduction along axes (triggers _reduceRecursive fallback paths)
        final p0 = prod(viewT, axis: 0); // Product along axis 0 -> shape [3]
        expect(p0.shape, [3]);
        expect(p0.toList(), [
          2.0,
          12.0,
          30.0,
        ]); // col 0: 1*2=2, col 1: 3*4=12, col 2: 5*6=30

        final p1 = prod(viewT, axis: 1); // Product along axis 1 -> shape [2]
        expect(p1.shape, [2]);
        expect(p1.toList(), [15.0, 48.0]); // row 0: 1*3*5=15, row 1: 2*4*6=48
      }),
    );

    test(
      'clip() with named out parameter recycler and sliced contiguous view',
      () {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );

        final view = parent.slice([Slice(start: 0, stop: 2), Slice.all()]);
        final out = NDArray.zeros([2, 2], DType.float64);

        final res = clip(view, min: 2.0, max: 3.0, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [2.0, 2.0, 3.0, 3.0]);

        expect(parent.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      },
    );

    test(
      'where() with out parameter recycler and shape validations',
      () => NDArray.scope(() {
        final cond = NDArray.fromList(
          [true, false, false, true],
          [2, 2],
          DType.boolean,
        );
        final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final y = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final out = NDArray.zeros([2, 2], DType.float64);
        final incompatibleOut = NDArray.zeros([3], DType.float64);

        // 1. Incompatible shape throws ArgumentError
        expect(() => where(cond, x, y, incompatibleOut), throwsArgumentError);

        // 2. Valid in-place recycling
        final res = where(cond, x, y, out);
        expect(identical(res, out), true);
        expect(out.toList(), [1.0, 20.0, 30.0, 4.0]);
      }),
    );

    test(
      'NDArray.fill() ufunc correctness and performance speedups verification',
      () {
        // 1. Contiguous Double Precision fill
        final a = NDArray.zeros([5], DType.float64);
        a.fill(42.5);
        expect(a.toList(), [42.5, 42.5, 42.5, 42.5, 42.5]);

        // 2. Contiguous Int32 Precision fill
        final b = NDArray.zeros([5], DType.int32);
        b.fill(99);
        expect(b.toList(), [99, 99, 99, 99, 99]);

        // 3. Strided view fallback JIT fill
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final view = parent.slice([
          Slice(start: 0, stop: 4, step: 2),
        ]); // indices: 0, 2

        expect(view.shape, [2]);
        expect(view.isContiguous, false);

        view.fill(77.0);
        expect(parent.toList(), [77.0, 2.0, 77.0, 4.0]);
      },
    );

    test(
      'isClose() and allClose() approximate equality ufunc correctness',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 1.00001, 2.0], [3], DType.float64);
        final b = NDArray.fromList([1.0, 1.00003, 2.0], [3], DType.float64);

        // 1. Default tolerances: rtol = 1e-05, atol = 1e-08
        final closeDefault = isClose(a, b);
        expect(closeDefault.toList(), [true, false, true]);

        // 2. Stretched tolerances: rtol = 1e-04
        final closeStretched = isClose(a, b, rtol: 1e-04);
        expect(closeStretched.toList(), [true, true, true]);

        // 3. allClose logic
        expect(allClose(a, b), false);
        expect(allClose(a, b, rtol: 1e-04), true);

        // 4. Infinite matching values
        final infA = NDArray.fromList(
          [double.infinity, double.negativeInfinity],
          [2],
          DType.float64,
        );
        final infB = NDArray.fromList(
          [double.infinity, double.negativeInfinity],
          [2],
          DType.float64,
        );
        final infC = NDArray.fromList(
          [double.negativeInfinity, double.infinity],
          [2],
          DType.float64,
        );

        final closeInf = isClose(infA, infB);
        final closeInfMismatch = isClose(infA, infC);

        expect(closeInf.toList(), [true, true]);
        expect(closeInfMismatch.toList(), [false, false]);

        // 5. NaN value equalNan checks
        final nanA = NDArray.fromList([double.nan], [1], DType.float64);
        final nanB = NDArray.fromList([double.nan], [1], DType.float64);

        final closeNanDefault = isClose(nanA, nanB);
        final closeNanEqual = isClose(nanA, nanB, equalNan: true);

        expect(closeNanDefault.toList(), [false]);
        expect(closeNanEqual.toList(), [true]);

        // 6. 2D and 3D arrays without .toList() nested list error
        final a2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b2d = NDArray.fromList(
          [1.0, 2.00001, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final close2d = isClose(a2d, b2d);
        expect(close2d.shape, [2, 2]);
        expect(allClose(a2d, b2d), true);

        final a3d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float64,
        );
        final b3d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float64,
        );
        expect(allClose(a3d, b3d), true);
      }),
    );

    test(
      'nan_to_num() dataset cleaning ufunc correctness',
      () => NDArray.scope(() {
        // 1. Default Float64 cleaning
        final a = NDArray.fromList(
          [1.0, double.nan, double.infinity, double.negativeInfinity],
          [4],
          DType.float64,
        );

        final cleanDefault = nan_to_num(a);

        expect(cleanDefault.toList()[0], 1.0);
        expect(cleanDefault.toList()[1], 0.0);
        expect(cleanDefault.toList()[2], double.maxFinite);
        expect(cleanDefault.toList()[3], -double.maxFinite);

        // 2. Custom parameters cleaning
        final cleanCustom = nan_to_num(
          a,
          nan: 99.0,
          posinf: 500.0,
          neginf: -500.0,
        );

        expect(cleanCustom.toList(), [1.0, 99.0, 500.0, -500.0]);

        // 3. View-safe in-place recycling
        final parent = NDArray.fromList(
          [double.nan, 2.0, double.nan, 4.0],
          [4],
          DType.float64,
        );
        final view = parent.slice([
          Slice(start: 0, stop: 4, step: 2),
        ]); // indices: 0, 2

        expect(view.isContiguous, false);
        nan_to_num(view, nan: 100.0, out: view);

        expect(parent.toList(), [100.0, 2.0, 100.0, 4.0]);

        // 4. 2D and 3D arrays cleaning
        final a2d = NDArray.fromList(
          [double.nan, 2.0, 3.0, double.infinity],
          [2, 2],
          DType.float64,
        );
        final res2d = nan_to_num(a2d, nan: 10.0, posinf: 20.0);
        expect(res2d.shape, [2, 2]);
        expect(res2d.data, [10.0, 2.0, 3.0, 20.0]);

        final a3d = NDArray.fromList(
          [double.nan, 1.0, 2.0, double.negativeInfinity, 3.0, 4.0, 5.0, 6.0],
          [2, 2, 2],
          DType.float64,
        );
        final res3d = nan_to_num(a3d, nan: -1.0, neginf: -99.0);
        expect(res3d.shape, [2, 2, 2]);
        expect(res3d.data[0], -1.0);
        expect(res3d.data[3], -99.0);
      }),
    );

    test(
      'Contiguous sub-slice view sum() and prod() FFI reductions correctness',
      () {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final view = parent.slice([
          Slice(start: 0, stop: 2),
        ]); // elements: 1.0, 2.0

        expect(view.isContiguous, true);
        expect(view.shape, [2]);

        // 1. sum() verification
        final s = sum(view);
        expect(s.scalar, 3.0);

        // 2. prod() verification
        final p = prod(view);
        expect(p.scalar, 2.0);
      },
    );

    test(
      'Non-contiguous/strided integer ufuncs fallback walks (tan, abs, ceil, floor, round)',
      () {
        final i = NDArray.fromList([-1, -2, -3, -4], [2, 2], DType.int64);
        final iT = i.transposed;

        final rAbs = abs(iT);
        expect(rAbs.toList(), [1, 3, 2, 4]);

        final rTan = tan(iT);
        expect(rTan.getCell([0, 0]), closeTo(math.tan(-1.0), 1e-9));

        final rCeil = ceil(iT);
        expect(rCeil.toList(), [-1, -3, -2, -4]);

        final rFloor = floor(iT);
        expect(rFloor.toList(), [-1, -3, -2, -4]);

        final rRound = round(iT);
        expect(rRound.toList(), [-1, -3, -2, -4]);
      },
    );

    test(
      'Contiguous and non-contiguous clip() precision ufuncs coverage',
      () => NDArray.scope(() {
        // 1. Contiguous Float32 clip
        final f32 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
        final resF32 = clip(f32, min: 2.0, max: 3.0);
        expect(resF32.dtype, DType.float32);
        expect(resF32.toList(), [2.0, 2.0, 3.0, 3.0]);

        // 2. Non-contiguous integer clip
        final i32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32T = i32.transposed;
        final resI32 = clip(i32T, min: 2, max: 3);
        expect(resI32.toList(), [
          2,
          3,
          2,
          3,
        ]); // transposed: 1, 3, 2, 4 -> clipped: 2, 3, 2, 3

        // 3. Non-contiguous double clip
        final f64 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final f64T = f64.transposed;
        final resF64 = clip(f64T, min: 2.0, max: 3.0);
        expect(resF64.toList(), [2.0, 3.0, 2.0, 3.0]);
      }),
    );

    test(
      'Advanced Vector Selection select() correctness and broadcasting',
      () => NDArray.scope(() {
        final cond1 = NDArray.fromList(
          [true, false, false],
          [3],
          DType.boolean,
        );
        final cond2 = NDArray.fromList(
          [false, true, false],
          [3],
          DType.boolean,
        );

        final choice1 = NDArray.fromList([10, 20, 30], [3], DType.int32);
        final choice2 = NDArray.fromList([100, 200, 300], [3], DType.int32);

        // 1. Standard select
        final res = select(
          [cond1, cond2],
          [choice1, choice2],
          defaultValue: 999,
        );

        expect(res.shape, [3]);
        expect(res.dtype, DType.int32);
        expect(
          res.toList(),
          [10, 200, 999],
        ); // cond1 true at 0 -> 10; cond2 true at 1 -> 200; default at 2 -> 999

        // 2. Broadcasting select (scalar condition, vector choices)
        final condScalar = NDArray.fromList([true], [1], DType.boolean);
        final resB = select([condScalar], [choice1], defaultValue: 999);

        expect(resB.shape, [3]); // broadcasted
        expect(resB.toList(), [10, 20, 30]);

        // 3. Verify ArgumentError throwing exceptions
        expect(() => select([], [choice1]), throwsArgumentError); // empty list
        expect(() => select([cond1], []), throwsArgumentError);
        expect(
          () => select([cond1, cond2], [choice1]),
          throwsArgumentError,
        ); // length mismatch

        final badShapeChoice = NDArray.fromList([1, 2], [2], DType.int32);
        expect(
          () => select([cond1], [badShapeChoice]),
          throwsArgumentError,
        ); // shape mismatch
      }),
    );

    test(
      'Type-preserving reductions min(), max(), nanmin(), nanmax() DType parity',
      () {
        // 1. Integer min() / max() DType preservation
        final aInt32 = NDArray.fromList(
          [10, 2, 30, 4, 50, 6],
          [3, 2],
          DType.int32,
        );

        final minI32 = min(aInt32, axis: 0);
        expect(minI32.shape, [2]);
        expect(minI32.dtype, DType.int32); // Preserves Int32!
        expect(minI32.toList(), [10, 2]);

        final maxI32 = max(aInt32, axis: 0);
        expect(maxI32.shape, [2]);
        expect(maxI32.dtype, DType.int32); // Preserves Int32!
        expect(maxI32.toList(), [50, 6]);

        // 2. Float32 min() / max() DType preservation
        final aFloat32 = NDArray.fromList(
          [10.0, 2.0, 30.0, 4.0, 50.0, 6.0],
          [3, 2],
          DType.float32,
        );

        final minF32 = min(aFloat32, axis: 0);
        expect(minF32.dtype, DType.float32); // Preserves Float32!

        // 3. nanmin() / nanmax() DType preservation
        final nanF64 = NDArray.fromList(
          [1.0, double.nan, 3.0, 4.0, double.nan, 6.0],
          [3, 2],
          DType.float64,
        );

        final nanMinF64 = nanmin(nanF64, axis: 0);
        expect(nanMinF64.shape, [2]);
        expect(nanMinF64.dtype, DType.float64);
        expect(nanMinF64.getCell([0]), 1.0);

        final nanMaxF64 = nanmax(nanF64, axis: 0);
        expect(nanMaxF64.shape, [2]);
        expect(nanMaxF64.dtype, DType.float64);
        expect(nanMaxF64.getCell([0]), 3.0);
        expect(nanMaxF64.getCell([1]), 6.0);
      },
    );

    test(
      'NDArray cross-type comparison operators coverage',
      () => NDArray.scope(() {
        final comp = NDArray.create([2], DType.complex128);
        comp.setCell([0], Complex(1.0, 0.0));
        comp.setCell([1], Complex(3.0, 0.0));

        final dbl = NDArray.fromList([2.0, 2.0], [2], DType.float64);
        final integer = NDArray.fromList([2, 2], [2], DType.int32);

        // Mixed-dtype array comparisons throw ArgumentError; explicit astype works
        expect(() => comp.eq(dbl), throwsArgumentError);
        final cDbl = comp.eq(dbl.astype(DType.complex128));
        expect(cDbl.toList(), [false, false]); // 1 != 2, 3 != 2

        expect(() => comp.eq(integer), throwsArgumentError);
        final cInt = comp.eq(integer.astype(DType.complex128));
        expect(cInt.toList(), [false, false]);

        expect(() => dbl.eq(comp), throwsArgumentError);
        final dblC = dbl.astype(DType.complex128).eq(comp);
        expect(dblC.toList(), [false, false]);

        expect(() => dbl.eq(integer), throwsArgumentError);
        final dblInt = dbl.eq(integer.astype(DType.float64));
        expect(dblInt.toList(), [true, true]);

        expect(() => integer.eq(comp), throwsArgumentError);
        final intC = integer.astype(DType.complex128).eq(comp);
        expect(intC.toList(), [false, false]);

        expect(() => integer.eq(dbl), throwsArgumentError);
        final intDbl = integer.astype(DType.float64).eq(dbl);
        expect(intDbl.toList(), [true, true]);
      }),
    );

    test(
      'Complex array reductions (sum, prod, mean) and stacking coverage',
      () {
        final a = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 0.0),
            Complex(3.0, 0.0),
            Complex(4.0, 0.0),
          ],
          [2, 2],
          DType.complex128,
        );

        // Test sum()
        final totalSum = sum(a);
        expect(totalSum.scalar, Complex(10.0, 1.0));

        // Test mean()
        final totalMean = mean(a);
        expect(totalMean.scalar, Complex(2.5, 0.25));

        // Test prod()
        final totalProd = prod(a);
        expect(totalProd.scalar, Complex(24.0, 24.0)); // (1+i)*2*3*4 = 24 + 24i

        // Test concatenate() and hstack()
        final b = NDArray.fromList(
          [Complex(10.0, 0.0), Complex(10.0, 0.0)],
          [1, 2],
          DType.complex128,
        );

        final row0 = a.slice([Index(0)]).reshape([1, 2]);
        final combined = concatenate([row0, b], axis: 0);
        expect(combined.shape, [2, 2]);
        expect(combined.dtype, DType.complex128);
        expect(combined.getCell([0, 0]), Complex(1.0, 1.0));
        expect(combined.getCell([1, 0]), Complex(10.0, 0.0));
      },
    );
  });

  group('Min/Max/NanMin/NanMax Branch Coverage', () {
    test(
      'nanmin/nanmax flat reduction multiple dtypes',
      () => NDArray.scope(() {
        final f32 = NDArray.fromList(
          [1.0, double.nan, 3.0, 2.0],
          [4],
          DType.float32,
        );
        final f64 = NDArray.fromList(
          [1.0, double.nan, 3.0, 2.0],
          [4],
          DType.float64,
        );
        final i32 = NDArray.fromList([1, 3, 2], [3], DType.int32);

        expect(nanmin(f32).scalar, 1.0);
        expect(nanmax(f32).scalar, 3.0);

        expect(nanmin(f64).scalar, 1.0);
        expect(nanmax(f64).scalar, 3.0);

        expect(nanmin(i32).scalar, 1);
        expect(nanmax(i32).scalar, 3);
      }),
    );

    test(
      'out parameter reuse',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 3.0, 2.0], [3], DType.float64);

        final outFlat = NDArray.zeros([], DType.float64);
        final resFlat = min(a, out: outFlat);
        expect(identical(resFlat, outFlat), true);
        expect(outFlat.scalar, 1.0);

        final outAxis = NDArray.zeros([2], DType.float64);
        final mat = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final resAxis = min(mat, axis: 1, out: outAxis);
        expect(identical(resAxis, outAxis), true);
        expect(outAxis.toList(), [1.0, 2.0]);
      }),
    );

    test(
      'disposed out array throws StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final out = NDArray.zeros([], DType.float64);
        out.dispose();

        expect(() => min(a, out: out), throwsStateError);
        expect(() => max(a, out: out), throwsStateError);
        expect(() => nanmin(a, out: out), throwsStateError);
        expect(() => nanmax(a, out: out), throwsStateError);
      }),
    );

    test(
      'incompatible out array throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        final outShape = NDArray.zeros([2], DType.float64);
        expect(() => min(a, out: outShape), throwsArgumentError);
        expect(() => max(a, out: outShape), throwsArgumentError);
        expect(() => nanmin(a, out: outShape), throwsArgumentError);
        expect(() => nanmax(a, out: outShape), throwsArgumentError);

        final outDType = NDArray.zeros([], DType.int32);
        expect(
          () => min(a, out: outDType as dynamic),
          throwsA(anyOf(isA<TypeError>(), isA<ArgumentError>())),
        );
        expect(
          () => max(a, out: outDType as dynamic),
          throwsA(anyOf(isA<TypeError>(), isA<ArgumentError>())),
        );
        expect(
          () => nanmin(a, out: outDType as dynamic),
          throwsA(anyOf(isA<TypeError>(), isA<ArgumentError>())),
        );
        expect(
          () => nanmax(a, out: outDType as dynamic),
          throwsA(anyOf(isA<TypeError>(), isA<ArgumentError>())),
        );
      }),
    );

    test(
      'non-contiguous out array support in min/max/nanmin/nanmax',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final parent = NDArray.zeros([2, 2], DType.float64);
        final nonContiguousOut = parent.slice([Slice.all(), Index(0)]);
        expect(nonContiguousOut.isContiguous, false);

        final rMin = min(a, axis: 1, out: nonContiguousOut);
        expect(rMin, same(nonContiguousOut));
        expect(rMin.toList(), [1.0, 3.0]);
      }),
    );

    test(
      'disposed input array throws StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();

        expect(() => min(a), throwsStateError);
        expect(() => max(a), throwsStateError);
        expect(() => nanmin(a), throwsStateError);
        expect(() => nanmax(a), throwsStateError);
      }),
    );

    test(
      'invalid axis throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => min(a, axis: 2), throwsArgumentError);
        expect(() => min(a, axis: -2), throwsArgumentError);
        expect(() => nanmin(a, axis: 2), throwsArgumentError);
        expect(() => nanmin(a, axis: -2), throwsArgumentError);
      }),
    );

    test(
      'empty arrays / zero-sized axis',
      () => NDArray.scope(() {
        final empty = NDArray.zeros([0], DType.float64);
        expect(() => min(empty), throwsArgumentError);
        expect(() => max(empty), throwsArgumentError);
        expect(() => nanmin(empty), throwsArgumentError);
        expect(() => nanmax(empty), throwsArgumentError);

        final empty2D = NDArray.zeros([2, 0], DType.float64);
        expect(() => min(empty2D, axis: 1), throwsArgumentError);
        expect(() => max(empty2D, axis: 1), throwsArgumentError);
        expect(() => nanmin(empty2D, axis: 1), throwsArgumentError);
        expect(() => nanmax(empty2D, axis: 1), throwsArgumentError);

        expect(min(empty2D, axis: 0).shape, [0]);
        expect(max(empty2D, axis: 0).shape, [0]);
        expect(nanmin(empty2D, axis: 0).shape, [0]);
        expect(nanmax(empty2D, axis: 0).shape, [0]);
      }),
    );
  });

  group("0-Dimensional (rank == 0) Ufuncs Tests", () {
    test("Unary and binary strided ufuncs on 0-D arrays", () {
      NDArray.scope(() {
        final a = NDArray.scalar(4.0, dtype: DType.float64);
        final b = NDArray.scalar(2.0, dtype: DType.float64);

        final rSqrt = sqrt(a);
        expect(rSqrt.rank, 0);
        expect(rSqrt.scalar, closeTo(2.0, 1e-7));

        final rAdd = add(a, b);
        expect(rAdd.rank, 0);
        expect(rAdd.scalar, closeTo(6.0, 1e-7));

        final rMul = multiply(a, b);
        expect(rMul.rank, 0);
        expect(rMul.scalar, closeTo(8.0, 1e-7));

        // Test with a sliced 0-D view (offsetElements > 0)
        final arr = NDArray.fromList([10.0, 16.0, 25.0], [3], DType.float64);
        final view0D = NDArray.view(
          arr,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final rViewSqrt = sqrt((view0D as NDArray<AnySpec>));
        expect(rViewSqrt.rank, 0);
        expect(rViewSqrt.scalar, closeTo(4.0, 1e-7));
      });
    });
  });

  group("Where Mask Optional Parameter Tests", () {
    test("Logical comparison and binary logical ufuncs with where mask", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final b = NDArray.fromList([1, 1, 4, 4], [4], DType.int32);
        final whereMask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );

        // Pre-fill result buffer with false so masked-out elements remain false
        final outEq = NDArray<Boolean>.zeros([4], DType.boolean);
        equal(a, b, where: whereMask, out: outEq);
        expect(outEq.toList(), [true, false, false, false]);

        final outGt = NDArray.fromList(
          [true, true, true, true],
          [4],
          DType.boolean,
        );
        greater(a, b, where: whereMask, out: outGt);
        expect(outGt.toList(), [false, true, false, true]);

        final boolA = NDArray.fromList(
          [true, true, false, false],
          [4],
          DType.boolean,
        );
        final boolB = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final outAnd = NDArray.fromList(
          [false, false, false, false],
          [4],
          DType.boolean,
        );
        logicalAnd(boolA, boolB, where: whereMask, out: outAnd);
        expect(outAnd.toList(), [true, false, false, false]);
      });
    });

    test("mod and clipArray with optional where mask", () {
      NDArray.scope(() {
        final a = NDArray.fromList([5, 6, 7, 8], [4], DType.int32);
        final b = NDArray.fromList([3, 3, 3, 3], [4], DType.int32);
        final mask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );

        final outMod = NDArray.fromList([-1, -1, -1, -1], [4], DType.int32);
        mod(a, b, where: mask, out: outMod);
        expect(outMod.toList(), [2, -1, 1, -1]);

        final outClip = NDArray.fromList([99, 99, 99, 99], [4], DType.int32);
        final minB = NDArray.scalar(6, dtype: DType.int32);
        final maxB = NDArray.scalar(7, dtype: DType.int32);
        clipArray(a, min: minB, max: maxB, where: mask, out: outClip);
        expect(outClip.toList(), [6, 99, 7, 99]);
      });
    });
  });
}
