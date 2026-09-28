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

import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('cond()', () {
    test('2D diagonal matrix across all supported p norms', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [2.0, 0.0, 0.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final cDefault = cond(a);
        expect(cDefault.dtype, equals(DType.float64));
        expect(cDefault.shape, equals(<int>[]));
        expect(cDefault.scalar, closeTo(2.0, 1e-12));

        final c2 = cond(a, p: 2);
        expect(c2.scalar, closeTo(2.0, 1e-12));

        final cNeg2 = cond(a, p: -2);
        expect(cNeg2.scalar, closeTo(0.5, 1e-12));

        final c1 = cond(a, p: 1);
        expect(c1.scalar, closeTo(2.0, 1e-12));

        final cNeg1 = cond(a, p: -1);
        expect(cNeg1.scalar, closeTo(0.5, 1e-12));

        final cInf = cond(a, p: double.infinity);
        expect(cInf.scalar, closeTo(2.0, 1e-12));

        final cNegInf = cond(a, p: -double.infinity);
        expect(cNegInf.scalar, closeTo(0.5, 1e-12));

        final cFro = cond(a, p: NormKind.frobenius);
        expect(cFro.scalar, closeTo(2.5, 1e-12));

        final cFroStr = cond(a, p: 'fro');
        expect(cFroStr.scalar, closeTo(2.5, 1e-12));

        expect(cond(a, p: NormKind.l2).scalar, closeTo(2.0, 1e-12));
        expect(cond(a, p: NormKind.negL2).scalar, closeTo(0.5, 1e-12));
        expect(cond(a, p: NormKind.l1).scalar, closeTo(2.0, 1e-12));
        expect(cond(a, p: NormKind.negL1).scalar, closeTo(0.5, 1e-12));
        expect(cond(a, p: NormKind.infinity).scalar, closeTo(2.0, 1e-12));
        expect(cond(a, p: NormKind.negInfinity).scalar, closeTo(0.5, 1e-12));
      });
    });

    test('2D non-diagonal matrix with p = 1, -1, inf, -inf, frobenius', () {
      NDArray.scope(() {
        final b = NDArray<Float64>.fromList(
          [1.0, 2.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        expect(cond(b, p: 1).scalar, closeTo(9.0, 1e-12));
        expect(cond(b, p: -1).scalar, closeTo(1.0, 1e-12));
        expect(cond(b, p: double.infinity).scalar, closeTo(9.0, 1e-12));
        expect(cond(b, p: -double.infinity).scalar, closeTo(1.0, 1e-12));
        expect(cond(b, p: NormKind.frobenius).scalar, closeTo(6.0, 1e-12));
      });
    });

    test('non-square matrix with default p (2-norm) and p = -2', () {
      NDArray.scope(() {
        final rect = NDArray<Float64>.fromList(
          [2.0, 0.0, 0.0, 0.0, 4.0, 0.0],
          [2, 3],
          DType.float64,
        );
        expect(cond(rect).scalar, closeTo(2.0, 1e-12));
        expect(cond(rect, p: -2).scalar, closeTo(0.5, 1e-12));
        expect(() => cond(rect, p: 1), throwsArgumentError);
      });
    });

    test(
      'singular and zero matrices return double.infinity and NaN matrices match NumPy',
      () {
        NDArray.scope(() {
          final sing = NDArray<Float64>.fromList(
            [1.0, 0.0, 0.0, 0.0],
            [2, 2],
            DType.float64,
          );
          expect(cond(sing).scalar, equals(double.infinity));
          expect(cond(sing, p: 2).scalar, equals(double.infinity));
          expect(cond(sing, p: -2).scalar, equals(0.0));
          expect(cond(sing, p: 1).scalar, equals(double.infinity));
          expect(cond(sing, p: -1).scalar, equals(double.infinity));
          expect(
            cond(sing, p: double.infinity).scalar,
            equals(double.infinity),
          );
          expect(
            cond(sing, p: -double.infinity).scalar,
            equals(double.infinity),
          );
          expect(
            cond(sing, p: NormKind.frobenius).scalar,
            equals(double.infinity),
          );

          final zeros = NDArray<Float64>.zeros([2, 2], DType.float64);
          expect(cond(zeros).scalar, equals(double.infinity));
          expect(cond(zeros, p: 2).scalar, equals(double.infinity));
          expect(cond(zeros, p: -2).scalar, equals(double.infinity));
          expect(cond(zeros, p: 1).scalar, equals(double.infinity));
          expect(cond(zeros, p: -1).scalar, equals(double.infinity));
          expect(
            cond(zeros, p: double.infinity).scalar,
            equals(double.infinity),
          );
          expect(
            cond(zeros, p: -double.infinity).scalar,
            equals(double.infinity),
          );
          expect(
            cond(zeros, p: NormKind.frobenius).scalar,
            equals(double.infinity),
          );

          final nans = NDArray<Float64>.fromList(
            [double.nan, 0.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );
          expect(() => cond(nans), throwsA(isA<LinAlgException>()));
          expect(() => cond(nans, p: 2), throwsA(isA<LinAlgException>()));
          expect(() => cond(nans, p: -2), throwsA(isA<LinAlgException>()));
          expect(cond(nans, p: 1).scalar.isNaN, isTrue);
          expect(cond(nans, p: -1).scalar.isNaN, isTrue);
          expect(cond(nans, p: double.infinity).scalar.isNaN, isTrue);
          expect(cond(nans, p: -double.infinity).scalar.isNaN, isTrue);
          expect(cond(nans, p: NormKind.frobenius).scalar.isNaN, isTrue);
        });
      },
    );

    test('3D batch matrices including singular slice and out parameter', () {
      NDArray.scope(() {
        // Batch of 2 matrices:
        // Matrix 0: [[2, 0], [0, 4]] -> cond_2 = 2.0, cond_1 = 2.0, cond_fro = 2.5
        // Matrix 1: [[1, 2], [0, 1]] -> cond_1 = 9.0, cond_fro = 6.0
        final batch = NDArray<Float64>.fromList(
          [
            2.0, 0.0, 0.0, 4.0, //
            1.0, 2.0, 0.0, 1.0,
          ],
          [2, 2, 2],
          DType.float64,
        );

        final resFro = cond(batch, p: NormKind.frobenius);
        expect(resFro.shape, equals([2]));
        expect(resFro.toList()[0], closeTo(2.5, 1e-12));
        expect(resFro.toList()[1], closeTo(6.0, 1e-12));

        final out1 = NDArray<Float64>.zeros([2], DType.float64);
        final res1 = cond(batch, p: 1, out: out1);
        expect(identical(res1, out1), isTrue);
        expect(out1.toList()[0], closeTo(2.0, 1e-12));
        expect(out1.toList()[1], closeTo(9.0, 1e-12));

        // Batch with one invertible and one singular matrix
        final batchSing = NDArray<Float64>.fromList(
          [
            2.0, 0.0, 0.0, 4.0, //
            1.0, 0.0, 0.0, 0.0,
          ],
          [2, 2, 2],
          DType.float64,
        );
        final resSing2 = cond(batchSing);
        expect(resSing2.toList()[0], closeTo(2.0, 1e-12));
        expect(resSing2.toList()[1], equals(double.infinity));

        final resSing1 = cond(batchSing, p: 1);
        expect(resSing1.toList()[0], closeTo(2.0, 1e-12));
        expect(resSing1.toList()[1], equals(double.infinity));
      });
    });

    test(
      'out parameter for 2D matrix, aliasing, and float32 / integer inputs',
      () {
        NDArray.scope(() {
          final a32 = NDArray<Float32>.fromList(
            [2.0, 0.0, 0.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final out32 = NDArray<Float32>.scalar(0.0, dtype: DType.float32);
          final res32 = cond(a32, out: out32);
          expect(identical(res32, out32), isTrue);
          expect(res32.dtype, equals(DType.float32));
          expect(res32.scalar, closeTo(2.0, 1e-5));

          final aInt = NDArray<Int32>.fromList(
            [2, 0, 0, 4],
            [2, 2],
            DType.int32,
          );
          final resInt = cond(aInt, p: 1);
          expect(resInt.dtype, equals(DType.float64));
          expect(resInt.scalar, closeTo(2.0, 1e-12));

          // out aliasing with a
          final aAlias = NDArray<Float64>.fromList(
            [2.0, 0.0, 0.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outAlias = NDArray<Float64>.view(
            aAlias,
            shape: [],
            strides: [],
            offsetElements: 0,
          );
          cond(aAlias, out: outAlias);
          expect(outAlias.scalar, closeTo(2.0, 1e-12));
        });
      },
    );
  });

  group('matmul() expanded DType support', () {
    test('float16 matmul and out parameter', () {
      NDArray.scope(() {
        final a = NDArray<Float16>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float16,
        );
        final b = NDArray<Float16>.fromList(
          [2.0, 0.0, 1.0, 2.0],
          [2, 2],
          DType.float16,
        );
        final res = matmul<Float16>(a, b);
        expect(res.dtype, equals(DType.float16));
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([4.0, 4.0, 10.0, 8.0]));

        final out = NDArray<Float16>.zeros([2, 2], DType.float16);
        final resOut = matmul<Float16>(a, b, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.toList(), equals([4.0, 4.0, 10.0, 8.0]));
      });
    });

    test('bfloat16 matmul and out parameter', () {
      NDArray.scope(() {
        final a = NDArray<BFloat16>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.bfloat16,
        );
        final b = NDArray<BFloat16>.fromList(
          [2.0, 0.0, 1.0, 2.0],
          [2, 2],
          DType.bfloat16,
        );
        final res = matmul<BFloat16>(a, b);
        expect(res.dtype, equals(DType.bfloat16));
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([4.0, 4.0, 10.0, 8.0]));

        final out = NDArray<BFloat16>.zeros([2, 2], DType.bfloat16);
        final resOut = matmul<BFloat16>(a, b, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.toList(), equals([4.0, 4.0, 10.0, 8.0]));
      });
    });

    test('int8 matmul and out parameter', () {
      NDArray.scope(() {
        final a = NDArray<Int8>.fromList([1, 2, 3, 4], [2, 2], DType.int8);
        final b = NDArray<Int8>.fromList([2, -1, 1, 2], [2, 2], DType.int8);
        final res = matmul<Int8>(a, b);
        expect(res.dtype, equals(DType.int8));
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([4, 3, 10, 5]));

        final out = NDArray<Int8>.zeros([2, 2], DType.int8);
        final resOut = matmul<Int8>(a, b, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.toList(), equals([4, 3, 10, 5]));
      });
    });

    test('uint16 matmul and out parameter', () {
      NDArray.scope(() {
        final a = NDArray<Uint16>.fromList(
          [10, 20, 30, 40],
          [2, 2],
          DType.uint16,
        );
        final b = NDArray<Uint16>.fromList([2, 1, 1, 2], [2, 2], DType.uint16);
        final res = matmul<Uint16>(a, b);
        expect(res.dtype, equals(DType.uint16));
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([40, 50, 100, 110]));

        final out = NDArray<Uint16>.zeros([2, 2], DType.uint16);
        final resOut = matmul<Uint16>(a, b, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.toList(), equals([40, 50, 100, 110]));
      });
    });

    test('uint32 matmul and out parameter', () {
      NDArray.scope(() {
        final a = NDArray<Uint32>.fromList(
          [100, 200, 300, 400],
          [2, 2],
          DType.uint32,
        );
        final b = NDArray<Uint32>.fromList([2, 1, 1, 2], [2, 2], DType.uint32);
        final res = matmul<Uint32>(a, b);
        expect(res.dtype, equals(DType.uint32));
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([400, 500, 1000, 1100]));

        final out = NDArray<Uint32>.zeros([2, 2], DType.uint32);
        final resOut = matmul<Uint32>(a, b, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.toList(), equals([400, 500, 1000, 1100]));
      });
    });

    test('uint64 matmul and out parameter including values >= 2^63', () {
      NDArray.scope(() {
        final a = NDArray<Uint64>.fromList(
          [1000, 2000, 3000, 4000],
          [2, 2],
          DType.uint64,
        );
        final b = NDArray<Uint64>.fromList([2, 1, 1, 2], [2, 2], DType.uint64);
        final res = matmul<Uint64>(a, b);
        expect(res.dtype, equals(DType.uint64));
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([4000, 5000, 10000, 11000]));

        final out = NDArray<Uint64>.zeros([2, 2], DType.uint64);
        final resOut = matmul<Uint64>(a, b, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.toList(), equals([4000, 5000, 10000, 11000]));

        // Test values >= 2^63 wrapping modulo 2^64 without clamping
        // val = 2^63 + 3 = 0x8000000000000003 (in signed 64-bit int: -9223372036854775805)
        // val * 2 mod 2^64 = 6
        final largeVal = -9223372036854775805;
        final aLarge = NDArray<Uint64>.fromList(
          [largeVal, 0, 0, largeVal],
          [2, 2],
          DType.uint64,
        );
        final bTwo = NDArray<Uint64>.fromList(
          [2, 0, 0, 2],
          [2, 2],
          DType.uint64,
        );
        final resLarge = matmul<Uint64>(aLarge, bTwo);
        expect(resLarge.toList(), equals([6, 0, 0, 6]));
      });
    });
  });
}
