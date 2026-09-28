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
import 'dart:math' as math;

void main() {
  group('Ufuncs - log2, log10, reciprocal, positive', () {
    group('log2', () {
      test('float64/float32 contiguous log2', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [4], DType.float64);
          final res = log2(a);
          expect(res.dtype, DType.float64);
          expect(res.toList()[0], closeTo(0.0, 1e-10));
          expect(res.toList()[1], closeTo(1.0, 1e-10));
          expect(res.toList()[2], closeTo(2.0, 1e-10));
          expect(res.toList()[3], closeTo(3.0, 1e-10));

          final b = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [4], DType.float32);
          final resB = log2(b);
          expect(resB.dtype, DType.float32);
          expect(resB.toList()[0], closeTo(0.0, 1e-7));
          expect(resB.toList()[1], closeTo(1.0, 1e-7));
          expect(resB.toList()[2], closeTo(2.0, 1e-7));
          expect(resB.toList()[3], closeTo(3.0, 1e-7));
        });
      });

      test('float64 strided log2', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 999.0, 2.0, 999.0, 4.0, 999.0, 8.0],
            [7],
            DType.float64,
          );
          final sliced = a.slice([
            const Slice(start: 0, stop: 7, step: 2),
          ]); // [1, 2, 4, 8]
          expect(sliced.isContiguous, false);

          final res = log2(sliced);
          expect(res.toList()[0], closeTo(0.0, 1e-10));
          expect(res.toList()[1], closeTo(1.0, 1e-10));
          expect(res.toList()[2], closeTo(2.0, 1e-10));
          expect(res.toList()[3], closeTo(3.0, 1e-10));
        });
      });

      test('complex128/complex64 log2', () {
        NDArray.scope(() {
          // log2(1 + 1i) = log(1+1i)/log(2) = (log(sqrt(2)) + i*pi/4) / log(2)
          // = (0.5 * log(2) + i*pi/4) / log(2) = 0.5 + i * pi / (4 * log(2))
          final expectedVal = Complex(0.5, math.pi / (4.0 * math.log(2.0)));

          final a = NDArray.fromList(
            [Complex(1.0, 1.0)],
            [1],
            DType.complex128,
          );
          final res = log2(a);
          expect(res.dtype, DType.complex128);
          expect(res.toList()[0].real, closeTo(expectedVal.real, 1e-10));
          expect(res.toList()[0].imag, closeTo(expectedVal.imag, 1e-10));

          final b = NDArray.fromList([Complex(1.0, 1.0)], [1], DType.complex64);
          final resB = log2(b);
          expect(resB.dtype, DType.complex64);
          expect(resB.toList()[0].real, closeTo(expectedVal.real, 1e-6));
          expect(resB.toList()[0].imag, closeTo(expectedVal.imag, 1e-6));
        });
      });

      test('integer upcasting log2', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 4, 8], [4], DType.int32);
          final res = log2(a);
          expect(res.dtype, DType.float64);
          expect(res.toList()[0], closeTo(0.0, 1e-10));
          expect(res.toList()[1], closeTo(1.0, 1e-10));
          expect(res.toList()[2], closeTo(2.0, 1e-10));
          expect(res.toList()[3], closeTo(3.0, 1e-10));
        });
      });

      test('log2 edge cases (negative values)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([-1.0], [1], DType.float64);
          final res = log2(a);
          expect(res.toList()[0].isNaN, true);

          // For complex, log2(-1) = log(-1)/log(2) = i*pi / log(2)
          final b = NDArray.fromList(
            [Complex(-1.0, 0.0)],
            [1],
            DType.complex128,
          );
          final resB = log2(b);
          expect(resB.toList()[0].real, closeTo(0.0, 1e-10));
          expect(
            resB.toList()[0].imag,
            closeTo(math.pi / math.log(2.0), 1e-10),
          );
        });
      });
    });

    group('log10', () {
      test('float64/float32 contiguous log10', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 10.0, 100.0, 1000.0],
            [4],
            DType.float64,
          );
          final res = log10(a);
          expect(res.dtype, DType.float64);
          expect(res.toList()[0], closeTo(0.0, 1e-10));
          expect(res.toList()[1], closeTo(1.0, 1e-10));
          expect(res.toList()[2], closeTo(2.0, 1e-10));
          expect(res.toList()[3], closeTo(3.0, 1e-10));
        });
      });

      test('complex128 log10', () {
        NDArray.scope(() {
          // log10(10 + 0i) = 1
          final a = NDArray.fromList(
            [Complex(10.0, 0.0)],
            [1],
            DType.complex128,
          );
          final res = log10(a);
          expect(res.toList()[0].real, closeTo(1.0, 1e-10));
          expect(res.toList()[0].imag, closeTo(0.0, 1e-10));
        });
      });

      test('log10 edge cases', () {
        NDArray.scope(() {
          final a = NDArray.fromList([-10.0], [1], DType.float64);
          final res = log10(a);
          expect(res.toList()[0].isNaN, true);
        });
      });
    });

    group('reciprocal', () {
      test('float64/float32 reciprocal', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
          final res = reciprocal(a);
          expect(res.dtype, DType.float64);
          expect(res.toList()[0], closeTo(1.0, 1e-10));
          expect(res.toList()[1], closeTo(0.5, 1e-10));
          expect(res.toList()[2], closeTo(0.25, 1e-10));

          final b = NDArray.fromList([0.0], [1], DType.float64);
          final resB = reciprocal(b);
          expect(resB.toList()[0], double.infinity);
        });
      });

      test('complex reciprocal', () {
        NDArray.scope(() {
          // 1 / (1 + 1i) = (1 - 1i) / 2 = 0.5 - 0.5i
          final a = NDArray.fromList(
            [Complex(1.0, 1.0)],
            [1],
            DType.complex128,
          );
          final res = reciprocal(a);
          expect(res.toList()[0].real, closeTo(0.5, 1e-10));
          expect(res.toList()[0].imag, closeTo(-0.5, 1e-10));
        });
      });

      test('integer reciprocal and division by zero', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, -1], [3], DType.int32);
          final res = reciprocal(a);
          expect(res.dtype, DType.int32);
          expect(res.toList()[0], 1);
          expect(res.toList()[1], 0); // 1/2 = 0 in integer arithmetic
          expect(res.toList()[2], -1);

          final zero = NDArray.fromList([1, 0], [2], DType.int32);
          expect(() => reciprocal(zero), throwsA(isA<UnsupportedError>()));
        });
      });

      test('strided integer reciprocal', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 999, 2, 999, -1], [5], DType.int32);
          final sliced = a.slice([
            const Slice(start: 0, stop: 5, step: 2),
          ]); // [1, 2, -1]
          final res = reciprocal(sliced);
          expect(res.toList()[0], 1);
          expect(res.toList()[1], 0);
          expect(res.toList()[2], -1);
        });
      });
    });

    group('positive', () {
      test('positive returns copy for all numeric types', () {
        NDArray.scope(() {
          final dtypes = [
            DType.float64,
            DType.float32,
            DType.complex128,
            DType.complex64,
            DType.int64,
            DType.int32,
            DType.int16,
            DType.uint8,
          ];

          for (final dtype in dtypes) {
            final a = NDArray.ones([3], dtype);
            final res = positive(a);
            expect(res, isNot(same(a))); // Must be a copy
            expect(res.dtype, dtype);
            expect(res.shape, [3]);
            if (dtype.isComplex) {
              expect(res.toList()[0], Complex(1.0, 0.0));
            } else {
              expect(res.toList()[0], 1);
            }
          }
        });
      });

      test('positive strided', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 999.0, 2.0], [3], DType.float64);
          final sliced = a.slice([
            const Slice(start: 0, stop: 3, step: 2),
          ]); // [1.0, 2.0]
          final res = positive(sliced);
          expect(
            res.isContiguous,
            true,
          ); // Output of ufunc is contiguous by default if new
          expect(res.toList()[0], 1.0);
          expect(res.toList()[1], 2.0);
        });
      });

      test('positive on boolean throws UnsupportedError', () {
        NDArray.scope(() {
          final a = NDArray.fromList([true, false], [2], DType.boolean);
          expect(() => positive(a), throwsA(isA<UnsupportedError>()));
        });
      });
    });
  });
}
