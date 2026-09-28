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
  group('Nullable where= mask array support in ufuncs', () {
    test('Unary ufunc (sqrt) with contiguous boolean mask', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [4.0, 9.0, 16.0, 25.0],
          [4],
          DType.float64,
        );
        final out = NDArray<Float64>.fromList(
          [100.0, 100.0, 100.0, 100.0],
          [4],
          DType.float64,
        );
        final mask = NDArray<Boolean>.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );

        sqrt(a, out: out, where: mask);

        expect(out.toList(), [2.0, 100.0, 4.0, 100.0]);
      });
    });

    test('Unary ufunc (sqrt) with contiguous uint8 mask', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 4.0, 9.0, 16.0], [4], DType.float64);
        final mask = NDArray.fromList([1, 0, 1, 0], [4], DType.uint8);
        final out = NDArray.zeros([4], DType.float64);

        final res = sqrt(a, where: mask, out: out);
        expect(res.data, [1.0, 0.0, 3.0, 0.0]);
      });
    });

    test('Unary ufunc (exp) with null mask (where=null)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);
        final res = exp(a, where: null);
        expect(res.data[0], closeTo(1.0, 1e-5));
        expect(res.data[1], closeTo(2.71828, 1e-5));
      });
    });

    test('Unary ufunc (sin) with where mask - strided view', () {
      NDArray.scope(() {
        final aFull = NDArray<Float64>.fromList(
          [0.0, 1.0, 0.0, 2.0],
          [4],
          DType.float64,
        );
        final a = aFull.slice([Slice(start: 0, stop: 4, step: 2)]);
        final out = NDArray<Float64>.fromList([99.0, 99.0], [2], DType.float64);
        final mask = NDArray<Boolean>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );

        sin(a, out: out, where: mask);

        expect(out.toList()[0], closeTo(0.0, 1e-6));
        expect(out.toList()[1], 99.0);
      });
    });

    test('Binary ufunc (add) with contiguous boolean mask', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
        );
        final out = NDArray<Float64>.fromList(
          [0.0, 0.0, 0.0, 0.0],
          [4],
          DType.float64,
        );
        final mask = NDArray<Boolean>.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );

        add(a, b, out: out, where: mask);

        expect(out.toList(), [11.0, 0.0, 33.0, 0.0]);
      });
    });

    test('Binary ufunc (multiply) with Int32 and boolean mask', () {
      NDArray.scope(() {
        final a = NDArray<Int32>.fromList([2, 3, 4, 5], [4], DType.int32);
        final b = NDArray<Int32>.fromList([10, 10, 10, 10], [4], DType.int32);
        final out = NDArray<Int32>.fromList([-1, -1, -1, -1], [4], DType.int32);
        final mask = NDArray<Boolean>.fromList(
          [false, true, false, true],
          [4],
          DType.boolean,
        );

        multiply(a, b, out: out, where: mask);

        expect(out.toList(), [-1, 30, -1, 50]);
      });
    });

    test('Binary ufunc (multiply) with 2D inputs and uint8 mask', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final out = NDArray.zeros([2, 2], DType.float64);

        final mask = NDArray.fromList([1, 0, 0, 1], [2, 2], DType.uint8);
        final res = multiply(a, b, where: mask, out: out);

        expect(res.shape, [2, 2]);
        expect(res.data, [10.0, 0.0, 0.0, 160.0]);
      });
    });

    test(
      'Binary ufunc (subtract) with strided view inputs and boolean mask',
      () {
        NDArray.scope(() {
          final aFull = NDArray<Float64>.fromList(
            [10.0, 0.0, 20.0, 0.0],
            [4],
            DType.float64,
          );
          final bFull = NDArray<Float64>.fromList(
            [2.0, 0.0, 5.0, 0.0],
            [4],
            DType.float64,
          );
          final a = aFull.slice([Slice(start: 0, stop: 4, step: 2)]);
          final b = bFull.slice([Slice(start: 0, stop: 4, step: 2)]);

          final out = NDArray<Float64>.fromList(
            [-5.0, -5.0],
            [2],
            DType.float64,
          );
          final mask = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );

          subtract(a, b, out: out, where: mask);

          expect(out.toList(), [8.0, -5.0]);
        });
      },
    );

    test('Binary ufunc (divide) with Complex128 and boolean mask', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [Complex(4.0, 2.0), Complex(6.0, 8.0)],
          [2],
          DType.complex128,
        );
        final b = NDArray<Complex128>.fromList(
          [Complex(2.0, 0.0), Complex(2.0, 0.0)],
          [2],
          DType.complex128,
        );
        final out = NDArray<Complex128>.fromList(
          [Complex(0.0, 0.0), Complex(0.0, 0.0)],
          [2],
          DType.complex128,
        );
        final mask = NDArray<Boolean>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );

        divide(a, b, out: out, where: mask);

        expect(out.toList()[0], Complex(2.0, 1.0));
        expect(out.toList()[1], Complex(0.0, 0.0));
      });
    });

    test('Broadcasted where= mask array (1D mask applied to 2D array)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 5.0, 5.0, 5.0], [2, 2], DType.float64);
        final mask = NDArray.fromList([1, 0], [1, 2], DType.uint8);
        final out = NDArray.zeros([2, 2], DType.float64);

        final res = add(a, b, where: mask, out: out);
        expect(res.shape, [2, 2]);
        expect(res.data, [6.0, 0.0, 8.0, 0.0]);
      });
    });

    test('Clip ufunc with mask', () {
      NDArray.scope(() {
        final a = NDArray.fromList([-5.0, 5.0, 15.0], [3], DType.float64);
        final mask = NDArray.fromList([1, 0, 1], [3], DType.uint8);
        final out = NDArray.zeros([3], DType.float64);

        final res = clip(a, min: 0, max: 10, where: mask, out: out);
        expect(res.data, [0.0, 0.0, 10.0]);
      });
    });

    test('Bitwise ufunc (bitwiseAnd) with mask', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0xFF, 0xFF, 0xFF], [3], DType.int32);
        final b = NDArray.fromList([0x0F, 0xF0, 0xAA], [3], DType.int32);
        final mask = NDArray.fromList([1, 0, 1], [3], DType.uint8);
        final out = NDArray.zeros([3], DType.int32);

        final res = bitwiseAnd(a, b, where: mask, out: out);
        expect(res.data, [0x0F, 0, 0xAA]);
      });
    });

    test('Throws StateError on disposed where= mask array', () {
      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final mask = NDArray.fromList([1, 1], [2], DType.uint8);
      mask.dispose();

      expect(() => add(a, a, where: mask), throwsStateError);
      a.dispose();
    });

    test('Default null where mask behaves normally without overhead', () {
      final a = NDArray<Float64>.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [4],
        DType.float64,
      );
      final b = NDArray<Float64>.fromList(
        [10.0, 20.0, 30.0, 40.0],
        [4],
        DType.float64,
      );

      final res = add(a, b);
      expect(res.toList(), [11.0, 22.0, 33.0, 44.0]);
    });
  });
}
