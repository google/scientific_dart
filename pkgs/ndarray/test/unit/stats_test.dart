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
  group('Stats Fixes & Edge Cases Tests', () {
    group('Float32 mean() buffer safety & accuracy', () {
      test('1D Float32 global mean', () {
        final a = NDArray<Float32>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float32,
        );
        final m = mean(a);
        expect(m.dtype, equals(DType.float64));
        expect(m.size, equals(1));
        expect(m.scalar, closeTo(3.0, 1e-6));
        a.dispose();
        m.dispose();
      });

      test('2D Float32 axis=0 mean', () {
        final a = NDArray<Float32>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float32,
        );
        final m0 = mean(a, axis: 0);
        expect(m0.dtype, equals(DType.float64));
        expect(m0.shape, equals([3]));
        expect(m0.toList(), equals([2.5, 3.5, 4.5]));
        a.dispose();
        m0.dispose();
      });

      test('2D Float32 axis=1 mean', () {
        final a = NDArray<Float32>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float32,
        );
        final m1 = mean(a, axis: 1);
        expect(m1.dtype, equals(DType.float64));
        expect(m1.shape, equals([2]));
        expect(m1.toList(), equals([2.0, 5.0]));
        a.dispose();
        m1.dispose();
      });

      test('3D Float32 axis-wise mean', () {
        final a = NDArray<Float32>.arange(
          0,
          24,
          dtype: DType.float32,
        ).reshape([2, 3, 4]);
        final mAxis1 = mean(a, axis: 1);
        expect(mAxis1.dtype, equals(DType.float64));
        expect(mAxis1.shape, equals([2, 4]));
        expect(
          mAxis1.toList(),
          equals([4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]),
        );
        a.dispose();
        mAxis1.dispose();
      });

      test('2D Float32 mean with out parameter', () {
        final a = NDArray<Float32>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float32,
        );
        final outBuf = NDArray<Float64>.zeros([2], DType.float64);
        final res = mean(a, axis: 1, out: outBuf);
        expect(identical(res, outBuf), isTrue);
        expect(res.toList(), equals([15.0, 35.0]));
        a.dispose();
        outBuf.dispose();
      });
    });

    group('Empty Array Reductions (sum, prod, mean)', () {
      test('sum() on empty 1D Float64 array', () {
        final empty = NDArray<Float64>.zeros([0], DType.float64);
        final s = sum(empty);
        expect(s.shape, equals(<int>[]));
        expect(s.scalar, equals(0.0));
        empty.dispose();
        s.dispose();
      });

      test('sum() on empty 1D Int32 array', () {
        final empty = NDArray<Int32>.zeros([0], DType.int32);
        final s = sum(empty);
        expect(s.shape, equals(<int>[]));
        expect(s.scalar, equals(0));
        empty.dispose();
        s.dispose();
      });

      test('sum() on empty 1D Complex128 array', () {
        final empty = NDArray<Complex128>.zeros([0], DType.complex128);
        final s = sum(empty);
        expect(s.shape, equals(<int>[]));
        expect(s.scalar, equals(Complex(0.0, 0.0)));
        empty.dispose();
        s.dispose();
      });

      test('prod() on empty 1D Float64 array returns identity 1.0', () {
        final empty = NDArray<Float64>.zeros([0], DType.float64);
        final p = prod(empty);
        expect(p.shape, equals(<int>[]));
        expect(p.scalar, equals(1.0));
        empty.dispose();
        p.dispose();
      });

      test('prod() on empty 1D Int64 array returns identity 1', () {
        final empty = NDArray<Int64>.zeros([0], DType.int64);
        final p = prod(empty);
        expect(p.shape, equals(<int>[]));
        expect(p.scalar, equals(1));
        empty.dispose();
        p.dispose();
      });

      test(
        'prod() on empty 1D Complex128 array returns identity Complex(1, 0)',
        () {
          final empty = NDArray<Complex128>.zeros([0], DType.complex128);
          final p = prod(empty);
          expect(p.shape, equals(<int>[]));
          expect(p.scalar, equals(Complex(1.0, 0.0)));
          empty.dispose();
          p.dispose();
        },
      );

      test('mean() on empty 1D Float64 array returns NaN', () {
        final empty = NDArray<Float64>.zeros([0], DType.float64);
        final m = mean(empty);
        expect(m.shape, equals(<int>[]));
        expect(m.scalar.isNaN, isTrue);
        empty.dispose();
        m.dispose();
      });

      test('sum() on 2D empty array [0, 5] along axis 0 returns zeros [5]', () {
        final empty = NDArray<Float64>.zeros([0, 5], DType.float64);
        final s = sum(empty, axis: 0);
        expect(s.shape, equals([5]));
        expect(s.toList(), equals([0.0, 0.0, 0.0, 0.0, 0.0]));
        empty.dispose();
        s.dispose();
      });

      test('prod() on 2D empty array [0, 5] along axis 0 returns ones [5]', () {
        final empty = NDArray<Float64>.zeros([0, 5], DType.float64);
        final p = prod(empty, axis: 0);
        expect(p.shape, equals([5]));
        expect(p.toList(), equals([1.0, 1.0, 1.0, 1.0, 1.0]));
        empty.dispose();
        p.dispose();
      });
    });
  });
}
