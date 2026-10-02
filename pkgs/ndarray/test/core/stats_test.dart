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
  group('Stats Operations Tests', () {
    group('all() and any()', () {
      test(
        'all() global reduction',
        () => NDArray.scope(() {
          final a1 = NDArray<Boolean>.fromList(
            [true, true, true],
            [3],
            DType.boolean,
          );
          final a2 = NDArray<Boolean>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          expect(all(a1).scalar, isTrue);
          expect(all(a2).scalar, isFalse);

          final empty = NDArray<Boolean>.zeros([0], DType.boolean);
          expect(all(empty).scalar, isTrue);
        }),
      );

      test(
        'any() global reduction',
        () => NDArray.scope(() {
          final a1 = NDArray<Boolean>.fromList(
            [false, false, false],
            [3],
            DType.boolean,
          );
          final a2 = NDArray<Boolean>.fromList(
            [false, true, false],
            [3],
            DType.boolean,
          );
          expect(any(a1).scalar, isFalse);
          expect(any(a2).scalar, isTrue);

          final empty = NDArray<Boolean>.zeros([0], DType.boolean);
          expect(any(empty).scalar, isFalse);
        }),
      );

      test(
        'all() and any() axis reduction',
        () => NDArray.scope(() {
          final a = NDArray<Boolean>.fromList(
            [true, false, true, true],
            [2, 2],
            DType.boolean,
          );

          final allAxis0 = all(a, axis: 0);
          expect(allAxis0.shape, equals([2]));
          expect(allAxis0.toList(), equals([true, false]));

          final allAxis1 = all(a, axis: 1);
          expect(allAxis1.shape, equals([2]));
          expect(allAxis1.toList(), equals([false, true]));

          final anyAxis0 = any(a, axis: 0);
          expect(anyAxis0.shape, equals([2]));
          expect(anyAxis0.toList(), equals([true, true]));

          final anyAxis1 = any(a, axis: 1);
          expect(anyAxis1.shape, equals([2]));
          expect(anyAxis1.toList(), equals([true, true]));
        }),
      );

      test(
        'all() and any() axis range error',
        () => NDArray.scope(() {
          final a = NDArray<Boolean>.zeros([2, 3], DType.boolean);
          expect(() => all(a, axis: 2), throwsRangeError);
          expect(() => all(a, axis: -3), throwsRangeError);
          expect(() => any(a, axis: 2), throwsRangeError);
          expect(() => any(a, axis: -3), throwsRangeError);
        }),
      );
    });

    group('nanvar()', () {
      test(
        'nanvar() with ddof: 1 and axis',
        () => NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, double.nan, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final v0 = nanvar(a, axis: 0, ddof: 1);
          expect(v0.shape, equals([3]));
          expect(v0.getCellFlat(0), closeTo(4.5, 1e-6));
          expect(v0.getCellFlat(1).isNaN, isTrue);
          expect(v0.getCellFlat(2), closeTo(4.5, 1e-6));
        }),
      );

      test(
        'nanvar() axis out of range error',
        () => NDArray.scope(() {
          final a = NDArray<Float64>.zeros([2, 2], DType.float64);
          expect(() => nanvar(a, axis: 2), throwsRangeError);
          expect(() => nanvar(a, axis: -3), throwsRangeError);
        }),
      );
    });

    group('Complex min() and max()', () {
      test(
        'Complex128 min() and max() global',
        () => NDArray.scope(() {
          final a = NDArray<Complex128>.fromList(
            [
              Complex(1.0, 2.0),
              Complex(0.0, 5.0),
              Complex(1.0, -1.0),
              Complex(-2.0, 0.0),
            ],
            [4],
            DType.complex128,
          );

          final mn = min(a);
          expect(mn.scalar, equals(Complex(-2.0, 0.0)));

          final mx = max(a);
          expect(mx.scalar, equals(Complex(1.0, 2.0)));
        }),
      );

      test(
        'Complex128 min() and max() axis',
        () => NDArray.scope(() {
          final a = NDArray<Complex128>.fromList(
            [
              Complex(1.0, 2.0),
              Complex(3.0, 4.0),
              Complex(2.0, 1.0),
              Complex(0.0, 5.0),
            ],
            [2, 2],
            DType.complex128,
          );

          final mn0 = min(a, axis: 0);
          expect(mn0.shape, equals([2]));
          expect(mn0.toList(), equals([Complex(1.0, 2.0), Complex(0.0, 5.0)]));

          final mx0 = max(a, axis: 0);
          expect(mx0.shape, equals([2]));
          expect(mx0.toList(), equals([Complex(2.0, 1.0), Complex(3.0, 4.0)]));
        }),
      );

      test(
        'Complex64 min() and max() axis',
        () => NDArray.scope(() {
          final a = NDArray<Complex64>.fromList(
            [
              Complex(1.0, 2.0),
              Complex(3.0, 4.0),
              Complex(2.0, 1.0),
              Complex(0.0, 5.0),
            ],
            [2, 2],
            DType.complex64,
          );

          final mn0 = min(a, axis: 0);
          expect(mn0.shape, equals([2]));
          expect(mn0.toList(), equals([Complex(1.0, 2.0), Complex(0.0, 5.0)]));

          final mx0 = max(a, axis: 0);
          expect(mx0.shape, equals([2]));
          expect(mx0.toList(), equals([Complex(2.0, 1.0), Complex(3.0, 4.0)]));
        }),
      );
    });

    group('nansum() across all 14 DTypes', () {
      test(
        'Float64 nansum',
        () => NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, double.nan, 3.0, 4.0, 5.0, double.nan],
            [2, 3],
            DType.float64,
          );

          final sGlobal = nansum(a);
          expect(sGlobal.dtype, equals(DType.float64));
          expect(sGlobal.scalar, equals(13.0));

          final sAxis0 = nansum(a, axis: 0);
          expect(sAxis0.shape, equals([3]));
          expect(sAxis0.toList(), equals([5.0, 5.0, 3.0]));
        }),
      );

      test(
        'Float32 nansum',
        () => NDArray.scope(() {
          final a = NDArray<Float32>.fromList(
            [1.0, double.nan, 3.0, 4.0, 5.0, double.nan],
            [2, 3],
            DType.float32,
          );

          final sGlobal = nansum(a);
          expect(sGlobal.dtype, equals(DType.float32));
          expect(sGlobal.scalar, equals(13.0));

          final sAxis0 = nansum(a, axis: 0);
          expect(sAxis0.shape, equals([3]));
          expect(sAxis0.toList(), equals([5.0, 5.0, 3.0]));
        }),
      );

      test(
        'Float16 nansum',
        () => NDArray.scope(() {
          final a = NDArray<Float16>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float16,
          );
          final s = nansum(a);
          expect(s.scalar, equals(6.0));
        }),
      );

      test(
        'BFloat16 nansum',
        () => NDArray.scope(() {
          final a = NDArray<BFloat16>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.bfloat16,
          );
          final s = nansum(a);
          expect(s.scalar, equals(6.0));
        }),
      );

      test(
        'Int8 nansum (delegates to sum, default Int64)',
        () => NDArray.scope(() {
          final a = NDArray<Int8>.fromList([1, 2, 3], [3], DType.int8);
          final s = nansum(a);
          expect(s.dtype, equals(DType.int64));
          expect(s.scalar, equals(6));
        }),
      );

      test(
        'Int16 nansum (delegates to sum, default Int64)',
        () => NDArray.scope(() {
          final a = NDArray<Int16>.fromList([10, 20, 30], [3], DType.int16);
          final s = nansum(a);
          expect(s.dtype, equals(DType.int64));
          expect(s.scalar, equals(60));
        }),
      );

      test(
        'Int32 nansum (delegates to sum, default Int64)',
        () => NDArray.scope(() {
          final a = NDArray<Int32>.fromList([100, 200, 300], [3], DType.int32);
          final s = nansum(a);
          expect(s.dtype, equals(DType.int64));
          expect(s.scalar, equals(600));
        }),
      );

      test(
        'Int64 nansum',
        () => NDArray.scope(() {
          final a = NDArray<Int64>.fromList(
            [1000, 2000, 3000],
            [3],
            DType.int64,
          );
          final s = nansum(a);
          expect(s.dtype, equals(DType.int64));
          expect(s.scalar, equals(6000));
        }),
      );

      test(
        'Uint8 nansum (delegates to sum, default Uint64)',
        () => NDArray.scope(() {
          final a = NDArray<Uint8>.fromList([1, 2, 3], [3], DType.uint8);
          final s = nansum(a);
          expect(s.dtype, equals(DType.uint64));
          expect(s.scalar, equals(6));
        }),
      );

      test(
        'Uint16 nansum (delegates to sum, default Uint64)',
        () => NDArray.scope(() {
          final a = NDArray<Uint16>.fromList([10, 20, 30], [3], DType.uint16);
          final s = nansum(a);
          expect(s.dtype, equals(DType.uint64));
          expect(s.scalar, equals(60));
        }),
      );

      test(
        'Uint32 nansum (delegates to sum, default Uint64)',
        () => NDArray.scope(() {
          final a = NDArray<Uint32>.fromList(
            [100, 200, 300],
            [3],
            DType.uint32,
          );
          final s = nansum(a);
          expect(s.dtype, equals(DType.uint64));
          expect(s.scalar, equals(600));
        }),
      );

      test(
        'Uint64 nansum',
        () => NDArray.scope(() {
          final a = NDArray<Uint64>.fromList(
            [1000, 2000, 3000],
            [3],
            DType.uint64,
          );
          final s = nansum(a);
          expect(s.dtype, equals(DType.uint64));
          expect(s.scalar, equals(6000));
        }),
      );

      test(
        'Boolean nansum (delegates to sum, default Int64)',
        () => NDArray.scope(() {
          final a = NDArray<Boolean>.fromList(
            [true, false, true, true],
            [4],
            DType.boolean,
          );
          final s = nansum(a);
          expect(s.dtype, equals(DType.int64));
          expect(s.scalar, equals(3));
        }),
      );

      test(
        'Complex128 nansum with NaN elements',
        () => NDArray.scope(() {
          final a = NDArray<Complex128>.fromList(
            [
              Complex(1.0, 2.0),
              Complex(double.nan, 0.0),
              Complex(3.0, 4.0),
              Complex(0.0, double.nan),
            ],
            [2, 2],
            DType.complex128,
          );

          final sGlobal = nansum(a);
          expect(sGlobal.scalar, equals(Complex(4.0, 6.0)));

          final sAxis0 = nansum(a, axis: 0);
          expect(sAxis0.shape, equals([2]));
          expect(
            sAxis0.toList(),
            equals([Complex(4.0, 6.0), Complex(0.0, 0.0)]),
          );

          final sAxis1 = nansum(a, axis: 1);
          expect(sAxis1.shape, equals([2]));
          expect(
            sAxis1.toList(),
            equals([Complex(1.0, 2.0), Complex(3.0, 4.0)]),
          );
        }),
      );

      test(
        'Complex64 nansum with NaN elements',
        () => NDArray.scope(() {
          final a = NDArray<Complex64>.fromList(
            [
              Complex(1.0, 2.0),
              Complex(double.nan, 0.0),
              Complex(3.0, 4.0),
              Complex(0.0, double.nan),
            ],
            [2, 2],
            DType.complex64,
          );

          final sGlobal = nansum(a);
          expect(sGlobal.scalar, equals(Complex(4.0, 6.0)));

          final sAxis0 = nansum(a, axis: 0);
          expect(sAxis0.shape, equals([2]));
          expect(
            sAxis0.toList(),
            equals([Complex(4.0, 6.0), Complex(0.0, 0.0)]),
          );

          final sAxis1 = nansum(a, axis: 1);
          expect(sAxis1.shape, equals([2]));
          expect(
            sAxis1.toList(),
            equals([Complex(1.0, 2.0), Complex(3.0, 4.0)]),
          );
        }),
      );

      test(
        'nansum axis out of bounds throws RangeError',
        () => NDArray.scope(() {
          final a = NDArray<Float64>.zeros([2, 3], DType.float64);
          expect(() => nansum(a, axis: 2), throwsRangeError);
          expect(() => nansum(a, axis: -3), throwsRangeError);
        }),
      );
    });

    group('Cumulative operations axis RangeError', () {
      test(
        'cumsum, cumprod, cummin, cummax axis out of bounds',
        () => NDArray.scope(() {
          final a = NDArray<Float64>.zeros([2, 3], DType.float64);
          expect(() => cumsum(a, axis: 2), throwsRangeError);
          expect(() => cumsum(a, axis: -3), throwsRangeError);
          expect(() => cumprod(a, axis: 2), throwsRangeError);
          expect(() => cumprod(a, axis: -3), throwsRangeError);
          expect(() => cummin(a, axis: 2), throwsRangeError);
          expect(() => cummin(a, axis: -3), throwsRangeError);
          expect(() => cummax(a, axis: 2), throwsRangeError);
          expect(() => cummax(a, axis: -3), throwsRangeError);
        }),
      );
    });

    group('nanmean and quantile axis RangeError', () {
      test(
        'nanmean and quantile axis out of bounds',
        () => NDArray.scope(() {
          final a = NDArray<Float64>.zeros([2, 3], DType.float64);
          expect(() => nanmean(a, axis: 2), throwsRangeError);
          expect(() => nanmean(a, axis: -3), throwsRangeError);
          expect(() => quantile(a, 0.5, axis: 2), throwsRangeError);
          expect(() => quantile(a, 0.5, axis: -3), throwsRangeError);
        }),
      );
    });
  });
}
