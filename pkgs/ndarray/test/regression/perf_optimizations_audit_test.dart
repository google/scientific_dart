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
  group('Performance Optimizations Audit Regression Tests', () {
    test(
      '1. interp handles extreme y1 - y0 slope overflow and xv == x0 without NaN',
      () {
        NDArray.scope(() {
          final xp = NDArray<Float64>.fromList(
            [0.0, 1.0, 2.0],
            [3],
            DType.float64,
          );
          final fp = NDArray<Float64>.fromList(
            [-1e308, 1e308, 1e308],
            [3],
            DType.float64,
          );
          final x = NDArray<Float64>.fromList(
            [0.0, 0.5, 1.0],
            [3],
            DType.float64,
          );

          final res = interp(x, xp, fp);
          expect(res.getCell([0]).isNaN, isFalse);
          expect(res.getCell([0]), equals(-1e308));
          expect(res.getCell([1]), closeTo(0.0, 1e293));
          expect(res.getCell([2]), equals(1e308));

          // Also test strided interp path
          final x2d = NDArray<Float64>.fromList(
            [0.0, 99.0, 0.5, 99.0, 1.0, 99.0],
            [3, 2],
            DType.float64,
          );
          final xCol = x2d.slice([Slice.all(), Index(0)]);
          expect(xCol.isContiguous, isFalse);
          final resStrided = interp(xCol, xp, fp);
          expect(resStrided.getCell([0]).isNaN, isFalse);
          expect(resStrided.getCell([0]), equals(-1e308));
          expect(resStrided.getCell([1]), closeTo(0.0, 1e293));
          expect(resStrided.getCell([2]), equals(1e308));
        });
      },
    );

    test(
      '2. partition and argpartition handle duplicate k values and NaNs with multiple k',
      () {
        NDArray.scope(() {
          // Duplicate k values [0, 2, 2, 4] previously caused k=4 to be skipped
          final a = NDArray<Float64>.fromList(
            [50.0, 40.0, 30.0, 20.0, 10.0],
            [5],
            DType.float64,
          );
          final p = partition(a, [0, 2, 2, 4]);
          expect(p.getCell([0]), equals(10.0));
          expect(p.getCell([2]), equals(30.0));
          expect(p.getCell([4]), equals(50.0));

          final ap = argpartition(a, [0, 2, 2, 4]);
          expect(
            a.getCell([
              ap.getCell([0]),
            ]),
            equals(10.0),
          );
          expect(
            a.getCell([
              ap.getCell([2]),
            ]),
            equals(30.0),
          );
          expect(
            a.getCell([
              ap.getCell([4]),
            ]),
            equals(50.0),
          );

          // Array with NaNs and multiple k values
          final withNan = NDArray<Float64>.fromList(
            [double.nan, 30.0, 10.0, double.nan, 20.0],
            [5],
            DType.float64,
          );
          final pNan = partition(withNan, [0, 1, 2, 3, 4]);
          expect(pNan.getCell([0]), equals(10.0));
          expect(pNan.getCell([1]), equals(20.0));
          expect(pNan.getCell([2]), equals(30.0));
          expect(pNan.getCell([3]).isNaN, isTrue);
          expect(pNan.getCell([4]).isNaN, isTrue);
        });
      },
    );

    test(
      '3. choose handles uint64 >= 2^63 correctly in clip, wrap, and raise modes',
      () {
        NDArray.scope(() {
          // -1 in signed 64-bit two's complement represents uint64 0xFFFFFFFFFFFFFFFF (2^64 - 1)
          // 2^64 - 1 = 18446744073709551615, which is divisible by 3 (18446744073709551615 % 3 == 0).
          final idx = NDArray<Uint64>.fromList([0, -1], [2], DType.uint64);
          final c0 = NDArray<Float64>.fromList(
            [10.0, 100.0],
            [2],
            DType.float64,
          );
          final c1 = NDArray<Float64>.fromList(
            [20.0, 200.0],
            [2],
            DType.float64,
          );
          final c2 = NDArray<Float64>.fromList(
            [30.0, 300.0],
            [2],
            DType.float64,
          );

          // In clip mode, 0xFFFFFFFFFFFFFFFF must clamp to nChoices - 1 = 2 (not 0!)
          final clipped = choose(idx, [c0, c1, c2], mode: ChooseMode.clip);
          expect(clipped.getCell([0]), equals(10.0));
          expect(clipped.getCell([1]), equals(300.0));

          // In wrap mode, (2^64 - 1) % 3 == 0 (not 2!)
          final wrapped = choose(idx, [c0, c1, c2], mode: ChooseMode.wrap);
          expect(wrapped.getCell([0]), equals(10.0));
          expect(wrapped.getCell([1]), equals(100.0));

          // In raise mode, 0xFFFFFFFFFFFFFFFF must throw RangeError
          expect(
            () => choose(idx, [c0, c1, c2], mode: ChooseMode.raise),
            throwsRangeError,
          );

          // Also test non-contiguous out buffer with choose
          final outFull = NDArray<Float64>.zeros([2, 2], DType.float64);
          final outCol = outFull.slice([Slice.all(), Index(1)]);
          expect(outCol.isContiguous, isFalse);
          choose(idx, [c0, c1, c2], mode: ChooseMode.clip, out: outCol);
          expect(outCol.getCell([0]), equals(10.0));
          expect(outCol.getCell([1]), equals(300.0));
        });
      },
    );

    test(
      '4. isClose and allClose avoid 64-bit integer overflow on int64 min value (-2^63)',
      () {
        NDArray.scope(() {
          const int64Min = -9223372036854775808;
          final a = NDArray<Int64>.fromList(
            [int64Min, int64Min, 100],
            [3],
            DType.int64,
          );
          final b = NDArray<Int64>.fromList(
            [0, int64Min, 100],
            [3],
            DType.int64,
          );

          final res = isClose(a, b);
          // Previously (int64Min - 0).abs() overflowed to negative in 64-bit int and returned true!
          expect(res.getCell([0]), isFalse);
          expect(res.getCell([1]), isTrue);
          expect(res.getCell([2]), isTrue);
          expect(allClose(a, b), isFalse);

          // Also test non-contiguous out buffer with isClose
          final outFull = NDArray<Boolean>.zeros([3, 2], DType.boolean);
          final outCol = outFull.slice([Slice.all(), Index(1)]);
          expect(outCol.isContiguous, isFalse);
          isClose(a, b, out: outCol);
          expect(outCol.getCell([0]), isFalse);
          expect(outCol.getCell([1]), isTrue);
          expect(outCol.getCell([2]), isTrue);
        });
      },
    );

    test(
      '5. nanstd with axis == null respects offsetElements on sliced out buffer',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, double.nan, 3.0],
            [3],
            DType.float64,
          );
          final outBacking = NDArray<Float64>.fromList(
            [999.0, 888.0],
            [2, 1],
            DType.float64,
          );
          // Slice second row with shape [1] so offsetElements == 1
          final outView = outBacking.slice([Index(1), Slice.all()]);
          expect(outView.shape, equals([1]));

          nanstd(a, keepdims: true, out: outView);
          // Variance of [1.0, 3.0] is 1.0, std is 1.0.
          // First row must remain untouched (999.0), second row must be 1.0.
          expect(outBacking.getCell([0, 0]), equals(999.0));
          expect(outBacking.getCell([1, 0]), closeTo(math.sqrt(1.0), 1e-12));
        });
      },
    );

    test(
      '6. financial functions with out buffer do not leak temporary views',
      () {
        NDArray.scope(() {
          final rate = NDArray<Float64>.fromList(
            [0.05, 0.06],
            [2, 1],
            DType.float64,
          );
          final nper = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0],
            [1, 3],
            DType.float64,
          );
          final pmtVal = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
          final pvVal = NDArray<Float64>.scalar(-1000.0, dtype: DType.float64);
          final out = NDArray<Float64>.zeros([2, 3], DType.float64);

          final res = fv(rate, nper, pmtVal, pvVal, out: out);
          expect(identical(res, out), isTrue);
          expect(out.getCell([0, 0]), closeTo(2886.68, 1e-1));

          // Test npv with scalar rate, non-contiguous values, and out buffer
          final cfFull = NDArray<Float64>.fromList(
            [-100.0, 0.0, 60.0, 0.0, 60.0, 0.0],
            [1, 3, 2],
            DType.float64,
          );
          final cfSliced = cfFull.slice([Slice.all(), Slice.all(), Index(0)]);
          expect(cfSliced.isContiguous, isFalse);
          final scalarRate = NDArray<Float64>.scalar(0.1, dtype: DType.float64);
          final npvOut = NDArray<Float64>.zeros([1], DType.float64);
          npv(scalarRate, cfSliced, out: npvOut);
          expect(
            npvOut.getCell([0]),
            closeTo(-100.0 + 60.0 / 1.1 + 60.0 / 1.21, 1e-10),
          );
        });
      },
    );

    test('7. norm handles empty matrices for ord == 2 and ord == -2', () {
      NDArray.scope(() {
        final empty = NDArray<Float64>.zeros([0, 4], DType.float64);
        final n2 = norm(empty, ord: 2);
        final nm2 = norm(empty, ord: -2);
        expect(n2.scalar, equals(0.0));
        expect(nm2.scalar, equals(0.0));
      });
    });
  });
}
