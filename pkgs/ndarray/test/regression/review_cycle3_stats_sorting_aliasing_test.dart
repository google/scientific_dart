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
  group('Review Cycle 3: Stats, Cumulative, Diff & Sorting Aliasing', () {
    test('sum(a, axis: 0, out: a.slice([Index(0)])) produces [4.0, 6.0]', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final outSlice = a.slice([Index(0)]);
        final res = sum(a, axis: 0, out: outSlice);
        expect(res, same(outSlice));
        expect(res.toList(), equals([4.0, 6.0]));
        expect(a.toList(), equals([4.0, 6.0, 3.0, 4.0]));
      });
    });

    test('prod(a, axis: 0, out: a.slice([Index(0)])) with aliasing', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2.0, 3.0, 4.0, 5.0], [2, 2], DType.float64);
        final outSlice = a.slice([Index(0)]);
        final res = prod(a, axis: 0, out: outSlice);
        expect(res, same(outSlice));
        expect(res.toList(), equals([8.0, 15.0]));
      });
    });

    test('all and any with out aliasing slice of a', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [true, false, true, true],
          [2, 2],
          DType.boolean,
        );
        final outSlice = a.slice([Index(0)]);
        final resAll = all(a, axis: 0, out: outSlice);
        expect(resAll.toList(), equals([true, false]));

        final b = NDArray.fromList(
          [false, false, true, false],
          [2, 2],
          DType.boolean,
        );
        final outSliceB = b.slice([Index(0)]);
        final resAny = any(b, axis: 0, out: outSliceB);
        expect(resAny.toList(), equals([true, false]));
      });
    });

    test('mean, std, variance with out aliasing slice of a', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 6.0], [2, 2], DType.float64);
        final outMean = a.slice([Index(0)]);
        mean(a, axis: 0, out: outMean);
        expect(outMean.toList(), equals([2.0, 4.0]));

        final b = NDArray.fromList([1.0, 2.0, 3.0, 6.0], [2, 2], DType.float64);
        final outVar = b.slice([Index(0)]);
        variance(b, axis: 0, out: outVar);
        expect(outVar.toList(), equals([1.0, 4.0]));

        final c = NDArray.fromList([1.0, 2.0, 3.0, 6.0], [2, 2], DType.float64);
        final outStd = c.slice([Index(0)]);
        std(c, axis: 0, out: outStd);
        expect(outStd.toList(), equals([1.0, 2.0]));
      });
    });

    test('min, max, ptp with out aliasing slice of a', () {
      NDArray.scope(() {
        final a = NDArray.fromList([5.0, 2.0, 1.0, 8.0], [2, 2], DType.float64);
        final outMin = a.slice([Index(0)]);
        min(a, axis: 0, out: outMin);
        expect(outMin.toList(), equals([1.0, 2.0]));

        final b = NDArray.fromList([5.0, 2.0, 1.0, 8.0], [2, 2], DType.float64);
        final outMax = b.slice([Index(0)]);
        max(b, axis: 0, out: outMax);
        expect(outMax.toList(), equals([5.0, 8.0]));

        final c = NDArray.fromList([5.0, 2.0, 1.0, 8.0], [2, 2], DType.float64);
        final outPtp = c.slice([Index(0)]);
        ptp(c, axis: 0, out: outPtp);
        expect(outPtp.toList(), equals([4.0, 6.0]));
      });
    });

    test(
      'nansum, nanmean, nanvar, nanstd, nanmin, nanmax with out aliasing slice of a',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, double.nan, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outNanSum = a.slice([Index(0)]);
          nansum(a, axis: 0, out: outNanSum);
          expect(outNanSum.toList(), equals([4.0, 4.0]));

          final b = NDArray.fromList(
            [1.0, double.nan, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outNanMean = b.slice([Index(0)]);
          nanmean(b, axis: 0, out: outNanMean);
          expect(outNanMean.toList(), equals([2.0, 4.0]));

          final c = NDArray.fromList(
            [1.0, double.nan, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outNanVar = c.slice([Index(0)]);
          nanvar(c, axis: 0, out: outNanVar);
          expect(outNanVar.toList(), equals([1.0, 0.0]));

          final d = NDArray.fromList(
            [1.0, double.nan, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outNanStd = d.slice([Index(0)]);
          nanstd(d, axis: 0, out: outNanStd);
          expect(outNanStd.toList(), equals([1.0, 0.0]));

          final e = NDArray.fromList(
            [5.0, double.nan, 1.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outNanMin = e.slice([Index(0)]);
          nanmin(e, axis: 0, out: outNanMin);
          expect(outNanMin.toList(), equals([1.0, 4.0]));

          final f = NDArray.fromList(
            [5.0, double.nan, 1.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outNanMax = f.slice([Index(0)]);
          nanmax(f, axis: 0, out: outNanMax);
          expect(outNanMax.toList(), equals([5.0, 4.0]));
        });
      },
    );

    test(
      'median, quantile, average with out aliasing slice of a or weights',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 10.0, 3.0, 20.0],
            [2, 2],
            DType.float64,
          );
          final outMedian = a.slice([Index(0)]);
          median(a, axis: 0, out: outMedian);
          expect(outMedian.toList(), equals([2.0, 15.0]));

          final b = NDArray.fromList(
            [1.0, 10.0, 3.0, 20.0],
            [2, 2],
            DType.float64,
          );
          final outQuantile = b.slice([Index(0)]);
          quantile(b, 0.5, axis: 0, out: outQuantile);
          expect(outQuantile.toList(), equals([2.0, 15.0]));

          final c = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final weights = NDArray.fromList(
            [1.0, 3.0, 3.0, 1.0],
            [2, 2],
            DType.float64,
          );
          // out aliases weights slice
          final outAvg = weights.slice([Index(0)]);
          final res = average(
            c,
            axis: 0,
            weights: weights,
            returned: true,
            out: outAvg,
          );
          expect(res.average, same(outAvg));
          expect(res.average.toList(), equals([2.5, 2.5]));
          expect(res.sumOfWeights?.toList(), equals([4.0, 4.0]));
        });
      },
    );

    test('cumsum(flip(a), out: a) on [1, 2, 3, 4] produces [4, 7, 9, 10]', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final res = cumsum(flip(a), out: a);
        expect(res, same(a));
        expect(a.toList(), equals([4.0, 7.0, 9.0, 10.0]));
      });
    });

    test('cumprod, cummin, cummax with reversed/aliased views', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        cumprod(flip(a), out: a);
        expect(a.toList(), equals([4.0, 12.0, 24.0, 24.0]));

        final b = NDArray.fromList([3.0, 1.0, 4.0, 2.0], [4], DType.float64);
        cummin(flip(b), out: b);
        expect(b.toList(), equals([2.0, 2.0, 1.0, 1.0]));

        final c = NDArray.fromList([3.0, 1.0, 4.0, 2.0], [4], DType.float64);
        cummax(flip(c), out: c);
        expect(c.toList(), equals([2.0, 4.0, 4.0, 4.0]));
      });
    });

    test(
      'diff(a, out: a.slice([Slice(start: 1)])) on [10, 25, 30, 50] produces [15, 5, 20] in a[1..3]',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 25.0, 30.0, 50.0],
            [4],
            DType.float64,
          );
          final outSlice = a.slice([Slice(start: 1)]);
          final res = diff(a, out: outSlice);
          expect(res, same(outSlice));
          expect(outSlice.toList(), equals([15.0, 5.0, 20.0]));
          expect(a.toList(), equals([10.0, 15.0, 5.0, 20.0]));
        });
      },
    );

    test('count_nonzero and argmax with out aliasing flipped slice of a', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1, 2, 0, 3, 0], [2, 3], DType.int64);
        final outSlice = flip(a.slice([Index(0)]));
        final resCount = count_nonzero(a, axis: 0, out: outSlice);
        expect(resCount, same(outSlice));
        // Column 0: [0, 0] -> 0 nonzeros
        // Column 1: [1, 3] -> 2 nonzeros
        // Column 2: [2, 0] -> 1 nonzero
        // Since outSlice is flipped row 0, writing [0, 2, 1] into outSlice means:
        // outSlice[0] = 0 (which is a[0, 2]), outSlice[1] = 2 (a[0, 1]), outSlice[2] = 1 (a[0, 0])
        expect(outSlice.toList(), equals([0, 2, 1]));
        expect(a.slice([Index(0)]).toList(), equals([1, 2, 0]));

        final b = NDArray.fromList(
          [10, 50, 20, 40, 30, 60],
          [2, 3],
          DType.int64,
        );
        final outArgmax = flip(b.slice([Index(0)]));
        final resArgmax = argmax(b, axis: 0, out: outArgmax);
        expect(resArgmax, same(outArgmax));
        // Column 0: [10, 40] -> argmax 1
        // Column 1: [50, 30] -> argmax 0
        // Column 2: [20, 60] -> argmax 1
        expect(outArgmax.toList(), equals([1, 0, 1]));

        final c = NDArray.fromList(
          [10, 50, 20, 40, 30, 60],
          [2, 3],
          DType.int64,
        );
        final outArgmin = flip(c.slice([Index(0)]));
        final resArgmin = argmin(c, axis: 0, out: outArgmin);
        expect(resArgmin, same(outArgmin));
        // Column 0: [10, 40] -> argmin 0
        // Column 1: [50, 30] -> argmin 1
        // Column 2: [20, 60] -> argmin 0
        expect(outArgmin.toList(), equals([0, 1, 0]));
      });
    });
  });
}
