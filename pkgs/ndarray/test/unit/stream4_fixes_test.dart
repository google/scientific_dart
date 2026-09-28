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
  group('Stream 4 Fixes Verification Tests', () {
    test(
      '1. Sorting/partitioning with targetAxis != rank - 1 and out != null',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [3.0, 1.0, 2.0, 6.0, 4.0, 5.0],
            [2, 3],
            DType.float64,
          );
          final outSort = NDArray<Float64>.zeros([2, 3], DType.float64);
          final resSort = sort(a, axis: 0, out: outSort);
          expect(identical(resSort, outSort), isTrue);
          expect(outSort.toList(), equals([3.0, 1.0, 2.0, 6.0, 4.0, 5.0]));

          final outArgSort = NDArray.zeros([2, 3], DType.int64);
          final resArgSort = argsort(a, axis: 0, out: outArgSort);
          expect(identical(resArgSort, outArgSort), isTrue);
          expect(outArgSort.toList(), equals([0, 0, 0, 1, 1, 1]));

          final outPart = NDArray<Float64>.zeros([2, 3], DType.float64);
          final resPart = partition(a, 0, axis: 0, out: outPart);
          expect(identical(resPart, outPart), isTrue);

          final outArgPart = NDArray.zeros([2, 3], DType.int64);
          final resArgPart = argpartition(a, 0, axis: 0, out: outArgPart);
          expect(identical(resArgPart, outArgPart), isTrue);

          // 0-D scalar sort with out
          final scalarA = NDArray.scalar(42.0, dtype: DType.float64);
          final outScalar = NDArray<Float64>.zeros([], DType.float64);
          sort(scalarA, out: outScalar);
          expect(outScalar.scalar, equals(42.0));

          final outScalarIdx = NDArray.zeros([], DType.int64);
          argsort(scalarA, out: outScalarIdx);
          expect(outScalarIdx.scalar, equals(0));
        });
      },
    );

    test('2. Random multinomial strided pvals and choice empty array', () {
      NDArray.scope(() {
        final pBase = NDArray.fromList(
          [0.2, 99.0, 0.3, 99.0, 0.5],
          [5],
          DType.float64,
        );
        final pvals = pBase.slice([Slice(step: 2)]); // [0.2, 0.3, 0.5]
        expect(pvals.isContiguous, isFalse);

        final multiRes = multinomial(100, pvals, size: [2]);
        expect(multiRes.shape, equals([2, 3]));
        final row0Sum =
            multiRes.getCell([0, 0]) +
            multiRes.getCell([0, 1]) +
            multiRes.getCell([0, 2]);
        expect(row0Sum, equals(100));

        // choice empty array
        final emptyA = NDArray.fromList(<double>[], [0], DType.float64);
        final c0 = choice(emptyA, size: [0]);
        expect(c0.shape, equals([0]));

        expect(() => choice(emptyA, size: [2]), throwsArgumentError);

        // shuffle on 1-D and 2-D
        final arr1D = NDArray.fromList([1, 2, 3, 4, 5], [5], DType.int32);
        shuffle(arr1D, seed: 42);
        expect(arr1D.size, equals(5));

        final arr2D = NDArray.fromList([1, 2, 3, 4, 5, 6], [3, 2], DType.int32);
        shuffle(arr2D, seed: 42);
        expect(arr2D.shape, equals([3, 2]));
      });
    });

    test('3. Stats _reductionTargetShape axis validation', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      expect(() => mean(a, axis: 5), throwsRangeError);
      expect(() => mean(a, axis: -5), throwsRangeError);
      expect(() => variance(a, axis: 5), throwsRangeError);
    });

    test('4. Optimize brentq with NaN bracket evaluation', () {
      expect(() => brentq((x) => double.nan, 0.0, 1.0), throwsArgumentError);
    });

    test('5. Manipulation concatenate error message contains axis info', () {
      final a = NDArray.fromList([1, 2], [1, 2], DType.int32);
      final b = NDArray.fromList([3, 4, 5], [1, 3], DType.int32);
      expect(
        () => concatenate([a, b], axis: 0),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('dimension 0'),
          ),
        ),
      );
    });

    test(
      '6. Spacers logspace and geomspace with numSamples == 0 and named records',
      () {
        NDArray.scope(() {
          final ls0 = logspace(1.0, 3.0, 0, dtype: DType.float64);
          expect(ls0.shape, equals([0]));
          expect(ls0.toList(), equals([]));

          final gs0 = geomspace(1.0, 100.0, 0, dtype: DType.float64);
          expect(gs0.shape, equals([0]));
          expect(gs0.toList(), equals([]));

          final (:samples, :step) = linspaceWithStep(
            0.0,
            10.0,
            5,
            dtype: DType.float64,
          );
          expect(samples.toList(), equals([0.0, 2.5, 5.0, 7.5, 10.0]));
          expect(step, equals(2.5));

          final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stop = NDArray.fromList([1.0, 12.0], [2], DType.float64);
          final (samples: gridSamples, step: gridStep) = linspaceGridWithStep(
            start,
            stop,
            3,
          );
          expect(gridSamples.shape, equals([3, 2]));
          expect(gridStep.toList(), equals([0.5, 1.0]));
        });
      },
    );

    test('7. Financial PaymentDue strongly-typed enum and doc contracts', () {
      NDArray.scope(() {
        final rate = NDArray.fromList([0.05 / 12], [1], DType.float64);
        final nper = NDArray.fromList([120.0], [1], DType.float64);
        final pmt = NDArray.fromList([-100.0], [1], DType.float64);
        final pvVal = NDArray.fromList([-100.0], [1], DType.float64);

        final resEnd = fv(rate, nper, pmt, pvVal, when: PaymentDue.end);
        final resBegin = fv(rate, nper, pmt, pvVal, when: PaymentDue.begin);
        expect(resBegin.getCell([0]), greaterThan(resEnd.getCell([0])));
      });
    });

    test('8. Binning histogram returns named record', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1, 2, 1], [3], DType.int32);
        final (:hist, :binEdges) = histogram(x, bins: 2);
        expect(hist.shape, equals([2]));
        expect(binEdges.shape, equals([3]));
      });
    });

    test('9. Shaping meshes and interpolation use NDArray<Float64>', () {
      NDArray.scope(() {
        final og = ogrid([GridRange(0.0, 2.0), GridRange(0.0, 3.0)]);
        expect(og, isA<List<NDArray<Float64>>>());
        expect(og.length, equals(2));

        final mg = mgrid([GridRange(0.0, 2.0), GridRange(0.0, 3.0)]);
        expect(mg, isA<NDArray<Float64>>());
        expect(mg.shape, equals([2, 2, 3]));

        final x = NDArray.fromList([2.5], [1], DType.float64);
        final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final fp = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
        final interpRes = interp(x, xp, fp);
        expect(interpRes, isA<NDArray<Float64>>());
        expect(interpRes.getCell([0]), equals(25.0));
      });
    });
    test(
      '10. Spacers linspace, logspace, geomspace out: parameter support',
      () {
        NDArray.scope(() {
          final outLin = NDArray<Float64>.zeros([5], DType.float64);
          final resLin = linspace(
            0.0,
            10.0,
            5,
            dtype: DType.float64,
            out: outLin,
          );
          expect(identical(resLin, outLin), isTrue);
          expect(outLin.toList(), equals([0.0, 2.5, 5.0, 7.5, 10.0]));

          final outLog = NDArray<Float64>.zeros([3], DType.float64);
          final resLog = logspace(
            0.0,
            2.0,
            3,
            dtype: DType.float64,
            out: outLog,
          );
          expect(identical(resLog, outLog), isTrue);
          expect(outLog.toList(), equals([1.0, 10.0, 100.0]));

          final outGeom = NDArray<Float64>.zeros([3], DType.float64);
          final resGeom = geomspace(
            1.0,
            100.0,
            3,
            dtype: DType.float64,
            out: outGeom,
          );
          expect(identical(resGeom, outGeom), isTrue);
          expect(outGeom.toList(), equals([1.0, 10.0, 100.0]));
        });
      },
    );
  });
}
