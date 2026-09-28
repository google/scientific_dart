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
  group('Bincount Tests', () {
    test('Basic contiguous unweighted (int64)', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 3, 2, 1, 7], [7], DType.int64);
        final counts = bincount(x);
        expect(counts.dtype, DType.int64);
        expect(counts.shape, [8]);
        expect(counts.toList(), [1, 3, 1, 1, 0, 0, 0, 1]);
      });
    });

    test('Basic contiguous unweighted (int32)', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 3, 2, 1, 7], [7], DType.int32);
        final out = NDArray.zeros([8], DType.int32);
        final counts = bincount(x, out: out);
        expect(counts.dtype, DType.int32);
        expect(counts.toList(), [1, 3, 1, 1, 0, 0, 0, 1]);
      });
    });

    test('Minlength parameter', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 2], [4], DType.int64);
        final counts = bincount(x, minlength: 5);
        expect(counts.shape, [5]);
        expect(counts.toList(), [1, 2, 1, 0, 0]);

        final counts2 = bincount(x, minlength: 2);
        expect(counts2.shape, [3]);
        expect(counts2.toList(), [1, 2, 1]);
      });
    });

    test('Strided unweighted', () {
      NDArray.scope(() {
        final xFull = NDArray.fromList(
          [0, 99, 1, 99, 1, 99, 2],
          [7],
          DType.int64,
        );
        final x = xFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        expect(x.isContiguous, false);
        final counts = bincount(x);
        expect(counts.toList(), [1, 2, 1]);
      });
    });

    test('Weighted contiguous (double)', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 2], [4], DType.int64);
        final weights = NDArray.fromList(
          [0.5, 1.0, 2.0, 1.5],
          [4],
          DType.float64,
        );
        final counts = bincount(x, weights: weights);
        expect(counts.dtype, DType.float64);
        expect(counts.toList(), [0.5, 3.0, 1.5]);
      });
    });

    test('Weighted contiguous (float32)', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 2], [4], DType.int64);
        final weights = NDArray.fromList(
          [0.5, 1.0, 2.0, 1.5],
          [4],
          DType.float32,
        );
        final counts = bincount(x, weights: weights);
        expect(counts.dtype, DType.float32);
        final list = counts
            .toList()
            .map((e) => double.parse(e.toStringAsFixed(1)))
            .toList();
        expect(list, [0.5, 3.0, 1.5]);
      });
    });

    test('Weighted strided', () {
      NDArray.scope(() {
        final xFull = NDArray.fromList(
          [0, 99, 1, 99, 1, 99, 2],
          [7],
          DType.int64,
        );
        final x = xFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final wFull = NDArray.fromList(
          [0.5, 99.0, 1.0, 99.0, 2.0, 99.0, 1.5],
          [7],
          DType.float64,
        );
        final weights = wFull.slice([Slice(start: 0, stop: 7, step: 2)]);

        final counts = bincount(x, weights: weights);
        expect(counts.toList(), [0.5, 3.0, 1.5]);
      });
    });

    test('Empty input', () {
      NDArray.scope(() {
        final x = NDArray.zeros([0], DType.int64);
        final counts = bincount(x);
        expect(counts.shape, [0]);
        expect(counts.toList(), <int>[]);

        final counts2 = bincount(x, minlength: 3);
        expect(counts2.shape, [3]);
        expect(counts2.toList(), [0, 0, 0]);
      });
    });

    test('Validation errors', () {
      NDArray.scope(() {
        final xNeg = NDArray.fromList([0, -1, 2], [3], DType.int64);
        expect(() => bincount(xNeg), throwsArgumentError);

        final x2D = NDArray.fromList([0, 1, 2, 3], [2, 2], DType.int64);
        expect(() => bincount(x2D), throwsArgumentError);

        final x = NDArray.fromList([0, 1, 2], [3], DType.int64);
        final wBad = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => bincount(x, weights: wBad), throwsArgumentError);
      });
    });

    test('int32 strided unweighted', () {
      NDArray.scope(() {
        final xFull = NDArray.fromList(
          [0, 99, 1, 99, 1, 99, 2],
          [7],
          DType.int32,
        );
        final x = xFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final out = NDArray.zeros([3], DType.int32);
        final counts = bincount(x, out: out);
        expect(counts.toList(), [1, 2, 1]);
      });
    });

    test('int32 x float64 contiguous weighted', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 2], [4], DType.int32);
        final weights = NDArray.fromList(
          [0.5, 1.0, 2.0, 1.5],
          [4],
          DType.float64,
        );
        final counts = bincount(x, weights: weights);
        expect(counts.dtype, DType.float64);
        expect(counts.toList(), [0.5, 3.0, 1.5]);
      });
    });

    test('int32 x float64 strided weighted', () {
      NDArray.scope(() {
        final xFull = NDArray.fromList(
          [0, 99, 1, 99, 1, 99, 2],
          [7],
          DType.int32,
        );
        final x = xFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final wFull = NDArray.fromList(
          [0.5, 99.0, 1.0, 99.0, 2.0, 99.0, 1.5],
          [7],
          DType.float64,
        );
        final weights = wFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final counts = bincount(x, weights: weights);
        expect(counts.toList(), [0.5, 3.0, 1.5]);
      });
    });

    test('int32 x float32 contiguous weighted', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 2], [4], DType.int32);
        final weights = NDArray.fromList(
          [0.5, 1.0, 2.0, 1.5],
          [4],
          DType.float32,
        );
        final counts = bincount(x, weights: weights);
        expect(counts.dtype, DType.float32);
        final list = counts
            .toList()
            .map((e) => double.parse(e.toStringAsFixed(1)))
            .toList();
        expect(list, [0.5, 3.0, 1.5]);
      });
    });

    test('int32 x float32 strided weighted', () {
      NDArray.scope(() {
        final xFull = NDArray.fromList(
          [0, 99, 1, 99, 1, 99, 2],
          [7],
          DType.int32,
        );
        final x = xFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final wFull = NDArray.fromList(
          [0.5, 99.0, 1.0, 99.0, 2.0, 99.0, 1.5],
          [7],
          DType.float32,
        );
        final weights = wFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final counts = bincount(x, weights: weights);
        expect(counts.dtype, DType.float32);
        final list = counts
            .toList()
            .map((e) => double.parse(e.toStringAsFixed(1)))
            .toList();
        expect(list, [0.5, 3.0, 1.5]);
      });
    });

    test('int64 x float32 strided weighted', () {
      NDArray.scope(() {
        final xFull = NDArray.fromList(
          [0, 99, 1, 99, 1, 99, 2],
          [7],
          DType.int64,
        );
        final x = xFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final wFull = NDArray.fromList(
          [0.5, 99.0, 1.0, 99.0, 2.0, 99.0, 1.5],
          [7],
          DType.float32,
        );
        final weights = wFull.slice([Slice(start: 0, stop: 7, step: 2)]);
        final counts = bincount(x, weights: weights);
        expect(counts.dtype, DType.float32);
        final list = counts
            .toList()
            .map((e) => double.parse(e.toStringAsFixed(1)))
            .toList();
        expect(list, [0.5, 3.0, 1.5]);
      });
    });

    test('Disposed arrays throw StateError', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1, 2], [2], DType.int64);
        x.dispose();
        expect(() => bincount(x), throwsStateError);

        final x2 = NDArray.fromList([1, 2], [2], DType.int64);
        final out = NDArray.zeros([3], DType.int64);
        out.dispose();
        expect(() => bincount(x2, out: out), throwsStateError);

        final x3 = NDArray.fromList([1, 2], [2], DType.int64);
        final weights = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        weights.dispose();
        expect(() => bincount(x3, weights: weights), throwsStateError);
      });
    });

    test('Validation errors extra', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1, 2], [2], DType.int64);
        expect(() => bincount(x, minlength: -1), throwsArgumentError);

        final outBadShape = NDArray.zeros([3, 1], DType.int64);
        expect(() => bincount(x, out: outBadShape), throwsArgumentError);

        final outBadSize = NDArray.zeros([2], DType.int64);
        expect(() => bincount(x, out: outBadSize), throwsArgumentError);
      });
    });

    test('Empty input with out parameter', () {
      NDArray.scope(() {
        final x = NDArray.zeros([0], DType.int64);
        final out = NDArray.fromList([1, 2, 3], [3], DType.int64);
        final counts = bincount(x, out: out);
        expect(counts.toList(), [0, 0, 0]);
      });
    });

    test('Non-standard and mismatched dtypes', () {
      NDArray.scope(() {
        final xUint8 = NDArray.fromList([0, 1, 1, 2], [4], DType.uint8);
        final counts = bincount(xUint8);
        expect(counts.dtype, DType.int64);
        expect(counts.toList(), [1, 2, 1]);

        final xInt32 = NDArray.fromList([0, 1, 1, 2], [4], DType.int32);
        final outInt64 = NDArray.zeros([3], DType.int64);
        final counts2 = bincount(xInt32, out: outInt64);
        expect(counts2.dtype, DType.int64);
        expect(counts2.toList(), [1, 2, 1]);

        final xInt64 = NDArray.fromList([0, 1, 1, 2], [4], DType.int64);
        final outInt32 = NDArray.zeros([3], DType.int32);
        final counts3 = bincount(xInt64, out: outInt32);
        expect(counts3.dtype, DType.int32);
        expect(counts3.toList(), [1, 2, 1]);

        final weights = NDArray.fromList(
          [0.5, 1.0, 2.0, 1.5],
          [4],
          DType.float64,
        );
        final counts4 = bincount(xUint8, weights: weights);
        expect(counts4.dtype, DType.float64);
        expect(counts4.toList(), [0.5, 3.0, 1.5]);

        final weightsInt32 = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final counts5 = bincount(xInt64, weights: weightsInt32);
        expect(counts5.dtype, DType.int32);
        expect(counts5.toList(), [1, 5, 4]);

        final weightsFloat32 = NDArray.fromList(
          [0.5, 1.0, 2.0, 1.5],
          [4],
          DType.float32,
        );
        final outFloat64 = NDArray.zeros([3], DType.float64);
        final counts6 = bincount(
          xInt64,
          weights: weightsFloat32,
          out: outFloat64,
        );
        expect(counts6.dtype, DType.float64);
        expect(counts6.toList(), [0.5, 3.0, 1.5]);
      });
    });

    test('Weighted with out parameter (matching dtype)', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 1, 2], [4], DType.int64);
        final weights = NDArray.fromList(
          [0.5, 1.0, 2.0, 1.5],
          [4],
          DType.float64,
        );
        final out = NDArray.zeros([3], DType.float64);
        final counts = bincount(x, weights: weights, out: out);
        expect(identical(counts, out), true);
        expect(out.toList(), [0.5, 3.0, 1.5]);
      });
    });
  });

  group('Digitize Tests', () {
    test('Increasing bins, right = false', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.2, 6.4, 3.0, 1.6, -1.0, 11.0],
          [6],
          DType.float64,
        );
        final bins = NDArray.fromList(
          [0.0, 1.0, 2.5, 4.0, 10.0],
          [5],
          DType.float64,
        );
        final inds = digitize(x, bins, right: false);
        expect(inds.toList(), [1, 4, 3, 2, 0, 5]);
      });
    });

    test('Increasing bins, right = true', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.0, 1.0, 2.5, 4.0, 10.0, -1.0, 11.0],
          [7],
          DType.float64,
        );
        final bins = NDArray.fromList(
          [0.0, 1.0, 2.5, 4.0, 10.0],
          [5],
          DType.float64,
        );
        final inds = digitize(x, bins, right: true);
        expect(inds.toList(), [0, 1, 2, 3, 4, 0, 5]);
      });
    });

    test('Decreasing bins, right = false', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [11.0, 8.0, 5.0, 2.0, 0.0],
          [5],
          DType.float64,
        );
        final bins = NDArray.fromList([10.0, 5.0, 1.0], [3], DType.float64);
        final inds = digitize(x, bins, right: false);
        expect(inds.toList(), [0, 1, 1, 2, 3]);
      });
    });

    test('Decreasing bins, right = true', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [11.0, 10.0, 8.0, 5.0, 2.0, 1.0, 0.0],
          [7],
          DType.float64,
        );
        final bins = NDArray.fromList([10.0, 5.0, 1.0], [3], DType.float64);
        final inds = digitize(x, bins, right: true);
        expect(inds.toList(), [0, 1, 1, 2, 2, 3, 3]);
      });
    });

    test('Validation errors', () {
      NDArray.scope(() {
        final binsBad = NDArray.fromList([0.0, 2.0, 1.0], [3], DType.float64);
        final x = NDArray.fromList([1.0], [1], DType.float64);
        expect(() => digitize(x, binsBad), throwsArgumentError);

        final bins2D = NDArray.fromList(
          [0.0, 1.0, 2.0, 3.0],
          [2, 2],
          DType.float64,
        );
        expect(() => digitize(x, bins2D), throwsArgumentError);

        final binsEmpty = NDArray.zeros([0], DType.float64);
        expect(() => digitize(x, binsEmpty), throwsArgumentError);
      });
    });

    test('Disposed arrays throw StateError', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0], [1], DType.float64);
        final bins = NDArray.fromList([0.0, 2.0], [2], DType.float64);
        x.dispose();
        expect(() => digitize(x, bins), throwsStateError);

        final x2 = NDArray.fromList([1.0], [1], DType.float64);
        final bins2 = NDArray.fromList([0.0, 2.0], [2], DType.float64);
        bins2.dispose();
        expect(() => digitize(x2, bins2), throwsStateError);
      });
    });
  });

  group('Histogram Tests', () {
    test('Uniform bins (int)', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 2.0, 1.0], [3], DType.float64);
        final (:hist, :binEdges) = histogram(x, bins: 2);
        expect(hist.dtype, DType.int64);
        expect(hist.toList(), [2, 1]);
        expect(binEdges.toList(), [1.0, 1.5, 2.0]);
      });
    });

    test('Custom bin edges (NDArray)', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.5, 0.5, 1.5, 2.5, 3.5],
          [5],
          DType.float64,
        );
        final bins = NDArray.fromList(
          [0.0, 1.0, 2.0, 3.0, 4.0],
          [5],
          DType.float64,
        );
        final (:hist, :binEdges) = histogram(x, bins: bins);
        expect(hist.toList(), [2, 1, 1, 1]);
        expect(binEdges.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);
      });
    });

    test('Range parameter', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final (:hist, :binEdges) = histogram(x, bins: 2, range: (1.5, 3.5));
        expect(hist.toList(), [1, 1]);
        expect(binEdges.toList(), [1.5, 2.5, 3.5]);
      });
    });

    test('Density normalization', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 2.0, 1.0], [3], DType.float64);
        final (:hist, :binEdges) = histogram(x, bins: 2, density: true);
        expect(hist.dtype, DType.float64);
        final histList = hist
            .toList()
            .map((e) => double.parse(e.toStringAsFixed(4)))
            .toList();
        expect(histList, [1.3333, 0.6667]);
      });
    });

    test('Weights parameter', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 2.0, 1.0], [3], DType.float64);
        final weights = NDArray.fromList([0.5, 1.0, 2.0], [3], DType.float64);
        final (:hist, :binEdges) = histogram(x, bins: 2, weights: weights);
        expect(hist.dtype, DType.float64);
        expect(hist.toList(), [2.5, 1.0]);
      });
    });

    test('Empty input', () {
      NDArray.scope(() {
        final x = NDArray.zeros([0], DType.float64);
        final (:hist, :binEdges) = histogram(x, bins: 3);
        expect(hist.toList(), [0, 0, 0]);
        expect(binEdges.toList(), [
          0.0,
          0.3333333333333333,
          0.6666666666666666,
          1.0,
        ]);
      });
    });

    test('Constant input', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 1.0, 1.0], [3], DType.float64);
        final (:hist, :binEdges) = histogram(x, bins: 2);
        expect(binEdges.toList(), [0.5, 1.0, 1.5]);
        expect(hist.toList(), [0, 3]);
      });
    });

    test('NDArray int32 bins', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0.5, 1.5, 2.5], [3], DType.float64);
        final bins = NDArray.fromList([0, 1, 2, 3], [4], DType.int32);
        final (:hist, :binEdges) = histogram(x, bins: bins);
        expect(hist.toList(), [1, 1, 1]);
        expect(binEdges.toList(), [0.0, 1.0, 2.0, 3.0]);
      });
    });

    test('NDArray float32 bins', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0.5, 1.5, 2.5], [3], DType.float64);
        final bins = NDArray.fromList([0.0, 1.0, 2.0, 3.0], [4], DType.float32);
        final (:hist, :binEdges) = histogram(x, bins: bins);
        expect(hist.toList(), [1, 1, 1]);
        expect(binEdges.dtype, DType.float64);
        expect(binEdges.toList(), [0.0, 1.0, 2.0, 3.0]);
      });
    });

    test('Invalid bins type', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0], [1], DType.float64);
        expect(() => histogram(x, bins: 'invalid'), throwsArgumentError);
      });
    });

    test('Weights size mismatch', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final weights = NDArray.fromList([1.0], [1], DType.float64);
        expect(
          () => histogram(x, bins: 2, weights: weights),
          throwsArgumentError,
        );
      });
    });

    test('Disposed arrays throw StateError', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0], [1], DType.float64);
        x.dispose();
        expect(() => histogram(x), throwsStateError);

        final x2 = NDArray.fromList([1.0], [1], DType.float64);
        final weights = NDArray.fromList([1.0], [1], DType.float64);
        weights.dispose();
        expect(() => histogram(x2, weights: weights), throwsStateError);
      });
    });

    test('Validation errors extra', () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0], [1], DType.float64);
        expect(() => histogram(x, bins: -1), throwsArgumentError);

        final bins2D = NDArray.fromList(
          [0.0, 1.0, 2.0, 3.0],
          [2, 2],
          DType.float64,
        );
        expect(() => histogram(x, bins: bins2D), throwsArgumentError);
      });
    });
  });
}
