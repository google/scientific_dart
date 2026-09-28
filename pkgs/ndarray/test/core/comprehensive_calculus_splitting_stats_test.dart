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

import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Comprehensive Calculus, Splitting & Stats Suite', () {
    test('Multi-dimensional splitting (split, array_split, hsplit, vsplit, dsplit)', () {
      NDArray.scope(() {
        final a2d = NDArray.fromList(List.generate(24, (i) => i.toDouble()), [
          4,
          6,
        ], DType.float64);

        final hsplits = hsplit(a2d, 3);
        expect(hsplits.length, 3);
        expect(hsplits[0].shape, [4, 2]);

        final vsplits = vsplit(a2d, 2);
        expect(vsplits.length, 2);
        expect(vsplits[0].shape, [2, 6]);

        final splitSections = split(a2d, 2, axis: 0);
        expect(splitSections.length, 2);

        final arrSplits = array_split(a2d, 5, axis: 1);
        expect(arrSplits.length, 5);

        // 3D dsplit
        final a3d = NDArray.fromList(List.generate(24, (i) => i.toDouble()), [
          2,
          3,
          4,
        ], DType.float64);

        final dsplits = dsplit(a3d, 2);
        expect(dsplits.length, 2);
        expect(dsplits[0].shape, [2, 3, 2]);

        final split3dIdx = split(a3d, 2, axis: 2);
        expect(split3dIdx.length, 2);
      });
    });

    test('Calculus (gradient, gradientArray, diff, trapz)', () {
      NDArray.scope(() {
        final a3d = NDArray.fromList(
          List.generate(24, (i) => (i * i).toDouble()),
          [2, 3, 4],
          DType.float64,
        );

        final g0 = gradient(a3d, axis: 0, spacing: const Spacing.step(2.0));
        expect(g0.shape, [2, 3, 4]);

        final g1 = gradient(a3d, axis: 1);
        expect(g1.shape, [2, 3, 4]);

        final g2 = gradient(a3d, axis: 2);
        expect(g2.shape, [2, 3, 4]);

        final gAll = gradientArray(a3d);
        expect(gAll.length, 3);
        expect(gAll[0].shape, [2, 3, 4]);
        expect(gAll[1].shape, [2, 3, 4]);
        expect(gAll[2].shape, [2, 3, 4]);

        final d1 = diff(a3d, n: 1, axis: 0);
        expect(d1.shape, [1, 3, 4]);

        final d2 = diff(a3d, n: 2, axis: 2);
        expect(d2.shape, [2, 3, 2]);

        final t0 = trapz(a3d, axis: 0, spacing: const Spacing.step(0.5));
        expect(t0.shape, [3, 4]);

        final t1 = trapz(a3d, axis: 1);
        expect(t1.shape, [2, 4]);

        final t2 = trapz(a3d, axis: 2);
        expect(t2.shape, [2, 3]);
      });
    });

    test(
      'Histogram, Binning & Digitalization (histogram, bincount, digitize)',
      () {
        NDArray.scope(() {
          final data1d = NDArray.fromList(
            [0.5, 1.2, 2.1, 3.8, 4.5, 2.3, 1.9, 0.1],
            [8],
            DType.float64,
          );
          final weights = NDArray.fromList(
            [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            [8],
            DType.float64,
          );

          final h1 = histogram(data1d, bins: 4);
          expect(h1.hist.shape, [4]);
          expect(h1.binEdges.shape, [5]);

          final h1w = histogram(
            data1d,
            bins: 4,
            weights: weights,
            density: true,
          );
          expect(h1w.hist.shape, [4]);

          final intData = NDArray.fromList(
            [1, 2, 1, 3, 2, 2, 4, 0],
            [8],
            DType.int64,
          );
          final bc = bincount(intData);
          expect(bc.shape, [5]);

          final bcw = bincount(intData, weights: weights);
          expect(bcw.shape, [5]);

          final bins = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          final dig = digitize(data1d, bins, right: true);
          expect(dig.shape, [8]);
        });
      },
    );
  });
}
