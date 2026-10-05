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
  group('Stream 4 Cycle 2 Unit Tests', () {
    test('diff: out buffer validation when n >= a.shape[targetAxis]', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

        // n >= 3 produces empty shape [0]
        final validOut = NDArray.create([0], DType.float64);
        final res = diff(a, n: 3, out: validOut);
        expect(res.shape, equals([0]));
        expect(identical(res, validOut), isTrue);

        // Mismatched shape throws ArgumentError
        final invalidShapeOut = NDArray.create([1], DType.float64);
        expect(() => diff(a, n: 3, out: invalidShapeOut), throwsArgumentError);

        // Mismatched dtype throws ArgumentError
        final invalidDtypeOut = NDArray.create([0], DType.int32);
        expect(
          () => diff(a, n: 3, out: invalidDtypeOut as dynamic),
          throwsArgumentError,
        );
      });
    });

    test('exponential: non-contiguous out buffer succeeds', () {
      NDArray.scope(() {
        final base = NDArray.zeros([4, 4], DType.float64);
        // Transposed view is non-contiguous
        final nonContig = base.transpose([1, 0]);
        expect(nonContig.isContiguous, isFalse);

        final res = exponential(nonContig.shape, scale: 1.0, out: nonContig);
        expect(identical(res, nonContig), isTrue);
        expect(nonContig.getCell([0, 0]), greaterThan(0.0));
      });
    });

    test('cov: empty array throws ArgumentError', () {
      NDArray.scope(() {
        final empty = NDArray.create([0], DType.float64);
        expect(() => cov(empty), throwsArgumentError);
      });
    });

    test('multinomial: seed reproducibility', () {
      NDArray.scope(() {
        final pvals = NDArray.fromList([0.2, 0.3, 0.5], [3], DType.float64);
        final res1 = multinomial(100, pvals, size: [5], seed: 42);
        final res2 = multinomial(100, pvals, size: [5], seed: 42);
        final res3 = multinomial(100, pvals, size: [5], seed: 99);

        expect(res1.toList(), equals(res2.toList()));
        // Different seed should differ
        expect(res1.toList(), isNot(equals(res3.toList())));
      });
    });

    test('distance: pdist cosine with typed pointers', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [3, 2],
          DType.float64,
        );

        final d = pdist(x, metric: DistanceMetric.cosine);
        expect(d.shape, equals([3]));
        // cosine dist (1,0) and (0,1) = 1.0
        expect(d.getCell([0]), closeTo(1.0, 1e-6));
        // cosine dist (1,0) and (1,1) = 1 - 1/sqrt(2) ~ 0.29289
        expect(d.getCell([1]), closeTo(1.0 - 1.0 / math.sqrt(2), 1e-5));
        // cosine dist (0,1) and (1,1) = 1 - 1/sqrt(2) ~ 0.29289
        expect(d.getCell([2]), closeTo(1.0 - 1.0 / math.sqrt(2), 1e-5));
      });
    });

    test(
      'interpolation: interp nearest method with strided non-contiguous inputs',
      () {
        NDArray.scope(() {
          final x = NDArray.fromList([0.5, 1.5, 2.5, 3.5], [4], DType.float64);
          final xp = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );
          final fp = NDArray.fromList(
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [5],
            DType.float64,
          );

          // Strided view
          final xStrided = x.transpose([0]);
          final res = interp(
            xStrided,
            xp,
            fp,
            method: InterpolationMethod.nearest,
          );
          expect(res.shape, equals([4]));
          expect(res.getCell([0]), equals(0.0));
          expect(res.getCell([1]), equals(10.0));
          expect(res.getCell([2]), equals(20.0));
          expect(res.getCell([3]), equals(30.0));
        });
      },
    );

    test(
      'sorting & search: boolean sort, argsort, searchsorted fallback paths',
      () {
        NDArray.scope(() {
          final b = NDArray.fromList(
            [true, false, true, false, false],
            [5],
            DType.boolean,
          );
          final sortedB = sort(b);
          expect(sortedB.toList(), equals([false, false, false, true, true]));

          final argsortB = argsort(b);
          expect(argsortB.shape, equals([5]));
          // Check that argsort indices index sorted elements
          for (var i = 0; i < 5; i++) {
            final idx = argsortB.getCell([i]);
            expect(b.getCell([idx]), equals(sortedB.getCell([i])));
          }

          final sortedArr = NDArray.fromList(
            [false, false, true, true],
            [4],
            DType.boolean,
          );
          final searchVals = NDArray.fromList(
            [false, true],
            [2],
            DType.boolean,
          );
          final searchRes = searchsorted(sortedArr, searchVals);
          expect(searchRes.getCell([0]), equals(0));
          expect(searchRes.getCell([1]), equals(2));
        });
      },
    );

    test(
      'stats: sum, prod, mean, variance, std scalar reductions across dtypes',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );

          expect(sum(a).scalar, equals(10.0));
          expect(prod(a).scalar, equals(24.0));
          expect(mean(a).scalar, equals(2.5));
          expect(variance(a).scalar, equals(1.25));
          expect(std(a).scalar, closeTo(math.sqrt(1.25), 1e-6));

          // keepdims: true scalar reduction
          final sKeep = sum(a, keepdims: true);
          expect(sKeep.shape, equals([1, 1]));
          expect(sKeep.getCell([0, 0]), equals(10.0));

          final mKeep = mean(a, keepdims: true);
          expect(mKeep.shape, equals([1, 1]));
          expect(mKeep.getCell([0, 0]), equals(2.5));
        });
      },
    );
    test('distance: chebyshev metric & NaN propagation', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [1.0, 2.0, 5.0, 0.0, 1.0, double.nan],
          [3, 2],
          DType.float64,
        );
        final d = pdist(x, metric: DistanceMetric.chebyshev);
        expect(d.shape, equals([3]));
        // max(|1-5|, |2-0|) = 4.0
        expect(d.getCell([0]), equals(4.0));
        // max(|1-1|, |2-NaN|) = NaN
        expect(d.getCell([1]).isNaN, isTrue);
        // max(|5-1|, |0-NaN|) = NaN
        expect(d.getCell([2]).isNaN, isTrue);

        // 0-dimensional space N=0
        final x0 = NDArray.create([2, 0], DType.float64);
        final d0 = pdist(x0, metric: DistanceMetric.chebyshev);
        expect(d0.getCell([0]), equals(0.0));
      });
    });

    test('interpolation: interp NaN propagation when xp_size == 1', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0.0, double.nan, 2.0], [3], DType.float64);
        final xp = NDArray.fromList([1.0], [1], DType.float64);
        final fp = NDArray.fromList([42.0], [1], DType.float64);

        final res = interp(x, xp, fp);
        expect(res.getCell([0]), equals(42.0));
        expect(res.getCell([1]).isNaN, isTrue);
        expect(res.getCell([2]), equals(42.0));
      });
    });

    test('polynomial: degree-0 constant polynomial NaN propagation', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [10.0, double.nan, 20.0],
          [3],
          DType.float64,
        );
        final c0 = NDArray.fromList([5.0], [1], DType.float64);

        final resPoly = polyval(c0, x);
        expect(resPoly.getCell([0]), equals(5.0));
        expect(resPoly.getCell([1]).isNaN, isTrue);
        expect(resPoly.getCell([2]), equals(5.0));

        final resCheb = chebval(x, c0);
        expect(resCheb.getCell([0]), equals(5.0));
        expect(resCheb.getCell([1]).isNaN, isTrue);
        expect(resCheb.getCell([2]), equals(5.0));
      });
    });

    test(
      'spacers, manipulation, random, shaping_meshes: optional out buffers and validation',
      () {
        NDArray.scope(() {
          // geomspaceGrid with out
          final start = NDArray.fromList([1.0], [1], DType.float64);
          final stop = NDArray.fromList([100.0], [1], DType.float64);
          final outGeom = NDArray.create([3, 1], DType.float64);
          final resGeom = geomspaceGrid(start, stop, 3, out: outGeom);
          expect(identical(resGeom, outGeom), isTrue);
          expect(resGeom.getCell([0, 0]), closeTo(1.0, 1e-6));
          expect(resGeom.getCell([2, 0]), closeTo(100.0, 1e-6));

          // vstack / hstack / copy with out
          final a = NDArray.fromList([1.0, 2.0], [1, 2], DType.float64);
          final b = NDArray.fromList([3.0, 4.0], [1, 2], DType.float64);
          final outVstack = NDArray.create([2, 2], DType.float64);
          final resV = vstack([a, b], out: outVstack);
          expect(identical(resV, outVstack), isTrue);

          final outHstack = NDArray.create([1, 4], DType.float64);
          final resH = hstack([a, b], out: outHstack);
          expect(identical(resH, outHstack), isTrue);

          final outCopy = NDArray.create([1, 2], DType.float64);
          final resCopy = copy(a, out: outCopy);
          expect(identical(resCopy, outCopy), isTrue);

          // permutation with out
          final pOut = NDArray.create([1, 2], DType.float64);
          final resPerm = permutation(a, out: pOut);
          expect(identical(resPerm, pOut), isTrue);

          // ogrid with out
          final r = GridRange(0.0, 2.0, numPoints: 3);
          final outOgrid = [
            NDArray<Float64>.create([3], DType.float64),
          ];
          final resOgrid = ogrid([r], out: outOgrid);
          expect(identical(resOgrid[0], outOgrid[0]), isTrue);

          // ogrid validation
          final badShape = [
            NDArray<Float64>.create([5], DType.float64),
          ];
          expect(() => ogrid([r], out: badShape), throwsArgumentError);

          final disposedOut = NDArray<Float64>.create([3], DType.float64)
            ..dispose();
          expect(() => ogrid([r], out: [disposedOut]), throwsStateError);
        });
      },
    );
  });
}
