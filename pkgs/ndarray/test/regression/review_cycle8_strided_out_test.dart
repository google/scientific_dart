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
  group('Review Cycle 8: set_operations strided/flipped/aliased out', () {
    test(
      'unique with negative stride (flip) and positive stride (step=2) out',
      () {
        NDArray.scope(() {
          final ar = NDArray.fromList([3.0, 1.0, 2.0, 1.0], [4], DType.float64);

          // Negative stride: flip(buf)
          final buf1 = NDArray<Float64>.zeros([3], DType.float64);
          final flippedOut = flip(buf1);
          unique<Float64>(ar, out: flippedOut);
          expect(flippedOut[0].scalar, closeTo(1.0, 1e-12));
          expect(flippedOut[1].scalar, closeTo(2.0, 1e-12));
          expect(flippedOut[2].scalar, closeTo(3.0, 1e-12));
          // Underlying buf1 should be reversed
          expect(buf1[0].scalar, closeTo(3.0, 1e-12));
          expect(buf1[1].scalar, closeTo(2.0, 1e-12));
          expect(buf1[2].scalar, closeTo(1.0, 1e-12));

          // Positive stride > 1: step=2
          final buf2 = NDArray<Float64>.fromList(List.filled(6, -99.0), [
            6,
          ], DType.float64);
          final stridedOut = buf2.slice([Slice(step: 2)]);
          unique<Float64>(ar, out: stridedOut);
          expect(buf2[0].scalar, closeTo(1.0, 1e-12));
          expect(buf2[1].scalar, closeTo(-99.0, 1e-12));
          expect(buf2[2].scalar, closeTo(2.0, 1e-12));
          expect(buf2[3].scalar, closeTo(-99.0, 1e-12));
          expect(buf2[4].scalar, closeTo(3.0, 1e-12));
          expect(buf2[5].scalar, closeTo(-99.0, 1e-12));
        });
      },
    );

    test(
      'intersect1d, setdiff1d, setxor1d, union1d with flip and step=2 out',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int64);
          final b = NDArray.fromList([2, 3, 4], [3], DType.int64);

          // intersect1d -> [2, 3]
          final bufInter = NDArray.zeros([2], DType.int64);
          intersect1d<DTypeTag>(a, b, out: flip(bufInter));
          expect(bufInter[0].scalar, equals(3));
          expect(bufInter[1].scalar, equals(2));

          final bufInterStep = NDArray.zeros([4], DType.int64);
          intersect1d<DTypeTag>(
            a,
            b,
            out: bufInterStep.slice([Slice(step: 2)]),
          );
          expect(bufInterStep[0].scalar, equals(2));
          expect(bufInterStep[1].scalar, equals(0));
          expect(bufInterStep[2].scalar, equals(3));
          expect(bufInterStep[3].scalar, equals(0));

          // setdiff1d -> [1]
          final bufDiff = NDArray.zeros([2], DType.int64);
          setdiff1d<DTypeTag>(
            a,
            b,
            out: bufDiff.slice([Slice(start: 1, stop: 2)]),
          );
          expect(bufDiff[0].scalar, equals(0));
          expect(bufDiff[1].scalar, equals(1));

          // setxor1d -> [1, 4]
          final bufXor = NDArray.zeros([2], DType.int64);
          setxor1d<DTypeTag>(a, b, out: flip(bufXor));
          expect(bufXor[0].scalar, equals(4));
          expect(bufXor[1].scalar, equals(1));

          final bufXorStep = NDArray.zeros([4], DType.int64);
          setxor1d<DTypeTag>(a, b, out: bufXorStep.slice([Slice(step: 2)]));
          expect(bufXorStep[0].scalar, equals(1));
          expect(bufXorStep[1].scalar, equals(0));
          expect(bufXorStep[2].scalar, equals(4));
          expect(bufXorStep[3].scalar, equals(0));

          // union1d -> [1, 2, 3, 4]
          final bufUnion = NDArray.zeros([4], DType.int64);
          union1d<DTypeTag>(a, b, out: flip(bufUnion));
          expect(bufUnion[0].scalar, equals(4));
          expect(bufUnion[1].scalar, equals(3));
          expect(bufUnion[2].scalar, equals(2));
          expect(bufUnion[3].scalar, equals(1));

          final bufUnionStep = NDArray.zeros([8], DType.int64);
          union1d<DTypeTag>(a, b, out: bufUnionStep.slice([Slice(step: 2)]));
          expect(bufUnionStep[0].scalar, equals(1));
          expect(bufUnionStep[2].scalar, equals(2));
          expect(bufUnionStep[4].scalar, equals(3));
          expect(bufUnionStep[6].scalar, equals(4));
        });
      },
    );

    test('isin with flip(buf), step=2, and aliased out', () {
      NDArray.scope(() {
        final elem = NDArray.fromList([10, 20, 30, 40], [4], DType.int64);
        final testElem = NDArray.fromList([20, 40], [2], DType.int64);

        final bufFlip = NDArray<Boolean>.create([4], DType.boolean);
        final flipped = flip(bufFlip);
        isin<DTypeTag>(elem, testElem, out: flipped);
        // Logical flipped result should be [false, true, false, true]
        expect(flipped[0].scalar, isFalse);
        expect(flipped[1].scalar, isTrue);
        expect(flipped[2].scalar, isFalse);
        expect(flipped[3].scalar, isTrue);
        // Underlying bufFlip is reversed: [true, false, true, false]
        expect(bufFlip[0].scalar, isTrue);
        expect(bufFlip[1].scalar, isFalse);
        expect(bufFlip[2].scalar, isTrue);
        expect(bufFlip[3].scalar, isFalse);

        final bufStep = NDArray<Boolean>.fromList(List.filled(8, false), [
          8,
        ], DType.boolean);
        isin<DTypeTag>(elem, testElem, out: bufStep.slice([Slice(step: 2)]));
        expect(bufStep[0].scalar, isFalse);
        expect(bufStep[2].scalar, isTrue);
        expect(bufStep[4].scalar, isFalse);
        expect(bufStep[6].scalar, isTrue);

        // Aliased boolean element / testElements with out
        final boolBuf = NDArray<Boolean>.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final boolTest = NDArray<Boolean>.fromList([true], [1], DType.boolean);
        isin<Boolean>(boolBuf, boolTest, out: boolBuf);
        expect(boolBuf[0].scalar, isTrue);
        expect(boolBuf[1].scalar, isFalse);
        expect(boolBuf[2].scalar, isTrue);
        expect(boolBuf[3].scalar, isFalse);
      });
    });
  });

  group('Review Cycle 8: spacers strided/flipped/aliased out', () {
    test('linspace, logspace, geomspace with flip(buf) and step=2 out', () {
      NDArray.scope(() {
        // linspace
        final bufLin = NDArray<Float64>.zeros([4], DType.float64);
        linspace<Float64>(0.0, 3.0, 4, dtype: DType.float64, out: flip(bufLin));
        expect(bufLin[0].scalar, closeTo(3.0, 1e-12));
        expect(bufLin[1].scalar, closeTo(2.0, 1e-12));
        expect(bufLin[2].scalar, closeTo(1.0, 1e-12));
        expect(bufLin[3].scalar, closeTo(0.0, 1e-12));

        final bufLinStep = NDArray<Float64>.zeros([8], DType.float64);
        linspace<Float64>(
          0.0,
          3.0,
          4,
          dtype: DType.float64,
          out: bufLinStep.slice([Slice(step: 2)]),
        );
        expect(bufLinStep[0].scalar, closeTo(0.0, 1e-12));
        expect(bufLinStep[2].scalar, closeTo(1.0, 1e-12));
        expect(bufLinStep[4].scalar, closeTo(2.0, 1e-12));
        expect(bufLinStep[6].scalar, closeTo(3.0, 1e-12));

        // logspace
        final bufLog = NDArray<Float64>.zeros([3], DType.float64);
        logspace<Float64>(0.0, 2.0, 3, dtype: DType.float64, out: flip(bufLog));
        expect(bufLog[0].scalar, closeTo(100.0, 1e-9));
        expect(bufLog[1].scalar, closeTo(10.0, 1e-9));
        expect(bufLog[2].scalar, closeTo(1.0, 1e-9));

        final bufLogStep = NDArray<Float64>.zeros([6], DType.float64);
        logspace<Float64>(
          0.0,
          2.0,
          3,
          dtype: DType.float64,
          out: bufLogStep.slice([Slice(step: 2)]),
        );
        expect(bufLogStep[0].scalar, closeTo(1.0, 1e-9));
        expect(bufLogStep[2].scalar, closeTo(10.0, 1e-9));
        expect(bufLogStep[4].scalar, closeTo(100.0, 1e-9));

        // geomspace
        final bufGeom = NDArray<Float64>.zeros([3], DType.float64);
        geomspace<Float64>(
          1.0,
          100.0,
          3,
          dtype: DType.float64,
          out: flip(bufGeom),
        );
        expect(bufGeom[0].scalar, closeTo(100.0, 1e-9));
        expect(bufGeom[1].scalar, closeTo(10.0, 1e-9));
        expect(bufGeom[2].scalar, closeTo(1.0, 1e-9));

        final bufGeomStep = NDArray<Float64>.zeros([6], DType.float64);
        geomspace<Float64>(
          1.0,
          100.0,
          3,
          dtype: DType.float64,
          out: bufGeomStep.slice([Slice(step: 2)]),
        );
        expect(bufGeomStep[0].scalar, closeTo(1.0, 1e-9));
        expect(bufGeomStep[2].scalar, closeTo(10.0, 1e-9));
        expect(bufGeomStep[4].scalar, closeTo(100.0, 1e-9));
      });
    });

    test('linspaceGrid with out sharing memory with start or stop', () {
      NDArray.scope(() {
        final shared = NDArray<Float64>.fromList(
          [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
          [3, 2],
          DType.float64,
        );
        final startView = shared.slice([Slice(start: 0, stop: 1)]).reshape([2]);
        final stop = NDArray<Float64>.fromList([2.0, 30.0], [2], DType.float64);
        linspaceGrid<Float64>(startView, stop, 3, out: shared);
        expect(shared[[0, 0]], closeTo(0.0, 1e-12));
        expect(shared[[0, 1]], closeTo(10.0, 1e-12));
        expect(shared[[1, 0]], closeTo(1.0, 1e-12));
        expect(shared[[1, 1]], closeTo(20.0, 1e-12));
        expect(shared[[2, 0]], closeTo(2.0, 1e-12));
        expect(shared[[2, 1]], closeTo(30.0, 1e-12));
      });
    });
  });

  group('Review Cycle 8: distance strided/flipped/aliased out', () {
    test(
      'pdist (cosine and euclidean) and cdist with flip(buf) and aliased out',
      () {
        NDArray.scope(() {
          final x = NDArray<Float64>.fromList(
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [3, 2],
            DType.float64,
          );

          // pdist cosine with flip(buf)
          final expectedCos = pdist(x, metric: DistanceMetric.cosine);
          final bufCos = NDArray<Float64>.zeros([3], DType.float64);
          pdist(x, metric: DistanceMetric.cosine, out: flip(bufCos));
          expect(bufCos[0].scalar, closeTo(expectedCos[2].scalar, 1e-12));
          expect(bufCos[1].scalar, closeTo(expectedCos[1].scalar, 1e-12));
          expect(bufCos[2].scalar, closeTo(expectedCos[0].scalar, 1e-12));

          // pdist euclidean with out sharing memory with x
          final sharedX = x.copy();
          final expectedEuc = pdist(sharedX, metric: DistanceMetric.euclidean);
          final outSlice = sharedX.flatten().slice([Slice(start: 0, stop: 3)]);
          pdist(sharedX, metric: DistanceMetric.euclidean, out: outSlice);
          expect(outSlice[0].scalar, closeTo(expectedEuc[0].scalar, 1e-12));
          expect(outSlice[1].scalar, closeTo(expectedEuc[1].scalar, 1e-12));
          expect(outSlice[2].scalar, closeTo(expectedEuc[2].scalar, 1e-12));

          // cdist with flip(buf) and aliased out
          final xa = NDArray<Float64>.fromList(
            [1.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final xb = NDArray<Float64>.fromList(
            [0.0, 0.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final expectedCdist = cdist(xa, xb);
          final bufCdist = NDArray<Float64>.zeros([2, 2], DType.float64);
          cdist(xa, xb, out: flip(bufCdist, axis: 0));
          expect(bufCdist[[0, 0]], closeTo(expectedCdist[[1, 0]], 1e-12));
          expect(bufCdist[[0, 1]], closeTo(expectedCdist[[1, 1]], 1e-12));
          expect(bufCdist[[1, 0]], closeTo(expectedCdist[[0, 0]], 1e-12));
          expect(bufCdist[[1, 1]], closeTo(expectedCdist[[0, 1]], 1e-12));

          // cdist with out sharing memory with xa
          final sharedXa = xa.copy();
          cdist(sharedXa, xb, out: sharedXa);
          expect(sharedXa[[0, 0]], closeTo(expectedCdist[[0, 0]], 1e-12));
          expect(sharedXa[[1, 1]], closeTo(expectedCdist[[1, 1]], 1e-12));
        });
      },
    );
  });

  group('Review Cycle 8: calculus aliased out & ScratchArena unwind', () {
    test(
      'trapz and gradient with out sharing memory and ScratchArena marker preserved',
      () {
        NDArray.scope(() {
          final markerBefore = ScratchArena.marker;

          final y = NDArray<Float64>.fromList(
            [1.0, 2.0, 4.0, 7.0],
            [2, 2],
            DType.float64,
          );
          final expectedTrapz = trapz<Float64>(y, axis: 1);
          final outTrapzView = y
              .slice([Slice(), Slice(start: 0, stop: 1)])
              .reshape([2]);
          trapz<Float64>(y, axis: 1, out: outTrapzView);
          expect(
            outTrapzView[0].scalar,
            closeTo(expectedTrapz[0].scalar, 1e-12),
          );
          expect(
            outTrapzView[1].scalar,
            closeTo(expectedTrapz[1].scalar, 1e-12),
          );

          final f = NDArray<Float64>.fromList(
            [1.0, 2.0, 4.0, 7.0],
            [4],
            DType.float64,
          );
          final expectedGrad = gradient<Float64>(f);
          gradient<Float64>(f, out: f);
          for (var i = 0; i < 4; i++) {
            expect(f[i].scalar, closeTo(expectedGrad[i].scalar, 1e-12));
          }

          final markerAfter = ScratchArena.marker;
          expect(markerAfter.pageIndex, equals(markerBefore.pageIndex));
          expect(markerAfter.offset, equals(markerBefore.offset));
        });
      },
    );
  });
}
