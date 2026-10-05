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

import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  group('GpuArray Indexing & Slicing', () {
    test('1D and 2D Multi-Axis Strided Slicing and Views', () {
      ResourceScope.scope(() {
        final a1 = GpuArray.fromList(
          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
          [6],
          DType.float64,
        );
        final sub1 = a1.slice([const Slice(1, 5, 2)]);
        expect(sub1.shape, equals([2]));
        expect(sub1.toList(), equals([1.0, 3.0]));

        final rev1 = a1.slice([const Slice(null, null, -1)]);
        expect(rev1.shape, equals([6]));
        expect(rev1.toList(), equals([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]));

        final a2 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final row1 = a2.slice([const Index(1), const All()]);
        expect(row1.shape, equals([3]));
        expect(row1.toList(), equals([4.0, 5.0, 6.0]));

        final col2 = a2.slice([const All(), const Index(2)]);
        expect(col2.shape, equals([3]));
        expect(col2.toList(), equals([3.0, 6.0, 9.0]));

        final sub2 = a2.slice([const Slice(0, 2), const Slice(1, 3)]);
        expect(sub2.shape, equals([2, 2]));
        expect(sub2.toList(), equals([2.0, 3.0, 5.0, 6.0]));

        // Strongly typed operator []
        final indexed = a2[[const Index(0), const All()]];
        expect(indexed.shape, equals([3]));
        expect(indexed.toList(), equals([1.0, 2.0, 3.0]));
      });
    });

    test('Ellipsis and NewAxis Slicing', () {
      ResourceScope.scope(() {
        final a3 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float32,
        );

        final sub = a3.slice([const Ellipsis(), const Index(1)]);
        expect(sub.shape, equals([2, 2]));
        expect(sub.toList(), equals([2.0, 4.0, 6.0, 8.0]));

        final exp = a3.slice([
          const NewAxis(),
          const All(),
          const All(),
          const All(),
        ]);
        expect(exp.shape, equals([1, 2, 2, 2]));
      });
    });

    test('where() conditional selection with broadcasting and out:', () {
      ResourceScope.scope(() {
        final cond = GpuArray.fromList(
          [true, false, false, true],
          [2, 2],
          DType.boolean,
        );
        final x = GpuArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final y = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final res = where<Float64>(cond, x, y);
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([10.0, 2.0, 3.0, 40.0]));

        final outBuf = GpuArray.empty([2, 2], DType.float64);
        final resOut = where<Float64>(cond, x, y, out: outBuf);
        expect(identical(resOut, outBuf), isTrue);
        expect(outBuf.toList(), equals([10.0, 2.0, 3.0, 40.0]));

        final coords = nonzero(cond);
        expect(coords.length, equals(2));
        expect(coords[0].toList(), equals([0, 1]));
        expect(coords[1].toList(), equals([0, 1]));
      });
    });

    test('select() and extract()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final cond1 = a.lessThan(2.5);
        final cond2 = a.greaterThan(4.0);

        final choice1 = GpuArray.fromList(
          [10.0, 10.0, 10.0, 10.0, 10.0],
          [5],
          DType.float64,
        );
        final choice2 = GpuArray.fromList(
          [50.0, 50.0, 50.0, 50.0, 50.0],
          [5],
          DType.float64,
        );

        final selected = select<Float64>([cond1, cond2], [choice1, choice2]);
        expect(selected.toList(), equals([10.0, 10.0, 0.0, 0.0, 50.0]));

        final outSelect = GpuArray.empty([5], DType.float64);
        select<Float64>([cond1, cond2], [choice1, choice2], out: outSelect);
        expect(outSelect.toList(), equals([10.0, 10.0, 0.0, 0.0, 50.0]));

        final extracted = extract(cond1, a);
        expect(extracted.shape, equals([2]));
        expect(extracted.toList(), equals([1.0, 2.0]));
      });
    });

    test('take(), put(), takeAlongAxis(), and putAlongAxis()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [10.0, 30.0, 20.0, 60.0, 40.0, 50.0],
          [2, 3],
          DType.float64,
        );

        final flatIdx = GpuArray.fromList([0, 3, 5], [3], DType.int32);
        final takenFlat = take(a, flatIdx);
        expect(takenFlat.shape, equals([3]));
        expect(takenFlat.toList(), equals([10.0, 60.0, 50.0]));

        final colIdx = GpuArray.fromList([2, 0], [2], DType.int32);
        final takenAxis = take(a, colIdx, axis: 1);
        expect(takenAxis.shape, equals([2, 2]));
        expect(takenAxis.toList(), equals([20.0, 10.0, 50.0, 60.0]));

        final indices = GpuArray.fromList(
          [0, 2, 1, 1, 0, 2],
          [2, 3],
          DType.int32,
        );

        final taken = takeAlongAxis(a, indices, 1);
        expect(taken.shape, equals([2, 3]));
        expect(taken.toList(), equals([10.0, 20.0, 30.0, 40.0, 60.0, 50.0]));

        final values = GpuArray.fromList(
          [99.0, 99.0, 99.0, 88.0, 88.0, 88.0],
          [2, 3],
          DType.float64,
        );

        putAlongAxis(a, indices, values, 1);
        expect(a.toList(), equals([99.0, 99.0, 99.0, 88.0, 88.0, 88.0]));

        final putIdx = GpuArray.fromList([0, 5], [2], DType.int32);
        final putVals = GpuArray.fromList([111.0, 222.0], [2], DType.float64);
        put(a, putIdx, putVals);
        expect(a.toList(), equals([111.0, 99.0, 99.0, 88.0, 88.0, 222.0]));
      });
    });

    test('nonzero(), flatnonzero(), argwhere()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final flat = flatnonzero(a);
        expect(flat, isA<GpuArray<Int64>>());
        expect(flat.dtype, equals(DType.int64));
        expect(flat.toList(), equals([0, 4, 6, 8]));

        final nz = a.nonzero();
        expect(nz.length, equals(2));
        expect(nz[0].dtype, equals(DType.int64));
        expect(nz[1].dtype, equals(DType.int64));
        expect(nz[0].toList(), equals([0, 1, 2, 2]));
        expect(nz[1].toList(), equals([0, 1, 0, 2]));

        final coords = argwhere(a);
        expect(coords, isA<GpuArray<Int64>>());
        expect(coords.dtype, equals(DType.int64));
        expect(coords.shape, equals([4, 2]));
        expect(coords.toList(), equals([0, 0, 1, 1, 2, 0, 2, 2]));
      });
    });

    test(
      'nonzero(), flatnonzero(), argwhere() with Complex64 and Complex128',
      () {
        ResourceScope.scope(() {
          final c64Data = [
            Complex(0.0, 2.0),
            Complex(0.0, 0.0),
            Complex(3.0, 0.0),
            Complex(4.0, -5.0),
          ];
          final aC64 = GpuArray.fromList(c64Data, [2, 2], DType.complex64);

          final flatC64 = flatnonzero(aC64);
          expect(flatC64.dtype, equals(DType.int64));
          expect(flatC64.toList(), equals([0, 2, 3]));

          final nzC64 = nonzero(aC64);
          expect(nzC64.length, equals(2));
          expect(nzC64[0].dtype, equals(DType.int64));
          expect(nzC64[0].toList(), equals([0, 1, 1]));
          expect(nzC64[1].toList(), equals([0, 0, 1]));

          final coordsC64 = argwhere(aC64);
          expect(coordsC64.dtype, equals(DType.int64));
          expect(coordsC64.shape, equals([3, 2]));
          expect(coordsC64.toList(), equals([0, 0, 1, 0, 1, 1]));

          final c128Data = [
            Complex(0.0, 0.0),
            Complex(0.0, 7.5),
            Complex(-1.0, 0.0),
          ];
          final aC128 = GpuArray.fromList(c128Data, [3], DType.complex128);

          final flatC128 = flatnonzero(aC128);
          expect(flatC128.dtype, equals(DType.int64));
          expect(flatC128.toList(), equals([1, 2]));

          final nzC128 = nonzero(aC128);
          expect(nzC128.length, equals(1));
          expect(nzC128[0].dtype, equals(DType.int64));
          expect(nzC128[0].toList(), equals([1, 2]));

          final coordsC128 = argwhere(aC128);
          expect(coordsC128.dtype, equals(DType.int64));
          expect(coordsC128.shape, equals([2, 1]));
          expect(coordsC128.toList(), equals([1, 2]));
        });
      },
    );

    test('sort() and argsort() across axes and dtypes', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [3.0, 1.0, 2.0, 6.0, 4.0, 5.0],
          [2, 3],
          DType.float64,
        );

        final sortedLast = sort(a);
        expect(sortedLast.shape, equals([2, 3]));
        expect(sortedLast.toList(), equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        final idxLast = argsort(a);
        expect(idxLast.dtype, equals(DType.int64));
        expect(idxLast.shape, equals([2, 3]));
        expect(idxLast.toList(), equals([1, 2, 0, 1, 2, 0]));

        final aAx0 = GpuArray.fromList(
          [9.0, 2.0, 7.0, 3.0, 8.0, 1.0],
          [2, 3],
          DType.float64,
        );
        final sortedAx0 = aAx0.sort(axis: 0);
        expect(sortedAx0.toList(), equals([3.0, 2.0, 1.0, 9.0, 8.0, 7.0]));
        final idxAx0 = aAx0.argsort(axis: 0);
        expect(idxAx0.dtype, equals(DType.int64));
        expect(idxAx0.toList(), equals([1, 0, 1, 0, 1, 0]));

        final sortedFlat = sort(aAx0, axis: null);
        expect(sortedFlat.shape, equals([6]));
        expect(sortedFlat.toList(), equals([1.0, 2.0, 3.0, 7.0, 8.0, 9.0]));

        final idxFlat = argsort(aAx0, axis: null);
        expect(idxFlat.dtype, equals(DType.int64));
        expect(idxFlat.shape, equals([6]));
        expect(idxFlat.toList(), equals([5, 1, 3, 2, 4, 0]));

        // Stable tie-breaking and out: parameter
        final ties = GpuArray.fromList([20, 10, 20, 10], [4], DType.int32);
        final outIdx = GpuArray.empty([4], DType.int64);
        final resIdx = argsort(ties, out: outIdx);
        expect(identical(resIdx, outIdx), isTrue);
        expect(outIdx.toList(), equals([1, 3, 0, 2]));
      });
    });

    test('topk(), partition(), and argpartition()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [10.0, 50.0, 20.0, 40.0, 30.0, 60.0],
          [2, 3],
          DType.float64,
        );

        final top2 = topk(a, 2, axis: 1, largest: true);
        expect(top2.values.shape, equals([2, 2]));
        expect(top2.values.toList(), equals([50.0, 20.0, 60.0, 40.0]));
        expect(top2.indices.dtype, equals(DType.int64));
        expect(top2.indices.shape, equals([2, 2]));
        expect(top2.indices.toList(), equals([1, 2, 2, 0]));

        final bot2 = a.topk(2, axis: 1, largest: false);
        expect(bot2.values.toList(), equals([10.0, 20.0, 30.0, 40.0]));
        expect(bot2.indices.toList(), equals([0, 2, 1, 0]));

        final pInput = GpuArray.fromList([7, 2, 9, 1, 5, 3], [6], DType.int32);
        final part = partition(pInput, [1, 4]);
        final partList = part.toList().cast<int>();
        expect(partList[1], equals(2));
        expect(partList[4], equals(7));
        expect(partList[0] <= partList[1], isTrue);
        expect(partList[5] >= partList[4], isTrue);

        final argp = pInput.argpartition(2);
        expect(argp.dtype, equals(DType.int64));
        final argpIdx = argp.toList().cast<int>();
        final origList = pInput.toList().cast<int>();
        expect(origList[argpIdx[2]], equals(3));

        expect(() => partition(pInput, 10), throwsRangeError);
        expect(() => topk(a, 5, axis: 1), throwsRangeError);
      });
    });

    test('searchsorted() with left, right, scalar, ND values, and sorter', () {
      ResourceScope.scope(() {
        final sorted = GpuArray.fromList(
          [10, 20, 20, 30, 40],
          [5],
          DType.int32,
        );

        final leftIdx = searchsorted(sorted, [5, 20, 25, 40, 50]);
        expect(leftIdx.dtype, equals(DType.int64));
        expect(leftIdx.shape, equals([5]));
        expect(leftIdx.toList(), equals([0, 1, 3, 4, 5]));

        final rightIdx = sorted.searchsorted([
          5,
          20,
          25,
          40,
          50,
        ], side: SearchSide.right);
        expect(rightIdx.dtype, equals(DType.int64));
        expect(rightIdx.toList(), equals([0, 3, 3, 5, 5]));

        final scalarIdx = searchsorted(sorted, 20);
        expect(scalarIdx.dtype, equals(DType.int64));
        expect(scalarIdx.shape, equals(const <int>[]));
        expect(scalarIdx.scalar, equals(1));

        final unsorted = GpuArray.fromList([30, 10, 40, 20], [4], DType.int32);
        final perm = argsort(unsorted);
        final viaSorter = searchsorted(unsorted, [10, 25, 40], sorter: perm);
        expect(viaSorter.toList(), equals([0, 2, 3]));
      });
    });

    test(
      'unique(), uniqueWithIndex(), uniqueWithInverse(), uniqueWithCounts(), uniqueAll()',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList([3, 1, 2, 1, 3, 3, 2], [7], DType.int32);

          final u = unique(a);
          expect(u.toList(), equals([1, 2, 3]));

          final uIdx = uniqueWithIndex(a);
          expect(uIdx.values.toList(), equals([1, 2, 3]));
          expect(uIdx.indices.dtype, equals(DType.int64));
          expect(uIdx.indices.toList(), equals([1, 2, 0]));

          final uInv = uniqueWithInverse(a);
          expect(uInv.values.toList(), equals([1, 2, 3]));
          expect(uInv.inverse.dtype, equals(DType.int64));
          expect(uInv.inverse.toList(), equals([2, 0, 1, 0, 2, 2, 1]));

          final uCnt = uniqueWithCounts(a);
          expect(uCnt.values.toList(), equals([1, 2, 3]));
          expect(uCnt.counts.dtype, equals(DType.int64));
          expect(uCnt.counts.toList(), equals([2, 2, 3]));

          final allRes = a.uniqueAll();
          expect(allRes.values.toList(), equals([1, 2, 3]));
          expect(allRes.indices.toList(), equals([1, 2, 0]));
          expect(allRes.inverse.toList(), equals([2, 0, 1, 0, 2, 2, 1]));
          expect(allRes.counts.toList(), equals([2, 2, 3]));

          // 2D unique along axis 0
          final mat = GpuArray.fromList(
            [1, 2, 3, 4, 1, 2],
            [3, 2],
            DType.int32,
          );
          final matU = uniqueAll(mat, axis: 0);
          expect(matU.values.shape, equals([2, 2]));
          expect(matU.values.toList(), equals([1, 2, 3, 4]));
          expect(matU.indices.toList(), equals([0, 1]));
          expect(matU.inverse.toList(), equals([0, 1, 0]));
          expect(matU.counts.toList(), equals([2, 1]));
        });
      },
    );

    test('bincount() unweighted and weighted', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList([0, 1, 1, 3, 2, 1, 7], [7], DType.int32);
        final counts = bincount(x);
        expect(counts.dtype, equals(DType.int64));
        expect(counts.shape, equals([8]));
        expect(counts.toList(), equals([1, 3, 1, 1, 0, 0, 0, 1]));

        final minLenCounts = x.bincount(minlength: 10);
        expect(minLenCounts.shape, equals([10]));
        expect(minLenCounts.toList(), equals([1, 3, 1, 1, 0, 0, 0, 1, 0, 0]));

        final w = GpuArray.fromList(
          [0.5, 1.0, 2.0, 4.0, 1.5, 0.5, 3.0],
          [7],
          DType.float64,
        );
        final weighted = bincount(x, weights: w);
        expect(weighted.dtype, equals(DType.float64));
        expect(weighted.shape, equals([8]));
        expect(
          weighted.toList(),
          equals([0.5, 3.5, 1.5, 4.0, 0.0, 0.0, 0.0, 3.0]),
        );

        final neg = GpuArray.fromList([1, -1, 2], [3], DType.int32);
        expect(() => bincount(neg), throwsArgumentError);
      });
    });

    test('cumsum(), cumprod(), and diff() across axes and dtypes', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);

        final csFlat = cumsum(a);
        expect(csFlat.dtype, equals(DType.int64));
        expect(csFlat.shape, equals([6]));
        expect(csFlat.toList(), equals([1, 3, 6, 10, 15, 21]));

        final csAx0 = a.cumsum(axis: 0);
        expect(csAx0.dtype, equals(DType.int64));
        expect(csAx0.shape, equals([2, 3]));
        expect(csAx0.toList(), equals([1, 2, 3, 5, 7, 9]));

        final csAx1 = cumsum(a, axis: 1);
        expect(csAx1.dtype, equals(DType.int64));
        expect(csAx1.toList(), equals([1, 3, 6, 4, 9, 15]));

        final cpFlat = cumprod(a);
        expect(cpFlat.dtype, equals(DType.int64));
        expect(cpFlat.toList(), equals([1, 2, 6, 24, 120, 720]));

        final cpAx1 = a.cumprod(axis: 1);
        expect(cpAx1.dtype, equals(DType.int64));
        expect(cpAx1.toList(), equals([1, 2, 6, 4, 20, 120]));

        // Multi-block scan (>256 elements)
        final large = GpuArray.ones([300], DType.int32);
        final largeCs = cumsum(large, axis: 0);
        expect(largeCs.dtype, equals(DType.int64));
        final largeList = largeCs.toList().cast<int>();
        expect(largeList.first, equals(1));
        expect(largeList[255], equals(256));
        expect(largeList[256], equals(257));
        expect(largeList.last, equals(300));

        // diff
        final dInput = GpuArray.fromList([1, 2, 4, 7, 0], [5], DType.int32);
        final d1 = diff(dInput);
        expect(d1.shape, equals([4]));
        expect(d1.toList(), equals([1, 2, 3, -7]));

        final d2 = dInput.diff(n: 2);
        expect(d2.shape, equals([3]));
        expect(d2.toList(), equals([1, 1, -10]));

        final dPrepApp = diff(dInput, prepend: 0, append: 10);
        expect(dPrepApp.shape, equals([6]));
        expect(dPrepApp.toList(), equals([1, 1, 2, 3, -7, 10]));
      });
    });

    test('Rejects broadcasted view as out destination', () {
      final base = GpuArray<Float32>.fromList([1.0], [1], DType.float32);
      final broadcasted = base.broadcastTo([4]);

      final cond = GpuArray.fromList(
        [true, false, true, false],
        [4],
        DType.boolean,
      );
      final inA = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
      final inB = GpuArray.fromList(
        [10.0, 20.0, 30.0, 40.0],
        [4],
        DType.float32,
      );
      final idx = GpuArray.fromList([0, 1, 2, 3], [4], DType.int32);

      expect(
        () => where(cond, inA, inB, out: broadcasted),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('Must be writeable and not a broadcasted view.'),
          ),
        ),
      );

      expect(
        () => take(inA, idx, out: broadcasted),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('Must be writeable and not a broadcasted view.'),
          ),
        ),
      );

      expect(
        () => takeAlongAxis(inA, idx, 0, out: broadcasted),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('Must be writeable and not a broadcasted view.'),
          ),
        ),
      );

      base.dispose();
      broadcasted.dispose();
      cond.dispose();
      inA.dispose();
      inB.dispose();
      idx.dispose();
    });
  });
}
