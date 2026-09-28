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
  group('Review Cycle 6: Sorting, Searching & Aliasing Regression Tests', () {
    test('searchsorted with non-contiguous out (step=2 and step=-1)', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
        );
        final v = NDArray<Float64>.fromList(
          [15.0, 35.0, 5.0],
          [3],
          DType.float64,
        );

        // Stepped view (step = 2)
        final full = NDArray<Int32>.fromList(
          [-1, -1, -1, -1, -1, -1],
          [6],
          DType.int32,
        );
        final outStepped = full.slice([Slice(start: 0, stop: 6, step: 2)]);
        final res = searchsorted(a, v, out: outStepped);
        expect(identical(res, outStepped), isTrue);
        expect(outStepped.toList(), equals([1, 3, 0]));
        // Neighboring elements must remain untouched (-1)
        expect(full.toList(), equals([1, -1, 3, -1, 0, -1]));

        // Reversed view (step = -1)
        final revFull = NDArray<Int32>.fromList([99, 99, 99], [3], DType.int32);
        final outRev = revFull.slice([Slice(step: -1)]);
        searchsorted(a, v, out: outRev);
        expect(outRev.toList(), equals([1, 3, 0]));
        expect(revFull.toList(), equals([0, 3, 1]));
      });
    });

    test('searchsorted with out aliasing a, v, or sorter (Int32 arrays)', () {
      NDArray.scope(() {
        // out aliases a
        final a1 = NDArray<Int32>.fromList([10, 20, 30], [3], DType.int32);
        final v1 = NDArray<Int32>.fromList([25, 5, 15], [3], DType.int32);
        searchsorted(a1, v1, out: a1);
        expect(a1.toList(), equals([2, 0, 1]));

        // out aliases v
        final a2 = NDArray<Int32>.fromList([10, 20, 30], [3], DType.int32);
        final v2 = NDArray<Int32>.fromList([25, 5, 15], [3], DType.int32);
        searchsorted(a2, v2, out: v2);
        expect(v2.toList(), equals([2, 0, 1]));

        // out aliases sorter
        final a3 = NDArray<Int32>.fromList([30, 10, 20], [3], DType.int32);
        final sorter3 = NDArray<Int32>.fromList([1, 2, 0], [3], DType.int32);
        final v3 = NDArray<Int32>.fromList([25, 5, 15], [3], DType.int32);
        searchsorted(a3, v3, sorter: sorter3, out: sorter3);
        expect(sorter3.toList(), equals([2, 0, 1]));
      });
    });

    test(
      'searchsorted with uint64 values > 2^53 verifies exact comparison',
      () {
        NDArray.scope(() {
          const base = (1 << 53) + 100;
          final a = NDArray<Uint64>.fromList(
            [base, base + 1, base + 2, base + 3],
            [4],
            DType.uint64,
          );
          final v = NDArray<Uint64>.fromList(
            [base + 2, base + 1, base],
            [3],
            DType.uint64,
          );
          final res = searchsorted(a, v);
          expect(res.toList(), equals([2, 1, 0]));
        });
      },
    );

    test('argsort and argpartition with out aliasing a (Int32 in-place)', () {
      NDArray.scope(() {
        // argsort with out: a
        final aSort = NDArray<Int32>.fromList([30, 10, 20], [3], DType.int32);
        final resSort = argsort(aSort, out: aSort);
        expect(identical(resSort, aSort), isTrue);
        expect(aSort.toList(), equals([1, 2, 0]));

        // argpartition with out: a
        final aPart = NDArray<Int32>.fromList(
          [40, 10, 30, 20],
          [4],
          DType.int32,
        );
        final origValues = [40, 10, 30, 20];
        final resPart = argpartition(aPart, 2, out: aPart);
        expect(identical(resPart, aPart), isTrue);
        final partIndices = aPart.toList().cast<int>();
        // Element at kth=2 in sorted [10, 20, 30, 40] is 30 (original index 2)
        expect(origValues[partIndices[2]], equals(30));
        expect(origValues[partIndices[0]], lessThanOrEqualTo(30));
        expect(origValues[partIndices[1]], lessThanOrEqualTo(30));
        expect(origValues[partIndices[3]], greaterThanOrEqualTo(30));
      });
    });

    test(
      'where with rank = 8 succeeds accurately and rank = 9 throws UnsupportedError without leaking memory',
      () {
        final initialMarker = ScratchArena.marker;
        NDArray.scope(() {
          final shape8 = [2, 1, 1, 1, 1, 1, 1, 2]; // rank 8, size 4
          final cond8 = NDArray<Boolean>.fromList(
            [true, false, false, true],
            shape8,
            DType.boolean,
          );
          final x8 = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            shape8,
            DType.float64,
          );
          final y8 = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0],
            shape8,
            DType.float64,
          );
          final res8 = where(cond8, x8, y8) as NDArray<Float64>;
          expect(res8.shape, equals(shape8));
          expect(res8.reshape([4]).toList(), equals([1.0, 20.0, 30.0, 4.0]));

          final shape9 = [2, 1, 1, 1, 1, 1, 1, 1, 2]; // rank 9, size 4
          final cond9 = NDArray<Boolean>.fromList(
            [true, false, false, true],
            shape9,
            DType.boolean,
          );
          final x9 = NDArray<Int32>.fromList([1, 2, 3, 4], shape9, DType.int32);
          final y9 = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0],
            shape9,
            DType.float64,
          );
          expect(() => where(cond9, x9, y9), throwsUnsupportedError);
        });
        expect(ScratchArena.marker.offset, equals(initialMarker.offset));
      },
    );

    test('exception paths do not leak ScratchArena markers', () {
      final initialMarker = ScratchArena.marker;
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final vInt = NDArray<Int32>.fromList([1, 2], [2], DType.int32);
        final badSorter = NDArray<Int32>.fromList([0, 99, 1], [3], DType.int32);

        // searchsorted mismatched dtype
        expect(() => searchsorted<DTypeTag>(a, vInt), throwsArgumentError);
        // searchsorted out-of-bounds sorter index
        expect(
          () => searchsorted(a, a, sorter: badSorter),
          throwsA(isA<IndexError>()),
        );
        // argpartition invalid kth
        expect(() => argpartition(a, 10), throwsRangeError);
        // kron incompatible out shape
        final badOut = NDArray<Float64>.zeros([2], DType.float64);
        expect(() => kron<Float64>(a, a, out: badOut), throwsArgumentError);
        // fft invalid axis
        expect(() => fft(a, axis: 5), throwsRangeError);
        // rfft invalid n
        expect(() => rfft(a, n: 0), throwsArgumentError);
      });
      expect(ScratchArena.marker.offset, equals(initialMarker.offset));
    });
  });
}
