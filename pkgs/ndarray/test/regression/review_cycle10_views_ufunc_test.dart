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
  group('Review Cycle 10: Ufunc, Clip, Sorting & Stats View Lifecycles', () {
    test(
      '1a. atUfunc with numIndices = 20 and DType.int32 / DType.int16 / 2D indices (no 8x buffer overflow)',
      () {
        NDArray.scope(() {
          final a32 = NDArray.zeros([10], DType.float64);
          final idxList = List<int>.generate(20, (i) => i % 10);
          final indices32 = NDArray.fromList(idxList, [20], DType.int32);
          final b32 = NDArray.fromList(List<double>.generate(20, (i) => 1.0), [
            20,
          ], DType.float64);

          at(a32, indices32, b32, op: BinaryOp.add);
          for (var i = 0; i < 10; i++) {
            expect(a32.getCell([i]), closeTo(2.0, 1e-12));
          }

          // Also test DType.int16 and 2D indices (verifying getCellFlat works on rank != 1)
          final a16 = NDArray.zeros([6], DType.int32);
          final indices16 = NDArray.fromList(
            List<int>.generate(20, (i) => i % 6),
            [4, 5],
            DType.int16,
          );
          final b16 = NDArray.scalar(3, dtype: DType.int32);

          at(a16, indices16, b16, op: BinaryOp.add);
          // Indices 0..5: 0,1 take 4 hits (12); 2,3,4,5 take 3 hits (9)
          expect(a16.toList(), equals([12, 12, 9, 9, 9, 9]));
        });
      },
    );

    test(
      '1b. reduceatUfunc with DType.int16 / DType.uint8 indices and out sharing memory with a',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );
          final idx16 = NDArray.fromList([0, 2, 4], [3], DType.int16);
          final res16 = reduceat(a, idx16, op: BinaryOp.add);
          expect(res16.toList(), equals([3.0, 7.0, 11.0]));

          final idxU8 = NDArray.fromList([0, 3], [2], DType.uint8);
          final resU8 = reduceat(a, idxU8, op: BinaryOp.add);
          expect(resU8.toList(), equals([6.0, 15.0]));

          // Test out sharing memory with a
          final aOverlap = NDArray.fromList(
            [10, 20, 30, 40, 50, 60],
            [6],
            DType.int32,
          );
          final outSlice = aOverlap.slice([Slice(start: 0, stop: 3)]);
          final idxOverlap = NDArray.fromList([0, 2, 4], [3], DType.int16);
          reduceat(aOverlap, idxOverlap, op: BinaryOp.add, out: outSlice);
          expect(outSlice.toList(), equals([30, 70, 110]));
        });
      },
    );

    test(
      '2. sort(a, axis: 0) on a 2D array disposes temporary views and allows a.dispose()',
      () {
        final a = NDArray.fromList(
          [4.0, 1.0, 3.0, 2.0, 6.0, 0.0],
          [3, 2],
          DType.float64,
        );
        final result = sort(a, axis: 0);
        expect(result.shape, equals([3, 2]));
        expect(result.toList(), equals([3.0, 0.0, 4.0, 1.0, 6.0, 2.0]));

        // Both a and result must dispose cleanly without StateError: Cannot dispose NDArray with active view(s)
        expect(() => a.dispose(), returnsNormally);
        expect(() => result.dispose(), returnsNormally);

        // Also verify with out parameter
        final a2 = NDArray.fromList([5, 2, 1, 4], [2, 2], DType.int32);
        final out2 = NDArray.zeros([2, 2], DType.int32);
        sort(a2, axis: 0, out: out2);
        expect(out2.toList(), equals([1, 2, 5, 4]));
        expect(() => a2.dispose(), returnsNormally);
        expect(() => out2.dispose(), returnsNormally);
      },
    );

    test(
      '3. clip and clipArray with min/max and 1D/2D arrays dispose broadcast views cleanly',
      () {
        final a = NDArray.fromList(
          [-2.0, 0.5, 3.0, 7.0],
          [2, 2],
          DType.float64,
        );
        final cMin = clip(a, min: 0.0);
        final cMax = clip(a, max: 5.0);
        expect(cMin.toList(), equals([0.0, 0.5, 3.0, 7.0]));
        expect(cMax.toList(), equals([-2.0, 0.5, 3.0, 5.0]));
        cMin.dispose();
        cMax.dispose();

        // clipArray with min only (max == null), max only (min == null), and broadcasting 1D against 2D
        final minBound1D = NDArray.fromList([-1.0, 1.0], [2], DType.float64);
        final maxBound1D = NDArray.fromList([2.0, 6.0], [2], DType.float64);

        final caMinOnly = clipArray(a, min: minBound1D);
        expect(caMinOnly.toList(), equals([-1.0, 1.0, 3.0, 7.0]));
        caMinOnly.dispose();

        final caMaxOnly = clipArray(a, max: maxBound1D);
        expect(caMaxOnly.toList(), equals([-2.0, 0.5, 2.0, 6.0]));
        caMaxOnly.dispose();

        final caBoth = clipArray(a, min: minBound1D, max: maxBound1D);
        expect(caBoth.toList(), equals([-1.0, 1.0, 2.0, 6.0]));
        caBoth.dispose();

        expect(() => minBound1D.dispose(), returnsNormally);
        expect(() => maxBound1D.dispose(), returnsNormally);
        expect(() => a.dispose(), returnsNormally);
      },
    );

    test(
      '4a. cov(v) on a 1D vector inside and outside NDArray.scope returns undisposed 0D array',
      () {
        // Outside NDArray.scope
        final v = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final cOutside = cov(v);
        expect(cOutside.isDisposed, isFalse);
        expect(cOutside.shape, isEmpty);
        expect(cOutside.scalar, closeTo(1.0, 1e-12));
        expect(() => v.dispose(), returnsNormally);
        expect(() => cOutside.dispose(), returnsNormally);

        // Inside NDArray.scope
        NDArray.scope(() {
          final vScoped = NDArray.fromList([2.0, 4.0, 6.0], [3], DType.float64);
          final cInside = cov(vScoped);
          expect(cInside.isDisposed, isFalse);
          expect(cInside.shape, isEmpty);
          expect(cInside.scalar, closeTo(4.0, 1e-12));
        });
      },
    );

    test(
      '4b. average(m, axis: 0, returned: true) with and without weights on 2D matrix',
      () {
        NDArray.scope(() {
          final m = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final resNoWeights = average<Float64, Float64, Float64>(
            m,
            axis: 0,
            returned: true,
          );
          expect(resNoWeights.average.isDisposed, isFalse);
          expect(resNoWeights.sumOfWeights, isNotNull);
          expect(resNoWeights.sumOfWeights!.isDisposed, isFalse);
          expect(resNoWeights.average.toList(), equals([2.5, 3.5, 4.5]));
          expect(resNoWeights.sumOfWeights!.toList(), equals([2.0, 2.0, 2.0]));

          final weights = NDArray.fromList([1.0, 3.0], [2], DType.float64);
          final resWithWeights = average<Float64, Float64, Float64>(
            m,
            axis: 0,
            weights: weights,
            returned: true,
          );
          expect(resWithWeights.average.isDisposed, isFalse);
          expect(resWithWeights.sumOfWeights, isNotNull);
          expect(resWithWeights.sumOfWeights!.isDisposed, isFalse);
          expect(resWithWeights.average.toList(), equals([3.25, 4.25, 5.25]));
          expect(
            resWithWeights.sumOfWeights!.toList(),
            equals([4.0, 4.0, 4.0]),
          );
        });
      },
    );

    test(
      '5. minimum / maximum / fmin / fmax with where mask (broadcast & matching) and out sharing memory with a',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 2.0, 8.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 3.0, 9.0],
            [2, 2],
            DType.float64,
          );

          // Broadcast 1D where mask (both bool and int32 dtypes)
          final where1D = NDArray.fromList([true, false], [2], DType.boolean);
          final whereInt1D = NDArray.fromList([1, 0], [2], DType.int32);
          final outMin = NDArray.fromList(
            [-1.0, -1.0, -1.0, -1.0],
            [2, 2],
            DType.float64,
          );
          binaryUfunc(a, b, op: BinaryOp.minimum, where: where1D, out: outMin);
          expect(outMin.toList(), equals([5.0, -1.0, 3.0, -1.0]));

          final outMax = NDArray.fromList(
            [-1.0, -1.0, -1.0, -1.0],
            [2, 2],
            DType.float64,
          );
          binaryUfunc(
            a,
            b,
            op: BinaryOp.maximum,
            where: whereInt1D,
            out: outMax,
          );
          expect(outMax.toList(), equals([10.0, -1.0, 8.0, -1.0]));

          // Out sharing memory with a (with NaN handling in fmin / fmax)
          final aNan = NDArray.fromList(
            [double.nan, 2.0, 8.0, double.nan],
            [2, 2],
            DType.float64,
          );
          final bVal = NDArray.fromList(
            [5.0, double.nan, 3.0, 9.0],
            [2, 2],
            DType.float64,
          );
          final maskMatch = NDArray.fromList(
            [true, true, false, true],
            [2, 2],
            DType.boolean,
          );
          binaryUfunc(
            aNan,
            bVal,
            op: BinaryOp.fmin,
            where: maskMatch,
            out: aNan,
          );
          expect(aNan.toList(), equals([5.0, 2.0, 8.0, 9.0]));

          final aNan2 = NDArray.fromList(
            [double.nan, 2.0, 8.0, 4.0],
            [2, 2],
            DType.float64,
          );
          binaryUfunc(
            aNan2,
            bVal,
            op: BinaryOp.fmax,
            where: maskMatch,
            out: aNan2,
          );
          expect(aNan2.toList(), equals([5.0, 2.0, 8.0, 9.0]));
        });
      },
    );
  });
}
