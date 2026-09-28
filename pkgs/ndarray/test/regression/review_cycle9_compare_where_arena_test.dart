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
  group('Review Cycle 9: Comparison, Where Mask & Arena Regressions', () {
    test(
      'equal, notEqual, greater, greaterEqual, less, lessEqual with where mask do not leak tempRes',
      () {
        NDArray.clearTrackedAllocations();
        final a = NDArray.fromList([1, 5, 3, 4], [4], DType.int32);
        final b = NDArray.fromList([1, 2, 6, 4], [4], DType.int32);
        final mask = NDArray<Boolean>.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final baselineCount = NDArray.trackedAllocations.length;
        final baselineMarker = ScratchArena.marker;

        final rEq = equal(a, b, where: mask);
        expect(rEq.getCell([0]), isTrue);
        expect(rEq.getCell([2]), isFalse);
        rEq.dispose();
        expect(NDArray.trackedAllocations.length, equals(baselineCount));

        final rNe = notEqual(a, b, where: mask);
        expect(rNe.getCell([0]), isFalse);
        expect(rNe.getCell([2]), isTrue);
        rNe.dispose();
        expect(NDArray.trackedAllocations.length, equals(baselineCount));

        final rGt = greater(a, b, where: mask);
        expect(rGt.getCell([0]), isFalse);
        expect(rGt.getCell([2]), isFalse);
        rGt.dispose();
        expect(NDArray.trackedAllocations.length, equals(baselineCount));

        final rGe = greaterEqual(a, b, where: mask);
        expect(rGe.getCell([0]), isTrue);
        expect(rGe.getCell([2]), isFalse);
        rGe.dispose();
        expect(NDArray.trackedAllocations.length, equals(baselineCount));

        final rLt = less(a, b, where: mask);
        expect(rLt.getCell([0]), isFalse);
        expect(rLt.getCell([2]), isTrue);
        rLt.dispose();
        expect(NDArray.trackedAllocations.length, equals(baselineCount));

        final rLe = lessEqual(a, b, where: mask);
        expect(rLe.getCell([0]), isTrue);
        expect(rLe.getCell([2]), isTrue);
        rLe.dispose();
        expect(NDArray.trackedAllocations.length, equals(baselineCount));
        expect(ScratchArena.marker.pageIndex, equals(baselineMarker.pageIndex));
        expect(ScratchArena.marker.offset, equals(baselineMarker.offset));

        NDArray.scope(() {
          final out = NDArray<Boolean>.fromList(
            [false, true, false, true],
            [4],
            DType.boolean,
          );
          equal(a, b, where: mask, out: out);
          expect(out.toList(), equals([true, true, false, true]));
        });
        expect(NDArray.trackedAllocations.length, equals(baselineCount));

        a.dispose();
        b.dispose();
        mask.dispose();
        NDArray.checkNoLeaks();
      },
    );

    test(
      'comparison ops with out: flip(a) handle overlapping buffers without corruption',
      () {
        // equal([true, false, false, false], [true, true, true, true], out: flip(a))
        // Expected comparison result: [true, false, false, false]
        // Written into flip(a): flip(a) is [true, false, false, false], so a becomes [false, false, false, true]
        final aEq = NDArray<Boolean>.fromList(
          [true, false, false, false],
          [4],
          DType.boolean,
        );
        final bEq = NDArray<Boolean>.fromList(
          [true, true, true, true],
          [4],
          DType.boolean,
        );
        final flipAEq = flip(aEq);
        equal(aEq, bEq, out: flipAEq);
        expect(flipAEq.toList(), equals([true, false, false, false]));
        expect(aEq.toList(), equals([false, false, false, true]));
        flipAEq.dispose();
        aEq.dispose();
        bEq.dispose();

        // notEqual([true, false, false, false], [true, true, true, true], out: flip(a))
        // Expected comparison result: [false, true, true, true]
        // Written into flip(a): a becomes [true, true, true, false]
        final aNe = NDArray<Boolean>.fromList(
          [true, false, false, false],
          [4],
          DType.boolean,
        );
        final bNe = NDArray<Boolean>.fromList(
          [true, true, true, true],
          [4],
          DType.boolean,
        );
        final flipANe = flip(aNe);
        notEqual(aNe, bNe, out: flipANe);
        expect(flipANe.toList(), equals([false, true, true, true]));
        expect(aNe.toList(), equals([true, true, true, false]));
        flipANe.dispose();
        aNe.dispose();
        bNe.dispose();

        // greater([true, false, false, false], [false, false, false, false], out: flip(a))
        // Expected comparison result: [true, false, false, false]
        // Written into flip(a): a becomes [false, false, false, true]
        final aGt = NDArray<Boolean>.fromList(
          [true, false, false, false],
          [4],
          DType.boolean,
        );
        final bGt = NDArray<Boolean>.fromList(
          [false, false, false, false],
          [4],
          DType.boolean,
        );
        final flipAGt = flip(aGt);
        greater(aGt, bGt, out: flipAGt);
        expect(flipAGt.toList(), equals([true, false, false, false]));
        expect(aGt.toList(), equals([false, false, false, true]));
        flipAGt.dispose();
        aGt.dispose();
        bGt.dispose();

        // less([false, true, true, true], [true, true, true, true], out: flip(a))
        // Expected comparison result: [true, false, false, false]
        // Written into flip(a): a becomes [false, false, false, true]
        final aLt = NDArray<Boolean>.fromList(
          [false, true, true, true],
          [4],
          DType.boolean,
        );
        final bLt = NDArray<Boolean>.fromList(
          [true, true, true, true],
          [4],
          DType.boolean,
        );
        final flipALt = flip(aLt);
        less(aLt, bLt, out: flipALt);
        expect(flipALt.toList(), equals([true, false, false, false]));
        expect(aLt.toList(), equals([false, false, false, true]));
        flipALt.dispose();
        aLt.dispose();
        bLt.dispose();
      },
    );

    test(
      'invalid where mask throws ArgumentError without leaking NDArray or ScratchArena marker',
      () {
        NDArray.clearTrackedAllocations();
        final aF64 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final bF64 = NDArray<Float64>.fromList(
          [4.0, 5.0, 6.0],
          [3],
          DType.float64,
        );
        final aInt = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final badShapeMask = NDArray<Boolean>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );
        final badDTypeMask = NDArray<Float64>.fromList(
          [1.0, 0.0, 1.0],
          [3],
          DType.float64,
        );

        final baselineCount = NDArray.trackedAllocations.length;
        final baselineMarker = ScratchArena.marker;

        void verifyNoLeak() {
          expect(NDArray.trackedAllocations.length, equals(baselineCount));
          expect(
            ScratchArena.marker.pageIndex,
            equals(baselineMarker.pageIndex),
          );
          expect(ScratchArena.marker.offset, equals(baselineMarker.offset));
        }

        for (final badMask in <NDArray<AnySpec>>[badShapeMask, badDTypeMask]) {
          expect(() => equal(aF64, bF64, where: badMask), throwsArgumentError);
          verifyNoLeak();

          expect(() => logicalNot(aF64, where: badMask), throwsArgumentError);
          verifyNoLeak();

          expect(
            () => logicalAnd(aF64, bF64, where: badMask),
            throwsArgumentError,
          );
          verifyNoLeak();

          expect(() => sin(aF64, where: badMask), throwsArgumentError);
          verifyNoLeak();

          expect(() => add(aF64, bF64, where: badMask), throwsArgumentError);
          verifyNoLeak();

          expect(() => exp(aF64, where: badMask), throwsArgumentError);
          verifyNoLeak();

          expect(() => floor(aF64, where: badMask), throwsArgumentError);
          verifyNoLeak();

          expect(() => invert(aInt, where: badMask), throwsArgumentError);
          verifyNoLeak();
        }

        aF64.dispose();
        bF64.dispose();
        aInt.dispose();
        badShapeMask.dispose();
        badDTypeMask.dispose();
        NDArray.checkNoLeaks();
      },
    );
  });
}
