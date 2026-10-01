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

import 'dart:io';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Workstream 5 Fixes', () {
    test('1. count_nonzero negative axis support and bounds validation', () {
      final a = NDArray.fromList(
        [0, 1, 0, 2, 3, 0, 4, 0, 5],
        [3, 3],
        DType.int32,
      );

      // axis = -1 (columns)
      final cnzNeg1 = count_nonzero(a, axis: -1);
      expect(cnzNeg1.shape, equals([3]));
      expect(cnzNeg1.toList(), equals([1, 2, 2]));

      // axis = -2 (rows)
      final cnzNeg2 = count_nonzero(a, axis: -2);
      expect(cnzNeg2.shape, equals([3]));
      expect(cnzNeg2.toList(), equals([2, 2, 1]));

      // out-of-bounds negative axis
      expect(() => count_nonzero(a, axis: -3), throwsRangeError);
      expect(() => count_nonzero(a, axis: 3), throwsRangeError);
    });

    test('2. argmin / argmax bounds validation and negative axis', () {
      final a = NDArray.fromList(
        [10.0, 20.0, 5.0, 30.0, 1.0, 40.0],
        [2, 3],
        DType.float64,
      );

      // Valid negative axis
      final minCol = argmin(a, axis: -1);
      expect(minCol.shape, equals([2]));
      expect(minCol.toList(), equals([2, 1]));

      final minRow = argmin(a, axis: -2);
      expect(minRow.shape, equals([3]));
      expect(minRow.toList(), equals([0, 1, 0]));

      final maxCol = argmax(a, axis: -1);
      expect(maxCol.shape, equals([2]));
      expect(maxCol.toList(), equals([1, 2]));

      // Out of bounds axis should throw RangeError without mutating targetShape
      expect(() => argmin(a, axis: -3), throwsRangeError);
      expect(() => argmin(a, axis: 2), throwsRangeError);
      expect(() => argmax(a, axis: -3), throwsRangeError);
      expect(() => argmax(a, axis: 2), throwsRangeError);
    });

    test('3. where() with strided / non-contiguous out buffer', () {
      final cond = NDArray.fromList(
        [true, false, false, true],
        [2, 2],
        DType.boolean,
      );
      final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      final y = NDArray.fromList(
        [10.0, 20.0, 30.0, 40.0],
        [2, 2],
        DType.float64,
      );

      // Create a non-contiguous / strided out buffer of shape [2, 2] from a [4, 2] array
      final backing = NDArray.zeros([4, 2], DType.float64);
      final stridedOut = backing.slice([
        Slice(start: 0, stop: 4, step: 2),
        Slice.all(),
      ]);
      expect(stridedOut.shape, equals([2, 2]));
      expect(stridedOut.isContiguous, isFalse);

      final res = where(cond, x, y, stridedOut);
      expect(identical(res, stridedOut), isTrue);
      expect(stridedOut.toList(), equals([1.0, 20.0, 30.0, 4.0]));
    });

    test('4. diff(n: 0) on strided arrays and out buffer', () {
      final base = NDArray.fromList([10, 20, 30, 40, 50, 60], [6], DType.int32);
      final strided = base.slice([
        Slice(start: 0, stop: 6, step: 2),
      ]); // [10, 30, 50]
      expect(strided.isContiguous, isFalse);

      final d0 = diff(strided, n: 0);
      expect(d0.shape, equals([3]));
      expect(d0.toList(), equals([10, 30, 50]));

      final out = NDArray.zeros([3], DType.int32);
      final d0Out = diff(strided, n: 0, out: out);
      expect(identical(d0Out, out), isTrue);
      expect(out.toList(), equals([10, 30, 50]));
    });

    test('5. concatenate negative axis support and out buffer', () {
      final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final b = NDArray.fromList([5, 6, 7, 8], [2, 2], DType.int32);

      // axis = -1 (columns -> shape [2, 4])
      final concatCols = concatenate([a, b], axis: -1);
      expect(concatCols.shape, equals([2, 4]));
      expect(concatCols.toList(), equals([1, 2, 5, 6, 3, 4, 7, 8]));

      // axis = -2 (rows -> shape [4, 2])
      final concatRows = concatenate([a, b], axis: -2);
      expect(concatRows.shape, equals([4, 2]));
      expect(concatRows.toList(), equals([1, 2, 3, 4, 5, 6, 7, 8]));

      // with out buffer
      final out = NDArray.zeros([2, 4], DType.int32);
      final resOut = concatenate([a, b], axis: -1, out: out);
      expect(identical(resOut, out), isTrue);
      expect(out.toList(), equals([1, 2, 5, 6, 3, 4, 7, 8]));
    });

    test('6. StatLength.normalize on 0-sized dimensions and padding', () {
      final statLenUniform = StatLength.all(2, 2);
      final normalizedUniform = statLenUniform.normalize([0, 5]);
      expect(normalizedUniform, equals([(0, 0), (2, 2)]));

      final statLenAxes = StatLength.axes([(3, 3), (1, 1)]);
      final normalizedAxes = statLenAxes.normalize([0, 4]);
      expect(normalizedAxes, equals([(0, 0), (1, 1)]));

      // Padding a 0-sized array
      final emptyArr = NDArray.zeros([0, 5], DType.float64);
      final padded = pad(
        emptyArr,
        PadWidth.all(1),
        mode: PaddingMode.constant,
        constantValues: PadValues.all(0.0),
      );
      expect(padded.shape, equals([2, 7]));
    });

    test('7. linspace with 0 samples', () {
      final ls0 = linspace(0.0, 10.0, 0, dtype: DType.float64);
      expect(ls0.shape, equals([0]));
      expect(ls0.toList(), equals([]));

      final (samples: lsStepArr, step: step) = linspaceWithStep(
        0.0,
        10.0,
        0,
        dtype: DType.float64,
      );
      expect(lsStepArr.shape, equals([0]));
      expect((step as double).isNaN, isTrue);

      final start = NDArray.fromList([0.0, 1.0], [2], DType.float64);
      final stop = NDArray.fromList([10.0, 11.0], [2], DType.float64);
      final grid0 = linspaceGrid(start, stop, 0);
      expect(grid0.shape, equals([0, 2]));

      final (samples: gridWithStep, step: gridStep) = linspaceGridWithStep(
        start,
        stop,
        0,
      );
      expect(gridWithStep.shape, equals([0, 2]));
      expect(gridStep.shape, equals([2]));
      expect(gridStep.getCell([0]).isNaN, isTrue);
      expect(gridStep.getCell([1]).isNaN, isTrue);
    });

    test('8. variance and nanvar on non-contiguous arrays', () {
      final base = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [4, 2],
        DType.float64,
      );
      final strided = base.slice([
        Slice(step: 2),
        Slice.all(),
      ]); // shape [2, 2]: [[1, 2], [5, 6]]
      expect(strided.isContiguous, isFalse);

      final v = variance(strided, axis: 0);
      expect(v.shape, equals([2]));
      expect(v.toList(), equals([4.0, 4.0]));

      final nv = nanvar(strided, axis: 0);
      expect(nv.shape, equals([2]));
      expect(nv.toList(), equals([4.0, 4.0]));
    });

    test('9. nelder_mead and lbfgs optimization memory stability in scope', () {
      NDArray.scope(() {
        final x0 = NDArray<Float64>.fromList([0.0, 0.0], [2], DType.float64);
        final resNM = nelder_mead((x) {
          final px = x.getCell([0]);
          final py = x.getCell([1]);
          return (px - 3.0) * (px - 3.0) + (py + 2.0) * (py + 2.0);
        }, x0);
        expect(resNM.success, isTrue);
        expect(resNM.x.getCell([0]), closeTo(3.0, 1e-2));
        expect(resNM.x.getCell([1]), closeTo(-2.0, 1e-2));

        final resLBFGS = lbfgs(
          (x) => 0.0,
          x0,
          funAndGrad: (x) {
            final px = x.getCell([0]);
            final py = x.getCell([1]);
            final fVal = (px - 3.0) * (px - 3.0) + (py + 2.0) * (py + 2.0);
            final g = NDArray<Float64>.fromList(
              [2.0 * (px - 3.0), 2.0 * (py + 2.0)],
              [2],
              DType.float64,
            );
            return (fVal, g);
          },
        );
        expect(resLBFGS.success, isTrue);
        expect(resLBFGS.x.getCell([0]), closeTo(3.0, 1e-2));
        expect(resLBFGS.x.getCell([1]), closeTo(-2.0, 1e-2));
      });
    });

    test('10. financial.dart PaymentDue enum support', () {
      final rate = NDArray.fromList([0.05], [1], DType.float64);
      final nper = NDArray.fromList([10.0], [1], DType.float64);
      final pmt = NDArray.fromList([-100.0], [1], DType.float64);
      final pvVal = NDArray.fromList([0.0], [1], DType.float64);

      final fvEndEnum = fv(rate, nper, pmt, pvVal, when: PaymentDue.end);
      final fvBeginEnum = fv(rate, nper, pmt, pvVal, when: PaymentDue.begin);

      expect(fvBeginEnum.getCell([0]), isNot(equals(fvEndEnum.getCell([0]))));
    });

    test('11. loadz / savez roundtrip with Uint8List compatibility', () {
      final tmpDir = Directory.systemTemp.createTempSync('loadz_test');
      final filePath = '${tmpDir.path}/test_archive.npz';
      try {
        final arr1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final arr2 = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
        savez(filePath, {'arr1': arr1, 'arr2': arr2});

        final loaded = loadz(filePath);
        expect(loaded.containsKey('arr1'), isTrue);
        expect(loaded.containsKey('arr2'), isTrue);
        expect(loaded['arr1']!.toList(), equals([1.0, 2.0, 3.0]));
        expect(loaded['arr2']!.toList(), equals([10, 20, 30, 40]));
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test(
      '12. out parameter support on stack, roll, split, array_split, interp',
      () {
        // stack
        final a1 = NDArray.fromList([1, 2], [2], DType.int32);
        final a2 = NDArray.fromList([3, 4], [2], DType.int32);
        final outStack = NDArray.zeros([2, 2], DType.int32);
        final resStack = stack([a1, a2], axis: 0, out: outStack);
        expect(identical(resStack, outStack), isTrue);
        expect(outStack.toList(), equals([1, 2, 3, 4]));

        // roll
        final rArr = NDArray.fromList([1, 2, 3, 4, 5], [5], DType.int32);
        final outRoll = NDArray.zeros([5], DType.int32);
        final resRoll = roll(rArr, 2, out: outRoll);
        expect(identical(resRoll, outRoll), isTrue);
        expect(outRoll.toList(), equals([4, 5, 1, 2, 3]));

        // split & array_split
        final sArr = NDArray.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
        final outSplit = [
          NDArray.zeros([3], DType.int32),
          NDArray.zeros([3], DType.int32),
        ];
        final resSplit = split(sArr, 2, out: outSplit);
        expect(identical(resSplit[0], outSplit[0]), isTrue);
        expect(identical(resSplit[1], outSplit[1]), isTrue);
        expect(outSplit[0].toList(), equals([1, 2, 3]));
        expect(outSplit[1].toList(), equals([4, 5, 6]));

        final outArraySplit = [
          NDArray.zeros([2], DType.int32),
          NDArray.zeros([2], DType.int32),
          NDArray.zeros([2], DType.int32),
        ];
        final resArraySplit = array_split(sArr, 3, out: outArraySplit);
        expect(identical(resArraySplit[0], outArraySplit[0]), isTrue);
        expect(outArraySplit[0].toList(), equals([1, 2]));
        expect(outArraySplit[1].toList(), equals([3, 4]));
        expect(outArraySplit[2].toList(), equals([5, 6]));

        // interp
        final x = NDArray.fromList([2.5], [1], DType.float64);
        final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final fp = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
        final outInterp = NDArray.zeros([1], DType.float64);
        final resInterp = interp(x, xp, fp, out: outInterp);
        expect(identical(resInterp, outInterp), isTrue);
        expect(outInterp.getCell([0]), equals(25.0));

        final resInterpolate = interpolate(x, xp, fp, out: outInterp);
        expect(identical(resInterpolate, outInterp), isTrue);
        expect(outInterp.getCell([0]), equals(25.0));
      },
    );
  });
}
