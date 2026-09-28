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
  group('Stream 2 Cycle 2 Fixes', () {
    test(
      '1. atUfunc memory safety with non-int64 and non-contiguous indices',
      () {
        final a = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
        );
        // Int32 indices
        final idxInt32 = NDArray<int>.fromList([0, 2], [2], DType.int32);
        final b = NDArray<Float64>.fromList([5.0, 7.0], [2], DType.float64);

        atUfunc(a, idxInt32, b, op: BinaryOp.add);
        expect(a.toList(), equals([15.0, 20.0, 37.0, 40.0]));

        // Sliced non-contiguous indices
        final idx2D = NDArray<int>.fromList(
          [1, 99, 3, 99],
          [2, 2],
          DType.int64,
        );
        final idxSlice =
            idx2D[[const Slice(), 0]] as NDArray<int>; // non-contiguous [1, 3]
        final b2 = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);

        atUfunc(a, idxSlice, b2, op: BinaryOp.add);
        expect(a.toList(), equals([15.0, 21.0, 37.0, 42.0]));
      },
    );

    test('2. clipArray double pointer offset fix with sliced views', () {
      final a = NDArray<Float64>.fromList(
        [0.0, 10.0, 0.0, 20.0, 0.0, 30.0],
        [6],
        DType.float64,
      );
      final aSlice =
          a[const Slice(start: 1, stop: 6, step: 2)]
              as NDArray<Float64>; // [10, 20, 30], offset = 1

      final minArr = NDArray<Float64>.fromList(
        [0.0, 15.0, 0.0, 15.0, 0.0, 15.0],
        [6],
        DType.float64,
      );
      final minSlice =
          minArr[const Slice(start: 1, stop: 6, step: 2)]
              as NDArray<Float64>; // [15, 15, 15]

      final maxArr = NDArray<Float64>.fromList(
        [0.0, 25.0, 0.0, 25.0, 0.0, 25.0],
        [6],
        DType.float64,
      );
      final maxSlice =
          maxArr[const Slice(start: 1, stop: 6, step: 2)]
              as NDArray<Float64>; // [25, 25, 25]

      final result = clipArray(aSlice, min: minSlice, max: maxSlice);
      expect(result.toList(), equals([15.0, 20.0, 25.0]));
    });

    test('3. bitwise operations with where mask holder disposal', () {
      final a = NDArray<int>.fromList([1, 2, 3, 4], [4], DType.int32);
      final b = NDArray<int>.fromList([4, 3, 2, 1], [4], DType.int32);
      final mask = NDArray<Uint8>.fromList([1, 0, 1, 0], [4], DType.uint8);
      final out = NDArray<int>.zeros([4], DType.int32);

      final rAnd = bitwise_and(a, b, where: mask, out: out);
      expect(rAnd.toList(), equals([0, 0, 2, 0]));

      final rInv = invert(a, where: mask);
      expect(rInv.toList()[0], equals(~1));
    });

    test('4. arithmetic operations with where mask', () {
      final a = NDArray<Float64>.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [4],
        DType.float64,
      );
      final mask = NDArray<Uint8>.fromList([1, 0, 1, 0], [4], DType.uint8);
      final out = NDArray<Float64>.zeros([4], DType.float64);

      final rExpm1 = expm1(a, where: mask, out: out);
      expect(rExpm1.toList()[0], closeTo(math.exp(1.0) - 1.0, 1e-6));
      expect(rExpm1.toList()[1], equals(0.0));

      final rLog1p = log1p(a, where: mask);
      expect(rLog1p.toList()[0], closeTo(math.log(2.0), 1e-6));

      final rSquare = square(a, where: mask);
      expect(rSquare.toList()[0], equals(1.0));

      final rAbs = abs(a, where: mask);
      expect(rAbs.toList()[0], equals(1.0));

      final rRound = round(a, where: mask);
      expect(rRound.toList()[0], equals(1.0));

      final rCeil = ceil(a, where: mask);
      expect(rCeil.toList()[0], equals(1.0));

      final rFloor = floor(a, where: mask);
      expect(rFloor.toList()[0], equals(1.0));

      final rTrunc = trunc(a, where: mask);
      expect(rTrunc.toList()[0], equals(1.0));

      final rFix = fix(a, where: mask);
      expect(rFix.toList()[0], equals(1.0));
    });

    test(
      '5. deg2rad and rad2deg with 0D scalar inputs and sinc integer promotion',
      () {
        final deg0D = NDArray<Float64>.fromList([180.0], [], DType.float64);
        final rad0D = deg2rad(deg0D);
        expect(rad0D.shape, equals([]));
        expect(rad0D.scalar, closeTo(math.pi, 1e-6));

        final backDeg = rad2deg(rad0D);
        expect(backDeg.shape, equals([]));
        expect(backDeg.scalar, closeTo(180.0, 1e-6));

        final intArr = NDArray<int>.fromList([0, 1, 2], [3], DType.int32);
        final mask = NDArray<Uint8>.fromList([1, 1, 0], [3], DType.uint8);
        final out = NDArray<Float64>.zeros([3], DType.float64);
        final sincRes = sinc(intArr, where: mask, out: out);
        expect(sincRes.toList()[0], equals(1.0));
        expect(sincRes.toList()[1], closeTo(0.0, 1e-6));
        expect(sincRes.toList()[2], equals(0.0));
      },
    );

    test('6. hanning and hamming window copy to dynamic result', () {
      final outDouble = NDArray<Float64>.zeros([4], DType.float64);
      final h = hanning(4, out: outDouble);
      expect(h.shape, equals([4]));
      expect(h.dtype, equals(DType.float64));

      final hm = hamming(4, out: outDouble);
      expect(hm.shape, equals([4]));
      expect(hm.dtype, equals(DType.float64));
    });

    test('7. broadcast and broadcastBinaryStrides with generic typing', () {
      final a = NDArray<Float64>.fromList([1.0, 2.0], [2, 1], DType.float64);
      final b = NDArray<int>.fromList([10, 20, 30], [1, 3], DType.int32);
      final res = broadcast(a, b);
      expect(res.shape, equals([2, 3]));
      expect(res.stridesA, equals([1, 0]));
      expect(res.stridesB, equals([0, 1]));

      final resStrides = broadcastBinaryStrides([2, 1], [1, 1], [1, 3], [3, 1]);
      expect(resStrides.shape, equals([2, 3]));
    });

    test(
      '8. conj with strided complex array and scratch arena marker reset',
      () {
        final cArr = NDArray<Complex>.fromList(
          [
            Complex(1.0, 2.0),
            Complex(0.0, 0.0),
            Complex(3.0, -4.0),
            Complex(0.0, 0.0),
          ],
          [4],
          DType.complex128,
        );
        final cSlice =
            cArr[const Slice(start: 0, stop: 4, step: 2)]
                as NDArray<Complex>; // [1+2i, 3-4i]
        final cConj = conj(cSlice);
        expect(cConj.toList(), equals([Complex(1.0, -2.0), Complex(3.0, 4.0)]));
      },
    );

    test('9. nan_to_num generic typing, non-contiguous views, and complex handling', () {
      final a = NDArray<Float64>.fromList(
        [double.nan, 0.0, double.infinity, 0.0, double.negativeInfinity, 0.0],
        [6],
        DType.float64,
      );
      final aSlice =
          a[const Slice(start: 0, stop: 6, step: 2)]
              as NDArray<Float64>; // [nan, inf, -inf]
      final out = NDArray<Float64>.zeros([3], DType.float64);
      final cleaned = nan_to_num(
        aSlice,
        nan: 99.0,
        posinf: 100.0,
        neginf: -100.0,
        out: out,
      );
      expect(cleaned.toList(), equals([99.0, 100.0, -100.0]));

      final c = NDArray<Complex>.fromList(
        [Complex(double.nan, double.infinity)],
        [1],
        DType.complex128,
      );
      final cCleaned = nan_to_num(c, nan: 0.0, posinf: 50.0);
      expect(cCleaned.toList(), equals([Complex(0.0, 50.0)]));
    });
  });
}
