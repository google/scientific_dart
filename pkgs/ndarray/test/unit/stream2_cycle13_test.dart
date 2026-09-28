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
  group('Stream 2 Cycle 13 Remediation Tests', () {
    test('1. ternaryOp Physical vs Logical Offset Indexing', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [6],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
          [6],
          DType.float64,
        );
        final cond = NDArray<Boolean>.fromList(
          [true, true, false, false, true, true],
          [6],
          DType.boolean,
        );

        // Slice with stride 2
        final aSlice = a.slice([const Slice(start: 0, stop: 6, step: 2)]);
        final bSlice = b.slice([const Slice(start: 0, stop: 6, step: 2)]);
        final condSlice = cond.slice([const Slice(start: 0, stop: 6, step: 2)]);

        final res = where(condSlice, aSlice, bSlice);
        expect(res.toList(), equals([1.0, 30.0, 5.0]));
      });
    });

    test('2. clipArray without scalar bound data[0] write', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [-10.0, 0.0, 10.0],
          [3],
          DType.float64,
        );
        final clippedMinOnly = clip(a, min: 0.0);
        expect(clippedMinOnly.toList(), equals([0.0, 0.0, 10.0]));

        final clippedMaxOnly = clip(a, max: 5.0);
        expect(clippedMaxOnly.toList(), equals([-10.0, 0.0, 5.0]));
      });
    });

    test(
      '3. atUfunc missing C cases (floorDivide, remainder, power) & unsupported op',
      () {
        NDArray.scope(() {
          final aInt = NDArray.fromList([10, 20, 30], [3], DType.int32);
          final idx = NDArray.fromList([0, 1, 2], [3], DType.int64);
          final bInt = NDArray.fromList([3, 3, 2], [3], DType.int32);

          atUfunc(aInt, idx, bInt, op: BinaryOp.floorDivide);
          expect(aInt.toList(), equals([3, 6, 15]));

          final aIntMod = NDArray.fromList([10, 20, 30], [3], DType.int32);
          atUfunc(aIntMod, idx, bInt, op: BinaryOp.remainder);
          expect(aIntMod.toList(), equals([1, 2, 0]));

          final aIntPow = NDArray.fromList([2, 3, 4], [3], DType.int32);
          final bPow = NDArray.fromList([3, 2, 1], [3], DType.int32);
          atUfunc(aIntPow, idx, bPow, op: BinaryOp.power);
          expect(aIntPow.toList(), equals([8, 9, 4]));

          // Float cases
          final aFloat = NDArray<Float64>.fromList(
            [10.5, 20.5, 30.5],
            [3],
            DType.float64,
          );
          final bFloat = NDArray<Float64>.fromList(
            [3.0, 4.0, 5.0],
            [3],
            DType.float64,
          );
          atUfunc(aFloat, idx, bFloat, op: BinaryOp.floorDivide);
          expect(aFloat.toList(), equals([3.0, 5.0, 6.0]));

          // Unsupported binary op
          expect(
            () => atUfunc(aFloat, idx, bFloat, op: BinaryOp.gcd),
            throwsA(isA<UnsupportedError>()),
          );
        });
      },
    );

    test('4. gamma and erf switch DType fast paths', () {
      NDArray.scope(() {
        final aDouble = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final gDouble = gamma(aDouble);
        expect(gDouble.dtype, equals(DType.float64));

        final aFloat = NDArray<Float32>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float32,
        );
        final gFloat = gamma(aFloat);
        expect(gFloat.dtype, equals(DType.float32));

        final eDouble = erf(aDouble);
        expect(eDouble.dtype, equals(DType.float64));

        final eFloat = erf(aFloat);
        expect(eFloat.dtype, equals(DType.float32));
      });
    });
  });
}
