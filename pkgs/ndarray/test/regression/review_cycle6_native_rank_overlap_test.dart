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
  group(
    'Review Cycle 6: High-Rank (rank = 34 & rank = 32) Strided Ops via DECLARE_RANK_BUFFER',
    () {
      test(
        'rank = 34 non-contiguous strided unary, binary, cast, clip, and reduction succeed accurately',
        () {
          NDArray.scope(() {
            // Shape of length 34: [4, 1, ..., 1, 2] (total 8 elements)
            final fullShape34 = <int>[4, ...List<int>.filled(32, 1), 2];
            final full = NDArray<Float64>.fromList(
              <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
              fullShape34,
              DType.float64,
            );
            expect(full.shape.length, equals(34));

            // Slice along axis 0 with step 2 -> shape [2, 1, ..., 1, 2] (elements [1.0, 2.0, 5.0, 6.0])
            final a34 = full.slice([Slice(start: 0, stop: 4, step: 2)]);
            // Slice along axis 0 starting at 1 with step 2 -> shape [2, 1, ..., 1, 2] (elements [3.0, 4.0, 7.0, 8.0])
            final b34 = full.slice([Slice(start: 1, stop: 4, step: 2)]);

            expect(a34.shape.length, equals(34));
            expect(a34.isContiguous, isFalse);
            expect(b34.shape.length, equals(34));
            expect(b34.isContiguous, isFalse);

            // Unary strided ops
            final neg = negative(a34);
            expect(neg.shape.length, equals(34));
            expect(neg.flatten().toList(), equals([-1.0, -2.0, -5.0, -6.0]));

            final sinVal = sin(a34);
            expect(sinVal.shape.length, equals(34));
            expect(sinVal.getCellFlat(0), closeTo(math.sin(1.0), 1e-12));
            expect(sinVal.getCellFlat(1), closeTo(math.sin(2.0), 1e-12));
            expect(sinVal.getCellFlat(2), closeTo(math.sin(5.0), 1e-12));
            expect(sinVal.getCellFlat(3), closeTo(math.sin(6.0), 1e-12));

            // Binary strided ops
            final sumArr = add(a34, b34);
            expect(sumArr.shape.length, equals(34));
            expect(sumArr.flatten().toList(), equals([4.0, 6.0, 12.0, 14.0]));

            final diffArr = subtract(b34, a34);
            expect(diffArr.shape.length, equals(34));
            expect(diffArr.flatten().toList(), equals([2.0, 2.0, 2.0, 2.0]));

            // Cast strided op
            final casted = a34.astype<Int32>(DType.int32);
            expect(casted.shape.length, equals(34));
            expect(casted.flatten().toList(), equals([1, 2, 5, 6]));

            // Clip strided op
            final clipped = clip(a34, min: 1.5, max: 5.5);
            expect(clipped.shape.length, equals(34));
            expect(clipped.flatten().toList(), equals([1.5, 2.0, 5.0, 5.5]));

            // Axis reduction on rank-34 strided view
            final reduced = sum(a34, axis: 0);
            expect(reduced.shape.length, equals(33));
            expect(reduced.flatten().toList(), equals([6.0, 8.0]));
          });
        },
      );

      test(
        'rank = 32 strided unary, binary, cast, clip, and reduction succeed',
        () {
          NDArray.scope(() {
            final base = NDArray<Float64>.fromList(
              <double>[1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final shape32 = <int>[2, ...List<int>.filled(31, 1)];
            final strides32 = <int>[2, ...List<int>.filled(31, 1)];
            // Non-contiguous strided view with step 2 along axis 0 -> elements [1.0, 3.0]
            final a32 = NDArray<Float64>.view(
              base,
              shape: shape32,
              strides: strides32,
              offsetElements: 0,
            );
            // Non-contiguous strided view with step 2 starting at index 1 -> elements [2.0, 4.0]
            final b32 = NDArray<Float64>.view(
              base,
              shape: shape32,
              strides: strides32,
              offsetElements: 1,
            );

            expect(a32.shape.length, equals(32));
            expect(a32.isContiguous, isFalse);

            // Unary strided ops
            final neg = negative(a32);
            expect(neg.shape.length, equals(32));
            expect(neg.flatten().toList(), equals([-1.0, -3.0]));

            final sinVal = sin(a32);
            expect(sinVal.shape.length, equals(32));
            expect(sinVal.getCellFlat(0), closeTo(math.sin(1.0), 1e-12));
            expect(sinVal.getCellFlat(1), closeTo(math.sin(3.0), 1e-12));

            // Binary strided ops
            final sumArr = add(a32, b32);
            expect(sumArr.shape.length, equals(32));
            expect(sumArr.flatten().toList(), equals([3.0, 7.0]));

            final diffArr = subtract(b32, a32);
            expect(diffArr.shape.length, equals(32));
            expect(diffArr.flatten().toList(), equals([1.0, 1.0]));

            // Cast strided op
            final casted = a32.astype<Int32>(DType.int32);
            expect(casted.shape.length, equals(32));
            expect(casted.flatten().toList(), equals([1, 3]));

            // Clip strided op
            final clipped = clip(a32, min: 1.5, max: 2.5);
            expect(clipped.shape.length, equals(32));
            expect(clipped.flatten().toList(), equals([1.5, 2.5]));

            // Axis reduction on rank-32 strided view
            final reduced = sum(a32, axis: 0);
            expect(reduced.shape.length, equals(31));
            expect(reduced.flatten().toList(), equals([4.0]));
          });
        },
      );
    },
  );

  group(
    'Review Cycle 6: Float64 Strided Binary Overlap Protection (s_add/sub/mul/div_double)',
    () {
      test(
        'partially overlapping offset slices with out: produce uncorrupted results',
        () {
          NDArray.scope(() {
            final orig = <double>[10.0, 20.0, 30.0, 40.0, 50.0];

            // 1. add: a[1:5] = a[0:4] + a[1:5]
            final aAdd = NDArray<Float64>.fromList(orig, [5], DType.float64);
            final aAdd04 = aAdd[Slice(start: 0, stop: 4)] as NDArray<Float64>;
            final aAdd15 = aAdd[Slice(start: 1, stop: 5)] as NDArray<Float64>;
            final expectedAdd = add(aAdd04, aAdd15).toList();
            add(aAdd04, aAdd15, out: aAdd15);
            expect(aAdd15.toList(), equals(expectedAdd));
            expect(aAdd.toList(), equals([10.0, 30.0, 50.0, 70.0, 90.0]));

            // 2. subtract: a[1:5] = a[0:4] - a[1:5]
            final aSub = NDArray<Float64>.fromList(orig, [5], DType.float64);
            final aSub04 = aSub[Slice(start: 0, stop: 4)] as NDArray<Float64>;
            final aSub15 = aSub[Slice(start: 1, stop: 5)] as NDArray<Float64>;
            final expectedSub = subtract(aSub04, aSub15).toList();
            subtract(aSub04, aSub15, out: aSub15);
            expect(aSub15.toList(), equals(expectedSub));
            expect(aSub.toList(), equals([10.0, -10.0, -10.0, -10.0, -10.0]));

            // 3. multiply: a[1:5] = a[0:4] * a[1:5]
            final aMul = NDArray<Float64>.fromList(
              <double>[1.0, 2.0, 3.0, 4.0, 5.0],
              [5],
              DType.float64,
            );
            final aMul04 = aMul[Slice(start: 0, stop: 4)] as NDArray<Float64>;
            final aMul15 = aMul[Slice(start: 1, stop: 5)] as NDArray<Float64>;
            final expectedMul = multiply(aMul04, aMul15).toList();
            multiply(aMul04, aMul15, out: aMul15);
            expect(aMul15.toList(), equals(expectedMul));
            expect(aMul.toList(), equals([1.0, 2.0, 6.0, 12.0, 20.0]));

            // 4. divide: a[1:5] = a[0:4] / a[1:5]
            final aDiv = NDArray<Float64>.fromList(
              <double>[2.0, 4.0, 8.0, 16.0, 32.0],
              [5],
              DType.float64,
            );
            final aDiv04 = aDiv[Slice(start: 0, stop: 4)] as NDArray<Float64>;
            final aDiv15 = aDiv[Slice(start: 1, stop: 5)] as NDArray<Float64>;
            final expectedDiv = divide(aDiv04, aDiv15).toList();
            divide(aDiv04, aDiv15, out: aDiv15);
            expect(aDiv15.toList(), equals(expectedDiv));
            expect(aDiv.toList(), equals([2.0, 0.5, 0.5, 0.5, 0.5]));
          });
        },
      );

      test(
        'reverse-stride overlapping views (Slice(step: -1)) with out: produce uncorrupted results',
        () {
          NDArray.scope(() {
            final orig = <double>[1.0, 2.0, 3.0, 4.0, 5.0];
            final b = NDArray<Float64>.fromList(
              <double>[10.0, 20.0, 30.0, 40.0, 50.0],
              [5],
              DType.float64,
            );

            // Add into reversed view of a: a[::-1] = a + b
            final a = NDArray<Float64>.fromList(orig, [5], DType.float64);
            final aRev = a[Slice(step: -1)] as NDArray<Float64>;
            final expected = add(a, b).toList();
            add(a, b, out: aRev);
            expect(aRev.toList(), equals(expected));
            expect(a.toList(), equals(expected.reversed.toList()));

            // Subtract reversed view of a from b into a: a = a[::-1] - b
            final a2 = NDArray<Float64>.fromList(orig, [5], DType.float64);
            final a2Rev = a2[Slice(step: -1)] as NDArray<Float64>;
            final expected2 = subtract(a2Rev, b).toList();
            subtract(a2Rev, b, out: a2);
            expect(a2.toList(), equals(expected2));
          });
        },
      );
    },
  );
}
