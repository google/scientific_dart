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

import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Review Cycle 12 - Core Stream 1 Tests', () {
    test(
      '_sliceAssign on 0-sized arrays with step < 0 does not crash',
      () => NDArray.scope(() {
        final arr = NDArray.zeros([0, 3], DType.float64);
        // Slice with negative step on 0-sized dimension should not throw clamp(0, -1) error
        arr[[const Slice(step: -1), const Slice.all()]] = 10.0;
        expect(arr.shape, equals([0, 3]));

        final arr1d = NDArray.zeros([0], DType.float64);
        arr1d[[const Slice(step: -1)]] = 5.0;
        expect(arr1d.shape, equals([0]));
      }),
    );

    test(
      '_sliceAssign with negative step exact bounds and dimSize',
      () => NDArray.scope(() {
        final arr = NDArray.arange(0, 10, dtype: DType.float64);
        // Reverse assign elements from 8 down to 2 with step -2: indices 8, 6, 4, 2
        final vals = NDArray.fromList(
          Float64List.fromList([100.0, 200.0, 300.0, 400.0]),
          [4],
          DType.float64,
        );
        arr[[const Slice(start: 8, stop: 1, step: -2)]] = vals;
        expect(
          arr.toList(),
          equals([0.0, 1.0, 400.0, 3.0, 300.0, 5.0, 200.0, 7.0, 100.0, 9.0]),
        );
      }),
    );

    test(
      'Advanced slice() with step < 0 clamping bounds',
      () => NDArray.scope(() {
        final arr = NDArray.arange(0, 12, dtype: DType.float64).reshape([3, 4]);
        // Mix negative slice step with Indices selector
        final view = arr.slice([
          const Slice(start: 2, step: -1), // rows 2 down to 0
          Indices([3, 1]),
        ]);
        expect(view.shape, equals([3, 2]));
        expect(view.toList(), equals([11.0, 9.0, 7.0, 5.0, 3.0, 1.0]));
      }),
    );

    test(
      'setByMask accurately reads elements of sliced view values',
      () => NDArray.scope(() {
        final target = NDArray.zeros([4], DType.float64);
        final mask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final source = NDArray.fromList(
          Float64List.fromList([10.0, 20.0, 30.0, 40.0]),
          [4],
          DType.float64,
        );
        // View with step 2: contains [10.0, 30.0], backing data has 4 elements
        final slicedSource = source.slice([const Slice(step: 2)]);

        target.setByMask(mask, slicedSource);
        expect(target.toList(), equals([10.0, 0.0, 30.0, 0.0]));
      }),
    );

    test(
      'setIndices and setIndicesScalar accurately read sliced views',
      () => NDArray.scope(() {
        final target = NDArray.zeros([5], DType.float64);
        final rawIndices = NDArray.fromList(Int32List.fromList([4, 1, 3, 0]), [
          4,
        ], DType.int32);
        // Sliced indices view: [4, 3]
        final slicedIndices = rawIndices.slice([const Slice(step: 2)]);

        // Test setIndicesScalar with sliced indices view
        target.setIndicesScalar(slicedIndices, 99.0);
        expect(target.toList(), equals([0.0, 0.0, 0.0, 99.0, 99.0]));

        // Test setIndices with sliced indices AND sliced values
        final rawValues = NDArray.fromList(
          Float64List.fromList([7.0, 8.0, 9.0, 10.0]),
          [4],
          DType.float64,
        );
        final slicedValues = rawValues.slice([
          const Slice(start: 1, step: 2),
        ]); // [8.0, 10.0]

        target.setIndices(slicedIndices, slicedValues);
        expect(target.toList(), equals([0.0, 0.0, 0.0, 10.0, 8.0]));
      }),
    );

    test(
      '_coerceScalar and _sliceAssign detect 1-element scalar views',
      () => NDArray.scope(() {
        final arr = NDArray.zeros([3], DType.float64);
        final source = NDArray.fromList(
          Float64List.fromList([1.0, 42.0, 3.0]),
          [3],
          DType.float64,
        );
        // 1-element view where backing data.length is 3, size is 1
        final scalarView = source.slice([const Slice(start: 1, stop: 2)]);

        // _sliceAssign with scalar view
        arr[[const Slice(start: 0, stop: 2)]] = scalarView;
        expect(arr.toList(), equals([42.0, 42.0, 0.0]));
      }),
    );
  });
}
