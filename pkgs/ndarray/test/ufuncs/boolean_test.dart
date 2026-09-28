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
  group('NDArray Boolean Data Type (DType.boolean) Tests', () {
    test(
      'Create empty boolean array via zeros factory',
      () => NDArray.scope(() {
        final arr = NDArray<bool>.zeros([2, 3], DType.boolean);

        expect(arr.shape, [2, 3]);
        expect(arr.dtype, DType.boolean);
        expect(arr.size, 6);

        // Check true default zeros allocation (false)
        expect(arr.toList(), [false, false, false, false, false, false]);
      }),
    );

    test(
      'Mutating boolean values via BoolList indexing operators',
      () => NDArray.scope(() {
        final arr = NDArray<bool>.zeros([4], DType.boolean);

        arr[[0]] = true;
        arr[[2]] = true;

        expect(arr[[0]], true);
        expect(arr[[1]], false);
        expect(arr[[2]], true);
        expect(arr[[3]], false);

        expect(arr.toList(), [true, false, true, false]);
      }),
    );

    test(
      'Create boolean array filled with true via ones factory',
      () => NDArray.scope(() {
        final arr = NDArray<bool>.ones([5], DType.boolean);

        expect(arr.shape, [5]);
        expect(arr.dtype, DType.boolean);
        expect(arr.toList(), [true, true, true, true, true]);
      }),
    );

    test(
      'Boolean fromList contiguous block copying',
      () => NDArray.scope(() {
        final input = [true, false, false, true];
        final arr = NDArray.fromList(input, [2, 2], DType.boolean);

        expect(arr.shape, [2, 2]);
        expect(arr.dtype, DType.boolean);
        expect(arr.toList(), [true, false, false, true]);
      }),
    );

    test(
      'Zero-copy boolean sliced views memory sharing',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );

        // Slice a view on the latter elements (index 2 to 4 -> [true, false])
        final view = parent.slice([Slice(start: 2, stop: 4)]);

        expect(view.shape, [2]);
        expect(view.dtype, DType.boolean);
        expect(view.toList(), [true, false]);

        // Mutate via the view and check if parent reflects it instantly!
        view[[1]] = true; // view -> [true, true]

        expect(parent.toList(), [true, false, true, true]);
        // Success! Confirms zero-copy C heap pointer sharing works flawlessly on boolean bytes!
      }),
    );

    test(
      'Boolean array arithmetic (addition) falls through to uint8',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, false, true], [3], DType.boolean);
        final b = NDArray.fromList([true, true, false], [3], DType.boolean);

        final result = add(a, b);

        expect(result.dtype, DType.uint8);
        expect(result.toList(), [2, 1, 1]);
      }),
    );
  });

  group('Boolean argmin/argmax bug repro', () {
    test('argmax/argmin on boolean array', () {
      NDArray.scope(() {
        final a = NDArray.fromList([false, true, false], [3], DType.boolean);
        final maxIdx = argmax(a);
        expect(maxIdx.scalar, 1);
        final minIdx = argmin(a);
        expect(minIdx.scalar, 0);
        final a2D = NDArray.fromList(
          [false, true, false, false],
          [2, 2],
          DType.boolean,
        );
        final max2D = argmax(a2D, axis: 1);
        expect(max2D.toList(), [1, 0]);
      });
    });
  });

  group('Boolean cumsum bug repro', () {
    test('cumsum on boolean array', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [true, false, true, true],
          [4],
          DType.boolean,
        );
        final result = cumsum(a);
        expect(result.dtype, DType.int32);
        expect(result.shape, [4]);
        expect(result.toList(), [1, 1, 2, 3]);
        final a2D = NDArray.fromList(
          [true, false, true, true],
          [2, 2],
          DType.boolean,
        );
        final result2D = cumsum(a2D, axis: 0);
        expect(result2D.dtype, DType.int32);
        expect(result2D.shape, [2, 2]);
        expect(result2D.toList(), [1, 0, 2, 1]);
      });
    });
  });
}
