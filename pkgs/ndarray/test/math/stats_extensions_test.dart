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
  group('ptp tests', () {
    test('ptp flat contiguous axis=null', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 5.0, 3.0, 10.0, 2.0],
          [5],
          DType.float64,
        );
        final res = ptp(a);
        expect(res.shape, <int>[]);
        expect(res.dtype, DType.float64);
        expect(res.scalar, 9.0);
      });
    });

    test('ptp 2D contiguous axis=null', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 6.0, 4.0], [2, 2], DType.float64);
        final p = ptp(a);
        expect(p.shape, <int>[]);
        expect(p.dtype, DType.float64);
        expect(p.scalar, 5.0);
      });
    });

    test('ptp 2D contiguous axis=0', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final res = ptp(a, axis: 0);
        expect(res.shape, [3]);
        expect(res.dtype, DType.float64);
        expect(res.toList(), [3.0, 3.0, 3.0]);
      });
    });

    test('ptp 2D contiguous axis=1', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final res = ptp(a, axis: 1);
        expect(res.shape, [2]);
        expect(res.dtype, DType.float64);
        expect(res.toList(), [2.0, 2.0]);
      });
    });

    test('ptp strided (non-contiguous) array', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final transposed = a.transpose(); // shape [2, 3], non-contiguous
        // transposed is:
        // [[1.0, 3.0, 5.0],
        //  [2.0, 4.0, 6.0]]

        final res = ptp(transposed, axis: 1); // ptp along rows
        expect(res.shape, [2]);
        expect(res.toList(), [4.0, 4.0]); // (5-1), (6-2)
      });
    });

    test('ptp with out parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 5.0, 2.0, 10.0], [4], DType.float64);
        final out = NDArray<double>.zeros([], DType.float64);
        final res = ptp(a, out: out);
        expect(identical(res, out), true);
        expect(out.scalar, 9.0);
      });
    });

    test('ptp integer types', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 5, 2, 10], [4], DType.int32);
        final res = ptp(a);
        expect(res.dtype, DType.int32);
        expect(res.scalar, 9);
      });
    });

    test('ptp validation', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        expect(() => ptp(a), throwsStateError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final out = NDArray<double>.zeros([
          2,
        ], DType.float64); // incompatible shape, should be []
        expect(() => ptp(a, out: out), throwsArgumentError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final out = NDArray<double>.zeros(
          [],
          DType.float32,
        ); // incompatible dtype
        expect(() => ptp(a, out: out), throwsArgumentError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 6.0, 4.0], [2, 2], DType.float64);
        expect(() => ptp(a, axis: 3), throwsArgumentError);
        expect(() => ptp(a, axis: -3), throwsArgumentError);
      });
    });

    test('ptp negative axis', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 6.0, 4.0], [2, 2], DType.float64);
        final p = ptp(a, axis: -1); // equivalent to axis 1
        expect(p.shape, [2]);
        expect(p.toList(), [1.0, 2.0]);
      });
    });
  });

  group('average tests', () {
    test('average without weights (same as mean)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final res = average(a);
        expect(res.average.shape, <int>[]);
        expect(res.average.scalar, 2.5);
        expect(res.sumOfWeights, null);

        final a2 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final resAxis = average(a2, axis: 0);
        expect(resAxis.average.shape, [2]);
        expect(resAxis.average.toList(), [2.0, 3.0]);
        expect(resAxis.sumOfWeights, null);
      });
    });

    test('average with 1D weights axis=0 (1D input)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final res = average(a, weights: w, returned: true);

        // weighted sum = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
        // sum of weights = 1 + 2 + 3 + 4 = 10
        // avg = 30 / 10 = 3.0
        expect(res.average.shape, <int>[]);
        expect(res.average.scalar, 3.0);
        expect(res.sumOfWeights, isNotNull);
        expect(res.sumOfWeights!.shape, <int>[]);
        expect(res.sumOfWeights!.scalar, 10.0);
      });
    });

    test('average 1D weights, axis specified (2D input)', () {
      NDArray.scope(() {
        // [[1, 2],
        //  [3, 4]]
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        // Axis 0: weights apply to rows.
        // (1*1 + 3*2)/(1+2) = 7/3 = 2.333...
        // (2*1 + 4*2)/(1+2) = 10/3 = 3.333...
        final res0 = average(a, axis: 0, weights: w, returned: true);
        expect(res0.average.shape, [2]);
        expect(res0.average.toList()[0], closeTo(7 / 3, 1e-9));
        expect(res0.average.toList()[1], closeTo(10 / 3, 1e-9));
        expect(res0.sumOfWeights, isNotNull);
        expect(res0.sumOfWeights!.shape, [2]); // Broadcasted to average shape
        expect(res0.sumOfWeights!.toList(), [3.0, 3.0]);

        // Axis 1: weights apply to columns.
        // (1*1 + 2*2)/(1+2) = 5/3 = 1.666...
        // (3*1 + 4*2)/(1+2) = 11/3 = 3.666...
        final res1 = average(a, axis: 1, weights: w, returned: true);
        expect(res1.average.shape, [2]);
        expect(res1.average.toList()[0], closeTo(5 / 3, 1e-9));
        expect(res1.average.toList()[1], closeTo(11 / 3, 1e-9));
        expect(res1.sumOfWeights, isNotNull);
        expect(res1.sumOfWeights!.shape, [2]); // Broadcasted to average shape
        expect(res1.sumOfWeights!.toList(), [3.0, 3.0]);
      });
    });

    test('average 1D weights, axis=null (only allowed if a is 1D)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

        final res = average(a, weights: w);
        expect(res.average.shape, <int>[]);
        expect(res.average.scalar, closeTo(14 / 6, 1e-9));

        final a2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(() => average(a2d, weights: w), throwsArgumentError);
      });
    });

    test('average with 1D weights axis=1 (2D input [2, 3])', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final w = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final res = average(a, axis: 1, weights: w, returned: true);

        // Row 1: (1*1 + 2*2 + 3*3) / 6 = 14 / 6 = 2.333...
        // Row 2: (4*1 + 5*2 + 6*3) / 6 = 32 / 6 = 5.333...
        expect(res.average.shape, [2]);
        expect(res.average.toList()[0], closeTo(2.3333333333333335, 1e-9));
        expect(res.average.toList()[1], closeTo(5.333333333333333, 1e-9));
        expect(res.sumOfWeights, isNotNull);
        expect(res.sumOfWeights!.shape, [2]);
        expect(res.sumOfWeights!.toList(), [6.0, 6.0]);
      });
    });

    test('average with ND weights', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = average(a, weights: w, returned: true); // global average
        expect(res.average.shape, <int>[]);
        expect(res.average.scalar, 3.0);
        expect(res.sumOfWeights!.scalar, 10.0);
      });
    });

    test('average with ND weights along axis=0', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = average(a, axis: 0, weights: w, returned: true);
        expect(res.average.shape, [2]);
        expect(res.average.toList()[0], 2.5);
        expect(res.average.toList()[1], closeTo(3.3333333333333335, 1e-9));
        expect(res.sumOfWeights!.toList(), [4.0, 6.0]);
      });
    });

    test('average with out parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final out = NDArray<double>.zeros([], DType.float64);
        final res = average(a, weights: w, out: out);
        expect(identical(res.average, out), true);
        expect(out.scalar, 3.0);
      });
    });

    test('average integer types promotion', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final w = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final res = average<int, int, double>(a, weights: w, returned: true);

        expect(res.average.dtype, DType.float64);
        expect(res.average.scalar, 3.0);
        expect(res.sumOfWeights!.dtype, DType.float64);
        expect(res.sumOfWeights!.scalar, 10.0);
      });
    });

    test('average validation', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        expect(() => average(a), throwsStateError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        expect(() => average(a, weights: w), throwsArgumentError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => average(a, weights: w), throwsArgumentError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => average(a, axis: 2), throwsArgumentError);
        expect(() => average(a, axis: -2), throwsArgumentError);
      });
    });

    test('average negative axis', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final res = average(
          a,
          axis: -2,
          weights: w,
          returned: true,
        ); // axis -2 is axis 0
        expect(res.average.shape, [2]);
        expect(res.average.toList()[0], closeTo(7 / 3, 1e-9));
        expect(res.average.toList()[1], closeTo(10 / 3, 1e-9));
        expect(res.sumOfWeights!.toList(), [3.0, 3.0]);
      });
    });
  });
}
