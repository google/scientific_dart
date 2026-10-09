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
  group('Issue #1: Empty Axis Padding Safety', () {
    test('pad throws ArgumentError on empty axis for non-constant modes', () {
      NDArray.scope(() {
        final empty1D = NDArray<Float64>.zeros([0], DType.float64);
        final empty2D = NDArray<Float64>.zeros([0, 3], DType.float64);

        final nonConstantModes = [
          PadMode.edge,
          PadMode.reflect,
          PadMode.symmetric,
          PadMode.wrap,
          PadMode.linearRamp,
          PadMode.max,
          PadMode.mean,
          PadMode.median,
          PadMode.min,
        ];

        for (final mode in nonConstantModes) {
          expect(
            () => pad(empty1D, PadWidth.all(1), mode: mode),
            throwsArgumentError,
            reason: 'Expected ArgumentError for 1D [0] with mode $mode',
          );
          expect(
            () => pad(empty2D, PadWidth.axes([(1, 0), (0, 0)]), mode: mode),
            throwsArgumentError,
            reason:
                'Expected ArgumentError for 2D [0, 3] padded on axis 0 with mode $mode',
          );
        }
      });
    });

    test(
      'pad allows unpadded empty axis when only non-empty axis is padded',
      () {
        NDArray.scope(() {
          final emptyAxis0 = NDArray<Float64>.zeros([0, 3], DType.float64);
          final paddedEdge = pad(
            emptyAxis0,
            PadWidth.axes([(0, 0), (1, 2)]),
            mode: PadMode.edge,
          );
          expect(paddedEdge.shape, equals([0, 6]));
          expect(paddedEdge.size, equals(0));

          final paddedMedian = pad(
            emptyAxis0,
            PadWidth.axes([(0, 0), (1, 2)]),
            mode: PadMode.median,
          );
          expect(paddedMedian.shape, equals([0, 6]));
        });
      },
    );

    test(
      'pad with PadMode.constant on empty axes writes constant values in axis priority order',
      () {
        NDArray.scope(() {
          final empty1D = NDArray<Float64>.zeros([0], DType.float64);
          final res1D = pad(
            empty1D,
            PadWidth.all(2, 3),
            mode: PadMode.constant,
            constantValues: PadValues.all(7.0, 9.0),
          );
          expect(res1D.shape, equals([5]));
          expect([
            for (var i = 0; i < 5; i++) res1D[[i]].toDouble(),
          ], equals([7.0, 7.0, 9.0, 9.0, 9.0]));

          // 2D [0, 2] padded with [(1, 1), (1, 1)] and constants [(10, 11), (20, 21)]:
          // Axis 0 pads [0, 2] -> [2, 2] with row 0 = [10, 10], row 1 = [11, 11].
          // Axis 1 pads [2, 2] -> [2, 4] with col 0 = 20, col 3 = 21.
          final empty2D = NDArray<Float64>.zeros([0, 2], DType.float64);
          final res2D = pad(
            empty2D,
            PadWidth.all(1),
            mode: PadMode.constant,
            constantValues: PadValues.axes([(10.0, 11.0), (20.0, 21.0)]),
          );
          expect(res2D.shape, equals([2, 4]));
          expect([
            for (var c = 0; c < 4; c++) res2D[[0, c]].toDouble(),
          ], equals([20.0, 10.0, 10.0, 21.0]));
          expect([
            for (var c = 0; c < 4; c++) res2D[[1, c]].toDouble(),
          ], equals([20.0, 11.0, 11.0, 21.0]));
        });
      },
    );
  });

  group('Issue #4: 3D/ND Non-Uniform Constant Padding Axis Priority Order', () {
    test(
      '3D [1, 1, 1] with non-uniform PadValues matches sequential axis 0 -> 1 -> 2 order',
      () {
        NDArray.scope(() {
          final src = NDArray<Float64>.fromList(
            [99.0],
            [1, 1, 1],
            DType.float64,
          );
          final padValues = PadValues<Float64>.axes([
            (10.0, 11.0),
            (20.0, 21.0),
            (30.0, 31.0),
          ]);

          final actual = pad(
            src,
            PadWidth.all(1),
            mode: PadMode.constant,
            constantValues: padValues,
          );

          // Compute expected by padding one axis at a time in 0 -> 1 -> 2 order
          final step0 = pad(
            src,
            PadWidth.axes([(1, 1), (0, 0), (0, 0)]),
            mode: PadMode.constant,
            constantValues: PadValues.axes([
              (10.0, 11.0),
              (0.0, 0.0),
              (0.0, 0.0),
            ]),
          );
          final step1 = pad(
            step0,
            PadWidth.axes([(0, 0), (1, 1), (0, 0)]),
            mode: PadMode.constant,
            constantValues: PadValues.axes([
              (0.0, 0.0),
              (20.0, 21.0),
              (0.0, 0.0),
            ]),
          );
          final expected = pad(
            step1,
            PadWidth.axes([(0, 0), (0, 0), (1, 1)]),
            mode: PadMode.constant,
            constantValues: PadValues.axes([
              (0.0, 0.0),
              (0.0, 0.0),
              (30.0, 31.0),
            ]),
          );

          expect(actual.shape, equals([3, 3, 3]));
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 3; j++) {
              for (var k = 0; k < 3; k++) {
                expect(
                  actual[[i, j, k]].toDouble(),
                  equals(expected[[i, j, k]].toDouble()),
                  reason: 'Mismatch at [$i, $j, $k]',
                );
              }
            }
          }
        });
      },
    );

    test(
      '3D [2, 3, 2] with PadValues.all(1.0, 2.0) matches sequential axis order',
      () {
        NDArray.scope(() {
          final src = NDArray<Float64>.fromList(
            [for (var i = 0; i < 12; i++) i.toDouble()],
            [2, 3, 2],
            DType.float64,
          );
          final actual = pad(
            src,
            PadWidth.axes([(1, 2), (2, 1), (1, 1)]),
            mode: PadMode.constant,
            constantValues: PadValues.all(1.0, 2.0),
          );

          final step0 = pad(
            src,
            PadWidth.axes([(1, 2), (0, 0), (0, 0)]),
            mode: PadMode.constant,
            constantValues: PadValues.all(1.0, 2.0),
          );
          final step1 = pad(
            step0,
            PadWidth.axes([(0, 0), (2, 1), (0, 0)]),
            mode: PadMode.constant,
            constantValues: PadValues.all(1.0, 2.0),
          );
          final expected = pad(
            step1,
            PadWidth.axes([(0, 0), (0, 0), (1, 1)]),
            mode: PadMode.constant,
            constantValues: PadValues.all(1.0, 2.0),
          );

          expect(actual.shape, equals(expected.shape));
          for (var i = 0; i < actual.shape[0]; i++) {
            for (var j = 0; j < actual.shape[1]; j++) {
              for (var k = 0; k < actual.shape[2]; k++) {
                expect(
                  actual[[i, j, k]].toDouble(),
                  equals(expected[[i, j, k]].toDouble()),
                  reason: 'Mismatch at [$i, $j, $k]',
                );
              }
            }
          }
        });
      },
    );
  });

  group('Issue #2: reduceatUfunc Index Bounds Validation & Normalization', () {
    test(
      'reduceat throws RangeError when indices is non-empty on an empty axis',
      () {
        NDArray.scope(() {
          final empty1D = NDArray<Float64>.zeros([0], DType.float64);
          final indices = NDArray.fromList([0], [1], DType.int64);
          expect(
            () => empty1D.reduceat(indices, op: BinaryOp.add),
            throwsRangeError,
          );

          final emptyAxis1 = NDArray<Float64>.zeros([2, 0], DType.float64);
          expect(
            () => emptyAxis1.reduceat(indices, op: BinaryOp.add, axis: 1),
            throwsRangeError,
          );
        });
      },
    );

    test(
      'reduceat throws RangeError on out-of-bounds positive or negative indices',
      () {
        NDArray.scope(() {
          final a1D = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          final oobPos = NDArray.fromList([0, 4], [2], DType.int64);
          final oobNeg = NDArray.fromList([-5, 1], [2], DType.int64);

          expect(
            () => a1D.reduceat(oobPos, op: BinaryOp.add),
            throwsRangeError,
          );
          expect(
            () => a1D.reduceat(oobNeg, op: BinaryOp.add),
            throwsRangeError,
          );

          final a2D = a1D.reshape([2, 2]);
          final oob2DPos = NDArray.fromList([0, 2], [2], DType.int64);
          final oob2DNeg = NDArray.fromList([-3, 0], [2], DType.int64);
          expect(
            () => a2D.reduceat(oob2DPos, op: BinaryOp.add, axis: 1),
            throwsRangeError,
          );
          expect(
            () => a2D.reduceat(oob2DNeg, op: BinaryOp.add, axis: 0),
            throwsRangeError,
          );
        });
      },
    );

    test(
      'reduceat normalizes valid negative indices without mutating indices',
      () {
        NDArray.scope(() {
          final a1D = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          final negIndices = NDArray.fromList([-4, -2], [2], DType.int64);
          final posIndices = NDArray.fromList([1, 3], [2], DType.int64);

          final resNeg = a1D.reduceat(negIndices, op: BinaryOp.add);
          final resPos = a1D.reduceat(posIndices, op: BinaryOp.add);

          expect(
            [
              resNeg[[0]].toDouble(),
              resNeg[[1]].toDouble(),
            ],
            equals([
              resPos[[0]].toDouble(),
              resPos[[1]].toDouble(),
            ]),
          );
          // Ensure negIndices was not mutated in-place
          expect([
            negIndices[[0]],
            negIndices[[1]],
          ], equals([-4, -2]));
        });
      },
    );
  });

  group('Issue #3: atUfunc DType Coercion, Broadcasting, Bounds & Aliasing', () {
    test('at throws RangeError on empty axis 0 or out-of-bounds indices', () {
      NDArray.scope(() {
        final empty = NDArray<Float64>.zeros([0], DType.float64);
        final idx = NDArray.fromList([0], [1], DType.int64);
        final b = NDArray<Float64>.fromList([1.0], [1], DType.float64);
        expect(() => empty.at(idx, b, op: BinaryOp.add), throwsRangeError);

        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        expect(
          () => a.at(
            NDArray.fromList([3], [1], DType.int64),
            b,
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );
        expect(
          () => a.at(
            NDArray.fromList([-4], [1], DType.int64),
            b,
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );
      });
    });

    test('at coerces b when b.dtype != a.dtype', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final idx = NDArray.fromList([1, 3], [2], DType.int64);
        final bInt32 = NDArray.fromList([10, 20], [2], DType.int32);

        a.at(idx, bInt32, op: BinaryOp.add);
        expect([
          for (var i = 0; i < 4; i++) a[[i]].toDouble(),
        ], equals([1.0, 12.0, 3.0, 24.0]));
      });
    });

    test('at validates and broadcasts b to indexed target shape', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.zeros([4, 3], DType.float64);
        final idx = NDArray.fromList([1, 3], [2], DType.int64);

        // Incompatible shape [2, 2] for expected [2, 3] throws ArgumentError
        final badB = NDArray<Float64>.ones([2, 2], DType.float64);
        expect(() => a.at(idx, badB, op: BinaryOp.add), throwsArgumentError);

        // Broadcast [1, 3] across 2 indexed rows
        final rowB = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [1, 3],
          DType.float64,
        );
        a.at(idx, rowB, op: BinaryOp.add);
        expect([
          for (var c = 0; c < 3; c++) a[[1, c]].toDouble(),
        ], equals([1.0, 2.0, 3.0]));
        expect([
          for (var c = 0; c < 3; c++) a[[3, c]].toDouble(),
        ], equals([1.0, 2.0, 3.0]));

        // Broadcast [2, 1] across 3 columns
        final colB = NDArray<Float64>.fromList(
          [10.0, 20.0],
          [2, 1],
          DType.float64,
        );
        a.at(idx, colB, op: BinaryOp.add);
        expect([
          for (var c = 0; c < 3; c++) a[[1, c]].toDouble(),
        ], equals([11.0, 12.0, 13.0]));
        expect([
          for (var c = 0; c < 3; c++) a[[3, c]].toDouble(),
        ], equals([21.0, 22.0, 23.0]));
      });
    });

    test('at protects against memory aliasing when b shares memory with a', () {
      NDArray.scope(() {
        // Initial a = [1, 2, 3]
        // Indices = [1, 2, 0]
        // b = a (initial snapshot [1, 2, 3]):
        // Step 0: a[1] += b[0] (1) => a[1] becomes 2 + 1 = 3
        // Step 1: a[2] += b[1] (snapshot 2, NOT mutated 3!) => a[2] becomes 3 + 2 = 5
        // Step 2: a[0] += b[2] (snapshot 3, NOT mutated 5!) => a[0] becomes 1 + 3 = 4
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final idx = NDArray.fromList([1, 2, 0], [3], DType.int64);
        a.at(idx, a, op: BinaryOp.add);
        expect([
          for (var i = 0; i < 3; i++) a[[i]].toDouble(),
        ], equals([4.0, 3.0, 5.0]));
      });
    });
  });
}
