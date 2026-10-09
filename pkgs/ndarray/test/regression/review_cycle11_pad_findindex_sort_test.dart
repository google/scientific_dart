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
  group('Review Cycle 11 Regression Tests: pad, findIndex, argpartition', () {
    group('1. pad with PaddingMode.constant on float16 and bfloat16', () {
      for (final dtype in [
        DType.float16 as DType<AnySpec>,
        DType.bfloat16 as DType<AnySpec>,
      ]) {
        test('1D constant padding (uniform and per-axis) on $dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0, 3.0], [3], dtype);

            // Uniform constant value
            final paddedUniform = pad(
              a,
              PadWidth.all(2, 1),
              mode: PaddingMode.constant,
              constantValues: PadValues.all(5.5),
            );
            expect(paddedUniform.dtype, dtype);
            expect(paddedUniform.shape, [6]);
            expect(paddedUniform.toList(), [5.5, 5.5, 1.0, 2.0, 3.0, 5.5]);

            // Per-axis (before, after) constant values
            final paddedPerAxis = pad(
              a,
              PadWidth.axes([(1, 2)]),
              mode: PaddingMode.constant,
              constantValues: PadValues.axes([(-2.5, 7.5)]),
            );
            expect(paddedPerAxis.dtype, dtype);
            expect(paddedPerAxis.shape, [6]);
            expect(paddedPerAxis.toList(), [-2.5, 1.0, 2.0, 3.0, 7.5, 7.5]);
          });
        });

        test('2D constant padding (uniform and per-axis) on $dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype);

            // Uniform constant value (hits native_pad_2d with isUniform = 1)
            final paddedUniform = pad(
              a,
              PadWidth.all(1, 1),
              mode: PaddingMode.constant,
              constantValues: PadValues.all(-3.5),
            );
            expect(paddedUniform.dtype, dtype);
            expect(paddedUniform.shape, [4, 4]);
            expect(paddedUniform.toList(), [
              -3.5,
              -3.5,
              -3.5,
              -3.5,
              -3.5,
              1.0,
              2.0,
              -3.5,
              -3.5,
              3.0,
              4.0,
              -3.5,
              -3.5,
              -3.5,
              -3.5,
              -3.5,
            ]);

            // Per-axis constant values (hits native_pad_2d with isUniform = 0)
            final paddedPerAxis = pad(
              a,
              PadWidth.axes([(1, 1), (2, 1)]),
              mode: PaddingMode.constant,
              constantValues: PadValues.axes([(10.0, 20.0), (-1.5, -2.5)]),
            );
            expect(paddedPerAxis.dtype, dtype);
            expect(paddedPerAxis.shape, [4, 5]);
            expect(paddedPerAxis.toList(), [
              -1.5,
              -1.5,
              10.0,
              10.0,
              -2.5,
              -1.5,
              -1.5,
              1.0,
              2.0,
              -2.5,
              -1.5,
              -1.5,
              3.0,
              4.0,
              -2.5,
              -1.5,
              -1.5,
              20.0,
              20.0,
              -2.5,
            ]);
          });
        });

        test('3D constant padding (uniform and per-axis) on $dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [1, 1, 2], dtype);

            // Uniform constant value (hits native_pad_nd with isUniform = 1)
            final paddedUniform = pad(
              a,
              PadWidth.axes([(1, 0), (0, 1), (1, 1)]),
              mode: PaddingMode.constant,
              constantValues: PadValues.all(4.5),
            );
            expect(paddedUniform.dtype, dtype);
            expect(paddedUniform.shape, [2, 2, 4]);
            expect(paddedUniform.toList(), [
              4.5,
              4.5,
              4.5,
              4.5,
              4.5,
              4.5,
              4.5,
              4.5,
              4.5,
              1.0,
              2.0,
              4.5,
              4.5,
              4.5,
              4.5,
              4.5,
            ]);

            // Per-axis constant values (hits native_pad_nd with isUniform = 0)
            final paddedPerAxis = pad(
              a,
              PadWidth.axes([(1, 1), (1, 0), (1, 1)]),
              mode: PaddingMode.constant,
              constantValues: PadValues.axes([
                (9.0, 8.0),
                (7.0, 6.0),
                (-0.5, 0.5),
              ]),
            );
            expect(paddedPerAxis.dtype, dtype);
            expect(paddedPerAxis.shape, [3, 2, 4]);
            expect(paddedPerAxis.getCell([1, 1, 1]), 1.0);
            expect(paddedPerAxis.getCell([1, 1, 2]), 2.0);
            expect(paddedPerAxis.getCell([1, 1, 0]), -0.5);
            expect(paddedPerAxis.getCell([1, 1, 3]), 0.5);
            expect(paddedPerAxis.getCell([1, 0, 1]), 7.0);
            expect(paddedPerAxis.getCell([0, 1, 1]), 9.0);
            expect(paddedPerAxis.getCell([2, 1, 1]), 8.0);
          });
        });
      }
    });

    group('2. findIndex across all comparison operators on extra dtypes', () {
      test('findIndex on int8', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [-10, 0, 15, 20, -5, 30],
            [2, 3],
            DType.int8,
          );

          expect(findIndex<DTypeTag>(a, CompareOp.equal, -5), [1, 1]);
          expect(findIndex<DTypeTag>(a, CompareOp.notEqual, -10), [0, 1]);
          expect(findIndex<DTypeTag>(a, CompareOp.greater, 15), [1, 0]);
          expect(findIndex<DTypeTag>(a, CompareOp.greaterEqual, 15), [0, 2]);
          expect(findIndex<DTypeTag>(a, CompareOp.less, -10), isNull);
          expect(findIndex<DTypeTag>(a, CompareOp.less, -9), [0, 0]);
          expect(findIndex<DTypeTag>(a, CompareOp.lessEqual, -10), [0, 0]);
          expect(
            findIndex<DTypeTag>(a, CompareOp.greater, 10, directions: [-1, -1]),
            [1, 2],
          );
        });
      });

      for (final dtype in [
        DType.uint16 as DType<AnySpec>,
        DType.uint32 as DType<AnySpec>,
        DType.uint64 as DType<AnySpec>,
      ]) {
        test('findIndex on $dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [10, 200, 300, 400, 50, 600],
              [2, 3],
              dtype,
            );

            expect(findIndex<DTypeTag>(a, CompareOp.equal, 400), [1, 0]);
            expect(findIndex<DTypeTag>(a, CompareOp.equal, 999), isNull);
            expect(findIndex<DTypeTag>(a, CompareOp.notEqual, 10), [0, 1]);
            expect(findIndex<DTypeTag>(a, CompareOp.greater, 300), [1, 0]);
            expect(findIndex<DTypeTag>(a, CompareOp.greaterEqual, 300), [0, 2]);
            expect(findIndex<DTypeTag>(a, CompareOp.less, 50), [0, 0]);
            expect(
              findIndex<DTypeTag>(
                a,
                CompareOp.lessEqual,
                50,
                startCoords: [0, 1],
              ),
              [1, 1],
            );
          });
        });
      }

      for (final dtype in [
        DType.float16 as DType<AnySpec>,
        DType.bfloat16 as DType<AnySpec>,
      ]) {
        test('findIndex on $dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [-2.5, 0.5, 1.5, 3.5, -1.0, 4.0],
              [2, 3],
              dtype,
            );

            expect(findIndex<DTypeTag>(a, CompareOp.equal, -1.0), [1, 1]);
            expect(findIndex<DTypeTag>(a, CompareOp.equal, 99.0), isNull);
            expect(findIndex<DTypeTag>(a, CompareOp.notEqual, -2.5), [0, 1]);
            expect(findIndex<DTypeTag>(a, CompareOp.greater, 1.5), [1, 0]);
            expect(findIndex<DTypeTag>(a, CompareOp.greaterEqual, 1.5), [0, 2]);
            expect(findIndex<DTypeTag>(a, CompareOp.less, -1.0), [0, 0]);
            expect(
              findIndex<DTypeTag>(
                a,
                CompareOp.lessEqual,
                -1.0,
                startCoords: [0, 1],
              ),
              [1, 1],
            );
          });
        });
      }
    });

    group('3. argpartition with out == null, DType.int64, and DType.int32', () {
      void verifyPartition1D<T extends num>(
        List<T> values,
        List<int> indices,
        int kth,
      ) {
        final pivot = values[indices[kth]];
        for (var i = 0; i < kth; i++) {
          expect(values[indices[i]], lessThanOrEqualTo(pivot));
        }
        for (var i = kth + 1; i < indices.length; i++) {
          expect(values[indices[i]], greaterThanOrEqualTo(pivot));
        }
      }

      for (final dtype in [
        DType.float16 as DType<AnySpec>,
        DType.bfloat16 as DType<AnySpec>,
      ]) {
        test('argpartition on $dtype (default, int32 out, int64 out)', () {
          NDArray.scope(() {
            final data = [3.5, -1.5, 4.0, 0.5, 2.0];
            final a = NDArray.fromList(data, [5], dtype);

            // Default out == null
            final resDefault = argpartition(a, 2);
            expect(resDefault.dtype, DType.int64);
            verifyPartition1D(data, resDefault.toList().cast<int>(), 2);

            // out with DType.int32 via argpartitionAs
            final out32 = NDArray.zeros([5], DType.int32);
            final ret32 = argpartitionAs(a, 2, DType.int32, out: out32);
            expect(identical(ret32, out32), isTrue);
            expect(out32.dtype, DType.int32);
            verifyPartition1D(data, out32.toList().cast<int>(), 2);

            // out with DType.int64
            final out64 = NDArray.zeros([5], DType.int64);
            final ret64 = argpartition(a, 2, out: out64);
            expect(identical(ret64, out64), isTrue);
            expect(out64.dtype, DType.int64);
            verifyPartition1D(data, out64.toList().cast<int>(), 2);
          });
        });
      }

      for (final dtype in [
        DType.int8 as DType<AnySpec>,
        DType.uint16 as DType<AnySpec>,
        DType.uint32 as DType<AnySpec>,
      ]) {
        test('argpartition on $dtype (default, int32 out, int64 out)', () {
          NDArray.scope(() {
            final data = dtype == DType.int8
                ? [30, -10, 40, 5, 20]
                : [300, 10, 400, 50, 200];
            final a = NDArray.fromList(data, [5], dtype);

            // Default out == null
            final resDefault = argpartition(a, 2);
            expect(resDefault.dtype, DType.int64);
            verifyPartition1D(data, resDefault.toList().cast<int>(), 2);

            // out with DType.int32 via argpartitionAs
            final out32 = NDArray.zeros([5], DType.int32);
            final ret32 = argpartitionAs(a, 2, DType.int32, out: out32);
            expect(identical(ret32, out32), isTrue);
            expect(out32.dtype, DType.int32);
            verifyPartition1D(data, out32.toList().cast<int>(), 2);

            // out with DType.int64
            final out64 = NDArray.zeros([5], DType.int64);
            final ret64 = argpartition(a, 2, out: out64);
            expect(identical(ret64, out64), isTrue);
            expect(out64.dtype, DType.int64);
            verifyPartition1D(data, out64.toList().cast<int>(), 2);
          });
        });
      }
    });
  });
}
