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

List<int> rep(dynamic v) {
  if (v is int) return [v];
  if (v is List<int>) return v;
  throw ArgumentError();
}

void main() {
  group('Shape Manipulation Core Tests', () {
    group('Shape Manipulation Tests', () {
      group('expandDims Tests', () {
        test(
          'Basic expandDims at front (axis 0)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = a.expandDims(0);
            expect(b.shape, [1, 2, 2]);
            expect(b.strides, [2, 2, 1]); // Strides are correct for [1, 2, 2]
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

            // Verify it shares memory (view behavior)
            b.data[0] = 99.0;
            expect(a.data[0], 99.0);
          }),
        );

        test(
          'expandDims in the middle (axis 1)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = a.expandDims(1);
            expect(b.shape, [2, 1, 2]);
            expect(b.strides, [2, 1, 1]);
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);
          }),
        );

        test(
          'expandDims at the end (axis 2)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = a.expandDims(2);
            expect(b.shape, [2, 2, 1]);
            expect(b.strides, [2, 1, 1]);
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);
          }),
        );

        test(
          'expandDims negative axis (-1)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // -1 refers to the position right after the last dimension (equivalent to axis 2)
            final b = a.expandDims(-1);
            expect(b.shape, [2, 2, 1]);
          }),
        );

        test(
          'expandDims negative axis (-3)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // -3 refers to the very front (equivalent to axis 0)
            final b = a.expandDims(-3);
            expect(b.shape, [1, 2, 2]);
          }),
        );

        test(
          'expandDims invalid axis throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => a.expandDims(3), throwsRangeError);
            expect(() => a.expandDims(-4), throwsRangeError);
          }),
        );
      });

      group('squeeze Tests', () {
        test(
          'Squeeze all size 1 axes',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze();
            expect(b.shape, [2]);
            expect(b.strides, [1]);
            expect(b.toList(), [1.0, 2.0]);

            // View verification
            b.data[0] = 42.0;
            expect(a.data[0], 42.0);
          }),
        );

        test(
          'Squeeze specific int axis',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze(axis: 0);
            expect(b.shape, [2, 1]);
            expect(b.strides, [1, 1]);
          }),
        );

        test(
          'Squeeze specific negative axis',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze(axis: -1);
            expect(b.shape, [1, 2]);
            expect(b.strides, [2, 1]);
          }),
        );

        test(
          'Squeeze specific List<int> axes',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze(axis: [0, 2]);
            expect(b.shape, [2]);
            expect(b.strides, [1]);
          }),
        );

        test(
          'Squeeze axis with size > 1 throws',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            expect(() => a.squeeze(axis: 1), throwsArgumentError);
          }),
        );

        test(
          'Squeeze invalid axis range throws',
          () => NDArray.scope(() {
            final a = NDArray.create([1, 2], DType.float64);
            expect(() => a.squeeze(axis: 2), throwsRangeError);
            expect(() => a.squeeze(axis: -3), throwsRangeError);
          }),
        );

        test(
          'Squeeze 1x1 array to 0-D scalar array',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([5.5]), [
              1,
              1,
            ], DType.float64);
            final b = a.squeeze();
            expect(b.shape, isEmpty);
            expect(b.strides, isEmpty);
            expect(b[[]], 5.5);
          }),
        );
      });

      group('swapaxes Tests', () {
        test(
          'Basic swapaxes in 2D',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );
            // shape: [2, 3], strides: [3, 1]
            final b = a.swapaxes(0, 1);
            expect(b.shape, [3, 2]);
            expect(b.strides, [1, 3]);
            // Original elements:
            // Row 0: 1, 2, 3
            // Row 1: 4, 5, 6
            // Swapped matrix:
            // [1, 4]
            // [2, 5]
            // [3, 6]
            expect(b.toList(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

            // View verification
            b.data[0] = 11.0; // updates a.data[0]
            expect(a.data[0], 11.0);
          }),
        );

        test(
          'swapaxes negative index',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            final b = a.swapaxes(-3, -1); // swaps 0 and 2
            expect(b.shape, [4, 3, 2]);
          }),
        );

        test(
          'swapaxes same axis is no-op view',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = a.swapaxes(0, 0);
            expect(b.shape, [2]);
            expect(b.toList(), [1.0, 2.0]);
          }),
        );

        test(
          'swapaxes out of bounds throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => a.swapaxes(0, 2), throwsRangeError);
            expect(() => a.swapaxes(-3, 1), throwsRangeError);
          }),
        );
      });

      group('moveaxis Tests', () {
        test(
          'Move single axis 0 to 2 in 3D',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            final b = a.moveaxis(0, 2);
            expect(b.shape, [3, 4, 2]);
            // original strides were [12, 4, 1]
            // order becomes [1, 2, 0], so strides should be [4, 1, 12]
            expect(b.strides, [4, 1, 12]);
          }),
        );

        test(
          'Move single axis negative indexes (-1 to 0)',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            final b = a.moveaxis(-1, 0); // moves axis 2 to 0
            expect(b.shape, [4, 2, 3]);
            expect(b.strides, [1, 12, 4]);
          }),
        );

        test(
          'Move multiple axes lists',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            // Move axis 0 -> 1 and axis 1 -> 2
            final b = a.moveaxis([0, 1], [1, 2]);
            expect(b.shape, [
              4,
              2,
              3,
            ]); // axis 2 becomes 0, axis 0 becomes 1, axis 1 becomes 2
          }),
        );

        test(
          'moveaxis invalid lengths or duplicates throw',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            expect(() => a.moveaxis([0, 1], [2]), throwsArgumentError);
            expect(() => a.moveaxis([0, 0], [1, 2]), throwsArgumentError);
            expect(() => a.moveaxis([0, 1], [2, 2]), throwsArgumentError);
          }),
        );
      });

      group('tile Tests', () {
        test(
          'Tile 1D array with int scalar reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = tile(a, rep(3));
            expect(b.shape, [6]);
            expect(b.toList(), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

            // Verify it allocates new memory (copy behavior)
            b.data[0] = 99.0;
            expect(a.data[0], 1.0);
          }),
        );

        test(
          'Tile 1D array with 1D List reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = tile(a, rep([2]));
            expect(b.shape, [4]);
            expect(b.toList(), [1.0, 2.0, 1.0, 2.0]);
          }),
        );

        test(
          'Tile 1D array with higher-rank 2D List reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = tile(
              a,
              rep([2, 3]),
            ); // shape promoted from [2] to [1, 2], target [2*1, 3*2] = [2, 6]
            expect(b.shape, [2, 6]);
            expect(b.toList(), [
              1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 0
              1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 1
            ]);
          }),
        );

        test(
          'Tile 2D array with list reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = tile(
              a,
              rep([2, 1]),
            ); // repeat rows 2 times, columns 1 time
            expect(b.shape, [4, 2]);
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
          }),
        );

        test(
          'Tile 3D array',
          () => NDArray.scope(() {
            final a = NDArray<Int32>.arange(
              0,
              6,
              dtype: DType.int32,
            ).reshape([1, 2, 3]);
            final b = tile(a, [2, 2, 2]);
            expect(b.shape, [2, 4, 6]);
            expect(b[[0, 0, 0]], 0);
            expect(b[[0, 0, 3]], 0);
            expect(b[[0, 2, 0]], 0);
            expect(b[[1, 0, 0]], 0);
            expect(b[[0, 1, 2]], 5);
            expect(b[[1, 3, 5]], 5);
          }),
        );

        test(
          'Tile with strided non-contiguous source (transposed and sliced)',
          () => NDArray.scope(() {
            final a = NDArray<Float64>.arange(
              0,
              12,
              dtype: DType.float64,
            ).reshape([3, 4]);
            final transposed = a.transposed; // shape [4, 3], strided
            expect(transposed.isContiguous, false);

            final tiled = tile(transposed, [2, 3]);
            expect(tiled.shape, [8, 9]);
            for (var r = 0; r < 8; r++) {
              for (var c = 0; c < 9; c++) {
                expect(tiled[[r, c]], transposed[[r % 4, c % 3]]);
              }
            }
          }),
        );

        test(
          'Tile with explicit out buffer',
          () => NDArray.scope(() {
            final a = NDArray<Float32>.fromList(
              Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float32,
            );
            final out = NDArray<Float32>.zeros([4, 4], DType.float32);
            final result = tile(a, [2, 2], out: out);
            expect(identical(result, out), true);
            expect(out.shape, [4, 4]);
            expect(out[[0, 0]], 1.0);
            expect(out[[0, 2]], 1.0);
            expect(out[[2, 0]], 1.0);
            expect(out[[2, 2]], 1.0);
            expect(out[[3, 3]], 4.0);
          }),
        );

        test(
          'Tile all DTypes',
          () => NDArray.scope(() {
            // Int64
            final i64 = NDArray<Int64>.fromList([10, 20], [2], DType.int64);
            expect(tile(i64, [3]).toList(), [10, 20, 10, 20, 10, 20]);

            // Int16
            final i16 = NDArray<Int16>.fromList([1, 2], [2], DType.int16);
            expect(tile(i16, [2]).toList(), [1, 2, 1, 2]);

            // Uint8
            final u8 = NDArray<Uint8>.fromList([255, 0], [2], DType.uint8);
            expect(tile(u8, [2]).toList(), [255, 0, 255, 0]);

            // Boolean
            final b = NDArray<Boolean>.fromList(
              [true, false],
              [2],
              DType.boolean,
            );
            expect(tile(b, [2]).toList(), [true, false, true, false]);

            // Complex128
            final c128 = NDArray<Complex128>.fromList(
              [Complex(1.0, 2.0), Complex(3.0, 4.0)],
              [2],
              DType.complex128,
            );
            final tiledC128 = tile(c128, [2]);
            expect(tiledC128.shape, [4]);
            expect(tiledC128[[0]], Complex(1.0, 2.0));
            expect(tiledC128[[1]], Complex(3.0, 4.0));
            expect(tiledC128[[2]], Complex(1.0, 2.0));
            expect(tiledC128[[3]], Complex(3.0, 4.0));

            // Complex64
            final c64 = NDArray<Complex64>.fromList(
              [Complex(1.0, 2.0), Complex(3.0, 4.0)],
              [2],
              DType.complex64,
            );
            final tiledC64 = tile(c64, [2]);
            expect(tiledC64.shape, [4]);
            expect(tiledC64[[0]], Complex(1.0, 2.0));
            expect(tiledC64[[1]], Complex(3.0, 4.0));
            expect(tiledC64[[2]], Complex(1.0, 2.0));
            expect(tiledC64[[3]], Complex(3.0, 4.0));
          }),
        );

        test(
          'Tile 0D scalar array',
          () => NDArray.scope(() {
            final s = NDArray<Float64>.scalar(42.0, dtype: DType.float64);
            final t0 = tile(s, []);
            expect(t0.shape, isEmpty);
            expect(t0.scalar, 42.0);

            final t2d = tile(s, [2, 3]);
            expect(t2d.shape, [2, 3]);
            expect(t2d.toList(), [42.0, 42.0, 42.0, 42.0, 42.0, 42.0]);
          }),
        );

        test(
          'Tile with 0 in reps or 0-sized array',
          () => NDArray.scope(() {
            final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
            final t0 = tile(a, [0]);
            expect(t0.shape, [0]);
            expect(t0.size, 0);

            final empty = NDArray<Float64>.create([0, 5], DType.float64);
            final tEmpty = tile(empty, [3, 2]);
            expect(tEmpty.shape, [0, 10]);
            expect(tEmpty.size, 0);
          }),
        );

        test('Tile with disposed array throws StateError', () {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          a.dispose();
          expect(() => tile(a, [2]), throwsStateError);

          final a2 = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final out = NDArray<Float64>.create([4], DType.float64);
          out.dispose();
          expect(() => tile(a2, [2], out: out), throwsStateError);
          a2.dispose();
        });

        test(
          'Tile with negative or invalid reps throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2], DType.float64);
            expect(() => tile(a, rep(-1)), throwsArgumentError);
            expect(() => tile(a, rep([2, -3])), throwsArgumentError);
            expect(
              () => tile(a, 'invalid' as dynamic),
              throwsA(isA<TypeError>()),
            );
            expect(() => tile(a, 3.5 as dynamic), throwsA(isA<TypeError>()));
          }),
        );
      });

      group('repeat Tests', () {
        test(
          'Repeat elements of flat 1D array with scalar int',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
              3,
            ], DType.float64);
            final b = repeat(a, rep(2));
            expect(b.shape, [6]);
            expect(b.toList(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

            // Copy verification
            b.data[0] = 99.0;
            expect(a.data[0], 1.0);
          }),
        );

        test(
          'Repeat automatically flattens high-rank array if axis is null',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = repeat(a, rep(2)); // axis = null
            expect(b.shape, [8]);
            expect(b.toList(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
          }),
        );

        test(
          'Repeat along axis 0 with scalar int',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = repeat(a, rep(2), axis: 0);
            expect(b.shape, [4, 2]);
            expect(b.toList(), [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
          }),
        );

        test(
          'Repeat along axis 1 with custom List<int> counts',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // Repeat column 0 -> 2 times, column 1 -> 1 time. total cols = 3
            final b = repeat(a, rep([2, 1]), axis: 1);
            expect(b.shape, [2, 3]);
            expect(b.toList(), [
              1.0, 1.0, 2.0, // row 0
              3.0, 3.0, 4.0, // row 1
            ]);
          }),
        );

        test(
          'Repeat with an entry of 0 counts (deletes slice)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // Repeat column 0 -> 0 times, column 1 -> 2 times.
            final b = repeat(a, rep([0, 2]), axis: 1);
            expect(b.shape, [2, 2]);
            expect(b.toList(), [2.0, 2.0, 4.0, 4.0]);
          }),
        );

        test(
          'Repeat mismatched list counts throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(
              () => repeat(a, rep([1, 2, 3]), axis: 1),
              throwsArgumentError,
            );
          }),
        );

        test(
          'Repeat negative counts throws',
          () => NDArray.scope(() {
            final a = NDArray.create([3], DType.float64);
            expect(() => repeat(a, rep(-2)), throwsArgumentError);
            expect(() => repeat(a, rep([1, -1, 0])), throwsArgumentError);
          }),
        );

        test(
          'Repeat invalid axis throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => repeat(a, rep(2), axis: 2), throwsRangeError);
          }),
        );
      });

      group('NDArray deep duplicate copy() Tests', () {
        test(
          'copy() contiguous float64 array',
          () => NDArray.scope(() {
            final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
            final b = a.copy();

            expect(b.shape, [3]);
            expect(b.dtype, DType.float64);
            expect(b.toList(), [10.0, 20.0, 30.0]);

            b.data[0] = 99.0;
            expect(a.data[0], 10.0);
          }),
        );

        test(
          'copy() non-contiguous transposed strided float64 array view',
          () => NDArray.scope(() {
            final parent = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );

            final view = parent.transposed;
            expect(view.isContiguous, false);

            final b = view.copy();
            expect(b.shape, [2, 2]);
            expect(b.isContiguous, true);
            expect(b.toList(), [1.0, 3.0, 2.0, 4.0]);

            b.data[0] = 99.0;
            expect(parent.data[0], 1.0);
          }),
        );

        test(
          'disposed array throws StateError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            a.dispose();
            expect(() => a.copy(), throwsStateError);
          }),
        );

        test(
          'top-level copy() contiguous float64 array',
          () => NDArray.scope(() {
            final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
            final b = copy(a);

            expect(b.shape, [3]);
            expect(b.dtype, DType.float64);
            expect(b.toList(), [10.0, 20.0, 30.0]);

            b.data[0] = 99.0;
            expect(a.data[0], 10.0);
          }),
        );

        test(
          'top-level copy() disposed array throws StateError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            a.dispose();
            expect(() => copy(a), throwsStateError);
          }),
        );
      });
    });

    group('Rearranging Tests', () {
      group('flip Tests', () {
        test(
          'Basic flip 1D Float64',
          () => NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              <double>[1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final flipped = flip(a);
            expect(flipped.shape, [4]);
            expect(flipped.toList(), [4.0, 3.0, 2.0, 1.0]);

            // Verify it is a zero-copy view
            flipped.setCell([0], 99.0);
            expect(a.getCell([3]), 99.0);
          }),
        );

        test(
          'Flip 2D along specific axes',
          () => NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );

            // Flip all axes
            final flippedAll = flip(a);
            expect(flippedAll.toList(), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

            // Flip axis 0 only
            final flipped0 = flip(a, axis: 0);
            expect(flipped0.toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

            // Flip axis 1 only
            final flipped1 = flip(a, axis: 1);
            expect(flipped1.toList(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);

            // Flip negative axis
            final flippedNeg = flip(a, axis: -1);
            expect(flippedNeg.toList(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
          }),
        );

        test(
          'Flip 3D with list of axes',
          () => NDArray.scope(() {
            final a = NDArray.fromList(List.generate(8, (i) => i + 1), [
              2,
              2,
              2,
            ], DType.int32);
            // original:
            // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
            // Flip axis 0 and 2
            final flipped = flip(a, axis: [0, 2]);
            // expected:
            // [[[6, 5], [8, 7]], [[2, 1], [4, 3]]]
            expect(flipped.toList(), [6, 5, 8, 7, 2, 1, 4, 3]);
          }),
        );

        test(
          'Flip invalid axis throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => flip(a, axis: 2), throwsRangeError);
            expect(() => flip(a, axis: -3), throwsRangeError);
            expect(() => flip(a, axis: 'invalid'), throwsArgumentError);
          }),
        );
      });

      group('fliplr and flipud Tests', () {
        test(
          'fliplr 2D Int32',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final flipped = fliplr(a);
            expect(flipped.toList(), [2, 1, 4, 3]);
          }),
        );

        test(
          'fliplr invalid dimensions throws',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            expect(() => fliplr(a), throwsArgumentError);
          }),
        );

        test(
          'flipud 2D Int32',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final flipped = flipud(a);
            expect(flipped.toList(), [3, 4, 1, 2]);
          }),
        );

        test(
          'flipud invalid dimensions throws',
          () => NDArray.scope(() {
            final a = NDArray.scalar(5, dtype: DType.int32);
            expect(() => flipud(a), throwsArgumentError);
          }),
        );
      });

      group('roll Tests', () {
        test(
          'Basic roll 1D flat',
          () => NDArray.scope(() {
            final a = NDArray<Int32>.fromList(
              [10, 20, 30, 40, 50],
              [5],
              DType.int32,
            );

            // Roll positive shift
            final rolledPos = roll(a, 2);
            expect(rolledPos.toList(), [40, 50, 10, 20, 30]);

            // Roll negative shift
            final rolledNeg = roll(a, -2);
            expect(rolledNeg.toList(), [30, 40, 50, 10, 20]);

            // Roll larger than size
            final rolledLarge = roll(a, 7); // 7 % 5 = 2
            expect(rolledLarge.toList(), [40, 50, 10, 20, 30]);

            // Verify it is a copy (independent memory)
            rolledPos.setCell([0], 99);
            expect(a.getCell([0]), 10);
          }),
        );

        test(
          'Roll 2D with specific axis',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1, 2, 3, 4, 5, 6, 7, 8, 9],
              [3, 3],
              DType.int32,
            );

            // Roll axis null flattens first
            final rolledFlat = roll(a, 2);
            expect(rolledFlat.shape, [3, 3]);
            expect(rolledFlat.toList(), [8, 9, 1, 2, 3, 4, 5, 6, 7]);

            // Roll axis 0
            final rolledAxis0 = roll(a, 1, axis: 0);
            expect(rolledAxis0.toList(), [7, 8, 9, 1, 2, 3, 4, 5, 6]);

            // Roll axis 1
            final rolledAxis1 = roll(a, 1, axis: 1);
            expect(rolledAxis1.toList(), [3, 1, 2, 6, 4, 5, 9, 7, 8]);
          }),
        );

        test(
          'Roll with multiple shifts and axes',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1, 2, 3, 4, 5, 6, 7, 8, 9],
              [3, 3],
              DType.int32,
            );

            // Roll list shift and list axis
            final rolledList = roll(a, [1, 1], axis: [0, 1]);
            // axis 0 shift 1: [7, 8, 9, 1, 2, 3, 4, 5, 6]
            // axis 1 shift 1: [9, 7, 8, 3, 1, 2, 6, 4, 5]
            expect(rolledList.toList(), [9, 7, 8, 3, 1, 2, 6, 4, 5]);

            // Roll single shift with multiple axes
            final rolledSingleShift = roll(a, 1, axis: [0, 1]);
            expect(rolledSingleShift.toList(), [9, 7, 8, 3, 1, 2, 6, 4, 5]);
          }),
        );

        test(
          'Roll invalid parameters throw',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => roll(a, [1], axis: [0, 1]), throwsArgumentError);
            expect(() => roll(a, [1, 2], axis: [0]), throwsArgumentError);
            expect(() => roll(a, 'invalid', axis: 0), throwsArgumentError);
            expect(() => roll(a, 1, axis: 'invalid'), throwsArgumentError);
            expect(() => roll(a, 1, axis: 2), throwsRangeError);
          }),
        );

        test(
          'Roll with all DTypes',
          () => NDArray.scope(() {
            // float64
            final f64 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            expect(roll(f64, 1, axis: 0).toList(), [3.0, 4.0, 1.0, 2.0]);
            expect(roll(f64, 1, axis: 1).toList(), [2.0, 1.0, 4.0, 3.0]);

            // float32
            final f32 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float32,
            );
            expect(roll(f32, 1, axis: 0).toList(), [3.0, 4.0, 1.0, 2.0]);

            // int64
            final i64 = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int64);
            expect(roll(i64, 1, axis: 0).toList(), [30, 40, 10, 20]);

            // int16
            final i16 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int16);
            expect(roll(i16, 1, axis: 0).toList(), [3, 4, 1, 2]);

            // uint8
            final u8 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
            expect(roll(u8, 1, axis: 0).toList(), [3, 4, 1, 2]);

            // boolean
            final b = NDArray.fromList(
              [true, false, true, true],
              [2, 2],
              DType.boolean,
            );
            expect(roll(b, 1, axis: 0).toList(), [true, true, true, false]);

            // complex128
            final c128 = NDArray.fromList(
              [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
              [2, 2],
              DType.complex128,
            );
            final rolledC128 = roll(c128, 1, axis: 0);
            expect(rolledC128.getCell([0, 0]), Complex(5, 6));
            expect(rolledC128.getCell([0, 1]), Complex(7, 8));
            expect(rolledC128.getCell([1, 0]), Complex(1, 2));
            expect(rolledC128.getCell([1, 1]), Complex(3, 4));

            // complex64
            final c64 = NDArray.fromList(
              [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
              [2, 2],
              DType.complex64,
            );
            final rolledC64 = roll(c64, 1, axis: 1);
            expect(rolledC64.getCell([0, 0]), Complex(3, 4));
            expect(rolledC64.getCell([0, 1]), Complex(1, 2));
          }),
        );

        test(
          'Roll with non-contiguous strided input',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1, 2, 3, 4, 5, 6, 7, 8, 9],
              [3, 3],
              DType.int32,
            );
            final transposed = a.transpose(); // non-contiguous view
            expect(transposed.isContiguous, isFalse);

            // Transposed values:
            // [[1, 4, 7],
            //  [2, 5, 8],
            //  [3, 6, 9]]
            final rolledAx0 = roll(transposed, 1, axis: 0);
            expect(rolledAx0.toList(), [3, 6, 9, 1, 4, 7, 2, 5, 8]);

            final rolledAx1 = roll(transposed, 1, axis: 1);
            expect(rolledAx1.toList(), [7, 1, 4, 8, 2, 5, 9, 3, 6]);

            // Roll flat on non-contiguous view
            final rolledFlat = roll(transposed, 1);
            expect(rolledFlat.toList(), [9, 1, 4, 7, 2, 5, 8, 3, 6]);
          }),
        );

        test(
          'Roll with out parameter and in-place',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4, 5], [5], DType.int32);
            final out = NDArray<Int32>.create([5], DType.int32);

            final res = roll(a, 2, out: out);
            expect(identical(res, out), isTrue);
            expect(out.toList(), [4, 5, 1, 2, 3]);

            // In-place roll (out: a)
            roll(a, 2, out: a);
            expect(a.toList(), [4, 5, 1, 2, 3]);

            // Incompatible out throws
            final badOut = NDArray<Int32>.create([4], DType.int32);
            expect(() => roll(a, 1, out: badOut), throwsArgumentError);
          }),
        );

        test(
          'Roll 3D array along different axes',
          () => NDArray.scope(() {
            // Shape [2, 3, 2]
            final a = NDArray.fromList(List.generate(12, (i) => i + 1), [
              2,
              3,
              2,
            ], DType.int32);

            // Roll along axis 0
            final r0 = roll(a, 1, axis: 0);
            expect(r0.shape, [2, 3, 2]);
            expect(r0.getCell([0, 0, 0]), 7);
            expect(r0.getCell([1, 0, 0]), 1);

            // Roll along axis 1
            final r1 = roll(a, 1, axis: 1);
            expect(r1.shape, [2, 3, 2]);
            expect(r1.getCell([0, 0, 0]), 5);
            expect(r1.getCell([0, 1, 0]), 1);

            // Roll along axis 2 (negative index -1)
            final r2 = roll(a, 1, axis: -1);
            expect(r2.shape, [2, 3, 2]);
            expect(r2.getCell([0, 0, 0]), 2);
            expect(r2.getCell([0, 0, 1]), 1);
          }),
        );
      });

      group('Extension methods Tests', () {
        test(
          'All rearranging extensions',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);

            expect(flip(a).toList(), [4, 3, 2, 1]);
            expect(fliplr(a).toList(), [2, 1, 4, 3]);
            expect(flipud(a).toList(), [3, 4, 1, 2]);
            expect(roll(a, 1, axis: 0).toList(), [3, 4, 1, 2]);
          }),
        );
      });
    });

    group('Splitting Tests', () {
      group('split & array_split Tests', () {
        test('basic 1D equal split', () {
          NDArray.scope(() {
            final a = NDArray<Int32>.fromList([1, 2, 3, 4], [4], DType.int32);
            final splits = split(a, 2);

            expect(splits.length, 2);
            expect(splits[0].shape, [2]);
            expect(splits[0].toList(), [1, 2]);
            expect(splits[1].shape, [2]);
            expect(splits[1].toList(), [3, 4]);

            // Modifying sub-array view affects original
            splits[0].setCell([0], 99);
            expect(a.getCell([0]), 99);
          });
        });

        test('split unequal division throws', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            expect(() => split(a, 2), throwsArgumentError);
          });
        });

        test('array_split unequal division splits as equal as possible', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final splits = array_split(a, 2);

            expect(splits.length, 2);
            expect(splits[0].shape, [2]);
            expect(splits[0].toList(), [1, 2]);
            expect(splits[1].shape, [1]);
            expect(splits[1].toList(), [3]);
          });
        });

        test('split_at with list of indices', () {
          NDArray.scope(() {
            final a = NDArray<Int32>.fromList(
              [10, 20, 30, 40, 50],
              [5],
              DType.int32,
            );
            final splits = split_at(a, [1, 3]);

            expect(splits.length, 3);
            expect(splits[0].toList(), [10]);
            expect(splits[1].toList(), [20, 30]);
            expect(splits[2].toList(), [40, 50]);
          });
        });

        test('split along specific axis', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1, 2, 3, 4, 5, 6, 7, 8],
              [2, 4],
              DType.int32,
            );

            // Split columns (axis 1) into 2 equal parts
            final splits = split(a, 2, axis: 1);
            expect(splits.length, 2);
            expect(splits[0].toList(), [1, 2, 5, 6]);
            expect(splits[1].toList(), [3, 4, 7, 8]);
          });
        });
      });

      group('hsplit & hsplit_at Tests', () {
        test('hsplit 2D array', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final splits = hsplit(a, 2);

            expect(splits.length, 2);
            expect(splits[0].toList(), [1, 3]);
            expect(splits[1].toList(), [2, 4]);
          });
        });

        test('hsplit_at 2D array', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final splits = hsplit_at(a, [1]);

            expect(splits.length, 2);
            expect(splits[0].toList(), [1, 3]);
            expect(splits[1].toList(), [2, 4]);
          });
        });

        test('hsplit 1D array splits along axis 0', () {
          NDArray.scope(() {
            final a = NDArray<Int32>.fromList([1, 2, 3, 4], [4], DType.int32);
            final splits = hsplit(a, 2);

            expect(splits.length, 2);
            expect(splits[0].toList(), [1, 2]);
            expect(splits[1].toList(), [3, 4]);
          });
        });
      });

      group('vsplit & vsplit_at Tests', () {
        test('vsplit 2D array', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final splits = vsplit(a, 2);

            expect(splits.length, 2);
            expect(splits[0].toList(), [1, 2]);
            expect(splits[1].toList(), [3, 4]);
          });
        });

        test('vsplit_at 2D array', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final splits = vsplit_at(a, [1]);

            expect(splits.length, 2);
            expect(splits[0].toList(), [1, 2]);
            expect(splits[1].toList(), [3, 4]);
          });
        });

        test('vsplit 1D array throws', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            expect(() => vsplit(a, 2), throwsArgumentError);
          });
        });
      });

      group('dsplit & dsplit_at Tests', () {
        test('dsplit 3D array equal split', () {
          NDArray.scope(() {
            final a = NDArray<Int32>.fromList(
              List<int>.generate(16, (i) => i + 1),
              [2, 2, 4],
              DType.int32,
            );
            final splits = dsplit(a, 2);

            expect(splits.length, 2);
            expect(splits[0].shape, [2, 2, 2]);
            expect(splits[0].toList(), [1, 2, 5, 6, 9, 10, 13, 14]);
            expect(splits[1].shape, [2, 2, 2]);
            expect(splits[1].toList(), [3, 4, 7, 8, 11, 12, 15, 16]);

            // Zero-copy check: mutating sub-array affects original
            splits[0].setCell([0, 0, 0], 99);
            expect(a.getCell([0, 0, 0]), 99);
          });
        });

        test('dsplit_at 3D array at indices', () {
          NDArray.scope(() {
            final a = NDArray<Int32>.fromList(
              List<int>.generate(16, (i) => i + 1),
              [2, 2, 4],
              DType.int32,
            );
            final splits = dsplit_at(a, [1, 3]);

            expect(splits.length, 3);
            expect(splits[0].shape, [2, 2, 1]);
            expect(splits[0].toList(), [1, 5, 9, 13]);
            expect(splits[1].shape, [2, 2, 2]);
            expect(splits[1].toList(), [2, 3, 6, 7, 10, 11, 14, 15]);
            expect(splits[2].shape, [2, 2, 1]);
            expect(splits[2].toList(), [4, 8, 12, 16]);
          });
        });

        test('dsplit rank < 3 throws', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            expect(() => dsplit(a, 2), throwsArgumentError);
          });
        });

        test('dsplit_at rank < 3 throws', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            expect(() => dsplit_at(a, [1]), throwsArgumentError);
          });
        });

        test('dsplit unequal division throws', () {
          NDArray.scope(() {
            final a = NDArray.fromList(List.generate(12, (i) => i + 1), [
              2,
              2,
              3,
            ], DType.int32);
            expect(() => dsplit(a, 2), throwsArgumentError);
          });
        });

        test('dsplit disposed array throws', () {
          NDArray.scope(() {
            final a = NDArray<Int32>.fromList(
              List<int>.generate(16, (i) => i + 1),
              [2, 2, 4],
              DType.int32,
            );
            a.dispose();
            expect(() => dsplit(a, 2), throwsStateError);
            expect(() => dsplit_at(a, [1]), throwsStateError);
          });
        });
      });
    });

    group('Stack Tests', () {
      test(
        'stack() basic 1D arrays to 2D along axis 0',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final b = NDArray.fromList([3, 4], [2], DType.int32);

          final s0 = stack([a, b], axis: 0);
          expect(s0.shape, [2, 2]);
          expect(s0.dtype, DType.int32);
          expect(s0.toList(), [1, 2, 3, 4]);
        }),
      );

      test(
        'stack() basic 1D arrays to 2D along axis 1',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final b = NDArray.fromList([3, 4], [2], DType.int32);

          final s1 = stack([a, b], axis: 1);
          expect(s1.shape, [2, 2]);
          expect(s1.toList(), [1, 3, 2, 4]);
        }),
      );

      test(
        'stack() multidimensional 2D matrices checks',
        () => NDArray.scope(() {
          // two matrices of shape [2, 3]
          final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
          final b = NDArray.fromList(
            [10, 20, 30, 40, 50, 60],
            [2, 3],
            DType.int32,
          );

          // Stacks into shape [2, 2, 3] along axis 0
          final s0 = stack([a, b], axis: 0);
          expect(s0.shape, [2, 2, 3]);
          expect(s0.toList(), [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]);

          // Stacks into shape [2, 2, 3] along axis 1
          final s1 = stack([a, b], axis: 1);
          expect(s1.shape, [2, 2, 3]);
          expect(s1.toList(), [1, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60]);
        }),
      );

      test(
        'stack() resolved negative target axis',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final b = NDArray.fromList([3, 4], [2], DType.int32);

          final sNeg = stack([a, b], axis: -1); // targets axis 1
          expect(sNeg.shape, [2, 2]);
          expect(sNeg.toList(), [1, 3, 2, 4]);
        }),
      );

      test(
        'stack() validation errors throws exceptions',
        () => NDArray.scope(() {
          // Empty list throws ArgumentError
          expect(() => stack(<NDArray<AnySpec>>[]), throwsArgumentError);

          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final wrongShape = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final wrongDType = NDArray.fromList(
            Float64List.fromList([1.0, 2.0]),
            [2],
            DType.float64,
          );

          // Mismatch shape/DType throws ArgumentError
          expect(() => stack([a, wrongShape]), throwsArgumentError);
          expect(
            () => stack([a.asAnySpec, wrongDType.asAnySpec]),
            throwsArgumentError,
          );

          // Out of bounds axis throws RangeError
          expect(() => stack([a, a], axis: 5), throwsRangeError);
          expect(() => stack([a, a], axis: -5), throwsRangeError);
        }),
      );

      test(
        'disposed arrays throw StateError',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          a.dispose();
          expect(() => stack([a, a]), throwsStateError);
        }),
      );
    });

    group('Repeating & Tiling Tests', () {
      group('repeat Tests', () {
        test('1D array, scalar repeats', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final r = repeat(a, rep(2));
            expect(r.shape, [6]);
            expect(r.toList(), [1, 1, 2, 2, 3, 3]);
          });
        });

        test('1D array, list repeats', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final r = repeat(a, rep([1, 2, 0]));
            expect(r.shape, [3]);
            expect(r.toList(), [1, 2, 2]);
          });
        });

        test('2D array, axis = 0', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep(2), axis: 0);
            expect(r.shape, [4, 2]);
            expect(r.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
          });
        });

        test('2D array, axis = 1', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep([2, 1]), axis: 1);
            expect(r.shape, [2, 3]);
            expect(r.toList(), [1, 1, 2, 3, 3, 4]);
          });
        });

        test('2D array, negative axis', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep(2), axis: -1);
            expect(r.shape, [2, 4]);
            expect(r.toList(), [1, 1, 2, 2, 3, 3, 4, 4]);
          });
        });

        test('2D array, axis = null (flatten)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep(2));
            expect(r.shape, [8]);
            expect(r.toList(), [1, 1, 2, 2, 3, 3, 4, 4]);
          });
        });

        test('3D array, repeat along axis 1', () {
          NDArray.scope(() {
            final a = NDArray.fromList(List.generate(8, (i) => i + 1), [
              2,
              2,
              2,
            ], DType.int32);
            final r = repeat(a, rep([2, 1]), axis: 1);
            expect(r.shape, [2, 3, 2]);
            expect(r.toList(), [
              1, 2, 1, 2, 3, 4, // block 0
              5, 6, 5, 6, 7, 8, // block 1
            ]);
          });
        });

        test('Edge case: repeats with 0 (deletes elements)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final r = repeat(a, rep(0));
            expect(r.shape, [0]);
            expect(r.toList(), <int>[]);
          });
        });

        test('Precondition checks', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            expect(() => repeat(a, rep([-1, 1, 1])), throwsArgumentError);
            expect(() => repeat(a, rep([1, -1, 1])), throwsArgumentError);
            expect(() => repeat(a, rep([1, 2])), throwsArgumentError);
            expect(() => repeat(a, rep(2), axis: 2), throwsRangeError);
            expect(() => repeat(a, rep(2), axis: -2), throwsRangeError);
          });
        });

        test('out parameter: correct shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final out = NDArray.create([6], DType.int32);
            final r = repeat(a, rep(2), out: out);
            expect(identical(r, out), true);
            expect(r.toList(), [1, 1, 2, 2, 3, 3]);
          });
        });

        test('out parameter: incorrect shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final outWrongShape = NDArray.create([5], DType.int32);
            final outWrongDType = NDArray.create([6], DType.float64);

            expect(
              () => repeat(a, rep(2), out: outWrongShape),
              throwsArgumentError,
            );
            expect(
              () => repeat(a.asAnySpec, rep(2), out: outWrongDType.asAnySpec),
              throwsArgumentError,
            );
          });
        });

        test('Different dtypes (float64, complex128, boolean)', () {
          NDArray.scope(() {
            final f = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final rF = repeat(f, rep(2));
            expect(rF.dtype, DType.float64);
            expect(rF.toList(), [1.0, 1.0, 2.0, 2.0]);

            final c = NDArray.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              DType.complex128,
            );
            final rC = repeat(c, rep(2));
            expect(rC.dtype, DType.complex128);
            expect(rC.toList(), [
              Complex(1, 2),
              Complex(1, 2),
              Complex(3, 4),
              Complex(3, 4),
            ]);

            final b = NDArray.fromList([true, false], [2], DType.boolean);
            final rB = repeat(b, rep(2));
            expect(rB.dtype, DType.boolean);
            expect(rB.toList(), [true, true, false, false]);
          });
        });

        test('Non-contiguous strided view input', () {
          NDArray.scope(() {
            final parent = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final view = parent.transposed; // [[1, 3], [2, 4]]
            expect(view.isContiguous, false);

            final r = repeat(view, rep(2), axis: 0);
            expect(r.shape, [4, 2]);
            expect(r.toList(), [1, 3, 1, 3, 2, 4, 2, 4]);
          });
        });
      });

      group('tile Tests', () {
        test('1D array, scalar reps', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final t = tile(a, rep(3));
            expect(t.shape, [6]);
            expect(t.toList(), [1, 2, 1, 2, 1, 2]);
          });
        });

        test('1D array, list reps', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final t = tile(a, rep([3]));
            expect(t.shape, [6]);
            expect(t.toList(), [1, 2, 1, 2, 1, 2]);
          });
        });

        test('2D array, reps of same rank', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final t = tile(a, rep([2, 3]));
            expect(t.shape, [4, 6]);
            expect(t.toList(), [
              1,
              2,
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
              3,
              4,
              1,
              2,
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
              3,
              4,
            ]);
          });
        });

        test('2D array, reps with prepended 1s (reps.length < rank)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final t = tile(a, rep([2])); // treated as [1, 2]
            expect(t.shape, [2, 4]);
            expect(t.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
          });
        });

        test('2D array, reps with expanded rank (reps.length > rank)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final t = tile(a, rep([2, 1, 2])); // a is promoted to [1, 2, 2]
            expect(t.shape, [2, 2, 4]);
            expect(t.toList(), [
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
            ]);
          });
        });

        test('Edge case: reps with 0 (returns empty array)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final t = tile(a, rep(0));
            expect(t.shape, [0]);
            expect(t.toList(), <int>[]);
          });
        });

        test('Precondition checks', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            expect(() => tile(a, rep([-1])), throwsArgumentError);
            expect(() => tile(a, rep([1, -1])), throwsArgumentError);
          });
        });

        test('out parameter: correct shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final out = NDArray.create([4], DType.int32);
            final t = tile(a, rep(2), out: out);
            expect(identical(t, out), true);
            expect(t.toList(), [1, 2, 1, 2]);
          });
        });

        test('out parameter: incorrect shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final outWrongShape = NDArray.create([5], DType.int32);
            final outWrongDType = NDArray.create([4], DType.float64);

            expect(
              () => tile(a, rep(2), out: outWrongShape),
              throwsArgumentError,
            );
            expect(
              () => tile(a.asAnySpec, rep(2), out: outWrongDType.asAnySpec),
              throwsArgumentError,
            );
          });
        });

        test('Different dtypes (float64, complex128, boolean)', () {
          NDArray.scope(() {
            final f = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final tF = tile(f, rep(2));
            expect(tF.dtype, DType.float64);
            expect(tF.toList(), [1.0, 2.0, 1.0, 2.0]);

            final c = NDArray.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              DType.complex128,
            );
            final tC = tile(c, rep(2));
            expect(tC.dtype, DType.complex128);
            expect(tC.toList(), [
              Complex(1, 2),
              Complex(3, 4),
              Complex(1, 2),
              Complex(3, 4),
            ]);

            final b = NDArray.fromList([true, false], [2], DType.boolean);
            final tB = tile(b, rep(2));
            expect(tB.dtype, DType.boolean);
            expect(tB.toList(), [true, false, true, false]);
          });
        });

        test('Non-contiguous strided view input', () {
          NDArray.scope(() {
            final parent = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final view = parent.transposed; // [[1, 3], [2, 4]]
            expect(view.isContiguous, false);

            final t = tile(view, rep([2, 1])); // tile rows 2x, cols 1x
            expect(t.shape, [4, 2]);
            expect(t.toList(), [1, 3, 2, 4, 1, 3, 2, 4]);
          });
        });
      });
    });

    group('Padding Tests', () {
      test(
        'Constant Padding - 1D - Uniform',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(9),
          );
          expect(r.shape, [7]);
          expect(r.toList(), [9, 9, 1, 2, 3, 9, 9]);
        }),
      );

      test(
        'Constant Padding - 2D - Per Axis',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = pad(
            a,
            PadWidth.axes([(1, 1), (2, 2)]),
            mode: PaddingMode.constant,
            constantValues: PadValues.axes([(8, 8), (9, 9)]),
          );
          expect(r.shape, [4, 6]);
          expect(r.toList(), [
            9,
            9,
            8,
            8,
            9,
            9,
            9,
            9,
            1,
            2,
            9,
            9,
            9,
            9,
            3,
            4,
            9,
            9,
            9,
            9,
            8,
            8,
            9,
            9,
          ]);
        }),
      );

      test(
        'Constant Mode - 2D Int with different before/after',
        () => NDArray.scope(() {
          final arr = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final padded = pad(
            arr,
            PadWidth.axes([(1, 2), (2, 1)]),
            mode: PaddingMode.constant,
            constantValues: PadValues.axes([(10, 20), (30, 40)]),
          );
          expect(padded.shape, [5, 5]);
          expect(padded.toList(), [
            30,
            30,
            10,
            10,
            40,
            30,
            30,
            1,
            2,
            40,
            30,
            30,
            3,
            4,
            40,
            30,
            30,
            20,
            20,
            40,
            30,
            30,
            20,
            20,
            40,
          ]);
        }),
      );

      test(
        'Edge Padding - 1D',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.axes([(2, 3)]), mode: PaddingMode.edge);
          expect(r.shape, [8]);
          expect(r.toList(), [1, 1, 1, 2, 3, 3, 3, 3]);
        }),
      );

      test(
        'Edge Padding - 2D',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.edge);
          expect(r.shape, [4, 4]);
          expect(r.toList(), [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
        }),
      );

      test(
        'Wrap Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.axes([(2, 2)]), mode: PaddingMode.wrap);
          expect(r.shape, [7]);
          expect(r.toList(), [2, 3, 1, 2, 3, 1, 2]);
        }),
      );

      test(
        'Reflect Padding - 1D',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(2), mode: PaddingMode.reflect);
          expect(r.shape, [7]);
          expect(r.toList(), [3, 2, 1, 2, 3, 2, 1]);
        }),
      );

      test(
        'Reflect Padding - Large padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(5), mode: PaddingMode.reflect);
          expect(r.shape, [13]);
          expect(r.toList(), [2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2]);
        }),
      );

      test(
        'Symmetric Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(2), mode: PaddingMode.symmetric);
          expect(r.shape, [7]);
          expect(r.toList(), [2, 1, 1, 2, 3, 3, 2]);
        }),
      );

      test(
        'Symmetric Mode - 1D Large Pad',
        () => NDArray.scope(() {
          final arr = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final padded = pad(arr, PadWidth.all(5), mode: PaddingMode.symmetric);
          expect(padded.toList(), [
            2.0,
            3.0,
            3.0,
            2.0,
            1.0,
            1.0,
            2.0,
            3.0,
            3.0,
            2.0,
            1.0,
            1.0,
            2.0,
          ]);
        }),
      );

      test(
        'Linear Ramp Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.linearRamp,
            endValues: PadValues.all(5.0, 7.0),
          );
          expect(r.shape, [7]);
          expect(r.toList(), [5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 7.0]);
        }),
      );

      test(
        'Linear Ramp Padding - Int',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.linearRamp,
            endValues: PadValues.all(5, 7),
          );
          expect(r.shape, [7]);
          expect(r.toList(), [5, 3, 1, 2, 3, 5, 7]);
        }),
      );

      test(
        'Stats Padding - Maximum',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 5, 3, 9, 2], [5], DType.int32);
          final r = pad(a, PadWidth.all(2), mode: PaddingMode.maximum);
          expect(r.shape, [9]);
          expect(r.toList(), [9, 9, 1, 5, 3, 9, 2, 9, 9]);
        }),
      );

      test(
        'Stats Padding - Maximum with Window',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 5, 3, 9, 2], [5], DType.int32);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.maximum,
            statLength: StatLength.all(3),
          );
          expect(r.shape, [9]);
          expect(r.toList(), [5, 5, 1, 5, 3, 9, 2, 9, 9]);
        }),
      );

      test(
        'Stats Padding - Mean',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.mean);
          expect(r.shape, [6]);
          expect(r.toList(), [2.5, 1.0, 2.0, 3.0, 4.0, 2.5]);
        }),
      );

      test(
        'Stats Padding - Median - Odd',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 5, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
          expect(r.shape, [5]);
          expect(r.toList(), [3, 1, 5, 3, 3]);
        }),
      );

      test(
        'Stats Padding - Median - Even',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 5.0, 3.0, 9.0], [4], DType.float64);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
          expect(r.shape, [6]);
          expect(r.toList(), [4.0, 1.0, 5.0, 3.0, 9.0, 4.0]);
        }),
      );

      test(
        'Complex Padding - Constant',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4)],
            [2],
            DType.complex128,
          );
          final r = pad(
            a,
            PadWidth.all(1),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(Complex(9, 9)),
          );
          expect(r.shape, [4]);
          expect(r.toList(), [
            Complex(9, 9),
            Complex(1, 2),
            Complex(3, 4),
            Complex(9, 9),
          ]);
        }),
      );

      test(
        'Complex Padding - Median (Independent)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 6), Complex(5, 2), Complex(3, 4)],
            [3],
            DType.complex128,
          );
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
          expect(r.shape, [5]);
          expect(r.toList(), [
            Complex(3, 4),
            Complex(1, 6),
            Complex(5, 2),
            Complex(3, 4),
            Complex(3, 4),
          ]);
        }),
      );

      test(
        'Complex Padding - Min/Max (Lexicographical)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 10), Complex(2, 1), Complex(2, 2)],
            [3],
            DType.complex128,
          );
          final rMin = pad(a, PadWidth.all(1), mode: PaddingMode.minimum);
          final rMax = pad(a, PadWidth.all(1), mode: PaddingMode.maximum);

          expect(rMin.toList(), [
            Complex(1, 10),
            Complex(1, 10),
            Complex(2, 1),
            Complex(2, 2),
            Complex(1, 10),
          ]);
          expect(rMax.toList(), [
            Complex(2, 2),
            Complex(1, 10),
            Complex(2, 1),
            Complex(2, 2),
            Complex(2, 2),
          ]);
        }),
      );

      test(
        'Boolean Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([true, false, true], [3], DType.boolean);
          final r = pad(
            a,
            PadWidth.all(1),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(true),
          );
          expect(r.shape, [5]);
          expect(r.toList(), [true, true, false, true, true]);
        }),
      );

      test(
        'Zero Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(0));
          expect(r.shape, [3]);
          expect(r.toList(), [1, 2, 3]);
          expect(r, isNot(same(a)));
        }),
      );

      test(
        'Out Parameter Reuse',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final out = NDArray.zeros([5], DType.int32);
          final r = pad(
            a,
            PadWidth.all(1),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(9),
            out: out,
          );
          expect(r, same(out));
          expect(out.toList(), [9, 1, 2, 3, 9]);
        }),
      );

      test(
        '2D Reflect Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
          final r = pad(a, PadWidth.all(1), mode: PadMode.reflect);
          expect(r.shape, [4, 5]);
          expect(r.toList(), [
            5,
            4,
            5,
            6,
            5,
            2,
            1,
            2,
            3,
            2,
            5,
            4,
            5,
            6,
            5,
            2,
            1,
            2,
            3,
            2,
          ]);
        }),
      );

      test(
        '2D Symmetric Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
          final r = pad(a, PadWidth.all(1), mode: PadMode.symmetric);
          expect(r.shape, [4, 5]);
          expect(r.toList(), [
            1,
            1,
            2,
            3,
            3,
            1,
            1,
            2,
            3,
            3,
            4,
            4,
            5,
            6,
            6,
            4,
            4,
            5,
            6,
            6,
          ]);
        }),
      );

      test(
        '2D Wrap Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
          final r = pad(a, PadWidth.all(1), mode: PadMode.wrap);
          expect(r.shape, [4, 5]);
          expect(r.toList(), [
            6,
            4,
            5,
            6,
            4,
            3,
            1,
            2,
            3,
            1,
            6,
            4,
            5,
            6,
            4,
            3,
            1,
            2,
            3,
            1,
          ]);
        }),
      );

      test(
        '3D Padding - Constant, Edge, Reflect, Wrap, Symmetric',
        () => NDArray.scope(() {
          final list = List<double>.generate(24, (i) => i.toDouble());
          final a = NDArray.fromList(list, [2, 3, 4], DType.float64);

          // Constant
          final rConst = pad(
            a,
            PadWidth.all(1),
            mode: PadMode.constant,
            constantValues: PadValues.all(-1.0),
          );
          expect(rConst.shape, [4, 5, 6]);
          expect(rConst.getCell([0, 0, 0]), -1.0);
          expect(rConst.getCell([1, 1, 1]), 0.0);
          expect(rConst.getCell([2, 3, 4]), 23.0);
          expect(rConst.getCell([3, 4, 5]), -1.0);

          // Edge
          final rEdge = pad(a, PadWidth.all(1), mode: PadMode.edge);
          expect(rEdge.shape, [4, 5, 6]);
          expect(rEdge.getCell([0, 0, 0]), 0.0);
          expect(rEdge.getCell([3, 4, 5]), 23.0);

          // Reflect
          final rReflect = pad(a, PadWidth.all(1), mode: PadMode.reflect);
          expect(rReflect.shape, [4, 5, 6]);

          // Wrap
          final rWrap = pad(a, PadWidth.all(1), mode: PadMode.wrap);
          expect(rWrap.shape, [4, 5, 6]);

          // Symmetric
          final rSym = pad(a, PadWidth.all(1), mode: PadMode.symmetric);
          expect(rSym.shape, [4, 5, 6]);
        }),
      );

      test(
        '4D Padding - Edge',
        () => NDArray.scope(() {
          final a = NDArray.ones([2, 2, 2, 2], DType.float64);
          final r = pad(a, PadWidth.all(1), mode: PadMode.edge);
          expect(r.shape, [4, 4, 4, 4]);
          for (final val in r.toList()) {
            expect(val, 1.0);
          }
        }),
      );

      test(
        'Strided Input Array Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final transposed = a.transposed; // non-contiguous
          final r = pad(
            transposed,
            PadWidth.all(1),
            mode: PadMode.constant,
            constantValues: PadValues.all(0),
          );
          expect(r.shape, [4, 4]);
          expect(r.toList(), [0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 4, 0, 0, 0, 0, 0]);
        }),
      );

      test(
        'Padding All DTypes - Constant, Edge, Reflect',
        () => NDArray.scope(() {
          for (final dtype in [
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.int16,
            DType.uint8,
          ]) {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], dtype);
            final rEdge = pad(a, PadWidth.all(1), mode: PadMode.edge);
            expect(rEdge.shape, [4, 4]);
            expect(rEdge.dtype, dtype);

            final rReflect = pad(a, PadWidth.all(1), mode: PadMode.reflect);
            expect(rReflect.shape, [4, 4]);
            expect(rReflect.dtype, dtype);

            final rConst = pad(a, PadWidth.all(1), mode: PadMode.constant);
            expect(rConst.shape, [4, 4]);
            expect(rConst.dtype, dtype);
          }

          // Complex64
          final c64 = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
            [2, 2],
            DType.complex64,
          );
          final rC64 = pad(c64, PadWidth.all(1), mode: PadMode.edge);
          expect(rC64.shape, [4, 4]);
          expect(rC64.dtype, DType.complex64);

          // Complex128
          final c128 = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
            [2, 2],
            DType.complex128,
          );
          final rC128 = pad(c128, PadWidth.all(1), mode: PadMode.reflect);
          expect(rC128.shape, [4, 4]);
          expect(rC128.dtype, DType.complex128);
        }),
      );

      test(
        'Preconditions - Disposed',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          a.dispose();
          expect(() => pad(a, PadWidth.all(1)), throwsStateError);
        }),
      );

      test(
        'Preconditions - 0-D Array',
        () => NDArray.scope(() {
          final a = NDArray.scalar(1, dtype: DType.int32);
          expect(() => pad(a, PadWidth.all(1)), throwsArgumentError);
        }),
      );
    });

    test(
      'reshape() non-contiguous view copies memory first',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [2, 2],
          DType.float64,
        );

        // Transposed view is non-contiguous (strides: [1, 2])
        final transposed = parent.transposed;
        expect(transposed.isContiguous, false);
        expect(transposed.toList(), [1.0, 3.0, 2.0, 4.0]);

        // Reshaping the transposed view: it should automatically copy first
        final reshaped = transposed.reshape([4]);
        expect(reshaped.isContiguous, true);
        expect(reshaped.toList(), [1.0, 3.0, 2.0, 4.0]);

        // Verify memory decoupling (modifying reshaped does NOT affect parent)
        reshaped.setCell([0], 99.0);
        expect(parent.getCell([0, 0]), 1.0);
      }),
    );

    test(
      'transpose() negative axis support',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [2, 3],
          DType.float64,
        );
        // Reverse axes order via negative indices: [-1, -2] is equivalent to [1, 0]
        final b = a.transpose([-1, -2]);
        expect(b.shape, [3, 2]);
        expect(b.toList(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
      }),
    );

    test(
      'reshape() size mismatch throws ArgumentError coverage',
      () => NDArray.scope(() {
        final a = NDArray.ones([2, 2], DType.float64);
        expect(() => a.reshape([3]), throwsArgumentError);
      }),
    );

    test(
      'transpose() invalid axes length throws ArgumentError coverage',
      () => NDArray.scope(() {
        final a = NDArray.ones([2, 2], DType.float64);
        expect(() => a.transpose([0]), throwsArgumentError);
      }),
    );

    test(
      'broadcastShapes() incompatible shapes throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.ones([2], DType.float64);
        final b = NDArray.ones([3], DType.float64);

        expect(() => add(a, b), throwsArgumentError);
      }),
    );

    test(
      'Contiguous sub-slice flatten() optimization correctness',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        // Slice representing the first 2 elements (strides: [1], contiguous: true, totalSize < data.length!)
        final view = parent.slice([Slice(start: 0, stop: 2)]);

        expect(view.isContiguous, true);
        expect(view.shape, [2]);

        final flat = view.flatten();
        expect(flat.shape, [2]);
        expect(flat.isContiguous, true);
        expect(flat.toList(), [1.0, 2.0]);
      }),
    );

    test(
      'expand_dims() and squeeze() shape view manipulations ufunc correctness',
      () {
        // 1. expand_dims() verification
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

        final aExp0 = expand_dims(a, 0);
        final aExp1 = expand_dims(a, 1);
        final aExpNeg = expand_dims(a, -1); // normalized to axis 1

        expect(aExp0.shape, [1, 3]);
        expect(aExp0.strides, [1, 1]);

        expect(aExp1.shape, [3, 1]);
        expect(aExp1.strides, [1, 1]);

        expect(aExpNeg.shape, [3, 1]);

        // expand_dims out of bounds exception
        expect(() => expand_dims(a, 3), throwsRangeError);

        // 2. squeeze() verification
        final b = NDArray.zeros([1, 3, 1], DType.float64);

        final bSqueezedAll = squeeze(b);
        final bSqueezed0 = squeeze(b, axis: [0]);

        expect(bSqueezedAll.shape, [3]);
        expect(bSqueezedAll.strides, [1]);

        expect(bSqueezed0.shape, [3, 1]);

        // Squeeze non-unit dimension axis throws ArgumentError
        expect(() => squeeze(b, axis: [1]), throwsArgumentError);
      },
    );

    test(
      'concatenate() validation errors throws exceptions',
      () => NDArray.scope(() {
        expect(() => concatenate(<NDArray<AnySpec>>[]), throwsArgumentError);

        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final wrongRank = NDArray.fromList([1.0, 2.0], [1, 2], DType.float64);
        final wrongDType = NDArray.fromList([1, 2], [2], DType.int32);

        expect(() => concatenate([a, a], axis: 5), throwsRangeError);
        expect(() => concatenate([a, a], axis: -5), throwsRangeError);
        expect(
          () => concatenate(<NDArray<AnySpec>>[a, wrongDType]),
          throwsArgumentError,
        );

        expect(
          () => concatenate(<NDArray<AnySpec>>[a, wrongRank]),
          throwsArgumentError,
        );

        final mat1 = NDArray.fromList(
          [1.0, 1.0, 1.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final mat2 = NDArray.fromList(
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [2, 3],
          DType.float64,
        );
        expect(
          () => concatenate(<NDArray<AnySpec>>[mat1, mat2], axis: 0),
          throwsArgumentError,
        );
      }),
    );

    test(
      'Sliding Window Views slidingWindowView() zero-copy view correctness',
      () {
        // 1D array sliding window of size 3
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );

        final view = slidingWindowView(a, [3]);
        expect(view.shape, [3, 3]);
        expect(view.dtype, DType.float64);

        expect(view.getCell([0, 0]), 1.0);
        expect(view.getCell([0, 1]), 2.0);
        expect(view.getCell([0, 2]), 3.0);

        expect(view.getCell([1, 0]), 2.0);
        expect(view.getCell([1, 1]), 3.0);
        expect(view.getCell([1, 2]), 4.0);

        expect(view.getCell([2, 0]), 3.0);
        expect(view.getCell([2, 1]), 4.0);
        expect(view.getCell([2, 2]), 5.0);

        // 2D array sliding window along specified axis
        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 4],
          DType.float64,
        );

        // Slide window of size 2 along axis 1 (columns)
        final win2d = slidingWindowView(mat, [2], axis: [1]);
        expect(win2d.shape, [
          2,
          3,
          2,
        ]); // original [2, 4] -> columns: 4 - 2 + 1 = 3 -> [2, 3, 2]

        expect(win2d.getCell([0, 0, 0]), 1.0);
        expect(win2d.getCell([0, 0, 1]), 2.0);
        expect(win2d.getCell([0, 1, 0]), 2.0);
        expect(win2d.getCell([0, 1, 1]), 3.0);

        // Verify ArgumentError validations
        expect(
          () => slidingWindowView(a, [6]),
          throwsArgumentError,
        ); // window exceeds size
        expect(
          () => slidingWindowView(a, [0]),
          throwsArgumentError,
        ); // invalid positive size
        expect(
          () => slidingWindowView(a, [2, 2]),
          throwsArgumentError,
        ); // axes count mismatch
      },
    );
    group('Workstream 4: Manipulation & Meshes Memory and Ergonomics', () {
      test(
        '1D hstack stacks along axis 0',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
          final res = hstack([a, b]);
          expect(res.shape, [4]);
          expect(res.toList(), [1.0, 2.0, 3.0, 4.0]);
        }),
      );

      test('roll returns an owning array that can be disposed', () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final r = roll(a, 1);
        expect(r.toList(), [4.0, 1.0, 2.0, 3.0]);
        r.dispose(); // Should not throw and properly free memory
        a.dispose();
      });

      test('ogrid and mgrid scope detachment and dispose', () {
        final ogridRes = ogrid([GridRange(0, 3), GridRange(0, 4)]);
        expect(ogridRes.length, 2);
        expect(ogridRes[0].shape, [3, 1]);
        expect(ogridRes[1].shape, [1, 4]);
        ogridRes[0].dispose();
        ogridRes[1].dispose();

        final mgridRes = mgrid([GridRange(0, 2), GridRange(0, 3)]);
        expect(mgridRes.shape, [2, 2, 3]);
        mgridRes.dispose();
      });

      test('rot90 error message validation includes formatted rank', () {
        final a1d = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(
          () => rot90(a1d),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('(was rank 1).'),
            ),
          ),
        );

        final a2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(
          () => rot90(a2d, 1, [0, 5]),
          throwsA(
            isA<RangeError>().having(
              (e) => e.message,
              'message',
              contains('for array of rank 2.'),
            ),
          ),
        );
      });
    });
  });
}
