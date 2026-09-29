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
  group('Advanced Indexing Operations (Section 3.24)', () {
    group('take_along_axis', () {
      test(
        '2D float64 take_along_axis along axis 1',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [2, 3],
            DType.float64,
          );
          final indices = NDArray.fromList([2, 0, 1, 1], [2, 2], DType.int32);
          final res = take_along_axis(a, indices, 1);
          expect(res.shape, [2, 2]);
          expect(res.dtype, DType.float64);
          expect(res.toList(), [30.0, 10.0, 50.0, 50.0]);
        }),
      );

      test(
        '2D int32 take_along_axis along axis 0 with negative indices',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [3, 2], DType.int32);
          final indices = NDArray.fromList([-1, 0, 1, -2], [2, 2], DType.int32);
          final res = take_along_axis(a, indices, 0);
          expect(res.shape, [2, 2]);
          expect(res.toList(), [5, 2, 3, 4]);
        }),
      );

      test(
        'take_along_axis strided/non-contiguous array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 4],
            DType.int64,
          );
          final view = a.swapaxes(0, 1); // shape [4, 2], strided view
          final idx = NDArray.fromList([1, 0, 1, 0], [4, 1], DType.int64);
          final res = take_along_axis(view, idx, 1);
          expect(res.shape, [4, 1]);
          expect(res.toList(), [5, 2, 7, 4]);
        }),
      );

      test(
        'take_along_axis with out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final indices = NDArray.fromList([1, 0], [2, 1], DType.int32);
          final out = NDArray.create([2, 1], DType.float64);
          final res = take_along_axis(a, indices, 1, out: out);
          expect(identical(res, out), true);
          expect(res.toList(), [2.0, 3.0]);
        }),
      );

      test(
        'take_along_axis complex128 and boolean dtypes',
        () => NDArray.scope(() {
          final c = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
            [2, 2],
            DType.complex128,
          );
          final idx = NDArray.fromList([1, 0], [2, 1], DType.int32);
          final resC = take_along_axis(c, idx, 1);
          expect(resC.toList(), [Complex(3, 4), Complex(5, 6)]);

          final b = NDArray<Boolean>.fromList(
            [true, false, false, true],
            [2, 2],
            DType.boolean,
          );
          final resB = take_along_axis(b, idx, 1);
          expect(resB.toList(), [false, false]);
        }),
      );

      test(
        'take_along_axis all data types (float32, complex64, int16, uint8)',
        () => NDArray.scope(() {
          final f32 = NDArray<Float32>.fromList(
            [1.5, 2.5, 3.5, 4.5],
            [2, 2],
            DType.float32,
          );
          final idx = NDArray.fromList([1, 0], [2, 1], DType.int32);
          final resF32 = take_along_axis(f32, idx, 1);
          expect(resF32.dtype, DType.float32);
          expect(resF32.toList(), [2.5, 3.5]);

          final c64 = NDArray<Complex64>.fromList(
            [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
            [2, 2],
            DType.complex64,
          );
          final resC64 = take_along_axis(c64, idx, 1);
          expect(resC64.dtype, DType.complex64);
          expect(resC64.toList(), [Complex(3, 4), Complex(5, 6)]);

          final i16 = NDArray<Int16>.fromList(
            [10, 20, 30, 40],
            [2, 2],
            DType.int16,
          );
          final resI16 = take_along_axis(i16, idx, 1);
          expect(resI16.dtype, DType.int16);
          expect(resI16.toList(), [20, 30]);

          final u8 = NDArray<Uint8>.fromList(
            [100, 200, 50, 150],
            [2, 2],
            DType.uint8,
          );
          final resU8 = take_along_axis(u8, idx, 1);
          expect(resU8.dtype, DType.uint8);
          expect(resU8.toList(), [200, 50]);
        }),
      );

      test(
        'take_along_axis 1D and 3D with broadcasting',
        () => NDArray.scope(() {
          // 1D test
          final a1d = NDArray<Float64>.fromList(
            [100.0, 200.0, 300.0, 400.0],
            [4],
            DType.float64,
          );
          final idx1d = NDArray.fromList([3, 1, -1, 0], [4], DType.int32);
          final res1d = take_along_axis(a1d, idx1d, 0);
          expect(res1d.toList(), [400.0, 200.0, 400.0, 100.0]);

          // 3D test: arr shape [2, 1, 3], idx shape [2, 2, 2], axis = 2
          final a3d = NDArray.fromList(
            [1, 2, 3, 4, 5, 6],
            [2, 1, 3],
            DType.int32,
          );
          final idx3d = NDArray.fromList(
            [2, 0, 1, 1, 0, 2, 1, 0],
            [2, 2, 2],
            DType.int32,
          );
          final res3d = take_along_axis(a3d, idx3d, 2);
          expect(res3d.shape, [2, 2, 2]);
          expect(res3d.toList(), [3, 1, 2, 2, 4, 6, 5, 4]);
        }),
      );

      test(
        'take_along_axis error cases',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final idx1D = NDArray.fromList([0, 1], [2], DType.int32);
          expect(() => take_along_axis(a, idx1D, 0), throwsArgumentError);

          final idx2D = NDArray.fromList([5, 0], [2, 1], DType.int32);
          expect(() => take_along_axis(a, idx2D, 1), throwsRangeError);
          expect(() => take_along_axis(a, idx2D, 5), throwsRangeError);
          final idxNeg = NDArray.fromList([-5, 0], [2, 1], DType.int32);
          expect(() => take_along_axis(a, idxNeg, 1), throwsRangeError);
        }),
      );
    });

    group('put_along_axis', () {
      test(
        '2D put_along_axis in-place with array and scalar',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [2, 3],
            DType.float64,
          );
          final indices = NDArray.fromList([2, 0, 1, 1], [2, 2], DType.int32);
          final values = NDArray.fromList(
            [99.0, 88.0, 77.0, 66.0],
            [2, 2],
            DType.float64,
          );
          put_along_axis(a, indices, values, 1);
          expect(a.toList(), [88.0, 20.0, 99.0, 40.0, 66.0, 60.0]);

          final b = NDArray.zeros([2, 3], DType.int32);
          final bIdx = NDArray.fromList([2, 0], [2, 1], DType.int32);
          put_along_axis(b, bIdx, 99, 1);
          expect(b.toList(), [0, 0, 99, 99, 0, 0]);
        }),
      );

      test(
        'put_along_axis 1D and 3D with all data types',
        () => NDArray.scope(() {
          final a1d = NDArray<Float32>.fromList(
            [10.0, 20.0, 30.0],
            [3],
            DType.float32,
          );
          final idx1d = NDArray.fromList([2, 0], [2], DType.int32);
          final val1d = NDArray<Float32>.fromList(
            [99.0, 88.0],
            [2],
            DType.float32,
          );
          put_along_axis(a1d, idx1d, val1d, 0);
          expect(a1d.toList(), [88.0, 20.0, 99.0]);

          final c128 = NDArray<Complex128>.zeros([2, 2], DType.complex128);
          final idxC = NDArray.fromList([1, 0], [2, 1], DType.int32);
          put_along_axis(c128, idxC, Complex(7, 8), 1);
          expect(c128.toList(), [
            Complex(0, 0),
            Complex(7, 8),
            Complex(7, 8),
            Complex(0, 0),
          ]);
        }),
      );

      test(
        'put_along_axis with out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final indices = NDArray.fromList([1, 0], [2, 1], DType.int32);
          final values = NDArray.fromList([10, 20], [2, 1], DType.int32);
          final out = NDArray.create([2, 2], DType.int32);
          final res = put_along_axis(a, indices, values, 1, out: out);
          expect(identical(res, out), true);
          expect(out.toList(), [1, 10, 20, 4]);
          expect(a.toList(), [1, 2, 3, 4]);
        }),
      );

      test(
        'put_along_axis error cases',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final indices = NDArray.fromList([10, 0], [2, 1], DType.int32);
          final values = NDArray.fromList([9.0, 8.0], [2, 1], DType.float64);
          expect(() => put_along_axis(a, indices, values, 1), throwsRangeError);
          final idxNeg = NDArray.fromList([-10, 0], [2, 1], DType.int32);
          expect(() => put_along_axis(a, idxNeg, values, 1), throwsRangeError);
        }),
      );
    });

    group('choose', () {
      test(
        'choose basic raise mode',
        () => NDArray.scope(() {
          final choice0 = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final choice1 = NDArray.fromList(
            [10.0, 11.0, 12.0, 13.0],
            [2, 2],
            DType.float64,
          );
          final a = NDArray.fromList([0, 1, 1, 0], [2, 2], DType.int32);
          final res = choose(a, [choice0, choice1]);
          expect(res.shape, [2, 2]);
          expect(res.toList(), [0.0, 11.0, 12.0, 3.0]);
        }),
      );

      test(
        'choose wrap mode',
        () => NDArray.scope(() {
          final choice0 = NDArray.fromList([10, 20], [2], DType.int32);
          final choice1 = NDArray.fromList([30, 40], [2], DType.int32);
          final a = NDArray.fromList([2, -1], [2], DType.int32);
          final res = choose(a, [choice0, choice1], mode: ChooseMode.wrap);
          expect(res.toList(), [10, 40]);
        }),
      );

      test(
        'choose clip mode',
        () => NDArray.scope(() {
          final choice0 = NDArray.fromList([10, 20], [2], DType.int32);
          final choice1 = NDArray.fromList([30, 40], [2], DType.int32);
          final a = NDArray.fromList([-5, 10], [2], DType.int32);
          final res = choose(a, [choice0, choice1], mode: ChooseMode.clip);
          expect(res.toList(), [10, 40]);
        }),
      );

      test(
        'choose error cases',
        () => NDArray.scope(() {
          final choice0 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final a = NDArray.fromList([0, 5], [2], DType.int32);
          expect(() => choose(a, [choice0]), throwsRangeError);
          expect(() => choose(a, []), throwsArgumentError);
        }),
      );
    });

    group('select', () {
      test(
        'select basic boolean condition matching',
        () => NDArray.scope(() {
          final cond1 = NDArray<Boolean>.fromList(
            [true, false, false, false, false],
            [5],
            DType.boolean,
          );
          final cond2 = NDArray<Boolean>.fromList(
            [false, false, false, true, true],
            [5],
            DType.boolean,
          );
          final choice1 = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          final choice2 = NDArray.fromList(
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [5],
            DType.float64,
          );

          final res = select(
            [cond1, cond2],
            [choice1, choice2],
            defaultValue: -1.0,
          );
          expect(res.toList(), [10.0, -1.0, -1.0, 400.0, 500.0]);
        }),
      );

      test(
        'select with out parameter and broadcasting',
        () => NDArray.scope(() {
          final cond1 = NDArray<Boolean>.fromList(
            [true, false],
            [2, 1],
            DType.boolean,
          );
          final choice1 = NDArray.fromList([10, 20], [2, 1], DType.int32);
          final choice2 = NDArray.fromList([100, 200], [1, 2], DType.int32);
          final cond2 = NDArray<Boolean>.fromList(
            [false, true],
            [1, 2],
            DType.boolean,
          );

          final out = NDArray.create([2, 2], DType.int32);
          final res = select(
            [cond1, cond2],
            [choice1, choice2],
            defaultValue: 0,
            out: out,
          );
          expect(identical(res, out), true);
          expect(res.shape, [2, 2]);
        }),
      );

      test(
        'select error cases',
        () => NDArray.scope(() {
          final cond = NDArray<Boolean>.fromList([true], [1], DType.boolean);
          final choice = NDArray.fromList([1.0], [1], DType.float64);
          expect(() => select([], []), throwsArgumentError);
          expect(() => select([cond], []), throwsArgumentError);
          expect(() => select([cond], [choice, choice]), throwsArgumentError);
        }),
      );

      test(
        'select defaults pure integer scalar choices to DType.int64 and respects NEP 50 weak scalars',
        () => NDArray.scope(() {
          final cond = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );
          final res64 = select([cond], [10], defaultValue: 20);
          expect(res64.dtype, DType.int64);
          expect(res64.toList(), [10, 20]);

          // Weak scalar with Int32 array preserves Int32 if in range
          final choice32 = NDArray.fromList([10, 20], [2], DType.int32);
          final res32 = select([cond], [choice32], defaultValue: 5);
          expect(res32.dtype, DType.int32);
          expect(res32.toList(), [10, 5]);

          // Scalar exceeding Int32 promotes to Int64
          final bigVal = 1 << 40;
          final resPromoted = select([cond], [choice32], defaultValue: bigVal);
          expect(resPromoted.dtype, DType.int64);
          expect(resPromoted.toList(), [10, bigVal]);
        }),
      );
    });

    group('unravel_index and ravel_multi_index', () {
      test(
        'unravel_index C and Fortran order and round-trip with ravel_multi_index',
        () => NDArray.scope(() {
          final idx = NDArray.fromList([22, 41, 37], [3], DType.int64);
          final coordsC = unravel_index(idx, [7, 6]);
          expect(coordsC.length, 2);
          expect(coordsC[0].dtype, DType.int64);
          expect(coordsC[1].dtype, DType.int64);
          expect(coordsC[0].toList(), [3, 6, 6]);
          expect(coordsC[1].toList(), [4, 5, 1]);

          final raveledC = ravel_multi_index(coordsC, [7, 6]);
          expect(raveledC.dtype, DType.int64);
          expect(raveledC.toList(), [22, 41, 37]);

          final coordsF = unravelIndex(idx, [7, 6], order: IndexOrder.f);
          expect(coordsF[0].toList(), [1, 6, 2]);
          expect(coordsF[1].toList(), [3, 5, 5]);

          final raveledF = ravelMultiIndex(coordsF, [
            7,
            6,
          ], order: IndexOrder.f);
          expect(raveledF.toList(), [22, 41, 37]);
        }),
      );

      test(
        'unravel_index and ravel_multi_index support 64-bit shapes and flat indices > 2^31 - 1',
        () => NDArray.scope(() {
          const bigRows = 1 << 22;
          const bigCols = 1 << 22;
          const flatIdx = (3 * bigCols) + 1234567;
          final idx = NDArray.fromList([flatIdx], [1], DType.int64);
          final coords = unravel_index(idx, [bigRows, bigCols]);
          expect(coords[0].toList(), [3]);
          expect(coords[1].toList(), [1234567]);

          final back = ravel_multi_index(coords, [bigRows, bigCols]);
          expect(back.toList(), [flatIdx]);
        }),
      );

      test(
        'unravel_index supports all integer dtypes, strided inputs, and out aliasing',
        () => NDArray.scope(() {
          for (final dt in <DType<DTypeTag>>[
            DType.int8,
            DType.uint8,
            DType.int16,
            DType.uint16,
            DType.int32,
            DType.uint32,
            DType.int64,
            DType.uint64,
          ]) {
            final idx = NDArray.fromList([5, 11], [2], dt);
            final coords = unravel_index(idx, [3, 4]);
            expect(coords[0].dtype, DType.int64);
            expect(coords[0].toList(), [1, 2]);
            expect(coords[1].toList(), [1, 3]);
          }

          // Strided 2D indices and aliased out
          final mat = NDArray.fromList([0, 5, 7, 11], [2, 2], DType.int64);
          final matT = mat.transpose();
          final out0 = NDArray.create([2, 2], DType.int64);
          final res = unravel_index(matT, [3, 4], out: [matT, out0]);
          expect(identical(res[0], matT), isTrue);
          expect(identical(res[1], out0), isTrue);
          // matT was [[0, 7], [5, 11]] -> rows [[0, 1], [1, 2]], cols [[0, 3], [1, 3]]
          expect(matT.shape, [2, 2]);
          expect(matT.toList(), [0, 1, 1, 2]);
          expect(out0.shape, [2, 2]);
          expect(out0.toList(), [0, 3, 1, 3]);
        }),
      );

      test(
        'ravel_multi_index supports broadcasting, ChooseMode wrap/clip, and per-axis modes',
        () => NDArray.scope(() {
          final rows = NDArray.fromList([-1, 4], [2, 1], DType.int32);
          final cols = NDArray.fromList([-2, 1, 5], [1, 3], DType.int64);

          final wrapped = ravel_multi_index(
            [rows, cols],
            [3, 4],
            mode: ChooseMode.wrap,
          );
          expect(wrapped.shape, [2, 3]);
          // rows [-1, 4] mod 3 -> [2, 1]; cols [-2, 1, 5] mod 4 -> [2, 1, 1]
          expect(wrapped.toList(), [
            2 * 4 + 2,
            2 * 4 + 1,
            2 * 4 + 1,
            1 * 4 + 2,
            1 * 4 + 1,
            1 * 4 + 1,
          ]);

          final mixed = ravel_multi_index(
            [rows, cols],
            [3, 4],
            mode: [ChooseMode.wrap, ChooseMode.clip],
          );
          // rows mod 3 -> [2, 1]; cols clipped to [0..3] -> [0, 1, 3]
          expect(mixed.toList(), [
            2 * 4 + 0,
            2 * 4 + 1,
            2 * 4 + 3,
            1 * 4 + 0,
            1 * 4 + 1,
            1 * 4 + 3,
          ]);

          expect(
            () => ravel_multi_index([rows, cols], [3, 4]),
            throwsRangeError,
          );
          expect(
            () =>
                unravel_index(NDArray.fromList([12], [1], DType.int64), [3, 4]),
            throwsRangeError,
          );
        }),
      );
    });

    group(
      'indices, sparse_indices, diag_indices, tril/triu_indices, mask_indices',
      () {
        test(
          'indices and sparse_indices return DType.int64 by default and match grid coordinates',
          () => NDArray.scope(() {
            final grid = indices([2, 3]);
            expect(grid.dtype, DType.int64);
            expect(grid.shape, [2, 2, 3]);
            expect(grid.toList(), [0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2]);

            final sparse = sparse_indices([2, 3]);
            expect(sparse.length, 2);
            expect(sparse[0].dtype, DType.int64);
            expect(sparse[0].shape, [2, 1]);
            expect(sparse[0].toList(), [0, 1]);
            expect(sparse[1].dtype, DType.int64);
            expect(sparse[1].shape, [1, 3]);
            expect(sparse[1].toList(), [0, 1, 2]);
          }),
        );

        test(
          'diag_indices and diag_indices_from return DType.int64 coordinate arrays',
          () => NDArray.scope(() {
            final di = diag_indices(4);
            expect(di.length, 2);
            expect(di[0].dtype, DType.int64);
            expect(di[0].toList(), [0, 1, 2, 3]);
            expect(di[1].toList(), [0, 1, 2, 3]);

            final cube = NDArray.zeros([3, 3, 3], DType.float64);
            final diCube = diag_indices_from(cube);
            expect(diCube.length, 3);
            for (final c in diCube) {
              expect(c.dtype, DType.int64);
              expect(c.toList(), [0, 1, 2]);
            }

            final nonSquare = NDArray.zeros([3, 4], DType.float64);
            expect(() => diag_indices_from(nonSquare), throwsArgumentError);
          }),
        );

        test(
          'tril_indices, triu_indices, and mask_indices produce 64-bit row/col coordinates',
          () => NDArray.scope(() {
            final (row: lRows, col: lCols) = tril_indices(3);
            expect(lRows.dtype, DType.int64);
            expect(lCols.dtype, DType.int64);
            expect(lRows.toList(), [0, 1, 1, 2, 2, 2]);
            expect(lCols.toList(), [0, 0, 1, 0, 1, 2]);

            final (row: uRows, col: uCols) = triu_indices(3, k: 1);
            expect(uRows.dtype, DType.int64);
            expect(uCols.dtype, DType.int64);
            expect(uRows.toList(), [0, 0, 1]);
            expect(uCols.toList(), [1, 2, 2]);

            // Rectangular matrix with tril_indices_from / triu_indices_from
            final rect = NDArray.zeros([3, 4], DType.float64);
            final (row: rfRows, col: rfCols) = tril_indices_from(rect, k: -1);
            expect(rfRows.toList(), [1, 2, 2]);
            expect(rfCols.toList(), [0, 0, 1]);

            final (row: ufRows, col: ufCols) = triu_indices_from(rect, k: 2);
            expect(ufRows.toList(), [0, 0, 1]);
            expect(ufCols.toList(), [2, 3, 3]);

            // mask_indices with triu and tril
            final (row: mRows, col: mCols) = mask_indices(3, triu, k: 1);
            expect(mRows.dtype, DType.int64);
            expect(mCols.dtype, DType.int64);
            expect(mRows.toList(), uRows.toList());
            expect(mCols.toList(), uCols.toList());
          }),
        );
      },
    );
  });
}
