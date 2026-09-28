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
    });
  });
}
