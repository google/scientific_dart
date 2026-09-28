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

import 'dart:typed_data';

void main() {
  group('NDArray Advanced Indexing, Fancy Lists & Mask Overloads Tests', () {
    group('Explicit Static-Typed Addressing Methods tests', () {
      test(
        'getCell and setCell basic checks',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );

          expect(a.getCell([0, 0]), 1.0);
          expect(a.getCell([0, 1]), 2.0);

          a.setCell([1, 1], Float64(40.0));
          expect(a.getCell([1, 1]), 40.0);
          expect(a.toList(), [1.0, 2.0, 3.0, 40.0]);
        }),
      );

      test(
        'setByMask explicit boolean mask mutation scalar clipping',
        () => NDArray.scope(() {
          final arr = NDArray.fromList(
            Float64List.fromList([-5.0, 10.0, -2.5, 4.0]),
            [4],
            DType.float64,
          );
          final mask = arr < 0.0; // returns binary mask array

          // Explicit scalar clip
          arr.setByMaskScalar(mask, Float64(0.0));
          expect(arr.toList(), [0.0, 10.0, 0.0, 4.0]);
        }),
      );

      test(
        'setIndicesScalar and setIndices fancy explicit row mutations',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Int32List.fromList([1, 1, 1, 2, 2, 2, 3, 3, 3]),
            [3, 3],
            DType.int32,
          );

          final targetRows = NDArray.fromList([0, 2], [2], DType.int32);

          // Overwrite row 0 and row 2 to 9
          mat.setIndicesScalar(targetRows, Int32(9), axis: 0);
          expect(mat.toList(), [9, 9, 9, 2, 2, 2, 9, 9, 9]);
        }),
      );
    });

    group('Polymorphic Overload [] and []= Syntax tests (NumPy Equivalence)', () {
      test(
        'operator [] single int extracts direct sub-matrix row views',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Int32List.fromList(List.generate(12, (i) => i)),
            [3, 4],
            DType.int32,
          );

          // NumPy: a[1] extracts row 1 view
          final r1 = a[1];
          expect(r1.shape, [4]);
          expect(r1.toList(), [4, 5, 6, 7]);
        }),
      );

      test(
        'operator [] List<int> extracts fancy row stacks',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Int32List.fromList(List.generate(12, (i) => i)),
            [3, 4],
            DType.int32,
          );

          // NumPy: a[[0, 2]] extracts row 0 and row 2 stacked!
          final fancy =
              a[[
                [0, 2],
              ]];
          expect(fancy.shape, [2, 4]);
          expect(fancy.toList(), [
            0, 1, 2, 3, // row 0
            8, 9, 10, 11, // row 2
          ]);
        }),
      );

      test(
        'operator [] NDArray boolean criteria filters and flattens to 1D',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, -2.0, 3.0, -4.0]),
            [4],
            DType.float64,
          );

          // NumPy: a[a > 0] returns positive elements flattened vector!
          final mask = a > 0.0;
          final positives = a[mask];
          expect(positives.shape, [2]);
          expect(positives.toList(), [1.0, 3.0]);
        }),
      );

      test(
        'operator []= Boolean Mask assignments clips in-place',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([10.0, -1.0, 20.0, -3.0]),
            [4],
            DType.float64,
          );

          // NumPy: mat[mat < 0.0] = 0.0
          final mask = mat < 0.0;
          mat[mask] = 0.0;
          expect(mat.toList(), [10.0, 0.0, 20.0, 0.0]);
        }),
      );

      test(
        'operator []= Fancy list index assignment mutates specific rows stack',
        () => NDArray.scope(() {
          final mat = NDArray.arange(
            0.0,
            6.0,
            dtype: DType.float64,
          ).reshape([3, 2]);
          // [[0,1], [2,3], [4,5]]

          // Mutate row 0 and row 2 in-place to scalar 99.0
          mat[[
                [0, 2],
              ]] =
              99.0;
          expect(mat.toList(), [
            99.0, 99.0, // row 0 overwritten
            2.0, 3.0,
            99.0, 99.0, // row 2 overwritten
          ]);
        }),
      );

      group('Additional NDArray Coverage & Edge Cases', () {
        test(
          'Disposed array checks throw StateError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            a.dispose();
            expect(() => a.fill(Float64(1.0)), throwsStateError);
            expect(() => a.transpose(), throwsStateError);
            expect(() => a[0], throwsStateError);
            expect(() => a[0] = 1.0, throwsStateError);
          }),
        );

        test(
          'Contiguous fill for float32 and int64',
          () => NDArray.scope(() {
            final a = NDArray<double>.create([3], DType.float32);
            a.fill(42.0);
            expect(a.toList(), [42.0, 42.0, 42.0]);

            final b = NDArray<int>.create([3], DType.int64);
            b.fill(99);
            expect(b.toList(), [99, 99, 99]);
          }),
        );

        test(
          'Unsupported selector type for operator[] throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            expect(() => a['invalid'], throwsArgumentError);
          }),
        );

        test(
          'Integer array mask of same shape for operator[] throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final mask = NDArray.fromList([1, 0], [2], DType.int32);
            expect(() => a[mask], throwsArgumentError);
          }),
        );

        test(
          'Mismatched shape integer array selector for operator[] performs fancy take indexing',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final selector = NDArray.fromList([0, 2, 1], [3], DType.int32);

            final res = a[selector];
            expect(res.shape, [3]);
            expect(res.toList(), [10.0, 30.0, 20.0]);
          }),
        );

        test(
          'operator[]= single int index assignment (scalar and array)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            a[0] = 99.0; // scalar assignment to row 0
            expect(a.toList(), [99.0, 99.0, 3.0, 4.0]);

            final val = NDArray.fromList([10.0, 20.0], [2], DType.float64);
            a[1] = val; // array assignment to row 1
            expect(a.toList(), [99.0, 99.0, 10.0, 20.0]);
          }),
        );

        test(
          'operator[]= nested List<List<int>> index assignment with NDArray value',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [3, 2],
              DType.float64,
            );
            final val = NDArray.fromList(
              [99.0, 99.0, 88.0, 88.0],
              [2, 2],
              DType.float64,
            );

            // Target rows 0 and 2 using nested list [[0, 2]]
            a[[
                  [0, 2],
                ]] =
                val;
            expect(a.toList(), [99.0, 99.0, 3.0, 4.0, 88.0, 88.0]);
          }),
        );

        test(
          'operator[]= coordinate length mismatch throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            expect(() => a[[0]] = 99.0, throwsArgumentError);
          }),
        );

        test(
          'operator[]= boolean mask shape mismatch throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final mask = NDArray.fromList(
              [true, false, true],
              [3],
              DType.boolean,
            );
            expect(() => a[mask] = 99.0, throwsArgumentError);
          }),
        );

        test(
          'Integer array mask of same shape for operator[]= throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final mask = NDArray.fromList([1, 0], [2], DType.int32);
            expect(() => a[mask] = 99.0, throwsArgumentError);
          }),
        );

        test(
          'operator[]= mismatched shape integer array selector with NDArray value',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final selector = NDArray.fromList([0, 2], [2], DType.int32);
            final val = NDArray.fromList([99.0, 88.0], [2], DType.float64);

            a[selector] = val;
            expect(a.toList(), [99.0, 2.0, 88.0, 4.0]);
          }),
        );

        test(
          'operator[]= mismatched shape integer array selector with scalar value',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final selector = NDArray.fromList([0, 2], [2], DType.int32);

            a[selector] = 99.0;
            expect(a.toList(), [99.0, 2.0, 99.0, 4.0]);
          }),
        );

        test(
          'operator[]= invalid selector type throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            expect(() => a['invalid'] = 99.0, throwsArgumentError);
          }),
        );
      });
    });
    test(
      'setByMask() with NDArray values and capacity validations',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final mask = NDArray.fromList(
          [true, false, false, true],
          [2, 2],
          DType.boolean,
        );
        final values = NDArray.fromList([99.0, 100.0], [2], DType.float64);
        final insufficientValues = NDArray.fromList([99.0], [1], DType.float64);

        // 1. Incompatible capacity throws ArgumentError
        expect(
          () => parent.setByMask(mask, insufficientValues),
          throwsArgumentError,
        );

        // 2. Valid array value mask mutation
        parent.setByMask(mask, values);
        expect(parent.toList(), [99.0, 2.0, 3.0, 100.0]);
      }),
    );

    test(
      'setIndices() RangeError and ArgumentError boundaries coverage',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final indices = NDArray.fromList([0, 1], [2], DType.int32);
        final values = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final insufficientValues = NDArray.fromList([10.0], [1], DType.float64);

        // 1. Invalid axis throws RangeError
        expect(
          () => parent.setIndices(indices, values, axis: -1),
          throwsRangeError,
        );
        expect(
          () => parent.setIndices(indices, values, axis: 2),
          throwsRangeError,
        );

        // 2. Insufficient values capacity throws ArgumentError
        expect(
          () => parent.setIndices(indices, insufficientValues, axis: 0),
          throwsArgumentError,
        );
      }),
    );
  });
}
