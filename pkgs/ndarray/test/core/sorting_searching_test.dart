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
  group('NDArray Sorting, Searching & Counting Tests', () {
    group('Native QSort direct sort tests', () {
      test(
        'Sort contiguous Float64 row slices',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 2.0, 9.0, 5.0, 7.0]),
            [2, 3],
            DType.float64,
          );

          final sorted = sort(mat, axis: 1);
          expect(sorted.shape, [2, 3]);
          expect(sorted.dtype, DType.float64);
          // Verifies each row is independently sorted in C memory!
          expect(sorted.toList(), [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
        }),
      );

      test(
        'Sort contiguous Int32 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([5, -2, 10, 0]), [
            4,
          ], DType.int32);
          final s = sort(a);
          expect(s.toList(), [-2, 0, 5, 10]);
        }),
      );

      test(
        'Sort contiguous Int64 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int64List.fromList([100, 50, 200]), [
            3,
          ], DType.int64);
          final s = sort(a);
          expect(s.toList(), [50, 100, 200]);
        }),
      );
    });

    group('Complex Lexicographical Sorting tests', () {
      test(
        'Sort Complex128 array lexicographically',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(2.0, 5.0),
              Complex(1.0, 10.0),
              Complex(2.0, 1.0),
              Complex(0.0, 0.0),
            ],
            [4],
            DType.complex128,
          );

          final s = sort(a);
          expect(s.shape, [4]);
          expect(s.dtype, DType.complex128);
          final sList = s.toList();
          expect(sList[0], Complex(0.0, 0.0));
          expect(sList[1], Complex(1.0, 10.0));
          expect(sList[2], Complex(2.0, 1.0)); // comes *before* Complex(2,5)
          expect(sList[3], Complex(2.0, 5.0));
        }),
      );
    });

    group('Axis-Swapping & Internal Axis Sort tests', () {
      test(
        'Sort 2D matrix along column axis 0',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([5.0, 1.0, 4.0, 2.0, 6.0, 3.0]),
            [2, 3],
            DType.float64,
          );

          // Sorting columns across rows!
          final s = sort(mat, axis: 0);
          expect(s.shape, [2, 3]);
          expect(s.toList(), [
            2.0, 1.0, 3.0, // column 0: min(5,2)=2. col 2: min(4,3)=3
            5.0, 6.0, 4.0, // col 0: max(5,2)=5. col 2: max(4,3)=4
          ]);
        }),
      );

      test(
        'Sort 3D tensor along an internal axis',
        () => NDArray.scope(() {
          // shape [2, 2, 2]
          final tensor = NDArray.fromList(
            Float64List.fromList([
              10.0, 2.0, // slice 0, row 0
              5.0, 8.0, // slice 0, row 1
              3.0, 1.0, // slice 1, row 0
              7.0, 4.0, // slice 1, row 1
            ]),
            [2, 2, 2],
            DType.float64,
          );

          final s = sort(tensor, axis: 1); // sort along row axis
          expect(s.shape, [2, 2, 2]);
          expect(s.toList(), [
            5.0, 2.0, // row 0 vs 1 elements sorted
            10.0, 8.0,
            3.0, 1.0,
            7.0, 4.0,
          ]);
        }),
      );

      test(
        'Sort non-contiguous sliced array view',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([10.0, 2.0, 3.0, 4.0, 20.0, 6.0]),
            [2, 3],
            DType.float64,
          );

          // Create a non-contiguous view: column 0 (elements 10.0 and 4.0)
          final colView = mat.slice([Slice.all(), Index(0)]);
          expect(colView.isContiguous, false);

          final s = sort(colView);
          expect(s.shape, [2]);
          expect(s.toList(), [4.0, 10.0]); // sorted perfectly!
        }),
      );
    });

    group('argsort (Indirect index sorting) tests', () {
      test(
        'Argsort 1D Float64 list',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([40.0, 10.0, 30.0, 20.0]),
            [4],
            DType.float64,
          );
          final indices = argsort(a);
          expect(indices.shape, [4]);
          expect(indices.dtype, DType.int64);
          expect(indices.toList(), [
            1,
            3,
            2,
            0,
          ]); // 10(idx 1) < 20(idx 3) < 30(idx 2) < 40(idx 0)
        }),
      );

      test(
        'Argsort complex array indirect ranking',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(5.0, 0.0), Complex(2.0, 3.0), Complex(2.0, 1.0)],
            [3],
            DType.complex128,
          );

          final idx = argsort(a);
          expect(idx.toList(), [
            2,
            1,
            0,
          ]); // Complex(2,1) < Complex(2,3) < Complex(5,0)
        }),
      );

      test(
        'Argsort float32, int32, and int64 lists',
        () => NDArray.scope(() {
          final f32 = NDArray.fromList(
            Float32List.fromList([4.0, 1.0, 3.0, 2.0]),
            [4],
            DType.float32,
          );
          final resF32 = argsort(f32);
          expect(resF32.toList(), [1, 3, 2, 0]);

          final i32 = NDArray.fromList(Int32List.fromList([40, 10, 30, 20]), [
            4,
          ], DType.int32);
          final resI32 = argsort(i32);
          expect(resI32.toList(), [1, 3, 2, 0]);

          final i64 = NDArray.fromList(
            Int64List.fromList([400, 100, 300, 200]),
            [4],
            DType.int64,
          );
          final resI64 = argsort(i64);
          expect(resI64.toList(), [1, 3, 2, 0]);
        }),
      );

      test(
        'Argsort scalar 0D array returns single index 0',
        () => NDArray.scope(() {
          final scalar = NDArray.scalar(99.0, dtype: DType.float64);
          final idx = argsort(scalar);
          expect(idx.shape, []);
          expect(idx.toList(), [0]);
        }),
      );

      test(
        'Argsort invalid target axis throws RangeError',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => argsort(a, axis: 5), throwsRangeError);
          expect(() => argsort(a, axis: -5), throwsRangeError);
        }),
      );
    });

    group('count_nonzero & nonzero tests', () {
      test(
        'count_nonzero flat and axis reduced',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(Int32List.fromList([0, 5, 0, 2, 0, 3]), [
            2,
            3,
          ], DType.int32);

          expect(count_nonzero(mat).scalar, 3);

          final c0 = count_nonzero(mat, axis: 0); // count along columns
          expect(c0.shape, [3]);
          expect(c0.toList(), [1, 1, 1]); // col0 has 2, col1 has 5, col2 has 3
        }),
      );

      test(
        'nonzero indices tuple coordinate arrays',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(Int32List.fromList([0, 9, 0, 4, 0, 0]), [
            2,
            3,
          ], DType.int32);

          final coords = nonzero(mat);
          expect(coords.length, 2); // one per dimension
          expect(coords[0].toList(), [0, 1]); // row coordinates
          expect(coords[1].toList(), [
            1,
            0,
          ]); // col coordinates -> elements are at (0,1) and (1,0)
        }),
      );
    });

    group('where (Ternary Select & SIMD) tests', () {
      test(
        'where condition select with scalar/array broadcasting',
        () => NDArray.scope(() {
          final cond = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final x = NDArray.fromList(
            Float64List.fromList([10.0, 20.0, 30.0, 40.0]),
            [4],
            DType.float64,
          );
          final y = NDArray.fromList(
            Float64List.fromList([-5.0, -6.0, -7.0, -8.0]),
            [4],
            DType.float64,
          );

          final result = where(cond, x, y) as NDArray;
          expect(result.dtype, DType.float64);
          expect(result.shape, [4]);
          expect(result.toList(), [10.0, -6.0, 30.0, -8.0]);
        }),
      );

      test(
        'where behaves as nonzero if x and y are omitted',
        () => NDArray.scope(() {
          final cond = NDArray.fromList(
            [false, true, false, true],
            [4],
            DType.boolean,
          );
          final res = where(cond) as List<NDArray<AnySpec>>;
          expect(res.length, 1);
          expect(res[0].toList(), [1, 3]);
        }),
      );

      test(
        'where with integer inputs',
        () => NDArray.scope(() {
          final cond = NDArray.fromList([true, false], [2], DType.boolean);
          final x = NDArray.fromList([10, 20], [2], DType.int32);
          final y = NDArray.fromList([100, 200], [2], DType.int32);

          final res = where(cond, x, y) as NDArray;
          expect(res.dtype, DType.int32);
          expect(res.toList(), [10, 200]);
        }),
      );

      test(
        'where with Complex inputs',
        () => NDArray.scope(() {
          final cond = NDArray.fromList([true, false], [2], DType.boolean);
          final x = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 2.0)],
            [2],
            DType.complex128,
          );

          final y = NDArray.fromList(
            [Complex(10.0, 10.0), Complex(20.0, 20.0)],
            [2],
            DType.complex128,
          );

          final res = where(cond, x, y) as NDArray;
          expect(res.dtype, DType.complex128);
          final resList = res.toList();
          expect(resList[0], Complex(1.0, 1.0));
          expect(resList[1], Complex(20.0, 20.0));
        }),
      );
    });

    group('argmax / argmin (Extremes Reductions) tests', () {
      test(
        'Global flat argmax and argmin',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([10.0, 50.0, 5.0, 50.0, 20.0]),
            [5],
            DType.float64,
          );

          // In NumPy duplicate max extremes return the *first* occurrence!
          // max is 50.0 at index 1 and 3. NumPy returns 1!
          expect(argmax(a).scalar, 1);
          expect(argmin(a).scalar, 2); // min is 5.0 at index 2
        }),
      );

      test(
        'Argmax along axis reduction in 2D',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([10.0, 30.0, 5.0, 40.0, 15.0, 60.0]),
            [2, 3],
            DType.float64,
          );

          final am0 = argmax(mat, axis: 0); // cols max row indices
          expect(am0.shape, [3]);
          expect(am0.toList(), [
            1,
            0,
            1,
          ]); // col0 max is row 1(40), col1 max row 0(30), col2 max row 1(60)
        }),
      );

      group('Argmin/Argmax Branch Coverage', () {
        test(
          'multiple dtypes',
          () => NDArray.scope(() {
            final i32 = NDArray.fromList([3, 1, 2], [3], DType.int32);
            final i64 = NDArray.fromList([3, 1, 2], [3], DType.int64);
            final u8 = NDArray.fromList([3, 1, 2], [3], DType.uint8);
            final f32 = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float32);

            expect(argmin(i32).scalar, 1);
            expect(argmax(i32).scalar, 0);

            expect(argmin(i64).scalar, 1);
            expect(argmax(i64).scalar, 0);

            expect(argmin(u8).scalar, 1);
            expect(argmax(u8).scalar, 0);

            expect(argmin(f32).scalar, 1);
            expect(argmax(f32).scalar, 0);
          }),
        );

        test(
          'out parameter reuse',
          () => NDArray.scope(() {
            final a = NDArray.fromList([10.0, 50.0, 5.0], [3], DType.float64);

            final outFlat = NDArray.zeros([], DType.int64);
            final resFlat = argmin(a, out: outFlat);
            expect(identical(resFlat, outFlat), true);
            expect(outFlat.scalar, 2);

            final outAxis = NDArray.zeros([1], DType.int64);
            final mat = NDArray.fromList(
              [10.0, 50.0, 5.0],
              [1, 3],
              DType.float64,
            );
            final resAxis = argmin(mat, axis: 1, out: outAxis);
            expect(identical(resAxis, outAxis), true);
            expect(outAxis.toList(), [2]);
          }),
        );

        test(
          'disposed out array throws StateError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final out = NDArray.zeros([], DType.int64);
            out.dispose();

            expect(() => argmin(a, out: out), throwsStateError);
            expect(() => argmax(a, out: out), throwsStateError);
          }),
        );

        test(
          'incompatible out array throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);

            // Incompatible shape
            final outShape = NDArray.zeros([2], DType.int64);
            expect(() => argmin(a, out: outShape), throwsArgumentError);
            expect(() => argmax(a, out: outShape), throwsArgumentError);

            // Incompatible dtype
            final outDType = NDArray.zeros([], DType.float64);
            expect(
              () => argmin(a, out: outDType as dynamic),
              throwsA(anyOf(isA<TypeError>(), isA<ArgumentError>())),
            );
            expect(
              () => argmax(a, out: outDType as dynamic),
              throwsA(anyOf(isA<TypeError>(), isA<ArgumentError>())),
            );
          }),
        );

        test(
          'disposed input array throws StateError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            a.dispose();

            expect(() => argmin(a), throwsStateError);
            expect(() => argmax(a), throwsStateError);
          }),
        );

        test(
          'invalid axis throws ArgumentError/RangeError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            expect(
              () => argmin(a, axis: 2),
              throwsA(anyOf(isA<ArgumentError>(), isA<RangeError>())),
            );
            expect(
              () => argmin(a, axis: -2),
              throwsA(anyOf(isA<ArgumentError>(), isA<RangeError>())),
            );
          }),
        );

        test(
          'empty arrays throws ArgumentError',
          () => NDArray.scope(() {
            final empty = NDArray.zeros([0], DType.float64);
            expect(() => argmin(empty), throwsArgumentError);
            expect(() => argmax(empty), throwsArgumentError);

            expect(() => argmin(empty, axis: 0), throwsArgumentError);
            expect(() => argmax(empty, axis: 0), throwsArgumentError);
          }),
        );

        test(
          'zero-sized axis throws ArgumentError',
          () => NDArray.scope(() {
            final a = NDArray.zeros([2, 0], DType.float64);
            // Reducing along zero-sized axis (axis 1)
            expect(() => argmin(a, axis: 1), throwsArgumentError);
            expect(() => argmax(a, axis: 1), throwsArgumentError);

            // Reducing along non-zero-sized axis (axis 0), but overall size is 0
            expect(() => argmin(a, axis: 0), throwsArgumentError);
            expect(() => argmax(a, axis: 0), throwsArgumentError);
          }),
        );
      });
    });

    group('SortKind custom quicksort/heapsort/mergesort/stable tests', () {
      test(
        'Sort with quicksort',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [5.0, 3.0, 8.0, 1.0, 2.0],
            [5],
            DType.float64,
          );
          final s = sort(a, kind: SortKind.quicksort);
          expect(s.toList(), [1.0, 2.0, 3.0, 5.0, 8.0]);
        }),
      );

      test(
        'Sort with heapsort',
        () => NDArray.scope(() {
          final a = NDArray.fromList([5, 3, 8, 1, 2], [5], DType.int32);
          final s = sort(a, kind: SortKind.heapsort);
          expect(s.toList(), [1, 2, 3, 5, 8]);
        }),
      );

      test(
        'Sort with stable mergesort',
        () => NDArray.scope(() {
          final a = NDArray.fromList([5, 3, 8, 1, 2], [5], DType.int64);
          final s = sort(a, kind: SortKind.mergesort);
          expect(s.toList(), [1, 2, 3, 5, 8]);
        }),
      );

      test(
        'Argsort with different kinds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [5.0, 3.0, 8.0, 1.0, 2.0],
            [5],
            DType.float64,
          );
          final idxQ = argsort(a, kind: SortKind.quicksort);
          final idxH = argsort(a, kind: SortKind.heapsort);
          final idxM = argsort(a, kind: SortKind.mergesort);
          expect(idxQ.toList(), [3, 4, 1, 0, 2]);
          expect(idxH.toList(), [3, 4, 1, 0, 2]);
          expect(idxM.toList(), [3, 4, 1, 0, 2]);
        }),
      );

      test(
        'Boolean sort correctness',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [true, false, true, false, true],
            [5],
            DType.boolean,
          );
          final s = sort(a);
          expect(s.toList(), [false, false, true, true, true]);
        }),
      );
    });

    group('partition & argpartition tests', () {
      test(
        'Simple partition with single kth',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3.0, 4.0, 2.0, 1.0], [4], DType.float64);
          final p = partition(a, 1);
          final pList = p.toList();
          expect(pList[1], 2.0);
          expect(pList[0] <= 2.0, true);
          expect(pList[2] >= 2.0, true);
          expect(pList[3] >= 2.0, true);
        }),
      );

      test(
        'Partition with multiple kth',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [9, 5, 7, 1, 4, 2, 8, 6, 3],
            [9],
            DType.int32,
          );
          final p = partition(a, [2, 6]);
          final pList = p.toList();
          expect(pList[2], 3);
          expect(pList[6], 7);
          for (var i = 0; i < 2; i++) {
            expect(pList[i] <= 3, true);
          }
          for (var i = 3; i < 6; i++) {
            expect(pList[i] >= 3 && pList[i] <= 7, true);
          }
          for (var i = 7; i < 9; i++) {
            expect(pList[i] >= 7, true);
          }
        }),
      );

      test(
        'Partition non-contiguous view',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([5.0, 1.0, 4.0, 2.0, 6.0, 3.0]),
            [2, 3],
            DType.float64,
          );
          final colView = mat.slice([Slice.all(), Index(0)]);
          final p = partition(colView, 0);
          expect(p.toList(), [2.0, 5.0]);
        }),
      );

      test(
        'Argpartition with single kth',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3.0, 4.0, 2.0, 1.0], [4], DType.float64);
          final idx = argpartition(a, 1);
          final aList = a.toList();
          final idxList = idx.toList();
          expect(aList[idxList[1]], 2.0);
          expect(aList[idxList[0]] <= 2.0, true);
          expect(aList[idxList[2]] >= 2.0, true);
          expect(aList[idxList[3]] >= 2.0, true);
        }),
      );

      test(
        'Boolean partition',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final p = partition(a, 1);
          expect(p.toList(), [false, false, true, true]);
        }),
      );
    });

    group('searchsorted tests', () {
      test(
        'Simple searchsorted numeric inputs',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );

          final idxL = searchsorted(
            a,
            NDArray.fromList([2.5, 3.0, 5.5], [3], DType.float64),
            side: SearchSide.left,
          );
          expect(idxL.toList(), [2, 2, 5]);

          final idxR = searchsorted(
            a,
            NDArray.fromList([2.5, 3.0, 5.5], [3], DType.float64),
            side: SearchSide.right,
          );
          expect(idxR.toList(), [2, 3, 5]);
        }),
      );

      test(
        'searchsorted with sorter',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
          final sorter = argsort(a);

          final idxL = searchsorted(
            a,
            NDArray.fromList([1.5, 2.0], [2], DType.float64),
            side: SearchSide.left,
            sorter: sorter,
          );
          expect(idxL.toList(), [1, 1]);
        }),
      );

      test(
        'searchsorted multidimensional input shape matching',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final v = NDArray.fromList(
            [1.5, 2.5, 0.5, 3.5],
            [2, 2],
            DType.float64,
          );

          final idx = searchsorted(a, v);
          expect(idx.shape, [2, 2]);
          expect(idx.toList(), [1, 2, 0, 3]);
        }),
      );

      test(
        'searchsorted complex and boolean support',
        () => NDArray.scope(() {
          final aComp = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 2.0), Complex(3.0, 3.0)],
            [3],
            DType.complex128,
          );

          final vComp = NDArray.fromList(
            [Complex(1.5, 1.5), Complex(2.0, 2.0)],
            [2],
            DType.complex128,
          );

          final idxComp = searchsorted(aComp, vComp, side: SearchSide.left);
          expect(idxComp.toList(), [1, 1]);

          final aBool = NDArray.fromList(
            [false, false, true, true],
            [4],
            DType.boolean,
          );
          final vBool = NDArray.fromList([false, true], [2], DType.boolean);
          final idxBool = searchsorted(aBool, vBool, side: SearchSide.left);
          expect(idxBool.toList(), [0, 2]);
        }),
      );
    });

    group('findIndex tests', () {
      test(
        'Basic forward findIndex',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 2.0, 1.0],
            [5],
            DType.float64,
          );
          expect(findIndex(a, CompareOp.equal, 2.0), [1]);
          expect(findIndex(a, CompareOp.greater, 2.0), [2]);
          expect(findIndex(a, CompareOp.equal, 4.0), null);
        }),
      );

      test(
        'Backward findIndex',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 2.0, 1.0],
            [5],
            DType.float64,
          );
          expect(findIndex(a, CompareOp.equal, 2.0, directions: [-1]), [3]);
          expect(findIndex(a, CompareOp.equal, 1.0, directions: [-1]), [4]);
        }),
      );

      test(
        'Forward findIndex with startCoords',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 2.0, 1.0],
            [5],
            DType.float64,
          );
          expect(findIndex(a, CompareOp.equal, 2.0, startCoords: [2]), [3]);
          expect(findIndex(a, CompareOp.equal, 1.0, startCoords: [1]), [4]);
        }),
      );

      test(
        'Backward findIndex with startCoords',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 2.0, 1.0],
            [5],
            DType.float64,
          );
          expect(
            findIndex(
              a,
              CompareOp.equal,
              2.0,
              startCoords: [2],
              directions: [-1],
            ),
            [1],
          );
          expect(
            findIndex(
              a,
              CompareOp.equal,
              1.0,
              startCoords: [3],
              directions: [-1],
            ),
            [0],
          );
        }),
      );

      test(
        'findIndex on strided/sliced view',
        () => NDArray.scope(() {
          // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] -> slice ::2 -> [1.0, 3.0, 5.0]
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );
          final sliced = a.slice([Slice(start: 0, stop: 6, step: 2)]);
          expect(sliced.shape, [3]);
          expect(sliced.isContiguous, isFalse);

          expect(findIndex(sliced, CompareOp.equal, 3.0), [
            1,
          ]); // flat index in sliced is 1
          expect(findIndex(sliced, CompareOp.equal, 5.0), [2]);
          expect(
            findIndex(sliced, CompareOp.equal, 2.0),
            null,
          ); // 2.0 is skipped

          // Backward on strided
          expect(findIndex(sliced, CompareOp.greater, 2.0, directions: [-1]), [
            2,
          ]); // 5.0 is at index 2
        }),
      );

      test(
        'findIndex 2D with custom directions',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          // Traversal starting at [0, 2] with directions [1, -1] (forward rows, backward cols):
          // [0, 2] (3.0), [0, 1] (2.0), [0, 0] (1.0), [1, 2] (6.0), [1, 1] (5.0), [1, 0] (4.0)

          expect(
            findIndex(
              a,
              CompareOp.greater,
              2.0,
              startCoords: [0, 2],
              directions: [1, -1],
            ),
            [0, 2],
          ); // 3.0 matches immediately, flat index 2

          expect(
            findIndex(
              a,
              CompareOp.greater,
              4.0,
              startCoords: [0, 2],
              directions: [1, -1],
            ),
            [1, 2],
          ); // 6.0 is at [1, 2], flat index 5

          expect(
            findIndex(
              a,
              CompareOp.equal,
              2.0,
              startCoords: [1, 2],
              directions: [1, -1],
            ),
            null,
          ); // starts at [1, 2], doesn't visit row 0
        }),
      );

      test(
        'findIndex bounds and empty validation',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(
            () => findIndex(a, CompareOp.equal, 1.0, startCoords: [-1]),
            throwsRangeError,
          );
          expect(
            () => findIndex(a, CompareOp.equal, 1.0, startCoords: [2]),
            throwsRangeError,
          );
          expect(
            () => findIndex(a, CompareOp.equal, 1.0, startCoords: [0, 0]),
            throwsArgumentError,
          ); // wrong rank

          final empty = NDArray.create([0], DType.float64);
          expect(findIndex(empty, CompareOp.equal, 1.0), null);
        }),
      );
    });
    test(
      'count_nonzero() fallback path on non-contiguous views',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 0.0, 3.0, 0.0, 5.0, 6.0]),
          [2, 3],
          DType.float64,
        );

        // Non-contiguous column slice: select column 0 (elements [1.0, 0.0] -> strides are [3, 1], column 0 is data[0], data[3])
        // Column 2 is: [3.0, 6.0] (nonzero count is 2).
        final view = a.slice([Slice.all(), Index(2)]);
        expect(view.isContiguous, false);
        expect(view.toList(), [3.0, 6.0]);

        // In the old code, count_nonzero(view) would incorrectly return 0 due to discarded recursion yields!
        expect(count_nonzero(view).scalar, 2);
      }),
    );

    test(
      'qsort floating-point NaN sorting safety and stability',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [3.0, double.nan, 1.0, double.nan, 2.0],
          [5],
          DType.float64,
        );

        final b = sort(a);

        final bList = b.toList();
        // The 3 non-NaN elements must be sorted ascending: 1.0, 2.0, 3.0
        expect(bList[0], 1.0);
        expect(bList[1], 2.0);
        expect(bList[2], 3.0);

        // The NaNs must be pushed consistently to the end of the array
        expect(bList[3].isNaN, true);
        expect(bList[4].isNaN, true);
      }),
    );

    group('argwhere Tests', () {
      test(
        'argwhere with 1D, 2D, and 3D arrays of various DTypes',
        () => NDArray.scope(() {
          final a1 = NDArray.fromList([0, 1, 0, 3], [4], DType.int32);
          final res1 = argwhere(a1);
          expect(res1.shape, [2, 1]);
          expect(res1.toList(), [1, 3]);

          final a2 = NDArray.fromList(
            [0.0, 1.5, 0.0, 0.0, 2.5, 0.0],
            [2, 3],
            DType.float64,
          );
          final res2 = argwhere(a2);
          expect(res2.shape, [2, 2]);
          expect(res2.toList(), [0, 1, 1, 1]);

          final a3 = NDArray.fromList(
            [0, 0, 1, 0, 0, 0, 0, 1],
            [2, 2, 2],
            DType.uint8,
          );
          final res3 = argwhere(a3);
          expect(res3.shape, [2, 3]);
          expect(res3.toList(), [0, 1, 0, 1, 1, 1]);
        }),
      );

      test(
        'argwhere with 0-D scalar arrays (both zero and non-zero)',
        () => NDArray.scope(() {
          final nonzeroScalar = NDArray.scalar(5.0, dtype: DType.float64);
          final res1 = argwhere(nonzeroScalar);
          expect(res1.shape, [1, 0]);
          expect(res1.toList(), <int>[]);

          final zeroScalar = NDArray.scalar(0.0, dtype: DType.float64);
          final res2 = argwhere(zeroScalar);
          expect(res2.shape, [0, 0]);
          expect(res2.toList(), <int>[]);
        }),
      );

      test(
        'argwhere with strided non-contiguous views',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            [0, 1, 0, 2, 0, 0, 0, 0, 3],
            [3, 3],
            DType.int32,
          );

          final transposed = mat.transpose();
          expect(transposed.isContiguous, isFalse);
          final res = argwhere(transposed);
          expect(res.shape, [3, 2]);
          expect(res.toList(), [0, 1, 1, 0, 2, 2]);

          final sliced = mat.slice([
            Slice(start: 0, stop: 3, step: 2),
            Slice.all(),
          ]);
          expect(sliced.isContiguous, isFalse);
          final resSliced = argwhere(sliced);
          expect(resSliced.shape, [2, 2]);
          expect(resSliced.toList(), [0, 1, 1, 2]);
        }),
      );

      test(
        'argwhere on disposed array throws StateError',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          a.dispose();
          expect(() => argwhere(a), throwsStateError);
        }),
      );
    });

    group('nonzero Strided & DType Tests', () {
      test(
        'nonzero on strided non-contiguous views',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            [0, 1, 0, 2, 0, 0, 0, 0, 3],
            [3, 3],
            DType.int32,
          );
          final transposed = mat.transpose();
          final coords = nonzero(transposed);
          expect(coords.length, 2);
          expect(coords[0].toList(), [0, 1, 2]);
          expect(coords[1].toList(), [1, 0, 2]);
        }),
      );

      test(
        'nonzero on different DTypes',
        () => NDArray.scope(() {
          final f64 = NDArray.fromList([0.0, 1.5, 0.0], [3], DType.float64);
          final resF64 = nonzero(f64);
          expect(resF64[0].toList(), [1]);

          final c128 = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 2.0)],
            [3],
            DType.complex128,
          );
          final resC128 = nonzero(c128);
          expect(resC128[0].toList(), [1, 2]);

          final u8 = NDArray.fromList([0, 255, 0], [3], DType.uint8);
          final resU8 = nonzero(u8);
          expect(resU8[0].toList(), [1]);

          final boolean = NDArray.fromList(
            [false, true, false],
            [3],
            DType.boolean,
          );
          final resBool = nonzero(boolean);
          expect(resBool[0].toList(), [1]);

          final i16 = NDArray.fromList([0, 10, 0], [3], DType.int16);
          final resI16 = nonzero(i16);
          expect(resI16[0].toList(), [1]);
        }),
      );
    });

    group('partition & argpartition Expansion', () {
      test(
        'partitioning on Complex128 and Complex64 arrays',
        () => NDArray.scope(() {
          final c128 = NDArray.fromList(
            [
              Complex(3.0, 3.0),
              Complex(1.0, 1.0),
              Complex(4.0, 4.0),
              Complex(2.0, 2.0),
            ],
            [4],
            DType.complex128,
          );
          final p128 = partition(c128, 1);
          final p128List = p128.toList();
          expect(p128List[1], Complex(2.0, 2.0));
          expect(p128List[0].real <= 2.0, true);
          expect(p128List[2].real >= 2.0, true);

          final c64 = NDArray.fromList(
            [
              Complex(3.0, 3.0),
              Complex(1.0, 1.0),
              Complex(4.0, 4.0),
              Complex(2.0, 2.0),
            ],
            [4],
            DType.complex64,
          );
          final p64 = partition(c64, 1);
          final p64List = p64.toList();
          expect(p64List[1], Complex(2.0, 2.0));
        }),
      );

      test(
        'negative kth pivots',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3, 4, 2, 1], [4], DType.int32);
          final p1 = partition(a, -1);
          expect(p1.toList()[3], 4);
          for (var i = 0; i < 3; i++) {
            expect(p1.toList()[i] <= 4, true);
          }

          final p2 = partition(a, -2);
          expect(p2.toList()[2], 3);

          final p3 = partition(a, [-2, -1]);
          expect(p3.toList()[2], 3);
          expect(p3.toList()[3], 4);
        }),
      );

      test(
        'edge cases like uniqueK being empty',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3, 4, 2, 1], [4], DType.int32);
          final p = partition(a, <int>[]);
          expect(p.toList(), [3, 4, 2, 1]);

          final idx = argpartition(a, <int>[]);
          expect(idx.toList(), [0, 1, 2, 3]);
        }),
      );

      test(
        'out-of-bounds kth values or invalid types throw correct errors',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3, 4, 2, 1], [4], DType.int32);
          expect(() => partition(a, 4), throwsRangeError);
          expect(() => partition(a, -5), throwsRangeError);
          expect(() => partition(a, 'invalid'), throwsArgumentError);

          expect(() => argpartition(a, 4), throwsRangeError);
          expect(() => argpartition(a, -5), throwsRangeError);
          expect(() => argpartition(a, 'invalid'), throwsArgumentError);
        }),
      );
    });

    group('sort & argsort Expansion', () {
      test(
        'sorting on Float32, UInt8, and Complex64',
        () => NDArray.scope(() {
          final f32 = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float32);
          expect(sort(f32).toList(), [1.0, 2.0, 3.0]);

          final u8 = NDArray.fromList([3, 1, 2], [3], DType.uint8);
          expect(sort(u8).toList(), [1, 2, 3]);

          final c64 = NDArray.fromList(
            [Complex(3.0, 3.0), Complex(1.0, 1.0), Complex(2.0, 2.0)],
            [3],
            DType.complex64,
          );
          expect(sort(c64).toList(), [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
          ]);
        }),
      );

      test(
        'sorting empty arrays and 0-D scalar arrays',
        () => NDArray.scope(() {
          final empty = NDArray.create([0], DType.float64);
          expect(sort(empty).toList(), <double>[]);

          final scalar = NDArray.scalar(5.0, dtype: DType.float64);
          expect(sort(scalar).toList(), [5.0]);
        }),
      );
    });

    group('where Expansion', () {
      test(
        'where using a pre-allocated out recycler buffer',
        () => NDArray.scope(() {
          final cond = NDArray.fromList([true, false], [2], DType.boolean);
          final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final y = NDArray.fromList([10.0, 20.0], [2], DType.float64);
          final out = NDArray.zeros([2], DType.float64);

          final res = where(cond, x, y, out) as NDArray;
          expect(identical(res, out), true);
          expect(out.toList(), [1.0, 20.0]);
        }),
      );

      test(
        'where error conditions',
        () => NDArray.scope(() {
          final cond = NDArray.fromList([true, false], [2], DType.boolean);
          final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final y = NDArray.fromList([10.0, 20.0], [2], DType.float64);

          final cond9 = NDArray<Boolean>.zeros([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
          ], DType.boolean);
          final x9 = NDArray.zeros([1, 1, 1, 1, 1, 1, 1, 1, 2], DType.float64);
          final y9 = NDArray.zeros([1, 1, 1, 1, 1, 1, 1, 1, 2], DType.float64);
          expect(() => where(cond9, x9, y9), throwsUnsupportedError);

          final outBadShape = NDArray.zeros([3], DType.float64);
          expect(() => where(cond, x, y, outBadShape), throwsArgumentError);

          final outBadDType = NDArray.zeros([2], DType.int32);
          expect(
            () => where(cond, x, y, outBadDType as dynamic),
            throwsArgumentError,
          );

          final condBad = NDArray.fromList([1.0, 0.0], [2], DType.float64);
          expect(
            () => where(condBad as dynamic, x, y),
            throwsA(anyOf(isA<ArgumentError>(), isA<TypeError>())),
          );

          final xMismatch = NDArray.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          expect(() => where(cond, xMismatch, y), throwsArgumentError);
        }),
      );
    });
    group('Workstream 4: Sorting & Searching Enhancements', () {
      test(
        'sort, argsort, partition, argpartition with out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
          final outSort = NDArray.zeros([3], DType.float64);
          final resSort = sort(a, out: outSort);
          expect(identical(resSort, outSort), isTrue);
          expect(outSort.toList(), [1.0, 2.0, 3.0]);

          final outArgSort = NDArray.zeros([3], DType.int64);
          final resArgSort = argsortAs(a, DType.int64, out: outArgSort);
          expect(identical(resArgSort, outArgSort), isTrue);
          expect(outArgSort.toList(), [1, 2, 0]);

          final outPart = NDArray.zeros([3], DType.float64);
          final resPart = partition(a, 1, out: outPart);
          expect(identical(resPart, outPart), isTrue);
          expect(outPart.toList()[1], 2.0);

          final outArgPart = NDArray.zeros([3], DType.int64);
          final resArgPart = argpartitionAs(a, 1, DType.int64, out: outArgPart);
          expect(identical(resArgPart, outArgPart), isTrue);
        }),
      );

      test(
        'searchsorted with SearchSide enum and out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );
          final v = NDArray.fromList([2.0, 3.5], [2], DType.float64);
          final outLeft = NDArray.zeros([2], DType.int64);
          final resLeft = searchsortedAs(
            a,
            v,
            DType.int64,
            side: SearchSide.left,
            out: outLeft,
          );
          expect(identical(resLeft, outLeft), isTrue);
          expect(outLeft.toList(), [1, 3]);

          final outRight = NDArray.zeros([2], DType.int64);
          final resRight = searchsortedAs(
            a,
            v,
            DType.int64,
            side: SearchSide.right,
            out: outRight,
          );
          expect(identical(resRight, outRight), isTrue);
          expect(outRight.toList(), [2, 3]);
        }),
      );

      test(
        'argmax and argmin with keepdims: true',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            [1.0, 5.0, 2.0, 4.0, 0.0, 3.0],
            [2, 3],
            DType.float64,
          );
          final max0 = argmax(mat, axis: 0, keepdims: true);
          expect(max0.shape, [1, 3]);
          expect(max0.toList(), [1, 0, 1]);

          final min1 = argmin(mat, axis: 1, keepdims: true);
          expect(min1.shape, [2, 1]);
          expect(min1.toList(), [0, 1]);

          final maxFlat = argmax(mat, keepdims: true);
          expect(maxFlat.shape, [1, 1]);
          expect(maxFlat.toList(), [1]);
        }),
      );

      test(
        'Set operations with out parameter and non-contiguous inputs',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 2, 3, 4], [5], DType.int64);
          final b = NDArray.fromList([2, 4, 6], [3], DType.int64);

          // unique
          final outU = NDArray.zeros([4], DType.int64);
          final resU = unique(a, out: outU) as NDArray<AnySpec>;
          expect(identical(resU, outU), isTrue);
          expect(outU.toList(), [1, 2, 3, 4]);

          // intersect1d
          final outInter = NDArray.zeros([2], DType.int64);
          final resInter = intersect1d(a, b, out: outInter);
          expect(identical(resInter, outInter), isTrue);
          expect(outInter.toList(), [2, 4]);

          // setdiff1d
          final outDiff = NDArray.zeros([2], DType.int64);
          final resDiff = setdiff1d(a, b, out: outDiff);
          expect(identical(resDiff, outDiff), isTrue);
          expect(outDiff.toList(), [1, 3]);

          // setxor1d
          final outXor = NDArray.zeros([3], DType.int64);
          final resXor = setxor1d(a, b, out: outXor);
          expect(identical(resXor, outXor), isTrue);
          expect(outXor.toList(), [1, 3, 6]);

          // union1d
          final outUnion = NDArray.zeros([5], DType.int64);
          final resUnion = union1d(a, b, out: outUnion);
          expect(identical(resUnion, outUnion), isTrue);
          expect(outUnion.toList(), [1, 2, 3, 4, 6]);

          // isin
          final outIsin = NDArray<Boolean>.zeros([5], DType.boolean);
          final resIsin = isin(a, b, out: outIsin);
          expect(identical(resIsin, outIsin), isTrue);
          expect(outIsin.toList(), [false, true, true, false, true]);
        }),
      );
    });
  });
}
