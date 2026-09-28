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
  group('Exceptions Hierarchy Tests', () {
    test('Exception classes inheritance and toString', () {
      const ndEx = NdArrayException('base message');
      expect(ndEx, isA<Exception>());
      expect(ndEx.toString(), 'NdArrayException: base message');

      const linEx = LinAlgException('linalg error');
      expect(linEx, isA<NdArrayException>());
      expect(linEx.toString(), 'LinAlgException: linalg error');

      const singEx = SingularMatrixException('singular matrix');
      expect(singEx, isA<LinAlgException>());
      expect(singEx, isA<NdArrayException>());
      expect(singEx.toString(), 'SingularMatrixException: singular matrix');

      const noRealEx = NoRealSolutionException('no real solution');
      expect(noRealEx, isA<NdArrayException>());
      expect(noRealEx.toString(), 'NoRealSolutionException: no real solution');

      const iterEx = IterationsExceededException('max iterations');
      expect(iterEx, isA<LinAlgException>());
      expect(iterEx.toString(), 'IterationsExceededException: max iterations');

      const nonPosDefEx = NonPositiveDefiniteException('non positive definite');
      expect(nonPosDefEx, isA<LinAlgException>());
      expect(
        nonPosDefEx.toString(),
        'NonPositiveDefiniteException: non positive definite',
      );
    });
  });

  group('Comprehensive Sorting & Argsort Tests', () {
    test(
      'Sort across all 15 DTypes 1D',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final arr = NDArray.fromList([4.0, 1.0, 3.0, 2.0], [4], dtype);
          final sorted = sort(arr);
          expect(sorted.toList(), [1.0, 2.0, 3.0, 4.0]);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final arr = NDArray.fromList([4, 1, 3, 2], [4], dtype);
          final sorted = sort(arr);
          expect(sorted.toList().map((e) => (e as num).toInt()).toList(), [
            1,
            2,
            3,
            4,
          ]);
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final arr = NDArray.fromList(
            [
              Complex(3.0, 1.0),
              Complex(1.0, 5.0),
              Complex(2.0, -1.0),
              Complex(1.0, 2.0),
            ],
            [4],
            dtype,
          );
          final sorted = sort(arr);
          expect(sorted.toList(), [
            Complex(1.0, 2.0),
            Complex(1.0, 5.0),
            Complex(2.0, -1.0),
            Complex(3.0, 1.0),
          ]);
        }

        final arrBool = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final sortedBool = sort(arrBool);
        expect(sortedBool.toList(), [false, false, true, true]);
      }),
    );

    test(
      'Argsort across all 15 DTypes 1D',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final arr = NDArray.fromList([40.0, 10.0, 30.0, 20.0], [4], dtype);
          final indices = argsort(arr);
          expect(indices.toList(), [1, 3, 2, 0]);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final arr = NDArray.fromList([40, 10, 30, 20], [4], dtype);
          final indices = argsort(arr);
          expect(indices.toList(), [1, 3, 2, 0]);
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final arr = NDArray.fromList(
            [
              Complex(3.0, 1.0),
              Complex(1.0, 5.0),
              Complex(2.0, -1.0),
              Complex(1.0, 2.0),
            ],
            [4],
            dtype,
          );
          final indices = argsort(arr);
          expect(indices.toList(), [3, 1, 2, 0]);
        }

        final arrBool = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final indicesBool = argsort(arrBool);
        expect(indicesBool.toList(), [1, 3, 0, 2]);
      }),
    );

    test(
      'Sort and Argsort along different axes (0, 1, 2, negative axes)',
      () => NDArray.scope(() {
        final a2d = NDArray.fromList(
          [9.0, 2.0, 7.0, 4.0, 8.0, 1.0],
          [2, 3],
          DType.float64,
        );

        final sortedAxis0 = sort(a2d, axis: 0);
        expect(sortedAxis0.toList(), [4.0, 2.0, 1.0, 9.0, 8.0, 7.0]);

        final sortedAxisNeg2 = sort(a2d, axis: -2);
        expect(sortedAxisNeg2.toList(), sortedAxis0.toList());

        final sortedAxis1 = sort(a2d, axis: 1);
        expect(sortedAxis1.toList(), [2.0, 7.0, 9.0, 1.0, 4.0, 8.0]);

        final argsortAxis0 = argsort(a2d, axis: 0);
        expect(argsortAxis0.toList(), [1, 0, 1, 0, 1, 0]);

        final argsortAxis1 = argsort(a2d, axis: 1);
        expect(argsortAxis1.toList(), [1, 2, 0, 2, 0, 1]);

        final a3d = NDArray.fromList(
          [5, 2, 8, 1, 6, 3, 7, 4],
          [2, 2, 2],
          DType.int32,
        );

        final sorted3dAxis1 = sort(a3d, axis: 1);
        expect(sorted3dAxis1.shape, [2, 2, 2]);
        expect(sorted3dAxis1.toList(), [5, 1, 8, 2, 6, 3, 7, 4]);
      }),
    );

    test(
      'Sort and Argsort on non-contiguous strided arrays',
      () => NDArray.scope(() {
        final base = NDArray.fromList(
          [10, 50, 20, 40, 30, 60],
          [6],
          DType.int32,
        );
        final view = base.slice([Slice(step: 2)]);
        expect(view.isContiguous, false);

        final sorted = sort(view);
        expect(sorted.toList(), [10, 20, 30]);

        final indices = argsort(view);
        expect(indices.toList(), [0, 1, 2]);

        final base2 = NDArray.fromList(
          [30, 50, 10, 40, 20, 60],
          [6],
          DType.float64,
        );
        final revView = base2.slice([Slice(step: -2)]);
        expect(revView.isContiguous, false);
        expect(revView.toList(), [60.0, 40.0, 50.0]);

        final sortedRev = sort(revView);
        expect(sortedRev.toList(), [40.0, 50.0, 60.0]);
      }),
    );

    test(
      'Sort and Argsort with SortKind variations',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [5.0, 2.0, 9.0, 1.0, 7.0],
          [5],
          DType.float64,
        );
        for (final kind in [
          SortKind.quicksort,
          SortKind.heapsort,
          SortKind.mergesort,
          SortKind.stable,
        ]) {
          final sorted = sort(a, kind: kind);
          expect(sorted.toList(), [1.0, 2.0, 5.0, 7.0, 9.0]);

          final indices = argsort(a, kind: kind);
          expect(indices.toList(), [3, 1, 0, 4, 2]);
        }
      }),
    );

    test(
      'Sort and Argsort with out buffer and edge cases (0D, empty, disposed, error handling)',
      () => NDArray.scope(() {
        final scalar = NDArray.scalar(42.0, dtype: DType.float64);
        final sortedScalar = sort(scalar);
        expect(sortedScalar.shape, <int>[]);
        expect(sortedScalar.scalar, 42.0);

        final outScalar = NDArray.create(<int>[], DType.float64);
        sort(scalar, out: outScalar);
        expect(outScalar.scalar, 42.0);

        final argsortScalar = argsort(scalar);
        expect(argsortScalar.scalar, 0);

        final outArgsortScalar = NDArray.create(<int>[], DType.int32);
        argsort(scalar, out: outArgsortScalar);
        expect(outArgsortScalar.scalar, 0);

        final empty = NDArray<Float64>.create([0], DType.float64);
        final sortedEmpty = sort(empty);
        expect(sortedEmpty.shape, [0]);
        expect(argsort(empty).shape, [0]);

        final a = NDArray.fromList([3, 1, 2], [3], DType.int32);
        final outSort = NDArray.create([3], DType.int32);
        final resSort = sort(a, out: outSort);
        expect(identical(resSort, outSort), true);
        expect(outSort.toList(), [1, 2, 3]);

        final outArgsort64 = NDArray.create([3], DType.int64);
        final resArgsort64 = argsortAs(a, DType.int64, out: outArgsort64);
        expect(identical(resArgsort64, outArgsort64), true);
        expect(outArgsort64.toList(), [1, 2, 0]);

        final badOutShape = NDArray.create([2], DType.int32);
        expect(() => sort(a, out: badOutShape), throwsA(isA<ArgumentError>()));
        expect(
          () => argsort(a, out: badOutShape),
          throwsA(isA<ArgumentError>()),
        );

        final badOutDType = NDArray<Float64>.create([3], DType.float64);
        expect(
          () => sort(a, out: badOutDType as dynamic),
          throwsA(anyOf(isA<ArgumentError>(), isA<TypeError>())),
        );

        expect(() => sort(a, axis: 5), throwsA(isA<RangeError>()));
        expect(() => argsort(a, axis: -3), throwsA(isA<RangeError>()));

        final disposedArr = NDArray.fromList([1, 2], [2], DType.int32)
          ..dispose();
        expect(() => sort(disposedArr), throwsA(isA<StateError>()));
        expect(() => argsort(disposedArr), throwsA(isA<StateError>()));
      }),
    );
  });

  group('Comprehensive Partition & Argpartition Tests', () {
    test(
      'Partition across all 15 DTypes',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final arr = NDArray.fromList(
            [50.0, 10.0, 40.0, 20.0, 30.0],
            [5],
            dtype,
          );
          final p = partition(arr, 2);
          expect(p.shape, [5]);
          final pivot = (p.getCellFlat(2) as num).toDouble();
          expect(pivot, 30.0);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final arr = NDArray.fromList([50, 10, 40, 20, 30], [5], dtype);
          final p = partition(arr, 2);
          expect(p.shape, [5]);
          final pivot = (p.getCellFlat(2) as num).toInt();
          expect(pivot, 30);
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final arr = NDArray.fromList(
            [
              Complex(4.0, 1.0),
              Complex(1.0, 2.0),
              Complex(3.0, 0.0),
              Complex(2.0, 5.0),
            ],
            [4],
            dtype,
          );
          final p = partition(arr, 1);
          expect(p.shape, [4]);
          expect(p.getCellFlat(1), Complex(2.0, 5.0));
        }

        final arrBool = NDArray.fromList(
          [true, false, true, false, true],
          [5],
          DType.boolean,
        );
        final pBool = partition(arrBool, 1);
        expect(pBool.shape, [5]);
        expect(pBool.getCellFlat(0), false);
        expect(pBool.getCellFlat(1), false);
      }),
    );

    test(
      'Argpartition across all 15 DTypes',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final arr = NDArray.fromList(
            [50.0, 10.0, 40.0, 20.0, 30.0],
            [5],
            dtype,
          );
          final indices = argpartition(arr, 2);
          expect(indices.shape, [5]);
          final pivot = (arr.getCellFlat(indices.getCellFlat(2)) as num)
              .toDouble();
          expect(pivot, 30.0);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final arr = NDArray.fromList([50, 10, 40, 20, 30], [5], dtype);
          final indices = argpartition(arr, 2);
          expect(indices.shape, [5]);
          final pivot = (arr.getCellFlat(indices.getCellFlat(2)) as num)
              .toInt();
          expect(pivot, 30);
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final arr = NDArray.fromList(
            [
              Complex(4.0, 1.0),
              Complex(1.0, 2.0),
              Complex(3.0, 0.0),
              Complex(2.0, 5.0),
            ],
            [4],
            dtype,
          );
          final indices = argpartition(arr, 1);
          expect(indices.shape, [4]);
          expect(arr.getCellFlat(indices.getCellFlat(1)), Complex(2.0, 5.0));
        }

        final arrBool = NDArray.fromList(
          [true, false, true, false, true],
          [5],
          DType.boolean,
        );
        final indicesBool = argpartition(arrBool, 1);
        expect(indicesBool.shape, [5]);
        expect(arrBool.getCellFlat(indicesBool.getCellFlat(0)), false);
        expect(arrBool.getCellFlat(indicesBool.getCellFlat(1)), false);
      }),
    );

    test(
      'Partition & Argpartition with negative kth, multi-kth list, 2D/3D axes, and out buffer',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [9.0, 1.0, 7.0, 3.0, 5.0],
          [5],
          DType.float64,
        );

        final pNeg = partition(a, -2);
        expect(pNeg.getCellFlat(3), 7.0);

        final pMulti = partition(a, [1, 3]);
        expect(pMulti.getCellFlat(1), 3.0);
        expect(pMulti.getCellFlat(3), 7.0);

        final a2d = NDArray.fromList(
          [9.0, 2.0, 1.0, 8.0, 5.0, 4.0],
          [3, 2],
          DType.float64,
        );
        final p2d = partition(a2d, 1, axis: 0);
        expect(p2d.shape, [3, 2]);

        final out64 = NDArray.create([5], DType.int64);
        argpartitionAs(a, 2, DType.int64, out: out64);
        expect(out64.dtype, DType.int64);

        final scalar = NDArray.scalar(10, dtype: DType.int32);
        expect(partition(scalar, 0).scalar, 10);
        expect(argpartition(scalar, 0).scalar, 0);

        expect(() => partition(a, 10), throwsA(isA<RangeError>()));
        expect(() => partition(a, 'invalid'), throwsA(isA<ArgumentError>()));
        expect(() => partition(a, 2, axis: 5), throwsA(isA<RangeError>()));
      }),
    );
  });

  group('Comprehensive Searchsorted Tests', () {
    test(
      'Searchsorted across all 15 DTypes with left and right side',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final a = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5],
            dtype,
          );
          final v = NDArray.fromList([5.0, 20.0, 35.0, 60.0], [4], dtype);
          final idxLeft = searchsorted(a, v, side: SearchSide.left);
          expect(idxLeft.toList(), [0, 1, 3, 5]);
          final idxRight = searchsorted(a, v, side: SearchSide.right);
          expect(idxRight.toList(), [0, 2, 3, 5]);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final a = NDArray.fromList([10, 20, 30, 40, 50], [5], dtype);
          final v = NDArray.fromList([5, 20, 35, 60], [4], dtype);
          final idxLeft = searchsorted(a, v, side: SearchSide.left);
          expect(idxLeft.toList(), [0, 1, 3, 5]);
          final idxRight = searchsorted(a, v, side: SearchSide.right);
          expect(idxRight.toList(), [0, 2, 3, 5]);
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final a = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(2.0, 1.0),
              Complex(2.0, 5.0),
              Complex(3.0, 0.0),
            ],
            [4],
            dtype,
          );
          final v = NDArray.fromList(
            [Complex(2.0, 1.0), Complex(2.0, 3.0)],
            [2],
            dtype,
          );
          final idxLeft = searchsorted(a, v, side: SearchSide.left);
          expect(idxLeft.toList(), [1, 2]);
        }

        final aBool = NDArray.fromList(
          [false, false, true, true],
          [4],
          DType.boolean,
        );
        final vBool = NDArray.fromList([false, true], [2], DType.boolean);
        final idxLeft = searchsorted(aBool, vBool, side: SearchSide.left);
        expect(idxLeft.toList(), [0, 2]);
        final idxRight = searchsorted(aBool, vBool, side: SearchSide.right);
        expect(idxRight.toList(), [2, 4]);
      }),
    );

    test(
      'Searchsorted with sorter, multi-dimensional queries, and error cases',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [30.0, 10.0, 50.0, 20.0, 40.0],
          [5],
          DType.float64,
        );
        final sorter = NDArray.fromList([1, 3, 0, 4, 2], [5], DType.int32);
        final v2d = NDArray.fromList(
          [15.0, 30.0, 45.0, 5.0],
          [2, 2],
          DType.float64,
        );

        final res = searchsorted(a, v2d, sorter: sorter);
        expect(res.shape, [2, 2]);
        expect(res.toList(), [1, 2, 4, 0]);

        final out64 = NDArray.create([2, 2], DType.int64);
        searchsortedAs(a, v2d, DType.int64, sorter: sorter, out: out64);
        expect(out64.dtype, DType.int64);
        expect(out64.toList(), [1, 2, 4, 0]);

        final a2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(() => searchsorted(a2d, v2d), throwsA(isA<ArgumentError>()));

        final badSorter = NDArray.fromList([0, 1], [2], DType.int32);
        expect(
          () => searchsorted(a, v2d, sorter: badSorter),
          throwsA(isA<ArgumentError>()),
        );

        final vTypeMismatch = NDArray.fromList([1, 2], [2], DType.int32);
        expect(
          () => searchsorted(a, vTypeMismatch as dynamic),
          throwsA(anyOf(isA<ArgumentError>(), isA<TypeError>())),
        );
      }),
    );
  });

  group('Comprehensive Where Tests', () {
    test(
      'Where ternary select across all 15 DTypes and broadcasting',
      () => NDArray.scope(() {
        final cond = NDArray.fromList(
          [true, false, true, false],
          [2, 2],
          DType.boolean,
        );

        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype);
          final y = NDArray.fromList([10.0, 20.0, 30.0, 40.0], [2, 2], dtype);
          final res = where(cond, x, y) as NDArray;
          expect(res.toList(), [1.0, 20.0, 3.0, 40.0]);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final x = NDArray.fromList([1, 2, 3, 4], [2, 2], dtype);
          final y = NDArray.fromList([10, 20, 30, 40], [2, 2], dtype);
          final res = where(cond, x, y) as NDArray;
          expect(res.toList().map((e) => (e as num).toInt()).toList(), [
            1,
            20,
            3,
            40,
          ]);
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final x = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(2.0, 2.0),
              Complex(3.0, 3.0),
              Complex(4.0, 4.0),
            ],
            [2, 2],
            dtype,
          );
          final y = NDArray.fromList(
            [
              Complex(-1.0, -1.0),
              Complex(-2.0, -2.0),
              Complex(-3.0, -3.0),
              Complex(-4.0, -4.0),
            ],
            [2, 2],
            dtype,
          );
          final res = where(cond, x, y) as NDArray;
          expect(res.toList(), [
            Complex(1.0, 1.0),
            Complex(-2.0, -2.0),
            Complex(3.0, 3.0),
            Complex(-4.0, -4.0),
          ]);
        }

        final xBool = NDArray.fromList(
          [true, true, true, true],
          [2, 2],
          DType.boolean,
        );
        final yBool = NDArray.fromList(
          [false, false, false, false],
          [2, 2],
          DType.boolean,
        );
        final resBool = where(cond, xBool, yBool) as NDArray;
        expect(resBool.toList(), [1, 0, 1, 0]);
      }),
    );

    test(
      'Where coordinate selector mode and error cases',
      () => NDArray.scope(() {
        final cond = NDArray.fromList(
          [true, false, true, true],
          [2, 2],
          DType.boolean,
        );
        final coords = where(cond) as List<NDArray<AnySpec>>;
        expect(coords.length, 2);
        expect(coords[0].toList(), [0, 1, 1]);
        expect(coords[1].toList(), [0, 0, 1]);

        final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final y = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final c1d = NDArray.fromList([true, false], [2], DType.boolean);

        expect(() => where(c1d, x, null), throwsA(isA<ArgumentError>()));
        expect(() => where(c1d, null, y), throwsA(isA<ArgumentError>()));

        final outRecycler = NDArray<Float64>.create([2], DType.float64);
        expect(
          () => where(c1d, null, null, outRecycler),
          throwsA(isA<ArgumentError>()),
        );
      }),
    );
  });

  group('Comprehensive Nonzero, Argwhere & Count Nonzero Tests', () {
    test(
      'Nonzero and Argwhere across 0D, 1D, 2D, 3D and all 15 DTypes',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final a = NDArray.fromList([0.0, 5.0, 0.0, 8.0, 0.0], [5], dtype);
          final nz = nonzero(a);
          expect(nz[0].toList(), [1, 3]);
          expect(argwhere(a).shape, [2, 1]);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final a = NDArray.fromList([0, 5, 0, 8, 0], [5], dtype);
          final nz = nonzero(a);
          expect(nz[0].toList(), [1, 3]);
          expect(argwhere(a).shape, [2, 1]);
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final a = NDArray.fromList(
            [
              Complex(0.0, 0.0),
              Complex(1.0, 0.0),
              Complex(0.0, 2.0),
              Complex(0.0, 0.0),
            ],
            [4],
            dtype,
          );
          final nz = nonzero(a);
          expect(nz[0].toList(), [1, 2]);
          expect(argwhere(a).shape, [2, 1]);
        }

        final aBool = NDArray.fromList(
          [false, true, false, true],
          [2, 2],
          DType.boolean,
        );
        final nzBool = nonzero(aBool);
        expect(nzBool.length, 2);
        expect(nzBool[0].toList(), [0, 1]);
        expect(nzBool[1].toList(), [1, 1]);

        final awBool = argwhere(aBool);
        expect(awBool.shape, [2, 2]);
        expect(awBool.toList(), [0, 1, 1, 1]);

        final sNonzero = NDArray.scalar(5.0, dtype: DType.float64);
        expect(argwhere(sNonzero).shape, [1, 0]);

        final sZero = NDArray.scalar(0.0, dtype: DType.float64);
        expect(argwhere(sZero).shape, [0, 0]);
      }),
    );

    test(
      'Count Nonzero global and axis reductions across DTypes',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 0, 3, 0, 0, 6], [2, 3], DType.int32);

        expect(count_nonzero(a).scalar, 3);
        expect(count_nonzero(a, axis: 0).toList(), [1, 0, 2]);
        expect(count_nonzero(a, axis: 1).toList(), [2, 1]);
        expect(count_nonzero(a, axis: -1).toList(), [2, 1]);

        final out = NDArray.create([3], DType.int32);
        count_nonzero(a, axis: 0, out: out);
        expect(out.toList(), [1, 0, 2]);

        expect(() => count_nonzero(a, axis: 5), throwsA(isA<RangeError>()));
      }),
    );
  });

  group('Comprehensive Argmax & Argmin Tests', () {
    test(
      'Argmax and Argmin across numeric and boolean DTypes',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final a = NDArray.fromList(
            [10.0, 40.0, 20.0, 50.0, 30.0],
            [5],
            dtype,
          );
          expect(argmax(a).scalar, 3);
          expect(argmin(a).scalar, 0);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final a = NDArray.fromList([10, 40, 20, 50, 30], [5], dtype);
          expect(argmax(a).scalar, 3);
          expect(argmin(a).scalar, 0);
        }

        final aBool = NDArray.fromList(
          [false, true, false],
          [3],
          DType.boolean,
        );
        expect(argmax(aBool).scalar, 1);
        expect(argmin(aBool).scalar, 0);
      }),
    );

    test(
      'Argmax and Argmin multi-axis, keepdims, and error cases',
      () => NDArray.scope(() {
        final a2d = NDArray.fromList(
          [10.0, 50.0, 20.0, 40.0, 30.0, 60.0],
          [2, 3],
          DType.float64,
        );

        expect(argmax(a2d).scalar, 5);
        expect(argmin(a2d).scalar, 0);

        final amKeepdims = argmax(a2d, keepdims: true);
        expect(amKeepdims.shape, [1, 1]);
        expect(amKeepdims.getCellFlat(0), 5);

        final amAxis0 = argmax(a2d, axis: 0);
        expect(amAxis0.toList(), [1, 0, 1]);

        final amAxis1 = argmax(a2d, axis: 1);
        expect(amAxis1.toList(), [1, 2]);

        final amAxis1Keep = argmax(a2d, axis: 1, keepdims: true);
        expect(amAxis1Keep.shape, [2, 1]);
        expect(amAxis1Keep.toList(), [1, 2]);

        final cArr = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4)],
          [2],
          DType.complex128,
        );
        expect(() => argmax(cArr), throwsA(isA<UnsupportedError>()));
        expect(() => argmin(cArr), throwsA(isA<UnsupportedError>()));

        final empty = NDArray<Int32>.create([0], DType.int32);
        expect(() => argmax(empty), throwsA(isA<ArgumentError>()));
      }),
    );
  });

  group('Comprehensive FindIndex Tests', () {
    test(
      'FindIndex with all comparison operators and directions',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
          [2, 3],
          DType.float64,
        );

        expect(findIndex(a, CompareOp.equal, 30.0), [0, 2]);
        expect(findIndex(a, CompareOp.greater, 25.0), [0, 2]);
        expect(findIndex(a, CompareOp.less, 20.0), [0, 0]);
        expect(findIndex(a, CompareOp.greaterEqual, 50.0), [1, 1]);
        expect(findIndex(a, CompareOp.lessEqual, 10.0), [0, 0]);
        expect(findIndex(a, CompareOp.notEqual, 10.0), [0, 1]);

        expect(findIndex(a, CompareOp.greater, 25.0, directions: [-1, -1]), [
          1,
          2,
        ]);
        expect(findIndex(a, CompareOp.greater, 25.0, startCoords: [1, 0]), [
          1,
          0,
        ]);
        expect(findIndex(a, CompareOp.equal, 999.0), null);

        final cArr = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        expect(findIndex(cArr, CompareOp.equal, Complex(3.0, 4.0)), [1]);
        expect(findIndex(cArr, CompareOp.notEqual, Complex(1.0, 2.0)), [1]);
        expect(
          () => findIndex(cArr, CompareOp.less, Complex(3.0, 4.0)),
          throwsA(isA<UnsupportedError>()),
        );

        expect(
          () => findIndex(a, CompareOp.equal, 10.0, startCoords: [0]),
          throwsA(isA<ArgumentError>()),
        );
        expect(
          () => findIndex(a, CompareOp.equal, 10.0, directions: [1, 2]),
          throwsA(isA<ArgumentError>()),
        );
        expect(
          () => findIndex(a, CompareOp.equal, 10.0, startCoords: [5, 0]),
          throwsA(isA<RangeError>()),
        );
      }),
    );
  });

  group('Comprehensive Splitting Tests', () {
    test(
      'array_split and split with integer section counts across axes',
      () => NDArray.scope(() {
        final a1d = NDArray.fromList(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          [10],
          DType.int32,
        );

        final s2 = split(a1d, 2);
        expect(s2.length, 2);
        expect(s2[0].toList(), [1, 2, 3, 4, 5]);
        expect(s2[1].toList(), [6, 7, 8, 9, 10]);

        final as3 = array_split(a1d, 3);
        expect(as3.length, 3);
        expect(as3[0].toList(), [1, 2, 3, 4]);
        expect(as3[1].toList(), [5, 6, 7]);
        expect(as3[2].toList(), [8, 9, 10]);

        final as12 = array_split(a1d, 12);
        expect(as12.length, 12);
        expect(as12[0].toList(), [1]);
        expect(as12[9].toList(), [10]);
        expect(as12[10].toList(), <int>[]);
        expect(as12[11].toList(), <int>[]);

        final a2d = NDArray.fromList(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [3, 4],
          DType.int32,
        );

        final splitAx0 = array_split(a2d, 2, axis: 0);
        expect(splitAx0.length, 2);
        expect(splitAx0[0].shape, [2, 4]);
        expect(splitAx0[1].shape, [1, 4]);

        final splitAx1 = split(a2d, 2, axis: 1);
        expect(splitAx1.length, 2);
        expect(splitAx1[0].shape, [3, 2]);
        expect(splitAx1[1].shape, [3, 2]);

        final out1 = NDArray.create([3, 2], DType.int32);
        final out2 = NDArray.create([3, 2], DType.int32);
        split(a2d, 2, axis: 1, out: [out1, out2]);
        expect(out1.shape, [3, 2]);
        expect(out2.shape, [3, 2]);

        expect(() => split(a1d, 3), throwsA(isA<ArgumentError>()));
        expect(() => split(a1d, 0), throwsA(isA<ArgumentError>()));
        expect(() => split(a1d, 2, axis: 3), throwsA(isA<RangeError>()));
      }),
    );

    test(
      'array_split_at and split_at with coordinate indices',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [10, 20, 30, 40, 50, 60, 70],
          [7],
          DType.int32,
        );

        final parts = split_at(a, [2, 5]);
        expect(parts.length, 3);
        expect(parts[0].toList(), [10, 20]);
        expect(parts[1].toList(), [30, 40, 50]);
        expect(parts[2].toList(), [60, 70]);

        final clamped = array_split_at(a, [-1, 3, 100]);
        expect(clamped.length, 4);
        expect(clamped[0].toList(), <int>[]);
        expect(clamped[1].toList(), [10, 20, 30]);
        expect(clamped[2].toList(), [40, 50, 60, 70]);
        expect(clamped[3].toList(), <int>[]);
      }),
    );

    test(
      'hsplit, vsplit, and dsplit functions',
      () => NDArray.scope(() {
        final a1d = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final hs1d = hsplit(a1d, 2);
        expect(hs1d.length, 2);
        expect(hs1d[0].toList(), [1, 2]);
        expect(hs1d[1].toList(), [3, 4]);

        final a2d = NDArray.fromList(
          [1, 2, 3, 4, 5, 6, 7, 8],
          [2, 4],
          DType.int32,
        );

        final hs2d = hsplit(a2d, 2);
        expect(hs2d.length, 2);
        expect(hs2d[0].shape, [2, 2]);

        final vs2d = vsplit(a2d, 2);
        expect(vs2d.length, 2);
        expect(vs2d[0].shape, [1, 4]);

        final a3d = NDArray.fromList(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
          [2, 2, 4],
          DType.int32,
        );

        final ds3d = dsplit(a3d, 2);
        expect(ds3d.length, 2);
        expect(ds3d[0].shape, [2, 2, 2]);

        expect(hsplit_at(a2d, [1, 3]).length, 3);
        expect(vsplit_at(a2d, [1]).length, 2);
        expect(dsplit_at(a3d, [2]).length, 2);

        expect(() => vsplit(a1d, 2), throwsA(isA<ArgumentError>()));
        expect(() => dsplit(a2d, 2), throwsA(isA<ArgumentError>()));
      }),
    );
  });

  group('Comprehensive Padding Tests', () {
    test(
      'PadMode, PadWidth, PadValues, StatLength normalization and validation',
      () {
        expect(PadMode.minimum, PadMode.min);
        expect(PadMode.maximum, PadMode.max);

        final pwUniform = PadWidth.all(2, 3);
        expect(pwUniform.normalize(2), [(2, 3), (2, 3)]);

        final pwAxes = PadWidth.axes([(1, 2), (3, 4)]);
        expect(pwAxes.normalize(2), [(1, 2), (3, 4)]);
        expect(() => pwAxes.normalize(3), throwsA(isA<ArgumentError>()));
        expect(() => PadWidth.all(-1), throwsA(isA<RangeError>()));

        final pv = PadValues.all(0.0, 1.0);
        expect(pv.normalize(2, 0.0), [(0.0, 1.0), (0.0, 1.0)]);

        final sl = StatLength.all(2, 4);
        expect(sl.normalize([5, 10]), [(2, 4), (2, 4)]);
        expect(sl.normalize([1, 2]), [(1, 1), (2, 2)]);
        expect(() => StatLength.all(0), throwsA(isA<ArgumentError>()));
      },
    );

    test(
      'Fast native padding modes (constant, edge, reflect, symmetric, wrap) 1D, 2D, 3D',
      () => NDArray.scope(() {
        for (final dtype in [DType.float64, DType.float32]) {
          final a1d = NDArray.fromList([1.0, 2.0, 3.0], [3], dtype);
          expect(pad(a1d, PadWidth.all(1), mode: PadMode.edge).toList(), [
            1.0,
            1.0,
            2.0,
            3.0,
            3.0,
          ]);
          expect(pad(a1d, PadWidth.all(1), mode: PadMode.reflect).toList(), [
            2.0,
            1.0,
            2.0,
            3.0,
            2.0,
          ]);
          expect(pad(a1d, PadWidth.all(1), mode: PadMode.symmetric).toList(), [
            1.0,
            1.0,
            2.0,
            3.0,
            3.0,
          ]);
          expect(pad(a1d, PadWidth.all(1), mode: PadMode.wrap).toList(), [
            3.0,
            1.0,
            2.0,
            3.0,
            1.0,
          ]);

          final a2d = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype);
          final p2d = pad(a2d, PadWidth.all(1), mode: PadMode.edge);
          expect(p2d.shape, [4, 4]);

          final a3d = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            dtype,
          );
          final p3d = pad(a3d, PadWidth.all(1), mode: PadMode.constant);
          expect(p3d.shape, [4, 4, 4]);
        }

        for (final dtype in [
          DType.int32,
          DType.int64,
          DType.uint8,
          DType.int16,
        ]) {
          final a1d = NDArray.fromList([1, 2, 3], [3], dtype);
          expect(
            pad(
              a1d,
              PadWidth.all(1),
              mode: PadMode.edge,
            ).toList().map((e) => (e as num).toInt()).toList(),
            [1, 1, 2, 3, 3],
          );
          expect(
            pad(
              a1d,
              PadWidth.all(1),
              mode: PadMode.reflect,
            ).toList().map((e) => (e as num).toInt()).toList(),
            [2, 1, 2, 3, 2],
          );
          expect(
            pad(
              a1d,
              PadWidth.all(1),
              mode: PadMode.symmetric,
            ).toList().map((e) => (e as num).toInt()).toList(),
            [1, 1, 2, 3, 3],
          );
          expect(
            pad(
              a1d,
              PadWidth.all(1),
              mode: PadMode.wrap,
            ).toList().map((e) => (e as num).toInt()).toList(),
            [3, 1, 2, 3, 1],
          );
        }

        for (final dtype in [DType.complex128, DType.complex64]) {
          final a = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            dtype,
          );
          final pEdge = pad(a, PadWidth.all(1), mode: PadMode.edge);
          expect(pEdge.toList(), [
            Complex(1.0, 2.0),
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(3.0, 4.0),
          ]);
        }

        final aBool = NDArray.fromList([true, false], [2], DType.boolean);
        final pConst = pad(
          aBool,
          PadWidth.all(1),
          mode: PadMode.constant,
          constantValues: PadValues.all(false),
        );
        expect(pConst.toList(), [false, true, false, false]);
      }),
    );

    test(
      'Statistical and linearRamp padding modes across all numeric DTypes',
      () => NDArray.scope(() {
        for (final dtype in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.bfloat16,
        ]) {
          final a = NDArray.fromList([10.0, 20.0, 30.0, 40.0], [4], dtype);
          final pMax = pad(a, PadWidth.all(1), mode: PadMode.max);
          expect((pMax.getCellFlat(0) as num).toInt(), 40);
          expect((pMax.getCellFlat(5) as num).toInt(), 40);

          final pMin = pad(a, PadWidth.all(1), mode: PadMode.min);
          expect((pMin.getCellFlat(0) as num).toInt(), 10);
          expect((pMin.getCellFlat(5) as num).toInt(), 10);

          final pMean = pad(a, PadWidth.all(1), mode: PadMode.mean);
          expect(pMean.shape, [6]);

          final pMed = pad(a, PadWidth.all(1), mode: PadMode.median);
          expect(pMed.shape, [6]);

          final pRamp = pad(
            a,
            PadWidth.all(2),
            mode: PadMode.linearRamp,
            endValues: PadValues.all(0.0),
          );
          expect(pRamp.shape, [8]);
        }

        for (final dtype in [
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ]) {
          final a = NDArray.fromList([10, 20, 30, 40], [4], dtype);
          final pMax = pad(a, PadWidth.all(1), mode: PadMode.max);
          expect((pMax.getCellFlat(0) as num).toInt(), 40);
          expect((pMax.getCellFlat(5) as num).toInt(), 40);

          final pMin = pad(a, PadWidth.all(1), mode: PadMode.min);
          expect((pMin.getCellFlat(0) as num).toInt(), 10);
          expect((pMin.getCellFlat(5) as num).toInt(), 10);

          final pMean = pad(a, PadWidth.all(1), mode: PadMode.mean);
          expect(pMean.shape, [6]);

          final pMed = pad(a, PadWidth.all(1), mode: PadMode.median);
          expect(pMed.shape, [6]);

          final pRamp = pad(
            a,
            PadWidth.all(2),
            mode: PadMode.linearRamp,
            endValues: PadValues.all(0),
          );
          expect(pRamp.shape, [8]);
        }
      }),
    );

    test(
      'Pad with zero padding, preallocated out buffer, and error cases',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

        final zeroPad = pad(a, PadWidth.all(0));
        expect(zeroPad.toList(), [1.0, 2.0, 3.0]);

        final out = NDArray<Float64>.create([5], DType.float64);
        pad(a, PadWidth.all(1), mode: PadMode.edge, out: out);
        expect(out.toList(), [1.0, 1.0, 2.0, 3.0, 3.0]);

        expect(
          () => pad(
            a,
            PadWidth.all(1),
            out: NDArray<Float64>.create([4], DType.float64),
          ),
          throwsA(isA<ArgumentError>()),
        );
        final scalar = NDArray.scalar(42.0, dtype: DType.float64);
        expect(
          () => pad(scalar, PadWidth.all(1)),
          throwsA(isA<ArgumentError>()),
        );
      }),
    );
  });
}
