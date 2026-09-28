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

void main() {
  group('Native SIMD & Sorting Tests', () {
    group('Sort - int16 and all numeric dtypes', () {
      test(
        'Sort contiguous Int16 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Int16List.fromList([300, -100, 25000, -32000, 0, 42]),
            [6],
            DType.int16,
          );
          final s = sort(a);
          expect(s.dtype, DType.int16);
          expect(s.toList(), [-32000, -100, 0, 42, 300, 25000]);
        }),
      );

      test(
        'Sort Int16 multi-dimensional array along axis',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int16List.fromList([5, 2, 8, 1, 9, 3]), [
            2,
            3,
          ], DType.int16);
          final s = sort(a, axis: 1);
          expect(s.shape, [2, 3]);
          expect(s.toList(), [2, 5, 8, 1, 3, 9]);
        }),
      );

      test(
        'Sort with different algorithms (quicksort, stable, heapsort)',
        () => NDArray.scope(() {
          for (final kind in [
            SortKind.quicksort,
            SortKind.stable,
            SortKind.heapsort,
          ]) {
            final a16 = NDArray.fromList(Int16List.fromList([10, -5, 3, 0]), [
              4,
            ], DType.int16);
            expect(sort(a16, kind: kind).toList(), [-5, 0, 3, 10]);

            final a32 = NDArray.fromList(Int32List.fromList([10, -5, 3, 0]), [
              4,
            ], DType.int32);
            expect(sort(a32, kind: kind).toList(), [-5, 0, 3, 10]);

            final a64 = NDArray.fromList(Int64List.fromList([10, -5, 3, 0]), [
              4,
            ], DType.int64);
            expect(sort(a64, kind: kind).toList(), [-5, 0, 3, 10]);

            final af32 = NDArray.fromList(
              Float32List.fromList([10.0, -5.0, 3.0, 0.0]),
              [4],
              DType.float32,
            );
            expect(sort(af32, kind: kind).toList(), [-5.0, 0.0, 3.0, 10.0]);

            final af64 = NDArray.fromList(
              Float64List.fromList([10.0, -5.0, 3.0, 0.0]),
              [4],
              DType.float64,
            );
            expect(sort(af64, kind: kind).toList(), [-5.0, 0.0, 3.0, 10.0]);

            final au8 = NDArray.fromList(Uint8List.fromList([10, 255, 3, 0]), [
              4,
            ], DType.uint8);
            expect(sort(au8, kind: kind).toList(), [0, 3, 10, 255]);
          }
        }),
      );
    });

    group('Argsort size == 1 uninitialized bug fix', () {
      test(
        'Argsort single-element array returns [0] for all dtypes',
        () => NDArray.scope(() {
          for (final kind in [
            SortKind.quicksort,
            SortKind.stable,
            SortKind.heapsort,
          ]) {
            final f64 = NDArray.fromList(Float64List.fromList([42.0]), [
              1,
            ], DType.float64);
            expect(argsort(f64, kind: kind).toList(), [0]);

            final f32 = NDArray.fromList(Float32List.fromList([42.0]), [
              1,
            ], DType.float32);
            expect(argsort(f32, kind: kind).toList(), [0]);

            final i64 = NDArray.fromList(Int64List.fromList([42]), [
              1,
            ], DType.int64);
            expect(argsort(i64, kind: kind).toList(), [0]);

            final i32 = NDArray.fromList(Int32List.fromList([42]), [
              1,
            ], DType.int32);
            expect(argsort(i32, kind: kind).toList(), [0]);

            final i16 = NDArray.fromList(Int16List.fromList([42]), [
              1,
            ], DType.int16);
            expect(argsort(i16, kind: kind).toList(), [0]);

            final u8 = NDArray.fromList(Uint8List.fromList([42]), [
              1,
            ], DType.uint8);
            expect(argsort(u8, kind: kind).toList(), [0]);
          }
        }),
      );

      test(
        'Argsort multi-row with single element rows',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int16List.fromList([10, 20, 30]), [
            3,
            1,
          ], DType.int16);
          final idx = argsort(a, axis: 1);
          expect(idx.shape, [3, 1]);
          expect(idx.toList(), [0, 0, 0]);
        }),
      );
    });

    group('Argsort - int16 and uint8', () {
      test(
        'Argsort Int16 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int16List.fromList([50, -20, 100, 0]), [
            4,
          ], DType.int16);
          final idx = argsort(a);
          expect(idx.toList(), [1, 3, 0, 2]);
        }),
      );

      test(
        'Argsort Uint8 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Uint8List.fromList([50, 200, 10, 0]), [
            4,
          ], DType.uint8);
          final idx = argsort(a);
          expect(idx.toList(), [3, 2, 0, 1]);
        }),
      );
    });

    group('Partition & Argpartition - int16 and uint8', () {
      test(
        'Partition Int16 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Int16List.fromList([9, 1, 8, 2, 7, 3, 6, 4, 5]),
            [9],
            DType.int16,
          );
          final p = partition(a, 4);
          expect(p.dtype, DType.int16);
          expect(p.toList()[4], 5);
          for (var i = 0; i < 4; i++) {
            expect(p.toList()[i] <= 5, isTrue);
          }
          for (var i = 5; i < 9; i++) {
            expect(p.toList()[i] >= 5, isTrue);
          }
        }),
      );

      test(
        'Partition Uint8 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Uint8List.fromList([90, 10, 80, 20, 70, 30, 60, 40, 50]),
            [9],
            DType.uint8,
          );
          final p = partition(a, 4);
          expect(p.dtype, DType.uint8);
          expect(p.toList()[4], 50);
          for (var i = 0; i < 4; i++) {
            expect(p.toList()[i] <= 50, isTrue);
          }
          for (var i = 5; i < 9; i++) {
            expect(p.toList()[i] >= 50, isTrue);
          }
        }),
      );

      test(
        'Argpartition Int16 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Int16List.fromList([9, 1, 8, 2, 7, 3, 6, 4, 5]),
            [9],
            DType.int16,
          );
          final idx = argpartition(a, 4);
          final pVal = a.toList()[idx.toList()[4]];
          expect(pVal, 5);
        }),
      );

      test(
        'Argpartition Uint8 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Uint8List.fromList([90, 10, 80, 20, 70, 30, 60, 40, 50]),
            [9],
            DType.uint8,
          );
          final idx = argpartition(a, 4);
          final pVal = a.toList()[idx.toList()[4]];
          expect(pVal, 50);
        }),
      );
    });

    group('Searchsorted - int16', () {
      test(
        'Searchsorted Int16 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int16List.fromList([10, 20, 30, 40, 50]), [
            5,
          ], DType.int16);
          final v = NDArray.fromList(Int16List.fromList([5, 20, 25, 50, 60]), [
            5,
          ], DType.int16);
          final resLeft = searchsorted(a, v, side: SearchSide.left);
          expect(resLeft.toList(), [0, 1, 2, 4, 5]);

          final resRight = searchsorted(a, v, side: SearchSide.right);
          expect(resRight.toList(), [0, 2, 2, 5, 5]);
        }),
      );
    });

    group('HashCode & Equality - int16 and uint8', () {
      test(
        'HashCode and equality for Int16 and Uint8',
        () => NDArray.scope(() {
          final a1 = NDArray.fromList(Int16List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int16);
          final a2 = NDArray.fromList(Int16List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int16);
          final a3 = NDArray.fromList(Int16List.fromList([1, 2, 3, 5]), [
            2,
            2,
          ], DType.int16);

          expect(a1 == a2, isTrue);
          expect(a1.hashCode, equals(a2.hashCode));
          expect(a1 == a3, isFalse);

          final u1 = NDArray.fromList(Uint8List.fromList([10, 20, 30]), [
            3,
          ], DType.uint8);
          final u2 = NDArray.fromList(Uint8List.fromList([10, 20, 30]), [
            3,
          ], DType.uint8);
          final u3 = NDArray.fromList(Uint8List.fromList([10, 20, 31]), [
            3,
          ], DType.uint8);

          expect(u1 == u2, isTrue);
          expect(u1.hashCode, equals(u2.hashCode));
          expect(u1 == u3, isFalse);
        }),
      );
    });

    group('Padding - int16', () {
      test(
        'Pad Int16 with constant mode',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int16List.fromList([1, 2, 3]), [
            3,
          ], DType.int16);
          final p = pad(
            a,
            PadWidth.axes([(1, 2)]),
            constantValues: PadValues.all(0),
          );
          expect(p.shape, [6]);
          expect(p.dtype, DType.int16);
          expect(p.toList(), [0, 1, 2, 3, 0, 0]);
        }),
      );

      test(
        'Pad Int16 with edge, reflect, symmetric, wrap',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int16List.fromList([10, 20, 30]), [
            3,
          ], DType.int16);
          final edge = pad(a, PadWidth.all(1), mode: PaddingMode.edge);
          expect(edge.toList(), [10, 10, 20, 30, 30]);

          final reflect = pad(a, PadWidth.all(1), mode: PaddingMode.reflect);
          expect(reflect.toList(), [20, 10, 20, 30, 20]);

          final symmetric = pad(
            a,
            PadWidth.all(1),
            mode: PaddingMode.symmetric,
          );
          expect(symmetric.toList(), [10, 10, 20, 30, 30]);

          final wrap = pad(a, PadWidth.all(1), mode: PaddingMode.wrap);
          expect(wrap.toList(), [30, 10, 20, 30, 10]);
        }),
      );

      test(
        'Pad Int16 with statistics (min, max, mean, median)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int16List.fromList([10, 20, 30]), [
            3,
          ], DType.int16);
          final minPad = pad(a, PadWidth.all(1), mode: PaddingMode.minimum);
          expect(minPad.toList(), [10, 10, 20, 30, 10]);

          final maxPad = pad(a, PadWidth.all(1), mode: PaddingMode.maximum);
          expect(maxPad.toList(), [30, 10, 20, 30, 30]);

          final meanPad = pad(a, PadWidth.all(1), mode: PaddingMode.mean);
          expect(meanPad.toList(), [20, 10, 20, 30, 20]);

          final medianPad = pad(a, PadWidth.all(1), mode: PaddingMode.median);
          expect(medianPad.toList(), [20, 10, 20, 30, 20]);
        }),
      );
    });

    group('High-Rank Arrays (Rank > 32)', () {
      test(
        'High-rank (rank 34) array operations without stack buffer overflow',
        () => NDArray.scope(() {
          final shape = List<int>.filled(34, 1);
          shape[0] = 2;
          shape[33] = 2; // total elements = 4

          final a = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 3.0, 0.0]),
            shape,
            DType.float64,
          );

          expect(a.rank, 34);
          expect(count_nonzero(a).scalar, 2);

          final nonzeroCoords = nonzero(a);
          expect(nonzeroCoords.length, 34);
          expect(nonzeroCoords[0].toList(), [0, 1]);

          // Test hashCode for rank 34
          final a2 = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 3.0, 0.0]),
            shape,
            DType.float64,
          );
          expect(a.hashCode, equals(a2.hashCode));

          // Test flatten for rank 34
          final flat = a.flatten();
          expect(flat.shape, [4]);
          expect(flat.toList(), [1.0, 0.0, 3.0, 0.0]);
        }),
      );
    });
  });
}
