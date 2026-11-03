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

import 'dart:ffi' as ffi;
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

          expect(a1.equals(a2), isTrue);
          expect(a1.contentHashCode, equals(a2.contentHashCode));
          expect(a1.equals(a3), isFalse);

          final u1 = NDArray.fromList(Uint8List.fromList([10, 20, 30]), [
            3,
          ], DType.uint8);
          final u2 = NDArray.fromList(Uint8List.fromList([10, 20, 30]), [
            3,
          ], DType.uint8);
          final u3 = NDArray.fromList(Uint8List.fromList([10, 20, 31]), [
            3,
          ], DType.uint8);

          expect(u1.equals(u2), isTrue);
          expect(u1.contentHashCode, equals(u2.contentHashCode));
          expect(u1.equals(u3), isFalse);
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

          // Test contentHashCode for rank 34
          final a2 = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 3.0, 0.0]),
            shape,
            DType.float64,
          );
          expect(a.contentHashCode, equals(a2.contentHashCode));

          // Test flatten for rank 34
          final flat = a.flatten();
          expect(flat.shape, [4]);
          expect(flat.toList(), [1.0, 0.0, 3.0, 0.0]);
        }),
      );
    });

    group('Track 2 Native Kernel Edge Cases & Regressions', () {
      test(
        'partition on float64/float32 preserves -0.0 and +0.0 bit multiset on two-value arrays',
        () => NDArray.scope(() {
          for (final n in [16, 17, 32, 33, 63, 64, 65, 128]) {
            final f64 = NDArray.zeros([n], DType.float64);
            final ptr64 = f64.pointer.cast<ffi.Uint64>();
            var expectedNegZeros = 0;
            for (var i = 0; i < n; i++) {
              if (i.isEven) {
                ptr64[i] = 0x8000000000000000; // -0.0
                expectedNegZeros++;
              } else {
                ptr64[i] = 0; // +0.0
              }
            }
            final p64 = partition(f64, [0, n ~/ 2, n - 1]);
            final outPtr64 = p64.pointer.cast<ffi.Uint64>();
            var actualNegZeros64 = 0;
            for (var i = 0; i < n; i++) {
              if (outPtr64[i] == 0x8000000000000000) {
                actualNegZeros64++;
              } else {
                expect(outPtr64[i], 0);
              }
            }
            expect(actualNegZeros64, expectedNegZeros);

            final f32 = NDArray.zeros([n], DType.float32);
            final ptr32 = f32.pointer.cast<ffi.Uint32>();
            for (var i = 0; i < n; i++) {
              ptr32[i] = i.isEven ? 0x80000000 : 0;
            }
            final p32 = partition(f32, [0, n ~/ 2, n - 1]);
            final outPtr32 = p32.pointer.cast<ffi.Uint32>();
            var actualNegZeros32 = 0;
            for (var i = 0; i < n; i++) {
              if (outPtr32[i] == 0x80000000) {
                actualNegZeros32++;
              } else {
                expect(outPtr32[i], 0);
              }
            }
            expect(actualNegZeros32, expectedNegZeros);
          }
        }),
      );

      test(
        'stable argsort preserves relative order of -0.0 and +0.0 ties across threshold sizes',
        () => NDArray.scope(() {
          for (final n in [16, 32, 33, 64, 65]) {
            final f64 = NDArray.zeros([n], DType.float64);
            final ptr64 = f64.pointer.cast<ffi.Uint64>();
            for (var i = 0; i < n; i++) {
              ptr64[i] = i.isOdd ? 0x8000000000000000 : 0;
            }
            final idx64 = argsort(f64, kind: SortKind.stable);
            expect(idx64.toList(), List<int>.generate(n, (i) => i));

            final f32 = NDArray.zeros([n], DType.float32);
            final ptr32 = f32.pointer.cast<ffi.Uint32>();
            for (var i = 0; i < n; i++) {
              ptr32[i] = i.isOdd ? 0x80000000 : 0;
            }
            final idx32 = argsort(f32, kind: SortKind.stable);
            expect(idx32.toList(), List<int>.generate(n, (i) => i));
          }
        }),
      );

      test(
        'partition, argpartition, and argsort across threshold sizes with NaNs, unsorted/duplicate kth, uint64 high-bit, and int8 min',
        () => NDArray.scope(() {
          for (final n in [16, 17, 32, 33, 63, 64, 65]) {
            // Float64 with NaN at various positions and unsorted duplicate kth
            for (final nanPos in [0, 1, n ~/ 2, n - 1]) {
              final data = List<double>.generate(n, (i) => (n - i).toDouble());
              data[nanPos] = double.nan;
              final expectedMin = nanPos == n - 1 ? 2.0 : 1.0;
              final a = NDArray.fromList(data, [n], DType.float64);
              final kth = [n - 1, 0, n ~/ 2, 0];
              final p = partition(a, kth);
              expect(p.getCellFlat(n - 1).isNaN, isTrue);
              expect(p.getCellFlat(0), expectedMin);

              final ap = argpartition(a, kth);
              expect(ap.getCellFlat(n - 1), nanPos);
              expect(a.getCellFlat(ap.getCellFlat(0)), expectedMin);

              final asIdx = argsort(a, kind: SortKind.stable);
              expect(asIdx.getCellFlat(n - 1), nanPos);
              expect(a.getCellFlat(asIdx.getCellFlat(0)), expectedMin);
            }

            // Uint64 with high bit set (>= 2^63)
            final u64 = NDArray.zeros([n], DType.uint64);
            final uPtr = u64.pointer.cast<ffi.Uint64>();
            for (var i = 0; i < n; i++) {
              // Alternate high-bit values and small values
              uPtr[i] = i.isEven ? (0x8000000000000000 + (n - i)) : i;
            }
            final uSortIdx = argsort(u64);
            for (var i = 0; i < n - 1; i++) {
              final aIdx = uSortIdx.getCellFlat(i);
              final bIdx = uSortIdx.getCellFlat(i + 1);
              final va = BigInt.from(uPtr[aIdx]).toUnsigned(64);
              final vb = BigInt.from(uPtr[bIdx]).toUnsigned(64);
              expect(va <= vb, isTrue);
            }
            final uPartIdx = argpartition(u64, n ~/ 2);
            final pivotU = BigInt.from(
              uPtr[uPartIdx.getCellFlat(n ~/ 2)],
            ).toUnsigned(64);
            for (var i = 0; i < n ~/ 2; i++) {
              final v = BigInt.from(
                uPtr[uPartIdx.getCellFlat(i)],
              ).toUnsigned(64);
              expect(v <= pivotU, isTrue);
            }

            // Int8 including -128 and 127
            final i8Data = Int8List.fromList(
              List<int>.generate(
                n,
                (i) => i == 0 ? 127 : (i == n - 1 ? -128 : (i % 50) - 25),
              ),
            );
            final i8 = NDArray.fromList(i8Data, [n], DType.int8);
            final i8Idx = argsort(i8, kind: SortKind.stable);
            expect(i8.getCellFlat(i8Idx.getCellFlat(0)), -128);
            expect(i8.getCellFlat(i8Idx.getCellFlat(n - 1)), 127);
          }
        }),
      );

      test(
        'min and max preserve custom NaN payload across SIMD threshold sizes',
        () => NDArray.scope(() {
          for (final n in [16, 17, 32, 33, 64, 65]) {
            final f64 = NDArray.ones([n], DType.float64);
            final ptr64 = f64.pointer.cast<ffi.Uint64>();
            const customNan64 = 0xfff8000000000042;
            ptr64[n ~/ 2] = customNan64;

            final minRes64 = min(f64);
            final maxRes64 = max(f64);
            expect(minRes64.pointer.cast<ffi.Uint64>()[0], customNan64);
            expect(maxRes64.pointer.cast<ffi.Uint64>()[0], customNan64);

            final f32 = NDArray.ones([n], DType.float32);
            final ptr32 = f32.pointer.cast<ffi.Uint32>();
            const customNan32 = 0xffc0002a;
            ptr32[n ~/ 2] = customNan32;

            final minRes32 = min(f32);
            final maxRes32 = max(f32);
            expect(minRes32.pointer.cast<ffi.Uint32>()[0], customNan32);
            expect(maxRes32.pointer.cast<ffi.Uint32>()[0], customNan32);
          }
        }),
      );

      test(
        'argmin and argmax across SIMD threshold sizes and NaN positions',
        () => NDArray.scope(() {
          for (final n in [16, 17, 32, 33, 63, 64, 65, 129]) {
            for (final nanPos in [0, 1, n ~/ 2, n - 1]) {
              final f64 = NDArray.fromList(
                List<double>.generate(n, (i) => i.toDouble()),
                [n],
                DType.float64,
              );
              f64.setCellFlat(nanPos, double.nan);
              expect(argmin(f64).scalar, nanPos);
              expect(argmax(f64).scalar, nanPos);
            }
          }
        }),
      );

      test(
        'quantile on array with infinity and discrete/continuous methods',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, double.infinity],
            [2],
            DType.float64,
          );
          for (final method in [
            QuantileMethod.lower,
            QuantileMethod.higher,
            QuantileMethod.nearest,
            QuantileMethod.invertedCdf,
            QuantileMethod.closestObservation,
          ]) {
            expect(quantile(a, 0.0, method: method).scalar, 1.0);
            expect(quantile(a, 1.0, method: method).scalar, double.infinity);
          }
        }),
      );

      test(
        'repeat scalar and list across all 15 dtypes, non-contiguous axis: null, and tile itemsizes',
        () => NDArray.scope(() {
          for (final dtype in DType.values) {
            final a = NDArray.zeros([2, 3], dtype);
            final rUniform = repeat(a, 3, axis: 1);
            expect(rUniform.shape, [2, 9]);
            expect(rUniform.dtype, dtype);

            final rList = repeat(a, [1, 2, 0], axis: 1);
            expect(rList.shape, [2, 3]);
            expect(rList.dtype, dtype);

            final t = tile(a, [2, 3]);
            expect(t.shape, [4, 9]);
            expect(t.dtype, dtype);
          }

          // Non-contiguous input with axis: null
          final base = NDArray.arange(
            0,
            12,
            dtype: DType.int32,
          ).reshape([3, 4]);
          final nonContig = base.transpose();
          expect(nonContig.isContiguous, isFalse);
          final repFlat = repeat(nonContig, 2);
          expect(repFlat.shape, [24]);
          final expected = <int>[];
          for (final v in nonContig.ravel().toList()) {
            expected.add(v);
            expected.add(v);
          }
          expect(repFlat.toList(), expected);

          // Non-contiguous partition with out == null and out != null
          final pNonContig = partition(nonContig, 1);
          expect(pNonContig.shape, [4, 3]);
          final outContig = NDArray.zeros([4, 3], DType.int32);
          partition(nonContig, 1, out: outContig);
          expect(outContig.toList(), pNonContig.toList());
        }),
      );
    });
  });
}
