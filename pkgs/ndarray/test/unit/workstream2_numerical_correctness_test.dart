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

import 'dart:io';
import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Workstream 2: Numerical Correctness, Reductions, & DTypes', () {
    group('H3: Scalar DType Promotion', () {
      test('float32 + 1.0 maintains Float32 dtype', () {
        NDArray.scope(() {
          final a = NDArray<Float32>.fromList([1.0, 2.0], [2], DType.float32);
          final res = a + 1.0;
          expect(res.dtype, equals(DType.float32));
          expect((res.getCellFlat(0) as num).toDouble(), closeTo(2.0, 1e-5));
          expect((res.getCellFlat(1) as num).toDouble(), closeTo(3.0, 1e-5));
        });
      });

      test('int8 + 1 maintains Int8 dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([10, 20], [2], DType.int8);
          final res = a + 1;
          expect(res.dtype, equals(DType.int8));
          expect(res.getCellFlat(0), equals(11));
          expect(res.getCellFlat(1), equals(21));
        });
      });

      test('float16 + 2.0 maintains Float16 dtype', () {
        NDArray.scope(() {
          final a = NDArray<Float16>.fromList([1.0, 2.0], [2], DType.float16);
          final res = a + 2.0;
          expect(res.dtype, equals(DType.float16));
          expect((res.getCellFlat(0) as num).toDouble(), closeTo(3.0, 1e-3));
        });
      });

      test('int16 * 3 maintains Int16 dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([10, 20], [2], DType.int16);
          final res = a * 3;
          expect(res.dtype, equals(DType.int16));
          expect(res.getCellFlat(0), equals(30));
        });
      });

      test('complex64 * Complex(2, 0) maintains Complex64 dtype', () {
        NDArray.scope(() {
          final a = NDArray<Complex64>.fromList(
            [Complex(1.0, 2.0)],
            [1],
            DType.complex64,
          );
          final res = a * Complex(2.0, 0.0);
          expect(res.dtype, equals(DType.complex64));
          final c = res.getCellFlat(0);
          expect(c.real, closeTo(2.0, 1e-5));
          expect(c.imag, closeTo(4.0, 1e-5));
        });
      });
    });

    group('H4: NaN Propagation in Reductions', () {
      test('min and max propagate NaN on 1D float64 array', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, double.nan, 2.0],
            [3],
            DType.float64,
          );
          expect((min(a).scalar as num).toDouble().isNaN, isTrue);
          expect((max(a).scalar as num).toDouble().isNaN, isTrue);
          // But nanmin and nanmax ignore NaN
          expect((nanmin(a).scalar as num).toDouble(), equals(1.0));
          expect((nanmax(a).scalar as num).toDouble(), equals(2.0));
        });
      });

      test('min and max propagate NaN on 2D float32 array along axis', () {
        NDArray.scope(() {
          final a = NDArray<Float32>.fromList(
            [1.0, double.nan, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          // Col 0: [1.0, 3.0] -> min=1.0, max=3.0
          // Col 1: [nan, 4.0] -> min=nan, max=nan
          final mn = min(a, axis: 0);
          final mx = max(a, axis: 0);
          expect((mn.getCellFlat(0) as num).toDouble(), equals(1.0));
          expect((mn.getCellFlat(1) as num).toDouble().isNaN, isTrue);
          expect((mx.getCellFlat(0) as num).toDouble(), equals(3.0));
          expect((mx.getCellFlat(1) as num).toDouble().isNaN, isTrue);
        });
      });

      test('argmax and argmin return index of first NaN', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [10.0, 20.0, double.nan, 5.0],
            [4],
            DType.float64,
          );
          final maxIdx = argmax(a);
          final minIdx = argmin(a);
          expect(maxIdx.scalar, equals(2));
          expect(minIdx.scalar, equals(2));
        });
      });

      test('argmax and argmin return first NaN along 2D axis', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, double.nan, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          // Along axis 0 (rows):
          // Col 0: [1.0, nan, 5.0] -> first NaN is row 1
          // Col 1: [2.0, 4.0, 6.0] -> min at row 0 (2.0), max at row 2 (6.0)
          final minCol = argmin(a, axis: 0);
          final maxCol = argmax(a, axis: 0);
          expect(minCol.getCellFlat(0), equals(1));
          expect(minCol.getCellFlat(1), equals(0));
          expect(maxCol.getCellFlat(0), equals(1));
          expect(maxCol.getCellFlat(1), equals(2));
        });
      });

      test('median propagates NaN', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, double.nan, 3.0],
            [3],
            DType.float64,
          );
          final med = median(a);
          expect((med.scalar as num).toDouble().isNaN, isTrue);
        });
      });
    });

    group('H5: sum and prod Accumulator DType & Bool Counting', () {
      test(
        'sumAs of boolean array with DType.int64 returns int64 count of true values',
        () {
          NDArray.scope(() {
            final a = NDArray<Boolean>.fromList(
              [true, false, true, true],
              [4],
              DType.boolean,
            );
            final s = sumAs(a, DType.int64);
            expect(s.dtype, equals(DType.int64));
            expect(s.scalar, equals(3));
          });
        },
      );

      test(
        'prodAs of boolean array with DType.int64 returns int64 product',
        () {
          NDArray.scope(() {
            final a1 = NDArray<Boolean>.fromList(
              [true, true, true],
              [3],
              DType.boolean,
            );
            final p1 = prodAs(a1, DType.int64);
            expect(p1.dtype, equals(DType.int64));
            expect(p1.scalar, equals(1));

            final a2 = NDArray<Boolean>.fromList(
              [true, false, true],
              [3],
              DType.boolean,
            );
            final p2 = prodAs(a2, DType.int64);
            expect(p2.dtype, equals(DType.int64));
            expect(p2.scalar, equals(0));
          });
        },
      );

      test('sumAs along axis for boolean array', () {
        NDArray.scope(() {
          final a = NDArray<Boolean>.fromList(
            [true, false, true, true],
            [2, 2],
            DType.boolean,
          );
          final s0 = sumAs(a, DType.int64, axis: 0);
          expect(s0.dtype, equals(DType.int64));
          expect(s0.toList(), equals([2, 1]));

          final s1 = sumAs(a, DType.int64, axis: 1);
          expect(s1.dtype, equals(DType.int64));
          expect(s1.toList(), equals([1, 2]));
        });
      });

      test('sum with explicit dtype widening prevents overflow', () {
        NDArray.scope(() {
          // 100 + 100 = 200 (overflows int8 if not widened)
          final a = NDArray.fromList([100, 100], [2], DType.int8);
          final s = sumAs(a, DType.int64);
          expect(s.dtype, equals(DType.int64));
          expect(s.scalar, equals(200));
        });
      });

      test(
        'sum with explicit boolean dtype preserves boolean OR reduction',
        () {
          NDArray.scope(() {
            final a = NDArray<Boolean>.fromList(
              [true, false, true],
              [3],
              DType.boolean,
            );
            final s = sumAs(a, DType.boolean);
            expect(s.dtype, equals(DType.boolean));
            expect(s.scalar, equals(true));
          });
        },
      );
    });

    group('H9: uint64 Correct Handling & >= 2^63 Validation', () {
      test('uint64Compare correctly orders values around 2^63', () {
        const uMax = -1; // 0xFFFFFFFFFFFFFFFF in 64-bit two's complement
        const uHigh = -0x8000000000000000; // 0x8000000000000000 (2^63)
        const uMid = 0x7FFFFFFFFFFFFFFF; // 2^63 - 1
        const uLow = 100;

        expect(uint64Compare(uLow, uMid) < 0, isTrue);
        expect(uint64Compare(uMid, uHigh) < 0, isTrue);
        expect(uint64Compare(uHigh, uMax) < 0, isTrue);
        expect(uint64Compare(uMax, uLow) > 0, isTrue);
        expect(uint64Compare(uHigh, uHigh), equals(0));
      });

      test('uint64 min and max on values >= 2^63', () {
        NDArray.scope(() {
          const uMax = -1;
          const uHigh = -0x8000000000000000;
          const uLow = 100;

          final a = NDArray.fromList([uMax, uLow, uHigh], [3], DType.uint64);
          final mn = min(a);
          final mx = max(a);
          expect(mn.scalar, equals(uLow));
          expect(mx.scalar, equals(uMax));
        });
      });

      test('uint64 sort and argsort', () {
        NDArray.scope(() {
          const uMax = -1;
          const uHigh = -0x8000000000000000;
          const uLow = 100;

          final a = NDArray.fromList([uMax, uLow, uHigh], [3], DType.uint64);
          final sorted = sort(a);
          expect(sorted.toList(), equals([uLow, uHigh, uMax]));

          final order = argsort(a);
          expect(order.toList(), equals([1, 2, 0]));
        });
      });

      test('uint64 searchsorted with values >= 2^63', () {
        NDArray.scope(() {
          const uLow = 100;
          const uHigh = -0x8000000000000000;
          const uMax = -1;

          final sortedA = NDArray.fromList(
            [uLow, uHigh, uMax],
            [3],
            DType.uint64,
          );
          final query = NDArray.fromList([50, uHigh, -2], [3], DType.uint64);

          final idxs = searchsorted(sortedA, query);
          // 50 < 100 -> index 0
          // uHigh == uHigh -> index 1
          // -2 (0xFFFFFFFFFFFFFFFE) < -1 (0xFFFFFFFFFFFFFFFF) -> index 2
          expect(idxs.toList(), equals([0, 1, 2]));
        });
      });

      test('uint64 median with odd and even sizes', () {
        NDArray.scope(() {
          const uLow = 100;
          const uHigh = -0x8000000000000000;
          const uMax = -1;

          final odd = NDArray.fromList([uMax, uLow, uHigh], [3], DType.uint64);
          expect(median(odd).dtype, equals(DType.float64));
          expect(
            median(odd).scalar,
            equals(BigInt.from(uHigh).toUnsigned(64).toDouble()),
          );

          final even = NDArray.fromList([10, 20], [2], DType.uint64);
          expect(median(even).dtype, equals(DType.float64));
          expect(median(even).scalar, equals(15.0));
        });
      });
    });

    group('H10: searchsorted(sorter:) Validation', () {
      test('valid sorter works correctly', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [30.0, 10.0, 20.0],
            [3],
            DType.float64,
          );
          final sorter = NDArray.fromList([1, 2, 0], [3], DType.int32);
          final v = NDArray<Float64>.fromList([15.0], [1], DType.float64);

          final idx = searchsorted(a, v, sorter: sorter);
          expect(idx.getCellFlat(0), equals(1));
        });
      });

      test('sorter with out-of-bounds indices throws ArgumentError', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [30.0, 10.0, 20.0],
            [3],
            DType.float64,
          );
          final v = NDArray<Float64>.fromList([15.0], [1], DType.float64);

          final oobSorter1 = NDArray.fromList([1, 2, 5], [3], DType.int32);
          expect(
            () => searchsorted(a, v, sorter: oobSorter1),
            throwsArgumentError,
          );

          final oobSorter2 = NDArray.fromList([-1, 1, 2], [3], DType.int32);
          expect(
            () => searchsorted(a, v, sorter: oobSorter2),
            throwsArgumentError,
          );
        });
      });

      test('sorter with incompatible shape throws ArgumentError', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [30.0, 10.0, 20.0],
            [3],
            DType.float64,
          );
          final v = NDArray<Float64>.fromList([15.0], [1], DType.float64);

          final badShapeSorter = NDArray.fromList([1, 2], [2], DType.int32);
          expect(
            () => searchsorted(a, v, sorter: badShapeSorter),
            throwsArgumentError,
          );
        });
      });

      test('sorter with non-integer dtype throws ArgumentError', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [30.0, 10.0, 20.0],
            [3],
            DType.float64,
          );
          final v = NDArray<Float64>.fromList([15.0], [1], DType.float64);

          final floatSorter = NDArray<Float64>.fromList(
            [1.0, 2.0, 0.0],
            [3],
            DType.float64,
          );
          expect(
            () => searchsorted(a, v, sorter: floatSorter as dynamic),
            throwsA(anyOf(isA<ArgumentError>(), isA<TypeError>())),
          );
        });
      });
    });

    group('M1: Checked Shape Product Overflow', () {
      test('negative dimension throws ArgumentError', () {
        expect(
          () => NDArray<Float64>.create([-1], DType.float64),
          throwsArgumentError,
        );
        expect(
          () => NDArray<Float64>.create([2, -3], DType.float64),
          throwsArgumentError,
        );
      });

      test(
        'overflowing 64-bit shape product or byte size throws ArgumentError',
        () {
          expect(
            () => NDArray<Float64>.create([
              3037000500,
              3037000500,
            ], DType.float64),
            throwsArgumentError,
          );
          expect(
            () => NDArray<Float64>.create([0x2000000000000000], DType.float64),
            throwsArgumentError,
          );
        },
      );
    });

    group('M4: .npy Load Byte Verification', () {
      test(
        'corrupted/truncated .npy throws FormatException',
        () {
          final tempDir = Directory.systemTemp.createTempSync('npy_test');
          try {
            final filePath = '${tempDir.path}/truncated.npy';
            final file = File(filePath);

            // Save a valid NPY array
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            save(filePath, a);

            // Truncate the file by 8 bytes (incomplete data payload)
            final bytes = file.readAsBytesSync();
            final truncatedBytes = bytes.sublist(0, bytes.length - 8);
            file.writeAsBytesSync(truncatedBytes);

            // Loading the truncated file must throw FormatException
            expect(() => load(filePath), throwsA(isA<FormatException>()));
          } finally {
            tempDir.deleteSync(recursive: true);
          }
        },
        skip: const bool.fromEnvironment('dart.tool.dart2wasm')
            ? 'Uses dart:io File.readAsBytesSync/writeAsBytesSync'
            : false,
      );
    });
  });
}
