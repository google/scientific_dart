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
  group('Workstream 4: Counting Sort for uint8', () {
    test(
      'Sorts 1D uint8 array correctly with Counting Sort (kind = quicksort)',
      () {
        final data = [255, 0, 128, 42, 0, 255, 128, 7, 3, 250, 100, 100, 100];
        final a = NDArray.fromList(data, [data.length], DType.uint8);
        final sorted = sort(a, kind: SortKind.quicksort);

        final expected = List<int>.from(data)..sort();
        expect(sorted.toList(), equals(expected));
        expect(sorted.dtype, equals(DType.uint8));
      },
    );

    test('Sorts large 1D uint8 array (all values 0..255)', () {
      final data = <int>[];
      for (var i = 255; i >= 0; i--) {
        data.addAll(List.filled(10, i));
      }
      final a = NDArray.fromList(data, [data.length], DType.uint8);
      final sorted = sort(a);

      final expected = List<int>.from(data)..sort();
      expect(sorted.toList(), equals(expected));
    });

    test('Sorts 2D uint8 matrix along axis 0 and 1', () {
      final a = NDArray.fromList(
        [100, 20, 50, 10, 200, 30],
        [2, 3],
        DType.uint8,
      );

      final sortedAxis1 = sort(a, axis: 1);
      expect(sortedAxis1.toList(), equals([20, 50, 100, 10, 30, 200]));

      final sortedAxis0 = sort(a, axis: 0);
      expect(sortedAxis0.toList(), equals([10, 20, 30, 100, 200, 50]));
    });
  });

  group('Workstream 4: Level 2 BLAS GEMV', () {
    test('Matrix (2D) x Vector (1D) float64', () {
      // A: 2x3, x: 3 -> y: 2
      final a = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        DType.float64,
      );
      final x = NDArray.fromList([0.5, -1.0, 2.0], [3], DType.float64);

      final y = matmul(a, x);
      expect(y.shape, equals([2]));
      expect(y.dtype, equals(DType.float64));
      // 1*0.5 + 2*(-1) + 3*2 = 0.5 - 2 + 6 = 4.5
      // 4*0.5 + 5*(-1) + 6*2 = 2 - 5 + 12 = 9.0
      expect(y.data[0], closeTo(4.5, 1e-12));
      expect(y.data[1], closeTo(9.0, 1e-12));
    });

    test('Vector (1D) x Matrix (2D) float64', () {
      // x: 2, B: 2x3 -> y: 3
      final x = NDArray.fromList([2.0, 3.0], [2], DType.float64);
      final b = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        DType.float64,
      );

      final y = matmul(x, b);
      expect(y.shape, equals([3]));
      // 2*1 + 3*4 = 14
      // 2*2 + 3*5 = 19
      // 2*3 + 3*6 = 24
      expect(y.data[0], closeTo(14.0, 1e-12));
      expect(y.data[1], closeTo(19.0, 1e-12));
      expect(y.data[2], closeTo(24.0, 1e-12));
    });

    test('Matrix (2D) x Vector (1D) float32', () {
      final a = NDArray.fromList([2.0, 1.0, -1.0, 3.0], [2, 2], DType.float32);
      final x = NDArray.fromList([3.0, 4.0], [2], DType.float32);

      final y = matmul(a, x);
      expect(y.shape, equals([2]));
      expect(y.dtype, equals(DType.float32));
      // 2*3 + 1*4 = 10
      // -1*3 + 3*4 = 9
      expect(y.data[0], closeTo(10.0, 1e-5));
      expect(y.data[1], closeTo(9.0, 1e-5));
    });

    test('Vector (1D) x Matrix (2D) float32', () {
      final x = NDArray.fromList([1.0, 2.0], [2], DType.float32);
      final b = NDArray.fromList([3.0, 4.0, 5.0, 6.0], [2, 2], DType.float32);

      final y = matmul(x, b);
      expect(y.shape, equals([2]));
      // 1*3 + 2*5 = 13
      // 1*4 + 2*6 = 16
      expect(y.data[0], closeTo(13.0, 1e-5));
      expect(y.data[1], closeTo(16.0, 1e-5));
    });

    test('Transposed Matrix x Vector float64 (GEMV trans)', () {
      // A is 3x2, transposed to 2x3 view
      final a = NDArray.fromList(
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        [3, 2],
        DType.float64,
      );
      final aT = a.transpose(); // 2x3 view
      final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

      final y = matmul(aT, x);
      expect(y.shape, equals([2]));
      // aT row 0: [1, 2, 3] . [1, 2, 3] = 14
      // aT row 1: [4, 5, 6] . [1, 2, 3] = 4 + 10 + 18 = 32
      expect(y.data[0], closeTo(14.0, 1e-12));
      expect(y.data[1], closeTo(32.0, 1e-12));
    });

    test('Vector x Transposed Matrix float64 (GEMV trans)', () {
      final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final b = NDArray.fromList(
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        [3, 2],
        DType.float64,
      );
      final bT = b.transpose(); // 2x3 view

      final y = matmul(x, bT);
      expect(y.shape, equals([3]));
      // x . [1, 4] = 1*1 + 2*4 = 9
      // x . [2, 5] = 1*2 + 2*5 = 12
      // x . [3, 6] = 1*3 + 2*6 = 15
      expect(y.data[0], closeTo(9.0, 1e-12));
      expect(y.data[1], closeTo(12.0, 1e-12));
      expect(y.data[2], closeTo(15.0, 1e-12));
    });

    test('Batch 3D Matrix x Vector float64', () {
      // Shape [2, 2, 3] x [3] -> [2, 2]
      final a = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [2, 2, 3],
        DType.float64,
      );
      final x = NDArray.fromList([1.0, 0.0, -1.0], [3], DType.float64);

      final y = matmul(a, x);
      expect(y.shape, equals([2, 2]));
      // Batch 0: [1 - 3, 4 - 6] = [-2, -2]
      // Batch 1: [7 - 9, 10 - 12] = [-2, -2]
      expect(y.data[0], closeTo(-2.0, 1e-12));
      expect(y.data[1], closeTo(-2.0, 1e-12));
      expect(y.data[2], closeTo(-2.0, 1e-12));
      expect(y.data[3], closeTo(-2.0, 1e-12));
    });

    test('Complex128 Matrix x Vector', () {
      final a = NDArray.fromList(
        [
          Complex(1.0, 1.0),
          Complex(2.0, 0.0),
          Complex(0.0, 1.0),
          Complex(1.0, -1.0),
        ],
        [2, 2],
        DType.complex128,
      );
      final x = NDArray.fromList(
        [Complex(1.0, 0.0), Complex(0.0, 1.0)],
        [2],
        DType.complex128,
      );

      final y = matmul(a, x);
      expect(y.shape, equals([2]));
      expect(y.dtype, equals(DType.complex128));
      // row 0: (1+i)*1 + 2*i = 1 + 3i
      // row 1: i*1 + (1-i)*i = i + i - i^2 = 1 + 2i
      final c0 = y.data[0];
      final c1 = y.data[1];
      expect(c0.real, closeTo(1.0, 1e-12));
      expect(c0.imag, closeTo(3.0, 1e-12));
      expect(c1.real, closeTo(1.0, 1e-12));
      expect(c1.imag, closeTo(2.0, 1e-12));
    });
  });

  group('Workstream 4: Copying Anti-Patterns Eliminated', () {
    test('real() with out buffer on real array uses copy', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      final out = NDArray.zeros([2, 2], DType.float64);

      final r = real(a, out: out);
      expect(identical(r, out), isTrue);
      expect(r.toList(), equals([1.0, 2.0, 3.0, 4.0]));
    });

    test('save() and load() roundtrip with strided non-contiguous array', () {
      final a = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3, 2],
        DType.float64,
      );
      final aT = a.transpose(); // Non-contiguous [2, 3]

      final path = '/tmp/test_ws4_strided.npy';
      save(path, aT);
      final loaded = load(path);

      expect(loaded.shape, equals([2, 3]));
      expect(loaded.dtype, equals(DType.float64));
      expect(loaded.toList(), equals([1.0, 3.0, 5.0, 2.0, 4.0, 6.0]));
    });

    test('sort() and partition() on non-contiguous strided views', () {
      final a = NDArray.fromList(
        [5.0, 1.0, 3.0, 4.0, 2.0, 6.0],
        [3, 2],
        DType.float64,
      );
      final aT = a.transpose(); // [2, 3]: [[5, 3, 2], [1, 4, 6]]

      final sorted = sort(aT, axis: -1);
      expect(sorted.shape, equals([2, 3]));
      expect(sorted.toList(), equals([2.0, 3.0, 5.0, 1.0, 4.0, 6.0]));

      final part = partition(aT, 1, axis: -1);
      expect(part.shape, equals([2, 3]));
      expect(part.data[1], closeTo(3.0, 1e-12)); // partition pivot at index 1
      expect(part.data[4], closeTo(4.0, 1e-12));
    });
  });
}
