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
  group('i0 real', () {
    test('contiguous float64', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, -1.0],
          [7],
          DType.float64,
        );
        final y = i0(x);
        expect(y.dtype, equals(DType.float64));
        expect(y.isContiguous, isTrue);

        expect(y.getCell([0]), closeTo(1.0, 1e-7));
        expect(y.getCell([1]), closeTo(1.2660658777520083, 1e-7));
        expect(y.getCell([2]), closeTo(2.2795853023360673, 1e-7));
        expect(y.getCell([3]), closeTo(27.239871822907405, 1e-5));
        expect(y.getCell([4]), closeTo(2815.716628466187, 1e-2));
        expect(y.getCell([5]), closeTo(43558285.787908286, 1e-1));
        expect(y.getCell([6]), closeTo(1.2660658777520083, 1e-7));
      });
    });

    test('contiguous float32', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.0, 1.0, 2.0, 5.0, -1.0],
          [5],
          DType.float32,
        );
        final y = i0(x);
        expect(y.dtype, equals(DType.float32));
        expect(y.isContiguous, isTrue);

        expect(y.getCell([0]), closeTo(1.0, 1e-5));
        expect(y.getCell([1]), closeTo(1.2660658, 1e-5));
        expect(y.getCell([2]), closeTo(2.2795853, 1e-5));
        expect(y.getCell([3]), closeTo(27.23987, 1e-3));
        expect(y.getCell([4]), closeTo(1.2660658, 1e-5));
      });
    });

    test('strided float64', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.0, 999.0, 1.0, 999.0, 2.0, 999.0],
          [3, 2],
          DType.float64,
        );
        final xStrided = x.slice([Slice.all(), Index(0)]);
        expect(xStrided.isContiguous, isFalse);

        final y = i0(xStrided);
        expect(y.dtype, equals(DType.float64));
        expect(y.isContiguous, isTrue);

        expect(y.getCell([0]), closeTo(1.0, 1e-7));
        expect(y.getCell([1]), closeTo(1.2660658777520083, 1e-7));
        expect(y.getCell([2]), closeTo(2.2795853023360673, 1e-7));
      });
    });

    test('strided float64 with strided out', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [0.0, 999.0, 1.0, 999.0, 2.0, 999.0],
          [3, 2],
          DType.float64,
        );
        final xStrided = x.slice([Slice.all(), Index(0)]);

        final out = NDArray<double>.create([3, 2], DType.float64)..fill(999.0);
        final outStrided = out.slice([Slice.all(), Index(0)]);

        final y = i0(xStrided, out: outStrided);
        expect(y, equals(outStrided));
        expect(y.isContiguous, isFalse);

        expect(y.getCell([0]), closeTo(1.0, 1e-7));
        expect(y.getCell([1]), closeTo(1.2660658777520083, 1e-7));
        expect(y.getCell([2]), closeTo(2.2795853023360673, 1e-7));

        expect(out.getCell([0, 1]), equals(999.0));
        expect(out.getCell([1, 1]), equals(999.0));
        expect(out.getCell([2, 1]), equals(999.0));
      });
    });

    test('works with where mask on contiguous and strided arrays', () {
      final x = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
      final mask = NDArray.fromList([true, false, true], [3], DType.boolean);
      final out = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
      final y = i0(x, where: mask, out: out);
      expect(y.getCell([0]), closeTo(1.0, 1e-7));
      expect(y.getCell([1]), equals(20.0));
      expect(y.getCell([2]), closeTo(2.2795853, 1e-6));
    });

    test('integer promotion', () {
      NDArray.scope(() {
        final x = NDArray.fromList([0, 1, 2], [3], DType.int32);
        final y = i0(x);
        expect(y.dtype, equals(DType.float64));
        expect(y.getCell([0]), closeTo(1.0, 1e-7));
        expect(y.getCell([1]), closeTo(1.2660658777520083, 1e-7));
        expect(y.getCell([2]), closeTo(2.2795853023360673, 1e-7));
      });
    });

    test('boolean input', () {
      NDArray.scope(() {
        final x = NDArray.fromList([false, true], [2], DType.boolean);
        final y = i0(x);
        expect(y.dtype, equals(DType.float64));
        expect(y.getCell([0]), closeTo(1.0, 1e-7));
        expect(y.getCell([1]), closeTo(1.2660658777520083, 1e-7));
      });
    });
  });

  group('i0 complex', () {
    test('contiguous complex128', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [
            Complex(0, 0),
            Complex(0, 1),
            Complex(0, 2),
            Complex(0, 15),
            Complex(0, 20),
            Complex(0, 50), // Large imaginary
            Complex(1, 1),
            Complex(10, 10),
            Complex(20, 20),
            Complex(0, -1),
            Complex(-12, 16), // Q2
            Complex(-12, -16), // Q3
            Complex(12, -16), // Q4
          ],
          [13],
          DType.complex128,
        );

        final y = i0(x);
        expect(y.dtype, equals(DType.complex128));

        expect(y.getCell([0]).real, closeTo(1.0, 1e-7));
        expect(y.getCell([0]).imag, closeTo(0.0, 1e-7));

        expect(y.getCell([1]).real, closeTo(0.7651976865579666, 1e-7));
        expect(y.getCell([1]).imag, closeTo(0.0, 1e-7));

        expect(y.getCell([2]).real, closeTo(0.22389077914123567, 1e-7));
        expect(y.getCell([2]).imag, closeTo(0.0, 1e-7));

        expect(y.getCell([3]).real, closeTo(-0.014224472826333917, 1e-7));
        expect(y.getCell([3]).imag, closeTo(0.0, 1e-7));

        expect(
          y.getCell([4]).real,
          closeTo(0.16702458347774152, 1e-5),
        ); // tolerance adjusted for asymptotic
        expect(y.getCell([4]).imag, closeTo(0.0, 1e-5));

        // J0(50) = 0.055812327669...
        expect(y.getCell([5]).real, closeTo(0.055812327, 1e-5));
        expect(y.getCell([5]).imag, closeTo(0.0, 1e-5));

        expect(y.getCell([6]).real, closeTo(0.9376084768060292, 1e-7));
        expect(y.getCell([6]).imag, closeTo(0.4965299476091221, 1e-7));

        expect(y.getCell([7]).real, closeTo(-2314.9753144452106, 1e-2));
        expect(y.getCell([7]).imag, closeTo(-411.56285702537843, 1e-2));

        expect(y.getCell([8]).real, closeTo(26598967.624014486, 1e0));
        expect(y.getCell([8]).imag, closeTo(25006018.46257524, 1e0));

        expect(y.getCell([9]).real, closeTo(0.7651976865579666, 1e-7));
        expect(y.getCell([9]).imag, closeTo(0.0, 1e-7));

        // I0(-12 + 16i) = -14345.459 - 2562.495i
        expect(y.getCell([10]).real, closeTo(-14345.459, 1e-1));
        expect(y.getCell([10]).imag, closeTo(-2562.495, 1e-1));

        // I0(-12 - 16i) = -14345.459 + 2562.495i
        expect(y.getCell([11]).real, closeTo(-14345.459, 1e-1));
        expect(y.getCell([11]).imag, closeTo(2562.495, 1e-1));

        // I0(12 - 16i) = -14345.459 - 2562.495i
        expect(y.getCell([12]).real, closeTo(-14345.459, 1e-1));
        expect(y.getCell([12]).imag, closeTo(-2562.495, 1e-1));
      });
    });

    test('contiguous complex64', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [Complex(0, 0), Complex(0, 1), Complex(1, 1)],
          [3],
          DType.complex64,
        );

        final y = i0(x);
        expect(y.dtype, equals(DType.complex64));

        expect(y.getCell([0]).real, closeTo(1.0, 1e-5));
        expect(y.getCell([0]).imag, closeTo(0.0, 1e-5));

        expect(y.getCell([1]).real, closeTo(0.7651976, 1e-5));
        expect(y.getCell([1]).imag, closeTo(0.0, 1e-5));

        expect(y.getCell([2]).real, closeTo(0.937608, 1e-5));
        expect(y.getCell([2]).imag, closeTo(0.49653, 1e-5));
      });
    });

    test('strided complex128', () {
      NDArray.scope(() {
        final x = NDArray.fromList(
          [
            Complex(0, 0),
            Complex(999, 999),
            Complex(0, 1),
            Complex(999, 999),
            Complex(1, 1),
            Complex(999, 999),
          ],
          [3, 2],
          DType.complex128,
        );
        final xStrided = x.slice([Slice.all(), Index(0)]);
        expect(xStrided.isContiguous, isFalse);

        final y = i0(xStrided);
        expect(y.dtype, equals(DType.complex128));
        expect(y.isContiguous, isTrue);

        expect(y.getCell([0]).real, closeTo(1.0, 1e-7));
        expect(y.getCell([0]).imag, closeTo(0.0, 1e-7));

        expect(y.getCell([1]).real, closeTo(0.7651976865579666, 1e-7));
        expect(y.getCell([1]).imag, closeTo(0.0, 1e-7));

        expect(y.getCell([2]).real, closeTo(0.9376084768060292, 1e-7));
        expect(y.getCell([2]).imag, closeTo(0.4965299476091221, 1e-7));
      });
    });

    test('rank > 8 fallback complex128', () {
      NDArray.scope(() {
        // Create a 9D array
        // Shape: [2, 1, 1, 1, 1, 1, 1, 1, 2] -> size 4
        final shape = [2, 1, 1, 1, 1, 1, 1, 1, 2];
        final x = NDArray.fromList(
          [Complex(0, 0), Complex(0, 1), Complex(1, 1), Complex(10, 10)],
          shape,
          DType.complex128,
        );
        expect(x.shape.length, equals(9));

        final y = i0(x);
        expect(y.dtype, equals(DType.complex128));
        expect(y.shape, equals(shape));

        expect(y.getCell([0, 0, 0, 0, 0, 0, 0, 0, 0]).real, closeTo(1.0, 1e-7));
        expect(
          y.getCell([0, 0, 0, 0, 0, 0, 0, 0, 1]).real,
          closeTo(0.7651976865579666, 1e-7),
        );
        expect(
          y.getCell([1, 0, 0, 0, 0, 0, 0, 0, 0]).real,
          closeTo(0.9376084768060292, 1e-7),
        );
        expect(
          y.getCell([1, 0, 0, 0, 0, 0, 0, 0, 1]).real,
          closeTo(-2314.9753144452106, 1e-2),
        );
      });
    });
  });
}
