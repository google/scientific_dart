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

import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Matrix Multiplication', () {
    test('1D vector dot product', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = GpuArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

        final dot = a.dot(b);
        expect(dot.scalar, equals(32.0)); // 1*4 + 2*5 + 3*6 = 32
      });
    });

    test('2D matrix multiplication', () {
      ResourceScope.scope(() {
        // A: 2x3
        final a = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );

        // B: 3x2
        final b = GpuArray.fromList(
          [
            [7.0, 8.0],
            [9.0, 1.0],
            [2.0, 3.0],
          ],
          [3, 2],
          DType.float64,
        );

        // Result: 2x2
        // [1*7+2*9+3*2, 1*8+2*1+3*3] = [31, 19]
        // [4*7+5*9+6*2, 4*8+5*1+6*3] = [85, 55]
        final c = a.matmul(b);
        expect(c.shape, equals([2, 2]));
        expect(
          c.toNestedList(),
          equals([
            [31.0, 19.0],
            [85.0, 55.0],
          ]),
        );
      });
    });

    test('Batched 3D matrix multiplication', () {
      ResourceScope.scope(() {
        // 2 batches of 2x2 matrices
        final a = GpuArray.fromList(
          [
            [
              [1.0, 2.0],
              [3.0, 4.0],
            ],
            [
              [5.0, 6.0],
              [7.0, 8.0],
            ],
          ],
          [2, 2, 2],
          DType.float64,
        );

        // Identity 2x2
        final eye = GpuArray.fromList(
          [
            [1.0, 0.0],
            [0.0, 1.0],
          ],
          [1, 2, 2],
          DType.float64,
        );

        final res = a.matmul(eye);
        expect(res.shape, equals([2, 2, 2]));
        expect(
          res.toNestedList(),
          equals([
            [
              [1.0, 2.0],
              [3.0, 4.0],
            ],
            [
              [5.0, 6.0],
              [7.0, 8.0],
            ],
          ]),
        );
      });
    });

    test('Shape mismatch throws GpuShapeMismatchException', () {
      ResourceScope.scope(() {
        final a = GpuArray.zeros([2, 3], DType.float64);
        final b = GpuArray.zeros([4, 2], DType.float64);

        expect(() => a.matmul(b), throwsA(isA<GpuShapeMismatchException>()));
      });
    });

    test('Complex64 and Complex128 1D and 2D matrix multiplication', () {
      ResourceScope.scope(() {
        // 1D dot product with Complex64
        final v1 = GpuArray.fromList(
          [Complex(1.0, 2.0), Complex(3.0, -1.0)],
          [2],
          DType.complex64,
        );
        final v2 = GpuArray.fromList(
          [Complex(2.0, 1.0), Complex(0.0, 4.0)],
          [2],
          DType.complex64,
        );
        // (1+2i)(2+i) = (2-2) + (1+4)i = 5i
        // (3-i)(4i) = 4 + 12i
        // Total = 4 + 17i
        final dotC64 = v1.dot(v2);
        expect(dotC64.dtype, equals(DType.complex64));
        final dotScalar = dotC64.scalar as Complex;
        expect(dotScalar.real, closeTo(4.0, 1e-5));
        expect(dotScalar.imag, closeTo(17.0, 1e-5));

        // 2D matmul with Complex128
        final a = GpuArray.fromList(
          [
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [Complex(0.0, 1.0), Complex(-2.0, 0.0)],
          ],
          [2, 2],
          DType.complex128,
        );

        final b = GpuArray.fromList(
          [
            [Complex(2.0, -1.0), Complex(0.0, 3.0)],
            [Complex(1.0, 1.0), Complex(4.0, 2.0)],
          ],
          [2, 2],
          DType.complex128,
        );

        final c = a.matmul(b);
        expect(c.dtype, equals(DType.complex128));
        expect(c.shape, equals([2, 2]));

        final expected = [
          [Complex(3.0, 10.0), Complex(-2.0, 25.0)],
          [Complex(-1.0, 0.0), Complex(-11.0, -4.0)],
        ];

        final actual = c.toNestedList();
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 2; j++) {
            final actVal = actual[i][j] as Complex;
            final expVal = expected[i][j];
            expect(actVal.real, closeTo(expVal.real, 1e-5));
            expect(actVal.imag, closeTo(expVal.imag, 1e-5));
          }
        }
      });
    });

    test('matmul and dot support out: destination array', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ],
          [2, 2],
          DType.float64,
        );
        final b = GpuArray.fromList(
          [
            [2.0, 0.0],
            [1.0, 2.0],
          ],
          [2, 2],
          DType.float64,
        );
        final out = GpuArray.zeros([2, 2], DType.float64);
        final res = a.matmul(b, out: out);
        expect(identical(res, out), isTrue);
        expect(
          out.toNestedList(),
          equals([
            [4.0, 4.0],
            [10.0, 8.0],
          ]),
        );

        final badOut = GpuArray.zeros([3, 3], DType.float64);
        expect(
          () => a.matmul(b, out: badOut),
          throwsA(isA<GpuShapeMismatchException>()),
        );
      });
    });
  });
}
