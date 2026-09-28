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
import 'package:ndarray/ndarray.dart';

void main() {
  group('eigh', () {
    test('simple real symmetric Float64', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);

        final res = eigh(a);
        final w = res.eigenvalues;
        final v = res.eigenvectors;

        expect(w.shape, equals([2]));
        expect(v.shape, equals([2, 2]));

        expect(w.dtype, equals(DType.float64));
        expect(v.dtype, equals(DType.float64));

        expect(w[[0]], closeTo(1.0, 1e-10));
        expect(w[[1]], closeTo(3.0, 1e-10));

        // Check eigenvectors: A * v = v * diag(w)
        final av0 = matmul(
          a,
          NDArray.fromList(
            [
              v[[0, 0]],
              v[[1, 0]],
            ],
            [2, 1],
            DType.float64,
          ),
        );
        final w0v0 =
            NDArray.fromList(
              [
                v[[0, 0]],
                v[[1, 0]],
              ],
              [2, 1],
              DType.float64,
            ) *
            w[[0]];

        expect(av0[[0, 0]], closeTo(w0v0[[0, 0]], 1e-10));
        expect(av0[[1, 0]], closeTo(w0v0[[1, 0]], 1e-10));

        final av1 = matmul(
          a,
          NDArray.fromList(
            [
              v[[0, 1]],
              v[[1, 1]],
            ],
            [2, 1],
            DType.float64,
          ),
        );
        final w1v1 =
            NDArray.fromList(
              [
                v[[0, 1]],
                v[[1, 1]],
              ],
              [2, 1],
              DType.float64,
            ) *
            w[[1]];

        expect(av1[[0, 0]], closeTo(w1v1[[0, 0]], 1e-10));
        expect(av1[[1, 0]], closeTo(w1v1[[1, 0]], 1e-10));
      });
    });

    test('Float32 real symmetric', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2.0, 1.0, 1.0, 2.0], [2, 2], DType.float32);

        final res = eigh(a);
        final w = res.eigenvalues;
        final v = res.eigenvectors;

        expect(w.dtype, equals(DType.float32));
        expect(v.dtype, equals(DType.float32));

        expect(w[[0]], closeTo(1.0, 1e-5));
        expect(w[[1]], closeTo(3.0, 1e-5));
      });
    });

    test('Complex128 Hermitian', () {
      NDArray.scope(() {
        // Hermitian matrix: [[1, 1+i], [1-i, 2]]
        // eigenvalues are real: (3 +/- sqrt(5))/2
        // approx 2.6180339887 and 0.3819660113
        final a = NDArray.fromList(
          [
            Complex(1.0, 0.0),
            Complex(1.0, 1.0),
            Complex(1.0, -1.0),
            Complex(2.0, 0.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final res = eigh(a);
        final w = res.eigenvalues;
        final v = res.eigenvectors;

        expect(w.dtype, equals(DType.float64));
        expect(v.dtype, equals(DType.complex128));

        expect(w[[0]], closeTo(0.0, 1e-9));
        expect(w[[1]], closeTo(3.0, 1e-9));

        final v0 = NDArray.fromList(
          [
            v[[0, 0]],
            v[[1, 0]],
          ],
          [2, 1],
          DType.complex128,
        );
        final av0 = matmul(a, v0);

        final ev0_0 = v[[0, 0]] as Complex;
        final ev0_1 = v[[1, 0]] as Complex;
        final scale = w[[0]] as double;
        final expected_0 = ev0_0 * scale;
        final expected_1 = ev0_1 * scale;

        expect(av0[[0, 0]].real, closeTo(expected_0.real, 1e-9));
        expect(av0[[0, 0]].imag, closeTo(expected_0.imag, 1e-9));
        expect(av0[[1, 0]].real, closeTo(expected_1.real, 1e-9));
        expect(av0[[1, 0]].imag, closeTo(expected_1.imag, 1e-9));
      });
    });

    test('Integer promotion', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2, 1, 1, 2], [2, 2], DType.int32);

        final res = eigh(a);
        expect(res.eigenvalues.dtype, equals(DType.float64));
        expect(res.eigenvectors.dtype, equals(DType.float64));
        expect(res.eigenvalues[[0]], closeTo(1.0, 1e-10));
      });
    });

    test('Batching eigh', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0],
          [2, 2, 2],
          DType.float64,
        );

        final res = eigh(a);
        final w = res.eigenvalues;
        final v = res.eigenvectors;

        expect(w.shape, equals([2, 2]));
        expect(v.shape, equals([2, 2, 2]));

        expect(w[[0, 0]], closeTo(1.0, 1e-10));
        expect(w[[0, 1]], closeTo(3.0, 1e-10));

        expect(w[[1, 0]], closeTo(1.0, 1e-10));
        expect(w[[1, 1]], closeTo(5.0, 1e-10));
      });
    });

    test('eigvalsh only eigenvalues', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);

        final w = eigvalsh(a);
        expect(w.shape, equals([2]));
        expect(w.dtype, equals(DType.float64));
        expect(w[[0]], closeTo(1.0, 1e-10));
        expect(w[[1]], closeTo(3.0, 1e-10));
      });
    });

    test('eigh with out parameters', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
        final outW = NDArray<num>.zeros([2], DType.float64);
        final outV = NDArray<Float64>.zeros([2, 2], DType.float64);

        eigh(a, outEigenvalues: outW, outEigenvectors: outV);

        expect(outW[[0]], closeTo(1.0, 1e-10));
        expect(outW[[1]], closeTo(3.0, 1e-10));

        // Verify outV contains eigenvectors: A * outV_i = outW_i * outV_i
        final v0 = NDArray.fromList(
          [
            outV[[0, 0]],
            outV[[1, 0]],
          ],
          [2, 1],
          DType.float64,
        );
        final av0 = matmul(a, v0);
        final w0v0 = v0 * outW[[0]];
        expect(av0[[0, 0]], closeTo(w0v0[[0, 0]], 1e-10));
        expect(av0[[1, 0]], closeTo(w0v0[[1, 0]], 1e-10));
      });
    });

    test('eigvalsh with out parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
        final outW = NDArray<num>.zeros([2], DType.float64);

        eigvalsh(a, out: outW);

        expect(outW[[0]], closeTo(1.0, 1e-10));
        expect(outW[[1]], closeTo(3.0, 1e-10));
      });
    });

    test('invalid shapes', () {
      NDArray.scope(() {
        final a = NDArray.zeros([2, 3], DType.float64);
        expect(() => eigh(a), throwsArgumentError);
        expect(() => eigvalsh(a), throwsArgumentError);

        final b = NDArray.zeros([2], DType.float64);
        expect(() => eigh(b), throwsArgumentError);
        expect(() => eigvalsh(b), throwsArgumentError);
      });
    });
  });
}
