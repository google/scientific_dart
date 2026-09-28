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
  group('schur', () {
    test('simple real input, real output', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [5.0, 7.0, -2.0, -4.0],
          [2, 2],
          DType.float64,
        );

        final res = schur(a, output: SchurForm.real);
        final t = res.t;
        final z = res.z;

        expect(t.shape, equals([2, 2]));
        expect(z.shape, equals([2, 2]));

        expect(t.dtype, equals(DType.float64));
        expect(z.dtype, equals(DType.float64));

        // Check T is quasi-upper triangular (t[1,0] should be 0 since eigenvalues are real: 3 and -2)
        expect(t[[1, 0]], closeTo(0.0, 1e-10));

        // Check A = Z * T * Z^T
        final zT = z.transposed;
        final zMulT = matmul(z, t);
        final recon = matmul(zMulT, zT);

        for (var r = 0; r < 2; r++) {
          for (var c = 0; c < 2; c++) {
            expect(recon[[r, c]], closeTo(a[[r, c]], 1e-10));
          }
        }
      });
    });

    test('simple real input, complex output', () {
      NDArray.scope(() {
        // Matrix with complex eigenvalues: [[3, -2], [4, -1]] -> eigenvalues 1 +/- 2i
        final a = NDArray.fromList(
          [3.0, -2.0, 4.0, -1.0],
          [2, 2],
          DType.float64,
        );

        final res = schur(a, output: SchurForm.complex);
        final t = res.t;
        final z = res.z;

        expect(t.shape, equals([2, 2]));
        expect(z.shape, equals([2, 2]));

        expect(t.dtype, equals(DType.complex128));
        expect(z.dtype, equals(DType.complex128));

        // For complex Schur, T must be strictly upper triangular (t[1,0] == 0)
        expect(t[[1, 0]].real, closeTo(0.0, 1e-10));
        expect(t[[1, 0]].imag, closeTo(0.0, 1e-10));

        // Check A = Z * T * Z^H
        final recon = matmul(matmul(z, t), conj(z).transposed);

        for (var r = 0; r < 2; r++) {
          for (var c = 0; c < 2; c++) {
            // recon should be real since A was real
            expect(recon[[r, c]].real, closeTo(a[[r, c]], 1e-10));
            expect(recon[[r, c]].imag, closeTo(0.0, 1e-10));
          }
        }
      });
    });

    test('Complex input Schur', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final res = schur(a);
        final t = res.t;
        final z = res.z;

        expect(t.dtype, equals(DType.complex128));
        expect(z.dtype, equals(DType.complex128));

        // T must be upper triangular
        expect(t[[1, 0]].real, closeTo(0.0, 1e-10));
        expect(t[[1, 0]].imag, closeTo(0.0, 1e-10));

        // Check A = Z * T * Z^H
        final recon = matmul(matmul(z, t), conj(z).transposed);

        for (var r = 0; r < 2; r++) {
          for (var c = 0; c < 2; c++) {
            expect(recon[[r, c]].real, closeTo(a[[r, c]].real, 1e-10));
            expect(recon[[r, c]].imag, closeTo(a[[r, c]].imag, 1e-10));
          }
        }
      });
    });

    test('schur with out parameters', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [5.0, 7.0, -2.0, -4.0],
          [2, 2],
          DType.float64,
        );
        final outT = NDArray<Float64>.zeros([2, 2], DType.float64);
        final outZ = NDArray<Float64>.zeros([2, 2], DType.float64);

        schur(a, outT: outT, outZ: outZ);

        expect(outT[[1, 0]], closeTo(0.0, 1e-10));
        // Verify reconstruction A = Z * T * Z^T
        final recon = matmul(matmul(outZ, outT), outZ.transposed);
        for (var r = 0; r < 2; r++) {
          for (var c = 0; c < 2; c++) {
            expect(recon[[r, c]], closeTo(a[[r, c]], 1e-10));
          }
        }
      });
    });

    test('Batching Schur', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [
            // Matrix 1
            5.0, 7.0,
            -2.0, -4.0,
            // Matrix 2
            6.0, 8.0,
            -1.0, -3.0,
          ],
          [2, 2, 2],
          DType.float64,
        );

        final res = schur(a);
        expect(res.t.shape, equals([2, 2, 2]));
        expect(res.z.shape, equals([2, 2, 2]));

        // Check first matrix in batch
        expect(res.t[[0, 1, 0]], closeTo(0.0, 1e-10));
        // Check second matrix in batch
        expect(res.t[[1, 1, 0]], closeTo(0.0, 1e-10));
      });
    });
  });
}
