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
  group('hessenberg', () {
    test('simple real input Float64', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final res = hessenberg(a);
        final h = res.h;
        final q = res.q;

        expect(h.shape, equals([3, 3]));
        expect(q.shape, equals([3, 3]));

        expect(h.dtype, equals(DType.float64));
        expect(q.dtype, equals(DType.float64));

        // Check H is Hessenberg (h[2,0] should be 0)
        expect(h[[2, 0]], closeTo(0.0, 1e-10));

        // Check A = Q * H * Q^T
        final qT = q.transposed;
        final qH = matmul(q, h);
        final recon = matmul(qH, qT);

        for (var r = 0; r < 3; r++) {
          for (var c = 0; c < 3; c++) {
            expect(recon[[r, c]], closeTo(a[[r, c]], 1e-10));
          }
        }
      });
    });

    test('simple complex input Complex128', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
            Complex(5.0, 5.0),
            Complex(6.0, 6.0),
            Complex(7.0, 7.0),
            Complex(8.0, 8.0),
            Complex(9.0, 9.0),
          ],
          [3, 3],
          DType.complex128,
        );

        final res = hessenberg(a);
        final h = res.h;
        final q = res.q;

        expect(h.shape, equals([3, 3]));
        expect(q.shape, equals([3, 3]));

        expect(h.dtype, equals(DType.complex128));
        expect(q.dtype, equals(DType.complex128));

        // Check H is Hessenberg (h[2,0] should be 0)
        expect(h[[2, 0]].real, closeTo(0.0, 1e-10));
        expect(h[[2, 0]].imag, closeTo(0.0, 1e-10));

        // Check A = Q * H * Q^H
        final recon = matmul(matmul(q, h), conj(q).transposed);

        for (var r = 0; r < 3; r++) {
          for (var c = 0; c < 3; c++) {
            expect(recon[[r, c]].real, closeTo(a[[r, c]].real, 1e-10));
            expect(recon[[r, c]].imag, closeTo(a[[r, c]].imag, 1e-10));
          }
        }
      });
    });

    test('hessenberg with out parameters', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );
        final outH = NDArray<Float64>.zeros([3, 3], DType.float64);
        final outQ = NDArray<Float64>.zeros([3, 3], DType.float64);

        hessenberg(a, outH: outH, outQ: outQ);

        expect(outH[[2, 0]], closeTo(0.0, 1e-10));
        // Verify reconstruction A = Q * H * Q^T
        final recon = matmul(matmul(outQ, outH), outQ.transposed);
        for (var r = 0; r < 3; r++) {
          for (var c = 0; c < 3; c++) {
            expect(recon[[r, c]], closeTo(a[[r, c]], 1e-10));
          }
        }
      });
    });

    test('Batching Hessenberg', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [
            // Matrix 1
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            // Matrix 2
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
          ],
          [2, 3, 3],
          DType.float64,
        );

        final res = hessenberg(a);
        expect(res.h.shape, equals([2, 3, 3]));
        expect(res.q.shape, equals([2, 3, 3]));

        // Check H[0, 2, 0] == 0
        expect(res.h[[0, 2, 0]], closeTo(0.0, 1e-10));
        // Check H[1, 2, 0] == 0
        expect(res.h[[1, 2, 0]], closeTo(0.0, 1e-10));
      });
    });
  });
}
