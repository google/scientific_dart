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
import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('Decompositions', () {
    group('LAPACK Inversion (inv) tests', () {
      test(
        'Invert Float64 square matrix and check identity reconstruction',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([4.0, 7.0, 2.0, 6.0]),
            [2, 2],
            DType.float64,
          );

          final aInv = inv(a);
          expect(aInv.shape, [2, 2]);
          expect(aInv.dtype, DType.float64);

          // Check exact values (approx 0.6, -0.7, -0.2, 0.4)
          expect(aInv.toList()[0], closeTo(0.6, 1e-6));
          expect(aInv.toList()[1], closeTo(-0.7, 1e-6));
          expect(aInv.toList()[2], closeTo(-0.2, 1e-6));
          expect(aInv.toList()[3], closeTo(0.4, 1e-6));
        }),
      );

      test(
        'Invert Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float32,
          );

          final aInv = inv(a);
          expect(aInv.dtype, DType.float32);
          // expected inverse for [[1,2],[3,4]] is [[-2, 1], [1.5, -0.5]]
          expect(aInv.toList()[0], closeTo(-2.0, 1e-5));
          expect(aInv.toList()[1], closeTo(1.0, 1e-5));
          expect(aInv.toList()[2], closeTo(1.5, 1e-5));
          expect(aInv.toList()[3], closeTo(-0.5, 1e-5));
        }),
      );

      test(
        'Singular matrix throws SingularMatrixException',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float64,
          ); // dependent rows, det = 0

          expect(() => inv(a), throwsA(isA<SingularMatrixException>()));
        }),
      );

      test(
        'Singular Float32 matrix throws SingularMatrixException',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float32,
          ); // dependent rows, det = 0

          expect(() => inv(a), throwsA(isA<SingularMatrixException>()));
        }),
      );

      test(
        'Batched 3D inv for Float64 and Float32',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([4.0, 7.0, 2.0, 6.0, 1.0, 2.0, 3.0, 4.0]),
            [2, 2, 2],
            DType.float64,
          );

          final aInv = inv(a);
          expect(aInv.shape, [2, 2, 2]);
          expect(aInv.dtype, DType.float64);

          expect(aInv.toList()[0], closeTo(0.6, 1e-6));
          expect(aInv.toList()[1], closeTo(-0.7, 1e-6));
          expect(aInv.toList()[2], closeTo(-0.2, 1e-6));
          expect(aInv.toList()[3], closeTo(0.4, 1e-6));

          expect(aInv.toList()[4], closeTo(-2.0, 1e-6));
          expect(aInv.toList()[5], closeTo(1.0, 1e-6));
          expect(aInv.toList()[6], closeTo(1.5, 1e-6));
          expect(aInv.toList()[7], closeTo(-0.5, 1e-6));

          final out = NDArray.zeros([2, 2, 2], DType.float64);
          final res = inv(a, out: out);
          expect(identical(res, out), true);
          expect(out.toList()[0], closeTo(0.6, 1e-6));
          expect(out.toList()[4], closeTo(-2.0, 1e-6));
        }),
      );

      test(
        'Batched 4D inv Complex128',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(0.0, 0.0),
              Complex(0.0, 0.0),
              Complex(1.0, 0.0),
            ],
            [1, 1, 2, 2],
            DType.complex128,
          );
          final aInv = inv(a);
          expect(aInv.shape, [1, 1, 2, 2]);
          expect(aInv.toList()[0], Complex(1.0, 0.0));
          expect(aInv.toList()[3], Complex(1.0, 0.0));
        }),
      );
    });

    group('Cholesky Decomposition tests', () {
      test(
        'Cholesky lower triangular factor and L * L^T reconstruction',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              4.0,
              12.0,
              -16.0,
              12.0,
              37.0,
              -43.0,
              -16.0,
              -43.0,
              98.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final l = cholesky(a);

          expect(l.shape, [3, 3]);

          // Check that the strictly upper triangular elements are zeroed out!
          expect(l.toList()[1], 0.0);
          expect(l.toList()[2], 0.0);
          expect(l.toList()[5], 0.0);

          // Check exact expected lower triangular values
          // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
          expect(l.toList()[0], closeTo(2.0, 1e-6));
          expect(l.toList()[3], closeTo(6.0, 1e-6));
          expect(l.toList()[4], closeTo(1.0, 1e-6));
          expect(l.toList()[6], closeTo(-8.0, 1e-6));
          expect(l.toList()[7], closeTo(5.0, 1e-6));
          expect(l.toList()[8], closeTo(3.0, 1e-6));
        }),
      );

      test(
        'Cholesky Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([4.0, 2.0, 2.0, 2.0]),
            [2, 2],
            DType.float32,
          );
          final l = cholesky(a);
          expect(l.dtype, DType.float32);
          // L = [[2, 0], [1, 1]]
          expect(l.toList()[0], closeTo(2.0, 1e-5));
          expect(l.toList()[1], 0.0);
          expect(l.toList()[2], closeTo(1.0, 1e-5));
          expect(l.toList()[3], closeTo(1.0, 1e-5));
        }),
      );

      test(
        'Cholesky Complex128 matrix and check exact factor values',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(4.0, 0.0),
              Complex(2.0, 1.0),
              Complex(2.0, -1.0),
              Complex(3.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final l = cholesky(a);
          expect(l.dtype, DType.complex128);

          // Expected: L = [[2.0, 0.0], [1.0 - 0.5i, sqrt(1.75)]]
          expect(l.toList()[0].real, closeTo(2.0, 1e-9));
          expect(l.toList()[0].imag, closeTo(0.0, 1e-9));

          expect(l.toList()[1].real, closeTo(0.0, 1e-9));
          expect(l.toList()[1].imag, closeTo(0.0, 1e-9));

          expect(l.toList()[2].real, closeTo(1.0, 1e-9));
          expect(l.toList()[2].imag, closeTo(-0.5, 1e-9));

          expect(l.toList()[3].real, closeTo(math.sqrt(1.75), 1e-9));
          expect(l.toList()[3].imag, closeTo(0.0, 1e-9));
        }),
      );

      test(
        'Cholesky Complex64 matrix and check exact factor values',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(4.0, 0.0),
              Complex(2.0, 1.0),
              Complex(2.0, -1.0),
              Complex(3.0, 0.0),
            ],
            [2, 2],
            DType.complex64,
          );
          final l = cholesky(a);
          expect(l.dtype, DType.complex64);

          // Expected: L = [[2.0, 0.0], [1.0 - 0.5i, sqrt(1.75)]]
          expect(l.toList()[0].real, closeTo(2.0, 1e-5));
          expect(l.toList()[0].imag, closeTo(0.0, 1e-5));

          expect(l.toList()[1].real, closeTo(0.0, 1e-5));
          expect(l.toList()[1].imag, closeTo(0.0, 1e-5));

          expect(l.toList()[2].real, closeTo(1.0, 1e-5));
          expect(l.toList()[2].imag, closeTo(-0.5, 1e-5));

          expect(l.toList()[3].real, closeTo(math.sqrt(1.75), 1e-5));
          expect(l.toList()[3].imag, closeTo(0.0, 1e-5));
        }),
      );

      test(
        'Cholesky with pre-allocated contiguous out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              4.0,
              12.0,
              -16.0,
              12.0,
              37.0,
              -43.0,
              -16.0,
              -43.0,
              98.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final outL = NDArray.zeros([3, 3], DType.float64);
          final res = cholesky(a, out: outL);
          expect(identical(res, outL), true);

          // Check that the strictly upper triangular elements are zeroed out!
          expect(outL.toList()[1], 0.0);
          expect(outL.toList()[2], 0.0);
          expect(outL.toList()[5], 0.0);

          // Check exact expected lower triangular values
          expect(outL.toList()[0], closeTo(2.0, 1e-6));
          expect(outL.toList()[3], closeTo(6.0, 1e-6));
          expect(outL.toList()[4], closeTo(1.0, 1e-6));
          expect(outL.toList()[6], closeTo(-8.0, 1e-6));
          expect(outL.toList()[7], closeTo(5.0, 1e-6));
          expect(outL.toList()[8], closeTo(3.0, 1e-6));
        }),
      );

      test(
        'Batched 3D Cholesky decomposition',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([4.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0]),
            [2, 2, 2],
            DType.float64,
          );

          final l = cholesky(a);
          expect(l.shape, [2, 2, 2]);
          expect(l.toList()[0], closeTo(2.0, 1e-6));
          expect(l.toList()[1], 0.0);
          expect(l.toList()[2], closeTo(1.0, 1e-6));
          expect(l.toList()[3], closeTo(1.0, 1e-6));

          expect(l.toList()[4], closeTo(2.0, 1e-6));
          expect(l.toList()[5], 0.0);
          expect(l.toList()[6], closeTo(1.0, 1e-6));
          expect(l.toList()[7], closeTo(1.0, 1e-6));

          final out = NDArray.zeros([2, 2, 2], DType.float64);
          final res = cholesky(a, out: out);
          expect(identical(res, out), true);
          expect(out.toList()[1], 0.0);
          expect(out.toList()[5], 0.0);
        }),
      );
    });

    group('QR Decomposition tests', () {
      test(
        'QR orthogonal Q and upper triangular R verification',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final res = qr(a);
          final q = res.q;
          final r = res.r;

          expect(q.shape, [3, 3]);
          expect(r.shape, [3, 3]);

          // Verify R is upper triangular: elements under diagonal are exactly 0
          expect(r.toList()[3], 0.0);
          expect(r.toList()[6], 0.0);
          expect(r.toList()[7], 0.0);

          // Verify Q is orthogonal: Q * Q^T should equal Identity Matrix!
          // To perform matrix multiplication, we could loop or verify dot products.
          // Let's just do a custom check: row 0 dot row 0 should be 1.0, row 0 dot row 1 should be 0.0!
          double dotRow(int rA, int rB) {
            var s = 0.0;
            for (var i = 0; i < 3; i++) {
              s += q.toList()[rA * 3 + i] * q.toList()[rB * 3 + i];
            }
            return s;
          }

          expect(dotRow(0, 0), closeTo(1.0, 1e-9));
          expect(dotRow(1, 1), closeTo(1.0, 1e-9));
          expect(dotRow(2, 2), closeTo(1.0, 1e-9));
          expect(dotRow(0, 1), closeTo(0.0, 1e-9));
          expect(dotRow(0, 2), closeTo(0.0, 1e-9));
        }),
      );

      test(
        'QR Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float32,
          );
          final res = qr(a);
          expect(res.q.dtype, DType.float32);
          expect(res.r.dtype, DType.float32);
          expect(res.r.toList()[3], 0.0);
        }),
      );

      test(
        'QR non-contiguous view matrix',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            Float64List.fromList([
              12.0,
              6.0,
              -4.0,
              -51.0,
              167.0,
              24.0,
              4.0,
              -68.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );
          final view = parent.transpose();
          expect(view.isContiguous, false);
          final res = qr(view);
          expect(res.q.shape, [3, 3]);
          expect(res.r.shape, [3, 3]);
          expect(res.r.toList()[3], 0.0);
        }),
      );

      test(
        'QR with out parameter recycling buffers',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final qBuffer = NDArray.zeros([3, 3], DType.float64);
          final rBuffer = NDArray.zeros([3, 3], DType.float64);

          final res = qr(a, out: (q: qBuffer, r: rBuffer));

          expect(identical(res.q, qBuffer), true);
          expect(identical(res.r, rBuffer), true);

          // Verify correctness of QR values in the recycled buffers
          expect(res.r.toList()[3], 0.0);
          expect(res.r.toList()[6], 0.0);
          expect(res.r.toList()[7], 0.0);
        }),
      );

      test(
        'QRRecordDispose extension',
        () => NDArray.scope(() {
          // Check the extension method on records
          final a2 = NDArray.fromList(
            Float64List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );
          final res2 = qr(a2);
          expect(res2.q.isDisposed, false);
          expect(res2.r.isDisposed, false);

          res2.dispose();
          expect(res2.q.isDisposed, true);
          expect(res2.r.isDisposed, true);
        }),
      );

      test(
        'QR throws ArgumentError on incompatible shape of out parameter',
        () => NDArray.scope(() {
          final a = NDArray.zeros([3, 3], DType.float64);

          // Wrong shape for Q
          final qBufferWrongShape = NDArray.zeros([2, 3], DType.float64);
          final rBuffer = NDArray.zeros([3, 3], DType.float64);
          expect(
            () => qr(a, out: (q: qBufferWrongShape, r: rBuffer)),
            throwsArgumentError,
          );

          // Wrong shape for R
          final qBuffer = NDArray.zeros([3, 3], DType.float64);
          final rBufferWrongShape = NDArray.zeros([3, 2], DType.float64);
          expect(
            () => qr(a, out: (q: qBuffer, r: rBufferWrongShape)),
            throwsArgumentError,
          );
        }),
      );
    });

    group('SVD Decomposition tests', () {
      test(
        'SVD full matrices factorizations checks',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [3, 2],
            DType.float64,
          );

          final res = svd(a);
          final u = res.u;
          final s = res.s;
          final vh = res.vh;

          expect(u.shape, [3, 3]); // full matrix
          expect(s.shape, [2]); // min(3, 2) vector
          expect(vh.shape, [2, 2]); // full matrix

          // Singular values in S must be sorted in descending order by LAPACK rules!
          expect(s.toList()[0], greaterThan(s.toList()[1]));
          expect(s.toList()[0], greaterThan(0.0));
        }),
      );

      test(
        'SVD Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [3, 2],
            DType.float32,
          );
          final res = svd(a);
          expect(res.u.dtype, DType.float32);
          expect(res.s.dtype, DType.float32);
          expect(res.vh.dtype, DType.float32);
          expect(res.u.shape, [3, 3]);
        }),
      );

      test(
        'SVD non-contiguous view matrix',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            Float64List.fromList([1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
            [2, 3],
            DType.float64,
          );
          final view = parent.transpose();
          expect(view.isContiguous, false);
          final res = svd(view);
          expect(res.u.shape, [3, 3]);
          expect(res.s.shape, [2]);
        }),
      );
    });

    group('Determinant (det) tests', () {
      test(
        'Determinant of Float64 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          expect(det(a).scalar, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        'Determinant of Float32 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float32,
          );
          expect(det(a).scalar, closeTo(-2.0, 1e-5));
        }),
      );

      test(
        'det() throws ArgumentError on non-square matrix',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => det(a), throwsArgumentError);
        }),
      );
    });

    group('Linear Solver (solve) tests', () {
      test(
        'Solve Float64 system of equations',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float64);
          final x = solve(a, b);
          expect(x.shape, [2, 1]);
          expect(x.toList()[0], closeTo(2.0, 1e-9));
          expect(x.toList()[1], closeTo(3.0, 1e-9));
        }),
      );

      test(
        'Solve Float32 system of equations',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([3.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList(Float32List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float32);
          final x = solve(a, b);
          expect(x.shape, [2, 1]);
          expect(x.toList()[0], closeTo(2.0, 1e-5));
          expect(x.toList()[1], closeTo(3.0, 1e-5));
        }),
      );

      test(
        'Solve Complex128 system of equations',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(3.0, 0.0),
              Complex(1.0, 0.0),
              Complex(1.0, 0.0),
              Complex(2.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );

          final b = NDArray.fromList(
            [Complex(9.0, 0.0), Complex(8.0, 0.0)],
            [2, 1],
            DType.complex128,
          );

          final x = solve(a, b);
          expect(x.dtype, DType.complex128);
          expect(x.toList()[0], Complex(2.0, 0.0));
          expect(x.toList()[1], Complex(3.0, 0.0));
        }),
      );

      test(
        'Singular matrix solve throws SingularMatrixException',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float64);
          expect(() => solve(a, b), throwsA(isA<SingularMatrixException>()));
        }),
      );

      test(
        'Solve integer matrices throws ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([3, 1, 1, 2]), [
            2,
            2,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([9, 8]), [
            2,
            1,
          ], DType.int32);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );

      test(
        'Solve Float64 system of equations with out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float64);
          final outBuffer = NDArray.zeros([2, 1], DType.float64);
          final x = solve(a, b, out: outBuffer);
          expect(identical(x, outBuffer), true);
          expect(x.shape, [2, 1]);
          expect(x.toList()[0], closeTo(2.0, 1e-9));
          expect(x.toList()[1], closeTo(3.0, 1e-9));
        }),
      );

      test(
        'Solve Complex64 system of equations',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(3.0, 0.0),
              Complex(1.0, 0.0),
              Complex(1.0, 0.0),
              Complex(2.0, 0.0),
            ],
            [2, 2],
            DType.complex64,
          );

          final b = NDArray.fromList(
            [Complex(9.0, 0.0), Complex(8.0, 0.0)],
            [2, 1],
            DType.complex64,
          );

          final x = solve(a, b);
          expect(x.dtype, DType.complex64);
          expect(x.toList()[0].real, closeTo(2.0, 1e-5));
          expect(x.toList()[1].real, closeTo(3.0, 1e-5));
        }),
      );

      test(
        'Batched 3D solve with 1D vector RHS',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0]),
            [2, 2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
          ], DType.float64);

          final x = solve(a, b);
          expect(x.shape, [2, 2]);
          expect(x.toList()[0], closeTo(2.0, 1e-9));
          expect(x.toList()[1], closeTo(3.0, 1e-9));
          expect(x.toList()[2], closeTo(2.0, 1e-9));
          expect(x.toList()[3], closeTo(3.0, 1e-9));
        }),
      );

      test(
        'Batched 3D solve with 3D matrix RHS',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0]),
            [2, 2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            Float64List.fromList([9.0, 8.0, 9.0, 8.0]),
            [2, 2, 1],
            DType.float64,
          );

          final out = NDArray.zeros([2, 2, 1], DType.float64);
          final x = solve(a, b, out: out);
          expect(identical(x, out), true);
          expect(x.shape, [2, 2, 1]);
          expect(x.toList()[0], closeTo(2.0, 1e-9));
          expect(x.toList()[1], closeTo(3.0, 1e-9));
          expect(x.toList()[2], closeTo(2.0, 1e-9));
          expect(x.toList()[3], closeTo(3.0, 1e-9));
        }),
      );
    });

    group('Eigenvalues (eig) tests', () {
      test(
        'Eigenvalues/eigenvectors of Float64 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([0.0, 1.0, -2.0, -3.0]),
            [2, 2],
            DType.float64,
          );
          final (eigenvalues: w, eigenvectors: vr) = eig(a);

          expect(w.shape, [2]);
          expect(vr.shape, [2, 2]);

          expect(w.toList()[0].real, closeTo(-1.0, 1e-9));
          expect(w.toList()[0].imag, closeTo(0.0, 1e-9));
          expect(w.toList()[1].real, closeTo(-2.0, 1e-9));
          expect(w.toList()[1].imag, closeTo(0.0, 1e-9));
        }),
      );

      test(
        'Eigenvalues of Float32 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([0.0, 1.0, -2.0, -3.0]),
            [2, 2],
            DType.float32,
          );
          final (eigenvalues: w, eigenvectors: _) = eig(a);
          expect(w.dtype, DType.complex64);
          expect(w.toList()[0].real, closeTo(-1.0, 1e-5));
          expect(w.toList()[1].real, closeTo(-2.0, 1e-5));
        }),
      );

      test(
        'Eigenvalues/eigenvectors of Complex128 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(0.0, 0.0),
              Complex(1.0, 0.0),
              Complex(-2.0, 0.0),
              Complex(-3.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );

          final (eigenvalues: w, eigenvectors: _) = eig(a);
          expect(w.dtype, DType.complex128);
          expect(w.toList()[0].real, closeTo(-1.0, 1e-9));
          expect(w.toList()[1].real, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        'Eigenvalues/eigenvectors of Complex64 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(0.0, 0.0),
              Complex(1.0, 0.0),
              Complex(-2.0, 0.0),
              Complex(-3.0, 0.0),
            ],
            [2, 2],
            DType.complex64,
          );

          final (eigenvalues: w, eigenvectors: _) = eig(a);
          expect(w.dtype, DType.complex64);
          expect(w.toList()[0].real, closeTo(-1.0, 1e-5));
          expect(w.toList()[1].real, closeTo(-2.0, 1e-5));
        }),
      );

      test(
        'Eigenvalues/eigenvectors with out parameter recycler',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([0.0, 1.0, -2.0, -3.0]),
            [2, 2],
            DType.float64,
          );
          final outW = NDArray.zeros([2], DType.complex128);
          final outVR = NDArray.zeros([2, 2], DType.complex128);

          final (eigenvalues: w, eigenvectors: vr) = eig(
            a,
            out: (eigenvalues: outW, eigenvectors: outVR),
          );

          expect(identical(w, outW), true);
          expect(identical(vr, outVR), true);

          expect(w.toList()[0].real, closeTo(-1.0, 1e-9));
          expect(w.toList()[1].real, closeTo(-2.0, 1e-9));
        }),
      );
    });

    group('Linear Algebra Validation & Error Exceptions tests', () {
      test(
        'cholesky() throws ArgumentError on non-square matrix',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => cholesky(a), throwsArgumentError);
        }),
      );

      test(
        'cholesky() throws NonPositiveDefiniteException on non positive-definite matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([-1.0, 0.0, 0.0, -1.0]),
            [2, 2],
            DType.float64,
          );
          expect(
            () => cholesky(a),
            throwsA(isA<NonPositiveDefiniteException>()),
          );
        }),
      );

      test(
        'cholesky() throws ArgumentError on integer matrices (int32)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([4, 2, 2, 2]), [
            2,
            2,
          ], DType.int32);
          expect(() => cholesky(a), throwsArgumentError);
        }),
      );

      test(
        'qr() throws ArgumentError on non-2D tensor',
        () => NDArray.scope(() {
          final a = NDArray.zeros([3], DType.float64);
          expect(() => qr(a), throwsArgumentError);
        }),
      );

      test(
        'svd() throws ArgumentError on non-2D tensor',
        () => NDArray.scope(() {
          final a = NDArray.zeros([3], DType.float64);
          expect(() => svd(a), throwsArgumentError);
        }),
      );

      test(
        'eig() throws ArgumentError on non-square matrix',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => eig(a), throwsArgumentError);
        }),
      );

      test(
        'eig() throws UnimplementedError on boolean matrix',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 2], DType.boolean);
          expect(() => eig(a), throwsUnimplementedError);
        }),
      );

      test(
        'solve() throws ArgumentError on non-square matrix a',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          final b = NDArray.zeros([2], DType.float64);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );

      test(
        'solve() throws ArgumentError on mismatched b first dimension',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 2], DType.float64);
          final b = NDArray.zeros([3], DType.float64);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );

      test(
        'solve() throws ArgumentError on empty/scalar b array',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 2], DType.float64);
          final b = NDArray.zeros([], DType.float64);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );
    });

    group('Least-Squares Solver (lstsq) tests', () {
      test(
        'Solve over-determined system (m > n)',
        () => NDArray.scope(() {
          // y = c + m * x
          // Data points: (1, 2), (2, 3.9), (3, 6.1)
          // A = [[1, 1], [1, 2], [1, 3]]
          // B = [2.0, 3.9, 6.1]
          final a = NDArray.fromList(
            [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            [3, 2],
            DType.float64,
          );
          final b = NDArray.fromList([2.0, 3.9, 6.1], [3], DType.float64);

          final res = lstsq(a, b);

          expect(res.rank, 2);
          expect(res.s.shape, [2]);
          expect(res.residuals.shape, [1]);

          // Analytical solution: c = -0.1, m = 2.05
          expect(res.x.toList()[0], closeTo(-0.1, 1e-9));
          expect(res.x.toList()[1], closeTo(2.05, 1e-9));

          // Residual: sum of squared errors = 0.05^2 + (-0.1)^2 + 0.05^2 = 0.0025 + 0.01 + 0.0025 = 0.015
          expect(res.residuals.toList()[0], closeTo(0.015, 1e-9));
        }),
      );

      test(
        'Solve under-determined system (m < n) returns minimum norm solution',
        () => NDArray.scope(() {
          // A = [[1.0, 2.0, 3.0]]
          // B = [6.0]
          final a = NDArray.fromList([1.0, 2.0, 3.0], [1, 3], DType.float64);
          final b = NDArray.fromList([6.0], [1], DType.float64);

          final res = lstsq(a, b);

          expect(res.rank, 1);
          expect(res.residuals.shape, [0]); // no residuals computed

          // Minimum norm solution: x_i = 6 * a_i / sum(a_i^2) = 6 * a_i / 14
          expect(res.x.toList()[0], closeTo(6.0 / 14.0, 1e-9));
          expect(res.x.toList()[1], closeTo(12.0 / 14.0, 1e-9));
          expect(res.x.toList()[2], closeTo(18.0 / 14.0, 1e-9));
        }),
      );

      test(
        'Solve full rank square matrix matching solve()',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [3.0, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList([9.0, 8.0], [2], DType.float64);

          final res = lstsq(a, b);
          final xSolve = solve(a, b);

          expect(res.rank, 2);
          expect(res.residuals.shape, [
            0,
          ]); // square full rank -> empty residuals
          expect(res.x.toList()[0], closeTo(xSolve.toList()[0], 1e-9));
          expect(res.x.toList()[1], closeTo(xSolve.toList()[1], 1e-9));
        }),
      );

      test(
        'lstsq with float32 support',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList([5.0, 11.0], [2], DType.float32);

          final res = lstsq(a, b);
          expect(res.x.dtype, DType.float32);
          expect(res.x.toList()[0], closeTo(1.0, 1e-5));
          expect(res.x.toList()[1], closeTo(2.0, 1e-5));
        }),
      );

      test(
        'Complex128 least-squares solver',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(3.0, 0.0),
              Complex(1.0, 0.0),
              Complex(1.0, 0.0),
              Complex(2.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );

          final b = NDArray.fromList(
            [Complex(9.0, 0.0), Complex(8.0, 0.0)],
            [2, 1],
            DType.complex128,
          );

          final res = lstsq(a, b);
          expect(res.x.dtype, DType.complex128);
          expect(res.residuals.dtype, DType.float64);
          expect(res.x.toList()[0].real, closeTo(2.0, 1e-9));
          expect(res.x.toList()[0].imag, closeTo(0.0, 1e-9));
          expect(res.x.toList()[1].real, closeTo(3.0, 1e-9));
          expect(res.x.toList()[1].imag, closeTo(0.0, 1e-9));
        }),
      );

      test(
        'lstsq using out recycler buffer',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            [3, 2],
            DType.float64,
          );
          final b = NDArray.fromList([2.0, 3.9, 6.1], [3], DType.float64);
          final outBuffer = NDArray.zeros([2], DType.float64);

          final res = lstsq(a, b, out: outBuffer);

          expect(identical(res.x, outBuffer), true);
          expect(res.x.toList()[0], closeTo(-0.1, 1e-9));
          expect(res.x.toList()[1], closeTo(2.05, 1e-9));
        }),
      );
      test(
        'lstsq with integer matrices auto-upcasts to Float64',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 1, 1, 2, 1, 3], [3, 2], DType.int32);
          final b = NDArray.fromList([2, 4, 6], [3], DType.int32);
          final res = lstsq(a, b);
          expect(res.x.dtype, DType.float64);
          expect(res.x.toList()[0], closeTo(0.0, 1e-9));
          expect(res.x.toList()[1], closeTo(2.0, 1e-9));
        }),
      );
    });

    group('Optimal Matrix Chain Product (multi_dot) tests', () {
      test(
        'Multiply 3 matrices with optimal parenthesization',
        () => NDArray.scope(() {
          // shapes: A[2, 10] * B[10, 5] * C[5, 3] -> result [2, 3]
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5, 3], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, [2, 3]);
          // Each cell = sum_{k} (1 * sum_{m} (1 * 1)) = 10 * 5 = 50
          for (var i = 0; i < 6; i++) {
            expect(res.toList()[i], 50.0);
          }
        }),
      );

      test(
        'multi_dot with 1D squeezing at first array',
        () => NDArray.scope(() {
          // A[10] * B[10, 5] * C[5, 3] -> result [3]
          final a = NDArray.ones([10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5, 3], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, [3]);
          for (var i = 0; i < 3; i++) {
            expect(res.toList()[i], 50.0);
          }
        }),
      );

      test(
        'multi_dot with 1D squeezing at last array',
        () => NDArray.scope(() {
          // A[2, 10] * B[10, 5] * C[5] -> result [2]
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, [2]);
          for (var i = 0; i < 2; i++) {
            expect(res.toList()[i], 50.0);
          }
        }),
      );

      test(
        'multi_dot with squeezing at both first and last arrays',
        () => NDArray.scope(() {
          // A[10] * B[10, 5] * C[5] -> result [] (scalar)
          final a = NDArray.ones([10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, []);
          expect(res.scalar, 50.0);
        }),
      );

      test(
        'multi_dot using out recycler buffer',
        () => NDArray.scope(() {
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5, 3], DType.float64);

          final out = NDArray.zeros([2, 3], DType.float64);
          final res = multi_dot([a, b, c], out: out);

          expect(res == out, true);
          expect(out.shape, [2, 3]);
          for (var i = 0; i < 6; i++) {
            expect(out.toList()[i], 50.0);
          }
        }),
      );

      test(
        'Incompatible dimensions throw ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([9, 5], DType.float64); // 9 != 10
          final c = NDArray.ones([5, 3], DType.float64);

          expect(() => multi_dot([a, b, c]), throwsArgumentError);
        }),
      );
    });

    group('Stacked/Batch Linear Algebra Tests', () {
      test(
        'Stacked det() double and float32',
        () => NDArray.scope(() {
          // Double precision stacked matrix [2, 2, 2]
          final a64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );

          final d64 = det(a64);
          expect(d64.shape, [2]);
          expect(d64.dtype, DType.float64);
          expect(d64.toList()[0], closeTo(-2.0, 1e-9));
          expect(d64.toList()[1], closeTo(-2.0, 1e-9));

          // Float32 precision stacked matrix [2, 2, 2]
          final a32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float32,
          );

          final d32 = det(a32);
          expect(d32.shape, [2]);
          expect(d32.toList()[0], closeTo(-2.0, 1e-5));
          expect(d32.toList()[1], closeTo(-2.0, 1e-5));

          // High rank 4D stack det [2, 2, 2, 2]
          final a4d = NDArray.fromList(
            List<double>.generate(
              16,
              (i) => (i % 4) == 0 ? 2.0 : ((i % 4) == 3 ? 3.0 : 1.0),
            ),
            [2, 2, 2, 2],
            DType.float64,
          ); // Each matrix is [[2, 1], [1, 3]] -> det = 5.0
          final d4d = det(a4d);
          expect(d4d.shape, [2, 2]);
          for (var i = 0; i < 4; i++) {
            expect(d4d.toList()[i], closeTo(5.0, 1e-9));
          }
        }),
      );

      test(
        'Stacked eig()',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, -2.0, -3.0, 0.0, 1.0, -2.0, -3.0],
            [2, 2, 2],
            DType.float64,
          );

          final (eigenvalues: w, eigenvectors: vr) = eig(a);

          expect(w.shape, [2, 2]);
          expect(vr.shape, [2, 2, 2]);
          expect(w.dtype, DType.complex128);
          expect(vr.dtype, DType.complex128);

          // Slice 0
          expect(w.toList()[0].real, closeTo(-1.0, 1e-9));
          expect(w.toList()[1].real, closeTo(-2.0, 1e-9));
          // Slice 1
          expect(w.toList()[2].real, closeTo(-1.0, 1e-9));
          expect(w.toList()[3].real, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        'Stacked qr()',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ],
            [2, 3, 3],
            DType.float64,
          );

          final res = qr(a);
          final q = res.q;
          final r = res.r;

          expect(q.shape, [2, 3, 3]);
          expect(r.shape, [2, 3, 3]);

          // Verify upper triangular slice 0 and 1
          expect(r.toList()[3], 0.0); // r[0, 1, 0]
          expect(r.toList()[6], 0.0); // r[0, 2, 0]
          expect(r.toList()[7], 0.0); // r[0, 2, 1]

          expect(r.toList()[12], 0.0); // r[1, 1, 0]
          expect(r.toList()[15], 0.0); // r[1, 2, 0]
          expect(r.toList()[16], 0.0); // r[1, 2, 1]

          // Verify Q slice 0 orthogonality: dot product of row 0 and row 1 is 0
          double dotRow(NDArray mat, int slice, int rA, int rB) {
            var s = 0.0;
            final offset = slice * 9;
            for (var i = 0; i < 3; i++) {
              s +=
                  mat.toList()[offset + rA * 3 + i] *
                  mat.toList()[offset + rB * 3 + i];
            }
            return s;
          }

          expect(dotRow(q, 0, 0, 0), closeTo(1.0, 1e-9));
          expect(dotRow(q, 0, 0, 1), closeTo(0.0, 1e-9));
          expect(dotRow(q, 1, 0, 0), closeTo(1.0, 1e-9));
          expect(dotRow(q, 1, 0, 1), closeTo(0.0, 1e-9));
        }),
      );

      test(
        'Stacked svd() tall and wide',
        () => NDArray.scope(() {
          // 1. Tall matrix svd: [2, 3, 2]
          final aTall = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3, 2],
            DType.float64,
          );

          final resTall = svd(aTall);
          final uT = resTall.u;
          final sT = resTall.s;
          final vhT = resTall.vh;

          expect(uT.shape, [2, 3, 3]);
          expect(sT.shape, [2, 2]);
          expect(vhT.shape, [2, 2, 2]);
          expect(sT.toList()[0], greaterThan(sT.toList()[1]));
          expect(sT.toList()[2], greaterThan(sT.toList()[3]));

          // 2. Wide matrix svd: [2, 2, 3] (tests transposed wide matrix path under batch)
          final aWide = NDArray.fromList(
            [1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
            [2, 2, 3],
            DType.float64,
          );

          final resWide = svd(aWide);
          final uW = resWide.u;
          final sW = resWide.s;
          final vhW = resWide.vh;

          expect(uW.shape, [2, 2, 2]);
          expect(sW.shape, [2, 2]);
          expect(vhW.shape, [2, 3, 3]);
        }),
      );

      test(
        'Stacked operations with strided inputs (views)',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            [1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );

          // Take the transpose of the last two axes
          final axes = [0, 2, 1];
          final view = parent.transpose(
            axes,
          ); // Shape [2, 2, 2], non-contiguous
          expect(view.isContiguous, false);

          // Det of non-contiguous stacked view
          final d = det(view);
          expect(d.shape, [2]);
          // slice 0: [[1, 2], [3, 4]] -> det = -2
          // slice 1: [[5, 6], [7, 8]] -> det = -2
          expect(d.toList()[0], closeTo(-2.0, 1e-9));
          expect(d.toList()[1], closeTo(-2.0, 1e-9));
        }),
      );
    });

    group('Advanced Linear Algebra (kron, outer, cross, norm) tests', () {
      test(
        'Kronecker product (kron) tests',
        () => NDArray.scope(() {
          // 1D vector kron
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
          final res1 = kron(a, b);
          expect(res1.shape, [4]);
          expect(res1.dtype, DType.float64);
          expect(res1.toList(), [3.0, 4.0, 6.0, 8.0]);

          // 2D matrix kron
          final a2 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b2 = NDArray.fromList(
            [0.0, 5.0, 6.0, 7.0],
            [2, 2],
            DType.float64,
          );
          final res2 = kron(a2, b2);
          expect(res2.shape, [4, 4]);
          expect(res2.toList(), [
            0.0,
            5.0,
            0.0,
            10.0,
            6.0,
            7.0,
            12.0,
            14.0,
            0.0,
            15.0,
            0.0,
            20.0,
            18.0,
            21.0,
            24.0,
            28.0,
          ]);

          // Strided view inputs
          final viewA = a2.transpose();
          final resView = kron(viewA, b2);
          expect(resView.shape, [4, 4]);

          // Complex types
          final ac = NDArray.fromList(
            [Complex(1.0, 2.0)],
            [1],
            DType.complex128,
          );
          final bc = NDArray.fromList(
            [Complex(3.0, 4.0)],
            [1],
            DType.complex128,
          );
          final resc = kron(ac, bc);
          expect(resc.dtype, DType.complex128);
          expect(resc.toList()[0].real, -5.0);
          expect(resc.toList()[0].imag, 10.0);
        }),
      );

      test(
        'Vector Outer Product (outer) tests',
        () => NDArray.scope(() {
          // Basic outer product
          final u = NDArray.fromList([1, 2], [2], DType.int32);
          final v = NDArray.fromList([3, 4, 5], [3], DType.int32);
          final res = outer(u, v);
          expect(res.shape, [2, 3]);
          expect(res.toList(), [3, 4, 5, 6, 8, 10]);

          // Complex outer product
          final uc = NDArray.fromList(
            [Complex(1.0, 0.0)],
            [1],
            DType.complex128,
          );
          final vc = NDArray.fromList(
            [Complex(0.0, 2.0)],
            [1],
            DType.complex128,
          );
          final resc = outer(uc, vc);
          expect(resc.dtype, DType.complex128);
          expect(resc.toList()[0].real, 0.0);
          expect(resc.toList()[0].imag, 2.0);
        }),
      );

      test(
        'Vector Cross Product (cross) tests',
        () => NDArray.scope(() {
          // 3D cross product
          final v1 = NDArray.fromList([1.0, 0.0, 0.0], [3], DType.float64);
          final v2 = NDArray.fromList([0.0, 1.0, 0.0], [3], DType.float64);
          final res3d = cross(v1, v2);
          expect(res3d.shape, [3]);
          expect(res3d.toList(), [0.0, 0.0, 1.0]);

          // 2D cross product (returns scalar/z-component)
          final u1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final u2 = NDArray.fromList([3.0, 4.0], [2], DType.float64);
          final res2d = cross(u1, u2);
          expect(res2d.shape, []);
          expect(res2d.toList()[0], -2.0);

          // Stacked/multidimensional cross product
          final sa = NDArray.fromList(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [2, 3],
            DType.float64,
          );
          final sb = NDArray.fromList(
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [2, 3],
            DType.float64,
          );
          final resStack = cross(sa, sb);
          expect(resStack.shape, [2, 3]);
          expect(resStack.toList(), [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]);
        }),
      );

      test(
        'Vector/Matrix Norm (norm) tests',
        () => NDArray.scope(() {
          // 1D Vector Norms
          final x = NDArray.fromList(
            [1.0, -2.0, 3.0, -4.0],
            [4],
            DType.float64,
          );
          expect(norm(x, ord: 1).toList()[0], 10.0);
          expect(norm(x, ord: 2).toList()[0], closeTo(math.sqrt(30.0), 1e-9));
          expect(norm(x, ord: double.infinity).toList()[0], 4.0);
          expect(norm(x, ord: double.negativeInfinity).toList()[0], 1.0);

          // Matrix Norms
          final m = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(
            norm(m, ord: NormKind.frobenius).toList()[0],
            closeTo(math.sqrt(30.0), 1e-9),
          );
          expect(norm(m, ord: 1).toList()[0], 6.0); // max of col sums ([4, 6])
          expect(
            norm(m, ord: double.infinity).toList()[0],
            7.0,
          ); // max of row sums ([3, 7])

          // Singular value matrix norms (ord=2)
          final sNorm = norm(m, ord: 2);
          expect(sNorm.toList()[0], greaterThan(0.0));

          // keepdims support
          final kd = norm(x, ord: 1, keepdims: true);
          expect(kd.shape, [1]);
          expect(kd.toList()[0], 10.0);

          // Axis reductions
          final m3 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final normAxis = norm(m3, ord: 1, axis: 0);
          expect(normAxis.shape, [2]);
          expect(normAxis.toList()[0], 4.0); // 1 + 3
          expect(normAxis.toList()[1], 6.0); // 2 + 4
        }),
      );

      test(
        'Kronecker product (kron) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = <DType<AnySpec>>[
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final a = dtype == DType.boolean
                ? NDArray.fromList([true, false], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(1.0, 0.0), Complex(2.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([1.0, 2.0], [2], dtype)
                            : NDArray.fromList([1, 2], [2], dtype)));

            final b = dtype == DType.boolean
                ? NDArray.fromList([true, true], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(3.0, 0.0), Complex(4.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([3.0, 4.0], [2], dtype)
                            : NDArray.fromList([3, 4], [2], dtype)));

            final res = kron(a, b);
            expect(res.shape, [4]);
            expect(res.dtype, dtype);

            if (dtype == DType.boolean) {
              expect(res.toList(), [true, true, false, false]);
            } else if (dtype.isComplex) {
              expect(res.toList()[0].real, 3.0);
              expect(res.toList()[1].real, 4.0);
              expect(res.toList()[2].real, 6.0);
              expect(res.toList()[3].real, 8.0);
            } else {
              expect(
                res.toList(),
                [
                  3.0,
                  4.0,
                  6.0,
                  8.0,
                ].map((x) => dtype.isFloating ? x : x.toInt()).toList(),
              );
            }
          }
        }),
      );

      test(
        'Vector Outer Product (outer) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = <DType<AnySpec>>[
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final a = dtype == DType.boolean
                ? NDArray.fromList([true, false], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(2.0, 0.0), Complex(3.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([2.0, 3.0], [2], dtype)
                            : NDArray.fromList([2, 3], [2], dtype)));

            final b = dtype == DType.boolean
                ? NDArray.fromList([true, true], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(4.0, 0.0), Complex(5.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([4.0, 5.0], [2], dtype)
                            : NDArray.fromList([4, 5], [2], dtype)));

            final res = outer(a, b);
            expect(res.shape, [2, 2]);
            expect(res.dtype, dtype);

            if (dtype == DType.boolean) {
              expect(res.toList(), [true, true, false, false]);
            } else if (dtype.isComplex) {
              expect(res.toList()[0].real, 8.0);
              expect(res.toList()[1].real, 10.0);
              expect(res.toList()[2].real, 12.0);
              expect(res.toList()[3].real, 15.0);
            } else {
              expect(
                res.toList(),
                [
                  8.0,
                  10.0,
                  12.0,
                  15.0,
                ].map((x) => dtype.isFloating ? x : x.toInt()).toList(),
              );
            }
          }
        }),
      );

      test(
        'Vector Cross Product (cross) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = <DType<AnySpec>>[
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final a = dtype == DType.boolean
                ? NDArray.fromList([true, false, false], [3], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [
                            Complex(1.0, 0.0),
                            Complex(0.0, 0.0),
                            Complex(0.0, 0.0),
                          ],
                          [3],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([1.0, 0.0, 0.0], [3], dtype)
                            : NDArray.fromList([1, 0, 0], [3], dtype)));

            final b = dtype == DType.boolean
                ? NDArray.fromList([false, true, false], [3], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [
                            Complex(0.0, 0.0),
                            Complex(1.0, 0.0),
                            Complex(0.0, 0.0),
                          ],
                          [3],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([0.0, 1.0, 0.0], [3], dtype)
                            : NDArray.fromList([0, 1, 0], [3], dtype)));

            final res = cross(a, b);
            expect(res.shape, [3]);
            expect(res.dtype, dtype);

            if (dtype == DType.boolean) {
              expect(res.toList(), [false, false, true]);
            } else if (dtype.isComplex) {
              expect(res.toList()[0].real, 0.0);
              expect(res.toList()[1].real, 0.0);
              expect(res.toList()[2].real, 1.0);
            } else {
              expect(
                res.toList(),
                [
                  0.0,
                  0.0,
                  1.0,
                ].map((x) => dtype.isFloating ? x : x.toInt()).toList(),
              );
            }
          }
        }),
      );

      test(
        'Vector Norm (norm) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = [
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final x = dtype == DType.boolean
                ? NDArray.fromList([true, false, true], [3], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [
                            Complex(1.0, 0.0),
                            Complex(-2.0, 0.0),
                            Complex(3.0, 0.0),
                          ],
                          [3],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([1.0, -2.0, 3.0], [3], dtype)
                            : (dtype == DType.uint8
                                  ? NDArray.fromList([1, 2, 3], [3], dtype)
                                  : NDArray.fromList([1, -2, 3], [3], dtype))));

            final res = norm((x as NDArray<AnySpec>), ord: 1);
            expect(res.shape, []);
            final expectedDType =
                (dtype == DType.float32 || dtype == DType.complex64)
                ? DType.float32
                : DType.float64;
            expect(res.dtype, expectedDType);

            if (dtype == DType.boolean) {
              expect(res.toList()[0], 2.0);
            } else {
              expect(res.toList()[0], 6.0);
            }
          }
        }),
      );
    });
  });

  group('Determinant Tests', () {
    test(
      '2D Complex128 Matrix Determinant',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, -1.0),
            Complex(3.0, 0.0),
            Complex(4.0, 2.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final d = det<DTypeTag>(a);

        expect(d.shape, []);
        expect(d.scalar.real, closeTo(-4.0, 1e-9));
        expect(d.scalar.imag, closeTo(9.0, 1e-9));
      }),
    );

    test(
      '3D Stacked Complex64 Matrix Determinant',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            // Matrix 0
            Complex(1.0, 1.0), Complex(2.0, -1.0),
            Complex(3.0, 0.0), Complex(4.0, 2.0),
            // Matrix 1
            Complex(2.0, 0.0), Complex(0.0, 1.0),
            Complex(0.0, -1.0), Complex(3.0, 0.0),
          ],
          [2, 2, 2],
          DType.complex64,
        );

        final d = det<DTypeTag>(a);

        expect(d.shape, [2]);
        final dList = d.toList();
        expect(dList[0].real, closeTo(-4.0, 1e-4));
        expect(dList[0].imag, closeTo(9.0, 1e-4));
        expect(dList[1].real, closeTo(5.0, 1e-4));
        expect(dList[1].imag, closeTo(0.0, 1e-4));
      }),
    );

    test(
      '3D Stack Matrix Determinant',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([
            1.0, 2.0, 3.0, 4.0, // Matrix 0: det = -2
            5.0, 6.0, 7.0, 8.0, // Matrix 1: det = -2
          ]),
          [2, 2, 2],
          DType.float64,
        );

        final d = det(a);

        expect(d.shape, [2]);
        final dList = d.toList();
        expect(dList[0], closeTo(-2.0, 1e-9));
        expect(dList[1], closeTo(-2.0, 1e-9));
      }),
    );
  });

  group('Complex Matrix Multiplication Tests', () {
    test(
      'Complex128 Matrix Multiplication',
      () => NDArray.scope(() {
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

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final result = matmul(a, b);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);

        final Complex c00 = result.toList()[0];
        expect(c00.real, closeTo(-28.0, 1e-9));
        expect(c00.imag, closeTo(122.0, 1e-9));
      }),
    );

    test(
      'Complex64 Matrix Multiplication',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex64,
        );

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
          ],
          [2, 2],
          DType.complex64,
        );

        final result = matmul(a, b);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex64);

        final Complex c00 = result.toList()[0];
        expect(c00.real, closeTo(-28.0, 1e-5));
        expect(c00.imag, closeTo(122.0, 1e-5));
      }),
    );

    test(
      'Complex128 Transposed Matrix Multiplication',
      () => NDArray.scope(() {
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

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final result = matmul(a.transpose(), b);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);

        final Complex c00 = result.toList()[0];
        expect(c00.real, closeTo(-30.0, 1e-9));
        expect(c00.imag, closeTo(176.0, 1e-9));
      }),
    );

    test(
      'Complex128 Sliced Matrix Multiplication',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
          ],
          [2, 3],
          DType.complex128,
        );

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
            Complex(17.0, 18.0),
            Complex(19.0, 20.0),
          ],
          [3, 2],
          DType.complex128,
        );

        final aSlice = a.slice([const Slice(), const Slice(start: 0, stop: 2)]);
        final bSlice = b.slice([const Slice(start: 0, stop: 2), const Slice()]);

        final result = matmul(aSlice, bSlice);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);

        final Complex c00 = result.toList()[0];
        expect(c00.real, closeTo(-28.0, 1e-9));
        expect(c00.imag, closeTo(122.0, 1e-9));
      }),
    );
  });

  group('Matmul Stack Broadcasting Tests', () {
    group('Standard 3D Batch Matmul tests', () {
      test(
        'Verify uniform batch matrix stack multiply [2, 2, 2] x [2, 2, 2] -> [2, 2, 2]',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              1.0, 2.0, 3.0, 4.0, // batch 0
              5.0, 6.0, 7.0, 8.0, // batch 1
            ]),
            [2, 2, 2],
            DType.float64,
          );

          final b = NDArray.fromList(
            Float64List.fromList([
              1.0, 0.0, 0.0, 1.0, // batch 0 identity
              2.0, 2.0, 2.0, 2.0, // batch 1 twos
            ]),
            [2, 2, 2],
            DType.float64,
          );

          final res = matmul(a, b);

          expect(res.shape, [2, 2, 2]);
          expect(res.dtype, DType.float64);

          expect(res.toList().sublist(0, 4), [1.0, 2.0, 3.0, 4.0]);
          expect(res.toList().sublist(4, 8), [22.0, 22.0, 30.0, 30.0]);
        }),
      );
    });

    group('Asymmetric Stack Shape Broadcasting tests', () {
      test(
        'Verify broadcast mapping stretching [2, 1, 2, 2] x [3, 2, 2] -> [2, 3, 2, 2]',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              1.0, 0.0, 0.0, 1.0, // block 0 (identity)
              2.0, 0.0, 0.0, 2.0, // block 1 (scaled twos identity)
            ]),
            [2, 1, 2, 2],
            DType.float64,
          );

          final b = NDArray.fromList(
            Float64List.fromList([
              1.0, 2.0, 3.0, 4.0, // sub-block 0
              5.0, 6.0, 7.0, 8.0, // sub-block 1
              9.0, 10.0, 11.0, 12.0, // sub-block 2
            ]),
            [3, 2, 2],
            DType.float64,
          );

          final res = matmul(a, b);

          expect(res.shape, [2, 3, 2, 2]);

          final resList = res.toList();
          expect(resList.sublist(0, 4), [1.0, 2.0, 3.0, 4.0]); // b[0]
          expect(resList.sublist(4, 8), [5.0, 6.0, 7.0, 8.0]); // b[1]
          expect(resList.sublist(8, 12), [9.0, 10.0, 11.0, 12.0]); // b[2]

          expect(resList.sublist(12, 16), [2.0, 4.0, 6.0, 8.0]);
          expect(resList.sublist(16, 20), [10.0, 12.0, 14.0, 16.0]);
          expect(resList.sublist(20, 24), [18.0, 20.0, 22.0, 24.0]);
        }),
      );
    });

    group('1D Vector Promotions & Dot Products tests', () {
      test(
        'Verify pure 1D Vector Dot Product [3] x [3] -> [] (0D Scalar)',
        () => NDArray.scope(() {
          final v1 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final v2 = NDArray.fromList(Float64List.fromList([4.0, 5.0, 6.0]), [
            3,
          ], DType.float64);

          final dotRes = matmul(v1, v2);
          expect(dotRes.shape, <int>[]); // 0D scalar array
          expect(dotRes.scalar, 32.0);
        }),
      );

      test(
        'Verify Matrix-Vector multiplication [2, 3] x [3] -> [2]',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            DType.float64,
          );
          final vec = NDArray.fromList(Float64List.fromList([1.0, 1.0, 1.0]), [
            3,
          ], DType.float64);

          final res = matmul(mat, vec);
          expect(res.shape, [2]);
          expect(res.toList(), [6.0, 15.0]);
        }),
      );

      test(
        'Verify Vector-Matrix multiplication [3] x [3, 2] -> [2]',
        () => NDArray.scope(() {
          final vec = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            [3, 2],
            DType.float64,
          );

          final res = matmul(vec, mat);
          expect(res.shape, [2]);
          expect(res.toList(), [4.0, 5.0]);
        }),
      );

      test(
        'matmul() throws ArgumentError on incompatible 1D vector dot dimensions',
        () => NDArray.scope(() {
          final v1 = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
            2,
          ], DType.float64);
          final v2 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          expect(() => matmul(v1, v2), throwsArgumentError);
        }),
      );

      test(
        'matmul() throws ArgumentError on incompatible inner dimensions',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          final b = NDArray.zeros([2, 2], DType.float64);
          expect(() => matmul(a, b), throwsArgumentError);
        }),
      );
    });

    group('Float32 Single-Precision matmul tests', () {
      test(
        'Verify Float32 1D Vector Dot Product sdot',
        () => NDArray.scope(() {
          final v1 = NDArray.fromList(Float32List.fromList([1.0, 2.0]), [
            2,
          ], DType.float32);
          final v2 = NDArray.fromList(Float32List.fromList([3.0, 4.0]), [
            2,
          ], DType.float32);

          final res = matmul(v1, v2);
          expect(res.shape, []);
          expect(res.dtype, DType.float32);
          expect(res.scalar, closeTo(11.0, 1e-5));
        }),
      );

      test(
        'Verify Float32 2D Matrix Multiply sgemm',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList(
            Float32List.fromList([5.0, 6.0, 7.0, 8.0]),
            [2, 2],
            DType.float32,
          );

          final res = matmul(a, b);
          expect(res.shape, [2, 2]);
          expect(res.dtype, DType.float32);
          expect(res.toList(), [19.0, 22.0, 43.0, 50.0]);
        }),
      );
    });
  });

  group('Matmul Bug Repro', () {
    test('matmul on integer matrices', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final b = NDArray.fromList([5, 6, 7, 8], [2, 2], DType.int32);
        final result = matmul(a, b);
        expect(result.dtype, DType.int32);
        expect(result.shape, [2, 2]);
        expect(result.toList(), [19, 22, 43, 50]);
      });
    });
    test(
      'matmul() high-speed 1D vector dot product gate',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

        final c = matmul(a, b);

        expect(c.shape, []); // Must be a 0D scalar array
        expect(c.dtype, DType.float64);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        expect(c.scalar, 32.0);
      }),
    );

    test(
      'Cholesky decomposition on non-contiguous strided view correctness',
      () => NDArray.scope(() {
        // Symmetric positive-definite parent matrix
        final parentNonContig = NDArray.fromList(
          [4.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 37.0],
          [3, 3],
          DType.float64,
        );
        final viewNonContig = parentNonContig.slice([
          Slice(start: 0, stop: 3, step: 2),
          Slice(start: 0, stop: 3, step: 2),
        ]);
        expect(viewNonContig.isContiguous, false);
        expect(viewNonContig.toList(), [4.0, 12.0, 12.0, 37.0]);

        final l = cholesky(viewNonContig);
        expect(l.shape, [2, 2]);
        expect(l.toList(), [2.0, 0.0, 6.0, 1.0]);
      }),
    );

    test(
      'Matrix Inversion on non-contiguous transposed view with out recycler buffer',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [4.0, 7.0, 2.0, 6.0],
          [2, 2],
          DType.float64,
        );
        final viewT = parent.transposed; // non-contiguous!
        expect(viewT.isContiguous, false);

        final out = NDArray.create([2, 2], DType.float64);
        final result = inv(viewT, out: out);

        expect(result == out, true);
        final expected = [0.6, -0.2, -0.7, 0.4];
        final actual = result.toList();
        for (var i = 0; i < 4; i++) {
          expect(actual[i], closeTo(expected[i], 1e-9));
        }
      }),
    );

    test(
      'inv() in-place out buffer validations and solvers coverage',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final out = NDArray.zeros([2, 2], DType.float64);
        final incompatibleOut = NDArray.ones([3, 3], DType.float64);

        // 1. Incompatible out shape throws ArgumentError
        expect(() => inv(a, out: incompatibleOut), throwsArgumentError);

        // 2. Valid in-place solving
        final res = inv(a, out: out);
        expect(identical(res, out), true);
        final resList = res.toList();
        expect(resList[0], closeTo(-2.0, 1e-9));
        expect(resList[1], closeTo(1.0, 1e-9));
        expect(resList[2], closeTo(1.5, 1e-9));
        expect(resList[3], closeTo(-0.5, 1e-9));
      }),
    );

    test(
      'matmul() copy-free 100% transposed and sliced views multi-dimensional multiplication',
      () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);

        final aT = a.transposed;
        final bT = b.transposed;

        final result = matmul(aT, bT);

        expect(result.shape, [2, 2]);
        expect(result.toList(), [23.0, 31.0, 34.0, 46.0]);
      },
    );

    test(
      'diag() diagonal matrix ufunc correctness and zero-copy view validations',
      () {
        // 1. Main diagonal extraction (k = 0)
        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final mainDiag = diag(mat);

        expect(mainDiag.shape, [3]);
        expect(mainDiag.isContiguous, false); // spaced view
        expect(mainDiag.toList(), [1.0, 5.0, 9.0]);

        // 2. Strided views offset extractions (k = 1 and k = -1)
        final upperDiag = diag(mat, k: 1);
        final lowerDiag = diag(mat, k: -1);

        expect(upperDiag.shape, [2]);
        expect(upperDiag.toList(), [2.0, 6.0]);
        expect(lowerDiag.shape, [2]);
        expect(lowerDiag.toList(), [4.0, 8.0]);

        // 3. Construct diagonal 2D matrix from 1D vector
        final vec = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final dMat = diag(vec);
        final uDiagMat = diag(vec, k: 1);

        expect(dMat.shape, [2, 2]);
        expect(dMat.toList(), [10.0, 0.0, 0.0, 20.0]);

        expect(uDiagMat.shape, [3, 3]);
        expect(uDiagMat.toList(), [
          0.0,
          10.0,
          0.0,
          0.0,
          0.0,
          20.0,
          0.0,
          0.0,
          0.0,
        ]);

        // 4. Rank out of bounds throws ArgumentError
        final tensor3d = NDArray.zeros([2, 2, 2], DType.float64);
        expect(() => diag(tensor3d), throwsArgumentError);

        // 5. In-place recycler out reuse
        final outRecycler = NDArray.zeros([3, 3], DType.float64);
        final dMatRec = diag(vec, k: 1, out: outRecycler);
        expect(identical(dMatRec, outRecycler), true);
        expect(dMatRec.toList(), [
          0.0,
          10.0,
          0.0,
          0.0,
          0.0,
          20.0,
          0.0,
          0.0,
          0.0,
        ]);
      },
    );

    test(
      'Linear Algebra solvers det() and solve() singular and preconditions exceptions',
      () {
        // 1. det() singular matrix returns 0.0
        final singularMat = NDArray.fromList(
          [
            1.0, 2.0,
            2.0, 4.0, // linearly dependent rows!
          ],
          [2, 2],
          DType.float64,
        );
        expect(det(singularMat).scalar, 0.0);

        // 2. solve() non-square matrix throws ArgumentError
        final nonSquareA = NDArray.fromList(
          [1.0, 2.0, 3.0],
          [1, 3],
          DType.float64,
        );
        final b = NDArray.fromList([1.0], [1], DType.float64);
        expect(() => solve(nonSquareA, b), throwsArgumentError);

        // 3. solve() incompatible RHS shape throws ArgumentError
        final squareA = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final incompatibleB = NDArray.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        expect(() => solve(squareA, incompatibleB), throwsArgumentError);

        // 4. solve() singular Float64 matrix throws singular SingularMatrixException
        final singularFloat64A = NDArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final validB = NDArray.fromList([5.0, 6.0], [2], DType.float64);
        expect(
          () => solve(singularFloat64A, validB),
          throwsA(isA<SingularMatrixException>()),
        );

        // 5. solve() singular Float32 matrix throws singular SingularMatrixException
        final singularFloat32A = NDArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final validFloat32B = NDArray.fromList([5.0, 6.0], [2], DType.float32);
        expect(
          () => solve(singularFloat32A, validFloat32B),
          throwsA(isA<SingularMatrixException>()),
        );
      },
    );

    test(
      'solve() optimized contiguous block copy and non-contiguous view solvers correctness',
      () {
        // 1. Contiguous Float64 matrix solve
        final a = NDArray.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
        final b = NDArray.fromList([9.0, 8.0], [2], DType.float64);

        final x = solve(a, b);
        expect(x.toList(), [2.0, 3.0]);

        // 2. Non-contiguous transposed Float64 matrix solve
        final aParent = NDArray.fromList(
          [3.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );
        final aTransposed = aParent.transposed; // non-contiguous!
        // aTransposed is: [[3.0, 1.0], [1.0, 2.0]] which is symmetric, so solve is same
        final bParent = NDArray.fromList([9.0, 8.0], [2], DType.float64);

        final x2 = solve(aTransposed, bParent);
        expect(x2.toList(), [2.0, 3.0]);
      },
    );

    test(
      'linalg.tril() and linalg.triu() matrix extractions correctness',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );

        // 1. tril standard
        final l = tril(a);
        expect(l.toList(), [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);

        // 2. triu standard
        final u = triu(a);
        expect(u.toList(), [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);

        // 3. offsets (k=1, k=-1)
        final lK1 = tril(a, k: 1);
        expect(lK1.toList(), [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        final uKM1 = triu(a, k: -1);
        expect(uKM1.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]);

        // 4. in-place recycler out reuse
        final outBuffer = NDArray.zeros([3, 3], DType.float64);
        final lOut = tril(a, k: 0, out: outBuffer);
        expect(identical(lOut, outBuffer), true);
        expect(lOut.toList(), [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
      }),
    );

    test(
      'linalg.pinv() and linalg.matrix_power() correctness',
      () => NDArray.scope(() {
        // 1. pinv
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final aPlus = pinv(a);
        expect(aPlus.shape, [3, 2]);

        // Verify A * A+ * A = A (approximately)
        final temp = matmul(a, aPlus);
        final rec = matmul(temp, a);
        expect(allClose(rec, a, atol: 1e-7), true);

        temp.dispose();
        rec.dispose();
        aPlus.dispose();

        // 2. matrix_power
        final t = NDArray.fromList([0.8, 0.2, 0.1, 0.9], [2, 2], DType.float64);

        final t2 = matrix_power(t, 2);
        final tTimesT = matmul(t, t);
        expect(allClose(t2, tTimesT, atol: 1e-9), true);

        final t0 = matrix_power(t, 0);
        expect(t0.toList(), [1.0, 0.0, 0.0, 1.0]);

        final tInv = matrix_power(t, -1);
        final check = matmul(t, tInv);
        expect(allClose(check, t0, atol: 1e-9), true);

        t2.dispose();
        tTimesT.dispose();
        t0.dispose();
        tInv.dispose();
        check.dispose();
      }),
    );
  });

  group('Rank-deficient rectangular matrices', () {
    test('lstsq reports rank 1 and pinv satisfies A * A^+ * A == A', () {
      NDArray.scope(() {
        // Rank-1 3x2 matrix.
        final a = NDArray.fromList(
          [1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final ls = lstsq<Float64, Float64, Float64>(
          a,
          NDArray.zeros([3], DType.float64),
        );
        expect(ls.rank, equals(1));
        final aPlus = pinv<Float64>(a);
        // Moore-Penrose identity: A * A^+ * A == A.
        final recon = matmul(matmul(a, aPlus), a);
        expect(allClose(recon, a, atol: 1e-11), isTrue);
      });
    });
  });
}
