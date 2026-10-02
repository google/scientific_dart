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

import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:openblas/openblas.dart';
import 'package:test/test.dart';

const int lapackRowMajor = 101;

void main() {
  test('LAPACKE_dgetrf and LAPACKE_dgetri', () {
    final a = calloc<Double>(4);
    final ipiv = calloc<lapack_int>(2);

    // Matrix A = [[1.0, 2.0], [3.0, 4.0]] in row major
    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 4.0;

    // LU factorization
    final info1 = LAPACKE_dgetrf(lapackRowMajor, 2, 2, a, 2, ipiv);
    expect(info1, 0);

    // Matrix inversion
    final info2 = LAPACKE_dgetri(lapackRowMajor, 2, a, 2, ipiv);
    expect(info2, 0);

    // Expected inverse: [[-2.0, 1.0], [1.5, -0.5]]
    expect(a[0], closeTo(-2.0, 1e-5));
    expect(a[1], closeTo(1.0, 1e-5));
    expect(a[2], closeTo(1.5, 1e-5));
    expect(a[3], closeTo(-0.5, 1e-5));

    calloc.free(a);
    calloc.free(ipiv);
  });

  group('LAPACKE solvers, factorizations, and info return codes', () {
    test('LAPACKE_dgesv linear solve and dgetrf/dgetri LU inversion', () {
      final a = calloc<Double>(4);
      final b = calloc<Double>(2);
      final ipiv = calloc<lapack_int>(2);
      try {
        // Solve [[3, 1], [1, 2]] * x = [9, 8] -> x = [2, 3]
        a[0] = 3.0;
        a[1] = 1.0;
        a[2] = 1.0;
        a[3] = 2.0;
        b[0] = 9.0;
        b[1] = 8.0;
        final info = LAPACKE_dgesv(lapackRowMajor, 2, 1, a, 2, ipiv, b, 1);
        expect(info, equals(0));
        expect(b[0], closeTo(2.0, 1e-12));
        expect(b[1], closeTo(3.0, 1e-12));

        // Reset A and invert via dgetrf + dgetri: inv([[3, 1], [1, 2]]) = 1/5 * [[2, -1], [-1, 3]]
        a[0] = 3.0;
        a[1] = 1.0;
        a[2] = 1.0;
        a[3] = 2.0;
        expect(LAPACKE_dgetrf(lapackRowMajor, 2, 2, a, 2, ipiv), equals(0));
        expect(LAPACKE_dgetri(lapackRowMajor, 2, a, 2, ipiv), equals(0));
        expect(a[0], closeTo(0.4, 1e-12));
        expect(a[1], closeTo(-0.2, 1e-12));
        expect(a[2], closeTo(-0.2, 1e-12));
        expect(a[3], closeTo(0.6, 1e-12));
      } finally {
        calloc.free(a);
        calloc.free(b);
        calloc.free(ipiv);
      }
    });

    test(
      'LAPACKE_dpotrf Cholesky, LAPACKE_dsyevd eigenvalues, and getrf pointers',
      () {
        // Cholesky of [[4, 2], [2, 5]] -> L = [[2, 0], [1, 2]]
        final a = calloc<Double>(4);
        final w = calloc<Double>(2);
        try {
          a[0] = 4.0;
          a[1] = 2.0;
          a[2] = 2.0;
          a[3] = 5.0;
          final infoChol = LAPACKE_dpotrf(
            lapackRowMajor,
            76 /* 'L' */,
            2,
            a,
            2,
          );
          expect(infoChol, equals(0));
          expect(a[0], closeTo(2.0, 1e-12));
          expect(a[2], closeTo(1.0, 1e-12));
          expect(a[3], closeTo(2.0, 1e-12));

          // Symmetric eigenvalues of [[2, 1], [1, 2]] -> [1.0, 3.0]
          a[0] = 2.0;
          a[1] = 1.0;
          a[2] = 1.0;
          a[3] = 2.0;
          final infoEig = LAPACKE_dsyevd(
            lapackRowMajor,
            86 /* 'V' */,
            85 /* 'U' */,
            2,
            a,
            2,
            w,
          );
          expect(infoEig, equals(0));
          expect(w[0], closeTo(1.0, 1e-12));
          expect(w[1], closeTo(3.0, 1e-12));

          // Extension function pointers are non-null
          expect(get_dgetrf_ptr().address, isNot(0));
          expect(get_sgetrf_ptr().address, isNot(0));
          expect(get_zgetrf_ptr().address, isNot(0));
          expect(get_cgetrf_ptr().address, isNot(0));
        } finally {
          calloc.free(a);
          calloc.free(w);
        }
      },
    );

    test('Singular matrix in LAPACKE_dgetrf returns info > 0', () {
      final singular = calloc<Double>(4);
      final ipiv = calloc<lapack_int>(2);
      try {
        // Row 1 is 2 * Row 0 -> U[1,1] == 0, so info == 2
        singular[0] = 1.0;
        singular[1] = 2.0;
        singular[2] = 2.0;
        singular[3] = 4.0;
        final info = LAPACKE_dgetrf(lapackRowMajor, 2, 2, singular, 2, ipiv);
        expect(info, greaterThan(0));
      } finally {
        calloc.free(singular);
        calloc.free(ipiv);
      }
    });

    test('Non-positive-definite matrix in LAPACKE_dpotrf returns info > 0', () {
      final nonSpd = calloc<Double>(4);
      try {
        nonSpd[0] = 1.0;
        nonSpd[1] = 5.0;
        nonSpd[2] = 5.0;
        nonSpd[3] = 1.0;
        final info = LAPACKE_dpotrf(lapackRowMajor, 76 /* 'L' */, 2, nonSpd, 2);
        expect(info, greaterThan(0));
      } finally {
        calloc.free(nonSpd);
      }
    });
  });
}
