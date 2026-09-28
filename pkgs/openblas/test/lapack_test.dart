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
import 'package:openblas/src/openblas_bindings.dart';
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
}
