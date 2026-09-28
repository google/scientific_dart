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

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Eigenvalues (eigvals) tests', () {
    test(
      'Eigenvalues of Float64 2D square matrix',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([0.0, 1.0, -2.0, -3.0]),
          [2, 2],
          DType.float64,
        );
        final w = eigvals(a);
        final (eigenvalues: expectedW, eigenvectors: _) = eig(a);

        expect(w.shape, [2]);
        expect(w.dtype, DType.complex128);

        // Hardcoded expectations
        expect(w.toList()[0].real, closeTo(-1.0, 1e-9));
        expect(w.toList()[0].imag, closeTo(0.0, 1e-9));
        expect(w.toList()[1].real, closeTo(-2.0, 1e-9));
        expect(w.toList()[1].imag, closeTo(0.0, 1e-9));

        // Compare with eig
        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-9));
        expect(w.toList()[0].imag, closeTo(expectedW.toList()[0].imag, 1e-9));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-9));
        expect(w.toList()[1].imag, closeTo(expectedW.toList()[1].imag, 1e-9));
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
        final w = eigvals(a);
        final (eigenvalues: expectedW, eigenvectors: _) = eig(a);

        expect(w.dtype, DType.complex64);

        // Hardcoded expectations
        expect(w.toList()[0].real, closeTo(-1.0, 1e-5));
        expect(w.toList()[1].real, closeTo(-2.0, 1e-5));

        // Compare with eig
        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-5));
        expect(w.toList()[0].imag, closeTo(expectedW.toList()[0].imag, 1e-5));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-5));
        expect(w.toList()[1].imag, closeTo(expectedW.toList()[1].imag, 1e-5));
      }),
    );

    test(
      'Eigenvalues of Complex128 matrix',
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

        final w = eigvals(a);
        final (eigenvalues: expectedW, eigenvectors: _) = eig(a);

        expect(w.dtype, DType.complex128);

        // Hardcoded expectations
        expect(w.toList()[0].real, closeTo(-1.0, 1e-9));
        expect(w.toList()[1].real, closeTo(-2.0, 1e-9));

        // Compare with eig
        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-9));
        expect(w.toList()[0].imag, closeTo(expectedW.toList()[0].imag, 1e-9));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-9));
        expect(w.toList()[1].imag, closeTo(expectedW.toList()[1].imag, 1e-9));
      }),
    );

    test(
      'Eigenvalues of Complex64 matrix',
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

        final w = eigvals(a);
        final (eigenvalues: expectedW, eigenvectors: _) = eig(a);

        expect(w.dtype, DType.complex64);

        // Hardcoded expectations
        expect(w.toList()[0].real, closeTo(-1.0, 1e-5));
        expect(w.toList()[1].real, closeTo(-2.0, 1e-5));

        // Compare with eig
        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-5));
        expect(w.toList()[0].imag, closeTo(expectedW.toList()[0].imag, 1e-5));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-5));
        expect(w.toList()[1].imag, closeTo(expectedW.toList()[1].imag, 1e-5));
      }),
    );

    test(
      'Eigenvalues with out parameter recycler',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([0.0, 1.0, -2.0, -3.0]),
          [2, 2],
          DType.float64,
        );
        final outW = NDArray.zeros([2], DType.complex128);

        final w = eigvals(a, out: outW);

        expect(identical(w, outW), true);

        final (eigenvalues: expectedW, eigenvectors: _) = eig(a);
        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-9));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-9));
      }),
    );

    test(
      'Eigenvalues of 3D stack (broadcasting)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([
            0.0, 1.0, -2.0, -3.0, // matrix 1: eigenvalues -1, -2
            2.0, 0.0, 0.0, 2.0, // matrix 2: eigenvalues 2, 2
          ]),
          [2, 2, 2],
          DType.float64,
        );

        final w = eigvals(a);
        expect(w.shape, [2, 2]);
        expect(w.dtype, DType.complex128);

        final (eigenvalues: expectedW, eigenvectors: _) = eig(a);

        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-9));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-9));
        expect(w.toList()[2].real, closeTo(expectedW.toList()[2].real, 1e-9));
        expect(w.toList()[3].real, closeTo(expectedW.toList()[3].real, 1e-9));
      }),
    );

    test(
      'Eigenvalues of non-contiguous sliced view',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
          [3, 3],
          DType.float64,
        );
        final view = a.slice([
          Slice(start: 0, stop: 2),
          Slice(start: 0, stop: 2),
        ]);
        expect(view.isContiguous, false);

        final w = eigvals(view);

        final (eigenvalues: wEig, eigenvectors: _) = eig(view);
        expect(w.toList(), wEig.toList());
      }),
    );

    test(
      'Eigenvalues of non-contiguous transposed view',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          Float64List.fromList([0.0, -2.0, 1.0, -3.0]),
          [2, 2],
          DType.float64,
        );
        final view = parent.transpose(); // [[0, 1], [-2, -3]]
        expect(view.isContiguous, false);

        final w = eigvals(view);
        final (eigenvalues: expectedW, eigenvectors: _) = eig(view);

        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-9));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-9));
      }),
    );

    test(
      'eigvals() throws ArgumentError on non-square matrix',
      () => NDArray.scope(() {
        final a = NDArray.zeros([2, 3], DType.float64);
        expect(() => eigvals(a), throwsArgumentError);
      }),
    );

    test(
      'eigvals() auto-promotes integer matrix to float64',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
          2,
          2,
        ], DType.int32);
        final w = eigvals(a);
        final (eigenvalues: expectedW, eigenvectors: _) = eig(a);
        expect(w.dtype, equals(expectedW.dtype));
        expect(w.shape, equals([2]));
        expect(w.toList()[0].real, closeTo(expectedW.toList()[0].real, 1e-9));
        expect(w.toList()[1].real, closeTo(expectedW.toList()[1].real, 1e-9));
      }),
    );
  });
}
