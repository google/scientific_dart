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
  group('Review Cycle 10 - Linalg, Financial & Polynomial Regression Tests', () {
    group('1. outer view disposal and out aliasing', () {
      test(
        'outer(a2d, b2d) disposes ravel() views/copies and allows input disposal',
        () {
          final a2d = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b2d = NDArray<Float64>.fromList(
            [10.0, 20.0],
            [1, 2],
            DType.float64,
          );
          final res = outer<Float64>(a2d, b2d);

          expect(res.shape, equals([4, 2]));
          expect(
            res.toList(),
            equals([10.0, 20.0, 20.0, 40.0, 30.0, 60.0, 40.0, 80.0]),
          );

          a2d.dispose();
          b2d.dispose();
          expect(a2d.isDisposed, isTrue);
          expect(b2d.isDisposed, isTrue);
          res.dispose();
        },
      );

      test('outer(a, b, out: outView) where outView shares memory with a', () {
        final buf = NDArray<Float64>.fromList(
          [2.0, 3.0, 99.0, 99.0],
          [2, 2],
          DType.float64,
        );
        final a = buf.slice([Index(0), Slice.all()]);
        final b = NDArray<Float64>.fromList([4.0, 5.0], [2], DType.float64);

        final res = outer<Float64>(a, b, out: buf);
        expect(identical(res, buf), isTrue);
        // Expected outer([2, 3], [4, 5]) = [[8, 10], [12, 15]]
        expect(buf.toList(), equals([8.0, 10.0, 12.0, 15.0]));

        a.dispose();
        b.dispose();
        buf.dispose();
      });
    });

    group('2. cross in-place aliasing', () {
      test(
        'cross(a, b, out: a) with axis reordering or strided overlap computes accurately',
        () {
          // Basic 1D 3-vector in-place cross(a, b, out: a)
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final b = NDArray<Float64>.fromList(
            [4.0, 5.0, 6.0],
            [3],
            DType.float64,
          );
          final res = cross<Float64>(a, b, out: a);
          expect(identical(res, a), isTrue);
          // [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4] = [-3, 6, -3]
          expect(a.toList(), equals([-3.0, 6.0, -3.0]));
          a.dispose();
          b.dispose();

          // 2D cross where axisa=0, axisc=1 and out=a (writing across rows would overwrite ax/ay before reading)
          final aMat = NDArray<Float64>.fromList(
            [
              1.0, 2.0, 3.0, // ax
              4.0, 5.0, 6.0, // ay
              7.0, 8.0, 9.0, // az
            ],
            [3, 3],
            DType.float64,
          );
          final bMat = NDArray<Float64>.fromList(
            [
              9.0, 8.0, 7.0, // bx
              6.0, 5.0, 4.0, // by
              3.0, 2.0, 1.0, // bz
            ],
            [3, 3],
            DType.float64,
          );
          final expected = cross<Float64>(
            aMat,
            bMat,
            axisa: 0,
            axisb: 0,
            axisc: 1,
          );
          final aliasedRes = cross<Float64>(
            aMat,
            bMat,
            axisa: 0,
            axisb: 0,
            axisc: 1,
            out: aMat,
          );
          expect(identical(aliasedRes, aMat), isTrue);
          expect(aMat.toList(), equals(expected.toList()));

          expected.dispose();
          aMat.dispose();
          bMat.dispose();
        },
      );
    });

    group('3. norm where out shares memory with a', () {
      test(
        'norm(a, axis: 1, out: aSlice) computes accurately without overwriting rows prematurely',
        () {
          final a = NDArray<Float64>.fromList(
            [
              3.0, 4.0, // norm = 5.0
              6.0, 8.0, // norm = 10.0
              5.0, 12.0, // norm = 13.0
            ],
            [3, 2],
            DType.float64,
          );
          // Slice column 0: shape [3], non-contiguous and shares memory with a
          final aSlice = a.slice([Slice.all(), Index(0)]);
          final res = norm(a, axis: 1, out: aSlice);
          expect(identical(res, aSlice), isTrue);
          expect(aSlice.toList(), equals([5.0, 10.0, 13.0]));

          // Also test contiguous row slice sharing memory with a (e.g. axis: 0, out: row 0)
          final b = NDArray<Float64>.fromList(
            [3.0, 6.0, 5.0, 4.0, 8.0, 12.0],
            [2, 3],
            DType.float64,
          );
          final row0 = b.slice([Index(0), Slice.all()]);
          final resRow = norm(b, axis: 0, out: row0);
          expect(identical(resRow, row0), isTrue);
          expect(row0.toList(), equals([5.0, 10.0, 13.0]));

          aSlice.dispose();
          a.dispose();
          row0.dispose();
          b.dispose();
        },
      );
    });

    group(
      '4. irr, roots, and chebroots (_orthoRoots) validation and DType handling',
      () {
        test(
          'irr validates out shape upfront even when values.size == 1 or all same sign',
          () {
            final singleVal = NDArray<Float64>.fromList(
              [100.0],
              [1],
              DType.float64,
            );
            final badOut = NDArray<Float64>.zeros([1], DType.float64);

            expect(() => irr(singleVal, out: badOut), throwsArgumentError);

            final validOut = NDArray<Float64>.scalar(0.0, dtype: DType.float64);
            final res = irr(singleVal, out: validOut);
            expect(identical(res, validOut), isTrue);
            expect(validOut.scalar.isNaN, isTrue);

            singleVal.dispose();
            badOut.dispose();
            validOut.dispose();
          },
        );

        test(
          'roots with DType.complex64 input of degree 0 and degree 1 and out validation',
          () {
            // Degree 0 (1 coefficient)
            final pDeg0 = NDArray.fromList(
              [Complex(5.0, 0.0)],
              [1],
              DType.complex64,
            );
            final r0 = roots(pDeg0);
            expect(r0.dtype, equals(DType.complex64));
            expect(r0.shape, equals([0]));

            // Invalid out shape for degree 0
            final badOut0 = NDArray.zeros([1], DType.complex64);
            expect(() => roots(pDeg0, out: badOut0), throwsArgumentError);

            // Invalid out dtype for degree 0 (complex128 instead of complex64)
            final badDTypeOut0 = NDArray.zeros([0], DType.complex128);
            expect(() => roots(pDeg0, out: badDTypeOut0), throwsArgumentError);

            // Degree 1: 2x - 6 = 0 => x = 3
            final pDeg1 = NDArray.fromList(
              [Complex(2.0, 0.0), Complex(-6.0, 0.0)],
              [2],
              DType.complex64,
            );
            final r1 = roots(pDeg1);
            expect(r1.dtype, equals(DType.complex64));
            expect(r1.shape, equals([1]));
            expect(r1.getCell([0]).real, closeTo(3.0, 1e-5));
            expect(r1.getCell([0]).imag, closeTo(0.0, 1e-5));

            // Invalid out shape for degree 1
            final badOut1 = NDArray.zeros([2], DType.complex64);
            expect(() => roots(pDeg1, out: badOut1), throwsArgumentError);

            // Invalid out dtype for degree 1 (complex128 instead of complex64)
            final badDTypeOut1 = NDArray.zeros([1], DType.complex128);
            expect(() => roots(pDeg1, out: badDTypeOut1), throwsArgumentError);

            // Aliased out sharing memory with pDeg1
            final aliasedOut1 = pDeg1.slice([Slice(stop: 1)]);
            final r1Aliased = roots(pDeg1, out: aliasedOut1);
            expect(identical(r1Aliased, aliasedOut1), isTrue);
            expect(aliasedOut1.getCell([0]).real, closeTo(3.0, 1e-5));

            r0.dispose();
            pDeg0.dispose();
            badOut0.dispose();
            badDTypeOut0.dispose();
            r1.dispose();
            aliasedOut1.dispose();
            pDeg1.dispose();
            badOut1.dispose();
            badDTypeOut1.dispose();
          },
        );

        test(
          'chebroots (_orthoRoots) with DType.complex64 input of degree 0 and degree 1 and out validation',
          () {
            // Degree 0
            final cDeg0 = NDArray.fromList(
              [Complex(5.0, 0.0)],
              [1],
              DType.complex64,
            );
            final r0 = chebroots(cDeg0);
            expect(r0.dtype, equals(DType.complex64));
            expect(r0.shape, equals([0]));

            // Invalid out shape for degree 0
            final badOut0 = NDArray.zeros([1], DType.complex64);
            expect(() => chebroots(cDeg0, out: badOut0), throwsArgumentError);

            // Invalid out dtype for degree 0
            final badDTypeOut0 = NDArray.zeros([0], DType.complex128);
            expect(
              () => chebroots(cDeg0, out: badDTypeOut0),
              throwsArgumentError,
            );

            // Degree 1: c0 + c1 * T_1(x) = -6 + 2x = 0 => x = 3
            final cDeg1 = NDArray.fromList(
              [Complex(-6.0, 0.0), Complex(2.0, 0.0)],
              [2],
              DType.complex64,
            );
            final r1 = chebroots(cDeg1);
            expect(r1.dtype, equals(DType.complex64));
            expect(r1.shape, equals([1]));
            expect(r1.getCell([0]).real, closeTo(3.0, 1e-5));
            expect(r1.getCell([0]).imag, closeTo(0.0, 1e-5));

            // Invalid out shape for degree 1
            final badOut1 = NDArray.zeros([2], DType.complex64);
            expect(() => chebroots(cDeg1, out: badOut1), throwsArgumentError);

            // Invalid out dtype for degree 1
            final badDTypeOut1 = NDArray.zeros([1], DType.complex128);
            expect(
              () => chebroots(cDeg1, out: badDTypeOut1),
              throwsArgumentError,
            );

            // Aliased out sharing memory with cDeg1
            final aliasedOut1 = cDeg1.slice([Slice(stop: 1)]);
            final r1Aliased = chebroots(cDeg1, out: aliasedOut1);
            expect(identical(r1Aliased, aliasedOut1), isTrue);
            expect(aliasedOut1.getCell([0]).real, closeTo(3.0, 1e-5));

            r0.dispose();
            cDeg0.dispose();
            badOut0.dispose();
            badDTypeOut0.dispose();
            r1.dispose();
            aliasedOut1.dispose();
            cDeg1.dispose();
            badOut1.dispose();
            badDTypeOut1.dispose();
          },
        );
      },
    );
  });
}
