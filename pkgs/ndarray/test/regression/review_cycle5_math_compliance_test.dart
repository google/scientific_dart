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
  group('Review Cycle 5 Math & AGENTS.md Compliance Regression Tests', () {
    test(
      'i0, gamma, erf preserve unmasked elements in non-contiguous out buffer with where mask',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([0.0, 1.0, 2.0, 3.0], [4], DType.float64);
          final mask = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );

          // Non-contiguous out via step slice
          final fullOut = NDArray.fromList(
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            [8],
            DType.float64,
          );
          final stridedOut = fullOut.slice([
            const Slice(start: 0, stop: 8, step: 2),
          ]);
          expect(stridedOut.isContiguous, isFalse);

          i0(a, where: mask, out: stridedOut);
          expect(stridedOut.getCell([0]), closeTo(1.0, 1e-6));
          expect(stridedOut.getCell([1]), equals(300.0)); // Unmasked preserved!
          expect(stridedOut.getCell([2]), closeTo(2.279585, 1e-5));
          expect(stridedOut.getCell([3]), equals(700.0)); // Unmasked preserved!

          // Test gamma with non-contiguous out and where
          final fullOutGamma = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            [8],
            DType.float64,
          );
          final stridedOutGamma = fullOutGamma.slice([
            const Slice(start: 0, stop: 8, step: 2),
          ]);
          final aGamma = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          gamma(aGamma, where: mask, out: stridedOutGamma);
          expect(stridedOutGamma.getCell([0]), closeTo(1.0, 1e-6));
          expect(stridedOutGamma.getCell([1]), equals(30.0));
          expect(stridedOutGamma.getCell([2]), closeTo(2.0, 1e-6));
          expect(stridedOutGamma.getCell([3]), equals(70.0));

          // Test erf with non-contiguous out and where
          final fullOutErf = NDArray.fromList(
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
            [8],
            DType.float64,
          );
          final stridedOutErf = fullOutErf.slice([
            const Slice(start: 0, stop: 8, step: 2),
          ]);
          erf(a, where: mask, out: stridedOutErf);
          expect(stridedOutErf.getCell([0]), closeTo(0.0, 1e-6));
          expect(stridedOutErf.getCell([1]), equals(-3.0));
          expect(stridedOutErf.getCell([2]), closeTo(0.995322, 1e-5));
          expect(stridedOutErf.getCell([3]), equals(-7.0));
        });
      },
    );

    test(
      'clipArray handles memory aliasing (out: flip(a)) and broadcast shapes accurately',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 5.0, 10.0, 15.0],
            [4],
            DType.float64,
          );
          final mn = NDArray.fromList([2.0], [1], DType.float64);
          final mx = NDArray.fromList([8.0], [1], DType.float64);

          final flipped = flip(a);
          clipArray(a, min: mn, max: mx, out: flipped);

          // Original values clipped: [2.0, 5.0, 8.0, 8.0]
          // Written into flipped view -> a should now be reversed: [8.0, 8.0, 5.0, 2.0]
          expect(a.toList(), equals([8.0, 8.0, 5.0, 2.0]));
        });
      },
    );

    test(
      'real and imag handle Complex128 and Complex64 strided arrays and aliasing views',
      () {
        NDArray.scope(() {
          final c128 = NDArray.fromList(
            [
              Complex(1.0, 10.0),
              Complex(2.0, 20.0),
              Complex(3.0, 30.0),
              Complex(4.0, 40.0),
            ],
            [4],
            DType.complex128,
          );
          final stridedC128 = flip(c128);
          final r128 = real(stridedC128);
          final i128 = imag(stridedC128);
          expect(r128.toList(), equals([4.0, 3.0, 2.0, 1.0]));
          expect(i128.toList(), equals([40.0, 30.0, 20.0, 10.0]));

          final c64 = NDArray.fromList(
            [Complex(1.5, -1.5), Complex(2.5, -2.5), Complex(3.5, -3.5)],
            [3],
            DType.complex64,
          );
          final stridedC64 = flip(c64);
          final r64 = real(stridedC64);
          final i64 = imag(stridedC64);
          expect(r64.toList(), equals([3.5, 2.5, 1.5]));
          expect(i64.toList(), equals([-3.5, -2.5, -1.5]));

          // Aliasing test for real array input to real() with flipped out view
          final realArr = NDArray.fromList(
            [10.0, 20.0, 30.0],
            [3],
            DType.float64,
          );
          real(realArr, out: flip(realArr));
          expect(realArr.toList(), equals([30.0, 20.0, 10.0]));
        });
      },
    );

    test('nan_to_num handles in-place strided aliasing (out: flip(a))', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [double.nan, 1.0, double.infinity, 2.0],
          [4],
          DType.float64,
        );
        nan_to_num(a, nan: 0.0, posinf: 999.0, out: flip(a));

        // Expected nan_to_num(a): [0.0, 1.0, 999.0, 2.0]
        // Written into flip(a) -> a becomes [2.0, 999.0, 1.0, 0.0]
        expect(a.toList(), equals([2.0, 999.0, 1.0, 0.0]));
      });
    });
  });
}
