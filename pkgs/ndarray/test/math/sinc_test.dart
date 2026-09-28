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

void main() {
  group('Sinc Tests', () {
    test('Contiguous Real (Float64)', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, 0.5, 1.0, 2.0, -0.5],
          [5],
          DType.float64,
        );
        final res = sinc(a);
        expect(res.dtype, DType.float64);
        expect(res.getCell([0]), closeTo(1.0, 1e-15));
        expect(res.getCell([1]), closeTo(2.0 / math.pi, 1e-15));
        expect(res.getCell([2]), closeTo(0.0, 1e-15));
        expect(res.getCell([3]), closeTo(0.0, 1e-15));
        expect(res.getCell([4]), closeTo(2.0 / math.pi, 1e-15));
      });
    });

    test('Contiguous Real (Float32)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float32);
        final res = sinc(a);
        expect(res.dtype, DType.float32);
        expect(res.getCell([0]), closeTo(1.0, 1e-7));
        expect(res.getCell([1]), closeTo(2.0 / math.pi, 1e-7));
        expect(res.getCell([2]), closeTo(0.0, 1e-7));
      });
    });

    test('Strided Real (Float64)', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, 99.0, 0.5, 99.0, 1.0],
          [5],
          DType.float64,
        ).slice([const Slice(start: 0, stop: 5, step: 2)]);
        expect(a.isContiguous, isFalse);
        final res = sinc(a);
        expect(res.dtype, DType.float64);
        expect(res.shape, [3]);
        expect(res.getCell([0]), closeTo(1.0, 1e-15));
        expect(res.getCell([1]), closeTo(2.0 / math.pi, 1e-15));
        expect(res.getCell([2]), closeTo(0.0, 1e-15));
      });
    });

    test('Small values Real (Taylor expansion)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1e-5, -1e-5], [2], DType.float64);
        final res = sinc(a);

        double expected(double x) {
          final x2 = x * x;
          final pi2 = math.pi * math.pi;
          return 1.0 - (pi2 * x2) / 6.0 + (pi2 * pi2 * x2 * x2) / 120.0;
        }

        expect(res.getCell([0]), closeTo(expected(1e-5), 1e-15));
        expect(res.getCell([1]), closeTo(expected(-1e-5), 1e-15));
      });
    });

    test('Contiguous Complex (Complex128)', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(0.0, 0.0), Complex(0.5, 0.0), Complex(0.0, 0.5)],
          [3],
          DType.complex128,
        );
        final res = sinc(a);
        expect(res.dtype, DType.complex128);

        expect(res.getCell([0]), Complex(1.0, 0.0));
        expect(res.getCell([1]).real, closeTo(2.0 / math.pi, 1e-15));
        expect(res.getCell([1]).imag, closeTo(0.0, 1e-15));

        // sinc(0.5i) = sinh(pi*0.5) / (pi*0.5)
        final expectedVal =
            (math.exp(math.pi * 0.5) - math.exp(-math.pi * 0.5)) /
            (2.0 * math.pi * 0.5);
        expect(res.getCell([2]).real, closeTo(expectedVal, 1e-15));
        expect(res.getCell([2]).imag, closeTo(0.0, 1e-15));
      });
    });

    test('Contiguous Complex (Complex64)', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(0.0, 0.0), Complex(0.5, 0.0)],
          [2],
          DType.complex64,
        );
        final res = sinc(a);
        expect(res.dtype, DType.complex64);
        expect(res.getCell([0]), Complex(1.0, 0.0));
        expect(res.getCell([1]).real, closeTo(2.0 / math.pi, 1e-7));
        expect(res.getCell([1]).imag, closeTo(0.0, 1e-7));
      });
    });

    test('Small values Complex', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1e-5, 1e-5)],
          [1],
          DType.complex128,
        );
        final res = sinc(a);

        Complex expected(Complex z) {
          final z2 = z * z;
          final pi2 = math.pi * math.pi;
          return Complex(1.0, 0.0) -
              z2 * (pi2 / 6.0) +
              (z2 * z2) * (pi2 * pi2 / 120.0);
        }

        final exp = expected(Complex(1e-5, 1e-5));
        expect(res.getCell([0]).real, closeTo(exp.real, 1e-15));
        expect(res.getCell([0]).imag, closeTo(exp.imag, 1e-15));
      });
    });

    test('Strided Complex (Complex128)', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(0.0, 0.0), Complex(99.0, 99.0), Complex(0.5, 0.0)],
          [3],
          DType.complex128,
        );
        final sliced = a.slice([const Slice(start: 0, stop: 3, step: 2)]);
        expect(sliced.isContiguous, isFalse);
        final res = sinc(sliced);
        expect(res.shape, [2]);
        expect(res.getCell([0]), Complex(1.0, 0.0));
        expect(res.getCell([1]).real, closeTo(2.0 / math.pi, 1e-15));
        expect(res.getCell([1]).imag, closeTo(0.0, 1e-15));
      });
    });

    test('Integer fallback', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1, 2], [3], DType.int32);
        final res = sinc(a);
        expect(res.dtype, DType.float64);
        expect(res.getCell([0]), closeTo(1.0, 1e-15));
        expect(res.getCell([1]), closeTo(0.0, 1e-15));
        expect(res.getCell([2]), closeTo(0.0, 1e-15));
      });
    });

    test('Out parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final out = NDArray.zeros([3], DType.float64);
        final res = sinc(a, out: out);
        expect(identical(res, out), isTrue);
        expect(res.getCell([0]), closeTo(1.0, 1e-15));
        expect(res.getCell([1]), closeTo(2.0 / math.pi, 1e-15));
        expect(res.getCell([2]), closeTo(0.0, 1e-15));
      });
    });
  });
}
