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
import 'dart:math' as math;

void main() {
  group('expm1 tests', () {
    test('float64 contiguous', () {
      final a = NDArray.fromList([0.0, 1.0, -1.0, 1e-15], [4], DType.float64);
      final r = expm1(a);
      expect(r.dtype, DType.float64);
      expect(r.data[0], closeTo(0.0, 1e-15));
      expect(r.data[1], closeTo(math.exp(1.0) - 1.0, 1e-15));
      expect(r.data[2], closeTo(math.exp(-1.0) - 1.0, 1e-15));
      expect(r.data[3], closeTo(1e-15, 1e-25)); // High precision check
    });

    test('float32 contiguous', () {
      final a = NDArray.fromList([0.0, 1.0, -1.0, 1e-7], [4], DType.float32);
      final r = expm1(a);
      expect(r.dtype, DType.float32);
      expect(r.data[0], closeTo(0.0, 1e-7));
      expect(r.data[1], closeTo(math.exp(1.0) - 1.0, 1e-6));
      expect(r.data[2], closeTo(math.exp(-1.0) - 1.0, 1e-6));
      expect(
        r.data[3],
        closeTo(1e-7, 1e-13),
      ); // High precision check for float32
    });

    test('complex128 contiguous', () {
      final a = NDArray.fromList(
        [
          Complex(0.0, 0.0),
          Complex(1.0, 2.0),
          Complex(-1.0, -2.0),
          Complex(1e-15, 1e-15),
        ],
        [4],
        DType.complex128,
      );
      final r = expm1(a);
      expect(r.dtype, DType.complex128);

      // exp(0) - 1 = 0
      expect(r.data[0].real, closeTo(0.0, 1e-15));
      expect(r.data[0].imag, closeTo(0.0, 1e-15));

      // exp(1+2i) - 1 = (e*cos(2) - 1) + i*e*sin(2)
      final e1 = math.exp(1.0);
      expect(r.data[1].real, closeTo(e1 * math.cos(2.0) - 1.0, 1e-15));
      expect(r.data[1].imag, closeTo(e1 * math.sin(2.0), 1e-15));

      // exp(-1-2i) - 1 = (e^-1*cos(-2) - 1) + i*e^-1*sin(-2)
      final e_1 = math.exp(-1.0);
      expect(r.data[2].real, closeTo(e_1 * math.cos(-2.0) - 1.0, 1e-15));
      expect(r.data[2].imag, closeTo(e_1 * math.sin(-2.0), 1e-15));

      // exp(1e-15 + 1e-15i) - 1
      expect(r.data[3].real, closeTo(1e-15, 1e-25));
      expect(r.data[3].imag, closeTo(1e-15, 1e-25));
    });

    test('strided float64 & out', () {
      final a = NDArray.fromList(
        [0.0, 99.0, 1.0, 99.0, -1.0, 99.0],
        [3, 2],
        DType.float64,
      );
      final view = a.slice([Slice.all(), Index(0)]); // [0.0, 1.0, -1.0] strided
      final out = NDArray.create([3], DType.float64);
      final r = expm1(view, out: out);
      expect(r.data[0], closeTo(0.0, 1e-15));
      expect(r.data[1], closeTo(math.exp(1.0) - 1.0, 1e-15));
      expect(r.data[2], closeTo(math.exp(-1.0) - 1.0, 1e-15));

      // verify out buffer was filled
      expect(out.data[0], closeTo(0.0, 1e-15));
      expect(out.data[1], closeTo(math.exp(1.0) - 1.0, 1e-15));
      expect(out.data[2], closeTo(math.exp(-1.0) - 1.0, 1e-15));
    });
  });

  group('log1p tests', () {
    test('float64 contiguous', () {
      final a = NDArray.fromList([0.0, 1.0, -0.5, 1e-15], [4], DType.float64);
      final r = log1p(a);
      expect(r.dtype, DType.float64);
      expect(r.data[0], closeTo(0.0, 1e-15));
      expect(r.data[1], closeTo(math.log(2.0), 1e-15));
      expect(r.data[2], closeTo(math.log(0.5), 1e-15));
      expect(r.data[3], closeTo(1e-15, 1e-25)); // High precision check
    });

    test('complex128 contiguous', () {
      final a = NDArray.fromList(
        [Complex(0.0, 0.0), Complex(1.0, 2.0), Complex(1e-15, 1e-15)],
        [3],
        DType.complex128,
      );
      final r = log1p(a);
      expect(r.dtype, DType.complex128);

      expect(r.data[0].real, closeTo(0.0, 1e-15));
      expect(r.data[0].imag, closeTo(0.0, 1e-15));

      // log1p(1+2i) = log(2+2i) = 0.5*ln(8) + i*pi/4
      expect(r.data[1].real, closeTo(0.5 * math.log(8.0), 1e-15));
      expect(r.data[1].imag, closeTo(math.pi / 4.0, 1e-15));

      // log1p(1e-15 + 1e-15i) approx 1e-15 + 1e-15i
      expect(r.data[2].real, closeTo(1e-15, 1e-25));
      expect(r.data[2].imag, closeTo(1e-15, 1e-25));
    });
  });

  group('logaddexp tests', () {
    test('float64 contiguous', () {
      final x1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final x2 = NDArray.fromList([2.0, 2.0, 2.0], [3], DType.float64);
      final r = logaddexp(x1, x2);
      expect(
        r.data[0],
        closeTo(math.log(math.exp(1.0) + math.exp(2.0)), 1e-15),
      );
      expect(r.data[1], closeTo(2.0 + math.log(2.0), 1e-15)); // same values
      expect(
        r.data[2],
        closeTo(math.log(math.exp(3.0) + math.exp(2.0)), 1e-15),
      );
    });

    test('overflow prevention', () {
      final x1 = NDArray.fromList(
        [1000.0, -1000.0, double.infinity, double.negativeInfinity],
        [4],
        DType.float64,
      );
      final x2 = NDArray.fromList(
        [1000.0, -1000.0, 0.0, double.negativeInfinity],
        [4],
        DType.float64,
      );
      final r = logaddexp(x1, x2);

      // 1000 and 1000 should not overflow to inf
      expect(r.data[0], closeTo(1000.0 + math.log(2.0), 1e-12));

      // -1000 and -1000 should not underflow to NaN/0, it should be -1000 + log(2)
      expect(r.data[1], closeTo(-1000.0 + math.log(2.0), 1e-12));

      // inf and 0 should be inf
      expect(r.data[2], double.infinity);

      // -inf and -inf should be -inf
      expect(r.data[3], double.negativeInfinity);
    });

    test('broadcasting', () {
      final x1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final x2 = NDArray.fromList([2.0], [1], DType.float64);
      final r = logaddexp(x1, x2);
      expect(r.shape, [2]);
      expect(
        r.data[0],
        closeTo(math.log(math.exp(1.0) + math.exp(2.0)), 1e-15),
      );
      expect(r.data[1], closeTo(2.0 + math.log(2.0), 1e-15));
    });
  });

  group('logaddexp2 tests', () {
    test('float64 contiguous', () {
      final x1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final x2 = NDArray.fromList([2.0, 2.0, 2.0], [3], DType.float64);
      final r = logaddexp2(x1, x2);
      expect(r.data[0], closeTo(math.log(6.0) / math.log(2.0), 1e-15));
      expect(r.data[1], closeTo(3.0, 1e-15));
      expect(r.data[2], closeTo(math.log(12.0) / math.log(2.0), 1e-15));
    });

    test('overflow prevention', () {
      final x1 = NDArray.fromList([1000.0, -1000.0], [2], DType.float64);
      final x2 = NDArray.fromList([1000.0, -1000.0], [2], DType.float64);
      final r = logaddexp2(x1, x2);
      expect(r.data[0], closeTo(1001.0, 1e-12));
      expect(r.data[1], closeTo(-999.0, 1e-12));
    });
  });

  group('rint tests', () {
    test('float64 contiguous', () {
      final a = NDArray.fromList(
        [0.1, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5],
        [8],
        DType.float64,
      );
      final r = rint(a);
      expect(r.data[0], 0.0);
      // Ties to even
      expect(r.data[1], 0.0); // 0.5 -> 0.0
      expect(r.data[2], 2.0); // 1.5 -> 2.0
      expect(r.data[3], 2.0); // 2.5 -> 2.0
      expect(r.data[4], 4.0); // 3.5 -> 4.0
      expect(r.data[5], -0.0); // -0.5 -> -0.0
      expect(r.data[6], -2.0); // -1.5 -> -2.0
      expect(r.data[7], -2.0); // -2.5 -> -2.0
    });

    test('integer input conversion', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final r = rint(a);
      expect(r.dtype, DType.float64); // returns floats
      expect(r.data[0], 1.0);
      expect(r.data[1], 2.0);
      expect(r.data[2], 3.0);
    });
  });

  group('trunc and fix tests', () {
    test('float64 contiguous', () {
      final a = NDArray.fromList(
        [0.1, 0.9, 1.1, -0.1, -0.9, -1.1],
        [6],
        DType.float64,
      );
      final r = trunc(a);
      expect(r.data[0], 0.0);
      expect(r.data[1], 0.0);
      expect(r.data[2], 1.0);
      expect(r.data[3], -0.0);
      expect(r.data[4], -0.0);
      expect(r.data[5], -1.0);

      final f = fix(a);
      expect(f.data, r.data);
    });

    test('integer input conversion', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final r = trunc(a);
      expect(r.dtype, DType.float64); // returns floats
      expect(r.data[0], 1.0);
      expect(r.data[1], 2.0);
      expect(r.data[2], 3.0);
    });
  });
}
