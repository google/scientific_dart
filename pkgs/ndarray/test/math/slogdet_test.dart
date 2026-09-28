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
  group('slogdet Tests', () {
    group('2D Real Matrix slogdet', () {
      test('Float64 invertible matrix', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final (:sign, logabsdet: logdet) = slogdet(a);

          expect(sign.shape, <int>[]);
          expect(logdet.shape, <int>[]);
          expect(sign.scalar, -1.0);
          expect(logdet.scalar, closeTo(math.log(2.0), 1e-9));
        });
      });

      test('Float32 invertible matrix', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final (:sign, logabsdet: logdet) = slogdet(a);

          expect(sign.shape, <int>[]);
          expect(logdet.shape, <int>[]);
          expect(sign.scalar, -1.0);
          expect(logdet.scalar, closeTo(math.log(2.0), 1e-6));
        });
      });

      test('Float64 singular matrix', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 2.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final (:sign, logabsdet: logdet) = slogdet(a);

          expect(sign.scalar, 0.0);
          expect(logdet.scalar, double.negativeInfinity);
        });
      });
    });

    group('2D Complex Matrix slogdet', () {
      test('Complex128 invertible matrix', () {
        NDArray.scope(() {
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
          final (:sign, logabsdet: logdet) = slogdet<DTypeTag, DTypeTag>(a);

          expect(sign.shape, <int>[]);
          expect(logdet.shape, <int>[]);

          // det = -4 + 9i
          final expectedDet = Complex(-4.0, 9.0);
          final expectedAbs = math.sqrt(16.0 + 81.0);
          final expectedSign = Complex(
            expectedDet.real / expectedAbs,
            expectedDet.imag / expectedAbs,
          );

          expect(sign.scalar.real, closeTo(expectedSign.real, 1e-9));
          expect(sign.scalar.imag, closeTo(expectedSign.imag, 1e-9));
          expect(logdet.scalar, closeTo(math.log(expectedAbs), 1e-9));
        });
      });

      test('Complex64 invertible matrix', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(2.0, -1.0),
              Complex(3.0, 0.0),
              Complex(4.0, 2.0),
            ],
            [2, 2],
            DType.complex64,
          );
          final (:sign, logabsdet: logdet) = slogdet<DTypeTag, DTypeTag>(a);

          expect(sign.shape, <int>[]);
          expect(logdet.shape, <int>[]);

          final expectedDet = Complex(-4.0, 9.0);
          final expectedAbs = math.sqrt(16.0 + 81.0);
          final expectedSign = Complex(
            expectedDet.real / expectedAbs,
            expectedDet.imag / expectedAbs,
          );

          expect(sign.scalar.real, closeTo(expectedSign.real, 1e-5));
          expect(sign.scalar.imag, closeTo(expectedSign.imag, 1e-5));
          expect(logdet.scalar, closeTo(math.log(expectedAbs), 1e-5));
        });
      });

      test('Complex128 singular matrix', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(2.0, 2.0),
              Complex(2.0, 2.0),
              Complex(4.0, 4.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final (:sign, logabsdet: logdet) = slogdet<DTypeTag, DTypeTag>(a);

          expect(sign.scalar.real, 0.0);
          expect(sign.scalar.imag, 0.0);
          expect(logdet.scalar, double.negativeInfinity);
        });
      });
    });

    group('Stacked slogdet and broadcasting', () {
      test('Stacked Float64 matrices', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              1.0,
              2.0,
              3.0,
              4.0, // Matrix 0: det = -2, sign = -1, logdet = log(2)
              1.0, 2.0, 2.0, 4.0, // Matrix 1: det = 0, sign = 0, logdet = -inf
              5.0,
              6.0,
              7.0,
              9.0, // Matrix 2: det = 3, sign = 1, logdet = log(3)
            ]),
            [3, 2, 2],
            DType.float64,
          );

          final (:sign, logabsdet: logdet) = slogdet(a);

          expect(sign.shape, [3]);
          expect(logdet.shape, [3]);

          final signList = sign.toList();
          final logdetList = logdet.toList();

          expect(signList[0], -1.0);
          expect(logdetList[0], closeTo(math.log(2.0), 1e-9));

          expect(signList[1], 0.0);
          expect(logdetList[1], double.negativeInfinity);

          expect(signList[2], 1.0);
          expect(logdetList[2], closeTo(math.log(3.0), 1e-9));
        });
      });

      test('Stacked Complex128 matrices', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              // Matrix 0: det = -4 + 9i
              Complex(1.0, 1.0), Complex(2.0, -1.0),
              Complex(3.0, 0.0), Complex(4.0, 2.0),
              // Matrix 1: det = 0
              Complex(1.0, 1.0), Complex(2.0, 2.0),
              Complex(2.0, 2.0), Complex(4.0, 4.0),
            ],
            [2, 2, 2],
            DType.complex128,
          );

          final (:sign, logabsdet: logdet) = slogdet<DTypeTag, DTypeTag>(a);

          expect(sign.shape, [2]);
          expect(logdet.shape, [2]);

          final signList = sign.toList();
          final logdetList = logdet.toList();

          final expectedDet = Complex(-4.0, 9.0);
          final expectedAbs = math.sqrt(16.0 + 81.0);
          final expectedSign = Complex(
            expectedDet.real / expectedAbs,
            expectedDet.imag / expectedAbs,
          );

          expect(signList[0].real, closeTo(expectedSign.real, 1e-9));
          expect(signList[0].imag, closeTo(expectedSign.imag, 1e-9));
          expect(logdetList[0], closeTo(math.log(expectedAbs), 1e-9));

          expect(signList[1].real, 0.0);
          expect(signList[1].imag, 0.0);
          expect(logdetList[1], double.negativeInfinity);
        });
      });
    });

    group('Recycler buffers (outSign, outLogdet)', () {
      test('Float64 stack with out parameters', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              1.0, 2.0, 3.0, 4.0, // Matrix 0: det = -2
              5.0, 6.0, 7.0, 9.0, // Matrix 1: det = 3
            ]),
            [2, 2, 2],
            DType.float64,
          );

          final outSign = NDArray.zeros([2], DType.float64);
          final outLogdet = NDArray.zeros([2], DType.float64);

          final (:sign, logabsdet: logdet) = slogdet(
            a,
            outSign: outSign,
            outLogdet: outLogdet,
          );

          // Verify they are identical instances (mutated in place)
          expect(identical(sign, outSign), true);
          expect(identical(logdet, outLogdet), true);

          expect(outSign.toList()[0], -1.0);
          expect(outLogdet.toList()[0], closeTo(math.log(2.0), 1e-9));

          expect(outSign.toList()[1], 1.0);
          expect(outLogdet.toList()[1], closeTo(math.log(3.0), 1e-9));
        });
      });

      test('Incompatible out buffers throw ArgumentError', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );

          final badSign = NDArray.zeros([
            2,
          ], DType.float64); // bad shape, should be [] for 2D matrix
          final badLogdet = NDArray.zeros(
            [],
            DType.float32,
          ); // bad dtype, should be float64

          expect(() => slogdet(a, outSign: badSign), throwsArgumentError);
          expect(() => slogdet(a, outLogdet: badLogdet), throwsArgumentError);
        });
      });
    });
  });
}
