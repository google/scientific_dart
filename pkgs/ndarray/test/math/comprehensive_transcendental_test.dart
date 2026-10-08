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

import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('1. Trigonometric Functions Grid (sin, cos, tan)', () {
    test('Float64 and Float32 contiguous and strided', () {
      NDArray.scope(() {
        final angles = [
          0.0,
          math.pi / 6,
          math.pi / 4,
          math.pi / 3,
          math.pi / 2,
          math.pi,
        ];
        final a64 = NDArray.fromList(angles, [6], DType.float64);
        final a32 = NDArray.fromList(angles, [6], DType.float32);

        // Float64 sin, cos, tan
        final s64 = sin(a64);
        final c64 = cos(a64);
        final t64 = tan(a64);
        expect(s64.dtype, DType.float64);
        expect(c64.dtype, DType.float64);
        expect(t64.dtype, DType.float64);
        expect(s64.getCell([0]), closeTo(0.0, 1e-14));
        expect(s64.getCell([1]), closeTo(0.5, 1e-14));
        expect(s64.getCell([4]), closeTo(1.0, 1e-14));
        expect(c64.getCell([0]), closeTo(1.0, 1e-14));
        expect(c64.getCell([1]), closeTo(math.sqrt(3) / 2, 1e-14));
        expect(c64.getCell([4]), closeTo(0.0, 1e-14));
        expect(t64.getCell([0]), closeTo(0.0, 1e-14));
        expect(t64.getCell([1]), closeTo(1.0 / math.sqrt(3), 1e-14));

        // Float32 sin, cos, tan
        final s32 = sin(a32);
        final c32 = cos(a32);
        final t32 = tan(a32);
        expect(s32.dtype, DType.float32);
        expect(c32.dtype, DType.float32);
        expect(t32.dtype, DType.float32);
        expect(s32.getCell([0]), closeTo(0.0, 1e-6));
        expect(s32.getCell([1]), closeTo(0.5, 1e-6));
        expect(c32.getCell([0]), closeTo(1.0, 1e-6));
        expect(t32.getCell([0]), closeTo(0.0, 1e-6));

        // Strided (sliced with step 2)
        final sliced64 = a64.slice([const Slice(start: 0, stop: 6, step: 2)]);
        expect(sliced64.isContiguous, isFalse);
        final sSliced = sin(sliced64);
        expect(sSliced.shape, [3]);
        expect(sSliced.getCell([0]), closeTo(0.0, 1e-14));
        expect(sSliced.getCell([1]), closeTo(math.sin(math.pi / 4), 1e-14));
        expect(sSliced.getCell([2]), closeTo(1.0, 1e-14));

        final cSliced = cos(sliced64);
        expect(cSliced.getCell([0]), closeTo(1.0, 1e-14));
        expect(cSliced.getCell([1]), closeTo(math.cos(math.pi / 4), 1e-14));

        final tSliced = tan(sliced64);
        expect(tSliced.getCell([0]), closeTo(0.0, 1e-14));
        expect(tSliced.getCell([1]), closeTo(math.tan(math.pi / 4), 1e-14));
      });
    });

    test('Complex128 and Complex64 contiguous & strided', () {
      NDArray.scope(() {
        final c128List = [
          Complex(0.0, 0.0),
          Complex(math.pi / 2, 1.0),
          Complex(-math.pi / 4, 0.5),
          Complex(1.0, -2.0),
        ];
        final aC128 = NDArray.fromList(c128List, [4], DType.complex128);
        final aC64 = NDArray.fromList(c128List, [4], DType.complex64);

        // Sin complex: sin(x + iy) = sin(x)cosh(y) + i cos(x)sinh(y)
        final sC128 = sin(aC128);
        expect(sC128.dtype, DType.complex128);
        expect(sC128.getCell([0]).real, closeTo(0.0, 1e-14));
        expect(sC128.getCell([0]).imag, closeTo(0.0, 1e-14));

        final expectedSin1Real =
            math.sin(math.pi / 2) * (math.exp(1.0) + math.exp(-1.0)) / 2.0;
        final expectedSin1Imag =
            math.cos(math.pi / 2) * (math.exp(1.0) - math.exp(-1.0)) / 2.0;
        expect(sC128.getCell([1]).real, closeTo(expectedSin1Real, 1e-14));
        expect(sC128.getCell([1]).imag, closeTo(expectedSin1Imag, 1e-14));

        // Cos complex: cos(x + iy) = cos(x)cosh(y) - i sin(x)sinh(y)
        final cC128 = cos(aC128);
        expect(cC128.dtype, DType.complex128);
        expect(cC128.getCell([0]).real, closeTo(1.0, 1e-14));
        expect(cC128.getCell([0]).imag, closeTo(0.0, 1e-14));

        final expectedCos1Real =
            math.cos(math.pi / 2) * (math.exp(1.0) + math.exp(-1.0)) / 2.0;
        final expectedCos1Imag =
            -math.sin(math.pi / 2) * (math.exp(1.0) - math.exp(-1.0)) / 2.0;
        expect(cC128.getCell([1]).real, closeTo(expectedCos1Real, 1e-14));
        expect(cC128.getCell([1]).imag, closeTo(expectedCos1Imag, 1e-14));

        // Tan complex
        final tC128 = tan(aC128);
        expect(tC128.dtype, DType.complex128);
        expect(tC128.getCell([0]).real, closeTo(0.0, 1e-14));

        // Complex64 checks
        final sC64 = sin(aC64);
        expect(sC64.dtype, DType.complex64);
        expect(sC64.getCell([0]).real, closeTo(0.0, 1e-5));
        expect(sC64.getCell([1]).real, closeTo(expectedSin1Real, 1e-5));

        final cC64 = cos(aC64);
        expect(cC64.dtype, DType.complex64);
        expect(cC64.getCell([0]).real, closeTo(1.0, 1e-5));

        final tC64 = tan(aC64);
        expect(tC64.dtype, DType.complex64);

        // Strided Complex
        final stridedC128 = aC128.slice([
          const Slice(start: 0, stop: 4, step: 2),
        ]);
        expect(stridedC128.isContiguous, isFalse);
        final sStrided = sin(stridedC128);
        expect(sStrided.shape, [2]);
        expect(sStrided.getCell([0]).real, closeTo(0.0, 1e-14));

        final cStrided = cos(stridedC128);
        expect(cStrided.getCell([0]).real, closeTo(1.0, 1e-14));

        final tStrided = tan(stridedC128);
        expect(tStrided.getCell([0]).real, closeTo(0.0, 1e-14));
      });
    });

    test('Integer and Boolean promotions to Float64', () {
      NDArray.scope(() {
        final i64 = NDArray.fromList([0, 1, 2], [3], DType.int64);
        final i32 = NDArray.fromList([0, 1, 2], [3], DType.int32);
        final i16 = NDArray.fromList([0, 1, 2], [3], DType.int16);
        final i8 = NDArray.fromList([0, 1, 2], [3], DType.int8);
        final u64 = NDArray.fromList([0, 1, 2], [3], DType.uint64);
        final u32 = NDArray.fromList([0, 1, 2], [3], DType.uint32);
        final u16 = NDArray.fromList([0, 1, 2], [3], DType.uint16);
        final u8 = NDArray.fromList([0, 1, 2], [3], DType.uint8);
        final b = NDArray.fromList([false, true], [2], DType.boolean);

        expect(sin(i64).dtype, DType.float64);
        expect(sin(i32).dtype, DType.float64);
        expect(cos(i16).dtype, DType.float64);
        expect(tan(i8).dtype, DType.float64);
        expect(sin(u64).dtype, DType.float64);
        expect(sin(u32).dtype, DType.float64);
        expect(cos(u16).dtype, DType.float64);
        expect(tan(u8).dtype, DType.float64);
        expect(sin(b).dtype, DType.float64);
        expect(sin(b).getCell([0]), closeTo(0.0, 1e-14));
        expect(sin(b).getCell([1]), closeTo(math.sin(1.0), 1e-14));
      });
    });

    test('Out buffer reuse and where mask', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, math.pi / 2, math.pi],
          [3],
          DType.float64,
        );
        final outBuffer = NDArray.zeros([3], DType.float64);
        final mask = NDArray.fromList([true, false, true], [3], DType.boolean);

        final res = sin(a, where: mask, out: outBuffer);
        expect(identical(res, outBuffer), isTrue);
        expect(outBuffer.getCell([0]), closeTo(0.0, 1e-14));
        expect(outBuffer.getCell([1]), 0.0); // unmasked remained 0
        expect(outBuffer.getCell([2]), closeTo(0.0, 1e-14));

        // Incompatible shape or dtype throws ArgumentError
        final badShapeOut = NDArray.zeros([4], DType.float64);
        expect(() => sin(a, out: badShapeOut), throwsArgumentError);
        expect(() => cos(a, out: badShapeOut), throwsArgumentError);
        expect(() => tan(a, out: badShapeOut), throwsArgumentError);

        final badDTypeOut = NDArray.zeros([3], DType.float32);
        expect(() => sin(a, out: badDTypeOut), throwsArgumentError);
      });
    });

    test('Disposed array checks', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        expect(() => sin(a), throwsStateError);
        expect(() => cos(a), throwsStateError);
        expect(() => tan(a), throwsStateError);

        final validA = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final disposedOut = NDArray.zeros([2], DType.float64);
        disposedOut.dispose();
        expect(() => sin(validA, out: disposedOut), throwsStateError);

        final disposedWhere = NDArray.fromList(
          [true, false],
          [2],
          DType.boolean,
        );
        disposedWhere.dispose();
        expect(() => sin(validA, where: disposedWhere), throwsStateError);
      });
    });
  });

  group('2. Inverse Trigonometric Functions (asin, acos, atan, atan2)', () {
    test('asin, acos, atan across float64, float32, integer', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, 0.5, 1.0, -0.5, -1.0],
          [5],
          DType.float64,
        );
        final a32 = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float32);
        final aInt = NDArray.fromList([0, 1], [2], DType.int32);

        final resAsin = asin(a);
        final resAcos = acos(a);
        final resAtan = atan(a);

        expect(resAsin.getCell([0]), closeTo(0.0, 1e-14));
        expect(resAsin.getCell([1]), closeTo(math.asin(0.5), 1e-14));
        expect(resAsin.getCell([2]), closeTo(math.pi / 2, 1e-14));

        expect(resAcos.getCell([0]), closeTo(math.pi / 2, 1e-14));
        expect(resAcos.getCell([1]), closeTo(math.acos(0.5), 1e-14));
        expect(resAcos.getCell([2]), closeTo(0.0, 1e-14));

        expect(resAtan.getCell([0]), closeTo(0.0, 1e-14));
        expect(resAtan.getCell([2]), closeTo(math.pi / 4, 1e-14));

        // Float32
        expect(asin(a32).dtype, DType.float32);
        expect(acos(a32).dtype, DType.float32);
        expect(atan(a32).dtype, DType.float32);

        // Integer
        expect(asin(aInt).dtype, DType.float64);
        expect(acos(aInt).dtype, DType.float64);
        expect(atan(aInt).dtype, DType.float64);
      });
    });

    test('asin, acos, atan complex types', () {
      NDArray.scope(() {
        final cList = [Complex(0.0, 0.0), Complex(2.0, 0.0), Complex(0.0, 1.0)];
        final c128 = NDArray.fromList(cList, [3], DType.complex128);
        final c64 = NDArray.fromList(cList, [3], DType.complex64);

        final resAsin = asin(c128);
        expect(resAsin.dtype, DType.complex128);
        expect(resAsin.getCell([0]).real, closeTo(0.0, 1e-14));
        expect(resAsin.getCell([0]).imag, closeTo(0.0, 1e-14));

        final resAcos = acos(c128);
        expect(resAcos.dtype, DType.complex128);
        expect(resAcos.getCell([0]).real, closeTo(math.pi / 2, 1e-14));

        final resAtan = atan(c128);
        expect(resAtan.dtype, DType.complex128);
        expect(resAtan.getCell([0]).real, closeTo(0.0, 1e-14));

        // Complex64
        expect(asin(c64).dtype, DType.complex64);
        expect(acos(c64).dtype, DType.complex64);
        expect(atan(c64).dtype, DType.complex64);

        // Strided complex
        final strided = c128.slice([const Slice(start: 0, stop: 3, step: 2)]);
        expect(asin(strided).shape, [2]);
        expect(acos(strided).shape, [2]);
        expect(atan(strided).shape, [2]);
      });
    });

    test('atan2 4 quadrants, broadcasting, strided, error handling', () {
      NDArray.scope(() {
        final y = NDArray.fromList(
          [0.0, 1.0, 0.0, -1.0, 1.0, -1.0],
          [6],
          DType.float64,
        );
        final x = NDArray.fromList(
          [1.0, 0.0, -1.0, 0.0, 1.0, -1.0],
          [6],
          DType.float64,
        );
        final res = atan2(y, x);

        expect(res.getCell([0]), closeTo(0.0, 1e-14));
        expect(res.getCell([1]), closeTo(math.pi / 2, 1e-14));
        expect(res.getCell([2]), closeTo(math.pi, 1e-14));
        expect(res.getCell([3]), closeTo(-math.pi / 2, 1e-14));
        expect(res.getCell([4]), closeTo(math.pi / 4, 1e-14));
        expect(res.getCell([5]), closeTo(-3 * math.pi / 4, 1e-14));

        // Float32
        final y32 = NDArray.fromList([1.0, -1.0], [2], DType.float32);
        final x32 = NDArray.fromList([1.0, -1.0], [2], DType.float32);
        final res32 = atan2(y32, x32);
        expect(res32.dtype, DType.float32);
        expect(res32.getCell([0]), closeTo(math.pi / 4, 1e-6));

        // Broadcasting: y is [2, 1], x is [1, 3] -> result is [2, 3]
        final yMat = NDArray.fromList([1.0, -1.0], [2, 1], DType.float64);
        final xMat = NDArray.fromList([1.0, 0.0, -1.0], [1, 3], DType.float64);
        final resMat = atan2(yMat, xMat);
        expect(resMat.shape, [2, 3]);
        expect(resMat.getCell([0, 0]), closeTo(math.pi / 4, 1e-14));
        expect(resMat.getCell([0, 1]), closeTo(math.pi / 2, 1e-14));
        expect(resMat.getCell([0, 2]), closeTo(3 * math.pi / 4, 1e-14));

        // Integer broadcasting
        final yInt = NDArray.fromList([1, 0], [2], DType.int32);
        final xInt = NDArray.fromList([0, 1], [2], DType.int32);
        final resInt = atan2(yInt, xInt);
        expect(resInt.dtype, DType.float64);
        expect(resInt.getCell([0]), closeTo(math.pi / 2, 1e-14));
        expect(resInt.getCell([1]), closeTo(0.0, 1e-14));

        // Out buffer & where mask
        final outAtan2 = NDArray.zeros([6], DType.float64);
        final mask = NDArray.fromList(
          [true, false, true, false, true, false],
          [6],
          DType.boolean,
        );
        atan2(y, x, where: mask, out: outAtan2);
        expect(outAtan2.getCell([0]), closeTo(0.0, 1e-14));
        expect(outAtan2.getCell([1]), 0.0);
        expect(outAtan2.getCell([2]), closeTo(math.pi, 1e-14));

        // Complex throws UnsupportedError, mixed dtypes throw ArgumentError
        final cArray = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
        final rArray = NDArray.fromList([1.0], [1], DType.float64);
        expect(() => atan2(cArray, cArray), throwsUnsupportedError);
        expect(
          () => atan2<AnySpec, DTypeTag>(cArray, rArray),
          throwsArgumentError,
        );
        expect(
          () => atan2<AnySpec, DTypeTag>(rArray, cArray),
          throwsArgumentError,
        );

        // Disposed checks
        final disposedY = NDArray.fromList([1.0], [1], DType.float64)
          ..dispose();
        expect(() => atan2(disposedY, rArray), throwsStateError);
      });
    });
  });

  group(
    '3. Hyperbolic & Inverse Hyperbolic Functions (sinh, cosh, tanh, asinh, acosh, atanh)',
    () {
      test('Hyperbolic on Float64, Float32, strided', () {
        NDArray.scope(() {
          final vals = [0.0, 0.5, 1.0, 2.0, -1.0];
          final a64 = NDArray.fromList(vals, [5], DType.float64);
          final a32 = NDArray.fromList(vals, [5], DType.float32);

          final s = sinh(a64);
          final c = cosh(a64);
          final t = tanh(a64);

          expect(s.getCell([0]), closeTo(0.0, 1e-14));
          expect(c.getCell([0]), closeTo(1.0, 1e-14));
          expect(t.getCell([0]), closeTo(0.0, 1e-14));

          final expectedSinh1 = (math.exp(1.0) - math.exp(-1.0)) / 2.0;
          final expectedCosh1 = (math.exp(1.0) + math.exp(-1.0)) / 2.0;
          expect(s.getCell([2]), closeTo(expectedSinh1, 1e-14));
          expect(c.getCell([2]), closeTo(expectedCosh1, 1e-14));
          expect(t.getCell([2]), closeTo(expectedSinh1 / expectedCosh1, 1e-14));

          // Float32
          expect(sinh(a32).dtype, DType.float32);
          expect(cosh(a32).dtype, DType.float32);
          expect(tanh(a32).dtype, DType.float32);

          // Strided
          final strided = a64.slice([const Slice(start: 0, stop: 5, step: 2)]);
          expect(sinh(strided).shape, [3]);
          expect(cosh(strided).shape, [3]);
          expect(tanh(strided).shape, [3]);
        });
      });

      test('Hyperbolic on Complex128 and Complex64', () {
        NDArray.scope(() {
          final cVals = [
            Complex(0.0, 0.0),
            Complex(1.0, math.pi / 4),
            Complex(-0.5, 0.5),
          ];
          final a128 = NDArray.fromList(cVals, [3], DType.complex128);
          final a64 = NDArray.fromList(cVals, [3], DType.complex64);

          final s = sinh(a128);
          final c = cosh(a128);
          final t = tanh(a128);

          expect(s.dtype, DType.complex128);
          expect(c.dtype, DType.complex128);
          expect(t.dtype, DType.complex128);

          expect(s.getCell([0]).real, closeTo(0.0, 1e-14));
          expect(c.getCell([0]).real, closeTo(1.0, 1e-14));
          expect(t.getCell([0]).real, closeTo(0.0, 1e-14));

          // sinh(x+iy) = sinh(x)cos(y) + i cosh(x)sin(y)
          final exSinhReal =
              ((math.exp(1.0) - math.exp(-1.0)) / 2.0) * math.cos(math.pi / 4);
          final exSinhImag =
              ((math.exp(1.0) + math.exp(-1.0)) / 2.0) * math.sin(math.pi / 4);
          expect(s.getCell([1]).real, closeTo(exSinhReal, 1e-14));
          expect(s.getCell([1]).imag, closeTo(exSinhImag, 1e-14));

          // Complex64
          expect(sinh(a64).dtype, DType.complex64);
          expect(cosh(a64).dtype, DType.complex64);
          expect(tanh(a64).dtype, DType.complex64);
        });
      });

      test('Inverse Hyperbolic (asinh, acosh, atanh)', () {
        NDArray.scope(() {
          final asinhVals = [0.0, 1.0, 2.0, -1.0];
          final acoshVals = [1.0, 1.5, 2.0, 5.0];
          final atanhVals = [0.0, 0.5, -0.5, 0.8];

          final arrAsinh = NDArray.fromList(asinhVals, [4], DType.float64);
          final arrAcosh = NDArray.fromList(acoshVals, [4], DType.float64);
          final arrAtanh = NDArray.fromList(atanhVals, [4], DType.float64);

          final resAsinh = asinh(arrAsinh);
          final resAcosh = acosh(arrAcosh);
          final resAtanh = atanh(arrAtanh);

          expect(resAsinh.getCell([0]), closeTo(0.0, 1e-14));
          expect(
            resAsinh.getCell([1]),
            closeTo(math.log(1.0 + math.sqrt(2.0)), 1e-14),
          );

          expect(resAcosh.getCell([0]), closeTo(0.0, 1e-14));
          expect(
            resAcosh.getCell([2]),
            closeTo(math.log(2.0 + math.sqrt(3.0)), 1e-14),
          );

          expect(resAtanh.getCell([0]), closeTo(0.0, 1e-14));
          expect(
            resAtanh.getCell([1]),
            closeTo(0.5 * math.log(1.5 / 0.5), 1e-14),
          );

          // Complex inverse hyperbolic
          final cVals = [Complex(1.0, 0.5), Complex(0.0, 2.0)];
          final cArr = NDArray.fromList(cVals, [2], DType.complex128);
          expect(asinh(cArr).dtype, DType.complex128);
          expect(acosh(cArr).dtype, DType.complex128);
          expect(atanh(cArr).dtype, DType.complex128);

          // Strided views
          final stridedAsinh = arrAsinh.slice([
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          expect(asinh(stridedAsinh).shape, [2]);
        });
      });
    },
  );

  group('4. Sinc, Angle Conversions & Hypot', () {
    test('sinc on Float64, Float32, Complex, and strided', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, 0.5, 1.0, 2.0, -0.5],
          [5],
          DType.float64,
        );
        final res = sinc(a);
        expect(res.dtype, DType.float64);
        expect(res.getCell([0]), closeTo(1.0, 1e-14));
        expect(res.getCell([1]), closeTo(2.0 / math.pi, 1e-14));
        expect(res.getCell([2]), closeTo(0.0, 1e-14));
        expect(res.getCell([3]), closeTo(0.0, 1e-14));
        expect(res.getCell([4]), closeTo(2.0 / math.pi, 1e-14));

        // Small values (Taylor series expansion)
        final smallA = NDArray.fromList([1e-6, -1e-6], [2], DType.float64);
        final smallRes = sinc(smallA);
        expect(
          smallRes.getCell([0]),
          closeTo(1.0 - (math.pi * math.pi * 1e-12) / 6.0, 1e-14),
        );

        // Complex sinc
        final cA = NDArray.fromList(
          [Complex(0.0, 0.0), Complex(0.5, 0.0)],
          [2],
          DType.complex128,
        );
        final cRes = sinc(cA);
        expect(cRes.getCell([0]).real, closeTo(1.0, 1e-14));
        expect(cRes.getCell([1]).real, closeTo(2.0 / math.pi, 1e-14));

        // Sliced strided sinc
        final strided = a.slice([const Slice(start: 0, stop: 5, step: 2)]);
        final stridedRes = sinc(strided);
        expect(stridedRes.shape, [3]);
        expect(stridedRes.getCell([0]), closeTo(1.0, 1e-14));
        expect(stridedRes.getCell([1]), closeTo(0.0, 1e-14));
      });
    });

    test('deg2rad and rad2deg with Float64, Float32, and Int32', () {
      NDArray.scope(() {
        final deg = NDArray.fromList(
          [0.0, 90.0, 180.0, 270.0, 360.0, -45.0],
          [6],
          DType.float64,
        );
        final rad = deg2rad(deg);
        expect(rad.getCell([0]), closeTo(0.0, 1e-14));
        expect(rad.getCell([1]), closeTo(math.pi / 2, 1e-14));
        expect(rad.getCell([2]), closeTo(math.pi, 1e-14));
        expect(rad.getCell([3]), closeTo(3 * math.pi / 2, 1e-14));
        expect(rad.getCell([4]), closeTo(2 * math.pi, 1e-14));
        expect(rad.getCell([5]), closeTo(-math.pi / 4, 1e-14));

        final backToDeg = rad2deg(rad);
        expect(backToDeg.getCell([0]), closeTo(0.0, 1e-14));
        expect(backToDeg.getCell([1]), closeTo(90.0, 1e-14));
        expect(backToDeg.getCell([2]), closeTo(180.0, 1e-14));
        expect(backToDeg.getCell([3]), closeTo(270.0, 1e-14));

        // Float32
        final deg32 = NDArray.fromList([180.0, 90.0], [2], DType.float32);
        final rad32 = deg2rad(deg32);
        expect(rad32.dtype, DType.float32);
        expect(rad32.getCell([0]), closeTo(math.pi, 1e-5));

        // Int32
        final degInt = NDArray.fromList([180, 90], [2], DType.int32);
        final radInt = deg2rad(degInt);
        expect(radInt.dtype, DType.float64);
        expect(radInt.getCell([0]), closeTo(math.pi, 1e-14));

        // Complex throws UnsupportedError
        final cArr = NDArray.fromList([Complex(180, 0)], [1], DType.complex128);
        expect(() => deg2rad(cArr), throwsUnsupportedError);
        expect(() => rad2deg(cArr), throwsUnsupportedError);
      });
    });

    test('hypot real, complex, broadcasting, strided', () {
      NDArray.scope(() {
        final a = NDArray.fromList([3.0, 5.0, 8.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 12.0, 15.0], [3], DType.float64);
        final h = hypot(a, b);
        expect(h.getCell([0]), closeTo(5.0, 1e-14));
        expect(h.getCell([1]), closeTo(13.0, 1e-14));
        expect(h.getCell([2]), closeTo(17.0, 1e-14));

        // Broadcasting: [3, 1] and [1, 2]
        final aMat = NDArray.fromList([3.0, 5.0, 0.0], [3, 1], DType.float64);
        final bMat = NDArray.fromList([4.0, 12.0], [1, 2], DType.float64);
        final hMat = hypot(aMat, bMat);
        expect(hMat.shape, [3, 2]);
        expect(hMat.getCell([0, 0]), closeTo(5.0, 1e-14));
        expect(hMat.getCell([0, 1]), closeTo(math.sqrt(9 + 144), 1e-14));
        expect(hMat.getCell([1, 0]), closeTo(math.sqrt(25 + 16), 1e-14));
        expect(hMat.getCell([1, 1]), closeTo(13.0, 1e-14));

        // Complex hypot throws UnsupportedError (S7)
        final aCpx = NDArray.fromList(
          [Complex(3.0, 0.0)],
          [1],
          DType.complex128,
        );
        final bCpx = NDArray.fromList(
          [Complex(4.0, 0.0)],
          [1],
          DType.complex128,
        );
        expect(() => hypot(aCpx, bCpx), throwsUnsupportedError);

        // Infinities
        final aInf = NDArray.fromList(
          [double.infinity, 0.0],
          [2],
          DType.float64,
        );
        final bNan = NDArray.fromList([double.nan, 0.0], [2], DType.float64);
        final hInf = hypot(aInf, bNan);
        expect(hInf.getCell([0]), double.infinity);
        expect(hInf.getCell([1]), 0.0);
      });
    });
  });

  group(
    '5. Exponential & Logarithmic Functions (exp, log, log2, log10, expm1, log1p, logaddexp, logaddexp2)',
    () {
      test('exp, log, log2, log10 on Float64, Float32, Integers', () {
        NDArray.scope(() {
          final vals = [0.0, 1.0, 2.0, -1.0, 10.0];
          final a = NDArray.fromList(vals, [5], DType.float64);
          final a32 = NDArray.fromList(vals, [5], DType.float32);

          // exp
          final expRes = exp(a);
          expect(expRes.getCell([0]), closeTo(1.0, 1e-14));
          expect(expRes.getCell([1]), closeTo(math.e, 1e-14));
          expect(expRes.getCell([2]), closeTo(math.e * math.e, 1e-14));
          expect(expRes.getCell([3]), closeTo(1.0 / math.e, 1e-14));

          // log
          final logVals = [1.0, math.e, math.e * math.e, 10.0];
          final aLog = NDArray.fromList(logVals, [4], DType.float64);
          final logRes = log(aLog);
          expect(logRes.getCell([0]), closeTo(0.0, 1e-14));
          expect(logRes.getCell([1]), closeTo(1.0, 1e-14));
          expect(logRes.getCell([2]), closeTo(2.0, 1e-14));

          // log2
          final log2Vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
          final aLog2 = NDArray.fromList(log2Vals, [6], DType.float64);
          final log2Res = log2(aLog2);
          for (var i = 0; i < 6; i++) {
            expect(log2Res.getCell([i]), closeTo(i.toDouble(), 1e-14));
          }

          // log10
          final log10Vals = [1.0, 10.0, 100.0, 1000.0];
          final aLog10 = NDArray.fromList(log10Vals, [4], DType.float64);
          final log10Res = log10(aLog10);
          for (var i = 0; i < 4; i++) {
            expect(log10Res.getCell([i]), closeTo(i.toDouble(), 1e-14));
          }

          // Float32
          expect(exp(a32).dtype, DType.float32);
          expect(log(a32).dtype, DType.float32);
          expect(log2(a32).dtype, DType.float32);
          expect(log10(a32).dtype, DType.float32);

          // Strided views
          final strided = a.slice([const Slice(start: 0, stop: 5, step: 2)]);
          expect(exp(strided).shape, [3]);
          expect(log(strided).shape, [3]);
          expect(log2(strided).shape, [3]);
          expect(log10(strided).shape, [3]);
        });
      });

      test('Complex exponential & logarithmic functions', () {
        NDArray.scope(() {
          final cVals = [
            Complex(0.0, 0.0),
            Complex(1.0, math.pi),
            Complex(0.0, math.pi / 2),
            Complex(2.0, 1.0),
          ];
          final aC128 = NDArray.fromList(cVals, [4], DType.complex128);
          final aC64 = NDArray.fromList(cVals, [4], DType.complex64);

          // exp(0) = 1, exp(1 + i*pi) = -e, exp(i*pi/2) = i
          final expC = exp(aC128);
          expect(expC.dtype, DType.complex128);
          expect(expC.getCell([0]).real, closeTo(1.0, 1e-14));
          expect(expC.getCell([0]).imag, closeTo(0.0, 1e-14));
          expect(expC.getCell([1]).real, closeTo(-math.e, 1e-14));
          expect(expC.getCell([1]).imag, closeTo(0.0, 1e-14));
          expect(expC.getCell([2]).real, closeTo(0.0, 1e-14));
          expect(expC.getCell([2]).imag, closeTo(1.0, 1e-14));

          // log complex
          final logC = log(aC128);
          expect(logC.dtype, DType.complex128);
          expect(
            logC.getCell([1]).real,
            closeTo(
              math.sqrt(1 + math.pi * math.pi) > 0
                  ? math.log(Complex(1.0, math.pi).abs)
                  : 0,
              1e-14,
            ),
          );

          // log2 & log10 complex
          final log2C = log2(aC128);
          final log10C = log10(aC128);
          expect(log2C.dtype, DType.complex128);
          expect(log10C.dtype, DType.complex128);

          // Complex64 checks
          expect(exp(aC64).dtype, DType.complex64);
          expect(log(aC64).dtype, DType.complex64);
          expect(log2(aC64).dtype, DType.complex64);
          expect(log10(aC64).dtype, DType.complex64);
        });
      });

      test('expm1 and log1p high precision and complex branch', () {
        NDArray.scope(() {
          final smallVals = [0.0, 1e-15, -1e-15, 0.5, 1.0];
          final a = NDArray.fromList(smallVals, [5], DType.float64);

          final resExpm1 = expm1(a);
          expect(resExpm1.getCell([0]), closeTo(0.0, 1e-15));
          expect(resExpm1.getCell([1]), closeTo(1e-15, 1e-20));
          expect(resExpm1.getCell([4]), closeTo(math.e - 1.0, 1e-14));

          final resLog1p = log1p(a);
          expect(resLog1p.getCell([0]), closeTo(0.0, 1e-15));
          expect(resLog1p.getCell([1]), closeTo(1e-15, 1e-20));
          expect(resLog1p.getCell([4]), closeTo(math.log(2.0), 1e-14));

          // Complex expm1 & log1p
          final cList = [
            Complex(0.0, 0.0),
            Complex(1e-10, 1e-10),
            Complex(1.0, 1.0),
          ];
          final aCpx = NDArray.fromList(cList, [3], DType.complex128);
          final expm1C = expm1(aCpx);
          final log1pC = log1p(aCpx);
          expect(expm1C.getCell([0]).real, closeTo(0.0, 1e-14));
          expect(expm1C.getCell([0]).imag, closeTo(0.0, 1e-14));
          expect(log1pC.getCell([0]).real, closeTo(0.0, 1e-14));
          expect(log1pC.getCell([0]).imag, closeTo(0.0, 1e-14));
        });
      });

      test('logaddexp and logaddexp2 with broadcasting and strided views', () {
        NDArray.scope(() {
          final x1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final x2 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

          // logaddexp(x, x) = log(exp(x) + exp(x)) = log(2*exp(x)) = x + log(2)
          final res = logaddexp(x1, x2);
          expect(res.getCell([0]), closeTo(1.0 + math.log(2.0), 1e-14));
          expect(res.getCell([1]), closeTo(2.0 + math.log(2.0), 1e-14));
          expect(res.getCell([2]), closeTo(3.0 + math.log(2.0), 1e-14));

          // logaddexp2(x, x) = log2(2^x + 2^x) = log2(2^(x+1)) = x + 1
          final res2 = logaddexp2(x1, x2);
          expect(res2.getCell([0]), closeTo(2.0, 1e-14));
          expect(res2.getCell([1]), closeTo(3.0, 1e-14));
          expect(res2.getCell([2]), closeTo(4.0, 1e-14));

          // Broadcasting: [3, 1] and [1, 2]
          final m1 = NDArray.fromList([0.0, 1.0, 2.0], [3, 1], DType.float64);
          final m2 = NDArray.fromList([0.0, 1.0], [1, 2], DType.float64);
          final resBcast = logaddexp(m1, m2);
          expect(resBcast.shape, [3, 2]);
          expect(resBcast.getCell([0, 0]), closeTo(math.log(2.0), 1e-14));

          final resBcast2 = logaddexp2(m1, m2);
          expect(resBcast2.shape, [3, 2]);
          expect(resBcast2.getCell([0, 0]), closeTo(1.0, 1e-14));
        });
      });
    },
  );

  group('6. Special Functions (i0, gamma, erf)', () {
    test('i0 modified Bessel function order 0', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, 1.0, 2.0, 3.75, 5.0],
          [5],
          DType.float64,
        );
        final res = i0(a);
        expect(res.dtype, DType.float64);
        expect(res.getCell([0]), closeTo(1.0, 1e-14));
        expect(res.getCell([1]), closeTo(1.2660658777520084, 1e-6));
        expect(res.getCell([2]), closeTo(2.2795853023360673, 1e-6));

        // Float32
        final a32 = NDArray.fromList([0.0, 1.0, 5.0], [3], DType.float32);
        final res32 = i0(a32);
        expect(res32.dtype, DType.float32);
        expect(res32.getCell([0]), closeTo(1.0, 1e-6));

        // Complex series (|z| <= 15) and asymptotic (|z| > 15)
        final cList = [
          Complex(0.0, 0.0),
          Complex(1.0, 1.0),
          Complex(20.0, 0.0),
        ];
        final cArr = NDArray.fromList(cList, [3], DType.complex128);
        final cRes = i0(cArr);
        expect(cRes.dtype, DType.complex128);
        expect(cRes.getCell([0]).real, closeTo(1.0, 1e-14));
        expect(cRes.getCell([0]).imag, closeTo(0.0, 1e-14));

        // Strided i0
        final strided = a.slice([const Slice(start: 0, stop: 5, step: 2)]);
        final stridedRes = i0(strided);
        expect(stridedRes.shape, [3]);
        expect(stridedRes.getCell([0]), closeTo(1.0, 1e-14));
      });
    });

    test('gamma function', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
          [6],
          DType.float64,
        );
        final res = gamma(a);
        expect(res.getCell([0]), closeTo(1.0, 1e-14)); // 0! = 1
        expect(res.getCell([1]), closeTo(1.0, 1e-14)); // 1! = 1
        expect(res.getCell([2]), closeTo(2.0, 1e-14)); // 2! = 2
        expect(res.getCell([3]), closeTo(6.0, 1e-14)); // 3! = 6
        expect(res.getCell([4]), closeTo(24.0, 1e-14)); // 4! = 24
        expect(
          res.getCell([5]),
          closeTo(math.sqrt(math.pi), 1e-14),
        ); // gamma(1/2) = sqrt(pi)

        // Float32
        final a32 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
        expect(gamma(a32).dtype, DType.float32);

        // Integer promotion
        final aInt = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final resInt = gamma(aInt);
        expect(resInt.dtype, DType.float64);
        expect(resInt.getCell([3]), closeTo(6.0, 1e-14));

        // Complex throws UnsupportedError
        final cArr = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
        expect(() => gamma(cArr), throwsUnsupportedError);
      });
    });

    test('erf error function', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, 1.0, 2.0, -1.0, 10.0, -10.0],
          [6],
          DType.float64,
        );
        final res = erf(a);
        expect(res.getCell([0]), closeTo(0.0, 1e-14));
        expect(res.getCell([1]), closeTo(0.8427007929497148, 1e-6));
        expect(res.getCell([3]), closeTo(-0.8427007929497148, 1e-6));
        expect(res.getCell([4]), closeTo(1.0, 1e-6));
        expect(res.getCell([5]), closeTo(-1.0, 1e-6));

        // Float32
        final a32 = NDArray.fromList([0.0, 1.0], [2], DType.float32);
        expect(erf(a32).dtype, DType.float32);

        // Complex throws UnsupportedError
        final cArr = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
        expect(() => erf(cArr), throwsUnsupportedError);
      });
    });
  });

  group(
    '7. Floating Point Classification & Properties (isnan, isinf, isfinite, copysign, isClose, allClose)',
    () {
      test('isnan on real, complex, integer, boolean', () {
        NDArray.scope(() {
          final a64 = NDArray.fromList(
            [1.0, double.nan, 3.0, double.infinity],
            [4],
            DType.float64,
          );
          final a32 = NDArray.fromList(
            [1.0, double.nan, 3.0],
            [3],
            DType.float32,
          );
          final c128 = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(double.nan, 1.0),
              Complex(1.0, double.nan),
            ],
            [3],
            DType.complex128,
          );
          final i32 = NDArray.fromList([1, 2, 3], [3], DType.int32);

          final res64 = isnan(a64);
          expect(res64.dtype, DType.boolean);
          expect(res64.toList(), [false, true, false, false]);

          expect(isnan(a32).toList(), [false, true, false]);
          expect(isnan(c128).toList(), [false, true, true]);
          expect(isnan(i32).toList(), [false, false, false]);

          // Strided isnan
          final strided = a64.slice([const Slice(start: 0, stop: 4, step: 2)]);
          expect(isnan(strided).toList(), [false, false]);
        });
      });

      test('isinf on real, complex, integer', () {
        NDArray.scope(() {
          final a64 = NDArray.fromList(
            [1.0, double.infinity, double.negativeInfinity, double.nan],
            [4],
            DType.float64,
          );
          final c128 = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(double.infinity, 0.0),
              Complex(0.0, double.negativeInfinity),
              Complex(double.nan, 1.0),
            ],
            [4],
            DType.complex128,
          );
          final i32 = NDArray.fromList([1, 2], [2], DType.int32);

          expect(isinf(a64).toList(), [false, true, true, false]);
          expect(isinf(c128).toList(), [false, true, true, false]);
          expect(isinf(i32).toList(), [false, false]);

          // Strided isinf
          final strided = a64.slice([const Slice(start: 0, stop: 4, step: 2)]);
          expect(isinf(strided).toList(), [false, true]);
        });
      });

      test(
        'isfinite on real, complex, integer, boolean, float16, bfloat16',
        () {
          NDArray.scope(() {
            final a64 = NDArray.fromList(
              [1.0, double.infinity, double.negativeInfinity, double.nan, 0.0],
              [5],
              DType.float64,
            );
            final c128 = NDArray.fromList(
              [
                Complex(1.0, 2.0),
                Complex(double.infinity, 1.0),
                Complex(1.0, double.nan),
              ],
              [3],
              DType.complex128,
            );
            final i32 = NDArray.fromList([1, 2], [2], DType.int32);
            final b = NDArray.fromList([true, false], [2], DType.boolean);
            final f16 = NDArray.fromList([1.0, 2.0], [2], DType.float16);
            final bf16 = NDArray.fromList([1.0, 2.0], [2], DType.bfloat16);

            expect(isfinite(a64).toList(), [true, false, false, false, true]);
            expect(isfinite(c128).toList(), [true, false, false]);
            expect(isfinite(i32).toList(), [true, true]);
            expect(isfinite(b).toList(), [true, true]);
            expect(isfinite(f16).toList(), [true, true]);
            expect(isfinite(bf16).toList(), [true, true]);

            // Strided
            final strided = a64.slice([
              const Slice(start: 0, stop: 5, step: 2),
            ]);
            expect(isfinite(strided).toList(), [true, false, true]);
          });
        },
      );

      test('copysign values, broadcasting, strided, error handling', () {
        NDArray.scope(() {
          final x1 = NDArray.fromList(
            [1.0, -2.0, 3.0, -4.0],
            [4],
            DType.float64,
          );
          final x2 = NDArray.fromList(
            [-1.0, 1.0, -1.0, 1.0],
            [4],
            DType.float64,
          );
          final res = copysign(x1, x2);
          expect(res.toList(), [-1.0, 2.0, -3.0, 4.0]);

          // Float32
          final x1F32 = NDArray.fromList([1.0, -2.0], [2], DType.float32);
          final x2F32 = NDArray.fromList([-1.0, 1.0], [2], DType.float32);
          expect(copysign(x1F32, x2F32).toList(), [-1.0, 2.0]);

          // Int32
          final x1Int = NDArray.fromList([5, -10], [2], DType.int32);
          final x2Int = NDArray.fromList([-1, 1], [2], DType.int32);
          expect(copysign(x1Int, x2Int).toList(), [-5, 10]);

          // Broadcasting: [2, 1] and [1, 2] -> [2, 2]
          final m1 = NDArray.fromList([5.0, -5.0], [2, 1], DType.float64);
          final m2 = NDArray.fromList([-1.0, 1.0], [1, 2], DType.float64);
          final resBcast = copysign(m1, m2);
          expect(resBcast.shape, [2, 2]);
          expect(resBcast.getCell([0, 0]), -5.0);
          expect(resBcast.getCell([0, 1]), 5.0);
          expect(resBcast.getCell([1, 0]), -5.0);
          expect(resBcast.getCell([1, 1]), 5.0);

          // Complex throws UnsupportedError
          final cArr = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
          final rArr = NDArray.fromList([1.0], [1], DType.float64);
          expect(
            () => copysign(cArr.asAnySpec, rArr.asAnySpec),
            throwsUnsupportedError,
          );
          expect(
            () => copysign(rArr.asAnySpec, cArr.asAnySpec),
            throwsUnsupportedError,
          );
        });
      });

      test('isClose and allClose real, complex, tolerances, equalNan', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 1.00001, double.nan, double.infinity],
            [4],
            DType.float64,
          );
          final b = NDArray.fromList(
            [1.0, 1.000010001, double.nan, double.infinity],
            [4],
            DType.float64,
          );

          // Without equalNan
          final c1 = isClose(a, b, equalNan: false);
          expect(c1.toList(), [true, true, false, true]);
          expect(allClose(a, b, equalNan: false), isFalse);

          // With equalNan
          final c2 = isClose(a, b, equalNan: true);
          expect(c2.toList(), [true, true, true, true]);
          expect(allClose(a, b, equalNan: true), isTrue);

          // Complex isClose
          final cA = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(1.0, double.nan)],
            [2],
            DType.complex128,
          );
          final cB = NDArray.fromList(
            [Complex(1.0000001, 2.0000001), Complex(1.0, double.nan)],
            [2],
            DType.complex128,
          );
          expect(isClose(cA, cB, equalNan: true).toList(), [true, true]);
          expect(allClose(cA, cB, equalNan: true), isTrue);
        });
      });
    },
  );

  group('8. 0D Scalar Arrays & Multidimensional Contractions', () {
    test('0D Scalar operations', () {
      NDArray.scope(() {
        final s0 = NDArray.fromList([math.pi / 2], [], DType.float64);
        expect(s0.shape, isEmpty);
        expect(s0.rank, 0);

        final sinS0 = sin(s0);
        expect(sinS0.shape, isEmpty);
        expect(sinS0.getCell([]), closeTo(1.0, 1e-14));

        final cosS0 = cos(s0);
        expect(cosS0.shape, isEmpty);
        expect(cosS0.getCell([]), closeTo(0.0, 1e-14));

        final expS0 = exp(s0);
        expect(expS0.shape, isEmpty);
        expect(expS0.getCell([]), closeTo(math.exp(math.pi / 2), 1e-14));

        final logS0 = log(expS0);
        expect(logS0.shape, isEmpty);
        expect(logS0.getCell([]), closeTo(math.pi / 2, 1e-14));
      });
    });

    test('3D and 4D multidimensional tensor strided views', () {
      NDArray.scope(() {
        // [2, 3, 4] tensor
        final flatData = List<double>.generate(24, (i) => i * 0.1);
        final tensor3D = NDArray.fromList(flatData, [2, 3, 4], DType.float64);

        // Non-contiguous slice: tensor3D[:, 0:2, ::2] -> shape [2, 2, 2]
        final slice3D = tensor3D.slice([
          const Slice(),
          const Slice(start: 0, stop: 2),
          const Slice(start: 0, stop: 4, step: 2),
        ]);
        expect(slice3D.shape, [2, 2, 2]);
        expect(slice3D.isContiguous, isFalse);

        final sin3D = sin(slice3D);
        expect(sin3D.shape, [2, 2, 2]);
        expect(
          sin3D.getCell([0, 0, 0]),
          closeTo(math.sin(slice3D.getCell([0, 0, 0])), 1e-14),
        );
        expect(
          sin3D.getCell([1, 1, 1]),
          closeTo(math.sin(slice3D.getCell([1, 1, 1])), 1e-14),
        );

        final exp3D = exp(slice3D);
        expect(exp3D.shape, [2, 2, 2]);
        expect(
          exp3D.getCell([0, 0, 0]),
          closeTo(math.exp(slice3D.getCell([0, 0, 0])), 1e-14),
        );

        final i03D = i0(slice3D);
        expect(i03D.shape, [2, 2, 2]);

        final isfinite3D = isfinite(slice3D);
        expect(isfinite3D.shape, [2, 2, 2]);
        expect(isfinite3D.getCell([0, 0, 0]), isTrue);
      });
    });
  });

  group('9. Strided Angle Conversions & Out Buffer Reuse', () {
    test('deg2rad and rad2deg strided views and out destination', () {
      NDArray.scope(() {
        final deg = NDArray.fromList(
          [0.0, 45.0, 90.0, 135.0, 180.0],
          [5],
          DType.float64,
        );
        final stridedDeg = deg.slice([
          const Slice(start: 0, stop: 5, step: 2),
        ]); // [0.0, 90.0, 180.0]
        final outRad = NDArray.zeros([3], DType.float64);

        final resRad = deg2rad(stridedDeg, out: outRad);
        expect(identical(resRad, outRad), isTrue);
        expect(outRad.getCell([0]), closeTo(0.0, 1e-14));
        expect(outRad.getCell([1]), closeTo(math.pi / 2, 1e-14));
        expect(outRad.getCell([2]), closeTo(math.pi, 1e-14));

        final outDeg = NDArray.zeros([3], DType.float64);
        final resDeg = rad2deg(outRad, out: outDeg);
        expect(identical(resDeg, outDeg), isTrue);
        expect(outDeg.getCell([0]), closeTo(0.0, 1e-14));
        expect(outDeg.getCell([1]), closeTo(90.0, 1e-14));
        expect(outDeg.getCell([2]), closeTo(180.0, 1e-14));
      });
    });
  });

  group(
    '10. Binary Math Operations Error Handling & Broadcaster Validations',
    () {
      test('Incompatible shapes without broadcasting throw ArgumentError', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final b = NDArray.fromList([1.0, 2.0], [2], DType.float64);

          expect(() => atan2(a, b), throwsArgumentError);
          expect(() => hypot(a, b), throwsArgumentError);
          expect(() => logaddexp(a, b), throwsArgumentError);
          expect(() => logaddexp2(a, b), throwsArgumentError);
          expect(() => copysign(a, b), throwsArgumentError);
          expect(() => isClose(a, b), throwsArgumentError);
        });
      });

      test('Invalid where mask dtype throws ArgumentError', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final badWhere = NDArray.fromList([1.0, 0.0], [2], DType.float64);

          expect(() => sin(a, where: badWhere), throwsArgumentError);
          expect(() => cos(a, where: badWhere), throwsArgumentError);
          expect(() => exp(a, where: badWhere), throwsArgumentError);
          expect(() => log(a, where: badWhere), throwsArgumentError);
          expect(() => i0(a, where: badWhere), throwsArgumentError);
          expect(() => gamma(a, where: badWhere), throwsArgumentError);
          expect(() => erf(a, where: badWhere), throwsArgumentError);
        });
      });

      test('uint8 where masks work equivalently to boolean masks', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, math.pi / 2, math.pi],
            [3],
            DType.float64,
          );
          final outArr = NDArray.zeros([3], DType.float64);
          final u8Mask = NDArray.fromList([1, 0, 1], [3], DType.uint8);

          sin(a, where: u8Mask, out: outArr);
          expect(outArr.getCell([0]), closeTo(0.0, 1e-14));
          expect(outArr.getCell([1]), 0.0);
          expect(outArr.getCell([2]), closeTo(0.0, 1e-14));
        });
      });
    },
  );

  group('11. Special Functions Asymptotic and Extreme Regimes', () {
    test('i0 large complex asymptotic regime (|z| > 15)', () {
      NDArray.scope(() {
        final cLarge = NDArray.fromList(
          [Complex(20.0, 5.0), Complex(-25.0, 10.0)],
          [2],
          DType.complex128,
        );

        final res = i0(cLarge);
        expect(res.shape, [2]);
        expect(res.getCell([0]).real.isFinite, isTrue);
        expect(res.getCell([0]).imag.isFinite, isTrue);
        expect(res.getCell([1]).real.isFinite, isTrue);
        expect(res.getCell([1]).imag.isFinite, isTrue);
      });
    });

    test('gamma and erf extreme inputs', () {
      NDArray.scope(() {
        final gVals = NDArray.fromList(
          [0.0, -1.0, double.infinity],
          [3],
          DType.float64,
        );
        final gRes = gamma(gVals);
        expect(gRes.shape, [3]);

        final erfVals = NDArray.fromList(
          [double.infinity, double.negativeInfinity, double.nan],
          [3],
          DType.float64,
        );
        final erfRes = erf(erfVals);
        expect(erfRes.getCell([0]), 1.0);
        expect(erfRes.getCell([1]), -1.0);
        expect(erfRes.getCell([2]).isNaN, isTrue);
      });
    });
  });
}
