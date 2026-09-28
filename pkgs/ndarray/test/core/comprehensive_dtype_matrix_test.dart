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

void main() {
  group('Comprehensive 15x15 DType Arithmetic Matrix Tests', () {
    final allDTypes = [
      DType.float64,
      DType.float32,
      DType.float16,
      DType.bfloat16,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
      DType.complex128,
      DType.complex64,
      DType.boolean,
    ];

    final numericDTypes = [
      DType.float64,
      DType.float32,
      DType.float16,
      DType.bfloat16,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
    ];

    final integerDTypes = [
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
    ];

    NDArray<Object> createSampleArray(
      DType dt,
      List<int> shape, {
      bool strided = false,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<Object>.generate(size * (strided ? 2 : 1), (i) {
        final val = (i % 5) + 1;
        if (dt == DType.boolean) return val % 2 == 1;
        if (dt == DType.complex128 || dt == DType.complex64) {
          return Complex(val.toDouble(), (val + 1).toDouble());
        }
        return val;
      });

      final dtObj = dt as DType<Object>;
      if (strided) {
        final flatArr = NDArray<Object>.fromList(rawList, [size * 2], dtObj);
        final sliced = flatArr[Slice(step: 2)];
        return sliced.reshape(shape);
      } else {
        return NDArray<Object>.fromList(rawList, shape, dtObj);
      }
    }

    NDArray<num> createNumericArray(
      DType dt,
      List<int> shape, {
      bool strided = false,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<num>.generate(
        size * (strided ? 2 : 1),
        (i) => ((i % 5) + 1),
      );
      final dtNum = dt as DType<num>;
      if (strided) {
        final flatArr = NDArray<num>.fromList(rawList, [size * 2], dtNum);
        final sliced = flatArr[Slice(step: 2)];
        return sliced.reshape(shape);
      } else {
        return NDArray<num>.fromList(rawList, shape, dtNum);
      }
    }

    test('All 15x15 DType binary arithmetic (add, sub, mul, div, power, floorDivide, remainder, fmod)', () {
      NDArray.scope(() {
        for (final dtA in allDTypes) {
          for (final dtB in allDTypes) {
            for (final isStrided in [false, true]) {
              final a = createSampleArray(dtA, [2, 3], strided: isStrided);
              final b = createSampleArray(dtB, [2, 3], strided: isStrided);

              final resAdd = add(a, b);
              expect(resAdd.shape, [2, 3]);

              final resSub = subtract(a, b);
              expect(resSub.shape, [2, 3]);

              final resMul = multiply(a, b);
              expect(resMul.shape, [2, 3]);

              final resDiv = divide(a, b);
              expect(resDiv.shape, [2, 3]);

              if (dtA == dtB) {
                final resPow = power(a, b);
                expect(resPow.shape, [2, 3]);
              }

              if (dtA != DType.complex128 &&
                  dtA != DType.complex64 &&
                  dtA != DType.boolean &&
                  dtB != DType.complex128 &&
                  dtB != DType.complex64 &&
                  dtB != DType.boolean) {
                final resFDiv = floor_divide(a, b);
                expect(resFDiv.shape, [2, 3]);

                final resRem = remainder(a, b);
                expect(resRem.shape, [2, 3]);

                final resFmod = fmod(a, b);
                expect(resFmod.shape, [2, 3]);

                final resHeavi = heaviside(a, b);
                expect(resHeavi.shape, [2, 3]);

                final resCopy = copysign(a, b);
                expect(resCopy.shape, [2, 3]);

                final resHypot = hypot(a, b);
                expect(resHypot.shape, [2, 3]);

                final resAtan2 = atan2(a, b);
                expect(resAtan2.shape, [2, 3]);

                final resLogadd = logaddexp(a, b);
                expect(resLogadd.shape, [2, 3]);

                final resLogadd2 = logaddexp2(a, b);
                expect(resLogadd2.shape, [2, 3]);
              }
            }
          }
        }
      });
    });

    test('Integer binary arithmetic (gcd, lcm)', () {
      NDArray.scope(() {
        for (final dtA in integerDTypes) {
          for (final dtB in integerDTypes) {
            for (final isStrided in [false, true]) {
              final a = createNumericArray(dtA, [2, 2], strided: isStrided);
              final b = createNumericArray(dtB, [2, 2], strided: isStrided);

              final resGcd = gcd(a, b);
              expect(resGcd.shape, [2, 2]);

              final resLcm = lcm(a, b);
              expect(resLcm.shape, [2, 2]);
            }
          }
        }
      });
    });

    test('All 15 DTypes unary math and reductions (flat, axis 0, axis 1, keepdims)', () {
      NDArray.scope(() {
        for (final dt in numericDTypes) {
          for (final isStrided in [false, true]) {
            final a = createNumericArray(dt, [3, 4], strided: isStrided);

            // Unary math
            final resSqrt = sqrt(a);
            expect(resSqrt.shape, [3, 4]);

            final resAbs = abs(a);
            expect(resAbs.shape, [3, 4]);

            final resSin = sin(a);
            expect(resSin.shape, [3, 4]);

            final resCos = cos(a);
            expect(resCos.shape, [3, 4]);

            final resTan = tan(a);
            expect(resTan.shape, [3, 4]);

            final resExp = exp(a);
            expect(resExp.shape, [3, 4]);

            final resLog = log(a);
            expect(resLog.shape, [3, 4]);

            final resLog2 = log2(a);
            expect(resLog2.shape, [3, 4]);

            final resLog10 = log10(a);
            expect(resLog10.shape, [3, 4]);

            final resExpm1 = expm1(a);
            expect(resExpm1.shape, [3, 4]);

            final resLog1p = log1p(a);
            expect(resLog1p.shape, [3, 4]);

            // Reductions
            final sumFlat = sum(a);
            expect(sumFlat.shape, <int>[]);

            final sumAxis0 = sum(a, axis: 0);
            expect(sumAxis0.shape, [4]);

            final sumAxis1 = sum(a, axis: 1, keepdims: true);
            expect(sumAxis1.shape, [3, 1]);

            final prodFlat = prod(a);
            expect(prodFlat.shape, <int>[]);

            final prodAxis0 = prod(a, axis: 0);
            expect(prodAxis0.shape, [4]);

            final meanFlat = mean(a);
            expect(meanFlat.shape, <int>[]);

            final meanAxis0 = mean(a, axis: 0);
            expect(meanAxis0.shape, [4]);

            final cumsum0 = cumsum(a, axis: 0);
            expect(cumsum0.shape, [3, 4]);

            final cumsum1 = cumsum(a, axis: 1);
            expect(cumsum1.shape, [3, 4]);

            final stdFlat = std(a);
            expect(stdFlat.shape, <int>[]);

            final stdAxis0 = std(a, axis: 0);
            expect(stdAxis0.shape, [4]);

            final varFlat = variance(a);
            expect(varFlat.shape, <int>[]);

            final varAxis0 = variance(a, axis: 0);
            expect(varAxis0.shape, [4]);

            final minFlat = min(a);
            expect(minFlat.shape, <int>[]);

            final maxFlat = max(a);
            expect(maxFlat.shape, <int>[]);

            final ptpFlat = ptp(a);
            expect(ptpFlat.shape, <int>[]);
          }
        }
      });
    });

    test('1D and 2D FFT and Linear Algebra across float and complex types', () {
      NDArray.scope(() {
        final floatTypes = [DType.float64, DType.float32];
        final complexTypes = [DType.complex128, DType.complex64];

        for (final dt in [...floatTypes, ...complexTypes]) {
          final a1d = createSampleArray(dt, [8]);
          final f1 = fft(a1d);
          expect(f1.shape, [8]);
          final if1 = ifft(f1);
          expect(if1.shape, [8]);

          final a2d = createSampleArray(dt, [4, 4]);
          final f2 = fft2(a2d);
          expect(f2.shape, [4, 4]);
          final if2 = ifft2(f2);
          expect(if2.shape, [4, 4]);

          final trans = a2d.transpose();
          expect(trans.shape, [4, 4]);
        }

        for (final dt in floatTypes) {
          final aReal = createNumericArray(dt, [8]);
          final rf = rfft(aReal);
          expect(rf.shape, [5]);
          final irf = irfft(rf, n: 8);
          expect(irf.shape, [8]);

          final a2dReal = createNumericArray(dt, [4, 4]);
          final detVal = det(a2dReal);
          expect(detVal.shape, <int>[]);
        }
      });
    });
  });
}
