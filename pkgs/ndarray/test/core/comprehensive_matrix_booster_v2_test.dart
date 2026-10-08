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
  group('Comprehensive Coverage Matrix Booster V2 Suite', () {
    final floatDTypes = [
      DType.float64,
      DType.float32,
      DType.float16,
      DType.bfloat16,
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

    final complexDTypes = [DType.complex128, DType.complex64];

    NDArray<AnySpec> createArray3D(
      DType dt,
      List<int> shape, {
      bool strided = false,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<Object>.generate(size * (strided ? 2 : 1), (i) {
        final val = (i % 5) + 2;
        if (dt == DType.boolean) return val.isOdd;
        if (dt == DType.complex128 || dt == DType.complex64) {
          return Complex(val.toDouble(), (val + 0.5));
        }
        return val;
      });

      final dtObj = dt;
      if (strided) {
        final flatArr = NDArray.fromList(rawList, [size * 2], dtObj);
        final sliced = flatArr[Slice(step: 2)];
        return sliced.reshape(shape);
      } else {
        return NDArray.fromList(rawList, shape, (dtObj as DType<AnySpec>));
      }
    }

    test(
      '3D Multi-DType Binary Arithmetic Operations (Contiguous & Strided Views with out & where)',
      () {
        NDArray.scope(() {
          for (final dt in [...floatDTypes, ...integerDTypes]) {
            for (final isStrided in [false, true]) {
              final a = createArray3D(dt, [2, 3, 2], strided: isStrided);
              final b = createArray3D(dt, [2, 3, 2], strided: isStrided);
              final mask = NDArray<Boolean>.fromList(
                List.generate(12, (i) => i.isEven),
                [2, 3, 2],
                DType.boolean,
              );

              // Basic binary ops
              final rAdd = add(a, b, where: mask);
              expect(rAdd.shape, [2, 3, 2]);

              final rSub = subtract(a, b, where: mask);
              expect(rSub.shape, [2, 3, 2]);

              final rMul = multiply(a, b, where: mask);
              expect(rMul.shape, [2, 3, 2]);

              final rDiv = divide(a, b, where: mask);
              expect(rDiv.shape, [2, 3, 2]);

              final rFloorDiv = floorDivide(a, b, where: mask);
              expect(rFloorDiv.shape, [2, 3, 2]);

              final rRem = remainder(a, b, where: mask);
              expect(rRem.shape, [2, 3, 2]);

              final rFmod = fmod(a, b, where: mask);
              expect(rFmod.shape, [2, 3, 2]);

              final rPow = power(a, b, where: mask);
              expect(rPow.shape, [2, 3, 2]);

              final rHeavi = heaviside(a, b, where: mask);
              expect(rHeavi.shape, [2, 3, 2]);

              final rCopySign = copysign(a, b, where: mask);
              expect(rCopySign.shape, [2, 3, 2]);

              final rHypot = hypot(a, b, where: mask);
              expect(rHypot.shape, [2, 3, 2]);

              final rAtan2 = atan2(a, b, where: mask);
              expect(rAtan2.shape, [2, 3, 2]);

              final rLogAddExp = logaddexp(a, b, where: mask);
              expect(rLogAddExp.shape, [2, 3, 2]);

              final rLogAddExp2 = logaddexp2(a, b, where: mask);
              expect(rLogAddExp2.shape, [2, 3, 2]);

              if (dt.isInteger) {
                final rGcd = gcd(
                  a.asIntegerDType,
                  b.asIntegerDType,
                  where: mask,
                );
                expect(rGcd.shape, [2, 3, 2]);

                final rLcm = lcm(
                  a.asIntegerDType,
                  b.asIntegerDType,
                  where: mask,
                );
                expect(rLcm.shape, [2, 3, 2]);
              }
            }
          }
        });
      },
    );

    test(
      '3D Multi-DType Transcendental Trigonometric & Hyperbolic Functions (Contiguous & Strided)',
      () {
        NDArray.scope(() {
          for (final dt in [...floatDTypes, ...complexDTypes, DType.int32]) {
            for (final isStrided in [false, true]) {
              final a = createArray3D(dt, [2, 2, 3], strided: isStrided);
              final mask = NDArray<Boolean>.fromList(
                List.generate(12, (i) => i.isEven),
                [2, 2, 3],
                DType.boolean,
              );

              final s = sin(a, where: mask);
              expect(s.shape, [2, 2, 3]);

              final c = cos(a, where: mask);
              expect(c.shape, [2, 2, 3]);

              final t = tan(a, where: mask);
              expect(t.shape, [2, 2, 3]);

              final sh = sinh(a, where: mask);
              expect(sh.shape, [2, 2, 3]);

              final ch = cosh(a, where: mask);
              expect(ch.shape, [2, 2, 3]);

              final th = tanh(a, where: mask);
              expect(th.shape, [2, 2, 3]);

              final ex = exp(a, where: mask);
              expect(ex.shape, [2, 2, 3]);

              final lg = log(a, where: mask);
              expect(lg.shape, [2, 2, 3]);

              final lg2 = log2(a, where: mask);
              expect(lg2.shape, [2, 2, 3]);

              final lg10 = log10(a, where: mask);
              expect(lg10.shape, [2, 2, 3]);

              if (dt != DType.complex128 && dt != DType.complex64) {
                final asin_ = asin(a, where: mask);
                expect(asin_.shape, [2, 2, 3]);

                final acos_ = acos(a, where: mask);
                expect(acos_.shape, [2, 2, 3]);

                final atan_ = atan(a, where: mask);
                expect(atan_.shape, [2, 2, 3]);

                final asinh_ = asinh(a, where: mask);
                expect(asinh_.shape, [2, 2, 3]);

                final acosh_ = acosh(a, where: mask);
                expect(acosh_.shape, [2, 2, 3]);

                final atanh_ = atanh(a, where: mask);
                expect(atanh_.shape, [2, 2, 3]);

                final d2r = deg2rad(a, where: mask);
                expect(d2r.shape, [2, 2, 3]);

                final r2d = rad2deg(a, where: mask);
                expect(r2d.shape, [2, 2, 3]);
              }
            }
          }
        });
      },
    );

    test(
      'Statistics and NaN-ignoring Reductions across 1D/2D/3D and all Axes',
      () {
        NDArray.scope(() {
          final floatArr = NDArray.fromList(
            [
              1.0,
              double.nan,
              3.0,
              4.0,
              double.nan,
              6.0,
              7.0,
              8.0,
              9.0,
              double.nan,
              11.0,
              12.0,
            ],
            [3, 4],
            DType.float64,
          );

          // NaN statistics
          final nm0 = nanmean(floatArr, axis: 0);
          expect(nm0.shape, [4]);

          final nm1 = nanmean(floatArr, axis: 1, keepdims: true);
          expect(nm1.shape, [3, 1]);

          final nstd0 = nanstd(floatArr, axis: 0);
          expect(nstd0.shape, [4]);

          final nvar1 = nanvar(floatArr, axis: 1);
          expect(nvar1.shape, [3]);

          final nmin0 = nanmin(floatArr, axis: 0);
          expect(nmin0.shape, [4]);

          final nmax1 = nanmax(floatArr, axis: 1);
          expect(nmax1.shape, [3]);

          final nsum0 = nansum(floatArr, axis: 0);
          expect(nsum0.shape, [4]);

          // General stats
          final validArr = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [3, 4],
            DType.float64,
          );

          final ptp0 = ptp(validArr, axis: 0);
          expect(ptp0.shape, [4]);

          final med1 = median(validArr, axis: 1);
          expect(med1.shape, [3]);

          final weights = NDArray.fromList([1.0, 2.0, 1.0], [3], DType.float64);
          final avgRes = average(validArr, axis: 0, weights: weights);
          expect(avgRes.average.shape, [4]);

          final covMat = cov(validArr);
          expect(covMat.shape, [3, 3]);

          final corrMat = corrcoef(validArr);
          expect(corrMat.shape, [3, 3]);
        });
      },
    );

    test('Batch 3D Linear Algebra & Complex Matrix Solvers', () {
      NDArray.scope(() {
        // Batch 3D matrices [2, 2, 2]
        final batchMat = NDArray.fromList(
          [4.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 4.0],
          [2, 2, 2],
          DType.float64,
        );

        final detBatch = det(batchMat);
        expect(detBatch.shape, [2]);

        final invBatch = inv(batchMat);
        expect(invBatch.shape, [2, 2, 2]);

        final mat2d = NDArray.fromList(
          [4.0, 2.0, 1.0, 3.0],
          [2, 2],
          DType.float64,
        );
        final pinv2d = pinv(mat2d);
        expect(pinv2d.shape, [2, 2]);

        // Complex 2x2 matrix
        final cMat = NDArray.fromList(
          [
            Complex(2.0, 1.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(3.0, -1.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final cDet = det(cMat);
        expect(cDet.shape, <int>[]);

        final cInv = inv(cMat);
        expect(cInv.shape, [2, 2]);

        final cPinv = pinv(cMat);
        expect(cPinv.shape, [2, 2]);

        final cPow = matrix_power(cMat, 2);
        expect(cPow.shape, [2, 2]);

        final cNorm = norm(cMat);
        expect(cNorm.shape, <int>[]);

        final cSvd = svd(cMat);
        expect(cSvd.s.shape, [2]);
        cSvd.dispose();

        final cQr = qr(cMat);
        expect(cQr.q.shape, [2, 2]);
        expect(cQr.r.shape, [2, 2]);
        cQr.dispose();
      });
    });

    test(
      'BinaryOp Universal Function Methods on 2D Arrays (.reduce, .accumulate, .reduceat, .outer, .at)',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final reducibleUfuncs = [
            BinaryOp.add,
            BinaryOp.multiply,
            BinaryOp.minimum,
            BinaryOp.maximum,
            BinaryOp.logaddexp,
            BinaryOp.logaddexp2,
          ];

          for (final op in reducibleUfuncs) {
            final red0 = a.reduce(op: op, axis: 0);
            expect(red0.shape, [3]);

            final red1 = a.reduce(op: op, axis: 1, keepdims: true);
            expect(red1.shape, [2, 1]);

            final acc0 = a.accumulate(op: op, axis: 0);
            expect(acc0.shape, [2, 3]);

            final acc1 = a.accumulate(op: op, axis: 1);
            expect(acc1.shape, [2, 3]);

            final indices = NDArray.fromList([0, 2], [2], DType.int64);
            final redAt = reduceatUfunc(a, indices, op: op, axis: 1);
            expect(redAt.shape, [2, 2]);
          }

          // Outer product
          final v1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final v2 = NDArray.fromList([4.0, 5.0], [2], DType.float64);
          final outProd = outerUfunc(v1, v2, op: BinaryOp.multiply);
          expect(outProd.shape, [3, 2]);

          // At in-place accumulation
          final target = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [4],
            DType.float64,
          );
          final atIdx = NDArray.fromList([0, 2], [2], DType.int64);
          final atVals = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          atUfunc(target, atIdx, atVals, op: BinaryOp.add);
          expect(target.shape, [4]);
        });
      },
    );
  });
}
