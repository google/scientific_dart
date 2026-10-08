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
  group('Comprehensive Statistics, Reductions, NaN Operations & Binning', () {
    // =========================================================================
    // 1. REDUCTIONS & DISPERSION METRICS
    // =========================================================================
    group('1. Reductions and Dispersion Metrics', () {
      group('ptp (Peak-to-Peak Range)', () {
        test('1D float64 ptp flat contiguous and strided', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [3.0, 1.0, 9.0, 2.0, 7.0],
              [5],
              DType.float64,
            );
            final res = ptp(a);
            expect(res.shape, <int>[]);
            expect(res.dtype, DType.float64);
            expect(res.scalar, 8.0);

            // Strided view (step: 2) -> [3.0, 9.0, 7.0] -> max 9.0, min 3.0 -> ptp 6.0
            final strided = a.slice([Slice(start: 0, stop: 5, step: 2)]);
            expect(strided.isContiguous, false);
            final resStrided = ptp(strided);
            expect(resStrided.scalar, 6.0);
          });
        });

        test('2D ptp across axis 0, axis 1, and negative axis', () {
          NDArray.scope(() {
            // [[10, 2, 8],
            //  [ 4, 9, 1]]
            final a = NDArray.fromList(
              [10.0, 2.0, 8.0, 4.0, 9.0, 1.0],
              [2, 3],
              DType.float64,
            );
            final p0 = ptp(a, axis: 0);
            expect(p0.shape, [3]);
            expect(p0.toList(), [6.0, 7.0, 7.0]); // (10-4), (9-2), (8-1)

            final p1 = ptp(a, axis: 1);
            expect(p1.shape, [2]);
            expect(p1.toList(), [8.0, 8.0]); // (10-2), (9-1)

            final pNeg = ptp(a, axis: -1);
            expect(pNeg.shape, [2]);
            expect(pNeg.toList(), [8.0, 8.0]);

            final pNeg2 = ptp(a, axis: -2);
            expect(pNeg2.shape, [3]);
            expect(pNeg2.toList(), [6.0, 7.0, 7.0]);
          });
        });

        test('3D ptp across all axes with non-contiguous transpose', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              List<double>.generate(24, (i) => i.toDouble()),
              [2, 3, 4],
              DType.float64,
            );
            final p0 = ptp(a, axis: 0);
            expect(p0.shape, [3, 4]);
            for (final val in p0.toList()) {
              expect(val, 12.0);
            }

            final p1 = ptp(a, axis: 1);
            expect(p1.shape, [2, 4]);
            for (final val in p1.toList()) {
              expect(val, 8.0);
            }

            final p2 = ptp(a, axis: 2);
            expect(p2.shape, [2, 3]);
            for (final val in p2.toList()) {
              expect(val, 3.0);
            }

            // Transposed 3D array
            final aT = a.transpose([2, 0, 1]);
            final pT = ptp(aT, axis: 0);
            expect(pT.shape, [2, 3]);
            for (final val in pT.toList()) {
              expect(val, 3.0);
            }
          });
        });

        test('ptp out buffer reuse', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [5.0, 1.0, 8.0, 2.0],
              [2, 2],
              DType.float64,
            );
            final out = NDArray<Float64>.zeros([2], DType.float64);
            final res = ptp(a, axis: 0, out: out);
            expect(identical(res, out), true);
            expect(out.toList(), [3.0, 1.0]); // (5-2), (8-1)
          });
        });

        test('ptp across all numeric and integer DTypes', () {
          NDArray.scope(() {
            final dtypes = [
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

            for (final dt in dtypes) {
              final a = NDArray.fromList([2, 10, 5, 1], [4], dt);
              final p = ptp(a);
              expect(p.dtype, dt);
              expect(p.scalar, 9);
            }
          });
        });

        test('ptp error and edge validations', () {
          NDArray.scope(() {
            final empty = NDArray<Float64>.zeros([0], DType.float64);
            expect(() => ptp(empty), throwsArgumentError);

            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            expect(() => ptp(a, axis: 5), throwsArgumentError);
            expect(() => ptp(a, axis: -3), throwsArgumentError);

            final wrongOut = NDArray<Float64>.zeros([5], DType.float64);
            expect(() => ptp(a, axis: 0, out: wrongOut), throwsArgumentError);

            a.dispose();
            expect(() => ptp(a), throwsStateError);
          });
        });
      });

      group('std and var_ / variance', () {
        test('std and var_ 1D, 2D, 3D ddof=0 and ddof=1', () {
          NDArray.scope(() {
            // [1, 2, 3, 4] -> mean = 2.5
            // population var (ddof=0): ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
            // sample var (ddof=1): 5.0 / 3 = 1.6666666666666667
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );

            final v0 = var_(a, ddof: 0);
            expect(v0.scalar, closeTo(1.25, 1e-9));
            final s0 = std(a, ddof: 0);
            expect(s0.scalar, closeTo(math.sqrt(1.25), 1e-9));

            final v1 = var_(a, ddof: 1);
            expect(v1.scalar, closeTo(5.0 / 3.0, 1e-9));
            final s1 = std(a, ddof: 1);
            expect(s1.scalar, closeTo(math.sqrt(5.0 / 3.0), 1e-9));

            // variance alias check
            final vAlias = variance(a, ddof: 1);
            expect(vAlias.scalar, closeTo(5.0 / 3.0, 1e-9));

            // 2D keepdims
            final a2 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final v2D = var_(a2, axis: 1, keepdims: true);
            expect(v2D.shape, [2, 1]);
            expect(v2D.toList()[0], closeTo(2.0 / 3.0, 1e-9));
            expect(v2D.toList()[1], closeTo(2.0 / 3.0, 1e-9));

            final s2D = std(a2, axis: 1, keepdims: false);
            expect(s2D.shape, [2]);
            expect(s2D.toList()[0], closeTo(math.sqrt(2.0 / 3.0), 1e-9));
          });
        });

        test('std and var_ out buffer reuse', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final outVar = NDArray<Float64>.zeros([2], DType.float64);
            final resVar = var_(a, axis: 0, out: outVar);
            expect(identical(resVar, outVar), true);

            final outStd = NDArray<Float64>.zeros([2], DType.float64);
            final resStd = std(a, axis: 0, out: outStd);
            expect(identical(resStd, outStd), true);
          });
        });

        test('std and var_ with ddof >= N resulting in Inf or NaN', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            final v = var_(a, ddof: 3);
            expect(v.scalar, double.infinity);
            final s = std(a, ddof: 4);
            expect(s.scalar, double.infinity);

            final constArr = NDArray.fromList([5.0, 5.0], [2], DType.float64);
            expect(var_(constArr, ddof: 2).scalar, isNaN);
            expect(std(constArr, ddof: 2).scalar, isNaN);
          });
        });

        test('std across multiple integer and float DTypes', () {
          NDArray.scope(() {
            final dtypes = [
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

            for (final dt in dtypes) {
              final a = NDArray.fromList([10, 20, 30], [3], dt);
              final s = std(a);
              expect(s.dtype, DType.float64);
              expect(s.scalar, closeTo(math.sqrt(200.0 / 3.0), 1e-3));
            }
          });
        });
      });

      group('mean', () {
        test('mean 1D, 2D, 3D across axes, keepdims, and strided arrays', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final mAll = mean(a);
            expect(mAll.shape, <int>[]);
            expect(mAll.scalar, 3.5);

            final m0 = mean(a, axis: 0);
            expect(m0.shape, [3]);
            expect(m0.toList(), [2.5, 3.5, 4.5]);

            final m1Keep = mean(a, axis: 1, keepdims: true);
            expect(m1Keep.shape, [2, 1]);
            expect(m1Keep.toList(), [2.0, 5.0]);

            // Strided transposed
            final aT = a.transpose();
            final mT = mean(aT, axis: 0);
            expect(mT.toList(), [2.0, 5.0]);
          });
        });

        test('mean complex128 and complex64 arrays', () {
          NDArray.scope(() {
            final c128 = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(3.0, 4.0), Complex(5.0, 6.0)],
              [3],
              DType.complex128,
            );
            final mC128 = mean(c128);
            expect(mC128.dtype, DType.complex128);
            expect(mC128.scalar.real, closeTo(3.0, 1e-9));
            expect(mC128.scalar.imag, closeTo(4.0, 1e-9));

            final c64 = NDArray.fromList(
              [Complex(2.0, -1.0), Complex(4.0, 3.0)],
              [2],
              DType.complex64,
            );
            final mC64 = mean(c64);
            expect(mC64.dtype, DType.complex128);
            expect(mC64.scalar.real, closeTo(3.0, 1e-6));
            expect(mC64.scalar.imag, closeTo(1.0, 1e-6));
          });
        });

        test('mean out parameter buffer reuse', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final out = NDArray<Float64>.zeros([2], DType.float64);
            final res = mean(a, axis: 0, out: out);
            expect(identical(res, out), true);
            expect(out.toList(), [2.0, 3.0]);
          });
        });
      });

      group('median', () {
        test('median 1D odd and even lengths', () {
          NDArray.scope(() {
            final odd = NDArray.fromList([7.0, 1.0, 5.0], [3], DType.float64);
            final mOdd = median(odd);
            expect(mOdd.scalar, 5.0);

            final even = NDArray.fromList(
              [7.0, 1.0, 5.0, 3.0],
              [4],
              DType.float64,
            );
            final mEven = median(even);
            expect(mEven.scalar, 4.0);
          });
        });

        test('median 2D and 3D across axes with keepdims', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final m0 = median(a, axis: 0);
            expect(m0.shape, [3]);
            expect(m0.toList(), [2.5, 3.5, 4.5]);

            final m1Keep = median(a, axis: 1, keepdims: true);
            expect(m1Keep.shape, [2, 1]);
            expect(m1Keep.toList(), [3.0, 4.0]);
          });
        });

        test('median across all 15 DTypes', () {
          NDArray.scope(() {
            final dtypes = <DType<AnySpec>>[
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

            for (final dt in dtypes) {
              final a = NDArray.fromList([10, 30, 20], [3], dt);
              final m = median(a);
              expect(
                m.dtype,
                dt == DType.float32 ? DType.float32 : DType.float64,
              );
              expect(m.scalar, 20.0);
            }

            // boolean
            final b = NDArray.fromList([true, false, true], [3], DType.boolean);
            final mb = median(b);
            expect(mb.dtype, DType.float64);
            expect(mb.scalar, 1.0);

            // complex128 and complex64
            final c128 = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(3.0, 4.0), Complex(5.0, 6.0)],
              [3],
              DType.complex128,
            );
            final mc128 = median(c128);
            expect(mc128.dtype, DType.complex128);
            expect(mc128.scalar.real, 3.0);

            final c64 = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(3.0, 4.0), Complex(5.0, 6.0)],
              [3],
              DType.complex64,
            );
            final mc64 = median(c64);
            expect(mc64.dtype, DType.complex64);
            expect(mc64.scalar.real, 3.0);
          });
        });
      });

      group('percentile and quantile', () {
        test('percentile and quantile scalar q values', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [15.0, 20.0, 35.0, 40.0, 50.0],
              [5],
              DType.float64,
            );
            final q0 = quantile(a, 0.0);
            expect(q0.scalar, closeTo(15.0, 1e-9));

            final q50 = quantile(a, 0.5);
            expect(q50.scalar, closeTo(35.0, 1e-9));

            final q100 = quantile(a, 1.0);
            expect(q100.scalar, closeTo(50.0, 1e-9));

            final p0 = percentile(a, 0.0);
            expect(p0.scalar, closeTo(15.0, 1e-9));

            final p50 = percentile(a, 50.0);
            expect(p50.scalar, closeTo(35.0, 1e-9));

            final p100 = percentile(a, 100.0);
            expect(p100.scalar, closeTo(50.0, 1e-9));
          });
        });

        test('all 13 QuantileMethod variants', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final methods = [
              QuantileMethod.linear,
              QuantileMethod.lower,
              QuantileMethod.higher,
              QuantileMethod.midpoint,
              QuantileMethod.nearest,
              QuantileMethod.invertedCdf,
              QuantileMethod.averagedInvertedCdf,
              QuantileMethod.closestObservation,
              QuantileMethod.interpolatedInvertedCdf,
              QuantileMethod.hazen,
              QuantileMethod.weibull,
              QuantileMethod.medianUnbiased,
              QuantileMethod.normalUnbiased,
            ];

            for (final m in methods) {
              final q = quantile(a, 0.4, method: m);
              expect(q.shape, <int>[]);
              expect(q.scalar, greaterThanOrEqualTo(10.0));
              expect(q.scalar, lessThanOrEqualTo(40.0));
            }
          });
        });

        test('percentile 2D across axes with keepdims and out buffer', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final p0 = percentile(a, 50.0, axis: 0);
            expect(p0.shape, [3]);
            expect(p0.toList(), [2.5, 3.5, 4.5]);

            final p1Keep = percentile(a, 50.0, axis: 1, keepdims: true);
            expect(p1Keep.shape, [2, 1]);
            expect(p1Keep.toList(), [3.0, 4.0]);

            final out = NDArray<Float64>.zeros([3], DType.float64);
            final res = percentile(a, 50.0, axis: 0, out: out);
            expect(identical(res, out), true);
            expect(out.toList(), [2.5, 3.5, 4.5]);
          });
        });

        test('percentile and quantile validation errors', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            expect(() => quantile(a, -0.1), throwsArgumentError);
            expect(() => quantile(a, 1.1), throwsArgumentError);
            expect(() => percentile(a, -1.0), throwsArgumentError);
            expect(() => percentile(a, 101.0), throwsArgumentError);
          });
        });
      });

      group('average', () {
        test('average 1D weighted and unweighted with returned: true', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final w = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );

            final res = average(a, weights: w, returned: true);
            expect(res.average.scalar, closeTo(3.0, 1e-9));
            expect(res.sumOfWeights?.scalar, closeTo(10.0, 1e-9));

            final resUnweighted = average(a, returned: true);
            expect(resUnweighted.average.scalar, closeTo(2.5, 1e-9));
            expect(resUnweighted.sumOfWeights?.scalar, closeTo(4.0, 1e-9));
          });
        });

        test('average 2D across axes with 1D and 2D weights', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final w1D = NDArray.fromList([1.0, 3.0], [2], DType.float64);

            final res0 = average(a, axis: 0, weights: w1D);
            expect(res0.average.shape, [2]);
            expect(res0.average.toList(), [2.5, 3.5]);

            final res1 = average(a, axis: 1, weights: w1D);
            expect(res1.average.shape, [2]);
            expect(res1.average.toList(), [1.75, 3.75]);
          });
        });

        test('average type promotion (int data, float weights)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
            final w = NDArray.fromList(
              [1.0, 1.0, 1.0, 1.0],
              [4],
              DType.float64,
            );
            final res = average(a, weights: w);
            expect(res.average.dtype, DType.float64);
            expect(res.average.scalar, closeTo(2.5, 1e-9));
          });
        });

        test('average out buffer and error validation', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final out = NDArray<Float64>.zeros([], DType.float64);
            final res = average(a, out: out);
            expect(identical(res.average, out), true);
            expect(out.scalar, 1.5);

            final mismatchedW = NDArray.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            expect(() => average(a, weights: mismatchedW), throwsArgumentError);
          });
        });
      });

      group('sum and prod across all 15 DTypes', () {
        test('sum and prod 1D, 2D, 3D, keepdims, and empty arrays', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final sAll = sum(a);
            expect(sAll.scalar, 21.0);

            final pAll = prod(a);
            expect(pAll.scalar, 720.0);

            final s0Keep = sum(a, axis: 0, keepdims: true);
            expect(s0Keep.shape, [1, 3]);
            expect(s0Keep.toList(), [5.0, 7.0, 9.0]);

            final p1Keep = prod(a, axis: 1, keepdims: true);
            expect(p1Keep.shape, [2, 1]);
            expect(p1Keep.toList(), [6.0, 120.0]);

            final empty = NDArray<Float64>.zeros([0], DType.float64);
            expect(sum(empty).scalar, 0.0);
            expect(prod(empty).scalar, 1.0);

            final emptyComplex = NDArray.zeros([0], DType.complex128);
            expect(sum(emptyComplex).scalar, Complex(0.0, 0.0));
            expect(prod(emptyComplex).scalar, Complex(1.0, 0.0));
          });
        });

        test('sum and prod across all 15 DTypes', () {
          NDArray.scope(() {
            final dtypes = <DType<AnySpec>>[
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

            for (final dt in dtypes) {
              final a = NDArray.fromList([2, 3, 4], [3], dt);
              expect(sum(a).scalar, 9);
              expect(prod(a).scalar, 24);
            }

            final b = NDArray.fromList([true, false, true], [3], DType.boolean);
            expect(sum(b).scalar, 2);
            expect(prod(b).scalar, 0);
            expect(sumAs(b, DType.int64).scalar, 2);
            expect(prodAs(b, DType.int64).scalar, 0);

            final c128 = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(3.0, 4.0)],
              [2],
              DType.complex128,
            );
            final sC = sum(c128);
            expect(sC.scalar, Complex(4.0, 6.0));
            final pC = prod(c128);
            expect(pC.scalar.real, closeTo(-5.0, 1e-9));
            expect(pC.scalar.imag, closeTo(10.0, 1e-9));
          });
        });
      });

      group('cumsum, cumprod, cummin, cummax', () {
        test('cumsum and cumprod 1D and 2D with out buffer', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final cs = cumsum(a);
            expect(cs.toList(), [1.0, 3.0, 6.0, 10.0]);

            final cp = cumprod(a);
            expect(cp.toList(), [1.0, 2.0, 6.0, 24.0]);

            final a2 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final cs0 = cumsum(a2, axis: 0);
            expect(cs0.toList(), [1.0, 2.0, 4.0, 6.0]);

            final cs1 = cumsum(a2, axis: 1);
            expect(cs1.toList(), [1.0, 3.0, 3.0, 7.0]);

            final out = NDArray<Float64>.zeros([2, 2], DType.float64);
            final res = cumsum(a2, axis: 0, out: out);
            expect(identical(res, out), true);
            expect(out.toList(), [1.0, 2.0, 4.0, 6.0]);
          });
        });

        test('cummin and cummax 1D and 2D', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [3.0, 1.0, 4.0, 2.0],
              [4],
              DType.float64,
            );
            final cmn = cummin(a);
            expect(cmn.toList(), [3.0, 1.0, 1.0, 1.0]);

            final cmx = cummax(a);
            expect(cmx.toList(), [3.0, 3.0, 4.0, 4.0]);
          });
        });
      });

      group('all and any', () {
        test('all and any 1D, 2D, axis=null, and keepdims', () {
          NDArray.scope(() {
            final a = NDArray.fromList([true, true, false], [3], DType.boolean);
            expect(all(a).scalar, false);
            expect(any(a).scalar, true);

            final allTrue = NDArray.fromList([true, true], [2], DType.boolean);
            expect(all(allTrue).scalar, true);

            final allFalse = NDArray.fromList(
              [false, false],
              [2],
              DType.boolean,
            );
            expect(any(allFalse).scalar, false);

            final a2 = NDArray.fromList(
              [true, true, true, false],
              [2, 2],
              DType.boolean,
            );
            final all0 = all(a2, axis: 0);
            expect(all0.toList(), [true, false]);

            final any1Keep = any(a2, axis: 1, keepdims: true);
            expect(any1Keep.shape, [2, 1]);
            expect(any1Keep.toList(), [true, true]);

            final out = NDArray<Boolean>.zeros([2], DType.boolean);
            final res = all(a2, axis: 0, out: out);
            expect(identical(res, out), true);
            expect(out.toList(), [true, false]);
          });
        });
      });
    });

    // =========================================================================
    // 2. COMPLETE SUITE OF NAN-IGNORING STATISTICS
    // =========================================================================
    group('2. NaN-Ignoring Statistics Suite', () {
      group('nanmean', () {
        test('1D, 2D, 3D nanmean across axes and axis=null', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, double.nan, 3.0, 5.0],
              [4],
              DType.float64,
            );
            final mAll = nanmean(a);
            expect(mAll.scalar, closeTo(3.0, 1e-9));

            final a2 = NDArray.fromList(
              [1.0, double.nan, 3.0, double.nan, 4.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final m0 = nanmean(a2, axis: 0);
            expect(m0.shape, [3]);
            expect(m0.toList(), [1.0, 4.0, 4.5]);

            final m1 = nanmean(a2, axis: 1);
            expect(m1.shape, [2]);
            expect(m1.toList(), [2.0, 5.0]);

            final mKeep = nanmean(a2, axis: 1, keepdims: true);
            expect(mKeep.shape, [2, 1]);
            expect(mKeep.toList(), [2.0, 5.0]);
          });
        });

        test('nanmean all-NaN slices return NaN', () {
          NDArray.scope(() {
            final allNan = NDArray.fromList(
              [double.nan, double.nan],
              [2],
              DType.float64,
            );
            final m = nanmean(allNan);
            expect((m.scalar as num).isNaN, true);

            final a2 = NDArray.fromList(
              [double.nan, 2.0, double.nan, 4.0],
              [2, 2],
              DType.float64,
            );
            final m0 = nanmean(a2, axis: 0);
            expect((m0.toList()[0] as num).isNaN, true);
            expect(m0.toList()[1], 3.0);
          });
        });

        test('nanmean complex arrays', () {
          NDArray.scope(() {
            final c = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(double.nan, 4.0), Complex(3.0, 6.0)],
              [3],
              DType.complex128,
            );
            final m = nanmean<DTypeTag>(c);
            expect(m.dtype, DType.complex128);
            expect(m.scalar.real, closeTo(2.0, 1e-9));
            expect(m.scalar.imag, closeTo(4.0, 1e-9));
          });
        });

        test('nanmean out buffer and strided views', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 99.0, double.nan, 99.0, 5.0, 99.0],
              [6],
              DType.float64,
            );
            final strided = a.slice([Slice(start: 0, stop: 6, step: 2)]);
            final out = NDArray<Float64>.zeros([], DType.float64);
            final res = nanmean<Float64>(strided, out: out);
            expect(identical(res, out), true);
            expect(out.scalar, 3.0);
          });
        });
      });

      group('nanvar and nanstd', () {
        test('nanvar and nanstd 1D and 2D', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, double.nan, 3.0, 5.0],
              [4],
              DType.float64,
            );

            final v0 = nanvar(a);
            expect(v0.scalar, closeTo(8.0 / 3.0, 1e-9));

            final s0 = nanstd(a);
            expect(s0.scalar, closeTo(math.sqrt(8.0 / 3.0), 1e-9));

            final a2 = NDArray.fromList(
              [1.0, double.nan, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final v2D = nanvar(a2, axis: 0);
            expect(v2D.shape, [2]);
            expect(v2D.toList()[0], closeTo(1.0, 1e-9));
            expect(v2D.toList()[1], closeTo(0.0, 1e-9));
          });
        });

        test('nanvar and nanstd with all NaNs', () {
          NDArray.scope(() {
            final allNan = NDArray.fromList(
              [double.nan, double.nan],
              [2],
              DType.float64,
            );
            expect(nanvar(allNan).scalar.isNaN, true);
            expect(nanstd(allNan).scalar.isNaN, true);
          });
        });

        test('nanvar and nanstd out buffer reuse', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, double.nan, 3.0],
              [3],
              DType.float64,
            );
            final outVar = NDArray<Float64>.zeros([], DType.float64);
            final resVar = nanvar(a, out: outVar);
            expect(identical(resVar, outVar), true);
            expect(outVar.scalar, 1.0);

            final outStd = NDArray<Float64>.zeros([], DType.float64);
            final resStd = nanstd(a, out: outStd);
            expect(identical(resStd, outStd), true);
            expect(outStd.scalar, 1.0);
          });
        });
      });

      group('nanmin and nanmax', () {
        test('nanmin and nanmax 1D, 2D, 3D, and keepdims', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [double.nan, 5.0, 2.0, double.nan, 8.0],
              [5],
              DType.float64,
            );
            expect(nanmin(a).scalar, 2.0);
            expect(nanmax(a).scalar, 8.0);

            final a2 = NDArray.fromList(
              [double.nan, 10.0, 3.0, 4.0, double.nan, 7.0],
              [2, 3],
              DType.float64,
            );
            final min0 = nanmin(a2, axis: 0);
            expect(min0.shape, [3]);
            expect(min0.toList(), [4.0, 10.0, 3.0]);

            final max0 = nanmax(a2, axis: 0);
            expect(max0.shape, [3]);
            expect(max0.toList(), [4.0, 10.0, 7.0]);

            final min1Keep = nanmin(a2, axis: 1, keepdims: true);
            expect(min1Keep.shape, [2, 1]);
            expect(min1Keep.toList(), [3.0, 4.0]);

            final max1Keep = nanmax(a2, axis: 1, keepdims: true);
            expect(max1Keep.shape, [2, 1]);
            expect(max1Keep.toList(), [10.0, 7.0]);
          });
        });

        test(
          'nanmin and nanmax all-NaN handling and unsupported complex error',
          () {
            NDArray.scope(() {
              final allNan = NDArray.fromList(
                [double.nan, double.nan],
                [2],
                DType.float64,
              );
              expect(nanmin(allNan).scalar.isNaN, true);
              expect(nanmax(allNan).scalar.isNaN, true);

              final c = NDArray.fromList(
                [Complex(1.0, 2.0)],
                [1],
                DType.complex128,
              );
              expect(() => nanmin(c), throwsUnsupportedError);
              expect(() => nanmax(c), throwsUnsupportedError);

              final empty = NDArray<Float64>.zeros([0], DType.float64);
              expect(() => nanmin(empty), throwsArgumentError);
              expect(() => nanmax(empty), throwsArgumentError);
            });
          },
        );

        test('nanmin and nanmax across multiple numeric DTypes', () {
          NDArray.scope(() {
            final dtypes = [
              DType.float64,
              DType.float32,
              DType.float16,
              DType.bfloat16,
              DType.int64,
              DType.int32,
              DType.int16,
              DType.uint8,
            ];

            for (final dt in dtypes) {
              final a = NDArray.fromList([10, 5, 20], [3], dt);
              expect(nanmin(a).scalar, 5);
              expect(nanmax(a).scalar, 20);
            }
          });
        });

        test('nanmin and nanmax out buffer reuse', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [double.nan, 3.0, 1.0],
              [3],
              DType.float64,
            );
            final outMin = NDArray<Float64>.zeros([], DType.float64);
            final resMin = nanmin(a, out: outMin);
            expect(identical(resMin, outMin), true);
            expect(outMin.scalar, 1.0);

            final outMax = NDArray<Float64>.zeros([], DType.float64);
            final resMax = nanmax(a, out: outMax);
            expect(identical(resMax, outMax), true);
            expect(outMax.scalar, 3.0);
          });
        });
      });

      group('nansum', () {
        test('nansum 1D, 2D, 3D across axes, keepdims, and all-NaN slices', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, double.nan, 3.0, double.nan],
              [4],
              DType.float64,
            );
            expect(nansum(a).scalar, 4.0);

            final allNan = NDArray.fromList(
              [double.nan, double.nan],
              [2],
              DType.float64,
            );
            expect(nansum(allNan).scalar, 0.0);

            final a2 = NDArray.fromList(
              [1.0, double.nan, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final s0 = nansum(a2, axis: 0);
            expect(s0.toList(), [4.0, 4.0]);

            final s1Keep = nansum(a2, axis: 1, keepdims: true);
            expect(s1Keep.shape, [2, 1]);
            expect(s1Keep.toList(), [1.0, 7.0]);
          });
        });

        test('nansum complex and integer types', () {
          NDArray.scope(() {
            final c = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(double.nan, 4.0), Complex(3.0, 5.0)],
              [3],
              DType.complex128,
            );
            final sc = nansum(c);
            expect(sc.scalar, Complex(4.0, 7.0));

            final i = NDArray.fromList([10, 20, 30], [3], DType.int32);
            expect(nansum(i).scalar, 60);
          });
        });

        test('nansum out buffer reuse and strided arrays', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 99.0, double.nan, 99.0, 5.0, 99.0],
              [6],
              DType.float64,
            );
            final strided = a.slice([Slice(start: 0, stop: 6, step: 2)]);
            final out = NDArray<Float64>.zeros([], DType.float64);
            final res = nansum(strided, out: out);
            expect(identical(res, out), true);
            expect(out.scalar, 6.0);
          });
        });
      });
    });

    // =========================================================================
    // 3. CORRELATION & SIGNAL CONVOLUTION / DSP
    // =========================================================================
    group('3. Correlation & Signal Convolution / DSP', () {
      group('cov and corrcoef', () {
        test('cov 1D array and 2D matrix (rowvar: true vs false)', () {
          NDArray.scope(() {
            final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            final c1D = cov(x);
            expect(c1D.shape, <int>[]);
            expect(c1D.scalar, closeTo(1.0, 1e-9));

            final m = NDArray.fromList(
              [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final cRow = cov(m, rowvar: true);
            expect(cRow.shape, [2, 2]);
            expect(cRow.getCell([0, 0]), closeTo(1.0, 1e-9));
            expect(cRow.getCell([0, 1]), closeTo(2.0, 1e-9));
            expect(cRow.getCell([1, 0]), closeTo(2.0, 1e-9));
            expect(cRow.getCell([1, 1]), closeTo(4.0, 1e-9));

            final cCol = cov(m, rowvar: false);
            expect(cCol.shape, [3, 3]);
          });
        });

        test('cov with second array y, bias, ddof, fweights, and aweights', () {
          NDArray.scope(() {
            final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            final y = NDArray.fromList([2.0, 4.0, 6.0], [3], DType.float64);

            final cXY = cov(x, y: y);
            expect(cXY.shape, [2, 2]);
            expect(cXY.getCell([0, 1]), closeTo(2.0, 1e-9));

            final cBias = cov(x, y: y, bias: true);
            expect(cBias.getCell([0, 0]), closeTo(2.0 / 3.0, 1e-9));

            final cDdof = cov(x, ddof: 0);
            expect(cDdof.scalar, closeTo(2.0 / 3.0, 1e-9));

            final fw = NDArray.fromList([1, 2, 1], [3], DType.int64);
            final cFW = cov(x, fweights: fw);
            expect(cFW.scalar, greaterThan(0.0));

            final aw = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);
            final cAW = cov(x, aweights: aw);
            expect(cAW.scalar, greaterThan(0.0));

            final cBoth = cov(x, fweights: fw, aweights: aw);
            expect(cBoth.scalar, greaterThan(0.0));
          });
        });

        test('cov out parameter reuse and validation errors', () {
          NDArray.scope(() {
            final x = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final out = NDArray<Float64>.zeros([2, 2], DType.float64);
            final res = cov(x, out: out);
            expect(identical(res, out), true);
            expect(out.shape, [2, 2]);

            final d3 = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
            expect(() => cov(d3), throwsArgumentError);

            final empty = NDArray<Float64>.zeros([0], DType.float64);
            expect(() => cov(empty), throwsArgumentError);

            final negW = NDArray.fromList([-1, 1, 1], [3], DType.int64);
            final x1D = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            expect(() => cov(x1D, fweights: negW), throwsArgumentError);
          });
        });

        test('corrcoef 1D, 2D, with y, and zero variance handling', () {
          NDArray.scope(() {
            final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            final r1D = corrcoef(x);
            expect(r1D.shape, <int>[]);
            expect(r1D.scalar, closeTo(1.0, 1e-9));

            final y = NDArray.fromList([2.0, 4.0, 6.0], [3], DType.float64);
            final rXY = corrcoef(x, y: y);
            expect(rXY.shape, [2, 2]);
            expect(rXY.getCell([0, 0]), closeTo(1.0, 1e-9));
            expect(rXY.getCell([0, 1]), closeTo(1.0, 1e-9));
            expect(rXY.getCell([1, 0]), closeTo(1.0, 1e-9));
            expect(rXY.getCell([1, 1]), closeTo(1.0, 1e-9));

            final constRow = NDArray.fromList(
              [5.0, 5.0, 5.0],
              [3],
              DType.float64,
            );
            final rConst = corrcoef(x, y: constRow);
            expect(rConst.getCell([0, 1]).isNaN, true);
            expect(rConst.getCell([1, 0]).isNaN, true);
            expect(rConst.getCell([1, 1]).isNaN, true);

            final out = NDArray<Float64>.zeros([2, 2], DType.float64);
            final res = corrcoef(x, y: y, out: out);
            expect(identical(res, out), true);
            expect(out.getCell([0, 1]), closeTo(1.0, 1e-9));
          });
        });
      });

      group('correlate, convolve, and convolve2d', () {
        test(
          '1D correlate and convolve in full, valid, same modes (float64)',
          () {
            NDArray.scope(() {
              final in1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              final in2 = NDArray.fromList([0.0, 1.0, 0.5], [3], DType.float64);

              final corrFull = correlate(in1, in2, mode: ConvMode.full);
              expect(corrFull.shape, [5]);
              expect(corrFull.toList(), [0.5, 2.0, 3.5, 3.0, 0.0]);

              final corrValid = correlate(in1, in2, mode: ConvMode.valid);
              expect(corrValid.shape, [1]);
              expect(corrValid.toList(), [3.5]);

              final corrSame = correlate(in1, in2, mode: ConvMode.same);
              expect(corrSame.shape, [3]);
              expect(corrSame.toList(), [2.0, 3.5, 3.0]);

              final convFull = convolve(in1, in2, mode: ConvMode.full);
              expect(convFull.shape, [5]);
              expect(convFull.toList(), [0.0, 1.0, 2.5, 4.0, 1.5]);

              final convValid = convolve(in1, in2, mode: ConvMode.valid);
              expect(convValid.shape, [1]);
              expect(convValid.toList(), [2.5]);

              final convSame = convolve(in1, in2, mode: ConvMode.same);
              expect(convSame.shape, [3]);
              expect(convSame.toList(), [1.0, 2.5, 4.0]);
            });
          },
        );

        test('correlate and convolve with complex128 and complex64', () {
          NDArray.scope(() {
            final in1 = NDArray.fromList(
              [Complex(1.0, 1.0), Complex(2.0, 0.0)],
              [2],
              DType.complex128,
            );
            final in2 = NDArray.fromList(
              [Complex(1.0, 0.0), Complex(0.0, 1.0)],
              [2],
              DType.complex128,
            );

            final corr = correlate(in1, in2, mode: ConvMode.full);
            expect(corr.dtype, DType.complex128);
            expect(corr.shape, [3]);

            final conv = convolve(in1, in2, mode: ConvMode.full);
            expect(conv.dtype, DType.complex128);
            expect(conv.shape, [3]);

            final in1_64 = NDArray.fromList(
              [Complex(1.0, 1.0), Complex(2.0, 0.0)],
              [2],
              DType.complex64,
            );
            final in2_64 = NDArray.fromList(
              [Complex(1.0, 0.0), Complex(0.0, 1.0)],
              [2],
              DType.complex64,
            );
            final corr64 = correlate(in1_64, in2_64, mode: ConvMode.valid);
            expect(corr64.dtype, DType.complex64);
            expect(corr64.shape, [1]);
          });
        });

        test(
          'convolve2d 2D spatial convolution across full, valid, same modes',
          () {
            NDArray.scope(() {
              final image = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [3, 3],
                DType.float64,
              );
              final kernel = NDArray.fromList(
                [1.0, 0.0, 0.0, 1.0],
                [2, 2],
                DType.float64,
              );

              final full = convolve2d(image, kernel, mode: ConvMode.full);
              expect(full.shape, [4, 4]);

              final valid = convolve2d(image, kernel, mode: ConvMode.valid);
              expect(valid.shape, [2, 2]);

              final same = convolve2d(image, kernel, mode: ConvMode.same);
              expect(same.shape, [3, 3]);

              final out = NDArray<Float64>.zeros([2, 2], DType.float64);
              final res = convolve2d(
                image,
                kernel,
                mode: ConvMode.valid,
                out: out,
              );
              expect(identical(res, out), true);
            });
          },
        );

        test('correlate and convolve error validation', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final b = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            expect(
              () => correlate(a, b, mode: ConvMode.valid),
              throwsArgumentError,
            );

            final a2D = NDArray<Float64>.zeros([2, 2], DType.float64);
            expect(() => correlate(a, a2D), throwsArgumentError);

            expect(() => convolve2d(a, b), throwsArgumentError);
          });
        });
      });

      group('angle and unwrap', () {
        test('angle on complex128 and complex64 contiguous and strided', () {
          NDArray.scope(() {
            final c128 = NDArray.fromList(
              [
                Complex(1.0, 0.0),
                Complex(0.0, 1.0),
                Complex(-1.0, 0.0),
                Complex(0.0, -1.0),
              ],
              [4],
              DType.complex128,
            );
            final a128 = angle(c128);
            expect(a128.dtype, DType.float64);
            expect(a128.toList()[0], closeTo(0.0, 1e-9));
            expect(a128.toList()[1], closeTo(math.pi / 2.0, 1e-9));
            expect(a128.toList()[2], closeTo(math.pi, 1e-9));
            expect(a128.toList()[3], closeTo(-math.pi / 2.0, 1e-9));

            final out = NDArray<Float64>.zeros([4], DType.float64);
            final res = angle(c128, out: out);
            expect(identical(res, out), true);

            final c64 = NDArray.fromList(
              [Complex(1.0, 1.0)],
              [1],
              DType.complex64,
            );
            final a64 = angle(c64);
            expect(a64.dtype, DType.float32);
            expect(a64.toList()[0], closeTo(math.pi / 4.0, 1e-6));
          });
        });

        test('unwrap 1D and 2D along axes with custom discont', () {
          NDArray.scope(() {
            final phase = NDArray.fromList(
              [0.0, math.pi + 0.2, -(math.pi + 0.2)],
              [3],
              DType.float64,
            );
            final unwrapped = unwrap(phase);
            expect(unwrapped.dtype, DType.float64);
            expect(unwrapped.shape, [3]);

            final phase2D = NDArray.fromList(
              [0.0, math.pi + 0.5, 0.0, -(math.pi + 0.5)],
              [2, 2],
              DType.float64,
            );
            final unwrapped0 = unwrap(phase2D, axis: 0);
            expect(unwrapped0.shape, [2, 2]);

            final unwrapped1 = unwrap(phase2D, axis: 1);
            expect(unwrapped1.shape, [2, 2]);

            final out = NDArray<Float64>.zeros([3], DType.float64);
            final res = unwrap(phase, out: out);
            expect(identical(res, out), true);
          });
        });
      });
    });

    // =========================================================================
    // 4. MULTI-DIMENSIONAL BINNING
    // =========================================================================
    group('4. Multi-dimensional Binning', () {
      group('bincount', () {
        test('unweighted bincount across int64, int32, int16, uint8', () {
          NDArray.scope(() {
            final intTypes = [
              DType.int64,
              DType.int32,
              DType.int16,
              DType.uint8,
            ];

            for (final dt in intTypes) {
              final x = NDArray.fromList([0, 1, 1, 3, 2, 1, 5], [7], dt);
              final counts = bincount(x);
              expect(counts.shape, [6]);
              expect(counts.toList(), [1, 3, 1, 1, 0, 1]);
            }
          });
        });

        test('bincount with minlength (smaller and larger than max value)', () {
          NDArray.scope(() {
            final x = NDArray.fromList([0, 1, 2], [3], DType.int64);
            final cLarge = bincount(x, minlength: 6);
            expect(cLarge.shape, [6]);
            expect(cLarge.toList(), [1, 1, 1, 0, 0, 0]);

            final cSmall = bincount(x, minlength: 1);
            expect(cSmall.shape, [3]);
            expect(cSmall.toList(), [1, 1, 1]);
          });
        });

        test(
          'weighted bincount with float64 and float32 and strided slices',
          () {
            NDArray.scope(() {
              final x = NDArray.fromList([0, 1, 1, 2], [4], DType.int64);
              final w64 = NDArray.fromList(
                [0.5, 1.0, 2.0, 1.5],
                [4],
                DType.float64,
              );
              final c64 = bincount(x, weights: w64);
              expect(c64.dtype, DType.float64);
              expect(c64.toList(), [0.5, 3.0, 1.5]);

              final w32 = NDArray.fromList(
                [0.5, 1.0, 2.0, 1.5],
                [4],
                DType.float32,
              );
              final c32 = bincount(x, weights: w32);
              expect(c32.dtype, DType.float32);

              final xFull = NDArray.fromList(
                [0, 99, 1, 99, 1, 99, 2],
                [7],
                DType.int64,
              );
              final xStrided = xFull.slice([Slice(start: 0, stop: 7, step: 2)]);
              final wFull = NDArray.fromList(
                [0.5, 99.0, 1.0, 99.0, 2.0, 99.0, 1.5],
                [7],
                DType.float64,
              );
              final wStrided = wFull.slice([Slice(start: 0, stop: 7, step: 2)]);
              final cStrided = bincount(xStrided, weights: wStrided);
              expect(cStrided.toList(), [0.5, 3.0, 1.5]);
            });
          },
        );

        test('bincount empty input and out buffer reuse', () {
          NDArray.scope(() {
            final empty = NDArray.zeros([0], DType.int64);
            final cEmpty = bincount(empty);
            expect(cEmpty.shape, [0]);

            final cEmptyMin = bincount(empty, minlength: 4);
            expect(cEmptyMin.shape, [4]);
            expect(cEmptyMin.toList(), [0, 0, 0, 0]);

            final x = NDArray.fromList([0, 1, 1], [3], DType.int32);
            final out = NDArray.zeros([3], DType.int32);
            final res = bincount(x, out: out);
            expect(identical(res, out), true);
            expect(out.toList(), [1, 2, 0]);
          });
        });

        test('bincount validation errors', () {
          NDArray.scope(() {
            final xNeg = NDArray.fromList([0, -1, 2], [3], DType.int64);
            expect(() => bincount(xNeg), throwsArgumentError);

            final x2D = NDArray.zeros([2, 2], DType.int64);
            expect(() => bincount(x2D), throwsArgumentError);

            final x = NDArray.fromList([0, 1], [2], DType.int64);
            expect(() => bincount(x, minlength: -1), throwsArgumentError);
          });
        });
      });

      group('digitize', () {
        test(
          'monotonically increasing bins with right: false and right: true',
          () {
            NDArray.scope(() {
              final x = NDArray.fromList(
                [0.2, 6.4, 3.0, 1.6, 0.0, 10.0],
                [6],
                DType.float64,
              );
              final bins = NDArray.fromList(
                [0.0, 1.0, 2.5, 4.0, 10.0],
                [5],
                DType.float64,
              );

              final indLeft = digitize(x, bins, right: false);
              expect(indLeft.toList(), [1, 4, 3, 2, 1, 5]);

              final indRight = digitize(x, bins, right: true);
              expect(indRight.toList(), [1, 4, 3, 2, 0, 4]);
            });
          },
        );

        test('monotonically decreasing bins', () {
          NDArray.scope(() {
            final x = NDArray.fromList(
              [0.2, 6.4, 3.0, 1.6],
              [4],
              DType.float64,
            );
            final bins = NDArray.fromList(
              [10.0, 4.0, 2.5, 1.0, 0.0],
              [5],
              DType.float64,
            );

            final ind = digitize(x, bins, right: false);
            expect(ind.shape, [4]);
            for (final val in ind.toList()) {
              expect(val, greaterThanOrEqualTo(0));
              expect(val, lessThanOrEqualTo(5));
            }
          });
        });

        test('digitize 2D inputs and out buffer reuse', () {
          NDArray.scope(() {
            final x2D = NDArray.fromList(
              [0.5, 1.5, 2.5, 3.5],
              [2, 2],
              DType.float64,
            );
            final bins = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            final res = digitize(x2D, bins);
            expect(res.shape, [2, 2]);
            expect(res.toList(), [0, 1, 2, 3]);

            final out = NDArray.zeros([2, 2], DType.int64);
            final resOut = digitize(x2D, bins, out: out);
            expect(identical(resOut, out), true);
            expect(out.toList(), [0, 1, 2, 3]);
          });
        });

        test('digitize validation errors', () {
          NDArray.scope(() {
            final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final nonMono = NDArray.fromList(
              [1.0, 3.0, 2.0],
              [3],
              DType.float64,
            );
            expect(() => digitize(x, nonMono), throwsArgumentError);

            final emptyBins = NDArray<Float64>.zeros([0], DType.float64);
            expect(() => digitize(x, emptyBins), throwsArgumentError);

            final cBins = NDArray.fromList(
              [Complex(1.0, 0.0)],
              [1],
              DType.complex128,
            );
            expect(() => digitize(x, cBins as dynamic), throwsA(anything));
          });
        });
      });

      group('histogram', () {
        test('uniform histogram with integer bins and range', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 1.0, 3.0, 4.0, 2.0],
              [6],
              DType.float64,
            );
            final (:hist, :binEdges) = histogram(a, bins: 4, range: (1.0, 5.0));
            expect(hist.shape, [4]);
            expect(binEdges.shape, [5]);
            expect(binEdges.toList(), [1.0, 2.0, 3.0, 4.0, 5.0]);
            expect(hist.toList(), [2, 2, 1, 1]);
          });
        });

        test('non-uniform histogram with explicit NDArray bin edges', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [0.5, 1.5, 2.5, 3.5, 7.0],
              [5],
              DType.float64,
            );
            final edges = NDArray.fromList(
              [0.0, 1.0, 3.0, 10.0],
              [4],
              DType.float64,
            );
            final (:hist, :binEdges) = histogram(a, bins: edges);
            expect(hist.shape, [3]);
            expect(binEdges.shape, [4]);
            expect(hist.toList(), [1, 2, 2]);
          });
        });

        test('weighted histogram and density: true probability density', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final weights = NDArray.fromList(
              [0.5, 1.5, 2.0, 1.0],
              [4],
              DType.float64,
            );
            final (:hist, :binEdges) = histogram(
              a,
              bins: 2,
              range: (1.0, 5.0),
              weights: weights,
            );
            expect(hist.toList(), [2.0, 3.0]);

            final densityRes = histogram(
              a,
              bins: 2,
              range: (1.0, 5.0),
              density: true,
            );
            final dHist = densityRes.hist.toList();
            final dEdges = densityRes.binEdges.toList();
            var integral = 0.0;
            for (var i = 0; i < dHist.length; i++) {
              final width = dEdges[i + 1] - dEdges[i];
              integral += dHist[i] * width;
            }
            expect(integral, closeTo(1.0, 1e-9));
          });
        });

        test('histogram with strided input and identical min/max values', () {
          NDArray.scope(() {
            final constArray = NDArray.fromList(
              [5.0, 5.0, 5.0],
              [3],
              DType.float64,
            );
            final (:hist, :binEdges) = histogram(constArray, bins: 2);
            expect(hist.shape, [2]);
            expect(binEdges.shape, [3]);
            expect(sum(hist.asAnySpec).scalar, 3);

            final full = NDArray.fromList(
              [1.0, 99.0, 2.0, 99.0, 3.0],
              [5],
              DType.float64,
            );
            final strided = full.slice([Slice(start: 0, stop: 5, step: 2)]);
            final resStrided = histogram(strided, bins: 2, range: (1.0, 3.0));
            expect(resStrided.hist.toList(), [1, 2]);
          });
        });

        test('histogram validation errors', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            expect(() => histogram(a, bins: 0), throwsArgumentError);
            expect(() => histogram(a, bins: -5), throwsArgumentError);
            expect(() => histogram(a, range: (5.0, 2.0)), throwsArgumentError);

            final nonMonoEdges = NDArray.fromList(
              [0.0, 2.0, 1.0],
              [3],
              DType.float64,
            );
            expect(() => histogram(a, bins: nonMonoEdges), throwsArgumentError);

            final tooFewEdges = NDArray.fromList([1.0], [1], DType.float64);
            expect(() => histogram(a, bins: tooFewEdges), throwsArgumentError);

            final complexData = NDArray.fromList(
              [Complex(1.0, 2.0)],
              [1],
              DType.complex128,
            );
            expect(() => histogram(complexData as dynamic), throwsA(anything));
          });
        });
      });
    });
  });
}
