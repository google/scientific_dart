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
  group('Comprehensive Calculus Operations', () {
    group('trapz composite trapezoidal integration', () {
      test('1D uniform step integration (Float64, Float32)', () {
        NDArray.scope(() {
          final y64 = NDArray.fromList(
            [0.0, 1.0, 4.0, 9.0, 16.0],
            [5],
            DType.float64,
          );
          final res64 = trapz(y64, spacing: const Spacing.step(1.0));
          expect(res64.shape, equals([]));
          expect(res64.scalar, closeTo(22.0, 1e-9));

          final resStep = trapz(y64, spacing: const Spacing.step(2.5));
          expect(resStep.scalar, closeTo(55.0, 1e-9));

          final y32 = NDArray.fromList(
            [0.0, 1.0, 4.0, 9.0, 16.0],
            [5],
            DType.float32,
          );
          final res32 = trapz(y32, spacing: const Spacing.step(1.0));
          expect(res32.dtype, equals(DType.float32));
          expect(res32.scalar, closeTo(22.0, 1e-5));
        });
      });

      test('1D non-uniform coordinate spacing', () {
        NDArray.scope(() {
          final y = NDArray.fromList([1.0, 3.0, 7.0], [3], DType.float64);
          final res = trapz(
            y,
            spacing: const Spacing.coordinates([0.0, 1.0, 3.0]),
          );
          expect(res.scalar, closeTo(12.0, 1e-9));
        });
      });

      test('Complex numbers and contour integration', () {
        NDArray.scope(() {
          final yCpx128 = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(2.0, 4.0), Complex(4.0, 8.0)],
            [3],
            DType.complex128,
          );
          final resCpx128 = trapz(yCpx128, spacing: const Spacing.step(1.0));
          expect(resCpx128.dtype, equals(DType.complex128));
          expect(resCpx128.scalar.real, closeTo(4.5, 1e-9));
          expect(resCpx128.scalar.imag, closeTo(9.0, 1e-9));

          final yLine = NDArray.fromList(
            [Complex(0, 0), Complex(0, 1), Complex(0, 2)],
            [3],
            DType.complex128,
          );
          final resContour = trapz(
            yLine,
            spacing: Spacing.step(Complex(0.0, 0.5)),
          );
          expect(resContour.scalar.real, closeTo(-1.0, 1e-9));
          expect(resContour.scalar.imag, closeTo(0.0, 1e-9));

          final coords = [Complex(0, 0), Complex(1, 1), Complex(2, 2)];
          final resCoords = trapz(yLine, spacing: Spacing.coordinates(coords));
          expect(resCoords.dtype, equals(DType.complex128));

          final yCpx64 = NDArray.fromList(
            [Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(4.0, 0.0)],
            [3],
            DType.complex64,
          );
          final resCpx64 = trapz(yCpx64);
          expect(resCpx64.dtype, equals(DType.complex64));
          expect(resCpx64.scalar.real, closeTo(4.5, 1e-5));
        });
      });

      test('2D and 3D multi-axis integrations', () {
        NDArray.scope(() {
          final m2d = NDArray.fromList(
            [1.0, 2.0, 4.0, 3.0, 6.0, 12.0],
            [2, 3],
            DType.float64,
          );

          final resAxis1 = trapz(m2d, axis: 1);
          expect(resAxis1.shape, equals([2]));
          expect(resAxis1.toList(), equals([4.5, 13.5]));

          final resAxisNeg1 = trapz(m2d, axis: -1);
          expect(resAxisNeg1.toList(), equals([4.5, 13.5]));

          final resAxis0 = trapz(m2d, axis: 0);
          expect(resAxis0.shape, equals([3]));
          expect(resAxis0.toList(), equals([2.0, 4.0, 8.0]));

          final t3d = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );
          final res3dAxis2 = trapz(t3d, axis: 2);
          expect(res3dAxis2.shape, equals([2, 2]));
          expect(res3dAxis2.getCell([0, 0]), closeTo(1.5, 1e-9));
          expect(res3dAxis2.getCell([0, 1]), closeTo(3.5, 1e-9));
          expect(res3dAxis2.getCell([1, 0]), closeTo(5.5, 1e-9));
          expect(res3dAxis2.getCell([1, 1]), closeTo(7.5, 1e-9));
        });
      });

      test('Strided views and out buffer reuse for trapz', () {
        NDArray.scope(() {
          final parent = NDArray.fromList(
            [1.0, 99.0, 2.0, 99.0, 4.0],
            [5],
            DType.float64,
          );
          final view = parent.slice([const Slice(start: 0, stop: 5, step: 2)]);
          expect(view.isContiguous, isFalse);

          final out = NDArray<double>.zeros([], DType.float64);
          final res = trapz(view, out: out);
          expect(identical(res, out), isTrue);
          expect(out.scalar, closeTo(4.5, 1e-9));
        });
      });

      test('trapz error handling', () {
        NDArray.scope(() {
          final valid = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

          final disp = NDArray.fromList([1.0, 2.0], [2], DType.float64)
            ..dispose();
          expect(() => trapz(disp), throwsStateError);

          final dispOut = NDArray<double>.zeros([], DType.float64)..dispose();
          expect(() => trapz(valid, out: dispOut), throwsStateError);

          final intArr = NDArray.fromList([1, 2, 3], [3], DType.int64);
          expect(() => trapz(intArr), throwsArgumentError);

          final boolArr = NDArray.fromList([true, false], [2], DType.boolean);
          expect(() => trapz(boolArr), throwsArgumentError);

          expect(() => trapz(valid, axis: 3), throwsArgumentError);
          expect(() => trapz(valid, axis: -3), throwsArgumentError);

          expect(
            () => trapz(valid, spacing: const Spacing.coordinates([0.0, 1.0])),
            throwsArgumentError,
          );

          expect(
            () => trapz(valid, spacing: Spacing.step(Complex(0.0, 1.0))),
            throwsArgumentError,
          );

          final badOutShape = NDArray<double>.zeros([2], DType.float64);
          expect(() => trapz(valid, out: badOutShape), throwsArgumentError);
        });
      });
    });

    group('gradient and gradientArray', () {
      test('1D gradient with edgeOrder=1 and edgeOrder=2', () {
        NDArray.scope(() {
          final f = NDArray.fromList(
            [0.0, 1.0, 4.0, 9.0, 16.0],
            [5],
            DType.float64,
          );

          final g1 = gradient(
            f,
            spacing: const Spacing.step(1.0),
            edgeOrder: 1,
          );
          expect(g1.shape, equals([5]));
          expect(g1.getCell([0]), closeTo(1.0, 1e-9));
          expect(g1.getCell([1]), closeTo(2.0, 1e-9));
          expect(g1.getCell([2]), closeTo(4.0, 1e-9));
          expect(g1.getCell([3]), closeTo(6.0, 1e-9));
          expect(g1.getCell([4]), closeTo(7.0, 1e-9));

          final g2 = gradient(
            f,
            spacing: const Spacing.step(1.0),
            edgeOrder: 2,
          );
          expect(g2.getCell([0]), closeTo(0.0, 1e-9));
          expect(g2.getCell([1]), closeTo(2.0, 1e-9));
          expect(g2.getCell([2]), closeTo(4.0, 1e-9));
          expect(g2.getCell([3]), closeTo(6.0, 1e-9));
          expect(g2.getCell([4]), closeTo(8.0, 1e-9));
        });
      });

      test('gradient with non-uniform coordinate spacing', () {
        NDArray.scope(() {
          final f = NDArray.fromList([0.0, 1.0, 9.0], [3], DType.float64);
          final g = gradient(
            f,
            spacing: const Spacing.coordinates([0.0, 1.0, 3.0]),
          );
          expect(g.getCell([1]), closeTo(2.0, 1e-9));
        });
      });

      test('2D multi-axis gradient and gradientArray', () {
        NDArray.scope(() {
          final grid = NDArray.fromList(
            [0.0, 3.0, 6.0, 2.0, 5.0, 8.0, 4.0, 7.0, 10.0],
            [3, 3],
            DType.float64,
          );

          final grads = gradientArray(grid);
          expect(grads.length, equals(2));

          final dfDx = grads[0];
          expect(dfDx.shape, equals([3, 3]));
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 3; j++) {
              expect(dfDx.getCell([i, j]), closeTo(2.0, 1e-9));
            }
          }

          final dfDy = grads[1];
          expect(dfDy.shape, equals([3, 3]));
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 3; j++) {
              expect(dfDy.getCell([i, j]), closeTo(3.0, 1e-9));
            }
          }

          final gradsCustom = gradientArray(
            grid,
            axis: [1],
            spacings: [const Spacing.step(2.0)],
          );
          expect(gradsCustom.length, equals(1));
          expect(gradsCustom[0].getCell([0, 0]), closeTo(1.5, 1e-9));

          // gradientArray with out parameter
          final out0 = NDArray<double>.zeros([3, 3], DType.float64);
          final out1 = NDArray<double>.zeros([3, 3], DType.float64);
          final gradsOut = gradientArray(grid, out: [out0, out1]);
          expect(identical(gradsOut[0], out0), isTrue);
          expect(identical(gradsOut[1], out1), isTrue);
        });
      });

      test('gradient error handling', () {
        NDArray.scope(() {
          final f = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

          final disp = NDArray.fromList([1.0, 2.0], [2], DType.float64)
            ..dispose();
          expect(() => gradient(disp), throwsStateError);
          expect(() => gradientArray(disp), throwsStateError);

          expect(() => gradient(f, edgeOrder: 0), throwsArgumentError);
          expect(() => gradient(f, edgeOrder: 3), throwsArgumentError);

          final fSmall = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => gradient(fSmall, edgeOrder: 2), throwsArgumentError);

          expect(
            () => gradientArray(
              f,
              spacing: const Spacing.step(1.0),
              spacings: [const Spacing.step(1.0)],
            ),
            throwsArgumentError,
          );

          expect(() => gradientArray(f, axis: [0, 0]), throwsArgumentError);
        });
      });
    });

    group('diff discrete difference operator', () {
      test('1D diff across various orders n=0, 1, 2, 3, 4', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 4.0, 7.0, 11.0],
            [5],
            DType.float64,
          );

          final d0 = diff(a, n: 0);
          expect(d0.toList(), equals([1.0, 2.0, 4.0, 7.0, 11.0]));

          final d1 = diff(a, n: 1);
          expect(d1.toList(), equals([1.0, 2.0, 3.0, 4.0]));

          final d2 = diff(a, n: 2);
          expect(d2.toList(), equals([1.0, 1.0, 1.0]));

          final d3 = diff(a, n: 3);
          expect(d3.toList(), equals([0.0, 0.0]));

          final d4 = diff(a, n: 4);
          expect(d4.toList(), equals([0.0]));

          final d5 = diff(a, n: 5);
          expect(d5.shape, equals([0]));
        });
      });

      test('diff across integer, float, and complex DTypes', () {
        NDArray.scope(() {
          final aInt = NDArray.fromList([10, 25, 45], [3], DType.int32);
          final dInt = diff(aInt);
          expect(dInt.dtype, equals(DType.int32));
          expect(dInt.toList(), equals([15, 20]));

          final aCpx = NDArray.fromList(
            [Complex(1, 1), Complex(3, 4), Complex(6, 9)],
            [3],
            DType.complex128,
          );
          final dCpx = diff(aCpx);
          expect(dCpx.getCell([0]), equals(Complex(2.0, 3.0)));
          expect(dCpx.getCell([1]), equals(Complex(3.0, 5.0)));
        });
      });

      test('2D diff along axes 0 and 1', () {
        NDArray.scope(() {
          final m = NDArray.fromList(
            [1.0, 2.0, 4.0, 3.0, 5.0, 9.0],
            [2, 3],
            DType.float64,
          );

          final dAxis1 = diff(m, axis: 1);
          expect(dAxis1.shape, equals([2, 2]));
          expect(dAxis1.toList(), equals([1.0, 2.0, 2.0, 4.0]));

          final dAxis0 = diff(m, axis: 0);
          expect(dAxis0.shape, equals([1, 3]));
          expect(dAxis0.toList(), equals([2.0, 3.0, 5.0]));
        });
      });
    });
  });

  group('Comprehensive Statistics Operations', () {
    group('multi-DType reductions: sum, prod, mean, std, var, min, max, ptp', () {
      test(
        'sum across various integer, float, complex, and boolean DTypes',
        () {
          NDArray.scope(() {
            // Float64
            final f64 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            expect(sum(f64).scalar, closeTo(10.0, 1e-9));

            // Float32
            final f32 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float32,
            );
            expect(sum(f32).scalar, closeTo(10.0, 1e-5));

            // Int64
            final i64 = NDArray.fromList([10, 20, 30], [3], DType.int64);
            expect(sum(i64).scalar, equals(60));

            // Int32
            final i32 = NDArray.fromList([10, 20, 30], [3], DType.int32);
            expect(sum(i32).scalar, equals(60));

            // Uint8
            final u8 = NDArray.fromList([5, 10, 15], [3], DType.uint8);
            expect(sum(u8).scalar, equals(30));

            // Complex128
            final c128 = NDArray.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              DType.complex128,
            );
            expect(sum(c128).scalar, equals(Complex(4.0, 6.0)));

            // Boolean
            final bools = NDArray.fromList(
              [true, false, true, true],
              [4],
              DType.boolean,
            );
            expect(sum(bools).scalar, isTrue);

            // Empty array sum
            final empty = NDArray.zeros([0], DType.float64);
            expect(sum(empty).scalar, equals(0.0));
          });
        },
      );

      test(
        'prod, mean, std, variance, min, max, ptp multi-axis and keepdims',
        () {
          NDArray.scope(() {
            final m = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );

            // prod
            expect(prod(m).scalar, closeTo(720.0, 1e-9));
            expect(prod(m, axis: 0).toList(), equals([4.0, 10.0, 18.0]));
            expect(prod(m, axis: 1).toList(), equals([6.0, 120.0]));

            // mean
            expect(mean(m).scalar, closeTo(3.5, 1e-9));
            expect(mean(m, axis: 0, keepdims: true).shape, equals([1, 3]));
            expect(mean(m, axis: 0).toList(), equals([2.5, 3.5, 4.5]));
            expect(mean(m, axis: 1).toList(), equals([2.0, 5.0]));

            // std and variance
            expect(
              variance(m).scalar,
              closeTo(35.0 / 12.0, 1e-5),
            ); // population var: ( (1-3.5)^2 + ... ) / 6 = 17.5 / 6 = 2.916667
            expect(std(m).scalar, closeTo(math.sqrt(17.5 / 6.0), 1e-5));
            expect(
              variance(m, ddof: 1).scalar,
              closeTo(17.5 / 5.0, 1e-5),
            ); // sample var: 17.5 / 5 = 3.5

            // min, max, ptp
            expect(min(m).scalar, equals(1.0));
            expect(max(m).scalar, equals(6.0));
            expect(ptp(m).scalar, equals(5.0)); // 6.0 - 1.0 = 5.0
            expect(ptp(m, axis: 0).toList(), equals([3.0, 3.0, 3.0]));
            expect(ptp(m, axis: 1).toList(), equals([2.0, 2.0]));
          });
        },
      );
    });

    group('average (weighted and unweighted)', () {
      test('1D unweighted and weighted average', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);

          final unweighted = average(a);
          expect(unweighted.average.shape, equals([]));
          expect(unweighted.average.scalar, closeTo(2.5, 1e-9));
          expect(unweighted.sumOfWeights, isNull);

          final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final weighted = average(a, weights: w, returned: true);
          expect(weighted.average.scalar, closeTo(3.0, 1e-9));
          expect(weighted.sumOfWeights, isNotNull);
          expect(weighted.sumOfWeights!.scalar, closeTo(10.0, 1e-9));
        });
      });

      test('2D weighted average along axis 0 and 1', () {
        NDArray.scope(() {
          final m = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final wAxis0 = NDArray.fromList([1.0, 3.0], [2], DType.float64);

          final resAxis0 = average(m, axis: 0, weights: wAxis0);
          expect(resAxis0.average.shape, equals([2]));
          expect(resAxis0.average.toList(), equals([2.5, 3.5]));

          final resAxis1 = average(m, axis: 1, weights: wAxis0);
          expect(resAxis1.average.shape, equals([2]));
          expect(resAxis1.average.toList(), equals([1.75, 3.75]));
        });
      });
    });

    group('cov and corrcoef', () {
      test(
        '1D and 2D covariance matrix computation with rowvar, bias, and ddof',
        () {
          NDArray.scope(() {
            final x = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final c1d = cov(x);
            expect(c1d.shape, equals([]));
            expect(c1d.scalar, closeTo(5.0 / 3.0, 1e-5));

            // 2D: rowvar = true
            final m2d = NDArray.fromList(
              [1.0, 2.0, 3.0, 1.0, 2.0, 4.0],
              [2, 3],
              DType.float64,
            );
            final c2d = cov(m2d);
            expect(c2d.shape, equals([2, 2]));
            expect(c2d.getCell([0, 0]), closeTo(1.0, 1e-5));
            expect(c2d.getCell([1, 1]), closeTo(7.0 / 3.0, 1e-5));
            expect(c2d.getCell([0, 1]), closeTo(1.5, 1e-5));

            // rowvar = false (columns are variables: shape [3, 3])
            final c2dCol = cov(m2d, rowvar: false);
            expect(c2dCol.shape, equals([3, 3]));

            // bias = true (normalize by N instead of N-1)
            final c2dBias = cov(x, bias: true);
            expect(c2dBias.scalar, closeTo(1.25, 1e-5)); // 5.0 / 4 = 1.25
          });
        },
      );

      test('corrcoef normalized Pearson correlation matrix', () {
        NDArray.scope(() {
          final m2d = NDArray.fromList(
            [1.0, 2.0, 3.0, 1.0, 2.0, 4.0],
            [2, 3],
            DType.float64,
          );
          final r = corrcoef(m2d);
          expect(r.shape, equals([2, 2]));
          expect(r.getCell([0, 0]), closeTo(1.0, 1e-5));
          expect(r.getCell([1, 1]), closeTo(1.0, 1e-5));
          expect(r.getCell([0, 1]), closeTo(0.98198, 1e-4));
          expect(r.getCell([1, 0]), closeTo(0.98198, 1e-4));

          // Constant input correlation returns NaN
          final cConst = NDArray.fromList([2.0, 2.0, 2.0], [3], DType.float64);
          final rConst = corrcoef(cConst);
          expect(rConst.scalar.isNaN, isTrue);
        });
      });
    });

    group('percentile and quantile (all 13 methods)', () {
      test('All QuantileMethod types on uniform data', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );

          for (final method in QuantileMethod.values) {
            final q0 = quantile(a, 0.0, method: method);
            expect(q0.scalar, closeTo(1.0, 1e-4));

            final q100 = quantile(a, 1.0, method: method);
            expect(q100.scalar, closeTo(5.0, 1e-4));

            final q50 = quantile(a, 0.5, method: method);
            if (method == QuantileMethod.closestObservation) {
              expect(q50.scalar, closeTo(2.0, 1e-4));
            } else if (method == QuantileMethod.interpolatedInvertedCdf) {
              expect(q50.scalar, closeTo(2.5, 1e-4));
            } else {
              expect(q50.scalar, closeTo(3.0, 1e-4));
            }
          }
        });
      });

      test('percentile vs quantile equivalence', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          final p25 = percentile(a, 25.0);
          final q25 = quantile(a, 0.25);
          expect(p25.scalar, equals(q25.scalar));

          final p75 = percentile(a, 75.0);
          final q75 = quantile(a, 0.75);
          expect(p75.scalar, equals(q75.scalar));
        });
      });
    });

    group('median multi-axis and dtypes', () {
      test('median on even and odd size arrays', () {
        NDArray.scope(() {
          final odd = NDArray.fromList([5.0, 1.0, 3.0], [3], DType.float64);
          expect(median(odd).scalar, equals(3.0));

          final even = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          expect(median(even).scalar, equals(2.5));

          final m = NDArray.fromList(
            [1.0, 5.0, 2.0, 6.0],
            [2, 2],
            DType.float64,
          );
          expect(median(m, axis: 0).toList(), equals([1.5, 5.5]));
          expect(median(m, axis: 1).toList(), equals([3.0, 4.0]));
        });
      });
    });

    group(
      'NaN-aware statistics: nanmean, nanstd, nanvar, nanmin, nanmax, nansum',
      () {
        test('1D and 2D arrays with NaNs', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, double.nan, 3.0, double.nan, 5.0],
              [5],
              DType.float64,
            );

            expect(nanmean(a).scalar, closeTo(3.0, 1e-9));
            expect(nansum(a).scalar, closeTo(9.0, 1e-9));
            expect(nanmin(a).scalar, equals(1.0));
            expect(nanmax(a).scalar, equals(5.0));
            expect(nanvar(a).scalar, closeTo(8.0 / 3.0, 1e-5));
            expect(nanstd(a).scalar, closeTo(math.sqrt(8.0 / 3.0), 1e-5));
          });
        });

        test('2D multi-axis nan reductions', () {
          NDArray.scope(() {
            final m = NDArray.fromList(
              [1.0, double.nan, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );

            final mean0 = nanmean(m, axis: 0);
            expect(mean0.shape, equals([2]));
            expect(mean0.toList(), equals([2.0, 4.0]));

            final mean1 = nanmean(m, axis: 1);
            expect(mean1.shape, equals([2]));
            expect(mean1.toList(), equals([1.0, 3.5]));
          });
        });
      },
    );

    group('Cumulative operations: cumsum, cumprod, cummin, cummax', () {
      test('1D and 2D cumulative reductions', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 3.0, 2.0, 5.0, 4.0],
            [5],
            DType.float64,
          );

          expect(cumsum(a).toList(), equals([1.0, 4.0, 6.0, 11.0, 15.0]));
          expect(cumprod(a).toList(), equals([1.0, 3.0, 6.0, 30.0, 120.0]));
          expect(cummin(a).toList(), equals([1.0, 1.0, 1.0, 1.0, 1.0]));
          expect(cummax(a).toList(), equals([1.0, 3.0, 3.0, 5.0, 5.0]));

          final m = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(cumsum(m, axis: 1).toList(), equals([1.0, 3.0, 3.0, 7.0]));
        });
      });
    });

    group('Logical reductions: all, any', () {
      test('all and any across boolean and numeric dtypes', () {
        NDArray.scope(() {
          final bAllTrue = NDArray.fromList(
            [true, true, true],
            [3],
            DType.boolean,
          );
          final bMixed = NDArray.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          final bAllFalse = NDArray.fromList(
            [false, false, false],
            [3],
            DType.boolean,
          );

          expect(all(bAllTrue).scalar, isTrue);
          expect(all(bMixed).scalar, isFalse);
          expect(any(bMixed).scalar, isTrue);
          expect(any(bAllFalse).scalar, isFalse);

          final m = NDArray.fromList(
            [true, false, true, true],
            [2, 2],
            DType.boolean,
          );
          expect(all(m, axis: 0).toList(), equals([true, false]));
          expect(any(m, axis: 1).toList(), equals([true, true]));
        });
      });
    });

    group('Binning operations: histogram, bincount, digitize', () {
      test('histogram uniform and non-uniform bins', () {
        NDArray.scope(() {
          final data = NDArray.fromList(
            [0.5, 1.5, 1.8, 2.2, 3.5],
            [5],
            DType.float64,
          );

          final hUniform = histogram(data, bins: 4, range: (0.0, 4.0));
          expect(hUniform.hist.toList(), equals([1, 2, 1, 1]));
          expect(hUniform.binEdges.toList(), equals([0.0, 1.0, 2.0, 3.0, 4.0]));

          final edges = NDArray.fromList(
            [0.0, 1.0, 3.0, 5.0],
            [4],
            DType.float64,
          );
          final hCustom = histogram(data, bins: edges);
          expect(hCustom.hist.toList(), equals([1, 3, 1]));
        });
      });

      test('bincount unweighted and weighted', () {
        NDArray.scope(() {
          final x = NDArray.fromList([0, 1, 1, 3, 2, 1, 7], [7], DType.int64);

          final counts = bincount(x);
          expect(counts.toList(), equals([1, 3, 1, 1, 0, 0, 0, 1]));

          final counts10 = bincount(x, minlength: 10);
          expect(counts10.shape, equals([10]));
          expect(counts10.toList(), equals([1, 3, 1, 1, 0, 0, 0, 1, 0, 0]));

          final weights = NDArray.fromList(
            [0.5, 1.0, 2.0, 0.2, 0.3, 0.5, 1.5],
            [7],
            DType.float64,
          );
          final wCounts = bincount(x, weights: weights);
          expect(wCounts.getCell([0]), closeTo(0.5, 1e-9));
          expect(wCounts.getCell([1]), closeTo(3.5, 1e-9));
          expect(wCounts.getCell([2]), closeTo(0.3, 1e-9));
          expect(wCounts.getCell([3]), closeTo(0.2, 1e-9));
          expect(wCounts.getCell([7]), closeTo(1.5, 1e-9));
        });
      });

      test('digitize monotonic increasing and decreasing bins', () {
        NDArray.scope(() {
          final x = NDArray.fromList(
            [0.2, 6.4, 3.0, 1.6, -1.0, 15.0],
            [6],
            DType.float64,
          );
          final binsInc = NDArray.fromList(
            [0.0, 1.0, 2.5, 4.0, 10.0],
            [5],
            DType.float64,
          );

          final indsInc = digitize(x, binsInc);
          expect(indsInc.toList(), equals([1, 4, 3, 2, 0, 5]));

          final binsDec = NDArray.fromList(
            [10.0, 4.0, 2.5, 1.0, 0.0],
            [5],
            DType.float64,
          );
          final indsDec = digitize(x, binsDec);
          expect(indsDec.shape, equals([6]));
        });
      });
    });
  });
}
