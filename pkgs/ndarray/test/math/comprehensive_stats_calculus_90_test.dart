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
  group(
    'Comprehensive Workstream 2 Test Suite (Stats, Reductions, DSP, Calculus, Padding, Sets)',
    () {
      // =========================================================================
      // SECTION 1: STATISTICS & DISPERSION
      // =========================================================================
      group('1. Statistics, Reductions & Dispersion', () {
        group('average', () {
          test('1D average with weights: null (equivalent to mean)', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [4],
                DType.float64,
              );
              final res = average(a);
              expect(res.average.shape, <int>[]);
              expect(res.average.scalar.toDouble(), closeTo(2.5, 1e-9));
              expect(res.sumOfWeights, isNull);

              final resWithWeightsReturned = average(a, returned: true);
              expect(
                resWithWeightsReturned.average.scalar.toDouble(),
                closeTo(2.5, 1e-9),
              );
              expect(resWithWeightsReturned.sumOfWeights, isNotNull);
              expect(
                resWithWeightsReturned.sumOfWeights!.scalar.toDouble(),
                4.0,
              );
            });
          });

          test('1D weighted average with 1D weights', () {
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
              // (1*1 + 2*2 + 3*3 + 4*4) / (1 + 2 + 3 + 4) = 30 / 10 = 3.0
              expect(res.average.scalar.toDouble(), closeTo(3.0, 1e-9));
              expect(res.sumOfWeights!.scalar.toDouble(), closeTo(10.0, 1e-9));
            });
          });

          test(
            '2D weighted average along axis 0 and axis 1 with 1D weights',
            () {
              NDArray.scope(() {
                // [[1, 2],
                //  [3, 4]]
                final a = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [2, 2],
                  DType.float64,
                );
                final w0 = NDArray.fromList([1.0, 3.0], [2], DType.float64);

                // Along axis 0 (rows):
                // col 0: (1*1 + 3*3) / 4 = 10 / 4 = 2.5
                // col 1: (2*1 + 4*3) / 4 = 14 / 4 = 3.5
                final res0 = average(a, axis: 0, weights: w0, returned: true);
                expect(res0.average.shape, [2]);
                expect(res0.average.toList(), equals([2.5, 3.5]));
                expect(res0.sumOfWeights!.shape, [2]);
                expect(res0.sumOfWeights!.toList(), equals([4.0, 4.0]));

                // Along axis 1 (columns):
                final w1 = NDArray.fromList([2.0, 1.0], [2], DType.float64);
                // row 0: (1*2 + 2*1) / 3 = 4 / 3 = 1.3333333333333333
                // row 1: (3*2 + 4*1) / 3 = 10 / 3 = 3.3333333333333335
                final res1 = average(a, axis: 1, weights: w1, returned: true);
                expect(res1.average.shape, [2]);
                expect(
                  res1.average.getCell([0]).toDouble(),
                  closeTo(4.0 / 3.0, 1e-9),
                );
                expect(
                  res1.average.getCell([1]).toDouble(),
                  closeTo(10.0 / 3.0, 1e-9),
                );
                expect(res1.sumOfWeights!.toList(), equals([3.0, 3.0]));

                // Negative axis (-1 is axis 1)
                final resNeg = average(a, axis: -1, weights: w1);
                expect(
                  resNeg.average.getCell([0]).toDouble(),
                  closeTo(4.0 / 3.0, 1e-9),
                );
              });
            },
          );

          test('2D weighted average with 2D matching weights', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              final w = NDArray.fromList(
                [4.0, 3.0, 2.0, 1.0],
                [2, 2],
                DType.float64,
              );
              final res0 = average(a, axis: 0, weights: w, returned: true);
              // col 0: (1*4 + 3*2) / (4+2) = 10 / 6 = 1.6666666666666667
              // col 1: (2*3 + 4*1) / (3+1) = 10 / 4 = 2.5
              expect(
                res0.average.getCell([0]).toDouble(),
                closeTo(10.0 / 6.0, 1e-9),
              );
              expect(res0.average.getCell([1]).toDouble(), closeTo(2.5, 1e-9));
              expect(res0.sumOfWeights!.toList(), equals([6.0, 4.0]));
            });
          });

          test('average out buffer reuse and integer promotion', () {
            NDArray.scope(() {
              final aInt = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
              final wInt = NDArray.fromList([1, 1, 1, 1], [4], DType.int32);
              final out = NDArray<Float64>.zeros([], DType.float64);
              final res = average(aInt, weights: wInt, out: out);
              expect(identical(res.average, out), isTrue);
              expect((out.scalar as num).toDouble(), closeTo(2.5, 1e-9));
            });
          });

          test('average error handling', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
              final disp = NDArray.fromList([1.0, 2.0], [2], DType.float64)
                ..dispose();
              expect(() => average(disp), throwsStateError);
              expect(() => average(a, weights: disp), throwsStateError);

              final dispOut = NDArray<Float64>.zeros([], DType.float64)
                ..dispose();
              expect(() => average(a, out: dispOut), throwsStateError);

              expect(() => average(a, axis: 5), throwsArgumentError);

              // 1D weights with 2D array and null axis -> throws
              final a2d = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              final w1d = NDArray.fromList([1.0, 2.0], [2], DType.float64);
              expect(() => average(a2d, weights: w1d), throwsArgumentError);

              // Length mismatch
              final wBad = NDArray.fromList(
                [1.0, 2.0, 3.0],
                [3],
                DType.float64,
              );
              expect(() => average(a, weights: wBad), throwsArgumentError);
              expect(
                () => average(a2d, axis: 0, weights: wBad),
                throwsArgumentError,
              );

              // Shape mismatch for 2D weights
              final w2dBad = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2, 3],
                DType.float64,
              );
              expect(() => average(a2d, weights: w2dBad), throwsArgumentError);

              // Incompatible out buffer shape/dtype
              final outBad = NDArray<Float64>.zeros([5], DType.float64);
              expect(() => average(a, out: outBad), throwsArgumentError);
            });
          });
        });

        group('cov & corrcoef', () {
          test('1D cov (variance of single 1D variable)', () {
            NDArray.scope(() {
              final x = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
              // unbiased sample variance (N-1 = 2): ((0-1)^2 + (1-1)^2 + (2-1)^2) / 2 = 2 / 2 = 1.0
              final c = cov(x);
              expect(c.shape, <int>[]);
              expect((c.scalar as num).toDouble(), closeTo(1.0, 1e-9));

              // bias: true -> ddof=0 -> 2 / 3 = 0.6666666666666666
              final cBiased = cov(x, bias: true);
              expect(
                (cBiased.scalar as num).toDouble(),
                closeTo(2.0 / 3.0, 1e-9),
              );

              // custom ddof: 2 -> denominator = 3 - 2 = 1 -> 2 / 1 = 2.0
              final cDdof2 = cov(x, ddof: 2);
              expect((cDdof2.scalar as num).toDouble(), closeTo(2.0, 1e-9));
            });
          });

          test('2D cov with rowvar: true vs false', () {
            NDArray.scope(() {
              // 2 variables, 3 observations each
              // X = [[0, 1, 2],
              //      [2, 1, 0]]
              final m = NDArray.fromList(
                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                [2, 3],
                DType.float64,
              );
              final cRow = cov(m, rowvar: true);
              expect(cRow.shape, [2, 2]);
              expect(
                (cRow.getCell([0, 0]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );
              expect(
                (cRow.getCell([0, 1]) as num).toDouble(),
                closeTo(-1.0, 1e-9),
              );
              expect(
                (cRow.getCell([1, 0]) as num).toDouble(),
                closeTo(-1.0, 1e-9),
              );
              expect(
                (cRow.getCell([1, 1]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );

              // rowvar: false -> 3 variables, 2 observations each
              final cCol = cov(m, rowvar: false);
              expect(cCol.shape, [3, 3]);
            });
          });

          test('cov with two 1D arrays x and y', () {
            NDArray.scope(() {
              final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              final y = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              final c = cov(x, y: y);
              expect(c.shape, [2, 2]);
              expect((c.getCell([0, 0]) as num).toDouble(), closeTo(1.0, 1e-9));
              expect((c.getCell([0, 1]) as num).toDouble(), closeTo(1.0, 1e-9));
              expect((c.getCell([1, 0]) as num).toDouble(), closeTo(1.0, 1e-9));
              expect((c.getCell([1, 1]) as num).toDouble(), closeTo(1.0, 1e-9));
            });
          });

          test('cov with fweights and aweights', () {
            NDArray.scope(() {
              final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              final fw = NDArray.fromList([2, 1, 1], [3], DType.int64);
              final cFw = cov(x, fweights: fw);
              expect(cFw.shape, <int>[]);
              expect((cFw.scalar as num).isNaN, isFalse);

              final aw = NDArray.fromList([0.5, 1.0, 1.5], [3], DType.float64);
              final cAw = cov(x, aweights: aw);
              expect(cAw.shape, <int>[]);
              expect((cAw.scalar as num).isNaN, isFalse);

              final cBoth = cov(x, fweights: fw, aweights: aw);
              expect(cBoth.shape, <int>[]);
              expect((cBoth.scalar as num).isNaN, isFalse);
            });
          });

          test('cov out buffer reuse and error handling', () {
            NDArray.scope(() {
              final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              final out = NDArray<Float64>.zeros([], DType.float64);
              final res = cov(x, out: out);
              expect(identical(res, out), isTrue);

              final disp = NDArray.fromList([1.0], [1], DType.float64)
                ..dispose();
              expect(() => cov(disp), throwsStateError);
              expect(() => cov(x, y: disp), throwsStateError);
              expect(
                () => cov(
                  x,
                  fweights: NDArray.fromList([1], [1], DType.int64)..dispose(),
                ),
                throwsStateError,
              );
              expect(
                () => cov(
                  x,
                  aweights: NDArray.fromList([1.0], [1], DType.float64)
                    ..dispose(),
                ),
                throwsStateError,
              );

              final empty = NDArray.create([0], DType.float64);
              expect(() => cov(empty), throwsArgumentError);
              expect(() => cov(x, y: empty), throwsArgumentError);

              final t3d = NDArray.create([2, 2, 2], DType.float64);
              expect(() => cov(t3d), throwsArgumentError);

              // Mismatched / negative weights
              final badFw = NDArray.fromList([-1, 1, 1], [3], DType.int64);
              expect(() => cov(x, fweights: badFw), throwsArgumentError);
              final badAw = NDArray.fromList(
                [-0.5, 1.0, 1.0],
                [3],
                DType.float64,
              );
              expect(() => cov(x, aweights: badAw), throwsArgumentError);
              final wrongLenFw = NDArray.fromList([1, 1], [2], DType.int64);
              expect(() => cov(x, fweights: wrongLenFw), throwsArgumentError);
            });
          });

          test('corrcoef for 1D, 2D, and pairs of 1D arrays', () {
            NDArray.scope(() {
              final x = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [4],
                DType.float64,
              );
              final r1 = corrcoef(x);
              expect(r1.shape, <int>[]);
              expect((r1.scalar as num).toDouble(), closeTo(1.0, 1e-9));

              // Constant array -> 0 variance -> NaN
              final constArr = NDArray.fromList(
                [5.0, 5.0, 5.0],
                [3],
                DType.float64,
              );
              final rConst = corrcoef(constArr);
              expect((rConst.scalar as num).isNaN, isTrue);

              // 1D pair
              final y = NDArray.fromList(
                [4.0, 3.0, 2.0, 1.0],
                [4],
                DType.float64,
              );
              final rPair = corrcoef(x, y: y);
              expect(rPair.shape, [2, 2]);
              expect(
                (rPair.getCell([0, 0]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );
              expect(
                (rPair.getCell([0, 1]) as num).toDouble(),
                closeTo(-1.0, 1e-9),
              );
              expect(
                (rPair.getCell([1, 0]) as num).toDouble(),
                closeTo(-1.0, 1e-9),
              );
              expect(
                (rPair.getCell([1, 1]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );

              // 2D array
              final m2d = NDArray.fromList(
                [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                [2, 3],
                DType.float64,
              );
              final r2d = corrcoef(m2d);
              expect(r2d.shape, [2, 2]);
              expect(
                (r2d.getCell([0, 0]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );
              expect(
                (r2d.getCell([0, 1]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );
              expect(
                (r2d.getCell([1, 0]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );
              expect(
                (r2d.getCell([1, 1]) as num).toDouble(),
                closeTo(1.0, 1e-9),
              );

              // Out buffer reuse
              final out = NDArray<Float64>.zeros([], DType.float64);
              final rOut = corrcoef(x, out: out);
              expect(identical(rOut, out), isTrue);
              expect((out.scalar as num).toDouble(), closeTo(1.0, 1e-9));
            });
          });

          test('corrcoef error handling', () {
            NDArray.scope(() {
              final x = NDArray.fromList([1.0, 2.0], [2], DType.float64);
              final disp = NDArray.fromList([1.0, 2.0], [2], DType.float64)
                ..dispose();
              expect(() => corrcoef(disp), throwsStateError);
              expect(() => corrcoef(x, y: disp), throwsStateError);
              expect(
                () => corrcoef(
                  x,
                  out: NDArray<Float64>.zeros([], DType.float64)..dispose(),
                ),
                throwsStateError,
              );
            });
          });
        });

        group(
          'quantile and percentile with all 13 QuantileMethod variants',
          () {
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

            test('1D float64 quantile across all methods', () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [1.0, 2.0, 4.0, 7.0, 11.0],
                  [5],
                  DType.float64,
                );
                for (final method in methods) {
                  final q0 = quantile(a, 0.0, method: method);
                  expect((q0.scalar as num).toDouble(), closeTo(1.0, 1e-9));

                  final q1 = quantile(a, 1.0, method: method);
                  expect((q1.scalar as num).toDouble(), closeTo(11.0, 1e-9));

                  final qHalf = quantile(a, 0.5, method: method);
                  expect((qHalf.scalar as num).isNaN, isFalse);
                }
              });
            });

            test('percentile vs quantile equivalence across methods', () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [10.0, 20.0, 30.0, 40.0, 50.0],
                  [5],
                  DType.float64,
                );
                for (final method in methods) {
                  final p25 = percentile(a, 25.0, method: method);
                  final q25 = quantile(a, 0.25, method: method);
                  expect(
                    (p25.scalar as num).toDouble(),
                    closeTo((q25.scalar as num).toDouble(), 1e-9),
                  );

                  final p75 = percentile(a, 75.0, method: method);
                  final q75 = quantile(a, 0.75, method: method);
                  expect(
                    (p75.scalar as num).toDouble(),
                    closeTo((q75.scalar as num).toDouble(), 1e-9),
                  );
                }
              });
            });

            test('2D and 3D quantile along axes with keepdims', () {
              NDArray.scope(() {
                final a2d = NDArray.fromList(
                  [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
                  [2, 3],
                  DType.float64,
                );
                // axis 0
                final q0 = quantile(a2d, 0.5, axis: 0);
                expect(q0.shape, [3]);
                expect(q0.toList(), equals([1.5, 5.5, 9.5]));

                // axis 0 with keepdims
                final q0Keep = quantile(a2d, 0.5, axis: 0, keepdims: true);
                expect(q0Keep.shape, [1, 3]);
                expect(q0Keep.toList(), equals([1.5, 5.5, 9.5]));

                // axis 1
                final q1 = quantile(a2d, 0.5, axis: 1);
                expect(q1.shape, [2]);
                expect(q1.toList(), equals([5.0, 6.0]));

                // axis -1 with keepdims
                final qNegKeep = quantile(a2d, 0.5, axis: -1, keepdims: true);
                expect(qNegKeep.shape, [2, 1]);
                expect(qNegKeep.toList(), equals([5.0, 6.0]));
              });
            });

            test('quantile across integer and other numeric DTypes', () {
              NDArray.scope(() {
                final aInt32 = NDArray.fromList(
                  [1, 2, 3, 4, 5],
                  [5],
                  DType.int32,
                );
                expect(
                  (quantile(aInt32, 0.5).scalar as num).toDouble(),
                  closeTo(3.0, 1e-9),
                );

                final aInt64 = NDArray.fromList(
                  [10, 20, 30, 40],
                  [4],
                  DType.int64,
                );
                expect(
                  (quantile(aInt64, 0.5).scalar as num).toDouble(),
                  closeTo(25.0, 1e-9),
                );

                final aFloat32 = NDArray.fromList(
                  [1.0, 2.0, 3.0],
                  [3],
                  DType.float32,
                );
                expect(
                  (quantile(aFloat32, 0.5).scalar as num).toDouble(),
                  closeTo(2.0, 1e-5),
                );

                final aUint8 = NDArray.fromList([10, 20, 30], [3], DType.uint8);
                expect(
                  (quantile(aUint8, 0.5).scalar as num).toDouble(),
                  closeTo(20.0, 1e-9),
                );

                final aInt16 = NDArray.fromList(
                  [100, 200, 300],
                  [3],
                  DType.int16,
                );
                expect(
                  (quantile(aInt16, 0.5).scalar as num).toDouble(),
                  closeTo(200.0, 1e-9),
                );

                final aBool = NDArray.fromList(
                  [false, true, true],
                  [3],
                  DType.boolean,
                );
                expect((quantile(aBool, 0.5).scalar as num).isNaN, isFalse);
              });
            });

            test(
              'quantile non-contiguous strided views and out buffer reuse',
              () {
                NDArray.scope(() {
                  final parent = NDArray.fromList(
                    [1.0, 99.0, 3.0, 99.0, 5.0],
                    [5],
                    DType.float64,
                  );
                  final view = parent.slice([
                    const Slice(start: 0, stop: 5, step: 2),
                  ]);
                  expect(view.isContiguous, isFalse);
                  final out = NDArray<Float64>.zeros([], DType.float64);
                  final res = quantile(view, 0.5, out: out);
                  expect(identical(res, out), isTrue);
                  expect((out.scalar as num).toDouble(), closeTo(3.0, 1e-9));
                });
              },
            );

            test('quantile error handling', () {
              NDArray.scope(() {
                final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
                final empty = NDArray.create([0], DType.float64);
                expect(() => quantile(empty, 0.5), throwsArgumentError);
                expect(() => quantile(a, -0.1), throwsArgumentError);
                expect(() => quantile(a, 1.1), throwsArgumentError);
                expect(() => percentile(a, -1.0), throwsArgumentError);
                expect(() => percentile(a, 101.0), throwsArgumentError);
                expect(() => quantile(a, 0.5, axis: 5), throwsArgumentError);

                final disp = NDArray.fromList([1.0], [1], DType.float64)
                  ..dispose();
                expect(() => quantile(disp, 0.5), throwsStateError);
              });
            });
          },
        );

        group('median', () {
          test('1D median with odd and even length', () {
            NDArray.scope(() {
              final odd = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
              expect(
                (median(odd).scalar as num).toDouble(),
                closeTo(2.0, 1e-9),
              );

              final even = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [4],
                DType.float64,
              );
              expect(
                (median(even).scalar as num).toDouble(),
                closeTo(2.5, 1e-9),
              );
            });
          });

          test('2D and 3D median across axes with keepdims', () {
            NDArray.scope(() {
              final a2d = NDArray.fromList(
                [1.0, 6.0, 2.0, 8.0, 3.0, 7.0],
                [2, 3],
                DType.float64,
              );
              final med0 = median(a2d, axis: 0);
              expect(med0.shape, [3]);
              expect(med0.toList(), equals([4.5, 4.5, 4.5]));

              final med1Keep = median(a2d, axis: 1, keepdims: true);
              expect(med1Keep.shape, [2, 1]);
              expect(med1Keep.toList(), equals([2.0, 7.0]));
            });
          });

          test('median across all DTypes including Complex', () {
            NDArray.scope(() {
              final aF32 = NDArray.fromList(
                [1.0, 5.0, 3.0],
                [3],
                DType.float32,
              );
              expect(
                (median(aF32).scalar as num).toDouble(),
                closeTo(3.0, 1e-5),
              );

              final aI64 = NDArray.fromList([10, 50, 30], [3], DType.int64);
              expect(median(aI64).scalar, equals(30));

              final aI32 = NDArray.fromList([10, 50, 30], [3], DType.int32);
              expect(median(aI32).scalar, equals(30));

              final aU8 = NDArray.fromList([10, 50, 30], [3], DType.uint8);
              expect(median(aU8).scalar, equals(30));

              final aI16 = NDArray.fromList([10, 50, 30], [3], DType.int16);
              expect(median(aI16).scalar, equals(30));

              final aCpx128 = NDArray.fromList(
                [Complex(1.0, 2.0), Complex(5.0, 10.0), Complex(3.0, 6.0)],
                [3],
                DType.complex128,
              );
              final medCpx = median(aCpx128);
              expect(medCpx.dtype, DType.complex128);
              expect((medCpx.scalar).real, closeTo(3.0, 1e-9));

              final aCpx64 = NDArray.fromList(
                [Complex(1.0, 2.0), Complex(5.0, 10.0), Complex(3.0, 6.0)],
                [3],
                DType.complex64,
              );
              final medCpx64 = median(aCpx64);
              expect(medCpx64.dtype, DType.complex64);
            });
          });

          test('median out buffer reuse and error handling', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              final out = NDArray<Float64>.zeros([], DType.float64);
              final res = median(a, out: out);
              expect(identical(res, out), isTrue);
              expect((out.scalar as num).toDouble(), closeTo(2.0, 1e-9));

              final empty = NDArray.create([0], DType.float64);
              expect(() => median(empty), throwsArgumentError);

              final disp = NDArray.fromList([1.0], [1], DType.float64)
                ..dispose();
              expect(() => median(disp), throwsStateError);
            });
          });
        });

        group(
          'NaN reductions (nanmin, nanmax, nanmean, nanvar, nanstd, nansum)',
          () {
            test('1D and 2D with all-NaN slices and boundary NaNs', () {
              NDArray.scope(() {
                final allNan = NDArray.fromList(
                  [double.nan, double.nan],
                  [2],
                  DType.float64,
                );
                expect((nanmean(allNan).scalar as num).isNaN, isTrue);
                expect((nanvar(allNan).scalar as num).isNaN, isTrue);
                expect((nanstd(allNan).scalar as num).isNaN, isTrue);
                expect(
                  (nansum(allNan).scalar as num).toDouble(),
                  closeTo(0.0, 1e-9),
                );

                // Leading and trailing NaNs
                // [nan, 2.0, 4.0, nan] -> mean: 3.0, var: 1.0, std: 1.0, sum: 6.0, min: 2.0, max: 4.0
                final mixed = NDArray.fromList(
                  [double.nan, 2.0, 4.0, double.nan],
                  [4],
                  DType.float64,
                );
                expect(
                  (nanmean(mixed).scalar as num).toDouble(),
                  closeTo(3.0, 1e-9),
                );
                expect(
                  (nanvar(mixed).scalar as num).toDouble(),
                  closeTo(1.0, 1e-9),
                );
                expect(
                  (nanstd(mixed).scalar as num).toDouble(),
                  closeTo(1.0, 1e-9),
                );
                expect(
                  (nansum(mixed).scalar as num).toDouble(),
                  closeTo(6.0, 1e-9),
                );
                expect(
                  (nanmin(mixed).scalar as num).toDouble(),
                  closeTo(2.0, 1e-9),
                );
                expect(
                  (nanmax(mixed).scalar as num).toDouble(),
                  closeTo(4.0, 1e-9),
                );
              });
            });

            test('2D NaN reductions along axis 0 and 1 with keepdims', () {
              NDArray.scope(() {
                // [[nan, 2.0, 4.0],
                //  [1.0, nan, 5.0]]
                final a2d = NDArray.fromList(
                  [double.nan, 2.0, 4.0, 1.0, double.nan, 5.0],
                  [2, 3],
                  DType.float64,
                );

                final m0 = nanmean(a2d, axis: 0);
                expect(m0.shape, [3]);
                expect(m0.toList(), equals([1.0, 2.0, 4.5]));

                final m1Keep = nanmean(a2d, axis: 1, keepdims: true);
                expect(m1Keep.shape, [2, 1]);
                expect(
                  (m1Keep.getCell([0, 0]) as num).toDouble(),
                  closeTo(3.0, 1e-9),
                );
                expect(
                  (m1Keep.getCell([1, 0]) as num).toDouble(),
                  closeTo(3.0, 1e-9),
                );

                final v0 = nanvar(a2d, axis: 0);
                expect(v0.shape, [3]);
                expect((v0.getCell([0]) as num).toDouble(), closeTo(0.0, 1e-9));
                expect((v0.getCell([1]) as num).toDouble(), closeTo(0.0, 1e-9));
                expect(
                  (v0.getCell([2]) as num).toDouble(),
                  closeTo(0.25, 1e-9),
                ); // (4-4.5)^2 + (5-4.5)^2 / 2 = 0.5 / 2 = 0.25

                final s0 = nanstd(a2d, axis: 0);
                expect((s0.getCell([2]) as num).toDouble(), closeTo(0.5, 1e-9));

                final min0 = nanmin(a2d, axis: 0);
                expect(min0.toList(), equals([1.0, 2.0, 4.0]));

                final max0 = nanmax(a2d, axis: 0);
                expect(max0.toList(), equals([1.0, 2.0, 5.0]));

                final sum0 = nansum(a2d, axis: 0);
                expect(sum0.toList(), equals([1.0, 2.0, 9.0]));
              });
            });

            test('Complex array nan reductions', () {
              NDArray.scope(() {
                final cpx = NDArray.fromList(
                  [
                    Complex(1.0, 2.0),
                    Complex(double.nan, 0.0),
                    Complex(3.0, 4.0),
                  ],
                  [3],
                  DType.complex128,
                );
                final meanRes = nanmean(cpx);
                expect(meanRes.scalar.real, closeTo(2.0, 1e-9));
                expect(meanRes.scalar.imag, closeTo(3.0, 1e-9));

                final sumRes = nansum(cpx);
                expect((sumRes.scalar).real, closeTo(4.0, 1e-9));
                expect((sumRes.scalar).imag, closeTo(6.0, 1e-9));

                // Complex not supported for nanmin/nanmax
                expect(() => nanmin(cpx), throwsUnsupportedError);
                expect(() => nanmax(cpx), throwsUnsupportedError);
              });
            });

            test('NaN reductions out buffer reuse and error handling', () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [1.0, double.nan, 3.0],
                  [3],
                  DType.float64,
                );
                final out = NDArray<Float64>.zeros([], DType.float64);
                expect(identical(nanmean(a, out: out), out), isTrue);
                expect(identical(nanvar(a, out: out), out), isTrue);
                expect(identical(nanstd(a, out: out), out), isTrue);
                expect(identical(nansum(a, out: out), out), isTrue);
                expect(identical(nanmin(a, out: out), out), isTrue);
                expect(identical(nanmax(a, out: out), out), isTrue);

                final empty = NDArray.create([0], DType.float64);
                expect(() => nanmin(empty), throwsArgumentError);
                expect(() => nanmax(empty), throwsArgumentError);
              });
            });
          },
        );

        group(
          'General Reductions (sum, prod, all, any, min, max, ptp, variance, std, cumsum, cumprod, cummin, cummax)',
          () {
            test('all and any on boolean and numeric arrays', () {
              NDArray.scope(() {
                final bools = NDArray.fromList(
                  [true, true, false],
                  [3],
                  DType.boolean,
                );
                expect(all(bools).scalar, isFalse);
                expect(any(bools).scalar, isTrue);

                final allTrue = NDArray.fromList(
                  [true, true],
                  [2],
                  DType.boolean,
                );
                expect(all(allTrue).scalar, isTrue);

                final allFalse = NDArray.fromList(
                  [false, false],
                  [2],
                  DType.boolean,
                );
                expect(any(allFalse).scalar, isFalse);

                // 2D along axes
                final b2d = NDArray.fromList(
                  [true, false, true, true],
                  [2, 2],
                  DType.boolean,
                );
                expect(all(b2d, axis: 0).toList(), equals([true, false]));
                expect(any(b2d, axis: 1).toList(), equals([true, true]));
              });
            });

            test('sum and prod on empty arrays and various DTypes', () {
              NDArray.scope(() {
                final emptyF64 = NDArray.create([0], DType.float64);
                expect((sum(emptyF64).scalar as num).toDouble(), 0.0);
                expect((prod(emptyF64).scalar as num).toDouble(), 1.0);

                final emptyCpx = NDArray.create([0], DType.complex128);
                expect(sum(emptyCpx).scalar, equals(Complex(0.0, 0.0)));
                expect(prod(emptyCpx).scalar, equals(Complex(1.0, 0.0)));

                final emptyBool = NDArray.create([0], DType.boolean);
                expect(sum(emptyBool).scalar, equals(0));
                expect(prod(emptyBool).scalar, equals(1));
                expect(sumAs(emptyBool, DType.boolean).scalar, isFalse);
                expect(prodAs(emptyBool, DType.boolean).scalar, isTrue);

                final aI32 = NDArray.fromList([2, 3, 4], [3], DType.int32);
                expect(sum(aI32).scalar, 9);
                expect(prod(aI32).scalar, 24);

                final aU8 = NDArray.fromList([2, 3, 4], [3], DType.uint8);
                expect(sum(aU8).scalar, 9);
                expect(prod(aU8).scalar, 24);
              });
            });

            test('ptp, variance/var_, std with axis and ddof', () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
                  [8],
                  DType.float64,
                );
                // Mean = 5.0, Var = 4.0 (ddof=0)
                expect(
                  (variance(a, ddof: 0).scalar as num).toDouble(),
                  closeTo(4.0, 1e-9),
                );
                expect(
                  (var_(a, ddof: 0).scalar as num).toDouble(),
                  closeTo(4.0, 1e-9),
                );
                expect(
                  (std(a, ddof: 0).scalar as num).toDouble(),
                  closeTo(2.0, 1e-9),
                );
                expect((ptp(a).scalar as num).toDouble(), closeTo(7.0, 1e-9));

                // ddof = 1 -> 32 / 7 = 4.571428571428571
                expect(
                  (variance(a, ddof: 1).scalar as num).toDouble(),
                  closeTo(32.0 / 7.0, 1e-9),
                );
                expect(
                  (std(a, ddof: 1).scalar as num).toDouble(),
                  closeTo(math.sqrt(32.0 / 7.0), 1e-9),
                );

                // Out buffer reuse
                final out = NDArray<Float64>.zeros([], DType.float64);
                expect(identical(std(a, out: out), out), isTrue);
                expect(identical(variance(a, out: out), out), isTrue);
                expect(identical(ptp(a, out: out), out), isTrue);
              });
            });

            test('cumsum, cumprod, cummin, cummax in 1D and 2D', () {
              NDArray.scope(() {
                final a = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [4],
                  DType.float64,
                );
                expect(cumsum(a).toList(), equals([1.0, 3.0, 6.0, 10.0]));
                expect(cumprod(a).toList(), equals([1.0, 2.0, 6.0, 24.0]));

                final mix = NDArray.fromList(
                  [3.0, 1.0, 4.0, 2.0],
                  [4],
                  DType.float64,
                );
                expect(cummin(mix).toList(), equals([3.0, 1.0, 1.0, 1.0]));
                expect(cummax(mix).toList(), equals([3.0, 3.0, 4.0, 4.0]));

                // 2D axis 0
                final m2d = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0],
                  [2, 2],
                  DType.float64,
                );
                expect(
                  cumsum(m2d, axis: 0).toList(),
                  equals([1.0, 2.0, 4.0, 6.0]),
                );
                expect(
                  cumprod(m2d, axis: 0).toList(),
                  equals([1.0, 2.0, 3.0, 8.0]),
                );
                expect(
                  cummin(m2d, axis: 0).toList(),
                  equals([1.0, 2.0, 1.0, 2.0]),
                );
                expect(
                  cummax(m2d, axis: 0).toList(),
                  equals([1.0, 2.0, 3.0, 4.0]),
                );

                // 2D axis 1
                expect(
                  cumsum(m2d, axis: 1).toList(),
                  equals([1.0, 3.0, 3.0, 7.0]),
                );
                expect(
                  cumprod(m2d, axis: 1).toList(),
                  equals([1.0, 2.0, 3.0, 12.0]),
                );

                // Boolean cumsumAs -> returns int32
                final b = NDArray.fromList(
                  [true, false, true],
                  [3],
                  DType.boolean,
                );
                final cB = cumsumAs(b, DType.int32);
                expect(cB.dtype, DType.int32);
                expect(cB.toList(), equals([1, 1, 2]));
              });
            });
          },
        );
      });

      // =========================================================================
      // SECTION 2: CALCULUS & NUMERICAL DIFFERENTIATION / INTEGRATION
      // =========================================================================
      group('2. Calculus & Integration', () {
        group('gradient & gradientArray', () {
          test('1D gradient with edgeOrder: 1 and edgeOrder: 2', () {
            NDArray.scope(() {
              // f(x) = x^2 at x = [0, 1, 2, 3, 4] -> f = [0, 1, 4, 9, 16]
              // Exact f prime (x) = 2x -> [0, 2, 4, 6, 8]
              final f = NDArray.fromList(
                [0.0, 1.0, 4.0, 9.0, 16.0],
                [5],
                DType.float64,
              );

              // edgeOrder: 1
              final g1 = gradient(f, edgeOrder: 1);
              // x0: (1-0)/1 = 1.0; x1: (4-0)/2 = 2.0; x2: (9-1)/2 = 4.0; x3: (16-4)/2 = 6.0; x4: (16-9)/1 = 7.0
              expect(g1.toList(), equals([1.0, 2.0, 4.0, 6.0, 7.0]));

              // edgeOrder: 2 (second-order accurate boundaries)
              // Exact for quadratic!
              final g2 = gradient(f, edgeOrder: 2);
              expect(g2.toList(), equals([0.0, 2.0, 4.0, 6.0, 8.0]));
            });
          });

          test('gradient with Spacing.step and Spacing.coordinates', () {
            NDArray.scope(() {
              final f = NDArray.fromList([0.0, 4.0, 16.0], [3], DType.float64);

              // Spacing.step(2.0)
              final gStep = gradient(
                f,
                spacing: const Spacing.step(2.0),
                edgeOrder: 2,
              );
              expect(gStep.shape, [3]);

              // CoordinateSpacing non-uniform
              final coords = const Spacing.coordinates([0.0, 1.0, 3.0]);
              final fNonUniform = NDArray.fromList(
                [0.0, 1.0, 9.0],
                [3],
                DType.float64,
              );
              final gCoords = gradient(
                fNonUniform,
                spacing: coords,
                edgeOrder: 1,
              );
              expect(gCoords.shape, [3]);
              expect(gCoords.toList(), isNotNull);
            });
          });

          test('2D, 3D, 4D gradient along specific axes', () {
            NDArray.scope(() {
              // 2D array
              final f2d = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2, 3],
                DType.float64,
              );
              final gAxis0 = gradient(f2d, axis: 0);
              expect(gAxis0.shape, [2, 3]);
              expect(gAxis0.toList(), equals([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]));

              final gAxis1 = gradient(f2d, axis: 1, edgeOrder: 2);
              expect(gAxis1.shape, [2, 3]);
              expect(gAxis1.toList(), equals([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));

              // 3D array
              final f3d = NDArray.fromList(
                List<double>.generate(27, (i) => i.toDouble()),
                [3, 3, 3],
                DType.float64,
              );
              final g3d = gradient(f3d, axis: 2);
              expect(g3d.shape, [3, 3, 3]);

              // 4D array
              final f4d = NDArray.fromList(
                List<double>.generate(16, (i) => i.toDouble()),
                [2, 2, 2, 2],
                DType.float64,
              );
              final g4d = gradient(f4d, axis: 3);
              expect(g4d.shape, [2, 2, 2, 2]);
            });
          });

          test('gradient with Complex numbers and complex spacing', () {
            NDArray.scope(() {
              final fCpx = NDArray.fromList(
                [Complex(0, 0), Complex(1, 2), Complex(4, 8)],
                [3],
                DType.complex128,
              );
              final gCpx = gradient(
                fCpx,
                spacing: const Spacing.step(1.0),
                edgeOrder: 2,
              );
              expect(gCpx.dtype, DType.complex128);

              final gCpxStep = gradient(
                fCpx,
                spacing: Spacing.step(Complex(0.0, 1.0)),
                edgeOrder: 1,
              );
              expect(gCpxStep.dtype, DType.complex128);

              final fCpx64 = NDArray.fromList(
                [Complex(0, 0), Complex(1, 2), Complex(4, 8)],
                [3],
                DType.complex64,
              );
              final gCpx64 = gradient(fCpx64, edgeOrder: 1);
              expect(gCpx64.dtype, DType.complex64);
            });
          });

          test(
            'gradientArray on multi-axis with spacing and spacings list',
            () {
              NDArray.scope(() {
                final f = NDArray.fromList(
                  [1.0, 2.0, 4.0, 8.0],
                  [2, 2],
                  DType.float64,
                );
                // Default all axes
                final gradsAll = gradientArray(f);
                expect(gradsAll.length, 2);
                expect(gradsAll[0].shape, [2, 2]);
                expect(gradsAll[1].shape, [2, 2]);

                // Specified subset of axes with spacings list
                final gradsSubset = gradientArray(
                  f,
                  axis: [0, 1],
                  spacings: [const Spacing.step(1.0), const Spacing.step(2.0)],
                );
                expect(gradsSubset.length, 2);

                // Out list reuse
                final out0 = NDArray<Float64>.zeros([2, 2], DType.float64);
                final out1 = NDArray<Float64>.zeros([2, 2], DType.float64);
                final gradsOut = gradientArray(f, out: [out0, out1]);
                expect(identical(gradsOut[0], out0), isTrue);
                expect(identical(gradsOut[1], out1), isTrue);
              });
            },
          );

          test('gradient error handling', () {
            NDArray.scope(() {
              final f = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

              final disp = NDArray.fromList([1.0, 2.0], [2], DType.float64)
                ..dispose();
              expect(() => gradient(disp), throwsStateError);
              expect(() => gradientArray(disp), throwsStateError);

              // Boolean arrays not supported
              final boolArr = NDArray.fromList(
                [true, false, true],
                [3],
                DType.boolean,
              );
              expect(() => gradient(boolArr), throwsArgumentError);
              expect(() => gradientArray(boolArr), throwsArgumentError);

              // Integer arrays promoted to float64
              final intArr = NDArray.fromList([1, 2, 4], [3], DType.int32);
              final gradInt = gradient(intArr);
              expect(gradInt.dtype, DType.float64);
              expect(gradInt.toList(), equals([1.0, 1.5, 2.0]));
              final gradArrInt = gradientArray(intArr);
              expect(gradArrInt[0].dtype, DType.float64);
              expect(gradArrInt[0].toList(), equals([1.0, 1.5, 2.0]));

              // edgeOrder must be 1 or 2
              expect(() => gradient(f, edgeOrder: 3), throwsArgumentError);

              // Axis out of bounds
              expect(() => gradient(f, axis: 5), throwsArgumentError);

              // Dimension too small for edgeOrder
              final smallArr = NDArray.fromList([1.0, 2.0], [2], DType.float64);
              expect(
                () => gradient(smallArr, edgeOrder: 2),
                throwsArgumentError,
              );

              // Mismatched CoordinateSpacing length
              expect(
                () =>
                    gradient(f, spacing: const Spacing.coordinates([0.0, 1.0])),
                throwsArgumentError,
              );

              // Both spacing and spacings provided
              expect(
                () => gradientArray(
                  f,
                  spacing: const Spacing.step(1.0),
                  spacings: [const Spacing.step(1.0)],
                ),
                throwsArgumentError,
              );

              // Duplicate axes in gradientArray
              expect(() => gradientArray(f, axis: [0, 0]), throwsArgumentError);
            });
          });
        });

        group('diff (Numerical Differences)', () {
          test('diff with n=0, 1, 2, 3 and n >= size', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 4.0, 7.0, 11.0],
                [5],
                DType.float64,
              );

              // n = 0 -> returns copy
              final d0 = diff(a, n: 0);
              expect(d0.toList(), equals([1.0, 2.0, 4.0, 7.0, 11.0]));

              // n = 1 -> [1, 2, 3, 4]
              final d1 = diff(a, n: 1);
              expect(d1.shape, [4]);
              expect(d1.toList(), equals([1.0, 2.0, 3.0, 4.0]));

              // n = 2 -> [1, 1, 1]
              final d2 = diff(a, n: 2);
              expect(d2.shape, [3]);
              expect(d2.toList(), equals([1.0, 1.0, 1.0]));

              // n = 3 -> [0, 0]
              final d3 = diff(a, n: 3);
              expect(d3.shape, [2]);
              expect(d3.toList(), equals([0.0, 0.0]));

              // n >= size -> empty shape [0]
              final d5 = diff(a, n: 5);
              expect(d5.shape, [0]);
            });
          });

          test('2D diff across axis 0, axis 1, axis -1', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 3.0, 6.0, 2.0, 5.0, 9.0],
                [2, 3],
                DType.float64,
              );
              // axis 0 (rows): [2-1, 5-3, 9-6] = [1, 2, 3]
              final d0 = diff(a, axis: 0);
              expect(d0.shape, [1, 3]);
              expect(d0.toList(), equals([1.0, 2.0, 3.0]));

              // axis 1 (cols): [[2, 3], [3, 4]]
              final d1 = diff(a, axis: 1);
              expect(d1.shape, [2, 2]);
              expect(d1.toList(), equals([2.0, 3.0, 3.0, 4.0]));

              final dNeg = diff(a, axis: -1);
              expect(dNeg.shape, [2, 2]);
              expect(dNeg.toList(), equals([2.0, 3.0, 3.0, 4.0]));
            });
          });

          test(
            'diff across various DTypes (int32, int64, float32, complex128)',
            () {
              NDArray.scope(() {
                final aInt = NDArray.fromList([1, 4, 9], [3], DType.int32);
                expect(diff(aInt).toList(), equals([3, 5]));

                final aInt64 = NDArray.fromList([10, 30, 60], [3], DType.int64);
                expect(diff(aInt64).toList(), equals([20, 30]));

                final aFloat32 = NDArray.fromList(
                  [1.0, 3.0, 6.0],
                  [3],
                  DType.float32,
                );
                expect(diff(aFloat32).toList(), equals([2.0, 3.0]));

                final aCpx = NDArray.fromList(
                  [Complex(1, 1), Complex(3, 4), Complex(6, 8)],
                  [3],
                  DType.complex128,
                );
                final dCpx = diff(aCpx);
                expect(dCpx.getCell([0]), equals(Complex(2.0, 3.0)));
                expect(dCpx.getCell([1]), equals(Complex(3.0, 4.0)));
              });
            },
          );

          test('diff out buffer reuse and error handling', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1.0, 3.0, 6.0], [3], DType.float64);
              final out = NDArray<Float64>.zeros([2], DType.float64);
              final res = diff(a, out: out);
              expect(identical(res, out), isTrue);
              expect(out.toList(), equals([2.0, 3.0]));

              final disp = NDArray.fromList([1.0], [1], DType.float64)
                ..dispose();
              expect(() => diff(disp), throwsStateError);
              expect(() => diff(a, n: -1), throwsArgumentError);
              expect(() => diff(a, axis: 5), throwsArgumentError);
            });
          });
        });

        group('trapz (Composite Trapezoidal Rule)', () {
          test('4D trapz along arbitrary axes', () {
            NDArray.scope(() {
              final t4d = NDArray.fromList(
                List<double>.generate(16, (i) => i.toDouble()),
                [2, 2, 2, 2],
                DType.float64,
              );
              final resAxis0 = trapz(t4d, axis: 0);
              expect(resAxis0.shape, [2, 2, 2]);

              final resAxis3 = trapz(t4d, axis: 3);
              expect(resAxis3.shape, [2, 2, 2]);
            });
          });

          test('trapz with non-contiguous slice view and out buffer', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 99.0, 2.0, 99.0, 3.0],
                [5],
                DType.float64,
              );
              final slice = a.slice([const Slice(start: 0, stop: 5, step: 2)]);
              expect(slice.isContiguous, isFalse);

              final out = NDArray<Float64>.zeros([], DType.float64);
              final res = trapz(slice, out: out);
              expect(identical(res, out), isTrue);
              expect(
                (out.scalar as num).toDouble(),
                closeTo(4.0, 1e-9),
              ); // (1+2)/2 * 1 + (2+3)/2 * 1 = 1.5 + 2.5 = 4.0
            });
          });
        });
      });

      // =========================================================================
      // SECTION 3: DSP & PADDING
      // =========================================================================
      group('3. DSP & Padding', () {
        group('angle & unwrap', () {
          test('angle on all 4 quadrants for complex64 and complex128', () {
            NDArray.scope(() {
              final c128 = NDArray.fromList(
                [
                  Complex(1.0, 1.0), // pi/4
                  Complex(-1.0, 1.0), // 3pi/4
                  Complex(-1.0, -1.0), // -3pi/4
                  Complex(1.0, -1.0), // -pi/4
                  Complex(1.0, 0.0), // 0
                  Complex(0.0, 1.0), // pi/2
                ],
                [6],
                DType.complex128,
              );
              final ang128 = angle(c128);
              expect(ang128.dtype, DType.float64);
              expect(
                (ang128.getCell([0]) as num).toDouble(),
                closeTo(math.pi / 4.0, 1e-9),
              );
              expect(
                (ang128.getCell([1]) as num).toDouble(),
                closeTo(3.0 * math.pi / 4.0, 1e-9),
              );
              expect(
                (ang128.getCell([2]) as num).toDouble(),
                closeTo(-3.0 * math.pi / 4.0, 1e-9),
              );
              expect(
                (ang128.getCell([3]) as num).toDouble(),
                closeTo(-math.pi / 4.0, 1e-9),
              );
              expect(
                (ang128.getCell([4]) as num).toDouble(),
                closeTo(0.0, 1e-9),
              );
              expect(
                (ang128.getCell([5]) as num).toDouble(),
                closeTo(math.pi / 2.0, 1e-9),
              );

              // Complex64
              final c64 = NDArray.fromList(
                [Complex(1.0, 1.0)],
                [1],
                DType.complex64,
              );
              final ang64 = angle(c64);
              expect(ang64.dtype, DType.float32);
              expect(
                (ang64.getCell([0]) as num).toDouble(),
                closeTo(math.pi / 4.0, 1e-5),
              );

              // Non-contiguous strided angle
              final strided = c128.slice([
                const Slice(start: 0, stop: 6, step: 2),
              ]);
              expect(strided.isContiguous, isFalse);
              final angStrided = angle(strided);
              expect(angStrided.shape, [3]);
            });
          });

          test('unwrap along 1D and 2D with custom discont', () {
            NDArray.scope(() {
              final phase = NDArray.fromList(
                [0.0, math.pi + 0.5, -(math.pi + 0.5)],
                [3],
                DType.float64,
              );
              final unwrapped = unwrap(phase);
              expect(unwrapped.shape, [3]);

              // 2D unwrap along axis 0 and axis 1
              final p2d = NDArray.fromList(
                [0.0, math.pi + 0.2, 0.0, math.pi + 0.2],
                [2, 2],
                DType.float64,
              );
              final u0 = unwrap(p2d, axis: 0);
              expect(u0.shape, [2, 2]);

              final u1 = unwrap(p2d, axis: 1);
              expect(u1.shape, [2, 2]);

              // Float32 unwrap
              final p32 = NDArray.fromList([0.0, 3.5], [2], DType.float32);
              final u32 = unwrap(p32);
              expect(u32.dtype, DType.float32);
            });
          });

          test('angle & unwrap error handling', () {
            NDArray.scope(() {
              final boolArr = NDArray.fromList(
                [true, false],
                [2],
                DType.boolean,
              );
              expect(() => angle(boolArr as dynamic), throwsArgumentError);
              expect(() => unwrap(boolArr as dynamic), throwsArgumentError);

              final realArr = NDArray.fromList([1.0, 2.0], [2], DType.float64);
              final angleRes = angle(realArr);
              expect(angleRes.shape, [2]);
              expect(angleRes.dtype, DType.float64);

              final intArr = NDArray.fromList([1, 2], [2], DType.int32);
              final unwrapRes = unwrap(intArr);
              expect(unwrapRes.shape, [2]);
              expect(unwrapRes.dtype, DType.float64);

              expect(() => unwrap(realArr, axis: 5), throwsArgumentError);
            });
          });
        });

        group('convolve, correlate & convolve2d', () {
          test(
            '1D convolve and correlate in full, valid, same modes for Float64',
            () {
              NDArray.scope(() {
                // in1 = [1, 2, 3], in2 = [0, 1, 0.5]
                final in1 = NDArray.fromList(
                  [1.0, 2.0, 3.0],
                  [3],
                  DType.float64,
                );
                final in2 = NDArray.fromList(
                  [0.0, 1.0, 0.5],
                  [3],
                  DType.float64,
                );

                // full convolution: len = 3 + 3 - 1 = 5
                final cFull = convolve(in1, in2, mode: ConvMode.full);
                expect(cFull.shape, [5]);

                // valid convolution: len = 3 - 3 + 1 = 1
                final cValid = convolve(in1, in2, mode: ConvMode.valid);
                expect(cValid.shape, [1]);

                // same convolution: len = 3
                final cSame = convolve(in1, in2, mode: ConvMode.same);
                expect(cSame.shape, [3]);

                // correlate full, valid, same
                final corrFull = correlate(in1, in2, mode: ConvMode.full);
                expect(corrFull.shape, [5]);

                final corrValid = correlate(in1, in2, mode: ConvMode.valid);
                expect(corrValid.shape, [1]);

                final corrSame = correlate(in1, in2, mode: ConvMode.same);
                expect(corrSame.shape, [3]);
              });
            },
          );

          test(
            '1D convolve across Float32, Int32, Int64, Complex64, Complex128',
            () {
              NDArray.scope(() {
                final f32_1 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
                final f32_2 = NDArray.fromList([0.5, 1.0], [2], DType.float32);
                expect(
                  convolve(f32_1, f32_2, mode: ConvMode.valid).dtype,
                  DType.float32,
                );

                final i32_1 = NDArray.fromList([1, 2, 3], [3], DType.int32);
                final i32_2 = NDArray.fromList([1, 1], [2], DType.int32);
                expect(
                  convolve(i32_1, i32_2, mode: ConvMode.valid).toList(),
                  equals([3, 5]),
                );

                final i64_1 = NDArray.fromList([10, 20, 30], [3], DType.int64);
                final i64_2 = NDArray.fromList([1, 2], [2], DType.int64);
                expect(
                  convolve(i64_1, i64_2, mode: ConvMode.valid).toList(),
                  equals([40, 70]),
                );

                final c64_1 = NDArray.fromList(
                  [Complex(1, 0), Complex(2, 0)],
                  [2],
                  DType.complex64,
                );
                final c64_2 = NDArray.fromList(
                  [Complex(0, 1)],
                  [1],
                  DType.complex64,
                );
                final resC64 = convolve(c64_1, c64_2, mode: ConvMode.valid);
                expect(resC64.dtype, DType.complex64);

                final c128_1 = NDArray.fromList(
                  [Complex(1, 1), Complex(2, 2)],
                  [2],
                  DType.complex128,
                );
                final c128_2 = NDArray.fromList(
                  [Complex(1, 0), Complex(0, 1)],
                  [2],
                  DType.complex128,
                );
                final resC128 = convolve(c128_1, c128_2, mode: ConvMode.valid);
                expect(resC128.dtype, DType.complex128);
              });
            },
          );

          test('2D convolve2d in full, valid, same modes', () {
            NDArray.scope(() {
              final img = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [3, 3],
                DType.float64,
              );
              final kernel = NDArray.fromList(
                [1.0, 0.0, 0.0, 1.0],
                [2, 2],
                DType.float64,
              );

              final convFull = convolve2d(img, kernel, mode: ConvMode.full);
              expect(convFull.shape, [4, 4]);

              final convValid = convolve2d(img, kernel, mode: ConvMode.valid);
              expect(convValid.shape, [2, 2]);

              final convSame = convolve2d(img, kernel, mode: ConvMode.same);
              expect(convSame.shape, [3, 3]);
            });
          });

          test('convolve & correlate error handling', () {
            NDArray.scope(() {
              final a1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
              final a2 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

              // valid mode when in1 < in2
              expect(
                () => convolve(a1, a2, mode: ConvMode.valid),
                throwsArgumentError,
              );
              expect(
                () => correlate(a1, a2, mode: ConvMode.valid),
                throwsArgumentError,
              );

              // Mismatched dtypes
              final aInt = NDArray.fromList([1, 2], [2], DType.int32);
              expect(() => correlate(a1, aInt as dynamic), throwsArgumentError);

              // Mismatched ranks
              final a2d = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              expect(() => convolve(a1, a2d), throwsArgumentError);
            });
          });
        });

        group('pad with all 10 PadModes across 1D, 2D, 3D and various DTypes', () {
          final padModes = [
            PadMode.constant,
            PadMode.edge,
            PadMode.reflect,
            PadMode.symmetric,
            PadMode.wrap,
            PadMode.linearRamp,
            PadMode.mean,
            PadMode.median,
            PadMode.min,
            PadMode.max,
          ];

          test('1D array with all 10 pad modes and asymmetric widths', () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5],
                DType.float64,
              );
              final pw = PadWidth.axes([
                (2, 3),
              ]); // pad 2 before, 3 after -> total size 10

              for (final mode in padModes) {
                final padded = pad(
                  a,
                  pw,
                  mode: mode,
                  constantValues: PadValues.all(0.0),
                  endValues: PadValues.all(0.0),
                  statLength: StatLength.all(2),
                );
                expect(padded.shape, [10]);
                // Core elements preserved in interior
                expect(
                  (padded.getCell([2]) as num).toDouble(),
                  closeTo(1.0, 1e-9),
                );
                expect(
                  (padded.getCell([3]) as num).toDouble(),
                  closeTo(2.0, 1e-9),
                );
                expect(
                  (padded.getCell([4]) as num).toDouble(),
                  closeTo(3.0, 1e-9),
                );
                expect(
                  (padded.getCell([5]) as num).toDouble(),
                  closeTo(4.0, 1e-9),
                );
                expect(
                  (padded.getCell([6]) as num).toDouble(),
                  closeTo(5.0, 1e-9),
                );
              }
            });
          });

          test(
            '2D array with all 10 pad modes and fast native path vs axis-by-axis',
            () {
              NDArray.scope(() {
                final m = NDArray.fromList(
                  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                  [2, 3],
                  DType.float64,
                );
                // pad (top: 1, bottom: 2), (left: 2, right: 1) -> shape [2+1+2, 3+2+1] = [5, 6]
                final pw = PadWidth.axes([(1, 2), (2, 1)]);

                for (final mode in padModes) {
                  final padded = pad(
                    m,
                    pw,
                    mode: mode,
                    constantValues: PadValues.all(0.0),
                    endValues: PadValues.all(0.0),
                    statLength: StatLength.all(2),
                  );
                  expect(padded.shape, [5, 6]);
                  expect(
                    (padded.getCell([1, 2]) as num).toDouble(),
                    closeTo(1.0, 1e-9),
                  );
                }
              });
            },
          );

          test('3D array padding with uniform and per-axis PadWidth', () {
            NDArray.scope(() {
              final t3d = NDArray.fromList(
                List<double>.generate(8, (i) => i.toDouble() + 1.0),
                [2, 2, 2],
                DType.float64,
              );
              final pwUniform = PadWidth.all(1, 1);
              final pUniform = pad(t3d, pwUniform, mode: PadMode.edge);
              expect(pUniform.shape, [4, 4, 4]);

              final pwAxes = PadWidth.axes([(1, 0), (0, 1), (1, 1)]);
              final pAxes = pad(t3d, pwAxes, mode: PadMode.wrap);
              expect(pAxes.shape, [3, 3, 4]);
            });
          });

          test('pad across all DTypes', () {
            NDArray.scope(() {
              // Int32
              final aI32 = NDArray.fromList([1, 2, 3], [3], DType.int32);
              expect(pad(aI32, PadWidth.all(1), mode: PadMode.constant).shape, [
                5,
              ]);

              // Int64
              final aI64 = NDArray.fromList([10, 20], [2], DType.int64);
              expect(pad(aI64, PadWidth.all(1), mode: PadMode.edge).shape, [4]);

              // Float32
              final aF32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
              expect(pad(aF32, PadWidth.all(1), mode: PadMode.reflect).shape, [
                4,
              ]);

              // Uint8
              final aU8 = NDArray.fromList([10, 20], [2], DType.uint8);
              expect(pad(aU8, PadWidth.all(1), mode: PadMode.symmetric).shape, [
                4,
              ]);

              // Boolean
              final aBool = NDArray.fromList([true, false], [2], DType.boolean);
              expect(
                pad(aBool, PadWidth.all(1), mode: PadMode.constant).shape,
                [4],
              );

              // Complex128
              final aCpx128 = NDArray.fromList(
                [Complex(1, 2), Complex(3, 4)],
                [2],
                DType.complex128,
              );
              final pCpx128 = pad(
                aCpx128,
                PadWidth.all(1),
                mode: PadMode.constant,
              );
              expect(pCpx128.dtype, DType.complex128);
              expect(pCpx128.shape, [4]);

              // Complex64
              final aCpx64 = NDArray.fromList(
                [Complex(1, 2), Complex(3, 4)],
                [2],
                DType.complex64,
              );
              final pCpx64 = pad(aCpx64, PadWidth.all(1), mode: PadMode.edge);
              expect(pCpx64.dtype, DType.complex64);
              expect(pCpx64.shape, [4]);
            });
          });

          test('pad zero padding bypass and out buffer reuse', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
              // Zero padding
              final pZero = pad(a, PadWidth.all(0));
              expect(pZero.toList(), equals([1.0, 2.0, 3.0]));

              // Out buffer
              final out = NDArray<Float64>.zeros([5], DType.float64);
              final res = pad(
                a,
                PadWidth.all(1),
                mode: PadMode.constant,
                out: out,
              );
              expect(identical(res, out), isTrue);
              expect(out.shape, [5]);
            });
          });

          test('pad error handling', () {
            NDArray.scope(() {
              expect(() => PadWidth.all(-1), throwsRangeError);
              expect(() => StatLength.all(0), throwsArgumentError);

              final a0d = NDArray.fromList([1.0], [], DType.float64);
              expect(() => pad(a0d, PadWidth.all(1)), throwsArgumentError);

              final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
              // Mismatched padwidth rank
              final pwMismatched = PadWidth.axes([(1, 1), (1, 1)]);
              expect(() => pad(a, pwMismatched), throwsArgumentError);

              final disp = NDArray.fromList([1.0], [1], DType.float64)
                ..dispose();
              expect(() => pad(disp, PadWidth.all(1)), throwsStateError);
            });
          });
        });
      });

      // =========================================================================
      // SECTION 4: SET OPERATIONS
      // =========================================================================
      group('4. Set Operations', () {
        group('unique', () {
          test(
            'unique on 1D with returnIndex, returnInverse, returnCounts',
            () {
              NDArray.scope(() {
                // [3, 1, 2, 3, 1, 4] -> unique sorted: [1, 2, 3, 4]
                // indices: [1, 2, 0, 5]
                // inverse: [2, 0, 1, 2, 0, 3]
                // counts:  [2, 1, 2, 1]
                final a = NDArray.fromList(
                  [3, 1, 2, 3, 1, 4],
                  [6],
                  DType.int32,
                );

                // Simple unique
                final u = unique(a);
                expect(u.toList(), equals([1, 2, 3, 4]));

                // With all 3 flags
                final res = uniqueAll(a);
                expect(res.values.toList(), equals([1, 2, 3, 4]));
                expect(res.index.toList(), equals([1, 2, 0, 5]));
                expect(res.inverse.toList(), equals([2, 0, 1, 2, 0, 3]));
                expect(res.counts.toList(), equals([2, 1, 2, 1]));
              });
            },
          );

          test('unique on empty arrays and already unique arrays', () {
            NDArray.scope(() {
              final empty = NDArray.create([0], DType.float64);
              final uEmpty = unique(empty);
              expect(uEmpty.shape, [0]);

              final uEmptyAll = uniqueAll(empty);
              expect(uEmptyAll.values.shape, [0]);
              expect(uEmptyAll.index.shape, [0]);
              expect(uEmptyAll.inverse.shape, [0]);
              expect(uEmptyAll.counts.shape, [0]);

              // Multi-dimensional array (flattened)
              final a2d = NDArray.fromList(
                [2.0, 1.0, 3.0, 2.0],
                [2, 2],
                DType.float64,
              );
              final u2d = unique(a2d);
              expect(u2d.toList(), equals([1.0, 2.0, 3.0]));
            });
          });

          test(
            'unique across various DTypes (int64, float32, uint8, boolean)',
            () {
              NDArray.scope(() {
                final aI64 = NDArray.fromList(
                  [10, 30, 20, 10],
                  [4],
                  DType.int64,
                );
                expect(unique(aI64).toList(), equals([10, 20, 30]));

                final aF32 = NDArray.fromList(
                  [3.0, 1.0, 2.0, 1.0],
                  [4],
                  DType.float32,
                );
                expect(unique(aF32).toList(), equals([1.0, 2.0, 3.0]));

                final aU8 = NDArray.fromList(
                  [255, 0, 128, 0],
                  [4],
                  DType.uint8,
                );
                expect(unique(aU8).toList(), equals([0, 128, 255]));

                final aBool = NDArray.fromList(
                  [true, false, true],
                  [3],
                  DType.boolean,
                );
                expect(unique(aBool).toList(), equals([false, true]));
              });
            },
          );

          test('unique out buffer reuse and error handling', () {
            NDArray.scope(() {
              final a = NDArray.fromList([3, 1, 2, 1], [4], DType.int32);
              final out = NDArray<Int32>.zeros([3], DType.int32);
              final res = unique(a, out: out);
              expect(identical(res, out), isTrue);
              expect(out.toList(), equals([1, 2, 3]));

              final disp = NDArray.fromList([1], [1], DType.int32)..dispose();
              expect(() => unique(disp), throwsStateError);
            });
          });
        });

        group('intersect1d, union1d, setdiff1d, setxor1d & isin', () {
          test('intersect1d with assumeUnique: false vs true', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1, 2, 2, 3, 4], [5], DType.int32);
              final b = NDArray.fromList([2, 4, 6], [3], DType.int32);

              final inter = intersect1d(a, b);
              expect(inter.toList(), equals([2, 4]));

              final uA = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
              final uB = NDArray.fromList([2, 4, 6], [3], DType.int32);
              final interAssume = intersect1d(uA, uB, assumeUnique: true);
              expect(interAssume.toList(), equals([2, 4]));

              // Disjoint arrays -> empty [0]
              final c = NDArray.fromList([10, 20], [2], DType.int32);
              expect(intersect1d(a, c).shape, [0]);
            });
          });

          test('setdiff1d with disjoint, subset, and identical arrays', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1, 2, 3, 4, 5], [5], DType.int32);
              final b = NDArray.fromList([2, 4], [2], DType.int32);

              final diffRes = setdiff1d(a, b);
              expect(diffRes.toList(), equals([1, 3, 5]));

              // Identical -> empty [0]
              expect(setdiff1d(a, a).shape, [0]);

              // Disjoint -> all of a
              final c = NDArray.fromList([10, 20], [2], DType.int32);
              expect(setdiff1d(a, c).toList(), equals([1, 2, 3, 4, 5]));
            });
          });

          test('setxor1d with overlapping, identical, and disjoint arrays', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
              final b = NDArray.fromList([2, 3, 4], [3], DType.int32);
              // XOR -> elements in only one array -> [1, 4]
              expect(setxor1d(a, b).toList(), equals([1, 4]));

              // Identical -> empty [0]
              expect(setxor1d(a, a).shape, [0]);

              // Disjoint -> union [1, 2, 3, 10, 20]
              final c = NDArray.fromList([10, 20], [2], DType.int32);
              expect(setxor1d(a, c).toList(), equals([1, 2, 3, 10, 20]));
            });
          });

          test('union1d with overlapping and disjoint arrays', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
              final b = NDArray.fromList([2, 3, 4, 5], [4], DType.int32);
              expect(union1d(a, b).toList(), equals([1, 2, 3, 4, 5]));

              final empty = NDArray.create([0], DType.int32);
              expect(union1d(empty, empty).shape, [0]);
            });
          });

          test('isin with invert: false and true across 1D and 2D arrays', () {
            NDArray.scope(() {
              // 2D element array
              final elem = NDArray.fromList(
                [1, 2, 3, 4, 5, 6],
                [2, 3],
                DType.int32,
              );
              final testSet = NDArray.fromList([2, 4, 6, 8], [4], DType.int32);

              final res = isin(elem, testSet);
              expect(res.shape, [2, 3]);
              expect(
                res.toList(),
                equals([false, true, false, true, false, true]),
              );

              // invert: true
              final resInv = isin(elem, testSet, invert: true);
              expect(resInv.shape, [2, 3]);
              expect(
                resInv.toList(),
                equals([true, false, true, false, true, false]),
              );

              // assumeUnique: true
              final resUnique = isin(elem, testSet, assumeUnique: true);
              expect(
                resUnique.toList(),
                equals([false, true, false, true, false, true]),
              );
            });
          });

          test('set operations out buffer reuse and error handling', () {
            NDArray.scope(() {
              final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
              final b = NDArray.fromList([2, 3, 4], [3], DType.int32);

              final outInter = NDArray<Int32>.zeros([2], DType.int32);
              expect(
                identical(intersect1d(a, b, out: outInter), outInter),
                isTrue,
              );
              expect(outInter.toList(), equals([2, 3]));

              final outDiff = NDArray<Int32>.zeros([1], DType.int32);
              expect(identical(setdiff1d(a, b, out: outDiff), outDiff), isTrue);
              expect(outDiff.toList(), equals([1]));

              final outXor = NDArray<Int32>.zeros([2], DType.int32);
              expect(identical(setxor1d(a, b, out: outXor), outXor), isTrue);
              expect(outXor.toList(), equals([1, 4]));

              final outUnion = NDArray<Int32>.zeros([4], DType.int32);
              expect(identical(union1d(a, b, out: outUnion), outUnion), isTrue);
              expect(outUnion.toList(), equals([1, 2, 3, 4]));

              final outIsin = NDArray<Boolean>.create([3], DType.boolean);
              expect(identical(isin(a, b, out: outIsin), outIsin), isTrue);
              expect(outIsin.toList(), equals([false, true, true]));

              // Disposed error
              final disp = NDArray.fromList([1], [1], DType.int32)..dispose();
              expect(() => intersect1d(disp, a), throwsStateError);
              expect(() => setdiff1d(disp, a), throwsStateError);
              expect(() => setxor1d(disp, a), throwsStateError);
              expect(() => union1d(disp, a), throwsStateError);
              expect(() => isin(disp, a), throwsStateError);
            });
          });
        });
      });
    },
  );
}
