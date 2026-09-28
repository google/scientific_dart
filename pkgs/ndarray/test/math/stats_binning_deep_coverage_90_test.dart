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
  final allDTypes = <DType<AnySpec>>[
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

  final numericDTypes = <DType<AnySpec>>[
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

  final integerDTypes = <DType<AnySpec>>[
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint64,
    DType.uint32,
    DType.uint16,
    DType.uint8,
  ];

  group('Deep Coverage 90%+: Stats, Binning, NaN Metrics, Calculus & Padding', () {
    // =========================================================================
    // SECTION 1: HISTOGRAMS & MULTI-DIMENSIONAL BINNING
    // =========================================================================
    group('1. Histograms & Multi-dimensional Binning', () {
      group('bincount', () {
        test('empty input array handling', () {
          NDArray.scope(() {
            final empty = NDArray.fromList([], [0], DType.int64);
            final res1 = bincount(empty);
            expect(res1.shape, [0]);
            expect(res1.dtype, DType.int64);

            final res2 = bincount(empty, minlength: 5);
            expect(res2.shape, [5]);
            expect(res2.toList(), equals([0, 0, 0, 0, 0]));

            final out = NDArray.fromList([9, 9, 9], [3], DType.int64);
            final res3 = bincount(empty, minlength: 3, out: out);
            expect(identical(res3, out), isTrue);
            expect(out.toList(), equals([0, 0, 0]));

            final emptyW = NDArray<Float64>.fromList([], [0], DType.float64);
            final resW = bincount(empty, weights: emptyW, minlength: 4);
            expect(resW.shape, [4]);
            expect(resW.toList(), equals([0.0, 0.0, 0.0, 0.0]));
          });
        });

        test(
          'bincount across all integer types unweighted contiguous & strided',
          () {
            NDArray.scope(() {
              final data = [0, 2, 1, 2, 4, 1, 0, 2];
              final expected = [2, 2, 3, 0, 1];

              for (final dt in integerDTypes) {
                // Contiguous
                final arr = NDArray.fromList(data, [8], dt);
                final counts = bincount(arr);
                expect(counts.shape, [5]);
                expect(
                  counts.toList(),
                  equals(expected),
                  reason: 'Failed for dtype ',
                );

                // Out buffer
                final out = NDArray.zeros([5], dt);
                final resOut = bincount(arr, out: out);
                expect(identical(resOut, out), isTrue);
                expect(out.toList(), equals(expected));

                // Strided
                final stridedData = [
                  0,
                  99,
                  2,
                  99,
                  1,
                  99,
                  2,
                  99,
                  4,
                  99,
                  1,
                  99,
                  0,
                  99,
                  2,
                ];
                final stridedArr = NDArray.fromList(stridedData, [
                  15,
                ], dt).slice([Slice(start: 0, stop: 15, step: 2)]);
                final resStrided = bincount(stridedArr);
                expect(resStrided.toList(), equals(expected));
              }
            });
          },
        );

        test(
          'bincount with custom target output DTypes and out buffer recycling',
          () {
            NDArray.scope(() {
              final arr = NDArray.fromList([1, 3, 1, 2], [4], DType.int32);

              // Out buffer with int32
              final out32 = NDArray.zeros([5], DType.int32);
              final res32 = bincount(arr, out: out32);
              expect(identical(res32, out32), isTrue);
              expect(res32.toList(), equals([0, 2, 1, 1, 0]));

              // Out buffer with float64
              final outF64 = NDArray<Float64>.zeros([6], DType.float64);
              final resF64 = bincount(arr, out: outF64);
              expect(identical(resF64, outF64), isTrue);
              expect(resF64.toList(), equals([0.0, 2.0, 1.0, 1.0, 0.0, 0.0]));

              // Out buffer with uint8
              final outU8 = NDArray.zeros([4], DType.uint8);
              final resU8 = bincount(arr, out: outU8);
              expect(resU8.toList(), equals([0, 2, 1, 1]));
            });
          },
        );

        test(
          'weighted bincount across combinations of int types and float weights contiguous & strided',
          () {
            NDArray.scope(() {
              final xData = [0, 1, 1, 2, 0, 2];
              final wData = [0.5, 1.5, 2.0, 0.25, 1.0, 0.75];
              final expected = [1.5, 3.5, 1.0];

              for (final xDt in integerDTypes) {
                for (final wDt in [DType.float64, DType.float32]) {
                  final x = NDArray.fromList(xData, [6], xDt);
                  final w = NDArray.fromList(wData, [6], wDt);

                  // Contiguous
                  final res = bincount(x, weights: w);
                  expect(res.shape, [3]);
                  for (var i = 0; i < 3; i++) {
                    expect(
                      res.getCell([i]).toDouble(),
                      closeTo(expected[i], 1e-5),
                    );
                  }

                  // With out buffer
                  final out = NDArray.zeros([3], wDt);
                  final resOut = bincount(x, weights: w, out: out);
                  expect(identical(resOut, out), isTrue);

                  // Strided
                  final xFull = NDArray.fromList(
                    [0, 99, 1, 99, 1, 99, 2, 99, 0, 99, 2],
                    [11],
                    xDt,
                  );
                  final wFull = NDArray.fromList(
                    [0.5, 0, 1.5, 0, 2.0, 0, 0.25, 0, 1.0, 0, 0.75],
                    [11],
                    wDt,
                  );
                  final xStrided = xFull.slice([
                    Slice(start: 0, stop: 11, step: 2),
                  ]);
                  final wStrided = wFull.slice([
                    Slice(start: 0, stop: 11, step: 2),
                  ]);
                  final resStrided = bincount(xStrided, weights: wStrided);
                  for (var i = 0; i < 3; i++) {
                    expect(
                      resStrided.getCell([i]).toDouble(),
                      closeTo(expected[i], 1e-5),
                    );
                  }
                }
              }
            });
          },
        );

        test('bincount validation and error branches', () {
          NDArray.scope(() {
            final xDisposed = NDArray.fromList([1, 2], [2], DType.int32)
              ..dispose();
            expect(() => bincount(xDisposed), throwsStateError);

            final x2D = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            expect(() => bincount(x2D), throwsArgumentError);

            final xNegative = NDArray.fromList([1, -2, 3], [3], DType.int32);
            expect(() => bincount(xNegative), throwsArgumentError);

            final xValid = NDArray.fromList([1, 2, 3], [3], DType.int32);
            expect(() => bincount(xValid, minlength: -1), throwsArgumentError);

            final wDisposed = NDArray<Float64>.fromList(
              [1.0, 1.0, 1.0],
              [3],
              DType.float64,
            )..dispose();
            expect(
              () => bincount(xValid, weights: wDisposed),
              throwsStateError,
            );

            final wMismatched = NDArray<Float64>.fromList(
              [1.0, 1.0],
              [2],
              DType.float64,
            );
            expect(
              () => bincount(xValid, weights: wMismatched),
              throwsArgumentError,
            );

            final outDisposed = NDArray.zeros([4], DType.int64)..dispose();
            expect(() => bincount(xValid, out: outDisposed), throwsStateError);

            final outTooSmall = NDArray.zeros([2], DType.int64);
            expect(
              () => bincount(xValid, out: outTooSmall),
              throwsArgumentError,
            );

            final outWrongRank = NDArray.zeros([2, 2], DType.int64);
            expect(
              () => bincount(xValid, out: outWrongRank),
              throwsArgumentError,
            );
          });
        });
      });

      group('digitize', () {
        test(
          'monotonically increasing and decreasing bins with right = false and true',
          () {
            NDArray.scope(() {
              final x = NDArray<Float64>.fromList(
                [0.0, 1.0, 2.5, 4.0, 10.0, -1.0, 15.0],
                [7],
                DType.float64,
              );
              final bins = NDArray<Float64>.fromList(
                [0.0, 1.0, 2.5, 4.0, 10.0],
                [5],
                DType.float64,
              );

              final resLeft = digitize(x, bins, right: false);
              expect(resLeft.toList(), equals([1, 2, 3, 4, 5, 0, 5]));

              final resRight = digitize(x, bins, right: true);
              expect(resRight.toList(), equals([0, 1, 2, 3, 4, 0, 5]));

              final binsDesc = NDArray<Float64>.fromList(
                [10.0, 4.0, 2.5, 1.0, 0.0],
                [5],
                DType.float64,
              );
              final resDescLeft = digitize(x, binsDesc, right: false);
              expect(resDescLeft.toList(), equals([4, 3, 2, 1, 0, 5, 0]));

              final resDescRight = digitize(x, binsDesc, right: true);
              expect(resDescRight.toList(), equals([5, 4, 3, 2, 1, 5, 0]));
            });
          },
        );

        test('digitize 2D input array and custom DType conversion', () {
          NDArray.scope(() {
            final x2D = NDArray.fromList([0, 2, 5, 12], [2, 2], DType.int32);
            final bins = NDArray<Float64>.fromList(
              [1.0, 3.0, 10.0],
              [3],
              DType.float64,
            );

            final out = NDArray.zeros([2, 2], DType.int32);
            final res = digitize(x2D, bins, out: out);
            expect(identical(res, out), isTrue);
            expect(res.shape, equals([2, 2]));
            expect(res.toList(), equals([0, 1, 2, 3]));

            final binsInt = NDArray.fromList([1, 3, 10], [3], DType.int64);
            final resInt = digitize(x2D, binsInt);
            expect(resInt.toList(), equals([0, 1, 2, 3]));
          });
        });

        test('digitize error and validation branches', () {
          NDArray.scope(() {
            final x = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
            final bins = NDArray<Float64>.fromList(
              [1.0, 3.0],
              [2],
              DType.float64,
            );

            final xDisposed = NDArray<Float64>.fromList(
              [1.0],
              [1],
              DType.float64,
            )..dispose();
            expect(() => digitize(xDisposed, bins), throwsStateError);

            final binsDisposed = NDArray<Float64>.fromList(
              [1.0],
              [1],
              DType.float64,
            )..dispose();
            expect(() => digitize(x, binsDisposed), throwsStateError);

            final bins2D = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            expect(() => digitize(x, bins2D), throwsArgumentError);

            final emptyBins = NDArray<Float64>.fromList([], [0], DType.float64);
            expect(() => digitize(x, emptyBins), throwsArgumentError);

            final nonMonotonicBins = NDArray<Float64>.fromList(
              [1.0, 5.0, 2.0],
              [3],
              DType.float64,
            );
            expect(() => digitize(x, nonMonotonicBins), throwsArgumentError);

            final complexX = NDArray.fromList(
              [Complex(1, 0)],
              [1],
              DType.complex128,
            );
            expect(
              () => digitize(complexX as dynamic, bins),
              throwsA(anything),
            );

            final outDisposed = NDArray.zeros([2], DType.int32)..dispose();
            expect(() => digitize(x, bins, out: outDisposed), throwsStateError);

            final outBadShape = NDArray.zeros([3], DType.int32);
            expect(
              () => digitize(x, bins, out: outBadShape),
              throwsArgumentError,
            );
          });
        });
      });

      group('histogram', () {
        test('uniform bins with automatic range and explicit range', () {
          NDArray.scope(() {
            final data = NDArray<Float64>.fromList(
              [1.0, 2.0, 1.0, 4.0, 5.0, 2.0],
              [6],
              DType.float64,
            );

            final resAuto = histogram(data, bins: 4);
            expect(resAuto.hist.shape, [4]);
            expect(resAuto.binEdges.shape, [5]);
            expect(
              resAuto.binEdges.toList(),
              equals([1.0, 2.0, 3.0, 4.0, 5.0]),
            );
            expect(resAuto.hist.toList(), equals([2, 2, 0, 2]));

            final resRange = histogram(data, bins: 2, range: (0.0, 6.0));
            expect(resRange.binEdges.toList(), equals([0.0, 3.0, 6.0]));
            expect(resRange.hist.toList(), equals([4, 2]));
          });
        });

        test('density normalization with uniform and non-uniform bins', () {
          NDArray.scope(() {
            final data = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );

            final resDensity = histogram(data, bins: 2, density: true);
            expect(resDensity.hist.shape, [2]);
            expect(
              resDensity.hist.getCell([0]).toDouble(),
              closeTo(1.0 / 3.0, 1e-7),
            );
            expect(
              resDensity.hist.getCell([1]).toDouble(),
              closeTo(1.0 / 3.0, 1e-7),
            );

            final edges = NDArray<Float64>.fromList(
              [0.0, 2.0, 10.0],
              [3],
              DType.float64,
            );
            final resNonUniformDensity = histogram(
              data,
              bins: edges,
              density: true,
            );
            expect(
              resNonUniformDensity.hist.getCell([0]).toDouble(),
              closeTo(0.125, 1e-7),
            );
            expect(
              resNonUniformDensity.hist.getCell([1]).toDouble(),
              closeTo(0.09375, 1e-7),
            );
          });
        });

        test(
          'weighted histogram with contiguous and strided inputs, and non-Float64 edges',
          () {
            NDArray.scope(() {
              final dataFull = NDArray<Float64>.fromList(
                [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0],
                [7],
                DType.float64,
              );
              final weightsFull = NDArray<Float64>.fromList(
                [0.5, 0.0, 1.5, 0.0, 2.0, 0.0, 1.0],
                [7],
                DType.float64,
              );
              final data = dataFull.slice([Slice(start: 0, stop: 7, step: 2)]);
              final weights = weightsFull.slice([
                Slice(start: 0, stop: 7, step: 2),
              ]);

              final res = histogram(data, bins: 2, weights: weights);
              expect(res.hist.shape, [2]);
              expect(res.hist.toList(), equals([2.0, 3.0]));

              // Int edges
              final intEdges = NDArray.fromList([0, 2, 5], [3], DType.int32);
              final resInt = histogram(data, bins: intEdges, weights: weights);
              expect(resInt.hist.shape, [2]);
              expect(resInt.hist.toList(), equals([0.5, 4.5]));

              // Strided non-uniform bin edges
              final fullEdges = NDArray<Float64>.fromList(
                [0.0, 99.0, 2.0, 99.0, 5.0],
                [5],
                DType.float64,
              );
              final stridedEdges = fullEdges.slice([
                Slice(start: 0, stop: 5, step: 2),
              ]);
              final resStridedEdges = histogram(
                data,
                bins: stridedEdges,
                weights: weights,
              );
              expect(resStridedEdges.hist.toList(), equals([0.5, 4.5]));
            });
          },
        );

        test('multi-dimensional array histogram (ravels automatically)', () {
          NDArray.scope(() {
            final data2D = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final w2D = NDArray<Float64>.fromList(
              [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
              [2, 3],
              DType.float64,
            );
            final res = histogram(data2D, bins: 3, weights: w2D);
            expect(res.hist.shape, [3]);
            expect(res.hist.toList(), equals([2.0, 3.0, 4.0]));
          });
        });

        test(
          'histogram edge cases: constant input array, empty array, single range point',
          () {
            NDArray.scope(() {
              final constArr = NDArray<Float64>.fromList(
                [3.0, 3.0, 3.0],
                [3],
                DType.float64,
              );
              final resConst = histogram(constArr, bins: 2);
              expect(resConst.binEdges.getCell([0]), closeTo(2.5, 1e-9));
              expect(resConst.binEdges.getCell([2]), closeTo(3.5, 1e-9));
              expect(resConst.hist.toList(), equals([0, 3]));

              final emptyArr = NDArray<Float64>.fromList([], [
                0,
              ], DType.float64);
              final resEmpty = histogram(emptyArr, bins: 4);
              expect(resEmpty.hist.shape, [4]);
              expect(resEmpty.hist.toList(), equals([0, 0, 0, 0]));

              final resRangeEqual = histogram(
                constArr,
                bins: 2,
                range: (5.0, 5.0),
              );
              expect(resRangeEqual.binEdges.getCell([0]), closeTo(4.5, 1e-9));
              expect(resRangeEqual.binEdges.getCell([2]), closeTo(5.5, 1e-9));
            });
          },
        );

        test('histogram validation and error branches', () {
          NDArray.scope(() {
            final data = NDArray<Float64>.fromList(
              [1.0, 2.0],
              [2],
              DType.float64,
            );
            final disp = NDArray<Float64>.fromList(
              [1.0, 2.0],
              [2],
              DType.float64,
            )..dispose();
            expect(() => histogram(disp), throwsStateError);
            expect(() => histogram(data, weights: disp), throwsStateError);

            final complexData = NDArray.fromList(
              [Complex(1, 0)],
              [1],
              DType.complex128,
            );
            expect(() => histogram(complexData as dynamic), throwsA(anything));

            expect(() => histogram(data, bins: 0), throwsArgumentError);
            expect(() => histogram(data, bins: -5), throwsArgumentError);
            expect(
              () => histogram(data, range: (10.0, 5.0)),
              throwsArgumentError,
            );

            final badEdges = NDArray<Float64>.fromList(
              [1.0],
              [1],
              DType.float64,
            );
            expect(() => histogram(data, bins: badEdges), throwsArgumentError);

            final nonMonoEdges = NDArray<Float64>.fromList(
              [1.0, 3.0, 2.0],
              [3],
              DType.float64,
            );
            expect(
              () => histogram(data, bins: nonMonoEdges),
              throwsArgumentError,
            );

            final badWeights = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            expect(
              () => histogram(data, weights: badWeights),
              throwsArgumentError,
            );

            expect(
              () => histogram(data, bins: 'invalid' as dynamic),
              throwsArgumentError,
            );
          });
        });
      });
    });

    // =========================================================================
    // SECTION 2: COMPLETE SUITE OF REDUCTIONS & STATISTICS
    // =========================================================================
    group('2. Complete Suite of Reductions & Statistics', () {
      test('sum across all 15 DTypes with contiguous and strided slices', () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            final NDArray<AnySpec> arr = dt.isComplex
                ? NDArray.fromList(
                    [Complex(1, 2), Complex(3, 4), Complex(5, 6)],
                    [3],
                    dt,
                  )
                : dt == DType.boolean
                ? NDArray<Boolean>.fromList(
                    [true, false, true, true],
                    [4],
                    DType.boolean,
                  )
                : NDArray.fromList([1, 2, 3, 4], [4], dt);

            final res = sum(arr);
            expect(res.shape, <int>[]);
            if (dt.isComplex) {
              final c = res.scalar as Complex;
              expect(c.real, closeTo(9.0, 1e-5));
              expect(c.imag, closeTo(12.0, 1e-5));
            } else if (dt == DType.boolean) {
              expect(res.scalar, isTrue);
              expect(sumAs(arr, DType.int64).scalar, 3);
            } else {
              expect((res.scalar as num).toDouble(), closeTo(10.0, 1e-5));
            }

            if (arr.size >= 4) {
              final strided = arr.slice([
                Slice(start: 0, stop: arr.size, step: 2),
              ]);
              final resStrided = sum(strided);
              expect(resStrided.shape, <int>[]);
            }
          }
        });
      });

      test('prod across all 15 DTypes with empty arrays and axes', () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            final NDArray<AnySpec> empty = dt.isComplex
                ? NDArray.fromList([], [0], dt)
                : dt == DType.boolean
                ? NDArray<Boolean>.fromList([], [0], DType.boolean)
                : NDArray.fromList([], [0], dt);

            final resEmpty = prod(empty);
            expect(resEmpty.shape, <int>[]);
            if (dt.isComplex) {
              expect(resEmpty.scalar, equals(Complex(1.0, 0.0)));
            } else if (dt == DType.boolean) {
              expect(resEmpty.scalar, isTrue);
              expect(prodAs(empty, DType.int64).scalar, 1);
            } else {
              expect((resEmpty.scalar as num).toDouble(), closeTo(1.0, 1e-5));
            }

            final NDArray<AnySpec> arr = dt.isComplex
                ? NDArray.fromList([Complex(1, 1), Complex(2, 0)], [2], dt)
                : dt == DType.boolean
                ? NDArray<Boolean>.fromList(
                    [true, true, true],
                    [3],
                    DType.boolean,
                  )
                : NDArray.fromList([2, 3, 4], [3], dt);
            final res = prod(arr);
            if (dt.isComplex) {
              final c = res.scalar as Complex;
              expect(c.real, closeTo(2.0, 1e-5));
              expect(c.imag, closeTo(2.0, 1e-5));
            } else if (dt == DType.boolean) {
              expect(res.scalar, isTrue);
              expect(prodAs(arr, DType.int64).scalar, 1);
            } else {
              expect((res.scalar as num).toDouble(), closeTo(24.0, 1e-5));
            }
          }
        });
      });

      test(
        'mean, variance, std across all numeric & complex DTypes and axes',
        () {
          NDArray.scope(() {
            for (final dt in allDTypes) {
              if (dt == DType.boolean) continue;

              final NDArray<AnySpec> arr2D = dt.isComplex
                  ? NDArray.fromList(
                      [
                        Complex(1, 1),
                        Complex(2, 2),
                        Complex(3, 3),
                        Complex(4, 4),
                      ],
                      [2, 2],
                      dt,
                    )
                  : NDArray.fromList([1, 2, 3, 4], [2, 2], dt);

              // Global mean
              final mGlobal = mean(arr2D);
              expect(mGlobal.shape, <int>[]);

              // Axis 0 mean
              final m0 = mean(arr2D, axis: 0);
              expect(m0.shape, [2]);

              // Axis 1 mean with keepdims
              final m1Keep = mean(arr2D, axis: 1, keepdims: true);
              expect(m1Keep.shape, [2, 1]);

              if (!dt.isComplex && dt != DType.boolean) {
                final numArr = NDArray.fromList([1, 2, 3, 4], [2, 2], dt);
                // Variance
                final v = variance(numArr, axis: 0);
                expect(v.shape, [2]);

                // Std
                final s = std(numArr, axis: 1, keepdims: true);
                expect(s.shape, [2, 1]);
              }
            }
          });
        },
      );

      test(
        'all and any logic reductions across multi-axis arrays and booleans',
        () {
          NDArray.scope(() {
            final a = NDArray<Boolean>.fromList(
              [true, false, true, true],
              [2, 2],
              DType.boolean,
            );

            expect(all(a).scalar, isFalse);
            expect(any(a).scalar, isTrue);

            final all0 = all(a, axis: 0);
            expect(all0.toList(), equals([true, false]));

            final any1 = any(a, axis: 1);
            expect(any1.toList(), equals([true, true]));

            final out = NDArray<Boolean>.zeros([2], DType.boolean);
            final resAnyOut = any(a, axis: 0, out: out);
            expect(identical(resAnyOut, out), isTrue);
            expect(out.toList(), equals([true, true]));
          });
        },
      );

      test('std, variance and var_ with ddof = 0, 1, 2 and size <= ddof', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
            [8],
            DType.float64,
          );

          final v0 = variance(a, ddof: 0);
          expect(v0.scalar, closeTo(4.0, 1e-9));
          final s0 = std(a, ddof: 0);
          expect(s0.scalar, closeTo(2.0, 1e-9));

          final v1 = var_(a, ddof: 1);
          expect(v1.scalar, closeTo(32.0 / 7.0, 1e-9));

          final small = NDArray<Float64>.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
          );
          final vNaN = variance(small, ddof: 2);
          expect(vNaN.scalar.isNaN, isTrue);

          final a2D = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final vAxis0 = variance(a2D, axis: 0, ddof: 0);
          expect(vAxis0.shape, [2]);
          expect(vAxis0.toList(), equals([1.0, 1.0]));
        });
      });

      test('min and max across all numeric DTypes and axis reductions', () {
        NDArray.scope(() {
          for (final dt in numericDTypes) {
            final a = NDArray.fromList([10, 2, 45, 1, 88, 12], [2, 3], dt);

            final minGlobal = min(a);
            expect(minGlobal.scalar.toInt(), equals(1));
            expect(minGlobal.dtype, dt);

            final maxGlobal = max(a);
            expect(maxGlobal.scalar.toInt(), equals(88));
            expect(maxGlobal.dtype, dt);

            final minAxis0 = min(a, axis: 0);
            expect(minAxis0.shape, [3]);
            expect(
              minAxis0.toList().map((e) => e.toInt()).toList(),
              equals([1, 2, 12]),
            );

            final maxAxis1 = max(a, axis: 1, keepdims: true);
            expect(maxAxis1.shape, [2, 1]);
            expect(
              maxAxis1.toList().map((e) => e.toInt()).toList(),
              equals([45, 88]),
            );
          }
        });
      });

      test('cumsum, cumprod, cummin, cummax across axes and DTypes', () {
        NDArray.scope(() {
          final a = NDArray.fromList([3, 1, 4, 1, 5, 9], [2, 3], DType.int32);

          final cs = cumsum(a);
          expect(cs.shape, [6]);
          expect(cs.toList(), equals([3, 4, 8, 9, 14, 23]));

          final cs0 = cumsum(a, axis: 0);
          expect(cs0.shape, [2, 3]);
          expect(cs0.toList(), equals([3, 1, 4, 4, 6, 13]));

          final cp = cumprod(NDArray.fromList([1, 2, 3, 4], [4], DType.int32));
          expect(cp.toList(), equals([1, 2, 6, 24]));

          final cmin = cummin(
            NDArray.fromList([5, 2, 8, 1, 9], [5], DType.int32),
          );
          expect(cmin.toList(), equals([5, 2, 2, 1, 1]));

          final cmax = cummax(
            NDArray.fromList([5, 2, 8, 1, 9], [5], DType.int32),
          );
          expect(cmax.toList(), equals([5, 5, 8, 8, 9]));
        });
      });

      test(
        'quantile and percentile across all 9 QuantileMethods and DTypes',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
              [10],
              DType.float64,
            );

            final methods = [
              QuantileMethod.invertedCdf,
              QuantileMethod.averagedInvertedCdf,
              QuantileMethod.closestObservation,
              QuantileMethod.interpolatedInvertedCdf,
              QuantileMethod.hazen,
              QuantileMethod.weibull,
              QuantileMethod.linear,
              QuantileMethod.medianUnbiased,
              QuantileMethod.normalUnbiased,
              QuantileMethod.lower,
              QuantileMethod.higher,
              QuantileMethod.midpoint,
              QuantileMethod.nearest,
            ];

            for (final m in methods) {
              final q50 = quantile(a, 0.5, method: m);
              expect(q50.scalar, closeTo(5.5, 0.6), reason: 'Method  failed');

              final p75 = percentile(a, 75.0, method: m);
              expect(p75.scalar, greaterThan(6.0));
            }

            final a2D = NDArray<Float64>.fromList(
              [1.0, 5.0, 2.0, 6.0, 3.0, 7.0],
              [3, 2],
              DType.float64,
            );
            final qAxis0 = quantile(a2D, 0.5, axis: 0);
            expect(qAxis0.shape, [2]);
            expect(qAxis0.toList(), equals([2.0, 6.0]));
          });
        },
      );

      test('median across all 15 DTypes with axis and keepdims', () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            final NDArray<AnySpec> arr = dt.isComplex
                ? NDArray.fromList(
                    [
                      Complex(1, 1),
                      Complex(2, 2),
                      Complex(5, 5),
                      Complex(3, 3),
                      Complex(4, 4),
                      Complex(6, 6),
                    ],
                    [2, 3],
                    dt,
                  )
                : dt == DType.boolean
                ? NDArray<Boolean>.fromList(
                    [true, false, true, false, false, true],
                    [2, 3],
                    DType.boolean,
                  )
                : NDArray.fromList([1, 2, 5, 3, 4, 6], [2, 3], dt);

            final medGlobal = median(arr);
            expect(medGlobal.shape, <int>[]);

            final med0 = median(arr, axis: 0);
            expect(med0.shape, [3]);

            final med1Keep = median(arr, axis: 1, keepdims: true);
            expect(med1Keep.shape, [2, 1]);
          }
        });
      });

      test('ptp (peak to peak) across all numeric DTypes and axes', () {
        NDArray.scope(() {
          for (final dt in numericDTypes) {
            final a = NDArray.fromList([1, 10, 3, 20], [2, 2], dt);

            final ptpGlobal = ptp(a);
            expect(ptpGlobal.scalar.toInt(), equals(19));

            final ptp0 = ptp(a, axis: 0);
            expect(ptp0.shape, [2]);
            expect(
              ptp0.toList().map((e) => e.toInt()).toList(),
              equals([2, 10]),
            );

            final out = NDArray.zeros([2], dt);
            final ptp1 = ptp(a, axis: 1, out: out);
            expect(identical(ptp1, out), isTrue);
            expect(
              out.toList().map((e) => e.toInt()).toList(),
              equals([9, 17]),
            );
          }
        });
      });

      test(
        'average with 1D and ND weights, axis, and returned sum of weights',
        () {
          NDArray.scope(() {
            final a1D = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final w1D = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final avgGlobal = average(a1D, weights: w1D);
            expect(avgGlobal.average.scalar, closeTo(30.0 / 10.0, 1e-9));

            final a2D = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final wAx0 = NDArray<Float64>.fromList(
              [1.0, 3.0],
              [2],
              DType.float64,
            );
            final avgAx0 = average(a2D, axis: 0, weights: wAx0, returned: true);
            expect(avgAx0.average.shape, [2]);
            expect(avgAx0.sumOfWeights?.shape, [2]);
            expect(avgAx0.sumOfWeights?.toList(), equals([4.0, 4.0]));

            final avgAx1 = average(a2D, axis: 1, weights: wAx0);
            expect(avgAx1.average.shape, [2]);
          });
        },
      );

      test(
        'cov and corrcoef with 1D, 2D arrays, fweights, aweights and zero variance',
        () {
          NDArray.scope(() {
            final x = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            final c1D = cov(x);
            expect(c1D.shape, <int>[]);
            expect(c1D.scalar, closeTo(1.0, 1e-9));

            final y = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            final c2D = cov(x, y: y);
            expect(c2D.shape, [2, 2]);
            expect(c2D.getCell([0, 0]), closeTo(1.0, 1e-9));
            expect(c2D.getCell([0, 1]), closeTo(1.0, 1e-9));
            expect(c2D.getCell([1, 1]), closeTo(1.0, 1e-9));

            final fw = NDArray.fromList([1, 2, 1], [3], DType.int32);
            final aw = NDArray<Float64>.fromList(
              [1.0, 1.0, 1.0],
              [3],
              DType.float64,
            );
            final cWeights = cov(x, fweights: fw, aweights: aw);
            expect(cWeights.scalar, isA<double>());

            final r = corrcoef(x, y: y);
            expect(r.shape, [2, 2]);
            expect(r.getCell([0, 0]), closeTo(1.0, 1e-9));
            expect(r.getCell([0, 1]), closeTo(1.0, 1e-9));

            final zeroVar = NDArray<Float64>.fromList(
              [5.0, 5.0, 5.0],
              [3],
              DType.float64,
            );
            final rZero = corrcoef(zeroVar, y: x);
            expect(rZero.getCell([0, 0]).isNaN, isTrue);

            // rowvar = false & bias = true
            final m2D = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final cColVar = cov(m2D, rowvar: false);
            expect(cColVar.shape, [3, 3]);

            final cBias = cov(m2D, bias: true);
            expect(cBias.shape, [2, 2]);

            final rColVar = corrcoef(m2D, rowvar: false);
            expect(rColVar.shape, [3, 3]);
          });
        },
      );
    });

    // =========================================================================
    // SECTION 3: NAN-IGNORING REDUCTIONS & METRICS
    // =========================================================================
    group('3. Complete Suite of NaN-ignoring Statistics & Reductions', () {
      test(
        'nanmin and nanmax across all non-complex DTypes and axis reductions',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [double.nan, 3.0, 1.0, 4.0, double.nan, 2.0],
              [2, 3],
              DType.float64,
            );

            final nmin = nanmin(a);
            expect(nmin.scalar, equals(1.0));
            final nmax = nanmax(a);
            expect(nmax.scalar, equals(4.0));

            final nmin0 = nanmin(a, axis: 0);
            expect(nmin0.shape, [3]);
            expect(nmin0.toList(), equals([4.0, 3.0, 1.0]));

            final nmax0 = nanmax(a, axis: 0);
            expect(nmax0.shape, [3]);
            expect(nmax0.toList(), equals([4.0, 3.0, 2.0]));

            final allNaN = NDArray<Float64>.fromList(
              [double.nan, double.nan],
              [2],
              DType.float64,
            );
            final resAllNaN = nanmin(allNaN);
            expect(resAllNaN.scalar.isNaN, isTrue);

            final out = NDArray<Float64>.zeros([3], DType.float64);
            final resOut = nanmin(a, axis: 0, out: out);
            expect(identical(resOut, out), isTrue);
            expect(out.toList(), equals([4.0, 3.0, 1.0]));

            for (final dt in numericDTypes) {
              final numArr = NDArray.fromList([1, 5, 2, 8, 3, 7], [2, 3], dt);
              final nm = nanmin(numArr, axis: 0);
              expect(nm.shape, [3]);
              final nx = nanmax(numArr, axis: 1, keepdims: true);
              expect(nx.shape, [2, 1]);
            }
          });
        },
      );

      test(
        'nanmean, nanvar, nanstd with complex numbers and all numeric DTypes',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, double.nan, 3.0, 5.0],
              [4],
              DType.float64,
            );

            final m = nanmean<Float64>(a);
            expect(m.scalar, closeTo(3.0, 1e-9));

            final v = nanvar(a);
            expect(v.scalar, closeTo(8.0 / 3.0, 1e-9));

            final s = nanstd(a);
            expect(s.scalar, closeTo(math.sqrt(8.0 / 3.0), 1e-9));

            final cpx = NDArray.fromList(
              [Complex(1.0, 2.0), Complex(double.nan, 4.0), Complex(3.0, 6.0)],
              [3],
              DType.complex128,
            );
            final cpxMean = nanmean<DTypeTag>(cpx);
            final c = cpxMean.scalar;
            expect(c.real, closeTo(2.0, 1e-9));
            expect(c.imag, closeTo(4.0, 1e-9));

            final allNaN = NDArray<Float64>.fromList(
              [double.nan, double.nan],
              [2],
              DType.float64,
            );
            final mAllNaN = nanmean<Float64>(allNaN);
            expect(mAllNaN.scalar.isNaN, isTrue);

            final vAllNaN = nanvar(allNaN);
            expect(vAllNaN.scalar.isNaN, isTrue);

            for (final dt in numericDTypes) {
              final arr = NDArray.fromList([1, 5, 2, 8, 3, 7], [2, 3], dt);
              final meanRes = nanmean(arr, axis: 0);
              expect(meanRes.shape, [3]);
              final varRes = nanvar(arr, axis: 1);
              expect(varRes.shape, [2]);
              final stdRes = nanstd(arr, axis: 0);
              expect(stdRes.shape, [3]);
            }
          });
        },
      );

      test('nansum across 2D slices, keepdims, and complex numbers', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, double.nan, double.nan, 4.0],
            [2, 2],
            DType.float64,
          );

          final sGlobal = nansum(a);
          expect(sGlobal.scalar, equals(5.0));

          final s0 = nansum(a, axis: 0, keepdims: true);
          expect(s0.shape, equals([1, 2]));
          expect(s0.toList(), equals([1.0, 4.0]));

          final s1 = nansum(a, axis: 1);
          expect(s1.shape, equals([2]));
          expect(s1.toList(), equals([1.0, 4.0]));

          final cpx = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(double.nan, 10.0),
              Complex(3.0, double.nan),
              Complex(4.0, 5.0),
            ],
            [4],
            DType.complex128,
          );
          final cpxSum = nansum(cpx);
          expect(cpxSum.scalar, equals(Complex(5.0, 7.0)));
        });
      });
    });

    // =========================================================================
    // SECTION 4: CALCULUS (TRAPZ, GRADIENT, GRADIENTARRAY, DIFF)
    // =========================================================================
    group('4. Calculus & Numerical Differentiation / Integration', () {
      group('trapz', () {
        test(
          'trapz across float32, float64, complex64, complex128 with StepSpacing and CoordinateSpacing',
          () {
            NDArray.scope(() {
              // Float64 StepSpacing
              final yF64 = NDArray<Float64>.fromList(
                [1.0, 3.0, 7.0],
                [3],
                DType.float64,
              );
              final resF64 = trapz(yF64, spacing: Spacing.step(2.0));
              expect(resF64.scalar, closeTo(14.0, 1e-9));

              // Float32 StepSpacing
              final yF32 = NDArray<Float32>.fromList(
                [1.0, 3.0, 7.0],
                [3],
                DType.float32,
              );
              final resF32 = trapz(yF32, spacing: Spacing.step(2.0));
              expect(resF32.dtype, DType.float32);
              expect(resF32.scalar, closeTo(14.0, 1e-5));

              // Float32 CoordinateSpacing
              final resF32Coords = trapz(
                yF32,
                spacing: Spacing.coordinates([0.0, 2.0, 4.0]),
              );
              expect(resF32Coords.dtype, DType.float32);
              expect(resF32Coords.scalar, closeTo(14.0, 1e-5));

              // Complex64 StepSpacing
              final yC64 = NDArray.fromList(
                [Complex(1, 1), Complex(2, 2), Complex(4, 4)],
                [3],
                DType.complex64,
              );
              final resC64 = trapz(yC64, spacing: Spacing.step(Complex(2, 1)));
              expect(resC64.dtype, DType.complex64);

              // Complex64 CoordinateSpacing real
              final resC64RealCoords = trapz(
                yC64,
                spacing: Spacing.coordinates([0.0, 1.0, 2.0]),
              );
              expect(resC64RealCoords.dtype, DType.complex64);

              // Complex64 CoordinateSpacing complex
              final resC64CpxCoords = trapz(
                yC64,
                spacing: Spacing.coordinates([
                  Complex(0, 0),
                  Complex(1, 1),
                  Complex(2, 2),
                ]),
              );
              expect(resC64CpxCoords.dtype, DType.complex64);

              // Complex128 real Spacing.step
              final yC128 = NDArray.fromList(
                [Complex(1, 1), Complex(2, 2), Complex(4, 4)],
                [3],
                DType.complex128,
              );
              final resC128RealStep = trapz(yC128, spacing: Spacing.step(2.0));
              expect(resC128RealStep.dtype, DType.complex128);

              // Complex128 real Spacing.coordinates
              final resC128RealCoords = trapz(
                yC128,
                spacing: Spacing.coordinates([0.0, 1.0, 2.0]),
              );
              expect(resC128RealCoords.dtype, DType.complex128);

              // Complex128 complex Spacing.step
              final resC128CpxStep = trapz(
                yC128,
                spacing: Spacing.step(Complex(2.0, 1.0)),
              );
              final c = resC128CpxStep.scalar;
              expect(c.real, closeTo(4.5, 1e-9));
              expect(c.imag, closeTo(13.5, 1e-9));

              // Out buffer and multi-axis
              final y2D = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2, 3],
                DType.float64,
              );
              final resAx0 = trapz(y2D, axis: 0);
              expect(resAx0.shape, [3]);
              expect(resAx0.toList(), equals([2.5, 3.5, 4.5]));

              final out = NDArray<Float64>.zeros([2], DType.float64);
              final resOut = trapz(y2D, axis: 1, out: out);
              expect(identical(resOut, out), isTrue);
              expect(out.toList(), equals([4.0, 10.0]));
            });
          },
        );

        test('trapz validation errors', () {
          NDArray.scope(() {
            final yInt = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final intRes = trapz(yInt);
            expect(intRes.dtype, DType.float64);
            expect((intRes.scalar as num).toDouble(), closeTo(4.0, 1e-12));

            final yF64 = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            expect(
              () => trapz(yF64, spacing: Spacing.step(Complex(1, 0))),
              throwsArgumentError,
            );
            expect(() => trapz(yF64, axis: 5), throwsArgumentError);
            expect(
              () => trapz(yF64, spacing: Spacing.coordinates([0.0, 1.0])),
              throwsArgumentError,
            );

            final yDisp = NDArray<Float64>.fromList(
              [1.0, 2.0],
              [2],
              DType.float64,
            )..dispose();
            expect(() => trapz(yDisp), throwsStateError);

            final outDisp = NDArray<Float64>.zeros([2], DType.float64)
              ..dispose();
            expect(() => trapz(yF64, axis: 0, out: outDisp), throwsStateError);

            final outBadShape = NDArray<Float64>.zeros([5], DType.float64);
            expect(
              () => trapz(y2DWrong(yF64), axis: 0, out: outBadShape),
              throwsArgumentError,
            );
          });
        });
      });

      group('gradient & gradientArray', () {
        test('gradient with edgeOrder = 1 vs edgeOrder = 2', () {
          NDArray.scope(() {
            final f = NDArray<Float64>.fromList(
              [0.0, 1.0, 4.0, 9.0, 16.0],
              [5],
              DType.float64,
            );

            final g1 = gradient(f, edgeOrder: 1);
            expect(g1.toList(), equals([1.0, 2.0, 4.0, 6.0, 7.0]));

            final g2 = gradient(f, edgeOrder: 2);
            expect(g2.toList(), equals([0.0, 2.0, 4.0, 6.0, 8.0]));
          });
        });

        test(
          'gradient across float32, float64, complex64, complex128 with StepSpacing and CoordinateSpacing',
          () {
            NDArray.scope(() {
              // Float32 StepSpacing
              final fF32 = NDArray<Float32>.fromList(
                [0.0, 1.0, 4.0, 9.0],
                [4],
                DType.float32,
              );
              final gF32 = gradient(fF32, edgeOrder: 2);
              expect(gF32.dtype, DType.float32);

              // Float32 CoordinateSpacing
              final gF32Coords = gradient(
                fF32,
                spacing: Spacing.coordinates([0.0, 1.0, 2.0, 3.0]),
              );
              expect(gF32Coords.dtype, DType.float32);

              // Complex64 StepSpacing complex dx
              final fC64 = NDArray.fromList(
                [Complex(0, 0), Complex(1, 1), Complex(4, 4)],
                [3],
                DType.complex64,
              );
              final gC64 = gradient(fC64, spacing: Spacing.step(Complex(1, 1)));
              expect(gC64.dtype, DType.complex64);

              // Complex64 StepSpacing real dx
              final gC64RealStep = gradient(fC64, spacing: Spacing.step(1.0));
              expect(gC64RealStep.dtype, DType.complex64);

              // Complex64 CoordinateSpacing real
              final gC64RealCoords = gradient(
                fC64,
                spacing: Spacing.coordinates([0.0, 1.0, 2.0]),
              );
              expect(gC64RealCoords.dtype, DType.complex64);

              // Complex64 CoordinateSpacing complex
              final gC64CpxCoords = gradient(
                fC64,
                spacing: Spacing.coordinates([
                  Complex(0, 0),
                  Complex(1, 1),
                  Complex(2, 2),
                ]),
              );
              expect(gC64CpxCoords.dtype, DType.complex64);

              // Complex128 StepSpacing complex dx
              final fC128 = NDArray.fromList(
                [Complex(0, 0), Complex(1, 1), Complex(4, 4)],
                [3],
                DType.complex128,
              );
              final gC128 = gradient(
                fC128,
                spacing: Spacing.step(Complex(1, 1)),
              );
              expect(gC128.dtype, DType.complex128);

              // Complex128 StepSpacing real dx
              final gC128RealStep = gradient(fC128, spacing: Spacing.step(1.0));
              expect(gC128RealStep.dtype, DType.complex128);

              // Complex128 CoordinateSpacing real
              final gC128RealCoords = gradient(
                fC128,
                spacing: Spacing.coordinates([0.0, 1.0, 2.0]),
              );
              expect(gC128RealCoords.dtype, DType.complex128);

              // Complex128 CoordinateSpacing complex
              final gC128CpxCoords = gradient(
                fC128,
                spacing: Spacing.coordinates([
                  Complex(0, 0),
                  Complex(1, 1),
                  Complex(2, 2),
                ]),
              );
              expect(gC128CpxCoords.dtype, DType.complex128);

              // Negative axis
              final f2D = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0, 4.0],
                [2, 2],
                DType.float64,
              );
              final gNegAx = gradient(f2D, axis: -1);
              expect(gNegAx.shape, [2, 2]);

              // Out buffer
              final out = NDArray<Float64>.zeros([2, 2], DType.float64);
              final gOut = gradient(f2D, out: out);
              expect(identical(gOut, out), isTrue);
            });
          },
        );

        test('gradientArray full parameter combinations and out buffers', () {
          NDArray.scope(() {
            final f2D = NDArray<Float64>.fromList(
              [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
              [2, 3],
              DType.float64,
            );

            final grads = gradientArray(f2D);
            expect(grads.length, 2);
            expect(grads[0].shape, [2, 3]);
            expect(grads[1].shape, [2, 3]);

            // With out buffers
            final out0 = NDArray<Float64>.zeros([2, 3], DType.float64);
            final out1 = NDArray<Float64>.zeros([2, 3], DType.float64);
            final res = gradientArray(f2D, out: [out0, out1]);
            expect(identical(res[0], out0), isTrue);
            expect(identical(res[1], out1), isTrue);

            // Explicit axis with negative index
            final resAx = gradientArray(f2D, axis: [-1]);
            expect(resAx.length, 1);

            // Validation
            expect(
              () => gradientArray(
                f2D,
                spacing: Spacing.step(1.0),
                spacings: [Spacing.step(1.0)],
              ),
              throwsArgumentError,
            );
            expect(() => gradientArray(f2D, axis: [0, 0]), throwsArgumentError);
            expect(() => gradientArray(f2D, axis: [5]), throwsArgumentError);
          });
        });

        test(
          'diff across difference orders n = 0, 1, 2, 3, 4 and all DTypes',
          () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1, 2, 4, 7, 11, 16],
                [6],
                DType.int32,
              );

              final d0 = diff(a, n: 0);
              expect(d0.toList(), equals([1, 2, 4, 7, 11, 16]));

              final d1 = diff(a, n: 1);
              expect(d1.toList(), equals([1, 2, 3, 4, 5]));

              final d2 = diff(a, n: 2);
              expect(d2.toList(), equals([1, 1, 1, 1]));

              final d3 = diff(a, n: 3);
              expect(d3.toList(), equals([0, 0, 0]));

              final d4 = diff(a, n: 4);
              expect(d4.toList(), equals([0, 0]));

              final dEmpty = diff(a, n: 6);
              expect(dEmpty.shape, [0]);

              final a2D = NDArray<Float64>.fromList(
                [1.0, 2.0, 4.0, 2.0, 5.0, 8.0],
                [2, 3],
                DType.float64,
              );

              final dAx0 = diff(a2D, axis: 0);
              expect(dAx0.shape, [1, 3]);
              expect(dAx0.toList(), equals([1.0, 3.0, 4.0]));

              final dAx1 = diff(a2D, axis: 1);
              expect(dAx1.shape, [2, 2]);
              expect(dAx1.toList(), equals([1.0, 2.0, 3.0, 3.0]));
            });
          },
        );
      });
    });

    // =========================================================================
    // SECTION 5: PADDING (ALL MODES & DTYPES)
    // =========================================================================
    group('5. Padding Deep Coverage Across All 15 DTypes & Modes', () {
      test(
        'PadWidth, PadValues, and StatLength constructors and normalization',
        () {
          final pwAll = PadWidth.all(2, 3);
          expect(pwAll.normalize(2), equals([(2, 3), (2, 3)]));

          final pwAxes = PadWidth.axes([(1, 2), (3, 4)]);
          expect(pwAxes.normalize(2), equals([(1, 2), (3, 4)]));
          expect(() => pwAxes.normalize(3), throwsArgumentError);

          final pv = PadValues<DTypeTag>.all(1.0, 2.0);
          expect(pv.normalize(2, 0.0), equals([(1.0, 2.0), (1.0, 2.0)]));

          final sl = StatLength.all(3, 4);
          expect(sl.normalize([5, 5]), equals([(3, 4), (3, 4)]));
          expect(sl.normalize([2, 3]), equals([(2, 2), (3, 3)]));
        },
      );

      test('Fast native padding across all 15 DTypes (1D and 2D)', () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            _runPadTestsForDType(dt);
          }
        });
      });

      test('Axis-by-axis fallback padding across all 15 DTypes and modes', () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            _runFallbackPadTestsForDType(dt);
          }
        });
      });

      test(
        '3D tensor padding with asymmetrical pad widths and out buffer reuse',
        () {
          NDArray.scope(() {
            final a3D = NDArray.fromList(List<int>.generate(24, (i) => i + 1), [
              2,
              3,
              4,
            ], DType.int32);

            final pw3D = PadWidth.axes([(1, 0), (2, 1), (0, 3)]);
            final out = NDArray.zeros([3, 6, 7], DType.int32);

            final res = pad(a3D, pw3D, mode: PadMode.edge, out: out);
            expect(identical(res, out), isTrue);
            expect(out.shape, equals([3, 6, 7]));
            expect(out.getCell([1, 2, 0]), equals(a3D.getCell([0, 0, 0])));
          });
        },
      );

      test('pad zero padding and error branches', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);

          final pZero = pad(a, PadWidth.all(0, 0));
          expect(pZero.toList(), equals([1.0, 2.0]));

          final aDisposed = NDArray<Float64>.fromList([1.0], [1], DType.float64)
            ..dispose();
          expect(() => pad(aDisposed, PadWidth.all(1)), throwsStateError);

          final outDisposed = NDArray<Float64>.zeros([4], DType.float64)
            ..dispose();
          expect(
            () => pad(a, PadWidth.all(1), out: outDisposed),
            throwsStateError,
          );

          final outWrongShape = NDArray<Float64>.zeros([5], DType.float64);
          expect(
            () => pad(a, PadWidth.all(1), out: outWrongShape),
            throwsArgumentError,
          );
        });
      });
    });

    // =========================================================================
    // SECTION 6: STATS DEEP COVERAGE - ERROR PATHS, OUT BUFFERS & EDGE CASES
    // =========================================================================
    group('6. Stats Error Paths & Out Buffers Deep Coverage', () {
      test('all and any error paths and out buffer validation', () {
        NDArray.scope(() {
          final a = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );
          final aDisp = NDArray<Boolean>.fromList([true], [1], DType.boolean)
            ..dispose();
          expect(() => all(aDisp), throwsStateError);
          expect(() => any(aDisp), throwsStateError);

          final outDisp = NDArray<Boolean>.zeros([1], DType.boolean)..dispose();
          expect(() => all(a, axis: 0, out: outDisp), throwsStateError);
          expect(() => any(a, axis: 0, out: outDisp), throwsStateError);

          final outBadShape = NDArray<Boolean>.zeros([5], DType.boolean);
          expect(() => all(a, axis: 0, out: outBadShape), throwsArgumentError);
          expect(() => any(a, axis: 0, out: outBadShape), throwsArgumentError);

          expect(() => all(a, axis: 5), throwsArgumentError);
          expect(() => any(a, axis: -5), throwsArgumentError);
        });
      });

      test(
        'sum, prod, mean, std, variance error paths and out buffer validation',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final aDisp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
              ..dispose();

            expect(() => sum(aDisp), throwsStateError);
            expect(() => prod(aDisp), throwsStateError);
            expect(() => mean(aDisp), throwsStateError);
            expect(() => std(aDisp), throwsStateError);
            expect(() => variance(aDisp), throwsStateError);

            final outDisp = NDArray<Float64>.zeros([2], DType.float64)
              ..dispose();
            expect(() => sum(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => prod(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => mean(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => std(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => variance(a, axis: 0, out: outDisp), throwsStateError);

            final outBad = NDArray<Float64>.zeros([5], DType.float64);
            expect(() => sum(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => prod(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => mean(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => std(a, axis: 0, out: outBad), throwsArgumentError);
            expect(
              () => variance(a, axis: 0, out: outBad),
              throwsArgumentError,
            );

            expect(() => sum(a, axis: 10), throwsArgumentError);
            expect(() => prod(a, axis: 10), throwsArgumentError);
            expect(() => mean(a, axis: 10), throwsArgumentError);
            expect(() => std(a, axis: 10), throwsArgumentError);
            expect(() => variance(a, axis: 10), throwsArgumentError);

            // Global out buffers
            final outGlobalF64 = NDArray<Float64>.zeros([], DType.float64);
            final resSum = sum(a, out: outGlobalF64);
            expect(identical(resSum, outGlobalF64), isTrue);

            final resProd = prod(a, out: outGlobalF64);
            expect(identical(resProd, outGlobalF64), isTrue);

            final resMean = mean(a, out: outGlobalF64);
            expect(identical(resMean, outGlobalF64), isTrue);

            final resVar = variance(a, out: outGlobalF64);
            expect(identical(resVar, outGlobalF64), isTrue);

            final resStd = std(a, out: outGlobalF64);
            expect(identical(resStd, outGlobalF64), isTrue);
          });
        },
      );

      test('min, max, ptp error paths and empty array checks', () {
        NDArray.scope(() {
          final empty = NDArray.fromList([], [0], DType.int32);
          expect(() => min(empty), throwsArgumentError);
          expect(() => max(empty), throwsArgumentError);
          expect(() => ptp(empty), throwsArgumentError);

          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final aDisp = NDArray.fromList([1], [1], DType.int32)..dispose();
          expect(() => min(aDisp), throwsStateError);
          expect(() => max(aDisp), throwsStateError);
          expect(() => ptp(aDisp), throwsStateError);

          final outDisp = NDArray.zeros([2], DType.int32)..dispose();
          expect(() => min(a, axis: 0, out: outDisp), throwsStateError);
          expect(() => max(a, axis: 0, out: outDisp), throwsStateError);
          expect(() => ptp(a, axis: 0, out: outDisp), throwsStateError);

          final outBad = NDArray.zeros([5], DType.int32);
          expect(() => min(a, axis: 0, out: outBad), throwsArgumentError);
          expect(() => max(a, axis: 0, out: outBad), throwsArgumentError);
          expect(() => ptp(a, axis: 0, out: outBad), throwsArgumentError);

          expect(() => min(a, axis: 5), throwsArgumentError);
          expect(() => max(a, axis: 5), throwsArgumentError);
          expect(() => ptp(a, axis: 5), throwsArgumentError);

          final outGlobalInt = NDArray.zeros([], DType.int32);
          final resMin = min(a, out: outGlobalInt);
          expect(identical(resMin, outGlobalInt), isTrue);
          final resMax = max(a, out: outGlobalInt);
          expect(identical(resMax, outGlobalInt), isTrue);
          final resPtp = ptp(a, out: outGlobalInt);
          expect(identical(resPtp, outGlobalInt), isTrue);
        });
      });

      test('cumsum, cumprod, cummin, cummax error paths and out buffers', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final aDisp = NDArray.fromList([1], [1], DType.int32)..dispose();

          expect(() => cumsum(aDisp), throwsStateError);
          expect(() => cumprod(aDisp), throwsStateError);
          expect(() => cummin(aDisp), throwsStateError);
          expect(() => cummax(aDisp), throwsStateError);

          final outDisp = NDArray.zeros([2, 2], DType.int32)..dispose();
          expect(() => cumsum(a, axis: 0, out: outDisp), throwsStateError);
          expect(() => cumprod(a, axis: 0, out: outDisp), throwsStateError);
          expect(() => cummin(a, axis: 0, out: outDisp), throwsStateError);
          expect(() => cummax(a, axis: 0, out: outDisp), throwsStateError);

          final outBad = NDArray.zeros([5], DType.int32);
          expect(() => cumsum(a, axis: 0, out: outBad), throwsArgumentError);
          expect(() => cumprod(a, axis: 0, out: outBad), throwsArgumentError);
          expect(() => cummin(a, axis: 0, out: outBad), throwsArgumentError);
          expect(() => cummax(a, axis: 0, out: outBad), throwsArgumentError);

          expect(() => cumsum(a, axis: 5), throwsArgumentError);
          expect(() => cumprod(a, axis: 5), throwsArgumentError);
          expect(() => cummin(a, axis: 5), throwsArgumentError);
          expect(() => cummax(a, axis: 5), throwsArgumentError);

          final out2D = NDArray.zeros([2, 2], DType.int32);
          final resCs = cumsum(a, axis: 0, out: out2D);
          expect(identical(resCs, out2D), isTrue);

          final resCp = cumprod(a, axis: 0, out: out2D);
          expect(identical(resCp, out2D), isTrue);

          final resCmin = cummin(a, axis: 0, out: out2D);
          expect(identical(resCmin, out2D), isTrue);

          final resCmax = cummax(a, axis: 0, out: out2D);
          expect(identical(resCmax, out2D), isTrue);
        });
      });

      test('quantile and percentile error branches and validation', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          expect(() => quantile(a, -0.1), throwsArgumentError);
          expect(() => quantile(a, 1.1), throwsArgumentError);
          expect(() => percentile(a, -1.0), throwsArgumentError);
          expect(() => percentile(a, 101.0), throwsArgumentError);

          final aDisp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
            ..dispose();
          expect(() => quantile(aDisp, 0.5), throwsStateError);
          expect(() => percentile(aDisp, 50.0), throwsStateError);

          final empty = NDArray<Float64>.fromList([], [0], DType.float64);
          expect(() => quantile(empty, 0.5), throwsArgumentError);

          final outDisp = NDArray<Float64>.zeros([], DType.float64)..dispose();
          expect(() => quantile(a, 0.5, out: outDisp), throwsStateError);

          final outBad = NDArray<Float64>.zeros([5], DType.float64);
          expect(() => quantile(a, 0.5, out: outBad), throwsArgumentError);
        });
      });

      test('median error paths and out buffers', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final aDisp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
            ..dispose();
          expect(() => median(aDisp), throwsStateError);

          final empty = NDArray<Float64>.fromList([], [0], DType.float64);
          expect(() => median(empty), throwsArgumentError);

          final outDisp = NDArray<Float64>.zeros([2], DType.float64)..dispose();
          expect(() => median(a, axis: 0, out: outDisp), throwsStateError);

          final outBad = NDArray<Float64>.zeros([5], DType.float64);
          expect(() => median(a, axis: 0, out: outBad), throwsArgumentError);

          expect(() => median(a, axis: 5), throwsArgumentError);

          final out = NDArray<Float64>.zeros([2], DType.float64);
          final res = median(a, axis: 0, out: out);
          expect(identical(res, out), isTrue);
        });
      });

      test('cov and corrcoef scalar and error paths', () {
        NDArray.scope(() {
          final x = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final xDisp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
            ..dispose();
          expect(() => cov(xDisp), throwsStateError);
          expect(() => corrcoef(xDisp), throwsStateError);

          final yDisp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
            ..dispose();
          expect(() => cov(x, y: yDisp), throwsStateError);
          expect(() => corrcoef(x, y: yDisp), throwsStateError);

          final outDisp = NDArray<Float64>.zeros([], DType.float64)..dispose();
          expect(() => cov(x, out: outDisp), throwsStateError);
          expect(() => corrcoef(x, out: outDisp), throwsStateError);

          final outScalar = NDArray<Float64>.zeros([], DType.float64);
          final resC = cov(x, out: outScalar);
          expect(identical(resC, outScalar), isTrue);

          final resR = corrcoef(x, out: outScalar);
          expect(identical(resR, outScalar), isTrue);
        });
      });

      test(
        'nanmean, nanvar, nanstd, nanmin, nanmax, nansum error paths and out buffers',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, double.nan, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final aDisp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
              ..dispose();

            expect(() => nanmin(aDisp), throwsStateError);
            expect(() => nanmax(aDisp), throwsStateError);
            expect(() => nanmean(aDisp), throwsStateError);
            expect(() => nanvar(aDisp), throwsStateError);
            expect(() => nanstd(aDisp), throwsStateError);
            expect(() => nansum(aDisp), throwsStateError);

            final outDisp = NDArray<Float64>.zeros([2], DType.float64)
              ..dispose();
            expect(() => nanmin(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => nanmax(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => nanmean(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => nanvar(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => nanstd(a, axis: 0, out: outDisp), throwsStateError);
            expect(() => nansum(a, axis: 0, out: outDisp), throwsStateError);

            final outBad = NDArray<Float64>.zeros([5], DType.float64);
            expect(() => nanmin(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => nanmax(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => nanmean(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => nanvar(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => nanstd(a, axis: 0, out: outBad), throwsArgumentError);
            expect(() => nansum(a, axis: 0, out: outBad), throwsArgumentError);

            expect(() => nanmin(a, axis: 5), throwsArgumentError);
            expect(() => nanmax(a, axis: 5), throwsArgumentError);
            expect(() => nanmean(a, axis: 5), throwsArgumentError);
            expect(() => nanvar(a, axis: 5), throwsArgumentError);
            expect(() => nanstd(a, axis: 5), throwsArgumentError);
            expect(() => nansum(a, axis: 5), throwsArgumentError);

            // Integer arrays in nanmean (exercises promoteToDouble)
            final intArr = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final nmInt = nanmean(intArr, axis: 0);
            expect(nmInt.shape, [2]);

            // Integer arrays in nansum
            final nsInt = nansum(intArr, axis: 0);
            expect(nsInt.shape, [2]);
          });
        },
      );
    });

    // =========================================================================
    // SECTION 7: PADDING & CALCULUS ADVANCED CORNER CASES
    // =========================================================================
    group('7. Padding & Calculus Advanced Corner Cases', () {
      test('PadValues and StatLength constructors and edge validations', () {
        final pvAxes = PadValues<DTypeTag>.axes([(1.0, 2.0), (3.0, 4.0)]);
        expect(pvAxes.normalize(2, 0.0), equals([(1.0, 2.0), (3.0, 4.0)]));
        expect(() => pvAxes.normalize(3, 0.0), throwsArgumentError);

        final slAxes = StatLength.axes([(2, 2), (1, 3)]);
        expect(slAxes.normalize([5, 5]), equals([(2, 2), (1, 3)]));
        expect(slAxes.normalize([0, 5]), equals([(0, 0), (1, 3)]));
        expect(() => slAxes.normalize([5]), throwsArgumentError);

        expect(() => StatLength.all(0, 1), throwsArgumentError);
        expect(() => StatLength.all(1, -1), throwsArgumentError);
        expect(() => StatLength.axes([(0, 1)]), throwsArgumentError);
        expect(() => StatLength.axes([(1, -1)]), throwsArgumentError);
      });

      test(
        'pad 0-dimensional arrays, mismatched out rank/dtype, and zero pad with out',
        () {
          NDArray.scope(() {
            final scalarArr = NDArray<Float64>.scalar(
              5.0,
              dtype: DType.float64,
            );
            expect(() => pad(scalarArr, PadWidth.all(1)), throwsArgumentError);

            final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
            final outBadDType = NDArray.zeros([4], DType.int32);
            expect(
              () => pad(a, PadWidth.all(1), out: outBadDType),
              throwsArgumentError,
            );

            final outBadRank = NDArray<Float64>.zeros([2, 2], DType.float64);
            expect(
              () => pad(a, PadWidth.all(1), out: outBadRank),
              throwsArgumentError,
            );

            final outZero = NDArray<Float64>.zeros([2], DType.float64);
            final resZero = pad(a, PadWidth.all(0), out: outZero);
            expect(identical(resZero, outZero), isTrue);
            expect(outZero.toList(), equals([1.0, 2.0]));
          });
        },
      );

      test('gradientArray spacings list matching targetAxes', () {
        NDArray.scope(() {
          final f2D = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final res = gradientArray(
            f2D,
            spacings: [Spacing.step(0.5), Spacing.step(2.0)],
          );
          expect(res.length, 2);
          expect(res[0].shape, [2, 2]);
          expect(res[1].shape, [2, 2]);

          expect(
            () => gradientArray(f2D, spacings: [Spacing.step(1.0)]),
            throwsArgumentError,
          );

          final outWrongLen = [
            NDArray<Float64>.zeros([2, 2], DType.float64),
          ];
          expect(
            () => gradientArray(f2D, out: outWrongLen),
            throwsArgumentError,
          );

          final outDisp = [
            NDArray<Float64>.zeros([2, 2], DType.float64)..dispose(),
            NDArray<Float64>.zeros([2, 2], DType.float64),
          ];
          expect(() => gradientArray(f2D, out: outDisp), throwsStateError);

          final outNonContig = [
            NDArray<Float64>.zeros([
              2,
              4,
            ], DType.float64).slice([Slice(start: 0, stop: 4, step: 2)]),
            NDArray<Float64>.zeros([2, 2], DType.float64),
          ];
          expect(
            () => gradientArray(f2D, out: outNonContig),
            throwsArgumentError,
          );
        });
      });
    });

    // =========================================================================
    // SECTION 8: FULL DTYPE SWEEP OF STRIDED REDUCTIONS & QUANTILE/MEDIAN/COV
    // =========================================================================
    group('8. Full DType Sweep of Strided Reductions, Quantiles & Covariances', () {
      test(
        'prod, std, variance, and quantile across all numeric DTypes strided & contiguous',
        () {
          NDArray.scope(() {
            for (final dt in numericDTypes) {
              final fullData = [1, 99, 2, 99, 3, 99, 4, 99, 5];
              final fullArr = NDArray.fromList(fullData, [9], dt);
              final strided = fullArr.slice([
                Slice(start: 0, stop: 9, step: 2),
              ]); // [1, 2, 3, 4, 5]

              // prod on strided array (exercises strided copy & switch on dtype)
              final p = prod(strided);
              expect(p.scalar.toDouble(), closeTo(120.0, 1e-5));

              // std & variance on contiguous and strided arrays
              final vContig = variance(
                NDArray.fromList([1, 2, 3, 4, 5], [5], dt),
              );
              expect(vContig.scalar, closeTo(2.0, 1e-5));

              final sContig = std(NDArray.fromList([1, 2, 3, 4, 5], [5], dt));
              expect(sContig.scalar, closeTo(math.sqrt(2.0), 1e-5));

              final vStrided = variance(strided);
              expect(vStrided.scalar, closeTo(2.0, 1e-5));

              // quantile across all numeric types and negative axis
              final qVal = quantile(
                NDArray.fromList([1, 2, 3, 4, 5], [5], dt),
                0.5,
              );
              expect(qVal.scalar, closeTo(3.0, 1e-5));

              final arr2D = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], dt);
              final qNegAx = quantile(arr2D, 0.5, axis: -1);
              expect(qNegAx.shape, [2]);
            }
          });
        },
      );

      test('nanmin and nanmax across all individual numeric DTypes', () {
        NDArray.scope(() {
          for (final dt in numericDTypes) {
            final a = NDArray.fromList([5, 2, 8, 1, 9, 3], [2, 3], dt);
            final nmin = nanmin(a);
            expect(nmin.scalar.toInt(), equals(1));
            final nmax = nanmax(a);
            expect(nmax.scalar.toInt(), equals(9));

            final nminNeg = nanmin(a, axis: -1);
            expect(nminNeg.shape, [2]);

            final nmaxNeg = nanmax(a, axis: -1);
            expect(nmaxNeg.shape, [2]);
          }
        });
      });

      test('empty array reductions across all DTypes', () {
        NDArray.scope(() {
          // Complex sum & mean on empty
          final emptyCpx = NDArray.fromList([], [0], DType.complex128);
          final sCpx = sum(emptyCpx);
          expect(sCpx.scalar, equals(Complex(0.0, 0.0)));
          final mCpx = mean(emptyCpx);
          expect((mCpx.scalar as Complex).real.isNaN, isTrue);

          // Boolean sum on empty
          final emptyBool = NDArray<Boolean>.fromList([], [0], DType.boolean);
          final sBool = sum(emptyBool);
          expect(sBool.scalar, isFalse);
          expect(sumAs(emptyBool, DType.int64).scalar, equals(0));

          // Float sum on empty
          final emptyFloat = NDArray<Float64>.fromList([], [0], DType.float64);
          final sFloat = sum(emptyFloat);
          expect(sFloat.scalar, equals(0.0));
          final mFloat = mean(emptyFloat);
          expect((mFloat.scalar as double).isNaN, isTrue);

          // Int sum on empty
          final emptyInt = NDArray.fromList([], [0], DType.int32);
          final sInt = sum(emptyInt);
          expect(sInt.scalar, equals(0));
        });
      });

      test(
        'cov with integer input array, weights validation and rank errors',
        () {
          NDArray.scope(() {
            final intArr = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final cInt = cov(intArr);
            expect(cInt.shape, [2, 2]);

            final fweightsDisp = NDArray.fromList([1, 2], [2], DType.int32)
              ..dispose();
            expect(() => cov(intArr, fweights: fweightsDisp), throwsStateError);

            final aweightsDisp = NDArray<Float64>.fromList(
              [1.0, 1.0],
              [2],
              DType.float64,
            )..dispose();
            expect(() => cov(intArr, aweights: aweightsDisp), throwsStateError);

            final emptyArr = NDArray.fromList([], [0], DType.int32);
            expect(() => cov(emptyArr), throwsArgumentError);
            expect(() => cov(intArr, y: emptyArr), throwsArgumentError);

            final arr3D = NDArray.fromList(List.generate(8, (i) => i), [
              2,
              2,
              2,
            ], DType.int32);
            expect(() => cov(arr3D), throwsArgumentError);
          });
        },
      );

      test('calculus trapz and gradient remaining branch coverage', () {
        NDArray.scope(() {
          // trapz with real StepSpacing on Complex64
          final c64Arr = NDArray.fromList(
            [Complex(1, 1), Complex(2, 2), Complex(3, 3)],
            [3],
            DType.complex64,
          );
          final resRealStep = trapz(c64Arr, spacing: Spacing.step(1.0));
          expect(resRealStep.dtype, DType.complex64);

          // trapz with complex CoordinateSpacing on Complex128
          final c128Arr = NDArray.fromList(
            [Complex(1, 1), Complex(2, 2), Complex(3, 3)],
            [3],
            DType.complex128,
          );
          final resCpxCoords = trapz(
            c128Arr,
            spacing: Spacing.coordinates([
              Complex(0, 0),
              Complex(1, 1),
              Complex(2, 2),
            ]),
          );
          expect(resCpxCoords.dtype, DType.complex128);

          // trapz with double CoordinateSpacing on Float64
          final f64Arr = NDArray<Float64>.fromList(
            [1.0, 4.0, 9.0],
            [3],
            DType.float64,
          );
          final resDoubleCoords = trapz(
            f64Arr,
            spacing: Spacing.coordinates([0.0, 1.0, 3.0]),
          );
          expect(resDoubleCoords.scalar, closeTo(15.5, 1e-5));

          // gradient with double CoordinateSpacing on Float64
          final gDoubleCoords = gradient(
            f64Arr,
            spacing: Spacing.coordinates([0.0, 1.0, 3.0]),
          );
          expect(gDoubleCoords.shape, [3]);

          // gradient error validations
          final fDisp = NDArray<Float64>.fromList([1.0], [1], DType.float64)
            ..dispose();
          expect(() => gradient(fDisp), throwsStateError);

          final fValid = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final outDisp = NDArray<Float64>.zeros([3], DType.float64)..dispose();
          expect(() => gradient(fValid, out: outDisp), throwsStateError);

          expect(() => gradient(fValid, edgeOrder: 3), throwsArgumentError);

          final intArr = NDArray.fromList([1, 2, 3], [3], DType.int32);
          expect(gradient(intArr).toList(), equals([1.0, 1.0, 1.0]));

          final boolArr = NDArray<Boolean>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          expect(() => gradient(boolArr), throwsArgumentError);

          expect(
            () => gradient(fValid, spacing: Spacing.step(Complex(1, 0))),
            throwsArgumentError,
          );
          expect(() => gradient(fValid, axis: 5), throwsArgumentError);

          final shortArr = NDArray<Float64>.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
          );
          expect(() => gradient(shortArr, edgeOrder: 2), throwsArgumentError);

          expect(
            () => gradient(fValid, spacing: Spacing.coordinates([0.0, 1.0])),
            throwsArgumentError,
          );

          final outBad = NDArray<Float64>.zeros([5], DType.float64);
          expect(() => gradient(fValid, out: outBad), throwsArgumentError);

          // gradientArray errors
          expect(() => gradientArray(fDisp), throwsStateError);
          expect(() => gradientArray(boolArr), throwsArgumentError);
          expect(
            () => gradientArray(shortArr, edgeOrder: 2),
            throwsArgumentError,
          );
        });
      });
    });

    // =========================================================================
    // SECTION 9: REMAINING STRIDED FFI KERNELS & COMPLEX/BOOLEAN REDUCTIONS
    // =========================================================================
    group('9. Remaining Strided FFI Kernels, Complex & Boolean Reductions', () {
      test(
        'strided sum, prod, mean, std, variance on complex, uint8 and boolean',
        () {
          NDArray.scope(() {
            // Complex128 strided
            final c128Full = NDArray.fromList(
              [
                Complex(1, 1),
                Complex(99, 99),
                Complex(2, 2),
                Complex(99, 99),
                Complex(3, 3),
                Complex(99, 99),
              ],
              [6],
              DType.complex128,
            );
            final c128Str = c128Full.slice([Slice(start: 0, stop: 6, step: 2)]);

            final sumC128 = sum(c128Str);
            expect(sumC128.scalar, equals(Complex(6.0, 6.0)));

            final prodC128 = prod(c128Str);
            expect(prodC128.shape, <int>[]);

            final meanC128 = mean(c128Str);
            expect(meanC128.scalar, equals(Complex(2.0, 2.0)));

            // Complex64 strided
            final c64Full = NDArray.fromList(
              [Complex(1, 1), Complex(99, 99), Complex(2, 2), Complex(99, 99)],
              [4],
              DType.complex64,
            );
            final c64Str = c64Full.slice([Slice(start: 0, stop: 4, step: 2)]);

            final sumC64 = sum(c64Str);
            expect(sumC64.dtype, DType.complex64);

            final prodC64 = prod(c64Str);
            expect(prodC64.dtype, DType.complex64);

            final meanC64 = mean(c64Str);
            expect(meanC64.dtype, DType.complex128);

            // Boolean strided
            final boolFull = NDArray<Boolean>.fromList(
              [true, false, true, false, true],
              [5],
              DType.boolean,
            );
            final boolStr = boolFull.slice([Slice(start: 0, stop: 5, step: 2)]);
            expect(sumAs(boolStr, DType.int64).scalar, equals(3));
            expect(prodAs(boolStr, DType.int64).scalar, equals(1));

            // Uint8 strided & axis reductions
            final u8Full = NDArray.fromList(
              [2, 99, 4, 99, 6],
              [5],
              DType.uint8,
            );
            final u8Str = u8Full.slice([Slice(start: 0, stop: 5, step: 2)]);
            expect(mean(u8Str).scalar, closeTo(4.0, 1e-5));
            expect(variance(u8Str).scalar, closeTo(8.0 / 3.0, 1e-5));
            expect(std(u8Str).scalar, closeTo(math.sqrt(8.0 / 3.0), 1e-5));

            final u8_2D = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
            final varU8Ax = variance(u8_2D, axis: 0);
            expect(varU8Ax.shape, [2]);
          });
        },
      );

      test('nanmin, nanmax complex exception and uint8 support', () {
        NDArray.scope(() {
          final cpx = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
          expect(() => nanmin(cpx as dynamic), throwsUnsupportedError);
          expect(() => nanmax(cpx as dynamic), throwsUnsupportedError);

          final u8Arr = NDArray.fromList([10, 20, 5], [3], DType.uint8);
          expect(nanmin(u8Arr).scalar, equals(5));
          expect(nanmax(u8Arr).scalar, equals(20));
        });
      });

      test('cov with 2D y array and fweights size validation', () {
        NDArray.scope(() {
          final m = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final y = NDArray<Float64>.fromList(
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [2, 3],
            DType.float64,
          );
          final res = cov(m, y: y);
          expect(res.shape, [4, 4]);

          final fweightsBad = NDArray.fromList([1, 2], [2], DType.int32);
          expect(() => cov(m, fweights: fweightsBad), throwsArgumentError);
        });
      });

      test('strided quantile across multiple dtypes', () {
        NDArray.scope(() {
          final fFull = NDArray<Float64>.fromList(
            [1.0, 99.0, 2.0, 99.0, 3.0],
            [5],
            DType.float64,
          );
          final fStr = fFull.slice([Slice(start: 0, stop: 5, step: 2)]);
          final q = quantile(fStr, 0.5);
          expect(q.scalar, closeTo(2.0, 1e-5));

          final i32Full = NDArray.fromList(
            [10, 99, 20, 99, 30],
            [5],
            DType.int32,
          );
          final i32Str = i32Full.slice([Slice(start: 0, stop: 5, step: 2)]);
          final q32 = quantile(i32Str, 0.5);
          expect(q32.scalar, closeTo(20.0, 1e-5));

          final u8Full = NDArray.fromList([1, 99, 5, 99, 9], [5], DType.uint8);
          final u8Str = u8Full.slice([Slice(start: 0, stop: 5, step: 2)]);
          final qU8 = quantile(u8Str, 0.5);
          expect(qU8.scalar, closeTo(5.0, 1e-5));
        });
      });
    });

    // =========================================================================
    // SECTION 10: ULTIMATE COVERAGE SWEEP (STRIDED, COV, NAN AXES & MEDIAN)
    // =========================================================================
    group('10. Ultimate Coverage Sweep', () {
      test(
        'strided sweep of sum, prod, mean, std, variance across ALL 15 DTypes',
        () {
          NDArray.scope(() {
            for (final dt in allDTypes) {
              final NDArray<AnySpec> arrFull;
              if (dt.isComplex) {
                arrFull = NDArray.fromList(
                  [
                    Complex(1, 1),
                    Complex(99, 99),
                    Complex(2, 2),
                    Complex(99, 99),
                    Complex(3, 3),
                    Complex(99, 99),
                  ],
                  [6],
                  dt,
                );
              } else if (dt == DType.boolean) {
                arrFull = NDArray<Boolean>.fromList(
                  [true, false, true, false, true, false],
                  [6],
                  DType.boolean,
                );
              } else {
                arrFull = NDArray.fromList([1, 99, 2, 99, 3, 99], [6], dt);
              }

              final strided = arrFull.slice([
                Slice(start: 0, stop: 6, step: 2),
              ]);

              final s = sum(strided);
              expect(s.shape, <int>[]);

              final p = prod(strided);
              expect(p.shape, <int>[]);

              if (dt != DType.boolean) {
                final m = mean(strided);
                expect(m.shape, <int>[]);
              }

              if (!dt.isComplex && dt != DType.boolean) {
                final numArr = NDArray.fromList(
                  [1, 99, 2, 99, 3, 99],
                  [6],
                  dt,
                ).slice([Slice(start: 0, stop: 6, step: 2)]);
                final v = variance(numArr);
                expect(v.shape, <int>[]);

                final sd = std(numArr);
                expect(sd.shape, <int>[]);
              }
            }
          });
        },
      );

      test('cov advanced weights validation and 2D y shape adjustments', () {
        NDArray.scope(() {
          final m = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          // Negative fweights
          final negFw = NDArray.fromList([1, -1, 1], [3], DType.int32);
          expect(() => cov(m, fweights: negFw), throwsArgumentError);

          // aweights size mismatch
          final badAw = NDArray<Float64>.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
          );
          expect(() => cov(m, aweights: badAw), throwsArgumentError);

          // Negative aweights
          final negAw = NDArray<Float64>.fromList(
            [1.0, -1.0, 1.0],
            [3],
            DType.float64,
          );
          expect(() => cov(m, aweights: negAw), throwsArgumentError);

          final y2x3 = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final resCovY = cov(m, y: y2x3);
          expect(resCovY.shape, [4, 4]);

          final outBad = NDArray<Float64>.zeros([2, 2], DType.float64);
          expect(
            () => corrcoef(
              NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64),
              out: outBad,
            ),
            throwsArgumentError,
          );
        });
      });

      test('nanmean on complex 2D array along axes', () {
        NDArray.scope(() {
          final cpx2D = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(double.nan, 4.0),
              Complex(3.0, double.nan),
              Complex(4.0, 8.0),
            ],
            [2, 2],
            DType.complex128,
          );

          final nmAx0 = nanmean<DTypeTag>(cpx2D, axis: 0);
          expect(nmAx0.shape, [2]);

          final nmAx1 = nanmean<DTypeTag>(cpx2D, axis: 1);
          expect(nmAx1.shape, [2]);
        });
      });

      test(
        'median 1D float32 and float64 direct kernels and negative axis',
        () {
          NDArray.scope(() {
            final f32Arr = NDArray<Float32>.fromList(
              [3.0, 1.0, 2.0, 4.0],
              [4],
              DType.float32,
            );
            final medF32 = median(f32Arr);
            expect(medF32.scalar, closeTo(2.5, 1e-5));

            final f64Arr = NDArray<Float64>.fromList(
              [3.0, 1.0, 2.0, 4.0],
              [4],
              DType.float64,
            );
            final medF64 = median(f64Arr);
            expect(medF64.scalar, closeTo(2.5, 1e-5));

            final arr2D = NDArray<Float64>.fromList(
              [1.0, 3.0, 2.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final medNeg = median(arr2D, axis: -1);
            expect(medNeg.shape, [2]);
            expect(medNeg.toList(), equals([2.0, 3.0]));
          });
        },
      );

      test('nanmin, nanmax, min, max zero-sized axis validation', () {
        NDArray.scope(() {
          final empty2D = NDArray<Float64>.fromList([], [0, 5], DType.float64);
          expect(() => min(empty2D, axis: 0), throwsArgumentError);
          expect(() => max(empty2D, axis: 0), throwsArgumentError);
          expect(() => nanmin(empty2D, axis: 0), throwsArgumentError);
          expect(() => nanmax(empty2D, axis: 0), throwsArgumentError);
        });
      });
    });

    // =========================================================================
    // SECTION 11: FULL AXIS REDUCTIONS ACROSS ALL DTYPES (SUM, PROD, MEAN, VAR)
    // =========================================================================
    group('11. Full Axis Reductions Across All DTypes', () {
      test('sum and prod along axis 0 and axis 1 across all 15 DTypes', () {
        NDArray.scope(() {
          for (final dt in allDTypes) {
            final NDArray<AnySpec> arr2D = dt.isComplex
                ? NDArray.fromList(
                    [
                      Complex(1, 1),
                      Complex(2, 2),
                      Complex(3, 3),
                      Complex(4, 4),
                    ],
                    [2, 2],
                    dt,
                  )
                : dt == DType.boolean
                ? NDArray<Boolean>.fromList(
                    [true, false, true, true],
                    [2, 2],
                    DType.boolean,
                  )
                : NDArray.fromList([1, 2, 3, 4], [2, 2], dt);

            final s0 = sum(arr2D, axis: 0);
            expect(s0.shape, [2]);

            final s1 = sum(arr2D, axis: 1);
            expect(s1.shape, [2]);

            final p0 = prod(arr2D, axis: 0);
            expect(p0.shape, [2]);

            final p1 = prod(arr2D, axis: 1);
            expect(p1.shape, [2]);
          }
        });
      });

      test('mean, std, variance, nanmin, nanmax on uint8 2D along axes', () {
        NDArray.scope(() {
          final u8 = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.uint8);

          final m0 = mean(u8, axis: 0);
          expect(m0.shape, [2]);

          final v0 = variance(u8, axis: 0);
          expect(v0.shape, [2]);

          final s0 = std(u8, axis: 0);
          expect(s0.shape, [2]);

          final nmin0 = nanmin(u8, axis: 0);
          expect(nmin0.shape, [2]);

          final nmax0 = nanmax(u8, axis: 0);
          expect(nmax0.shape, [2]);
        });
      });

      test('cov with aweights alone (without fweights)', () {
        NDArray.scope(() {
          final m = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final aw = NDArray<Float64>.fromList(
            [1.0, 2.0, 1.0],
            [3],
            DType.float64,
          );
          final res = cov(m, aweights: aw);
          expect(res.shape, [2, 2]);
        });
      });
    });
  });
}

NDArray<Float64> y2DWrong(NDArray<Float64> arr) {
  return NDArray<Float64>.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
}

void _runPadTestsForDType<T extends DTypeTag>(DType<T> dt) {
  final NDArray<T> arr1D = dt.isComplex
      ? NDArray.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              dt as DType<AnySpec>,
            )
            as NDArray<T>
      : dt == DType.boolean
      ? NDArray<Boolean>.fromList([true, false], [2], DType.boolean)
            as NDArray<T>
      : NDArray.fromList([1, 2], [2], dt);

  final NDArray<T> arr2D = dt.isComplex
      ? NDArray.fromList(
              [Complex(1, 1), Complex(2, 2), Complex(3, 3), Complex(4, 4)],
              [2, 2],
              dt as DType<AnySpec>,
            )
            as NDArray<T>
      : dt == DType.boolean
      ? NDArray<Boolean>.fromList(
              [true, false, false, true],
              [2, 2],
              DType.boolean,
            )
            as NDArray<T>
      : NDArray.fromList([1, 2, 3, 4], [2, 2], dt);

  final modes = [
    PadMode.constant,
    PadMode.edge,
    PadMode.reflect,
    PadMode.symmetric,
    PadMode.wrap,
  ];

  for (final mode in modes) {
    final p1 = pad(arr1D, PadWidth.all(1, 1), mode: mode);
    expect(p1.shape, [4], reason: 'Failed 1D for dtype  in mode ');
    expect(p1.dtype, dt);

    final p2 = pad(arr2D, PadWidth.all(1, 1), mode: mode);
    expect(p2.shape, [4, 4], reason: 'Failed 2D for dtype  in mode ');
    expect(p2.dtype, dt);
  }
}

void _runFallbackPadTestsForDType<T extends DTypeTag>(DType<T> dt) {
  final fallbackModes = [
    PadMode.linearRamp,
    PadMode.mean,
    PadMode.median,
    PadMode.minimum,
    PadMode.maximum,
  ];

  final NDArray<T> arr = dt.isComplex
      ? NDArray.fromList(
              [Complex(1, 2), Complex(3, 4), Complex(5, 6)],
              [3],
              dt as DType<AnySpec>,
            )
            as NDArray<T>
      : dt == DType.boolean
      ? NDArray<Boolean>.fromList([true, false, true], [3], DType.boolean)
            as NDArray<T>
      : NDArray.fromList([1, 2, 4], [3], dt);

  for (final mode in fallbackModes) {
    final p = pad(arr, PadWidth.all(1, 1), mode: mode);
    expect(p.shape, [5], reason: 'Fallback pad failed for dtype  in mode ');
    expect(p.dtype, dt);
  }
}
