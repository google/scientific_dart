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

void main() {
  group('Review Cycle 18: multinomial, histogram, pad uint64, financial when', () {
    group(
      '1. multinomial() bounds, NaN, and zero-sum probability validation',
      () {
        test(
          'throws ArgumentError on empty pvals (shape: [0]) without heap corruption',
          () {
            NDArray.scope(() {
              final emptyPvals = NDArray<Float64>.zeros([0], DType.float64);
              expect(() => multinomial(10, emptyPvals), throwsArgumentError);
              expect(() => multinomial(0, emptyPvals), throwsArgumentError);
            });
          },
        );

        test('throws ArgumentError on NaN or negative probabilities', () {
          NDArray.scope(() {
            final nanPvals = NDArray<Float64>.fromList(
              [0.5, double.nan, 0.5],
              [3],
              DType.float64,
            );
            expect(() => multinomial(5, nanPvals), throwsArgumentError);

            final negPvals = NDArray<Float64>.fromList(
              [0.5, -0.1, 0.6],
              [3],
              DType.float64,
            );
            expect(() => multinomial(5, negPvals), throwsArgumentError);
          });
        });

        test('throws ArgumentError when sum of probabilities is zero', () {
          NDArray.scope(() {
            final zeroPvals = NDArray<Float64>.fromList(
              [0.0, 0.0, 0.0],
              [3],
              DType.float64,
            );
            expect(() => multinomial(10, zeroPvals), throwsArgumentError);
            expect(() => multinomial(0, zeroPvals), throwsArgumentError);
          });
        });
      },
    );

    group(
      '2. histogram() full DType coverage and uint64 unsigned MSB handling',
      () {
        test(
          'supports float16, bfloat16, int8, uint16, uint32, uint64 for x and weights',
          () {
            NDArray.scope(() {
              final dtypes = <DType<AnySpec>>[
                DType.float16,
                DType.bfloat16,
                DType.int8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
              ];

              for (final dt in dtypes) {
                final x = castNDArray<AnySpec>(
                  NDArray<Float64>.fromList(
                    [1.0, 2.0, 3.0, 4.0],
                    [4],
                    DType.float64,
                  ),
                  dt,
                );
                final weights = castNDArray<AnySpec>(
                  NDArray<Float64>.fromList(
                    [1.0, 2.0, 3.0, 4.0],
                    [4],
                    DType.float64,
                  ),
                  dt,
                );

                // Uniform bins without weights
                final resUnweighted = histogram(x, bins: 2);
                expect(
                  resUnweighted.hist.toList(),
                  equals([2, 2]),
                  reason: 'Unweighted uniform histogram failed for dtype $dt',
                );

                // Uniform bins with weights
                final resWeighted = histogram(x, bins: 2, weights: weights);
                final weightedList = resWeighted.hist
                    .toList()
                    .map((e) => e.toDouble())
                    .toList();
                expect(
                  weightedList,
                  equals([3.0, 7.0]),
                  reason: 'Weighted uniform histogram failed for dtype $dt',
                );

                // Non-uniform binEdges (binsearch) with weights
                final edges = NDArray<Float64>.fromList(
                  [0.5, 2.5, 4.5],
                  [3],
                  DType.float64,
                );
                final resBinsearch = histogram(
                  x,
                  bins: edges,
                  weights: weights,
                );
                final binsearchList = resBinsearch.hist
                    .toList()
                    .map((e) => e.toDouble())
                    .toList();
                expect(
                  binsearchList,
                  equals([3.0, 7.0]),
                  reason: 'Weighted binsearch histogram failed for dtype $dt',
                );
              }
            });
          },
        );

        test('handles uint64 values >= 2^63 as positive unsigned integers', () {
          NDArray.scope(() {
            final msbVal = BigInt.parse(
              '9223372036854775808',
            ).toSigned(64).toInt();
            final x = NDArray<Uint64>.fromList([0, msbVal], [2], DType.uint64);
            final (:hist, :binEdges) = histogram(x, bins: 2);
            expect(hist.toList(), equals([1, 1]));
            expect(binEdges.getCell([0]), equals(0.0));
            expect(binEdges.getCell([2]), closeTo(9223372036854775808.0, 1e3));
          });
        });
      },
    );

    group('3. pad() with DType.uint64 unsigned values >= 2^63', () {
      test(
        'preserves uint64 values >= 2^63 in constant and linearRamp modes',
        () {
          NDArray.scope(() {
            final msbVal = BigInt.parse(
              '9223372036854775808',
            ).toSigned(64).toInt();
            final src = NDArray<Uint64>.fromList([msbVal], [1], DType.uint64);

            final paddedConst = pad(
              src,
              PadWidth.all(1),
              mode: PadMode.constant,
              constantValues: PadValues.all(msbVal),
            );
            expect(paddedConst.shape, equals([3]));
            expect(paddedConst.getCell([0]), equals(msbVal));
            expect(paddedConst.getCell([1]), equals(msbVal));
            expect(paddedConst.getCell([2]), equals(msbVal));

            final paddedRamp = pad(
              src,
              PadWidth.all(1),
              mode: PadMode.linearRamp,
              endValues: PadValues.all(msbVal),
            );
            expect(paddedRamp.shape, equals([3]));
            expect(paddedRamp.getCell([0]), equals(msbVal));
            expect(paddedRamp.getCell([1]), equals(msbVal));
            expect(paddedRamp.getCell([2]), equals(msbVal));
          });
        },
      );
    });

    group('4. fv() and pv() PaymentDue enum parameter', () {
      test('defaults to PaymentDue.end and supports PaymentDue.begin', () {
        NDArray.scope(() {
          final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
          final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
          final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
          final pvVal = NDArray<Float64>.scalar(-1000.0, dtype: DType.float64);

          final fvDefault = fv(rate, nper, pmt, pvVal);
          final fvEnd = fv(rate, nper, pmt, pvVal, when: PaymentDue.end);
          final fvBegin = fv(rate, nper, pmt, pvVal, when: PaymentDue.begin);

          expect(fvDefault.scalar, closeTo(fvEnd.scalar, 1e-12));
          expect(fvBegin.scalar, greaterThan(fvEnd.scalar));

          final pvDefault = pv(rate, nper, pmt, fvEnd);
          final pvEnd = pv(rate, nper, pmt, fvEnd, when: PaymentDue.end);
          final pvBegin = pv(rate, nper, pmt, fvEnd, when: PaymentDue.begin);

          expect(pvDefault.scalar, closeTo(pvEnd.scalar, 1e-12));
          expect(pvDefault.scalar, closeTo(-1000.0, 1e-6));
          expect(pvBegin.scalar, greaterThan(pvEnd.scalar));
        });
      });
    });
  });
}
