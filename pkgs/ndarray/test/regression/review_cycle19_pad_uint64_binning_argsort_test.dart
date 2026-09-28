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
  group('Cycle 19 Regression Tests', () {
    group(
      '1. pad() preserves DType.uint64 and DType.int64 values > 2^53 across all modes',
      () {
        test(
          'DType.int64 > 2^53 across constant (including empty [0, 2]), linearRamp, max, min, mean, median',
          () {
            NDArray.scope(() {
              const v1 = (1 << 53) + 1;
              const v2 = (1 << 53) + 3;
              const vMid = (1 << 53) + 2;
              const vEndBefore = (1 << 53) - 1;
              const vEndAfter = (1 << 53) + 5;

              final src = NDArray<Int64>.fromList([v1, v2], [2], DType.int64);

              // Constant mode
              final pConst = pad(
                src,
                PadWidth.all(1),
                mode: PadMode.constant,
                constantValues: PadValues.all(vEndBefore, vEndAfter),
              );
              expect([
                for (var i = 0; i < pConst.size; i++) pConst.getCell([i]),
              ], equals([vEndBefore, v1, v2, vEndAfter]));

              // Constant mode on empty array [0, 2]
              final emptySrc = NDArray<Int64>.fromList(const <int>[], [
                0,
                2,
              ], DType.int64);
              final pEmptyConst = pad(
                emptySrc,
                PadWidth.all(1),
                mode: PadMode.constant,
                constantValues: PadValues.all(v1),
              );
              expect(pEmptyConst.shape, equals([2, 4]));
              for (var r = 0; r < 2; r++) {
                for (var c = 0; c < 4; c++) {
                  expect(pEmptyConst.getCell([r, c]), equals(v1));
                }
              }

              // linearRamp mode
              final pRamp = pad(
                src,
                PadWidth.all(2),
                mode: PadMode.linearRamp,
                endValues: PadValues.all(vEndBefore, vEndAfter),
              );
              expect(
                [
                  for (var i = 0; i < pRamp.size; i++) pRamp.getCell([i]),
                ],
                equals([vEndBefore, 1 << 53, v1, v2, (1 << 53) + 4, vEndAfter]),
              );

              // max mode
              final pMax = pad(src, PadWidth.all(1), mode: PadMode.max);
              expect([
                for (var i = 0; i < pMax.size; i++) pMax.getCell([i]),
              ], equals([v2, v1, v2, v2]));

              // min mode
              final pMin = pad(src, PadWidth.all(1), mode: PadMode.min);
              expect([
                for (var i = 0; i < pMin.size; i++) pMin.getCell([i]),
              ], equals([v1, v1, v2, v1]));

              // mean mode
              final pMean = pad(src, PadWidth.all(1), mode: PadMode.mean);
              expect([
                for (var i = 0; i < pMean.size; i++) pMean.getCell([i]),
              ], equals([vMid, v1, v2, vMid]));

              // median mode (even length 2 -> midpoint)
              final pMedian = pad(src, PadWidth.all(1), mode: PadMode.median);
              expect([
                for (var i = 0; i < pMedian.size; i++) pMedian.getCell([i]),
              ], equals([vMid, v1, v2, vMid]));
            });
          },
        );

        test(
          'DType.uint64 > 2^53 and >= 2^63 (up to 0xFFFFFFFFFFFFFFFE) across all modes',
          () {
            NDArray.scope(() {
              // 0xFFFFFFFFFFFFFFFC = -4, 0xFFFFFFFFFFFFFFFD = -3, 0xFFFFFFFFFFFFFFFE = -2
              const u1 = -4; // 0xFFFFFFFFFFFFFFFC
              const uMid = -3; // 0xFFFFFFFFFFFFFFFD
              const u2 = -2; // 0xFFFFFFFFFFFFFFFE
              const uBefore = -6; // 0xFFFFFFFFFFFFFFFA
              const uBeforeMid = -5; // 0xFFFFFFFFFFFFFFFB
              const uAfter = -4; // 0xFFFFFFFFFFFFFFFC

              final src = NDArray<Uint64>.fromList([u1, u2], [2], DType.uint64);

              // Constant mode
              final pConst = pad(
                src,
                PadWidth.all(1),
                mode: PadMode.constant,
                constantValues: PadValues.all(uBefore, u2),
              );
              expect([
                for (var i = 0; i < pConst.size; i++) pConst.getCell([i]),
              ], equals([uBefore, u1, u2, u2]));

              // Constant mode on empty array [0, 2]
              final emptySrc = NDArray<Uint64>.fromList(const <int>[], [
                0,
                2,
              ], DType.uint64);
              final pEmptyConst = pad(
                emptySrc,
                PadWidth.all(1),
                mode: PadMode.constant,
                constantValues: PadValues.all(u2),
              );
              expect(pEmptyConst.shape, equals([2, 4]));
              for (var r = 0; r < 2; r++) {
                for (var c = 0; c < 4; c++) {
                  expect(pEmptyConst.getCell([r, c]), equals(u2));
                }
              }

              // linearRamp mode
              final pRamp = pad(
                src,
                PadWidth.all(2),
                mode: PadMode.linearRamp,
                endValues: PadValues.all(uBefore, uAfter),
              );
              expect([
                for (var i = 0; i < pRamp.size; i++) pRamp.getCell([i]),
              ], equals([uBefore, uBeforeMid, u1, u2, uMid, uAfter]));

              // max mode (unsigned: u2 = 0xFFFFFFFFFFFFFFFE > u1 = 0xFFFFFFFFFFFFFFFC)
              final pMax = pad(src, PadWidth.all(1), mode: PadMode.max);
              expect([
                for (var i = 0; i < pMax.size; i++) pMax.getCell([i]),
              ], equals([u2, u1, u2, u2]));

              // min mode (unsigned: u1 = 0xFFFFFFFFFFFFFFFC < u2 = 0xFFFFFFFFFFFFFFFE)
              final pMin = pad(src, PadWidth.all(1), mode: PadMode.min);
              expect([
                for (var i = 0; i < pMin.size; i++) pMin.getCell([i]),
              ], equals([u1, u1, u2, u1]));

              // mean mode
              final pMean = pad(src, PadWidth.all(1), mode: PadMode.mean);
              expect([
                for (var i = 0; i < pMean.size; i++) pMean.getCell([i]),
              ], equals([uMid, u1, u2, uMid]));

              // median mode (including unsigned comparison across 2^63 boundary)
              final pMedian = pad(src, PadWidth.all(1), mode: PadMode.median);
              expect([
                for (var i = 0; i < pMedian.size; i++) pMedian.getCell([i]),
              ], equals([uMid, u1, u2, uMid]));

              // Also test values (1 << 53) + 1 and (1 << 53) + 3 for DType.uint64
              const p53_1 = (1 << 53) + 1;
              const p53_2 = (1 << 53) + 3;
              const p53Mid = (1 << 53) + 2;
              final src53 = NDArray<Uint64>.fromList(
                [p53_1, p53_2],
                [2],
                DType.uint64,
              );
              final p53Mean = pad(src53, PadWidth.all(1), mode: PadMode.mean);
              expect([
                for (var i = 0; i < p53Mean.size; i++) p53Mean.getCell([i]),
              ], equals([p53Mid, p53_1, p53_2, p53Mid]));
            });
          },
        );
      },
    );

    group(
      '2. histogram() and digitize() reject NaN in bins and non-finite bounds',
      () {
        test('digitize() throws ArgumentError when bins contains NaN', () {
          NDArray.scope(() {
            final x = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
            final binsStartNaN = NDArray<Float64>.fromList(
              [double.nan, 1.0, 2.0],
              [3],
              DType.float64,
            );
            final binsMidNaN = NDArray<Float64>.fromList(
              [0.0, double.nan, 2.0],
              [3],
              DType.float64,
            );
            final binsSingleNaN = NDArray<Float64>.fromList(
              [double.nan],
              [1],
              DType.float64,
            );

            expect(() => digitize(x, binsStartNaN), throwsArgumentError);
            expect(() => digitize(x, binsMidNaN), throwsArgumentError);
            expect(() => digitize(x, binsSingleNaN), throwsArgumentError);
          });
        });

        test(
          'histogram() throws ArgumentError on NaN in bins or non-finite range/x',
          () {
            NDArray.scope(() {
              final x = NDArray<Float64>.fromList(
                [1.0, 2.0, 3.0],
                [3],
                DType.float64,
              );

              // NaN in explicit bins
              final binsNaN0 = NDArray<Float64>.fromList(
                [double.nan, 1.0, 2.0],
                [3],
                DType.float64,
              );
              final binsNaN1 = NDArray<Float64>.fromList(
                [0.0, double.nan, 2.0],
                [3],
                DType.float64,
              );
              expect(() => histogram(x, bins: binsNaN0), throwsArgumentError);
              expect(() => histogram(x, bins: binsNaN1), throwsArgumentError);

              // Non-finite explicit range
              expect(
                () => histogram(x, bins: 4, range: (double.nan, 5.0)),
                throwsArgumentError,
              );
              expect(
                () => histogram(x, bins: 4, range: (0.0, double.nan)),
                throwsArgumentError,
              );
              expect(
                () => histogram(
                  x,
                  bins: 4,
                  range: (double.negativeInfinity, 5.0),
                ),
                throwsArgumentError,
              );
              expect(
                () => histogram(x, bins: 4, range: (0.0, double.infinity)),
                throwsArgumentError,
              );

              // Non-finite x with auto-range
              final xWithNaN = NDArray<Float64>.fromList(
                [1.0, double.nan, 3.0],
                [3],
                DType.float64,
              );
              final xWithInf = NDArray<Float64>.fromList(
                [1.0, double.infinity, 3.0],
                [3],
                DType.float64,
              );
              expect(() => histogram(xWithNaN, bins: 4), throwsArgumentError);
              expect(() => histogram(xWithInf, bins: 4), throwsArgumentError);
            });
          },
        );
      },
    );

    group(
      '3. argsort() on complex128 and complex64 compares imag when both real parts are NaN',
      () {
        test(
          'argsort() matches sort() order for [Complex(NaN, 2.0), Complex(NaN, 1.0)]',
          () {
            NDArray.scope(() {
              final c128 = NDArray<Complex128>.fromList(
                [Complex(double.nan, 2.0), Complex(double.nan, 1.0)],
                [2],
                DType.complex128,
              );
              final idx128 = argsort(c128);
              final sorted128 = sort(c128);
              expect([
                idx128.getCell([0]),
                idx128.getCell([1]),
              ], equals([1, 0]));
              expect(sorted128.getCell([0]).imag, equals(1.0));
              expect(sorted128.getCell([1]).imag, equals(2.0));

              final c64 = NDArray<Complex64>.fromList(
                [Complex(double.nan, 2.0), Complex(double.nan, 1.0)],
                [2],
                DType.complex64,
              );
              final idx64 = argsort(c64);
              final sorted64 = sort(c64);
              expect([
                idx64.getCell([0]),
                idx64.getCell([1]),
              ], equals([1, 0]));
              expect(sorted64.getCell([0]).imag, equals(1.0));
              expect(sorted64.getCell([1]).imag, equals(2.0));
            });
          },
        );
      },
    );
  });
}
