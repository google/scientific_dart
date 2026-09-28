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
  group('Review Cycle 12 Regressions', () {
    group(
      '1. solve() on transposed / non-contiguous 2D matrices (_copyStrided2DMatrix)',
      () {
        test('float64 transposed 2D matrix', () {
          NDArray.scope(() {
            final aOrig = NDArray<Float64>.fromList(
              [4.0, 1.0, 2.0, 3.0],
              [2, 2],
              DType.float64,
            );
            final aT = aOrig.transpose(); // stride1 != 1
            expect(aT.isContiguous, isFalse);
            expect(aT.strides[1], isNot(1));

            final b = NDArray<Float64>.fromList([9.0, 8.0], [2], DType.float64);
            final x = solve<Float64>(aT, b);
            // aT = [[4, 2], [1, 3]], b = [9, 8] => 4x0 + 2x1 = 9, x0 + 3x1 = 8 => x0 = 1.1, x1 = 2.3
            expect(x.getCell([0]), closeTo(1.1, 1e-12));
            expect(x.getCell([1]), closeTo(2.3, 1e-12));
          });
        });

        test('float32 transposed 2D matrix', () {
          NDArray.scope(() {
            final aOrig = NDArray<Float32>.fromList(
              [4.0, 1.0, 2.0, 3.0],
              [2, 2],
              DType.float32,
            );
            final aT = aOrig.transpose();
            final b = NDArray<Float32>.fromList([9.0, 8.0], [2], DType.float32);
            final x = solve<Float32>(aT, b);
            expect(x.getCell([0]), closeTo(1.1, 1e-5));
            expect(x.getCell([1]), closeTo(2.3, 1e-5));
          });
        });

        test('complex128 transposed 2D matrix', () {
          NDArray.scope(() {
            final aOrig = NDArray<Complex128>.fromList(
              [
                Complex(4.0, 0.0),
                Complex(1.0, 0.0),
                Complex(2.0, 0.0),
                Complex(3.0, 0.0),
              ],
              [2, 2],
              DType.complex128,
            );
            final aT = aOrig.transpose();
            final b = NDArray<Complex128>.fromList(
              [Complex(9.0, 0.0), Complex(8.0, 0.0)],
              [2],
              DType.complex128,
            );
            final x = solve<Complex128>(aT, b);
            expect(x.getCell([0]).real, closeTo(1.1, 1e-12));
            expect(x.getCell([0]).imag, closeTo(0.0, 1e-12));
            expect(x.getCell([1]).real, closeTo(2.3, 1e-12));
            expect(x.getCell([1]).imag, closeTo(0.0, 1e-12));
          });
        });

        test('complex64 transposed 2D matrix', () {
          NDArray.scope(() {
            final aOrig = NDArray<Complex64>.fromList(
              [
                Complex(4.0, 0.0),
                Complex(1.0, 0.0),
                Complex(2.0, 0.0),
                Complex(3.0, 0.0),
              ],
              [2, 2],
              DType.complex64,
            );
            final aT = aOrig.transpose();
            final b = NDArray<Complex64>.fromList(
              [Complex(9.0, 0.0), Complex(8.0, 0.0)],
              [2],
              DType.complex64,
            );
            final x = solve<Complex64>(aT, b);
            expect(x.getCell([0]).real, closeTo(1.1, 1e-5));
            expect(x.getCell([0]).imag, closeTo(0.0, 1e-5));
            expect(x.getCell([1]).real, closeTo(2.3, 1e-5));
            expect(x.getCell([1]).imag, closeTo(0.0, 1e-5));
          });
        });
      },
    );

    group('2. fv(), pv(), npv(), and irr() with explicit out: buffer', () {
      test('fv, pv, npv, irr with out: outside and inside NDArray.scope', () {
        // Test outside NDArray.scope
        final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
        final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
        final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
        final pvVal = NDArray<Float64>.scalar(-1000.0, dtype: DType.float64);
        final outScalar = NDArray<Float64>.create([], DType.float64);
        final cashflows = NDArray<Float64>.fromList(
          [-100.0, 39.0, 59.0, 55.0, 20.0],
          [5],
          DType.float64,
        );

        try {
          final expectedFv = fv(rate, nper, pmt, pvVal);
          final resFv = fv(rate, nper, pmt, pvVal, out: outScalar);
          expect(identical(resFv, outScalar), isTrue);
          expect(outScalar.scalar, closeTo(expectedFv.scalar, 1e-12));
          expectedFv.dispose();

          final expectedPv = pv(rate, nper, pmt, pvVal);
          final resPv = pv(rate, nper, pmt, pvVal, out: outScalar);
          expect(identical(resPv, outScalar), isTrue);
          expect(outScalar.scalar, closeTo(expectedPv.scalar, 1e-12));
          expectedPv.dispose();

          final expectedNpv = npv(rate, cashflows);
          final resNpv = npv(rate, cashflows, out: outScalar);
          expect(identical(resNpv, outScalar), isTrue);
          expect(outScalar.scalar, closeTo(expectedNpv.scalar, 1e-12));
          expectedNpv.dispose();

          final expectedIrr = irr(cashflows);
          final resIrr = irr(cashflows, out: outScalar);
          expect(identical(resIrr, outScalar), isTrue);
          expect(outScalar.scalar, closeTo(expectedIrr.scalar, 1e-12));
          expectedIrr.dispose();

          // Test inside NDArray.scope with out allocated in outer scope
          NDArray.scope(() {
            final rFv = fv(rate, nper, pmt, pvVal, out: outScalar);
            expect(identical(rFv, outScalar), isTrue);
            final rPv = pv(rate, nper, pmt, pvVal, out: outScalar);
            expect(identical(rPv, outScalar), isTrue);
            final rNpv = npv(rate, cashflows, out: outScalar);
            expect(identical(rNpv, outScalar), isTrue);
            final rIrr = irr(cashflows, out: outScalar);
            expect(identical(rIrr, outScalar), isTrue);
          });

          // Also test irr degenerate branches (all same sign -> NaN) with out:
          final posFlows = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0],
            [3],
            DType.float64,
          );
          try {
            final nanRes = irr(posFlows, out: outScalar);
            expect(identical(nanRes, outScalar), isTrue);
            expect(outScalar.scalar.isNaN, isTrue);
          } finally {
            posFlows.dispose();
          }
        } finally {
          rate.dispose();
          nper.dispose();
          pmt.dispose();
          pvVal.dispose();
          outScalar.dispose();
          cashflows.dispose();
        }
      });
    });

    group('3. lstsq() on empty matrices ([0, 3] and [3, 0])', () {
      test(
        'lstsq zero-fills out: buffer when m == 0 and n == 3 and does not leak casts',
        () {
          NDArray.scope(() {
            final aEmptyRows = NDArray.zeros([0, 3], DType.int32);
            final bEmptyRows = NDArray.zeros([0], DType.int32);
            final outBuf = NDArray<Float64>.fromList(
              [99.0, -42.0, 123.0],
              [3],
              DType.float64,
            );

            final res = lstsq<DTypeTag, DTypeTag, Float64>(
              aEmptyRows,
              bEmptyRows,
              out: outBuf,
            );
            expect(identical(res.x, outBuf), isTrue);
            expect(res.rank, equals(0));
            expect(outBuf.getCell([0]), equals(0.0));
            expect(outBuf.getCell([1]), equals(0.0));
            expect(outBuf.getCell([2]), equals(0.0));

            // Also test [3, 0] with float16 input
            final aEmptyCols = NDArray<Float16>.zeros([3, 0], DType.float16);
            final bCols = NDArray<Float16>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float16,
            );
            final outEmptyCols = NDArray<Float64>.zeros([0], DType.float64);
            final resCols = lstsq<Float16, Float16, Float64>(
              aEmptyCols,
              bCols,
              out: outEmptyCols,
            );
            expect(identical(resCols.x, outEmptyCols), isTrue);
            expect(resCols.rank, equals(0));
            expect(resCols.x.shape, equals([0]));
          });
        },
      );

      test(
        'lstsq zero-fills complex128 out: buffer when m == 0 and n == 2',
        () {
          NDArray.scope(() {
            final aEmpty = NDArray<Complex128>.zeros([0, 2], DType.complex128);
            final bEmpty = NDArray<Complex128>.zeros([0, 1], DType.complex128);
            final outBuf = NDArray<Complex128>.fromList(
              [Complex(5.0, -3.0), Complex(7.0, 2.0)],
              [2, 1],
              DType.complex128,
            );
            final res = lstsq<Complex128, Complex128, Complex128>(
              aEmpty,
              bEmpty,
              out: outBuf,
            );
            expect(identical(res.x, outBuf), isTrue);
            expect(outBuf.getCell([0, 0]), equals(Complex(0.0, 0.0)));
            expect(outBuf.getCell([1, 0]), equals(Complex(0.0, 0.0)));
          });
        },
      );
    });

    group('4. take_along_axis() out-of-bounds index cleanup', () {
      test(
        'throws RangeError without leaking allocated result when out == null',
        () {
          final arr = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );
          final badIndices = NDArray.fromList([0, 5], [1, 2], DType.int32);
          try {
            expect(
              () => take_along_axis(arr, badIndices, 0),
              throwsA(isA<RangeError>()),
            );
          } finally {
            arr.dispose();
            badIndices.dispose();
          }
        },
      );
    });
  });
}
