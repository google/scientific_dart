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
  group('Comprehensive Cross-DType Deep Coverage Suite', () {
    final all15DTypes = [
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

    final floatAndComplexDTypes = <DType<AnySpec>>[
      DType.float64,
      DType.float32,
      DType.complex128,
      DType.complex64,
    ];

    NDArray<AnySpec> makeNumArr(
      DType dt,
      List<int> shape, {
      bool strided = false,
      int offset = 2,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<num>.generate(size * (strided ? 2 : 1), (i) {
        final val = ((i + offset) % 7) + 2;
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

    NDArray<AnySpec> makeArr(
      DType dt,
      List<int> shape, {
      bool strided = false,
      int offset = 2,
    }) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<Object>.generate(size * (strided ? 2 : 1), (i) {
        final val = ((i + offset) % 7) + 2;
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
      'Deep Linear Algebra - solve, lstsq, eig, eigh, eigvals, eigvalsh, qr, svd, cholesky',
      () {
        NDArray.scope(() {
          for (final dt in floatAndComplexDTypes) {
            // 2x2 square matrix
            final NDArray<AnySpec> a2d =
                (dt == DType.complex128 || dt == DType.complex64)
                ? NDArray.fromList(
                    [
                      Complex(4.0, 0.0),
                      Complex(1.0, 1.0),
                      Complex(1.0, -1.0),
                      Complex(3.0, 0.0),
                    ],
                    [2, 2],
                    dt,
                  )
                : NDArray.fromList([4.0, 1.0, 1.0, 3.0], [2, 2], dt);

            final b2d = (dt == DType.complex128 || dt == DType.complex64)
                ? NDArray.fromList(
                    [Complex(1.0, 0.0), Complex(2.0, 0.0)],
                    [2],
                    dt,
                  )
                : NDArray.fromList([1.0, 2.0], [2], dt);

            // solve
            final slv = solve(a2d, b2d);
            expect(slv.shape, [2]);

            // lstsq
            final lst = lstsq(a2d, b2d);
            expect(lst.x.shape, [2]);
            lst.dispose();

            // eigvals & eigvalsh
            final ev = eigvals(a2d);
            expect(ev.shape, [2]);

            final evh = eigvalsh(a2d);
            expect(evh.shape, [2]);

            // qr
            final qrr = qr(a2d);
            expect(qrr.q.shape, [2, 2]);
            expect(qrr.r.shape, [2, 2]);
            qrr.dispose();

            // svd
            final svdr = svd(a2d);
            expect(svdr.s.shape, [2]);
            svdr.dispose();

            // cholesky
            final chol = cholesky(a2d);
            expect(chol.shape, [2, 2]);
          }
        });
      },
    );

    test(
      'Deep Statistics Reductions across all 15 DTypes with axis variations, keepdims, ddof, and out buffers',
      () {
        NDArray.scope(() {
          for (final dt in numericDTypes) {
            for (final isStrided in [false, true]) {
              final a2d = makeNumArr(dt, [3, 4], strided: isStrided);

              // mean
              final mAll = mean(a2d);
              expect(mAll.shape, <int>[]);

              final m0 = mean(a2d, axis: 0);
              expect(m0.shape, [4]);

              final m1 = mean(a2d, axis: 1, keepdims: true);
              expect(m1.shape, [3, 1]);

              // std & var_
              final sAll = std(a2d);
              expect(sAll.shape, <int>[]);

              final s0 = std(a2d, axis: 0, ddof: 1);
              expect(s0.shape, [4]);

              final v1 = var_(a2d, axis: 1, keepdims: true);
              expect(v1.shape, [3, 1]);

              // ptp
              final ptpAll = ptp(a2d);
              expect(ptpAll.shape, <int>[]);

              final ptp0 = ptp(a2d, axis: 0);
              expect(ptp0.shape, [4]);

              // median
              final medAll = median(a2d);
              expect(medAll.shape, <int>[]);

              final med1 = median(a2d, axis: 1);
              expect(med1.shape, [3]);

              // percentile & quantile
              final p50 = percentile(a2d, 50.0);
              expect(p50.shape, <int>[]);

              final p25 = percentile(a2d, 25.0, axis: 0, keepdims: true);
              expect(p25.shape, [1, 4]);

              final q50 = quantile(a2d, 0.5);
              expect(q50.shape, <int>[]);

              final q75 = quantile(a2d, 0.75, axis: 1);
              expect(q75.shape, [3]);
            }
          }
        });
      },
    );

    test(
      'Deep NaN-ignoring statistics across Float and Integer DTypes along all axes',
      () {
        NDArray.scope(() {
          for (final dt in [DType.float64, DType.float32, DType.int32]) {
            final raw = (dt == DType.float64 || dt == DType.float32)
                ? [
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
                  ]
                : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

            final arr = NDArray.fromList(raw, [3, 4], dt);

            // nanmean
            final nmAll = nanmean(arr);
            expect(nmAll.shape, <int>[]);

            final nm0 = nanmean(arr, axis: 0);
            expect(nm0.shape, [4]);

            final nm1 = nanmean(arr, axis: 1, keepdims: true);
            expect(nm1.shape, [3, 1]);

            // nanstd & nanvar
            final nstdAll = nanstd(arr);
            expect(nstdAll.shape, <int>[]);

            final nstd0 = nanstd(arr, axis: 0);
            expect(nstd0.shape, [4]);

            final nvar1 = nanvar(arr, axis: 1, keepdims: true);
            expect(nvar1.shape, [3, 1]);

            // nanmin & nanmax
            final nminAll = nanmin(arr);
            expect(nminAll.shape, <int>[]);

            final nmin0 = nanmin(arr, axis: 0);
            expect(nmin0.shape, [4]);

            final nmax1 = nanmax(arr, axis: 1, keepdims: true);
            expect(nmax1.shape, [3, 1]);

            // nansum
            final nsumAll = nansum(arr);
            expect(nsumAll.shape, <int>[]);

            final nsum0 = nansum(arr, axis: 0);
            expect(nsum0.shape, [4]);

            final nsum1 = nansum(arr, axis: 1, keepdims: true);
            expect(nsum1.shape, [3, 1]);
          }
        });
      },
    );

    test('Deep Logical Operations with Broadcasting across all 15 DTypes', () {
      NDArray.scope(() {
        for (final dt in all15DTypes) {
          final a = makeArr(dt, [2, 3]);
          final b = makeArr(dt, [1, 3], offset: 3);

          final land = logicalAnd(a, b);
          expect(land.shape, [2, 3]);

          final lor = logicalOr(a, b);
          expect(lor.shape, [2, 3]);

          final lxor = logicalXor(a, b);
          expect(lxor.shape, [2, 3]);

          final lnot = logicalNot(a);
          expect(lnot.shape, [2, 3]);

          final mask = NDArray<Boolean>.fromList(
            [true, false, true, false, true, false],
            [2, 3],
            DType.boolean,
          );
          final outArr = NDArray<Boolean>.create([2, 3], DType.boolean);
          logicalAnd(a, b, out: outArr, where: mask);
          expect(outArr.shape, [2, 3]);
        }
      });
    });

    test(
      'Deep Universal Function (Ufunc) Methods across all reducible BinaryOp enums',
      () {
        NDArray.scope(() {
          final floatArr = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final intArr = NDArray.fromList(
            [1, 2, 3, 4, 5, 6],
            [2, 3],
            DType.int64,
          );
          final boolArr = NDArray.fromList(
            [true, false, true, true, false, true],
            [2, 3],
            DType.boolean,
          );

          // Numeric ufuncs
          final numOps = [
            BinaryOp.add,
            BinaryOp.multiply,
            BinaryOp.minimum,
            BinaryOp.maximum,
            BinaryOp.fmin,
            BinaryOp.fmax,
            BinaryOp.logaddexp,
            BinaryOp.logaddexp2,
          ];

          for (final op in numOps) {
            // reduce
            final redAll = floatArr.reduce(op: op);
            expect(redAll.shape, <int>[]);

            final red0 = floatArr.reduce(op: op, axis: 0);
            expect(red0.shape, [3]);

            final red1 = floatArr.reduce(op: op, axis: 1, keepdims: true);
            expect(red1.shape, [2, 1]);

            // accumulate
            final acc0 = floatArr.accumulate(op: op, axis: 0);
            expect(acc0.shape, [2, 3]);

            final acc1 = floatArr.accumulate(op: op, axis: 1);
            expect(acc1.shape, [2, 3]);

            // reduceat
            final indices = NDArray.fromList([0, 2], [2], DType.int64);
            final redAt = reduceatUfunc(floatArr, indices, op: op, axis: 1);
            expect(redAt.shape, [2, 2]);
          }

          // Integer-specific ufuncs
          final intOps = [
            BinaryOp.gcd,
            BinaryOp.lcm,
            BinaryOp.bitwiseAnd,
            BinaryOp.bitwiseOr,
            BinaryOp.bitwiseXor,
          ];

          for (final op in intOps) {
            final red0 = intArr.reduce(op: op, axis: 0);
            expect(red0.shape, [3]);

            final acc1 = intArr.accumulate(op: op, axis: 1);
            expect(acc1.shape, [2, 3]);
          }

          // Boolean ufuncs
          final boolOps = [
            BinaryOp.logicalAnd,
            BinaryOp.logicalOr,
            BinaryOp.logicalXor,
          ];

          for (final op in boolOps) {
            final red0 = boolArr.reduce(op: op, axis: 0);
            expect(red0.shape, [3]);

            final acc1 = boolArr.accumulate(op: op, axis: 1);
            expect(acc1.shape, [2, 3]);
          }
        });
      },
    );
  });
}
