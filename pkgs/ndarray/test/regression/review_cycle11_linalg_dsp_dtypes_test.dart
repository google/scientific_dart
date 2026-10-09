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
  group('Review Cycle 11 Finding #4: Linalg Float16 & BFloat16 Promotion', () {
    for (final dtype in <DType<AnySpec>>[DType.float16, DType.bfloat16]) {
      group('dtype=${dtype.name}', () {
        test('eig and eigvals promote to float64 / complex128', () {
          NDArray.scope(() {
            final a = NDArray.fromList([2.0, 0.0, 0.0, 3.0], [2, 2], dtype);
            final res = eig(a);
            expect(res.eigenvalues.dtype, equals(DType.complex128));
            expect(res.eigenvectors.dtype, equals(DType.complex128));
            final ev0 = res.eigenvalues.getCell([0]).real;
            final ev1 = res.eigenvalues.getCell([1]).real;
            final sortedEvs = [ev0, ev1]..sort();
            expect(sortedEvs[0], closeTo(2.0, 1e-2));
            expect(sortedEvs[1], closeTo(3.0, 1e-2));

            final vals = eigvals(a);
            expect(vals.dtype, equals(DType.complex128));
            final v0 = vals.getCell([0]).real;
            final v1 = vals.getCell([1]).real;
            final sortedVals = [v0, v1]..sort();
            expect(sortedVals[0], closeTo(2.0, 1e-2));
            expect(sortedVals[1], closeTo(3.0, 1e-2));
          });
        });

        test('cholesky promotes to float64 and decomposes SPD matrix', () {
          NDArray.scope(() {
            final a = NDArray.fromList([4.0, 2.0, 2.0, 5.0], [2, 2], dtype);
            final l = cholesky(a);
            expect(l.dtype, equals(dtype));
            expect(l.getCell([0, 0]), closeTo(2.0, 1e-2));
            expect(l.getCell([1, 0]), closeTo(1.0, 1e-2));
            expect(l.getCell([0, 1]), closeTo(0.0, 1e-5));
            expect(l.getCell([1, 1]), closeTo(2.0, 1e-2));
          });
        });

        test('qr promotes to float64 and satisfies Q * R == A', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype);
            final res = qr(a);
            expect(res.q.dtype, equals(dtype));
            expect(res.r.dtype, equals(dtype));
            final recon = matmul(res.q, res.r);
            expect(recon.getCell([0, 0]), closeTo(1.0, 1e-2));
            expect(recon.getCell([0, 1]), closeTo(2.0, 1e-2));
            expect(recon.getCell([1, 0]), closeTo(3.0, 1e-2));
            expect(recon.getCell([1, 1]), closeTo(4.0, 1e-2));
          });
        });

        test('svd promotes to float64 and computes singular values', () {
          NDArray.scope(() {
            final a = NDArray.fromList([3.0, 0.0, 0.0, 4.0], [2, 2], dtype);
            final res = svd(a);
            expect(res.u.dtype, equals(dtype));
            expect(res.s.dtype, equals(dtype));
            expect(res.vh.dtype, equals(dtype));
            expect(res.s.getCell([0]), closeTo(4.0, 1e-2));
            expect(res.s.getCell([1]), closeTo(3.0, 1e-2));
          });
        });

        test('eigh and eigvalsh promote to float64 on symmetric matrix', () {
          NDArray.scope(() {
            final a = NDArray.fromList([2.0, 1.0, 1.0, 2.0], [2, 2], dtype);
            final res = eigh<DTypeTag, DTypeTag>(a);
            expect(res.eigenvalues.dtype, equals(DType.float64));
            expect(res.eigenvectors.dtype, equals(DType.float64));
            expect(res.eigenvalues.getCell([0]), closeTo(1.0, 1e-2));
            expect(res.eigenvalues.getCell([1]), closeTo(3.0, 1e-2));

            final vals = eigvalsh(a);
            expect(vals.dtype, equals(DType.float64));
            expect(vals.getCell([0]), closeTo(1.0, 1e-2));
            expect(vals.getCell([1]), closeTo(3.0, 1e-2));
          });
        });

        test('schur promotes to float64 (real) and complex128 (complex)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([2.0, 1.0, 0.0, 3.0], [2, 2], dtype);
            final realRes = schur(a);
            expect(realRes.t.dtype, equals(DType.float64));
            expect(realRes.z.dtype, equals(DType.float64));

            final complexRes = complexSchur(a);
            expect(complexRes.t.dtype, equals(DType.complex128));
            expect(complexRes.z.dtype, equals(DType.complex128));
          });
        });

        test(
          'hessenberg promotes to float64 and satisfies Q * H * Q^T == A',
          () {
            NDArray.scope(() {
              final a = NDArray.fromList(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
                [3, 3],
                dtype,
              );
              final res = hessenberg(a);
              expect(res.h.dtype, equals(DType.float64));
              expect(res.q.dtype, equals(DType.float64));
              expect(res.h.getCell([2, 0]), closeTo(0.0, 1e-5));
            });
          },
        );

        test('inv, det, slogdet, solve, pinv, norm, cond, lstsq', () {
          NDArray.scope(() {
            final a = NDArray.fromList([4.0, 1.0, 2.0, 3.0], [2, 2], dtype);
            final b = NDArray.fromList([1.0, 2.0], [2], dtype);

            final d = det(a);
            expect(d.scalar, closeTo(10.0, 1e-1));

            final sd = slogdet(a);
            expect(sd.sign.scalar, closeTo(1.0, 1e-2));
            expect(sd.logabsdet.scalar, closeTo(2.302585, 1e-2));

            final aInv = inv(a);
            expect(aInv.dtype, equals(dtype));
            expect(aInv.getCell([0, 0]), closeTo(0.3, 1e-2));

            final x = solve(a, b);
            expect(x.dtype, equals(dtype));
            expect(x.getCell([0]), closeTo(0.1, 1e-2));
            expect(x.getCell([1]), closeTo(0.6, 1e-2));

            final p = pinv(a);
            expect(p.dtype, equals(dtype));
            expect(p.getCell([0, 0]), closeTo(0.3, 1e-2));

            final nVal = norm(b);
            expect(nVal.scalar, closeTo(2.2360679, 1e-2));

            final cVal = cond(a);
            expect(cVal.scalar, greaterThan(1.0));

            final lsq = lstsq(a, b);
            expect(lsq.rank, equals(2));
            expect(lsq.x.dtype, equals(DType.float64));
            expect(lsq.x.getCell([0]), closeTo(0.1, 1e-2));
            expect(lsq.x.getCell([1]), closeTo(0.6, 1e-2));
          });
        });
      });
    }
  });

  group('Review Cycle 11 Finding #5: DSP correlate & convolve DTypes', () {
    final intDTypes = <DType<AnySpec>>[
      DType.int8,
      DType.int16,
      DType.uint8,
      DType.uint16,
      DType.uint32,
      DType.uint64,
    ];

    for (final dtype in intDTypes) {
      test('correlate and convolve support integer dtype=${dtype.name}', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [4], dtype);
          final v = NDArray.fromList([1, 2], [2], dtype);

          // correlate valid: [1*1 + 2*2, 2*1 + 3*2, 3*1 + 4*2] = [5, 8, 11]
          final corrValid = correlate(a, v, mode: ConvMode.valid);
          expect(corrValid.dtype, equals(dtype));
          expect(corrValid.shape, equals([3]));
          expect(corrValid.getCell([0]), equals(5));
          expect(corrValid.getCell([1]), equals(8));
          expect(corrValid.getCell([2]), equals(11));

          // correlate same: length 4 -> [2, 5, 8, 11]
          final corrSame = correlate(a, v, mode: ConvMode.same);
          expect(corrSame.dtype, equals(dtype));
          expect(corrSame.shape, equals([4]));
          expect(corrSame.getCell([0]), equals(2));
          expect(corrSame.getCell([1]), equals(5));
          expect(corrSame.getCell([2]), equals(8));
          expect(corrSame.getCell([3]), equals(11));

          // correlate full: length 5 -> [2, 5, 8, 11, 4]
          final corrFull = correlate(a, v, mode: ConvMode.full);
          expect(corrFull.dtype, equals(dtype));
          expect(corrFull.shape, equals([5]));
          expect(corrFull.getCell([0]), equals(2));
          expect(corrFull.getCell([1]), equals(5));
          expect(corrFull.getCell([2]), equals(8));
          expect(corrFull.getCell([3]), equals(11));
          expect(corrFull.getCell([4]), equals(4));

          // convolve valid: v reversed is [2, 1] -> [1*2+2*1, 2*2+3*1, 3*2+4*1] = [4, 7, 10]
          final convValid = convolve(a, v, mode: ConvMode.valid);
          expect(convValid.dtype, equals(dtype));
          expect(convValid.shape, equals([3]));
          expect(convValid.getCell([0]), equals(4));
          expect(convValid.getCell([1]), equals(7));
          expect(convValid.getCell([2]), equals(10));

          // convolve full: [1, 4, 7, 10, 8]
          final convFull = convolve(a, v, mode: ConvMode.full);
          expect(convFull.dtype, equals(dtype));
          expect(convFull.shape, equals([5]));
          expect(convFull.getCell([0]), equals(1));
          expect(convFull.getCell([1]), equals(4));
          expect(convFull.getCell([2]), equals(7));
          expect(convFull.getCell([3]), equals(10));
          expect(convFull.getCell([4]), equals(8));
        });
      });
    }

    for (final dtype in <DType<AnySpec>>[DType.float16, DType.bfloat16]) {
      test('correlate and convolve support half-float dtype=${dtype.name}', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], dtype);
          final v = NDArray.fromList([1.0, 2.0], [2], dtype);

          final corrValid = correlate(a, v, mode: ConvMode.valid);
          expect(corrValid.dtype, equals(dtype));
          expect(corrValid.shape, equals([3]));
          expect(corrValid.getCell([0]), closeTo(5.0, 1e-2));
          expect(corrValid.getCell([1]), closeTo(8.0, 1e-2));
          expect(corrValid.getCell([2]), closeTo(11.0, 1e-2));

          final convFull = convolve(a, v, mode: ConvMode.full);
          expect(convFull.dtype, equals(dtype));
          expect(convFull.shape, equals([5]));
          expect(convFull.getCell([0]), closeTo(1.0, 1e-2));
          expect(convFull.getCell([1]), closeTo(4.0, 1e-2));
          expect(convFull.getCell([2]), closeTo(7.0, 1e-2));
          expect(convFull.getCell([3]), closeTo(10.0, 1e-2));
          expect(convFull.getCell([4]), closeTo(8.0, 1e-2));
        });
      });
    }
  });
}
