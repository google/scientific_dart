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
  group('Linear Algebra - QR Decomposition', () {
    test(
      'QR decomposition 2D float64 & float32 (M >= N and M < N)',
      () => NDArray.scope(() {
        final aTall = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final qrTall = qr(aTall);
        expect(qrTall.q.shape, [3, 2]);
        expect(qrTall.r.shape, [2, 2]);

        final reconTall = matmul(qrTall.q, qrTall.r);
        for (var i = 0; i < aTall.size; i++) {
          expect(
            reconTall.getCellFlat(i),
            closeTo(aTall.getCellFlat(i), 1e-10),
          );
        }

        final aWide = NDArray<Float32>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float32,
        );

        final qrWide = qr(aWide);
        expect(qrWide.q.shape, [2, 2]);
        expect(qrWide.r.shape, [2, 3]);

        final reconWide = matmul(qrWide.q, qrWide.r);
        for (var i = 0; i < aWide.size; i++) {
          expect(reconWide.getCellFlat(i), closeTo(aWide.getCellFlat(i), 1e-4));
        }
      }),
    );

    test(
      'QR batched 3D stacks and out parameter',
      () => NDArray.scope(() {
        final a3d = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float64,
        );

        final outQ = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
        final outR = NDArray<Float64>.zeros([2, 2, 2], DType.float64);

        final qrRes = qr(a3d, out: (q: outQ, r: outR));
        expect(identical(qrRes.q, outQ), true);
        expect(identical(qrRes.r, outR), true);

        final a1d = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => qr(a1d), throwsA(isA<ArgumentError>()));
      }),
    );
  });

  group('Linear Algebra - SVD Decomposition', () {
    test(
      'SVD decomposition across float64 and complex128',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final svdRes = svd(a);
        expect(svdRes.u.shape, [3, 3]);
        expect(svdRes.s.shape, [2]);
        expect(svdRes.vh.shape, [2, 2]);

        final cMat = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex128,
        );
        final svdC = svd(cMat);
        expect(svdC.u.shape, [2, 2]);
        expect(svdC.s.shape, [2]);
        expect(svdC.vh.shape, [2, 2]);
      }),
    );

    test(
      'SVD batched 3D stacks and out parameter',
      () => NDArray.scope(() {
        final a3d = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float64,
        );

        final outU = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
        final outS = NDArray<Float64>.zeros([2, 2], DType.float64);
        final outVh = NDArray<Float64>.zeros([2, 2, 2], DType.float64);

        final res = svd(a3d, out: (u: outU, s: outS, vh: outVh));
        expect(identical(res.u, outU), true);
        expect(identical(res.s, outS), true);
        expect(identical(res.vh, outVh), true);

        final a1d = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => svd(a1d), throwsA(isA<ArgumentError>()));
      }),
    );
  });

  group('Linear Algebra - Cholesky Decomposition', () {
    test(
      'Cholesky on positive definite matrices',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
          [3, 3],
          DType.float64,
        );

        final l = cholesky(a);
        expect(l.shape, [3, 3]);
        expect(l.getCell([0, 0]), 2.0);
        expect(l.getCell([0, 1]), 0.0);
        expect(l.getCell([0, 2]), 0.0);

        final recon = matmul(l, l.transpose());
        for (var i = 0; i < a.size; i++) {
          expect(recon.getCellFlat(i), closeTo(a.getCellFlat(i), 1e-10));
        }
      }),
    );

    test(
      'Cholesky batched 3D, out parameter, and non-positive-definite exceptions',
      () => NDArray.scope(() {
        final a3d = NDArray<Float64>.fromList(
          [4.0, 2.0, 2.0, 5.0, 9.0, 3.0, 3.0, 10.0],
          [2, 2, 2],
          DType.float64,
        );

        final outL = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
        final res = cholesky(a3d, out: outL);
        expect(identical(res, outL), true);

        final nonPosDef = NDArray<Float64>.fromList(
          [-1.0, 0.0, 0.0, -1.0],
          [2, 2],
          DType.float64,
        );
        expect(
          () => cholesky(nonPosDef),
          throwsA(isA<NonPositiveDefiniteException>()),
        );

        final nonSquare = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        expect(
          () => cholesky(nonSquare),
          throwsA(anyOf(isA<ArgumentError>(), isA<LinAlgException>())),
        );
      }),
    );
  });

  group('Linear Algebra - Pseudo-Inverse (Pinv)', () {
    test(
      'Pinv 2D rectangular and square matrices',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final aPinv = pinv(a);
        expect(aPinv.shape, [2, 3]);

        final apa = matmul(matmul(a, aPinv), a);
        for (var i = 0; i < a.size; i++) {
          expect(apa.getCellFlat(i), closeTo(a.getCellFlat(i), 1e-10));
        }

        final outPinv = NDArray<Float64>.zeros([2, 3], DType.float64);
        pinv(a, out: outPinv);
        expect(outPinv.shape, [2, 3]);
      }),
    );

    test(
      'Pinv validation errors',
      () => NDArray.scope(() {
        final a1d = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => pinv(a1d), throwsA(isA<ArgumentError>()));
      }),
    );
  });

  group('Linear Algebra - Matrix Power', () {
    test(
      'matrix_power for n = 0, n > 0, and n < 0',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final a0 = matrix_power(a, 0);
        expect(a0.toList(), [1.0, 0.0, 0.0, 1.0]);

        final a1 = matrix_power(a, 1);
        expect(a1.toList(), [1.0, 2.0, 3.0, 4.0]);

        final a2 = matrix_power(a, 2);
        expect(a2.toList(), [7.0, 10.0, 15.0, 22.0]);

        final a3 = matrix_power(a, 3);
        final directA3 = matmul(matmul(a, a), a);
        for (var i = 0; i < a3.size; i++) {
          expect(a3.getCellFlat(i), closeTo(directA3.getCellFlat(i), 1e-10));
        }

        final aNeg1 = matrix_power(a, -1);
        final invA = inv(a);
        for (var i = 0; i < aNeg1.size; i++) {
          expect(aNeg1.getCellFlat(i), closeTo(invA.getCellFlat(i), 1e-10));
        }
      }),
    );

    test(
      'matrix_power non-square error validation',
      () => NDArray.scope(() {
        final nonSquare = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        expect(
          () => matrix_power(nonSquare, 2),
          throwsA(anyOf(isA<ArgumentError>(), isA<LinAlgException>())),
        );

        final non2D = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float64,
        );
        expect(
          () => matrix_power(non2D, 2),
          throwsA(anyOf(isA<ArgumentError>(), isA<LinAlgException>())),
        );
      }),
    );
  });

  group('Linear Algebra - Matrix and Vector Norms', () {
    test(
      'Vector norms with various ord parameters',
      () => NDArray.scope(() {
        final v = NDArray<Float64>.fromList([3.0, -4.0], [2], DType.float64);

        expect(norm(v).scalar, 5.0);
        expect(norm(v, ord: 1).scalar, 7.0);
        expect(norm(v, ord: 2).scalar, 5.0);
        expect(norm(v, ord: double.infinity).scalar, 4.0);
        expect(norm(v, ord: -double.infinity).scalar, 3.0);
        expect(norm(v, ord: 0).scalar, 2.0);
      }),
    );

    test(
      'Matrix norms with NormKind and axis/keepdims',
      () => NDArray.scope(() {
        final m = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        expect(
          norm(m, ord: NormKind.frobenius).scalar,
          closeTo(math.sqrt(1 + 4 + 9 + 16), 1e-10),
        );
        expect(
          norm(m, ord: NormKind.nuclear).scalar,
          closeTo(norm(m, ord: NormKind.nuclear).scalar, 1e-10),
        );
        expect(norm(m, ord: double.infinity).scalar, 7.0);
        expect(norm(m, ord: -double.infinity).scalar, 3.0);
        expect(norm(m, ord: 1).scalar, 6.0);
        expect(norm(m, ord: -1).scalar, 4.0);

        final normKeep = norm(m, ord: NormKind.frobenius, keepdims: true);
        expect(normKeep.shape, [1, 1]);
        expect(normKeep.getCell([0, 0]), closeTo(math.sqrt(30), 1e-10));

        final axisNorm = norm(m, axis: 0);
        expect(axisNorm.shape, [2]);
        expect(axisNorm.getCellFlat(0), closeTo(math.sqrt(10), 1e-10));
        expect(axisNorm.getCellFlat(1), closeTo(math.sqrt(20), 1e-10));
      }),
    );
  });

  group('Linear Algebra - Determinant and Slogdet', () {
    test(
      'det and slogdet on 2D and 3D batched stacks',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final d = det(a);
        expect(d.scalar, closeTo(-2.0, 1e-10));

        final sld = slogdet(a);
        expect(sld.sign.scalar, -1.0);
        expect(sld.logabsdet.scalar, closeTo(math.log(2.0), 1e-10));

        final a3d = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0],
          [2, 2, 2],
          DType.float64,
        );

        final d3d = det(a3d);
        expect(d3d.shape, [2]);
        expect(d3d.getCellFlat(0), closeTo(-2.0, 1e-10));
        expect(d3d.getCellFlat(1), closeTo(4.0, 1e-10));

        final singular = NDArray<Float64>.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(det(singular).scalar, closeTo(0.0, 1e-10));
      }),
    );
  });

  group('Linear Algebra - Eigenvalues and Eigenvectors', () {
    test(
      'eig and eigvals on real matrices',
      () => NDArray.scope(() {
        final diagMat = NDArray<Float64>.fromList(
          [2.0, 0.0, 0.0, 5.0],
          [2, 2],
          DType.float64,
        );

        final vals = eigvals(diagMat);
        expect(vals.shape, [2]);

        final resEig = eig(diagMat);
        expect(resEig.eigenvalues.shape, [2]);
        expect(resEig.eigenvectors.shape, [2, 2]);
      }),
    );

    test(
      'eigh and eigvalsh on symmetric matrices',
      () => NDArray.scope(() {
        final symm = NDArray<Float64>.fromList(
          [2.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );

        final valsH = eigvalsh(symm);
        expect(valsH.shape, [2]);
        expect(valsH.getCellFlat(0), closeTo(1.0, 1e-10));
        expect(valsH.getCellFlat(1), closeTo(3.0, 1e-10));

        final resEigh = eigh(symm, uplo: MatrixTriangle.lower);
        expect(resEigh.eigenvalues.shape, [2]);
        expect(resEigh.eigenvectors.shape, [2, 2]);

        final lambdaMat = NDArray<Float64>.fromList(
          [
            resEigh.eigenvalues.getCellFlat(0),
            0.0,
            0.0,
            resEigh.eigenvalues.getCellFlat(1),
          ],
          [2, 2],
          DType.float64,
        );

        final recon = matmul(
          matmul(resEigh.eigenvectors, lambdaMat),
          resEigh.eigenvectors.transpose(),
        );
        for (var i = 0; i < symm.size; i++) {
          expect(recon.getCellFlat(i), closeTo(symm.getCellFlat(i), 1e-10));
        }
      }),
    );
  });

  group('Linear Algebra - Solve and Lstsq', () {
    test(
      'solve linear systems Ax = b for 1D and 2D right-hand sides',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [3.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );

        final b1d = NDArray<Float64>.fromList([9.0, 8.0], [2], DType.float64);
        final x1d = solve(a, b1d);
        expect(x1d.shape, [2]);
        expect(x1d.getCellFlat(0), closeTo(2.0, 1e-10));
        expect(x1d.getCellFlat(1), closeTo(3.0, 1e-10));

        final b2d = NDArray<Float64>.fromList(
          [9.0, 6.0, 8.0, 7.0],
          [2, 2],
          DType.float64,
        );
        final x2d = solve(a, b2d);
        expect(x2d.shape, [2, 2]);
        final recon = matmul(a, x2d);
        for (var i = 0; i < b2d.size; i++) {
          expect(recon.getCellFlat(i), closeTo(b2d.getCellFlat(i), 1e-10));
        }

        final singular = NDArray<Float64>.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(
          () => solve(singular, b1d),
          throwsA(isA<SingularMatrixException>()),
        );
      }),
    );

    test(
      'lstsq least squares solution',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0],
          [4, 2],
          DType.float64,
        );

        final b = NDArray<Float64>.fromList(
          [-1.0, 0.2, 0.9, 2.1],
          [4],
          DType.float64,
        );

        final res = lstsq(a, b);
        expect(res.x.shape, [2]);
        expect(res.rank, 2);
        expect(res.s.shape, [2]);
      }),
    );
  });

  group('Linear Algebra - Matmul, Multi_dot, Outer, Cross', () {
    test(
      'matmul across dimensions and complex',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [5.0, 6.0, 7.0, 8.0],
          [2, 2],
          DType.float64,
        );

        final c = matmul(a, b);
        expect(c.toList(), [19.0, 22.0, 43.0, 50.0]);

        final v1 = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        final v2 = NDArray<Float64>.fromList([3.0, 4.0], [2], DType.float64);
        expect(matmul(v1, v2).scalar, 11.0);

        final vMat = matmul(v1, a);
        expect(vMat.shape, [2]);
        expect(vMat.toList(), [7.0, 10.0]);

        final matV = matmul(a, v1);
        expect(matV.shape, [2]);
        expect(matV.toList(), [5.0, 11.0]);

        final out = NDArray<Float64>.zeros([2, 2], DType.float64);
        matmul(a, b, out: out);
        expect(out.toList(), [19.0, 22.0, 43.0, 50.0]);
      }),
    );

    test(
      'multi_dot chain multiplication',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final c = NDArray<Float64>.fromList(
          [2.0, 0.0, 0.0, 2.0],
          [2, 2],
          DType.float64,
        );

        final res = multi_dot([a, b, c]);
        expect(res.toList(), [2.0, 4.0, 6.0, 8.0]);

        expect(() => multi_dot([a]), throwsA(isA<ArgumentError>()));
      }),
    );

    test(
      'outer product across DTypes and shapes',
      () => NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList([4.0, 5.0], [2], DType.float64);

        final outProd = outer(a, b);
        expect(outProd.shape, [3, 2]);
        expect(outProd.toList(), [4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);

        final a2d = NDArray<Int32>.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final b2d = NDArray<Int32>.fromList([10, 20], [2], DType.int32);
        final outProdInt = outer(a2d, b2d);
        expect(outProdInt.shape, [4, 2]);
      }),
    );

    test(
      'cross product 3D and 2D vectors',
      () => NDArray.scope(() {
        final u = NDArray<Float64>.fromList(
          [1.0, 0.0, 0.0],
          [3],
          DType.float64,
        );
        final v = NDArray<Float64>.fromList(
          [0.0, 1.0, 0.0],
          [3],
          DType.float64,
        );

        final w = cross(u, v);
        expect(w.shape, [3]);
        expect(w.toList(), [0.0, 0.0, 1.0]);

        final u2 = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        final v2 = NDArray<Float64>.fromList([3.0, 4.0], [2], DType.float64);
        final w2 = cross(u2, v2);
        expect(w2.scalar, -2.0);

        final u3d = NDArray<Float64>.fromList(
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [2, 3],
          DType.float64,
        );
        final v3d = NDArray<Float64>.fromList(
          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
          [2, 3],
          DType.float64,
        );

        final w3d = cross(u3d, v3d);
        expect(w3d.shape, [2, 3]);
        expect(w3d.toList(), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
      }),
    );
  });

  group('Financial Operations Tests', () {
    test(
      'fv future value with PaymentDue begin and end',
      () => NDArray.scope(() {
        final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
        final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
        final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
        final pv = NDArray<Float64>.scalar(-1000.0, dtype: DType.float64);

        final fvEnd = fv(rate, nper, pmt, pv, when: PaymentDue.end);
        expect(fvEnd.scalar, closeTo(2886.68, 0.01));

        final fvBegin = fv(rate, nper, pmt, pv, when: PaymentDue.begin);
        expect(fvBegin.scalar, closeTo(2949.57, 0.01));

        final zeroRate = NDArray<Float64>.scalar(0.0, dtype: DType.float64);
        final fvZeroRate = fv(zeroRate, nper, pmt, pv);
        expect(fvZeroRate.scalar, 2000.0);
      }),
    );

    test(
      'pv present value with PaymentDue begin and end',
      () => NDArray.scope(() {
        final rate = NDArray<Float64>.scalar(0.05, dtype: DType.float64);
        final nper = NDArray<Float64>.scalar(10.0, dtype: DType.float64);
        final pmt = NDArray<Float64>.scalar(-100.0, dtype: DType.float64);
        final futureVal = NDArray<Float64>.scalar(
          2886.68,
          dtype: DType.float64,
        );

        final pvEnd = pv(rate, nper, pmt, futureVal, when: PaymentDue.end);
        expect(pvEnd.scalar, closeTo(-1000.0, 0.01));

        final zeroRate = NDArray<Float64>.scalar(0.0, dtype: DType.float64);
        final pvZeroRate = pv(zeroRate, nper, pmt, futureVal);
        expect(pvZeroRate.scalar, closeTo(-(2886.68 - 1000.0), 0.01));
      }),
    );

    test(
      'npv net present value 1D and 2D batched',
      () => NDArray.scope(() {
        final rate = NDArray<Float64>.scalar(0.08, dtype: DType.float64);
        final values = NDArray<Float64>.fromList(
          [-40000.0, 5000.0, 8000.0, 12000.0, 30000.0],
          [5],
          DType.float64,
        );

        final resNpv = npv(rate, values);
        expect(resNpv.scalar, closeTo(3065.22, 0.01));

        final batchedValues = NDArray<Float64>.fromList(
          [
            -40000.0,
            5000.0,
            8000.0,
            12000.0,
            30000.0,
            -40000.0,
            5000.0,
            8000.0,
            12000.0,
            30000.0,
          ],
          [2, 5],
          DType.float64,
        );

        final batchedNpv = npv(rate, batchedValues);
        expect(batchedNpv.shape, [2]);
        expect(batchedNpv.getCellFlat(0), closeTo(3065.22, 0.01));
        expect(batchedNpv.getCellFlat(1), closeTo(3065.22, 0.01));
      }),
    );

    test(
      'irr internal rate of return and exception handling',
      () => NDArray.scope(() {
        final cashFlows = NDArray<Float64>.fromList(
          [-100.0, 39.0, 59.0, 55.0, 20.0],
          [5],
          DType.float64,
        );

        final rate = irr(cashFlows);
        expect(rate.scalar, closeTo(0.2809, 0.001));

        final allPositive = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0],
          [3],
          DType.float64,
        );
        expect(
          () => irr(allPositive, raiseExceptions: true),
          throwsA(isA<NoRealSolutionException>()),
        );

        final irrNoRaise = irr(allPositive, raiseExceptions: false);
        expect(irrNoRaise.scalar.isNaN, true);
      }),
    );
  });
}
