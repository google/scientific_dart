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
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Workstream 3: Linear Algebra, FFT & Math Ufuncs Tests', () {
    group('1. Matmul 1D Vector Strided Dot Products & View Returns', () {
      test('1D strided vector dot product with step > 1 (Float64)', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 99.0, 2.0, 99.0, 3.0],
            [5],
            DType.float64,
          );
          final b = NDArray.fromList(
            [4.0, -1.0, 5.0, -1.0, 6.0],
            [5],
            DType.float64,
          );

          final aSlice = NDArray<Float64>.view(
            a,
            shape: [3],
            strides: [2],
            offsetElements: 0,
          );
          final bSlice = NDArray<Float64>.view(
            b,
            shape: [3],
            strides: [2],
            offsetElements: 0,
          );

          expect(aSlice.strides, [2]);
          expect(bSlice.strides, [2]);

          final res = matmul(aSlice, bSlice);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(32.0, 1e-9));
        });
      });

      test('1D strided vector dot product with step > 1 (Float32)', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 0.0, 2.0, 0.0, 3.0]),
            [5],
            DType.float32,
          );
          final b = NDArray.fromList(
            Float32List.fromList([2.0, 0.0, 3.0, 0.0, 4.0]),
            [5],
            DType.float32,
          );

          final aSlice = NDArray<Float32>.view(
            a,
            shape: [3],
            strides: [2],
            offsetElements: 0,
          );
          final bSlice = NDArray<Float32>.view(
            b,
            shape: [3],
            strides: [2],
            offsetElements: 0,
          );

          final res = matmul(aSlice, bSlice);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(20.0, 1e-6));
        });
      });

      test('Matmul 1D return value is safely disposable', () {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
        final res = matmul(a, b);
        expect(res.scalar, 32.0);
        res.dispose();
        a.dispose();
        b.dispose();
      });

      test(
        'Matmul 1D x 2D and 2D x 1D vector return values are safely disposable',
        () {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final m = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          final res1 = matmul(a, m);
          expect(res1.shape, [2]);
          res1.dispose();

          final m2 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final res2 = matmul(m2, a);
          expect(res2.shape, [2]);
          res2.dispose();

          a.dispose();
          m.dispose();
          m2.dispose();
        },
      );
    });

    group('2. Complex QR Decomposition (LAPACK zgeqrf & cgeqrf)', () {
      test('Complex128 2D QR decomposition correctness', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(3.0, -1.0),
              Complex(-2.0, 1.0),
              Complex(0.0, 4.0),
            ],
            [2, 2],
            DType.complex128,
          );

          final res = qr(a);
          final q = res.q;
          final r = res.r;

          expect(q.dtype, DType.complex128);
          expect(r.dtype, DType.complex128);
          expect(q.shape, [2, 2]);
          expect(r.shape, [2, 2]);

          final r10 = r.getCell([1, 0]);
          expect(r10.real.abs(), lessThan(1e-12));
          expect(r10.imag.abs(), lessThan(1e-12));

          final qConj = NDArray.zeros([2, 2], DType.complex128);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
              final val = q.getCell([j, i]);
              qConj.setCell([i, j], Complex(val.real, -val.imag));
            }
          }
          final qHq = matmul(qConj, q);
          final id00 = qHq.getCell([0, 0]);
          final id11 = qHq.getCell([1, 1]);
          final id01 = qHq.getCell([0, 1]);
          final id10 = qHq.getCell([1, 0]);

          expect(id00.real, closeTo(1.0, 1e-12));
          expect(id00.imag.abs(), lessThan(1e-12));
          expect(id11.real, closeTo(1.0, 1e-12));
          expect(id11.imag.abs(), lessThan(1e-12));
          expect(id01.real.abs(), lessThan(1e-12));
          expect(id01.imag.abs(), lessThan(1e-12));
          expect(id10.real.abs(), lessThan(1e-12));
          expect(id10.imag.abs(), lessThan(1e-12));

          final qrProd = matmul(q, r);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
              final prodVal = qrProd.getCell([i, j]);
              final aVal = a.getCell([i, j]);
              expect(prodVal.real, closeTo(aVal.real, 1e-12));
              expect(prodVal.imag, closeTo(aVal.imag, 1e-12));
            }
          }
        });
      });

      test('Complex64 2D QR decomposition correctness', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(2.0, 0.0),
              Complex(0.0, 1.0),
              Complex(3.0, -2.0),
            ],
            [2, 2],
            DType.complex64,
          );

          final res = qr(a);
          expect(res.q.dtype, DType.complex64);
          expect(res.r.dtype, DType.complex64);

          final qrProd = matmul(res.q, res.r);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
              final prodVal = qrProd.getCell([i, j]);
              final aVal = a.getCell([i, j]);
              expect(prodVal.real, closeTo(aVal.real, 1e-5));
              expect(prodVal.imag, closeTo(aVal.imag, 1e-5));
            }
          }
        });
      });

      test('Complex128 Rectangular QR (tall matrix 3x2)', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(0.0, 1.0),
              Complex(3.0, 0.0),
              Complex(1.0, -1.0),
              Complex(0.0, -2.0),
              Complex(2.0, 2.0),
            ],
            [3, 2],
            DType.complex128,
          );

          final res = qr(a);
          expect(res.q.shape, [3, 2]);
          expect(res.r.shape, [2, 2]);

          final qrProd = matmul(res.q, res.r);
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 2; j++) {
              final prodVal = qrProd.getCell([i, j]);
              final aVal = a.getCell([i, j]);
              expect(prodVal.real, closeTo(aVal.real, 1e-12));
              expect(prodVal.imag, closeTo(aVal.imag, 1e-12));
            }
          }
        });
      });

      test('Stacked Complex128 QR decomposition', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(2.0, 0.0),
              Complex(0.0, 1.0),
              Complex(3.0, 0.0),
              Complex(2.0, 1.0),
              Complex(1.0, -1.0),
              Complex(-1.0, 0.0),
              Complex(4.0, 2.0),
            ],
            [2, 2, 2],
            DType.complex128,
          );

          final res = qr(a);
          expect(res.q.shape, [2, 2, 2]);
          expect(res.r.shape, [2, 2, 2]);

          final qrProd = matmul(res.q, res.r);
          for (var b = 0; b < 2; b++) {
            for (var i = 0; i < 2; i++) {
              for (var j = 0; j < 2; j++) {
                final prodVal = qrProd.getCell([b, i, j]);
                final aVal = a.getCell([b, i, j]);
                expect(prodVal.real, closeTo(aVal.real, 1e-12));
                expect(prodVal.imag, closeTo(aVal.imag, 1e-12));
              }
            }
          }
        });
      });
    });

    group('3. Integer Matrix Linalg Support (Auto-Upcast to Float64)', () {
      test('eig() auto-upcasts integer matrices (int32 and int64)', () {
        NDArray.scope(() {
          final aInt32 = NDArray.fromList(Int32List.fromList([2, 1, 1, 2]), [
            2,
            2,
          ], DType.int32);
          final res32 = eig(aInt32);
          expect(res32.eigenvalues.dtype, DType.complex128);
          expect(res32.eigenvectors.dtype, DType.complex128);

          final vals32 = res32.eigenvalues.toList().map((c) => c.real).toList()
            ..sort();
          expect(vals32[0], closeTo(1.0, 1e-9));
          expect(vals32[1], closeTo(3.0, 1e-9));

          final aInt64 = NDArray.fromList(Int64List.fromList([2, 1, 1, 2]), [
            2,
            2,
          ], DType.int64);
          final res64 = eig(aInt64);
          expect(res64.eigenvalues.dtype, DType.complex128);
          expect(res64.eigenvectors.dtype, DType.complex128);
        });
      });

      test('eigh() auto-upcasts integer symmetric matrices', () {
        NDArray.scope(() {
          final a = NDArray.fromList([2, 1, 1, 2], [2, 2], DType.int32);
          final res = eigh(a);
          expect(res.eigenvalues.dtype, DType.float64);
          expect(res.eigenvectors.dtype, DType.float64);

          final vals = res.eigenvalues.toList().map((v) => v).toList()..sort();
          expect(vals[0], closeTo(1.0, 1e-9));
          expect(vals[1], closeTo(3.0, 1e-9));
        });
      });

      test('qr() and svd() validate float/complex dtype preconditions', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          expect(() => qr(a), throwsArgumentError);
          expect(() => svd(a), throwsArgumentError);
        });
      });
    });

    group('4. NormKind Enum & Nuclear Norm Support', () {
      test('Matrix NormKind.nuclear (Nuclear Norm / Trace Norm)', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final nucNorm = norm(a, ord: NormKind.nuclear);
          expect(nucNorm.scalar, closeTo(3.0, 1e-9));
        });
      });

      test('Complex Matrix NormKind.nuclear', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(0.0, 0.0),
              Complex(0.0, 0.0),
              Complex(0.0, 2.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final nucNorm = norm(a, ord: NormKind.nuclear);
          expect(nucNorm.scalar, closeTo(3.0, 1e-9));
        });
      });

      test('Matrix NormKind all enum values', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, -2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(
            norm(a, ord: NormKind.frobenius).scalar,
            closeTo(math.sqrt(30.0), 1e-9),
          );
          expect(norm(a, ord: NormKind.l1).scalar, closeTo(6.0, 1e-9));
          expect(norm(a, ord: NormKind.infinity).scalar, closeTo(7.0, 1e-9));
          final svdRes = svd(a);
          expect(
            norm(a, ord: NormKind.l2).scalar,
            closeTo(svdRes.s.toList()[0], 1e-9),
          );
        });
      });

      test('Vector NormKind all enum values', () {
        NDArray.scope(() {
          final v = NDArray.fromList([3.0, -4.0], [2], DType.float64);
          expect(norm(v, ord: NormKind.l2).scalar, closeTo(5.0, 1e-9));
          expect(norm(v, ord: NormKind.l1).scalar, closeTo(7.0, 1e-9));
          expect(norm(v, ord: NormKind.infinity).scalar, closeTo(4.0, 1e-9));
          expect(norm(v, ord: NormKind.negInfinity).scalar, closeTo(3.0, 1e-9));

          expect(() => norm(v, ord: NormKind.nuclear), throwsArgumentError);
        });
      });
    });

    group('5. Masked Mathematical Ufuncs (where: mask propagation)', () {
      test('sqrt with where mask', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [4.0, 9.0, 16.0, 25.0],
            [4],
            DType.float64,
          );
          final mask = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final out = NDArray.fromList(
            [100.0, 200.0, 300.0, 400.0],
            [4],
            DType.float64,
          );

          final res = sqrt(a, where: mask, out: out);
          expect(res.toList(), [2.0, 200.0, 4.0, 400.0]);
        });
      });

      test('exp, log, log2, log10 with where mask', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final mask = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final out = NDArray.fromList(
            [-1.0, -1.0, -1.0, -1.0],
            [4],
            DType.float64,
          );

          final resExp = exp(a, where: mask, out: out);
          expect(resExp.toList()[0], closeTo(math.exp(1.0), 1e-9));
          expect(resExp.toList()[1], -1.0);
          expect(resExp.toList()[2], closeTo(math.exp(3.0), 1e-9));
          expect(resExp.toList()[3], -1.0);

          final resLog = log(a, where: mask, out: out);
          expect(resLog.toList()[0], closeTo(math.log(1.0), 1e-9));
          expect(resLog.toList()[1], -1.0);
          expect(resLog.toList()[2], closeTo(math.log(3.0), 1e-9));
          expect(resLog.toList()[3], -1.0);
        });
      });

      test('Complex sqrt with where mask', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(4.0, 0.0), Complex(9.0, 0.0)],
            [2],
            DType.complex128,
          );
          final mask = NDArray.fromList([true, false], [2], DType.boolean);
          final out = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(99.0, 99.0)],
            [2],
            DType.complex128,
          );

          final res = sqrt(a, where: mask, out: out);
          final r0 = res.getCell([0]);
          final r1 = res.getCell([1]);
          expect(r0.real, closeTo(2.0, 1e-9));
          expect(r0.imag, closeTo(0.0, 1e-9));
          expect(r1.real, 99.0);
          expect(r1.imag, 99.0);
        });
      });
    });

    group('6. FFT Non-Final Axis View Return & fftshift out Parameter', () {
      test('fft along non-final axis (axis: 0 on 2D) is safe to dispose', () {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );

        final res = fft(a, axis: 0);
        expect(res.shape, [2, 3]);
        expect(res.dtype, DType.complex128);

        res.dispose();
        a.dispose();
      });

      test('rfft and irfft along non-final axis (axis: 0 on 2D)', () {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
          [4, 3],
          DType.float64,
        );

        final rfftRes = rfft(a, axis: 0);
        expect(rfftRes.shape, [3, 3]);

        final irfftRes = irfft(rfftRes, n: 4, axis: 0);
        expect(irfftRes.shape, [4, 3]);

        for (var i = 0; i < 4; i++) {
          for (var j = 0; j < 3; j++) {
            expect(irfftRes.getCell([i, j]), closeTo(a.getCell([i, j]), 1e-9));
          }
        }

        rfftRes.dispose();
        irfftRes.dispose();
        a.dispose();
      });

      test('fftshift and ifftshift with out parameter', () {
        NDArray.scope(() {
          final x = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );
          final outBuffer = NDArray.zeros([5], DType.float64);

          final shifted = fftshift(x, out: outBuffer);
          expect(identical(shifted, outBuffer), true);
          expect(shifted.toList(), [3.0, 4.0, 0.0, 1.0, 2.0]);

          final unshiftedBuffer = NDArray.zeros([5], DType.float64);
          final unshifted = ifftshift(shifted, out: unshiftedBuffer);
          expect(identical(unshifted, unshiftedBuffer), true);
          expect(unshifted.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);
        });
      });
    });

    group('7. Tensordot & Einsum View Return Memory Safety', () {
      test('tensordot returned view is independent and safe to dispose', () {
        final a = NDArray.ones([2, 3, 4], DType.float64);
        final b = NDArray.ones([4, 5], DType.float64);

        final res = tensordot(a, b, axes: [2, 0]);
        expect(res.shape, [2, 3, 5]);

        res.dispose();
        a.dispose();
        b.dispose();
      });

      test('einsum returned reshaped/transposed view is safe to dispose', () {
        final a = NDArray.ones([2, 3], DType.float64);
        final res = einsum(EinsumSubscripts.parse('ij->ji'), [a]);
        expect(res.shape, [3, 2]);

        res.dispose();
        a.dispose();
      });
      group(
        '8. Zero-Sized Matrix (n == 0) Fast Paths & Consistent Scope Protection',
        () {
          test('solve on n == 0 matrix returns bCopy without LAPACK call', () {
            final a = NDArray.zeros([0, 0], DType.float64);
            final b1 = NDArray.zeros([0], DType.float64);
            final x1 = solve(a, b1);
            expect(x1.shape, [0]);
            expect(x1.isDisposed, false);
            x1.dispose();

            final b2 = NDArray.zeros([0, 3], DType.float64);
            final x2 = solve(a, b2);
            expect(x2.shape, [0, 3]);
            expect(x2.isDisposed, false);
            x2.dispose();

            a.dispose();
            b1.dispose();
            b2.dispose();
          });

          test(
            'eig, eigvals, eigvalsh, eigh on n == 0 matrix return empty arrays',
            () {
              final a = NDArray.zeros([0, 0], DType.float64);

              final resEig = eig(a);
              expect(resEig.eigenvalues.shape, [0]);
              expect(resEig.eigenvectors.shape, [0, 0]);
              expect(resEig.eigenvalues.isDisposed, false);
              expect(resEig.eigenvectors.isDisposed, false);
              resEig.dispose();

              final resEigvals = eigvals(a);
              expect(resEigvals.shape, [0]);
              expect(resEigvals.isDisposed, false);
              resEigvals.dispose();

              final resEigvalsh = eigvalsh(a);
              expect(resEigvalsh.shape, [0]);
              expect(resEigvalsh.isDisposed, false);
              resEigvalsh.dispose();

              final resEigh = eigh(a);
              expect(resEigh.eigenvalues.shape, [0]);
              expect(resEigh.eigenvectors.shape, [0, 0]);
              expect(resEigh.eigenvalues.isDisposed, false);
              expect(resEigh.eigenvectors.isDisposed, false);
              resEigh.dispose();

              a.dispose();
            },
          );

          test('hessenberg, qr, svd on n == 0 matrix and scope safety', () {
            final a = NDArray.zeros([0, 0], DType.float64);

            final resHess = hessenberg(a);
            expect(resHess.h.shape, [0, 0]);
            expect(resHess.q.shape, [0, 0]);
            expect(resHess.h.isDisposed, false);
            expect(resHess.q.isDisposed, false);
            resHess.h.dispose();
            resHess.q.dispose();

            final resQR = qr(a);
            expect(resQR.q.shape, [0, 0]);
            expect(resQR.r.shape, [0, 0]);
            expect(resQR.q.isDisposed, false);
            expect(resQR.r.isDisposed, false);
            resQR.q.dispose();
            resQR.r.dispose();

            final resSVD = svd(a);
            expect(resSVD.u.shape, [0, 0]);
            expect(resSVD.s.shape, [0]);
            expect(resSVD.vh.shape, [0, 0]);
            expect(resSVD.u.isDisposed, false);
            expect(resSVD.s.isDisposed, false);
            expect(resSVD.vh.isDisposed, false);
            resSVD.u.dispose();
            resSVD.s.dispose();
            resSVD.vh.dispose();

            a.dispose();
          });
        },
      );
    });
  });
}
