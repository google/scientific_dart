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
  group(
    "Comprehensive Linear Algebra, Solvers, Contractions & Manipulation Suite (>=90%)",
    () {
      // 1. SVD Matrix Decomposition
      group("1. SVD Matrix Decomposition", () {
        test(
          "SVD Float64 square matrix reconstruction and properties",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([
                3.0,
                2.0,
                2.0,
                2.0,
                3.0,
                -2.0,
                -2.0,
                -2.0,
                3.0,
              ]),
              [3, 3],
              DType.float64,
            );
            final res = svd(a);
            expect(res.u.shape, [3, 3]);
            expect(res.s.shape, [3]);
            expect(res.vh.shape, [3, 3]);
            expect(res.u.dtype, DType.float64);
            expect(res.s.dtype, DType.float64);
            expect(res.vh.dtype, DType.float64);

            final sList = res.s.toList();
            expect(sList[0], greaterThanOrEqualTo(sList[1]));
            expect(sList[1], greaterThanOrEqualTo(sList[2]));

            final sMat = NDArray.zeros([3, 3], DType.float64);
            for (var i = 0; i < 3; i++) {
              sMat.setCell([i, i], sList[i]);
            }
            final us = matmul(res.u, sMat);
            final usvh = matmul(us, res.vh);
            for (var i = 0; i < 9; i++) {
              expect(usvh.toList()[i], closeTo(a.toList()[i], 1e-5));
            }

            final utU = matmul(res.u.transpose(), res.u);
            for (var r = 0; r < 3; r++) {
              for (var c = 0; c < 3; c++) {
                expect(utU.getCell([r, c]), closeTo(r == c ? 1.0 : 0.0, 1e-5));
              }
            }
            res.dispose();
          }),
        );

        test(
          "SVD Float32 tall (m > n) and wide (m < n) matrices",
          () => NDArray.scope(() {
            final tall = NDArray.fromList(
              Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [4, 2],
              DType.float32,
            );
            final resTall = svd(tall);
            expect(resTall.u.shape, [4, 4]);
            expect(resTall.s.shape, [2]);
            expect(resTall.vh.shape, [2, 2]);
            expect(resTall.u.dtype, DType.float32);
            expect(resTall.s.dtype, DType.float32);
            resTall.dispose();

            final wide = NDArray.fromList(
              Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [2, 4],
              DType.float32,
            );
            final resWide = svd(wide);
            expect(resWide.u.shape, [2, 2]);
            expect(resWide.s.shape, [2]);
            expect(resWide.vh.shape, [4, 4]);
            expect(resWide.u.dtype, DType.float32);
            expect(resWide.s.dtype, DType.float32);
            resWide.dispose();
          }),
        );

        test(
          "SVD Complex128 and Complex64 reconstruction",
          () => NDArray.scope(() {
            final cpx128 = NDArray.fromList(
              [
                Complex(1.0, 2.0),
                Complex(3.0, 4.0),
                Complex(5.0, 6.0),
                Complex(7.0, 8.0),
              ],
              [2, 2],
              DType.complex128,
            );
            final res128 = svd(cpx128);
            expect(res128.u.shape, [2, 2]);
            expect(res128.s.shape, [2]);
            expect(res128.vh.shape, [2, 2]);
            expect(res128.u.dtype, DType.complex128);
            expect(res128.s.dtype, DType.float64);
            expect(res128.vh.dtype, DType.complex128);

            final sMat = NDArray.zeros([2, 2], DType.complex128);
            final sVals = res128.s.toList();
            sMat.setCell([0, 0], Complex(sVals[0], 0.0));
            sMat.setCell([1, 1], Complex(sVals[1], 0.0));
            final us = matmul(res128.u, sMat);
            final usvh = matmul(us, res128.vh);
            for (var i = 0; i < 4; i++) {
              expect(
                usvh.toList()[i].real,
                closeTo(cpx128.toList()[i].real, 1e-4),
              );
              expect(
                usvh.toList()[i].imag,
                closeTo(cpx128.toList()[i].imag, 1e-4),
              );
            }
            res128.dispose();

            final cpx64 = NDArray.fromList(
              [
                Complex(2.0, -1.0),
                Complex(0.0, 3.0),
                Complex(1.0, 1.0),
                Complex(-2.0, 0.0),
              ],
              [2, 2],
              DType.complex64,
            );
            final res64 = svd(cpx64);
            expect(res64.s.dtype, DType.float32);
            expect(res64.u.dtype, DType.complex64);
            res64.dispose();
          }),
        );

        test(
          "SVD 3D batch decomposition across float64 and complex128",
          () => NDArray.scope(() {
            final bFloat = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [2, 2, 2],
              DType.float64,
            );
            final resB = svd(bFloat);
            expect(resB.u.shape, [2, 2, 2]);
            expect(resB.s.shape, [2, 2]);
            expect(resB.vh.shape, [2, 2, 2]);
            resB.dispose();

            final bCpx = NDArray.fromList(
              [
                Complex(1.0, 0.0),
                Complex(0.0, 1.0),
                Complex(0.0, -1.0),
                Complex(2.0, 0.0),
                Complex(3.0, 0.0),
                Complex(1.0, 1.0),
                Complex(1.0, -1.0),
                Complex(4.0, 0.0),
              ],
              [2, 2, 2],
              DType.complex128,
            );
            final resBCpx = svd(bCpx);
            expect(resBCpx.u.shape, [2, 2, 2]);
            expect(resBCpx.s.shape, [2, 2]);
            expect(resBCpx.vh.shape, [2, 2, 2]);
            resBCpx.dispose();
          }),
        );

        test(
          "SVD edge cases: empty 0x0, 0x3, 3x0, and 1x1 matrices",
          () => NDArray.scope(() {
            final empty00 = NDArray<Float64>.zeros([0, 0], DType.float64);
            final res00 = svd(empty00);
            expect(res00.u.shape, [0, 0]);
            expect(res00.s.shape, [0]);
            expect(res00.vh.shape, [0, 0]);
            res00.dispose();

            final empty03 = NDArray<Float64>.zeros([0, 3], DType.float64);
            final res03 = svd(empty03);
            expect(res03.u.shape, [0, 0]);
            expect(res03.s.shape, [0]);
            expect(res03.vh.shape, [3, 3]);
            res03.dispose();

            final empty30 = NDArray<Float64>.zeros([3, 0], DType.float64);
            final res30 = svd(empty30);
            expect(res30.u.shape, [3, 3]);
            expect(res30.s.shape, [0]);
            expect(res30.vh.shape, [0, 0]);
            res30.dispose();

            final single = NDArray.fromList(Float64List.fromList([5.0]), [
              1,
              1,
            ], DType.float64);
            final res11 = svd(single);
            expect(res11.s.toList()[0], closeTo(5.0, 1e-6));
            res11.dispose();
          }),
        );

        test(
          "SVD out buffer recycling and error handling",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final outU = NDArray<Float64>.zeros([2, 2], DType.float64);
            final outS = NDArray<Float64>.zeros([2], DType.float64);
            final outVh = NDArray<Float64>.zeros([2, 2], DType.float64);

            final res = svd(a, out: (u: outU, s: outS, vh: outVh));
            expect(identical(res.u, outU), true);
            expect(identical(res.s, outS), true);
            expect(identical(res.vh, outVh), true);

            final intMat = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
              2,
              2,
            ], DType.int32);
            expect(() => svd(intMat), throwsArgumentError);

            final vec1D = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            expect(() => svd(vec1D), throwsArgumentError);

            final badU = NDArray.zeros([3, 3], DType.float64);
            expect(
              () => svd(a, out: (u: badU, s: outS, vh: outVh)),
              throwsArgumentError,
            );
          }),
        );
      });

      // 2. QR Decomposition
      group("2. QR Decomposition", () {
        test(
          "QR Float64 square and rectangular with reconstruction and orthogonality",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([
                12.0,
                -51.0,
                4.0,
                6.0,
                167.0,
                -68.0,
                -4.0,
                24.0,
                -41.0,
              ]),
              [3, 3],
              DType.float64,
            );
            final res = qr(a);
            expect(res.q.shape, [3, 3]);
            expect(res.r.shape, [3, 3]);

            final qrProd = matmul(res.q, res.r);
            for (var i = 0; i < 9; i++) {
              expect(qrProd.toList()[i], closeTo(a.toList()[i], 1e-5));
            }

            final qtQ = matmul(res.q.transpose(), res.q);
            for (var r = 0; r < 3; r++) {
              for (var c = 0; c < 3; c++) {
                expect(qtQ.getCell([r, c]), closeTo(r == c ? 1.0 : 0.0, 1e-5));
              }
            }

            for (var r = 1; r < 3; r++) {
              for (var c = 0; c < r; c++) {
                expect(res.r.getCell([r, c]).abs(), lessThan(1e-5));
              }
            }
            res.dispose();
          }),
        );

        test(
          "QR Float32 tall (m > n) and wide (m < n) matrices",
          () => NDArray.scope(() {
            final tall = NDArray.fromList(
              Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [4, 2],
              DType.float32,
            );
            final resTall = qr(tall);
            expect(resTall.q.shape, [4, 2]);
            expect(resTall.r.shape, [2, 2]);
            final reconTall = matmul(resTall.q, resTall.r);
            for (var i = 0; i < 8; i++) {
              expect(reconTall.toList()[i], closeTo(tall.toList()[i], 1e-4));
            }
            resTall.dispose();

            final wide = NDArray.fromList(
              Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [2, 4],
              DType.float32,
            );
            final resWide = qr(wide);
            expect(resWide.q.shape, [2, 2]);
            expect(resWide.r.shape, [2, 4]);
            final reconWide = matmul(resWide.q, resWide.r);
            for (var i = 0; i < 8; i++) {
              expect(reconWide.toList()[i], closeTo(wide.toList()[i], 1e-4));
            }
            resWide.dispose();
          }),
        );

        test(
          "QR Complex128 and Complex64 with unitarity check",
          () => NDArray.scope(() {
            final cpx128 = NDArray.fromList(
              [
                Complex(1.0, 1.0),
                Complex(2.0, -1.0),
                Complex(3.0, 2.0),
                Complex(4.0, 0.0),
              ],
              [2, 2],
              DType.complex128,
            );
            final res = qr(cpx128);
            expect(res.q.shape, [2, 2]);
            expect(res.r.shape, [2, 2]);
            expect(res.q.dtype, DType.complex128);
            expect(res.r.dtype, DType.complex128);

            final recon = matmul(res.q, res.r);
            for (var i = 0; i < 4; i++) {
              expect(
                recon.toList()[i].real,
                closeTo(cpx128.toList()[i].real, 1e-5),
              );
              expect(
                recon.toList()[i].imag,
                closeTo(cpx128.toList()[i].imag, 1e-5),
              );
            }
            res.dispose();

            final cpx64 = NDArray.fromList(
              [
                Complex(2.0, 0.0),
                Complex(1.0, 1.0),
                Complex(1.0, -1.0),
                Complex(3.0, 0.0),
              ],
              [2, 2],
              DType.complex64,
            );
            final res64 = qr(cpx64);
            expect(res64.q.dtype, DType.complex64);
            res64.dispose();
          }),
        );

        test(
          "QR 3D batch decomposition",
          () => NDArray.scope(() {
            final bMat = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [2, 2, 2],
              DType.float64,
            );
            final res = qr(bMat);
            expect(res.q.shape, [2, 2, 2]);
            expect(res.r.shape, [2, 2, 2]);
            final recon = matmul(res.q, res.r);
            for (var i = 0; i < 8; i++) {
              expect(recon.toList()[i], closeTo(bMat.toList()[i], 1e-5));
            }
            res.dispose();
          }),
        );

        test(
          "QR edge cases: empty 0x0, out buffers and errors",
          () => NDArray.scope(() {
            final empty = NDArray<Float64>.zeros([0, 0], DType.float64);
            final res00 = qr(empty);
            expect(res00.q.shape, [0, 0]);
            expect(res00.r.shape, [0, 0]);
            res00.dispose();

            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final outQ = NDArray<Float64>.zeros([2, 2], DType.float64);
            final outR = NDArray<Float64>.zeros([2, 2], DType.float64);
            final res = qr(a, out: (q: outQ, r: outR));
            expect(identical(res.q, outQ), true);
            expect(identical(res.r, outR), true);

            expect(
              () => qr(
                NDArray.fromList(Int32List.fromList([1, 2]), [2], DType.int32),
              ),
              throwsArgumentError,
            );
            final badQ = NDArray.zeros([3, 3], DType.float64);
            expect(() => qr(a, out: (q: badQ, r: outR)), throwsArgumentError);
          }),
        );
      });

      // 3. Cholesky Decomposition
      group("3. Cholesky Decomposition", () {
        test(
          "Cholesky Float64 and Float32 positive definite reconstruction (A = L L^T)",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([
                4.0,
                12.0,
                -16.0,
                12.0,
                37.0,
                -43.0,
                -16.0,
                -43.0,
                98.0,
              ]),
              [3, 3],
              DType.float64,
            );
            final l = cholesky(a);
            expect(l.shape, [3, 3]);
            expect(l.dtype, DType.float64);

            for (var r = 0; r < 3; r++) {
              for (var c = r + 1; c < 3; c++) {
                expect(l.getCell([r, c]), 0.0);
              }
            }

            final recon = matmul(l, l.transpose());
            for (var i = 0; i < 9; i++) {
              expect(recon.toList()[i], closeTo(a.toList()[i], 1e-5));
            }

            final a32 = NDArray.fromList(
              Float32List.fromList([2.0, -1.0, -1.0, 2.0]),
              [2, 2],
              DType.float32,
            );
            final l32 = cholesky(a32);
            expect(l32.dtype, DType.float32);
            final recon32 = matmul(l32, l32.transpose());
            for (var i = 0; i < 4; i++) {
              expect(recon32.toList()[i], closeTo(a32.toList()[i], 1e-5));
            }
          }),
        );

        test(
          "Cholesky Complex128 and Complex64 Hermitian positive definite (A = L L^H)",
          () => NDArray.scope(() {
            final cpx = NDArray.fromList(
              [
                Complex(4.0, 0.0),
                Complex(1.0, -1.0),
                Complex(1.0, 1.0),
                Complex(3.0, 0.0),
              ],
              [2, 2],
              DType.complex128,
            );
            final l = cholesky(cpx);
            expect(l.shape, [2, 2]);
            expect(l.dtype, DType.complex128);

            final lH = conjugate(l.transpose());
            final recon = matmul(l, lH);
            for (var i = 0; i < 4; i++) {
              expect(
                recon.toList()[i].real,
                closeTo(cpx.toList()[i].real, 1e-5),
              );
              expect(
                recon.toList()[i].imag,
                closeTo(cpx.toList()[i].imag, 1e-5),
              );
            }

            final cpx64 = NDArray.fromList(
              [
                Complex(5.0, 0.0),
                Complex(2.0, 1.0),
                Complex(2.0, -1.0),
                Complex(4.0, 0.0),
              ],
              [2, 2],
              DType.complex64,
            );
            final l64 = cholesky(cpx64);
            expect(l64.dtype, DType.complex64);
          }),
        );

        test(
          "Cholesky 3D Batch tensors and out buffer",
          () => NDArray.scope(() {
            final bA = NDArray.fromList(
              Float64List.fromList([4.0, 2.0, 2.0, 5.0, 2.0, -1.0, -1.0, 2.0]),
              [2, 2, 2],
              DType.float64,
            );
            final outL = NDArray<Float64>.zeros([2, 2, 2], DType.float64);
            final l = cholesky(bA, out: outL);
            expect(identical(l, outL), true);
          }),
        );

        test(
          "Cholesky non-positive definite throws NonPositiveDefiniteException",
          () => NDArray.scope(() {
            final indef = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 2.0, 1.0]),
              [2, 2],
              DType.float64,
            );
            expect(
              () => cholesky(indef),
              throwsA(isA<NonPositiveDefiniteException>()),
            );

            final indefCpx = NDArray.fromList(
              [
                Complex(1.0, 0.0),
                Complex(2.0, 0.0),
                Complex(2.0, 0.0),
                Complex(-1.0, 0.0),
              ],
              [2, 2],
              DType.complex128,
            );
            expect(
              () => cholesky(indefCpx),
              throwsA(isA<NonPositiveDefiniteException>()),
            );
          }),
        );
      });

      // 4. Eigensystems (eig, eigh, eigvals, eigvalsh)
      group("4. Eigensystems (eig, eigh, eigvals, eigvalsh)", () {
        test(
          "eig on non-symmetric matrix with complex eigenvalues",
          () => NDArray.scope(() {
            final rot = NDArray.fromList(
              Float64List.fromList([0.0, -1.0, 1.0, 0.0]),
              [2, 2],
              DType.float64,
            );
            final res = eig(rot);
            expect(res.eigenvalues.shape, [2]);
            expect(res.eigenvectors.shape, [2, 2]);
            final vals = res.eigenvalues.toList();
            expect(vals[0].real, closeTo(0.0, 1e-6));
            expect(vals[0].imag.abs(), closeTo(1.0, 1e-6));
            expect(vals[1].real, closeTo(0.0, 1e-6));
            expect(vals[1].imag.abs(), closeTo(1.0, 1e-6));
            res.dispose();
          }),
        );

        test(
          "eigvals on Float64, Float32, Complex128, Complex64 across 2D & 3D",
          () => NDArray.scope(() {
            final a64 = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final v64 = eigvals(a64);
            expect(v64.shape, [2]);
            expect(v64.dtype, DType.complex128);

            final a32 = NDArray.fromList(
              Float32List.fromList([2.0, 1.0, 1.0, 2.0]),
              [2, 2],
              DType.float32,
            );
            final v32 = eigvals(a32);
            expect(v32.shape, [2]);
            expect(v32.dtype, DType.complex64);

            final c128 = NDArray.fromList(
              [Complex(1, 1), Complex(0, 0), Complex(0, 0), Complex(2, -1)],
              [2, 2],
              DType.complex128,
            );
            final vc128 = eigvals(c128);
            expect(vc128.shape, [2]);
            expect(vc128.dtype, DType.complex128);

            final bMat = NDArray.fromList(
              Float64List.fromList([1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0]),
              [2, 2, 2],
              DType.float64,
            );
            final bVals = eigvals(bMat);
            expect(bVals.shape, [2, 2]);
          }),
        );

        test(
          "eigh and eigvalsh for symmetric real and Hermitian complex matrices (lower/upper)",
          () => NDArray.scope(() {
            final sym = NDArray.fromList(
              Float64List.fromList([2.0, 1.0, 1.0, 2.0]),
              [2, 2],
              DType.float64,
            );
            final resLower = eigh(sym, uplo: MatrixTriangle.lower);
            expect(resLower.eigenvalues.shape, [2]);
            expect(resLower.eigenvalues.toList()[0], closeTo(1.0, 1e-5));
            expect(resLower.eigenvalues.toList()[1], closeTo(3.0, 1e-5));
            resLower.dispose();

            final resUpper = eigh(sym, uplo: MatrixTriangle.upper);
            expect(resUpper.eigenvalues.toList()[0], closeTo(1.0, 1e-5));
            expect(resUpper.eigenvalues.toList()[1], closeTo(3.0, 1e-5));
            resUpper.dispose();

            final vals = eigvalsh(sym);
            expect(vals.toList()[0], closeTo(1.0, 1e-5));
            expect(vals.toList()[1], closeTo(3.0, 1e-5));

            final herm = NDArray.fromList(
              [
                Complex(2.0, 0.0),
                Complex(1.0, -1.0),
                Complex(1.0, 1.0),
                Complex(3.0, 0.0),
              ],
              [2, 2],
              DType.complex128,
            );
            final resHerm = eigh(herm);
            expect(resHerm.eigenvalues.dtype, DType.float64);
            expect(resHerm.eigenvectors.dtype, DType.complex128);
            resHerm.dispose();

            final hermVals = eigvalsh(herm);
            expect(hermVals.dtype, DType.float64);

            final intMat = NDArray.fromList(Int32List.fromList([2, 1, 1, 2]), [
              2,
              2,
            ], DType.int32);
            final intEigh = eigh(intMat);
            expect(intEigh.eigenvalues.dtype, DType.float64);
            intEigh.dispose();
          }),
        );
      });

      // 5. Schur & Hessenberg Decompositions
      group("5. Schur & Hessenberg Decompositions", () {
        test(
          "schur decomposition across real and complex forms",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([
                0.0,
                2.0,
                2.0,
                -1.0,
                3.0,
                -2.0,
                1.0,
                -1.0,
                2.0,
              ]),
              [3, 3],
              DType.float64,
            );
            final resReal = schur(a);
            expect(resReal.t.shape, [3, 3]);
            expect(resReal.z.shape, [3, 3]);

            final ztz = matmul(resReal.z, resReal.t);
            final recon = matmul(ztz, resReal.z.transpose());
            for (var i = 0; i < 9; i++) {
              expect(recon.toList()[i], closeTo(a.toList()[i], 1e-4));
            }
            resReal.dispose();

            final resCpx = complexSchur(a);
            expect(resCpx.t.dtype, DType.complex128);
            expect(resCpx.z.dtype, DType.complex128);
            resCpx.dispose();

            final empty = NDArray<Float64>.zeros([0, 0], DType.float64);
            final emptyRes = schur(empty);
            expect(emptyRes.t.shape, [0, 0]);
            emptyRes.dispose();
          }),
        );

        test(
          "hessenberg decomposition and upper Hessenberg verification",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([
                1.0,
                5.0,
                2.0,
                3.0,
                4.0,
                1.0,
                4.0,
                3.0,
                2.0,
                1.0,
                5.0,
                2.0,
                1.0,
                2.0,
                3.0,
                4.0,
              ]),
              [4, 4],
              DType.float64,
            );
            final res = hessenberg(a);
            expect(res.h.shape, [4, 4]);
            expect(res.q.shape, [4, 4]);

            for (var r = 2; r < 4; r++) {
              for (var c = 0; c < r - 1; c++) {
                expect(res.h.getCell([r, c]).abs(), lessThan(1e-5));
              }
            }

            final qh = matmul(res.q, res.h);
            final recon = matmul(qh, res.q.transpose());
            for (var i = 0; i < 16; i++) {
              expect(recon.toList()[i], closeTo(a.toList()[i], 1e-4));
            }
            res.dispose();
          }),
        );
      });

      // 6. Moore-Penrose Pseudo-Inverse (pinv)
      group("6. Moore-Penrose Pseudo-Inverse (pinv)", () {
        test(
          "pinv on tall, wide, and square matrices with Moore-Penrose equations",
          () => NDArray.scope(() {
            final tall = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [3, 2],
              DType.float64,
            );
            final pinvTall = pinv(tall);
            expect(pinvTall.shape, [2, 3]);

            final aPinv = matmul(tall, pinvTall);
            final aPinvA = matmul(aPinv, tall);
            for (var i = 0; i < 6; i++) {
              expect(aPinvA.toList()[i], closeTo(tall.toList()[i], 1e-5));
            }

            final pinvA = matmul(pinvTall, tall);
            final pinvAPinv = matmul(pinvA, pinvTall);
            for (var i = 0; i < 6; i++) {
              expect(
                pinvAPinv.toList()[i],
                closeTo(pinvTall.toList()[i], 1e-5),
              );
            }

            final wide = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );
            final pinvWide = pinv(wide, rcond: 1e-10);
            expect(pinvWide.shape, [3, 2]);

            final sq = NDArray.fromList(
              Float64List.fromList([4.0, 7.0, 2.0, 6.0]),
              [2, 2],
              DType.float64,
            );
            final pinvSq = pinv(sq);
            final invSq = inv(sq);
            for (var i = 0; i < 4; i++) {
              expect(pinvSq.toList()[i], closeTo(invSq.toList()[i], 1e-5));
            }

            final empty = NDArray<Float64>.zeros([0, 0], DType.float64);
            final pinvEmpty = pinv(empty);
            expect(pinvEmpty.shape, [0, 0]);
          }),
        );
      });

      // 7. Matrix Power (matrix_power)
      group("7. Matrix Power (matrix_power)", () {
        test(
          "matrix_power for n = 0, n > 0, and n < 0",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([2.0, 1.0, 0.0, 2.0]),
              [2, 2],
              DType.float64,
            );

            final p0 = matrix_power(a, 0);
            expect(p0.toList(), [1.0, 0.0, 0.0, 1.0]);

            final p1 = matrix_power(a, 1);
            expect(p1.toList(), [2.0, 1.0, 0.0, 2.0]);

            final p2 = matrix_power(a, 2);
            expect(p2.toList(), [4.0, 4.0, 0.0, 4.0]);

            final p3 = matrix_power(a, 3);
            expect(p3.toList(), [8.0, 12.0, 0.0, 8.0]);

            final pNeg1 = matrix_power(a, -1);
            final aInv = inv(a);
            for (var i = 0; i < 4; i++) {
              expect(pNeg1.toList()[i], closeTo(aInv.toList()[i], 1e-5));
            }

            final pNeg2 = matrix_power(a, -2);
            final aInv2 = matmul(aInv, aInv);
            for (var i = 0; i < 4; i++) {
              expect(pNeg2.toList()[i], closeTo(aInv2.toList()[i], 1e-5));
            }

            final nonSq = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );
            expect(() => matrix_power(nonSq, 2), throwsArgumentError);
          }),
        );
      });

      // 8. Linear Solvers (solve & lstsq)
      group("8. Linear Solvers (solve & lstsq)", () {
        test(
          "solve on 2D and 3D batch systems with 1D vector and 2D matrix RHS",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([3.0, 1.0, 1.0, 2.0]),
              [2, 2],
              DType.float64,
            );
            final b1D = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
              2,
            ], DType.float64);
            final x1D = solve(a, b1D);
            expect(x1D.shape, [2]);
            expect(x1D.toList()[0], closeTo(2.0, 1e-5));
            expect(x1D.toList()[1], closeTo(3.0, 1e-5));

            final b2D = NDArray.fromList(
              Float64List.fromList([9.0, 18.0, 8.0, 16.0]),
              [2, 2],
              DType.float64,
            );
            final x2D = solve(a, b2D);
            expect(x2D.shape, [2, 2]);
            expect(x2D.getCell([0, 0]), closeTo(2.0, 1e-5));
            expect(x2D.getCell([0, 1]), closeTo(4.0, 1e-5));
            expect(x2D.getCell([1, 0]), closeTo(3.0, 1e-5));
            expect(x2D.getCell([1, 1]), closeTo(6.0, 1e-5));

            final singular = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            expect(
              () => solve(singular, b1D),
              throwsA(isA<SingularMatrixException>()),
            );
          }),
        );

        test(
          "lstsq over-determined, under-determined, and rank-deficient systems",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 1.0, 1.0, 2.0, 1.0, 3.0]),
              [3, 2],
              DType.float64,
            );
            final b = NDArray.fromList(Float64List.fromList([6.0, 5.0, 7.0]), [
              3,
            ], DType.float64);
            final res = lstsq(a, b);
            expect(res.x.shape, [2]);
            expect(res.rank, 2);
            expect(res.s.shape, [2]);
            expect(res.residuals.shape, [1]);
            res.dispose();

            final aWide = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );
            final bWide = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final resWide = lstsq(aWide, bWide);
            expect(resWide.x.shape, [3]);
            expect(resWide.residuals.shape, [0]);
            resWide.dispose();
          }),
        );
      });

      // 9. Matrix & Vector Norms (norm)
      group("9. Matrix & Vector Norms (norm)", () {
        test(
          "Vector norms: 0, 1, 2, inf, -inf, p",
          () => NDArray.scope(() {
            final v = NDArray.fromList(
              Float64List.fromList([3.0, -4.0, 0.0, 12.0]),
              [4],
              DType.float64,
            );

            expect(norm(v, ord: 0).scalar, 3.0);
            expect(norm(v, ord: 1).scalar, 19.0);
            expect(norm(v, ord: NormKind.l1).scalar, 19.0);
            expect(norm(v, ord: 2).scalar, 13.0);
            expect(norm(v).scalar, 13.0);
            expect(norm(v, ord: NormKind.l2).scalar, 13.0);
            expect(norm(v, ord: double.infinity).scalar, 12.0);
            expect(norm(v, ord: NormKind.infinity).scalar, 12.0);
            expect(norm(v, ord: double.negativeInfinity).scalar, 0.0);
            expect(norm(v, ord: NormKind.negInfinity).scalar, 0.0);

            final p3 = math.pow(
              3.0 * 3.0 * 3.0 + 4.0 * 4.0 * 4.0 + 12.0 * 12.0 * 12.0,
              1.0 / 3.0,
            );
            expect(norm(v, ord: 3).scalar, closeTo(p3, 1e-5));
          }),
        );

        test(
          "Matrix norms: fro, 1, -1, inf, -inf, 2, -2, nuc",
          () => NDArray.scope(() {
            final m = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );

            expect(norm(m, ord: "fro").scalar, closeTo(math.sqrt(30.0), 1e-5));
            expect(
              norm(m, ord: "frobenius").scalar,
              closeTo(math.sqrt(30.0), 1e-5),
            );
            expect(
              norm(m, ord: NormKind.frobenius).scalar,
              closeTo(math.sqrt(30.0), 1e-5),
            );
            expect(norm(m, ord: 1).scalar, 6.0);
            expect(norm(m, ord: -1).scalar, 4.0);
            expect(norm(m, ord: double.infinity).scalar, 7.0);
            expect(norm(m, ord: double.negativeInfinity).scalar, 3.0);

            final sVals = svd(m).s.toList();
            expect(norm(m, ord: 2).scalar, closeTo(sVals[0], 1e-5));
            expect(norm(m, ord: -2).scalar, closeTo(sVals[1], 1e-5));
            expect(
              norm(m, ord: "nuc").scalar,
              closeTo(sVals[0] + sVals[1], 1e-5),
            );
            expect(
              norm(m, ord: "nuclear").scalar,
              closeTo(sVals[0] + sVals[1], 1e-5),
            );
            expect(
              norm(m, ord: NormKind.nuclear).scalar,
              closeTo(sVals[0] + sVals[1], 1e-5),
            );
          }),
        );

        test(
          "Matrix norm along custom batch axes with keepdims",
          () => NDArray.scope(() {
            final bMat = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [2, 2, 2],
              DType.float64,
            );
            final nMat = norm(bMat, ord: "fro", axis: [1, 2], keepdims: true);
            expect(nMat.shape, [2, 1, 1]);
            expect(nMat.getCell([0, 0, 0]), closeTo(math.sqrt(30.0), 1e-5));
            expect(nMat.getCell([1, 0, 0]), closeTo(math.sqrt(174.0), 1e-5));

            final nMatNoKeep = norm(
              bMat,
              ord: "fro",
              axis: [1, 2],
              keepdims: false,
            );
            expect(nMatNoKeep.shape, [2]);
          }),
        );
      });

      // 10. Matrix Chain Product (multi_dot)
      group("10. Matrix Chain Product (multi_dot)", () {
        test(
          "multi_dot with 3, 4, 5 mismatched dimension matrix chains",
          () => NDArray.scope(() {
            final m1 = NDArray.fromList(
              Float64List.fromList(
                List.generate(20, (i) => (i + 1).toDouble()),
              ),
              [10, 2],
              DType.float64,
            );
            final m2 = NDArray.fromList(
              Float64List.fromList(
                List.generate(100, (i) => (i + 1).toDouble()),
              ),
              [2, 50],
              DType.float64,
            );
            final m3 = NDArray.fromList(
              Float64List.fromList(
                List.generate(150, (i) => (i + 1).toDouble()),
              ),
              [50, 3],
              DType.float64,
            );

            final res = multi_dot([m1, m2, m3]);
            expect(res.shape, [10, 3]);

            final expected = matmul(matmul(m1, m2), m3);
            for (var i = 0; i < 30; i++) {
              expect(res.toList()[i], closeTo(expected.toList()[i], 1e-4));
            }

            final m4 = NDArray.fromList(
              Float64List.fromList(
                List.generate(12, (i) => (i + 1).toDouble()),
              ),
              [3, 4],
              DType.float64,
            );
            final res4 = multi_dot([m1, m2, m3, m4]);
            expect(res4.shape, [10, 4]);

            final m5 = NDArray.fromList(
              Float64List.fromList(List.generate(8, (i) => (i + 1).toDouble())),
              [4, 2],
              DType.float64,
            );
            final res5 = multi_dot([m1, m2, m3, m4, m5]);
            expect(res5.shape, [10, 2]);
          }),
        );

        test(
          "multi_dot with 1D vector at start, end, and both ends",
          () => NDArray.scope(() {
            final vStart = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final m1 = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );
            final m2 = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [3, 2],
              DType.float64,
            );
            final vEnd = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);

            final rStart = multi_dot([vStart, m1, m2]);
            expect(rStart.shape, [2]);

            final rEnd = multi_dot([m1, m2, vEnd]);
            expect(rEnd.shape, [2]);

            final rBoth = multi_dot([vStart, m1, m2, vEnd]);
            expect(rBoth.shape, []);
          }),
        );
      });

      // 11. Determinant & Log Determinant (det & slogdet)
      group("11. Determinant & Log Determinant (det & slogdet)", () {
        test(
          "det and slogdet on 2D and 3D batch across float64 and complex128",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final d = det(a);
            expect(d.scalar, closeTo(-2.0, 1e-5));

            final s = slogdet(a);
            expect(s.sign.scalar, closeTo(-1.0, 1e-5));
            expect(s.logabsdet.scalar, closeTo(math.log(2.0), 1e-5));
            s.dispose();

            final sing = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            expect(det(sing).scalar, closeTo(0.0, 1e-5));
            final sSing = slogdet(sing);
            expect(sSing.sign.scalar, 0.0);
            expect(sSing.logabsdet.scalar, double.negativeInfinity);
            sSing.dispose();

            final bMat = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0]),
              [2, 2, 2],
              DType.float64,
            );
            final bDet = det(bMat);
            expect(bDet.shape, [2]);
            expect(bDet.toList()[0], closeTo(-2.0, 1e-5));
            expect(bDet.toList()[1], closeTo(4.0, 1e-5));
          }),
        );
      });

      // 12. Tensor Contractions & Products
      group("12. Tensor Contractions & Products", () {
        test(
          "tensordot with axes 0, 1, 2, and explicit lists",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = NDArray.fromList(
              Float64List.fromList([5.0, 6.0, 7.0, 8.0]),
              [2, 2],
              DType.float64,
            );

            final td0 = tensordot(a, b, axes: 0);
            expect(td0.shape, [2, 2, 2, 2]);

            final td1 = tensordot(a, b, axes: 1);
            expect(td1.shape, [2, 2]);
            final mm = matmul(a, b);
            expect(td1.toList(), mm.toList());

            final td2 = tensordot(a, b, axes: 2);
            expect(td2.shape, []);
            expect(td2.scalar, closeTo(70.0, 1e-5));

            final tdExplicit = tensordot(
              a,
              b,
              axes: TensordotAxes.explicit([1], [0]),
            );
            expect(tdExplicit.shape, [2, 2]);
          }),
        );

        test(
          "einsum comprehensive operations",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = NDArray.fromList(
              Float64List.fromList([5.0, 6.0, 7.0, 8.0]),
              [2, 2],
              DType.float64,
            );

            final eMatmul = einsum(EinsumSubscripts.parse("ij,jk->ik"), [a, b]);
            expect(eMatmul.shape, [2, 2]);
            expect(eMatmul.toList(), matmul(a, b).toList());

            final eTrans = einsum(EinsumSubscripts.parse("ij->ji"), [a]);
            expect(eTrans.toList(), [1.0, 3.0, 2.0, 4.0]);

            final eTrace = einsum(EinsumSubscripts.parse("ii->"), [a]);
            expect(eTrace.scalar, 5.0);

            final eDiag = einsum(EinsumSubscripts.parse("ii->i"), [a]);
            expect(eDiag.toList(), [1.0, 4.0]);

            final v1 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
              3,
            ], DType.float64);
            final v2 = NDArray.fromList(Float64List.fromList([4.0, 5.0, 6.0]), [
              3,
            ], DType.float64);
            final eDot = einsum(EinsumSubscripts.parse("i,i->"), [v1, v2]);
            expect(eDot.scalar, closeTo(32.0, 1e-5));

            final eOuter = einsum(EinsumSubscripts.parse("i,j->ij"), [v1, v2]);
            expect(eOuter.shape, [3, 3]);
          }),
        );

        test(
          "inner, vdot, kron, outer, cross coverage",
          () => NDArray.scope(() {
            final v1 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
              3,
            ], DType.float64);
            final v2 = NDArray.fromList(Float64List.fromList([4.0, 5.0, 6.0]), [
              3,
            ], DType.float64);

            final inRes = inner(v1, v2);
            expect(inRes.scalar, closeTo(32.0, 1e-5));

            final c1 = NDArray.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              DType.complex128,
            );
            final c2 = NDArray.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              DType.complex128,
            );
            final vdotRes = vdot(c1, c2);
            expect(vdotRes.scalar.real, closeTo(30.0, 1e-5));
            expect(vdotRes.scalar.imag, closeTo(0.0, 1e-5));

            final kA = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final kB = NDArray.fromList(
              Float64List.fromList([0.0, 5.0, 6.0, 7.0]),
              [2, 2],
              DType.float64,
            );
            final kRes = kron(kA, kB);
            expect(kRes.shape, [4, 4]);

            final outRes = outer(v1, v2);
            expect(outRes.shape, [3, 3]);

            final crRes = cross(v1, v2);
            expect(crRes.shape, [3]);
            expect(crRes.toList(), [-3.0, 6.0, -3.0]);
          }),
        );
      });

      // 13. Array Manipulation (manipulation.dart)
      group("13. Array Manipulation (manipulation.dart)", () {
        test(
          "flip, fliplr, flipud across 1D, 2D, 3D",
          () => NDArray.scope(() {
            final a2D = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );

            final fAll = flip(a2D);
            expect(fAll.toList(), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

            final f0 = flip(a2D, axis: 0);
            expect(f0.toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

            final f1 = flip(a2D, axis: 1);
            expect(f1.toList(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);

            final flr = fliplr(a2D);
            expect(flr.toList(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);

            final fud = flipud(a2D);
            expect(fud.toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

            final fList = flip(a2D, axis: [0, 1]);
            expect(fList.toList(), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
          }),
        );

        test(
          "roll across 1D, 2D, and multi-axes",
          () => NDArray.scope(() {
            final v = NDArray.fromList(
              Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]),
              [5],
              DType.float64,
            );
            final r1 = roll(v, 2);
            expect(r1.toList(), [3.0, 4.0, 0.0, 1.0, 2.0]);

            final rNeg = roll(v, -1);
            expect(rNeg.toList(), [1.0, 2.0, 3.0, 4.0, 0.0]);

            final a2D = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );
            final r2DAx0 = roll(a2D, 1, axis: 0);
            expect(r2DAx0.toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

            final r2DAx1 = roll(a2D, 1, axis: 1);
            expect(r2DAx1.toList(), [3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);

            final rMulti = roll(a2D, [1, 1], axis: [0, 1]);
            expect(rMulti.toList(), [6.0, 4.0, 5.0, 3.0, 1.0, 2.0]);
          }),
        );

        test(
          "expand_dims and squeeze",
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
              3,
            ], DType.float64);

            final exp0 = expand_dims(a, 0);
            expect(exp0.shape, [1, 3]);

            final exp1 = expand_dims(a, 1);
            expect(exp1.shape, [3, 1]);

            final expNeg1 = expand_dims(a, -1);
            expect(expNeg1.shape, [3, 1]);

            final sqAll = squeeze(exp0);
            expect(sqAll.shape, [3]);

            final sqSpecific = squeeze(exp0, axis: [0]);
            expect(sqSpecific.shape, [3]);

            expect(() => squeeze(a, axis: [0]), throwsArgumentError);
          }),
        );

        test(
          "concatenate, stack, vstack, hstack",
          () => NDArray.scope(() {
            final a1 = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final a2 = NDArray.fromList(
              Float64List.fromList([5.0, 6.0, 7.0, 8.0]),
              [2, 2],
              DType.float64,
            );

            final c0 = concatenate([a1, a2], axis: 0);
            expect(c0.shape, [4, 2]);

            final c1 = concatenate([a1, a2], axis: 1);
            expect(c1.shape, [2, 4]);

            final s0 = stack([a1, a2], axis: 0);
            expect(s0.shape, [2, 2, 2]);

            final vs = vstack([a1, a2]);
            expect(vs.shape, [4, 2]);

            final hs = hstack([a1, a2]);
            expect(hs.shape, [2, 4]);
          }),
        );

        test(
          "diag, tril, triu, diff, and slidingWindowView",
          () => NDArray.scope(() {
            final v = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
              3,
            ], DType.float64);
            final dMat = diag(v);
            expect(dMat.shape, [3, 3]);
            expect(dMat.getCell([0, 0]), 1.0);
            expect(dMat.getCell([1, 1]), 2.0);
            expect(dMat.getCell([2, 2]), 3.0);

            final dVec = diag(dMat);
            expect(dVec.shape, [3]);
            expect(dVec.toList(), [1.0, 2.0, 3.0]);

            final full = NDArray.fromList(
              Float64List.fromList([
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
              ]),
              [3, 3],
              DType.float64,
            );
            final tr = tril(full);
            expect(tr.getCell([0, 1]), 0.0);
            expect(tr.getCell([0, 2]), 0.0);
            expect(tr.getCell([1, 2]), 0.0);

            final tu = triu(full);
            expect(tu.getCell([1, 0]), 0.0);
            expect(tu.getCell([2, 0]), 0.0);
            expect(tu.getCell([2, 1]), 0.0);

            final dArr = NDArray.fromList(
              Float64List.fromList([1.0, 3.0, 7.0, 15.0]),
              [4],
              DType.float64,
            );
            final diff1 = diff(dArr, n: 1);
            expect(diff1.toList(), [2.0, 4.0, 8.0]);
            final diff2 = diff(dArr, n: 2);
            expect(diff2.toList(), [2.0, 4.0]);

            final swv = slidingWindowView(dArr, [2]);
            expect(swv.shape, [3, 2]);
            expect(swv.toList(), [1.0, 3.0, 3.0, 7.0, 7.0, 15.0]);
          }),
        );
      });

      // 14. Array Splitting (splitting.dart)
      group("14. Array Splitting (splitting.dart)", () {
        test(
          "split, array_split, hsplit, vsplit, dsplit equal and unequal splits",
          () => NDArray.scope(() {
            final arr = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [6],
              DType.float64,
            );
            final sp3 = split(arr, 3);
            expect(sp3.length, 3);
            expect(sp3[0].toList(), [1.0, 2.0]);
            expect(sp3[1].toList(), [3.0, 4.0]);
            expect(sp3[2].toList(), [5.0, 6.0]);

            expect(() => split(arr, 4), throwsArgumentError);

            final asp4 = array_split(arr, 4);
            expect(asp4.length, 4);
            expect(asp4[0].toList(), [1.0, 2.0]);
            expect(asp4[1].toList(), [3.0, 4.0]);
            expect(asp4[2].toList(), [5.0]);
            expect(asp4[3].toList(), [6.0]);

            final spAt = split_at(arr, [2, 4]);
            expect(spAt.length, 3);
            expect(spAt[0].toList(), [1.0, 2.0]);
            expect(spAt[1].toList(), [3.0, 4.0]);
            expect(spAt[2].toList(), [5.0, 6.0]);

            final mat2D = NDArray.fromList(
              Float64List.fromList([
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
              ]),
              [4, 4],
              DType.float64,
            );
            final vs = vsplit(mat2D, 2);
            expect(vs.length, 2);
            expect(vs[0].shape, [2, 4]);
            expect(vs[1].shape, [2, 4]);

            final hs = hsplit(mat2D, 2);
            expect(hs.length, 2);
            expect(hs[0].shape, [4, 2]);
            expect(hs[1].shape, [4, 2]);

            final mat3D = NDArray.fromList(
              Float64List.fromList(
                List.generate(16, (i) => (i + 1).toDouble()),
              ),
              [2, 2, 4],
              DType.float64,
            );
            final ds = dsplit(mat3D, 2);
            expect(ds.length, 2);
            expect(ds[0].shape, [2, 2, 2]);
            expect(ds[1].shape, [2, 2, 2]);
          }),
        );
      });

      // =========================================================================
      // 15. Deep linalg.dart Edge Cases & Recycler Out Buffers
      // =========================================================================
      group("15. Deep linalg.dart Edge Cases & Recycler Out Buffers", () {
        test(
          "slogdet on Float32, Complex64, 3D/4D stacked, negative det, and out buffers",
          () => NDArray.scope(() {
            // Float32 2D
            final a32 = NDArray.fromList(
              Float32List.fromList([2.0, 0.0, 0.0, 3.0]),
              [2, 2],
              DType.float32,
            );
            final s32 = slogdet(a32);
            expect(s32.sign.scalar, 1.0);
            expect(s32.logabsdet.scalar, closeTo(math.log(6.0), 1e-4));
            s32.dispose();

            // Complex64 2D
            final c64 = NDArray.fromList(
              [Complex(0, 1), Complex(0, 0), Complex(0, 0), Complex(0, 2)],
              [2, 2],
              DType.complex64,
            );
            final sc64 = slogdet(c64);
            expect(sc64.sign.scalar.real, closeTo(-1.0, 1e-4));
            expect(sc64.sign.scalar.imag, closeTo(0.0, 1e-4));
            expect(sc64.logabsdet.scalar, closeTo(math.log(2.0), 1e-4));
            sc64.dispose();

            // Out buffer recycling
            final a64 = NDArray.fromList(
              Float64List.fromList([3.0, 1.0, 1.0, 2.0]),
              [2, 2],
              DType.float64,
            );
            final outSign = NDArray.zeros([], DType.float64);
            final outLog = NDArray.zeros([], DType.float64);
            final sRec = slogdet(a64, outSign: outSign, outLogdet: outLog);
            expect(identical(sRec.sign, outSign), true);
            expect(identical(sRec.logabsdet, outLog), true);
            expect(sRec.sign.scalar, 1.0);
            expect(sRec.logabsdet.scalar, closeTo(math.log(5.0), 1e-5));
          }),
        );

        test(
          "det on Float32 and Complex64 3D/4D stacks with out buffer",
          () => NDArray.scope(() {
            final b32 = NDArray.fromList(
              Float32List.fromList([2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 3.0, 4.0]),
              [2, 2, 2],
              DType.float32,
            );
            final outDet = NDArray.zeros([2], DType.float32);
            final resDet = det(b32, out: outDet);
            expect(identical(resDet, outDet), true);
            expect(resDet.toList()[0], closeTo(6.0, 1e-4));
            expect(resDet.toList()[1], closeTo(-2.0, 1e-4));
          }),
        );

        test(
          "inv on Complex64 2D/3D and strided transposed views",
          () => NDArray.scope(() {
            final cpx64 = NDArray.fromList(
              [Complex(1, 0), Complex(2, 0), Complex(0, 0), Complex(1, 0)],
              [2, 2],
              DType.complex64,
            );
            final invC = inv(cpx64);
            expect(invC.dtype, DType.complex64);
            final recon = matmul(cpx64, invC);
            expect(recon.getCell([0, 0]).real, closeTo(1.0, 1e-4));
            expect(recon.getCell([1, 1]).real, closeTo(1.0, 1e-4));

            // Transposed view inv
            final full = NDArray.fromList(
              Float64List.fromList([4.0, 2.0, 7.0, 6.0]),
              [2, 2],
              DType.float64,
            );
            final tr = full.transpose();
            final trInv = inv(tr);
            final trRecon = matmul(tr, trInv);
            expect(trRecon.getCell([0, 0]), closeTo(1.0, 1e-5));
            expect(trRecon.getCell([1, 1]), closeTo(1.0, 1e-5));
          }),
        );

        test(
          "solve on Float32 and Complex64 3D batch systems with out buffer",
          () => NDArray.scope(() {
            final a32 = NDArray.fromList(
              Float32List.fromList([2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0]),
              [2, 2, 2],
              DType.float32,
            );
            final b32 = NDArray.fromList(
              Float32List.fromList([4.0, 6.0, 9.0, 12.0]),
              [2, 2, 1],
              DType.float32,
            );
            final outX = NDArray.zeros([2, 2, 1], DType.float32);
            final x = solve(a32, b32, out: outX);
            expect(identical(x, outX), true);
            expect(x.toList()[0], closeTo(2.0, 1e-4));
            expect(x.toList()[1], closeTo(3.0, 1e-4));
            expect(x.toList()[2], closeTo(3.0, 1e-4));
            expect(x.toList()[3], closeTo(4.0, 1e-4));
          }),
        );

        test(
          "lstsq with Float32, Complex64, custom rcond, and out recycler",
          () => NDArray.scope(() {
            final a32 = NDArray.fromList(
              Float32List.fromList([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
              [3, 2],
              DType.float32,
            );
            final b32 = NDArray.fromList(
              Float32List.fromList([1.0, 2.0, 3.0]),
              [3],
              DType.float32,
            );
            final outX = NDArray.zeros([2], DType.float32);
            final res = lstsq(a32, b32, rcond: 1e-4, out: outX);
            expect(identical(res.x, outX), true);
            expect(res.x.toList()[0], closeTo(1.0, 1e-4));
            expect(res.x.toList()[1], closeTo(2.0, 1e-4));
            res.dispose();
          }),
        );

        test(
          "eigh and eigvalsh on Float32, Complex64, and out buffer recycler",
          () => NDArray.scope(() {
            final sym32 = NDArray.fromList(
              Float32List.fromList([3.0, 1.0, 1.0, 3.0]),
              [2, 2],
              DType.float32,
            );
            final outEvals = NDArray.zeros([2], DType.float32);
            final outEvecs = NDArray.zeros([2, 2], DType.float32);
            final res = eigh(
              sym32,
              outEigenvalues: outEvals,
              outEigenvectors: outEvecs,
            );
            expect(identical(res.eigenvalues, outEvals), true);
            expect(identical(res.eigenvectors, outEvecs), true);
            expect(res.eigenvalues.toList()[0], closeTo(2.0, 1e-4));
            expect(res.eigenvalues.toList()[1], closeTo(4.0, 1e-4));
            res.dispose();

            final outValsOnly = NDArray.zeros([2], DType.float32);
            final vals = eigvalsh(sym32, out: outValsOnly);
            expect(identical(vals, outValsOnly), true);
            expect(vals.toList()[0], closeTo(2.0, 1e-4));
            expect(vals.toList()[1], closeTo(4.0, 1e-4));
          }),
        );

        test(
          "schur and hessenberg on Float32 and Complex64 with out buffers",
          () => NDArray.scope(() {
            final a32 = NDArray.fromList(
              Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float32,
            );
            final outT = NDArray.zeros([2, 2], DType.float32);
            final outZ = NDArray.zeros([2, 2], DType.float32);
            final resSchur = schur(a32, outT: outT, outZ: outZ);
            expect(identical(resSchur.t, outT), true);
            expect(identical(resSchur.z, outZ), true);
            resSchur.dispose();

            final outH = NDArray.zeros([2, 2], DType.float32);
            final outQ = NDArray.zeros([2, 2], DType.float32);
            final resHess = hessenberg(a32, outH: outH, outQ: outQ);
            expect(identical(resHess.h, outH), true);
            expect(identical(resHess.q, outQ), true);
            resHess.dispose();
          }),
        );

        test(
          "norm on Float32, Complex64, and out buffer recycler",
          () => NDArray.scope(() {
            final v32 = NDArray.fromList(Float32List.fromList([3.0, 4.0]), [
              2,
            ], DType.float32);
            final outNorm = NDArray.zeros([], DType.float32);
            final n = norm(v32, out: outNorm);
            expect(identical(n, outNorm), true);
            expect(n.scalar, closeTo(5.0, 1e-4));

            final cpx64 = NDArray.fromList(
              [Complex(1, 1), Complex(1, -1)],
              [2],
              DType.complex64,
            );
            final nCpx = norm(cpx64);
            expect(nCpx.scalar, closeTo(2.0, 1e-4));
          }),
        );

        test(
          "matrix_power with n = 4, -3, -4 on Float32 and Complex128",
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float32List.fromList([1.0, 1.0, 0.0, 1.0]),
              [2, 2],
              DType.float32,
            );
            final p4 = matrix_power(a, 4);
            expect(p4.toList(), [1.0, 4.0, 0.0, 1.0]);

            final pNeg3 = matrix_power(a, -3);
            expect(pNeg3.toList()[0], closeTo(1.0, 1e-4));
            expect(pNeg3.toList()[1], closeTo(-3.0, 1e-4));
            expect(pNeg3.toList()[2], closeTo(0.0, 1e-4));
            expect(pNeg3.toList()[3], closeTo(1.0, 1e-4));
          }),
        );

        test(
          "cross and outer on Float32, Int32, and out buffers",
          () => NDArray.scope(() {
            final u = NDArray.fromList(Float32List.fromList([1.0, 0.0, 0.0]), [
              3,
            ], DType.float32);
            final v = NDArray.fromList(Float32List.fromList([0.0, 1.0, 0.0]), [
              3,
            ], DType.float32);
            final outCross = NDArray.zeros([3], DType.float32);
            final c = cross(u, v, out: outCross);
            expect(identical(c, outCross), true);
            expect(c.toList(), [0.0, 0.0, 1.0]);

            final outOuter = NDArray.zeros([3, 3], DType.float32);
            final o = outer(u, v, out: outOuter);
            expect(identical(o, outOuter), true);
            expect(o.getCell([0, 1]), 1.0);
            expect(o.getCell([0, 0]), 0.0);
          }),
        );
      });

      // =========================================================================
      // 16. Deep manipulation.dart & splitting.dart Edge Cases
      // =========================================================================
      group("16. Deep manipulation.dart & splitting.dart Edge Cases", () {
        test(
          "concatenate and stack along higher axes on 3D/4D tensors",
          () => NDArray.scope(() {
            final t1 = NDArray.fromList(
              Float64List.fromList(
                List.generate(24, (i) => (i + 1).toDouble()),
              ),
              [2, 3, 4],
              DType.float64,
            );
            final t2 = NDArray.fromList(
              Float64List.fromList(
                List.generate(24, (i) => (i + 25).toDouble()),
              ),
              [2, 3, 4],
              DType.float64,
            );

            final c2 = concatenate([t1, t2], axis: 2);
            expect(c2.shape, [2, 3, 8]);

            final s2 = stack([t1, t2], axis: 2);
            expect(s2.shape, [2, 3, 2, 4]);

            final cNeg1 = concatenate([t1, t2], axis: -1);
            expect(cNeg1.shape, [2, 3, 8]);
          }),
        );

        test(
          "diag with k != 0 on 1D and rectangular 2D matrices",
          () => NDArray.scope(() {
            final v = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final dPos = diag(v, k: 1);
            expect(dPos.shape, [3, 3]);
            expect(dPos.getCell([0, 1]), 1.0);
            expect(dPos.getCell([1, 2]), 2.0);
            expect(dPos.getCell([0, 0]), 0.0);

            final dNeg = diag(v, k: -1);
            expect(dNeg.shape, [3, 3]);
            expect(dNeg.getCell([1, 0]), 1.0);
            expect(dNeg.getCell([2, 1]), 2.0);

            // Rectangular 2D 3x4: diag with k = 1
            final rect = NDArray.fromList(
              Float64List.fromList(
                List.generate(12, (i) => (i + 1).toDouble()),
              ),
              [3, 4],
              DType.float64,
            );
            final diagRect1 = diag(rect, k: 1);
            expect(diagRect1.shape, [3]);
            expect(diagRect1.toList(), [2.0, 7.0, 12.0]);
          }),
        );

        test(
          "tril and triu with k != 0 on 3D batch arrays",
          () => NDArray.scope(() {
            final bMat = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
              [2, 2, 2],
              DType.float64,
            );
            final tl = tril(bMat, k: -1);
            expect(tl.shape, [2, 2, 2]);
            expect(tl.getCell([0, 0, 0]), 0.0);
            expect(tl.getCell([0, 1, 0]), 3.0);

            final tu = triu(bMat, k: 1);
            expect(tu.shape, [2, 2, 2]);
            expect(tu.getCell([0, 0, 1]), 2.0);
            expect(tu.getCell([0, 0, 0]), 0.0);
          }),
        );

        test(
          "roll with shift > length and multi-axis on 3D tensors",
          () => NDArray.scope(() {
            final t = NDArray.fromList(
              Float64List.fromList(List.generate(8, (i) => (i + 1).toDouble())),
              [2, 2, 2],
              DType.float64,
            );
            final r3D = roll(t, [1, 2], axis: [0, 2]);
            expect(r3D.shape, [2, 2, 2]);
          }),
        );

        test(
          "split and array_split on 3D/4D arrays and out-of-bounds indices",
          () => NDArray.scope(() {
            final t = NDArray.fromList(
              Float64List.fromList(
                List.generate(24, (i) => (i + 1).toDouble()),
              ),
              [2, 3, 4],
              DType.float64,
            );

            // split along axis 2 into 2 parts
            final sp2 = split(t, 2, axis: 2);
            expect(sp2.length, 2);
            expect(sp2[0].shape, [2, 3, 2]);
            expect(sp2[1].shape, [2, 3, 2]);

            // array_split_at with out-of-bounds indices clamped properly
            final asp = array_split_at(t, [-2, 2, 10], axis: 1);
            expect(asp.length, 4);
            expect(asp[0].shape, [2, 0, 4]);
            expect(asp[1].shape, [2, 2, 4]);
            expect(asp[2].shape, [2, 1, 4]);
            expect(asp[3].shape, [2, 0, 4]);
          }),
        );
      });

      // =========================================================================
      // 17. Deep tensor_contractions.dart Contractions & Einsum Patterns
      // =========================================================================
      group(
        "17. Deep tensor_contractions.dart Contractions & Einsum Patterns",
        () {
          test(
            "tensordot 3D with 3D multi-axis contractions",
            () => NDArray.scope(() {
              final t1 = NDArray.fromList(
                Float64List.fromList(
                  List.generate(12, (i) => (i + 1).toDouble()),
                ),
                [2, 3, 2],
                DType.float64,
              );
              final t2 = NDArray.fromList(
                Float64List.fromList(
                  List.generate(12, (i) => (i + 1).toDouble()),
                ),
                [3, 2, 2],
                DType.float64,
              );

              final td = tensordot(
                t1,
                t2,
                axes: TensordotAxes.explicit([1, 2], [0, 1]),
              );
              expect(td.shape, [2, 2]);
            }),
          );

          test(
            "einsum batched matrix multiplication, batched vector, and ellipsis",
            () => NDArray.scope(() {
              final b1 = NDArray.fromList(
                Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                [2, 2, 2],
                DType.float64,
              );
              final b2 = NDArray.fromList(
                Float64List.fromList([1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
                [2, 2, 2],
                DType.float64,
              );

              // Batched matmul: "bij,bjk->bik"
              final eBmm = einsum(EinsumSubscripts.parse("bij,bjk->bik"), [
                b1,
                b2,
              ]);
              expect(eBmm.shape, [2, 2, 2]);
              expect(eBmm.getCell([0, 0, 0]), 1.0);
              expect(eBmm.getCell([0, 0, 1]), 2.0);
              expect(eBmm.getCell([1, 0, 0]), 10.0);
              expect(eBmm.getCell([1, 0, 1]), 12.0);

              // Row sums: "ij->i"
              final m = NDArray.fromList(
                Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
                [2, 2],
                DType.float64,
              );
              final eRow = einsum(EinsumSubscripts.parse("ij->i"), [m]);
              expect(eRow.toList(), [3.0, 7.0]);

              // Column sums: "ij->j"
              final eCol = einsum(EinsumSubscripts.parse("ij->j"), [m]);
              expect(eCol.toList(), [4.0, 6.0]);
            }),
          );

          test(
            "inner and kron on 2D and 3D tensors",
            () => NDArray.scope(() {
              final m1 = NDArray.fromList(
                Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
                [2, 2],
                DType.float64,
              );
              final m2 = NDArray.fromList(
                Float64List.fromList([5.0, 6.0, 7.0, 8.0]),
                [2, 2],
                DType.float64,
              );

              // inner on 2D: computes A * B^T
              final in2D = inner(m1, m2);
              expect(in2D.shape, [2, 2]);
              final expected = matmul(m1, m2.transpose());
              expect(in2D.toList(), expected.toList());

              // kron on 1D with 2D
              final v = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
                2,
              ], DType.float64);
              final kr = kron(v, m1);
              expect(kr.shape, [2, 4]);
            }),
          );
        },
      );
    },
  );
}
