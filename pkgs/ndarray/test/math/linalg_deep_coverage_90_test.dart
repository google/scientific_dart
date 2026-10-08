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
  group('Linalg Deep Coverage 90% Suite', () {
    // ------------------------------------------------------------------------
    // Group 1: Matrix Multiplication & Multi-Dot Dynamic Programming Chains
    // ------------------------------------------------------------------------
    group('1. matmul and multi_dot', () {
      test(
        'matmul 1D dot product across all 4 floating/complex types',
        () => NDArray.scope(() {
          final v1 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final v2 = NDArray.fromList(Float64List.fromList([4.0, 5.0, 6.0]), [
            3,
          ], DType.float64);
          final dot64 = matmul(v1, v2);
          expect(dot64.shape, <int>[]);
          expect(dot64.scalar, closeTo(32.0, 1e-6));

          final vf1 = NDArray.fromList(Float32List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float32);
          final vf2 = NDArray.fromList(Float32List.fromList([4.0, 5.0, 6.0]), [
            3,
          ], DType.float32);
          final dot32 = matmul(vf1, vf2);
          expect(dot32.shape, <int>[]);
          expect(dot32.scalar, closeTo(32.0, 1e-5));

          final c1 = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            DType.complex128,
          );
          final c2 = NDArray.fromList(
            [Complex(5.0, 6.0), Complex(7.0, 8.0)],
            [2],
            DType.complex128,
          );
          final dotC128 = matmul(c1, c2);
          expect(dotC128.shape, <int>[]);
          expect(dotC128.scalar.real, closeTo(-18.0, 1e-6));
          expect(dotC128.scalar.imag, closeTo(68.0, 1e-6));

          final cf1 = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            DType.complex64,
          );
          final cf2 = NDArray.fromList(
            [Complex(5.0, 6.0), Complex(7.0, 8.0)],
            [2],
            DType.complex64,
          );
          final dotC64 = matmul(cf1, cf2);
          expect(dotC64.shape, <int>[]);
          expect(dotC64.scalar.real, closeTo(-18.0, 1e-4));
          expect(dotC64.scalar.imag, closeTo(68.0, 1e-4));
        }),
      );

      test(
        'matmul vector-matrix (1D x 2D) and matrix-vector (2D x 1D)',
        () => NDArray.scope(() {
          final vec = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
            2,
          ], DType.float64);
          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            DType.float64,
          );

          final res1 = matmul(vec, mat);
          expect(res1.shape, [3]);
          expect(res1.toList(), [
            closeTo(9.0, 1e-6),
            closeTo(12.0, 1e-6),
            closeTo(15.0, 1e-6),
          ]);

          final vec3 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final res2 = matmul(mat, vec3);
          expect(res2.shape, [2]);
          expect(res2.toList(), [closeTo(14.0, 1e-6), closeTo(32.0, 1e-6)]);
        }),
      );

      test(
        'matmul non-contiguous views and out buffer recycling',
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
          final outBuf = NDArray.zeros([2, 2], DType.float64);

          final res = matmul(m1.transpose(), m2, out: outBuf);
          expect(identical(res, outBuf), isTrue);
          // [[1, 3], [2, 4]] x [[5, 6], [7, 8]] = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]] = [[26, 30], [38, 44]]
          expect(res.toList(), [26.0, 30.0, 38.0, 44.0]);

          // Incompatible out buffer shape throws error
          final badOut = NDArray.zeros([3, 3], DType.float64);
          expect(() => matmul(m1, m2, out: badOut), throwsArgumentError);
        }),
      );

      test(
        'matmul batch broadcasting stack: [2, 1, 2, 2] x [1, 3, 2, 2] -> [2, 3, 2, 2]',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
            [2, 1, 2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
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
            ]),
            [1, 3, 2, 2],
            DType.float64,
          );
          final c = matmul(a, b);
          expect(c.shape, [2, 3, 2, 2]);
          expect(c.dtype, DType.float64);
        }),
      );

      test(
        'multi_dot with 3, 4, 5 matrix chains of mismatched dimensions',
        () => NDArray.scope(() {
          final m1 = NDArray.ones([2, 3], DType.float64);
          final m2 = NDArray.ones([3, 4], DType.float64);
          final m3 = NDArray.ones([4, 2], DType.float64);
          final res3 = multi_dot([m1, m2, m3]);
          expect(res3.shape, [2, 2]);
          for (final val in res3.toList()) {
            expect(val, closeTo(12.0, 1e-6));
          }

          final a1 = NDArray.ones([2, 5], DType.float64);
          final a2 = NDArray.ones([5, 2], DType.float64);
          final a3 = NDArray.ones([2, 4], DType.float64);
          final a4 = NDArray.ones([4, 3], DType.float64);
          final res4 = multi_dot([a1, a2, a3, a4]);
          expect(res4.shape, [2, 3]);
          for (final val in res4.toList()) {
            expect(val, closeTo(40.0, 1e-6));
          }

          final b1 = NDArray.ones([2, 3], DType.float64);
          final b2 = NDArray.ones([3, 2], DType.float64);
          final b3 = NDArray.ones([2, 4], DType.float64);
          final b4 = NDArray.ones([4, 2], DType.float64);
          final b5 = NDArray.ones([2, 2], DType.float64);
          final res5 = multi_dot([b1, b2, b3, b4, b5]);
          expect(res5.shape, [2, 2]);
          for (final val in res5.toList()) {
            expect(val, closeTo(48.0, 1e-6));
          }
        }),
      );

      test(
        'multi_dot with 1D vector at beginning and end',
        () => NDArray.scope(() {
          final v = NDArray.ones([3], DType.float64);
          final m1 = NDArray.ones([3, 4], DType.float64);
          final m2 = NDArray.ones([4, 2], DType.float64);
          final resStart = multi_dot([v, m1, m2]);
          expect(resStart.shape, [2]);
          for (final val in resStart.toList()) {
            expect(val, closeTo(12.0, 1e-6));
          }

          final n1 = NDArray.ones([2, 3], DType.float64);
          final n2 = NDArray.ones([3, 4], DType.float64);
          final vEnd = NDArray.ones([4], DType.float64);
          final resEnd = multi_dot([n1, n2, vEnd]);
          expect(resEnd.shape, [2]);
          for (final val in resEnd.toList()) {
            expect(val, closeTo(12.0, 1e-6));
          }

          final resScalar = multi_dot([v, n2, vEnd]);
          expect(resScalar.shape, <int>[]);
          expect(resScalar.scalar, closeTo(12.0, 1e-6));
        }),
      );

      test(
        'multi_dot error conditions',
        () => NDArray.scope(() {
          expect(() => multi_dot<DTypeTag>([]), throwsArgumentError);
          final m = NDArray.ones([2, 2], DType.float64);
          expect(() => multi_dot<DTypeTag>([m]), throwsArgumentError);
          final m3d = NDArray.ones([2, 2, 2], DType.float64);
          expect(() => multi_dot<DTypeTag>([m, m3d]), throwsArgumentError);
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 2: Matrix Inversion, Pseudo-Inverse & Matrix Power
    // ------------------------------------------------------------------------
    group('2. inv, pinv and matrix_power', () {
      test(
        'inv for Float64, Float32, Complex128, and Complex64',
        () => NDArray.scope(() {
          // Float64
          final a64 = NDArray.fromList(
            Float64List.fromList([4.0, 7.0, 2.0, 6.0]),
            [2, 2],
            DType.float64,
          );
          final a64Inv = inv(a64);
          expect(a64Inv.shape, [2, 2]);
          final id64 = matmul(a64, a64Inv);
          expect(id64.getCell([0, 0]), closeTo(1.0, 1e-5));
          expect(id64.getCell([0, 1]), closeTo(0.0, 1e-5));
          expect(id64.getCell([1, 0]), closeTo(0.0, 1e-5));
          expect(id64.getCell([1, 1]), closeTo(1.0, 1e-5));

          // Float32 with out buffer
          final a32 = NDArray.fromList(
            Float32List.fromList([4.0, 7.0, 2.0, 6.0]),
            [2, 2],
            DType.float32,
          );
          final out32 = NDArray.zeros([2, 2], DType.float32);
          final a32Inv = inv(a32, out: out32);
          expect(identical(a32Inv, out32), isTrue);

          // Complex128
          final a = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(3.0, 4.0),
              Complex(5.0, 6.0),
              Complex(7.0, 8.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final aInv = inv(a);
          expect(aInv.shape, [2, 2]);
          expect(aInv.dtype, DType.complex128);

          final ident = matmul(a, aInv);
          expect(ident.getCell([0, 0]).real, closeTo(1.0, 1e-5));
          expect(ident.getCell([0, 0]).imag, closeTo(0.0, 1e-5));
          expect(ident.getCell([0, 1]).real, closeTo(0.0, 1e-5));
          expect(ident.getCell([0, 1]).imag, closeTo(0.0, 1e-5));
          expect(ident.getCell([1, 0]).real, closeTo(0.0, 1e-5));
          expect(ident.getCell([1, 0]).imag, closeTo(0.0, 1e-5));
          expect(ident.getCell([1, 1]).real, closeTo(1.0, 1e-5));
          expect(ident.getCell([1, 1]).imag, closeTo(0.0, 1e-5));

          // Singular matrix throws SingularMatrixException
          final sing = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          expect(() => inv(sing), throwsA(isA<SingularMatrixException>()));
        }),
      );

      test(
        'pinv with tall [4, 2], wide [2, 4], custom rcond, and Complex64',
        () => NDArray.scope(() {
          final tall = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0]),
            [4, 2],
            DType.float64,
          );
          final pTall = pinv(tall);
          expect(pTall.shape, [2, 4]);
          expect(pTall.dtype, DType.float64);
          final pA = matmul(pTall, tall);
          expect(pA.getCell([0, 0]), closeTo(1.0, 1e-5));
          expect(pA.getCell([0, 1]), closeTo(0.0, 1e-5));
          expect(pA.getCell([1, 0]), closeTo(0.0, 1e-5));
          expect(pA.getCell([1, 1]), closeTo(1.0, 1e-5));

          final wide = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            [2, 4],
            DType.float32,
          );
          final pWide = pinv(wide, rcond: 1e-3);
          expect(pWide.shape, [4, 2]);
          expect(pWide.dtype, DType.float32);

          final cpx = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(0.0, 1.0),
              Complex(1.0, 0.0),
              Complex(2.0, 1.0),
              Complex(0.0, 0.0),
              Complex(1.0, 2.0),
            ],
            [3, 2],
            DType.complex64,
          );
          final pCpx = pinv(cpx);
          expect(pCpx.shape, [2, 3]);
          expect(pCpx.dtype, DType.complex64);
        }),
      );

      test(
        'matrix_power with powers: 0, 1, 2, 4, -1, -2, -3',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );

          final p0 = matrix_power(a, 0);
          expect(p0.toList(), [1.0, 0.0, 0.0, 1.0]);

          final p1 = matrix_power(a, 1);
          expect(p1.toList(), a.toList());

          final p2 = matrix_power(a, 2);
          expect(p2.toList(), [7.0, 10.0, 15.0, 22.0]);

          final p4 = matrix_power(a, 4);
          final p2_2 = matmul(p2, p2);
          for (var i = 0; i < 4; i++) {
            expect(p4.toList()[i], closeTo(p2_2.toList()[i], 1e-6));
          }

          final pNeg1 = matrix_power(a, -1);
          final aInv = inv(a);
          for (var i = 0; i < 4; i++) {
            expect(pNeg1.toList()[i], closeTo(aInv.toList()[i], 1e-6));
          }

          final pNeg2 = matrix_power(a, -2);
          final aInv2 = matmul(aInv, aInv);
          for (var i = 0; i < 4; i++) {
            expect(pNeg2.toList()[i], closeTo(aInv2.toList()[i], 1e-6));
          }

          final pNeg3 = matrix_power(a, -3);
          final aInv3 = matmul(aInv2, aInv);
          for (var i = 0; i < 4; i++) {
            expect(pNeg3.toList()[i], closeTo(aInv3.toList()[i], 1e-6));
          }
        }),
      );

      test(
        'matrix_power on 3D and complex matrices',
        () => NDArray.scope(() {
          final batch = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0]),
            [2, 2, 2],
            DType.float64,
          );
          expect(() => matrix_power(batch, 3), throwsArgumentError);

          final cMat = NDArray.fromList(
            [
              Complex(0.0, 1.0),
              Complex(0.0, 0.0),
              Complex(0.0, 0.0),
              Complex(0.0, 1.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final cPow2 = matrix_power(cMat, 2);
          expect(cPow2.getCell([0, 0]).real, closeTo(-1.0, 1e-6));
          expect(cPow2.getCell([0, 0]).imag, closeTo(0.0, 1e-6));
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 3: Determinant & Slogdet Across All Shapes and DTypes
    // ------------------------------------------------------------------------
    group('3. det and slogdet', () {
      test(
        'det for 2D, 3D, 4D batch matrices across Float32, Float64, Complex64, Complex128',
        () => NDArray.scope(() {
          final a64 = NDArray.fromList(
            Float64List.fromList([3.0, 8.0, 4.0, 6.0]),
            [2, 2],
            DType.float64,
          );
          final d64 = det(a64);
          expect(d64.shape, <int>[]);
          expect(d64.scalar, closeTo(-14.0, 1e-6));

          final a32 = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0]),
            [2, 2, 2],
            DType.float32,
          );
          final d32 = det(a32);
          expect(d32.shape, [2]);
          expect(d32.toList()[0], closeTo(-2.0, 1e-5));
          expect(d32.toList()[1], closeTo(4.0, 1e-5));

          final cpx4d = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(0.0, 0.0),
              Complex(0.0, 0.0),
              Complex(2.0, 0.0),
              Complex(3.0, 0.0),
              Complex(0.0, 0.0),
              Complex(0.0, 0.0),
              Complex(4.0, 0.0),
            ],
            [1, 2, 2, 2],
            DType.complex128,
          );
          final dCpx4d = det(cpx4d);
          expect(dCpx4d.shape, [1, 2]);
          expect(dCpx4d.toList()[0].real, closeTo(2.0, 1e-6));
          expect(dCpx4d.toList()[1].real, closeTo(12.0, 1e-6));
        }),
      );

      test(
        'slogdet for positive, negative, singular, complex, and batch matrices',
        () => NDArray.scope(() {
          final posMat = NDArray.fromList(
            Float64List.fromList([2.0, 0.0, 0.0, 3.0]),
            [2, 2],
            DType.float64,
          );
          final sPos = slogdet(posMat);
          expect(sPos.sign.scalar, closeTo(1.0, 1e-6));
          expect(sPos.logabsdet.scalar, closeTo(math.log(6.0), 1e-6));
          sPos.dispose();

          final negMat = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final sNeg = slogdet(negMat);
          expect(sNeg.sign.scalar, closeTo(-1.0, 1e-6));
          expect(sNeg.logabsdet.scalar, closeTo(math.log(2.0), 1e-6));
          sNeg.dispose();

          final singMat = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final sSing = slogdet(singMat);
          expect(sSing.sign.scalar, closeTo(0.0, 1e-6));
          expect(sSing.logabsdet.scalar, equals(double.negativeInfinity));
          sSing.dispose();

          final cMat = NDArray.fromList(
            [
              Complex(0.0, 1.0),
              Complex(0.0, 0.0),
              Complex(0.0, 0.0),
              Complex(0.0, 2.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final sCpx = slogdet(cMat);
          expect(sCpx.sign.scalar.real, closeTo(-1.0, 1e-6));
          expect(sCpx.sign.scalar.imag, closeTo(0.0, 1e-6));
          expect(sCpx.logabsdet.scalar, closeTo(math.log(2.0), 1e-6));
          sCpx.dispose();
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 4: Linear System Solvers (solve and lstsq)
    // ------------------------------------------------------------------------
    group('4. solve and lstsq', () {
      test(
        'solve with 1D RHS and 2D RHS across Float64, Float32, Complex128',
        () => NDArray.scope(() {
          final a64 = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float64,
          );
          final b1d = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
          ], DType.float64);
          final x1d = solve(a64, b1d);
          expect(x1d.shape, [2]);
          expect(x1d.toList()[0], closeTo(2.0, 1e-6));
          expect(x1d.toList()[1], closeTo(3.0, 1e-6));

          final b2d = NDArray.fromList(
            Float64List.fromList([9.0, 3.0, 8.0, 1.0]),
            [2, 2],
            DType.float64,
          );
          final x2d = solve(a64, b2d);
          expect(x2d.shape, [2, 2]);
          expect(x2d.getCell([0, 0]), closeTo(2.0, 1e-6));
          expect(x2d.getCell([1, 0]), closeTo(3.0, 1e-6));
          expect(x2d.getCell([0, 1]), closeTo(1.0, 1e-6));
          expect(x2d.getCell([1, 1]), closeTo(0.0, 1e-6));

          final aCpx = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(0.0, 1.0),
              Complex(1.0, -1.0),
              Complex(2.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final bCpx = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 1.0)],
            [2],
            DType.complex128,
          );
          final xCpx = solve(aCpx, bCpx);
          expect(xCpx.shape, [2]);
          expect(xCpx.dtype, DType.complex128);
          final ax = matmul(aCpx, xCpx);
          expect(ax.toList()[0].real, closeTo(1.0, 1e-5));
          expect(ax.toList()[0].imag, closeTo(2.0, 1e-5));
          expect(ax.toList()[1].real, closeTo(3.0, 1e-5));
          expect(ax.toList()[1].imag, closeTo(1.0, 1e-5));
        }),
      );

      test(
        'lstsq with overdetermined, underdetermined, singular, and custom rcond',
        () => NDArray.scope(() {
          final aOver = NDArray.fromList(
            Float64List.fromList([1.0, 1.0, 1.0, 2.0, 1.0, 3.0]),
            [3, 2],
            DType.float64,
          );
          final bOver = NDArray.fromList(
            Float64List.fromList([6.0, 5.0, 7.0]),
            [3],
            DType.float64,
          );
          final resOver = lstsq(aOver, bOver);
          expect(resOver.x.shape, [2]);
          expect(resOver.rank, equals(2));
          expect(resOver.residuals.size, equals(1));
          expect(resOver.s.shape, [2]);
          resOver.dispose();

          final aUnder = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            DType.float64,
          );
          final bUnder = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
            2,
          ], DType.float64);
          final resUnder = lstsq(aUnder, bUnder);
          expect(resUnder.x.shape, [3]);
          expect(resUnder.rank, equals(2));
          expect(resUnder.residuals.size, equals(0));
          resUnder.dispose();

          final aCpx = NDArray.fromList(
            [
              Complex(1.0, 0.0),
              Complex(0.0, 1.0),
              Complex(0.0, 1.0),
              Complex(1.0, 0.0),
              Complex(1.0, 1.0),
              Complex(1.0, -1.0),
            ],
            [3, 2],
            DType.complex64,
          );
          final bCpx = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 0.0), Complex(0.0, 2.0)],
            [3],
            DType.complex64,
          );
          final resCpx = lstsq(aCpx, bCpx, rcond: 1e-4);
          expect(resCpx.x.shape, [2]);
          expect(resCpx.x.dtype, DType.complex64);
          resCpx.dispose();
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 5: QR, Cholesky, and SVD Decompositions
    // ------------------------------------------------------------------------
    group('5. qr, cholesky, and svd', () {
      test(
        'qr decomposition for square, tall, wide, batch, and all 4 dtypes',
        () => NDArray.scope(() {
          final tall = NDArray.fromList(
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
          final resQr = qr(tall);
          expect(resQr.q.shape, [3, 3]);
          expect(resQr.r.shape, [3, 3]);
          expect(resQr.q.dtype, DType.float64);
          expect(resQr.r.dtype, DType.float64);

          final qrRecon = matmul(resQr.q, resQr.r);
          for (var i = 0; i < 9; i++) {
            expect(qrRecon.toList()[i], closeTo(tall.toList()[i], 1e-5));
          }

          final qtQ = matmul(resQr.q.transpose(), resQr.q);
          for (var r = 0; r < 3; r++) {
            for (var c = 0; c < 3; c++) {
              expect(qtQ.getCell([r, c]), closeTo(r == c ? 1.0 : 0.0, 1e-5));
            }
          }
          resQr.dispose();

          final batch = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            [2, 2, 2],
            DType.float32,
          );
          final batchQr = qr(batch);
          expect(batchQr.q.shape, [2, 2, 2]);
          expect(batchQr.r.shape, [2, 2, 2]);
          batchQr.dispose();

          final cpxMat = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(3.0, 4.0),
              Complex(5.0, 6.0),
              Complex(7.0, 8.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final cpxQr = qr(cpxMat);
          expect(cpxQr.q.shape, [2, 2]);
          expect(cpxQr.r.shape, [2, 2]);
          expect(cpxQr.q.dtype, DType.complex128);
          cpxQr.dispose();
        }),
      );

      test(
        'cholesky decomposition with lower and upper triangular forms',
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
          final L = cholesky(a);
          expect(L.shape, [3, 3]);
          expect(L.dtype, DType.float64);

          final recon = matmul(L, L.transpose());
          for (var i = 0; i < 9; i++) {
            expect(recon.toList()[i], closeTo(a.toList()[i], 1e-5));
          }

          final batchA = NDArray.fromList(
            Float32List.fromList([4.0, 2.0, 2.0, 5.0, 9.0, 3.0, 3.0, 10.0]),
            [2, 2, 2],
            DType.float32,
          );
          final batchL = cholesky(batchA);
          expect(batchL.shape, [2, 2, 2]);
          expect(batchL.dtype, DType.float32);

          final nonPosDef = NDArray.fromList(
            Float64List.fromList([-1.0, 0.0, 0.0, -1.0]),
            [2, 2],
            DType.float64,
          );
          expect(() => cholesky(nonPosDef), throwsA(anything));
        }),
      );

      test(
        'svd decomposition verification',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [3, 2],
            DType.float64,
          );

          final s1 = svd(a);
          expect(s1.u.shape, [3, 3]);
          expect(s1.s.shape, [2]);
          expect(s1.vh.shape, [2, 2]);
          expect(s1.s.toList()[0], greaterThanOrEqualTo(s1.s.toList()[1]));
          s1.dispose();
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 6: Eigensystems, Schur & Hessenberg Reductions
    // ------------------------------------------------------------------------
    group('6. eig, eigh, eigvals, eigvalsh, schur, hessenberg', () {
      test(
        'eig and eigvals for general non-symmetric matrices with complex eigenvalues',
        () => NDArray.scope(() {
          final rot = NDArray.fromList(
            Float64List.fromList([0.0, -1.0, 1.0, 0.0]),
            [2, 2],
            DType.float64,
          );
          final resEig = eig(rot);
          expect(resEig.eigenvalues.shape, [2]);
          expect(resEig.eigenvectors.shape, [2, 2]);
          expect(resEig.eigenvalues.dtype, DType.complex128);

          final vals = resEig.eigenvalues.toList();
          expect(vals.any((c) => (c.imag - 1.0).abs() < 1e-5), isTrue);
          expect(vals.any((c) => (c.imag + 1.0).abs() < 1e-5), isTrue);
          resEig.dispose();

          final justVals = eigvals(rot);
          expect(justVals.shape, [2]);
          expect(justVals.dtype, DType.complex128);
        }),
      );

      test(
        'eigh and eigvalsh for symmetric / Hermitian matrices with upper and lower options',
        () => NDArray.scope(() {
          final sym = NDArray.fromList(
            Float64List.fromList([2.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float64,
          );

          final resLower = eigh(sym, uplo: MatrixTriangle.lower);
          expect(resLower.eigenvalues.shape, [2]);
          expect(resLower.eigenvectors.shape, [2, 2]);
          expect(resLower.eigenvalues.toList()[0], closeTo(1.0, 1e-5));
          expect(resLower.eigenvalues.toList()[1], closeTo(3.0, 1e-5));
          resLower.dispose();

          final resUpper = eigh(sym, uplo: MatrixTriangle.upper);
          expect(resUpper.eigenvalues.toList()[0], closeTo(1.0, 1e-5));
          expect(resUpper.eigenvalues.toList()[1], closeTo(3.0, 1e-5));
          resUpper.dispose();

          final justValsh = eigvalsh(sym);
          expect(justValsh.shape, [2]);
          expect(justValsh.toList()[0], closeTo(1.0, 1e-5));
          expect(justValsh.toList()[1], closeTo(3.0, 1e-5));
        }),
      );

      test(
        'schur and hessenberg reductions',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            [3, 3],
            DType.float64,
          );

          final resSchur = schur(a);
          expect(resSchur.t.shape, [3, 3]);
          expect(resSchur.z.shape, [3, 3]);
          resSchur.dispose();

          final resHess = hessenberg(a);
          expect(resHess.h.shape, [3, 3]);
          expect(resHess.q.shape, [3, 3]);
          expect(resHess.h.getCell([2, 0]), closeTo(0.0, 1e-5));
          resHess.dispose();
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 7: Vector and Matrix Norms & Condition Numbers
    // ------------------------------------------------------------------------
    group('7. norm and conditioning', () {
      test(
        'vector norms: 0, 1, 2, 3, inf, -inf, -1, -2 across real and complex vectors',
        () => NDArray.scope(() {
          final v = NDArray.fromList(Float64List.fromList([3.0, -4.0, 0.0]), [
            3,
          ], DType.float64);

          final n0 = norm(v, ord: 0);
          expect(n0.scalar, equals(2.0));

          final n1 = norm(v, ord: 1);
          expect(n1.scalar, closeTo(7.0, 1e-6));

          final n2 = norm(v, ord: 2);
          expect(n2.scalar, closeTo(5.0, 1e-6));

          final nInf = norm(v, ord: double.infinity);
          expect(nInf.scalar, closeTo(4.0, 1e-6));

          final nNegInf = norm(v, ord: -double.infinity);
          expect(nNegInf.scalar, closeTo(0.0, 1e-6));

          final vCpx = NDArray.fromList(
            [Complex(3.0, 4.0), Complex(0.0, 0.0)],
            [2],
            DType.complex128,
          );
          final nCpx2 = norm(vCpx, ord: 2);
          expect(nCpx2.scalar, closeTo(5.0, 1e-6));
        }),
      );

      test(
        'matrix norms: fro, nuc, 1, -1, 2, -2, inf, -inf with axis and keepdims',
        () => NDArray.scope(() {
          final m = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );

          final nFro = norm(m, ord: 'fro');
          expect(nFro.scalar, closeTo(math.sqrt(30.0), 1e-5));

          final nNuc = norm(m, ord: 'nuc');
          final svdRes = svd(m);
          final expectedNuc = svdRes.s.toList().reduce((a, b) => a + b);
          expect(nNuc.scalar, closeTo(expectedNuc, 1e-5));
          svdRes.dispose();

          final n1 = norm(m, ord: 1);
          expect(n1.scalar, closeTo(6.0, 1e-5));

          final nNeg1 = norm(m, ord: -1);
          expect(nNeg1.scalar, closeTo(4.0, 1e-5));

          final nInf = norm(m, ord: double.infinity);
          expect(nInf.scalar, closeTo(7.0, 1e-5));

          final nNegInf = norm(m, ord: -double.infinity);
          expect(nNegInf.scalar, closeTo(3.0, 1e-5));

          final n2 = norm(m, ord: 2);
          final svdRes2 = svd(m);
          final maxSing = svdRes2.s.toList()[0];
          svdRes2.dispose();
          expect(n2.scalar, closeTo(maxSing, 1e-5));

          final nKeep = norm(m, ord: 2, keepdims: true);
          expect(nKeep.shape, [1, 1]);
          expect(nKeep.getCell([0, 0]), closeTo(maxSing, 1e-5));
        }),
      );

      test(
        'matrix conditioning verification using norm products',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([4.0, 7.0, 2.0, 6.0]),
            [2, 2],
            DType.float64,
          );
          final aInv = inv(a);

          for (final ord in [
            1,
            -1,
            2,
            double.infinity,
            -double.infinity,
            'fro',
          ]) {
            final normA = norm(a, ord: ord).scalar;
            final normAInv = norm(aInv, ord: ord).scalar;
            final condVal = normA * normAInv;
            expect(condVal, greaterThanOrEqualTo(1.0 - 1e-5));
          }
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 8: Outer Product & Cross Product
    // ------------------------------------------------------------------------
    group('8. outer and cross', () {
      test(
        'outer product across different numerical types and shapes',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final b = NDArray.fromList(Int32List.fromList([10, 20]), [
            2,
          ], DType.int32);

          expect(() => outer(a.asAnySpec, b.asAnySpec), throwsArgumentError);
          final res = outer(a, b.astype(DType.float64));
          expect(res.shape, [3, 2]);
          expect(res.dtype, DType.float64);
          expect(res.toList(), [10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);

          final c1 = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 0.0)],
            [2],
            DType.complex128,
          );
          final c2 = NDArray.fromList(
            [Complex(0.0, 1.0), Complex(3.0, 0.0)],
            [2],
            DType.complex128,
          );
          final cOut = outer(c1, c2);
          expect(cOut.shape, [2, 2]);
          expect(cOut.dtype, DType.complex128);
        }),
      );

      test(
        'cross product of 2D and 3D vectors along various axes',
        () => NDArray.scope(() {
          final x = NDArray.fromList(Float64List.fromList([1.0, 0.0, 0.0]), [
            3,
          ], DType.float64);
          final y = NDArray.fromList(Float64List.fromList([0.0, 1.0, 0.0]), [
            3,
          ], DType.float64);
          final z = cross(x, y);
          expect(z.shape, [3]);
          expect(z.toList(), [0.0, 0.0, 1.0]);

          final v1 = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
            2,
          ], DType.float64);
          final v2 = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
            2,
          ], DType.float64);
          final cross2d = cross(v1, v2);
          expect(cross2d.scalar, closeTo(-2.0, 1e-6));

          final b1 = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            [2, 3],
            DType.float64,
          );
          final b2 = NDArray.fromList(
            Float64List.fromList([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            [2, 3],
            DType.float64,
          );
          final bCross = cross(b1, b2, axis: -1);
          expect(bCross.shape, [2, 3]);
          expect(bCross.toList(), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 9: Array Manipulation Operations (manipulation.dart)
    // ------------------------------------------------------------------------
    group('9. manipulation.dart operations', () {
      test(
        'rot90 in 2D and 3D tensors with custom k and axes',
        () => NDArray.scope(() {
          final m = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);

          expect(rot90(m, 0).toList(), [1, 2, 3, 4]);
          expect(rot90(m, 1).toList(), [2, 4, 1, 3]);
          expect(rot90(m, 2).toList(), [4, 3, 2, 1]);
          expect(rot90(m, 3).toList(), [3, 1, 4, 2]);
          expect(rot90(m, 4).toList(), [1, 2, 3, 4]);
          expect(rot90(m, -1).toList(), [3, 1, 4, 2]);
          expect(rot90(m, -2).toList(), [4, 3, 2, 1]);
          expect(rot90(m, -3).toList(), [2, 4, 1, 3]);

          final t3d = NDArray.fromList(
            Int32List.fromList(List.generate(8, (i) => i)),
            [2, 2, 2],
            DType.int32,
          );
          final rot3d = rot90(t3d, 1, [0, 2]);
          expect(rot3d.shape, [2, 2, 2]);
        }),
      );

      test(
        'flip, fliplr, flipud across 1D, 2D, 3D',
        () => NDArray.scope(() {
          final a1d = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            4,
          ], DType.int32);
          final f1d = flip(a1d);
          expect(f1d.toList(), [4, 3, 2, 1]);

          final a2d = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          expect(fliplr(a2d).toList(), [2, 1, 4, 3]);
          expect(flipud(a2d).toList(), [3, 4, 1, 2]);
          expect(flip(a2d).toList(), [4, 3, 2, 1]);
          expect(flip(a2d, axis: 0).toList(), [3, 4, 1, 2]);
          expect(flip(a2d, axis: 1).toList(), [2, 1, 4, 3]);
        }),
      );

      test(
        'roll with 1D, 2D, multi-axis, positive/negative shifts',
        () => NDArray.scope(() {
          final a1d = NDArray.fromList(Int32List.fromList([0, 1, 2, 3, 4]), [
            5,
          ], DType.int32);
          expect(roll(a1d, 2).toList(), [3, 4, 0, 1, 2]);
          expect(roll(a1d, -2).toList(), [2, 3, 4, 0, 1]);
          expect(roll(a1d, 0).toList(), [0, 1, 2, 3, 4]);
          expect(roll(a1d, 7).toList(), [3, 4, 0, 1, 2]);

          final a2d = NDArray.fromList(
            Int32List.fromList(List.generate(6, (i) => i)),
            [2, 3],
            DType.int32,
          );
          final r0 = roll(a2d, 1, axis: 0);
          expect(r0.toList(), [3, 4, 5, 0, 1, 2]);

          final r1 = roll(a2d, 1, axis: 1);
          expect(r1.toList(), [2, 0, 1, 5, 3, 4]);

          final rMulti = roll(a2d, [1, 1], axis: [0, 1]);
          expect(rMulti.toList(), [5, 3, 4, 2, 0, 1]);
        }),
      );

      test(
        'squeeze and expand_dims',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            1,
            2,
            1,
            2,
          ], DType.int32);
          final sqAll = squeeze(a);
          expect(sqAll.shape, [2, 2]);

          final sq0 = squeeze(a, axis: [0]);
          expect(sq0.shape, [2, 1, 2]);

          expect(() => squeeze(a, axis: [1]), throwsArgumentError);

          final exp0 = expand_dims(a, 0);
          expect(exp0.shape, [1, 1, 2, 1, 2]);

          final expNeg1 = expand_dims(a, -1);
          expect(expNeg1.shape, [1, 2, 1, 2, 1]);
        }),
      );

      test(
        'broadcastTo and broadcast',
        () => NDArray.scope(() {
          final v = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
            2,
          ], DType.float64);
          final b2d = broadcastTo(v, [3, 2]);
          expect(b2d.shape, [3, 2]);
          expect(b2d.toList(), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

          final b3d = broadcastTo(v, [2, 3, 2]);
          expect(b3d.shape, [2, 3, 2]);

          expect(() => broadcastTo(v, [3, 3]), throwsArgumentError);

          final m1 = NDArray.ones([2, 1], DType.float64);
          final m2 = NDArray.ones([1, 3], DType.float64);
          final bRes = broadcast(m1, m2);
          expect(bRes.shape, [2, 3]);
        }),
      );

      test(
        'concatenate, stack, vstack, hstack',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([5, 6, 7, 8]), [
            2,
            2,
          ], DType.int32);

          final c0 = concatenate([a, b], axis: 0);
          expect(c0.shape, [4, 2]);
          expect(c0.toList(), [1, 2, 3, 4, 5, 6, 7, 8]);

          final c1 = concatenate([a, b], axis: 1);
          expect(c1.shape, [2, 4]);
          expect(c1.toList(), [1, 2, 5, 6, 3, 4, 7, 8]);

          final s0 = stack([a, b], axis: 0);
          expect(s0.shape, [2, 2, 2]);

          final vs = vstack([a, b]);
          expect(vs.shape, [4, 2]);

          final hs = hstack([a, b]);
          expect(hs.shape, [2, 4]);
        }),
      );

      test(
        'diag, tril, triu',
        () => NDArray.scope(() {
          final v = NDArray.fromList(Int32List.fromList([1, 2, 3]), [
            3,
          ], DType.int32);
          final d0 = diag(v);
          expect(d0.shape, [3, 3]);
          expect(d0.toList(), [1, 0, 0, 0, 2, 0, 0, 0, 3]);

          final d1 = diag(v, k: 1);
          expect(d1.shape, [4, 4]);
          expect(d1.getCell([0, 1]), equals(1));
          expect(d1.getCell([1, 2]), equals(2));
          expect(d1.getCell([2, 3]), equals(3));

          final m = NDArray.fromList(
            Int32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            [3, 3],
            DType.int32,
          );
          expect(diag(m).toList(), [1, 5, 9]);
          expect(diag(m, k: 1).toList(), [2, 6]);
          expect(diag(m, k: -1).toList(), [4, 8]);

          final l0 = tril(m);
          expect(l0.toList(), [1, 0, 0, 4, 5, 0, 7, 8, 9]);

          final u0 = triu(m);
          expect(u0.toList(), [1, 2, 3, 0, 5, 6, 0, 0, 9]);
        }),
      );

      test(
        'diff with orders n=1, 2, 3 and various axes',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 4.0, 7.0, 11.0]),
            [5],
            DType.float64,
          );
          final d1 = diff(a, n: 1);
          expect(d1.toList(), [1.0, 2.0, 3.0, 4.0]);

          final d2 = diff(a, n: 2);
          expect(d2.toList(), [1.0, 1.0, 1.0]);

          final d3 = diff(a, n: 3);
          expect(d3.toList(), [0.0, 0.0]);
        }),
      );

      test(
        'slidingWindowView on 1D and 2D arrays',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 2, 3, 4, 5]), [
            5,
          ], DType.int32);
          final sw = slidingWindowView(a, [3]);
          expect(sw.shape, [3, 3]);
          expect(sw.toList(), [1, 2, 3, 2, 3, 4, 3, 4, 5]);

          final a2d = NDArray.fromList(
            Int32List.fromList(List.generate(9, (i) => i)),
            [3, 3],
            DType.int32,
          );
          final sw2d = slidingWindowView(a2d, [2, 2]);
          expect(sw2d.shape, [2, 2, 2, 2]);
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 10: Array Splitting Operations (splitting.dart)
    // ------------------------------------------------------------------------
    group('10. splitting.dart operations', () {
      test(
        'split and array_split equal and unequal splits',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 2, 3, 4, 5, 6]), [
            6,
          ], DType.int32);

          final s3 = split(a, 3);
          expect(s3.length, 3);
          expect(s3[0].toList(), [1, 2]);
          expect(s3[1].toList(), [3, 4]);
          expect(s3[2].toList(), [5, 6]);

          expect(() => split(a, 4), throwsArgumentError);

          final as4 = array_split(a, 4);
          expect(as4.length, 4);
          expect(as4[0].toList(), [1, 2]);
          expect(as4[1].toList(), [3, 4]);
          expect(as4[2].toList(), [5]);
          expect(as4[3].toList(), [6]);

          final as8 = array_split(a, 8);
          expect(as8.length, 8);
          expect(as8[0].toList(), [1]);
          expect(as8[5].toList(), [6]);
          expect(as8[6].toList(), <int>[]);
          expect(as8[7].toList(), <int>[]);
        }),
      );

      test(
        'split_at and array_split_at coordinate slices',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Int32List.fromList([10, 20, 30, 40, 50, 60]),
            [6],
            DType.int32,
          );
          final res = split_at(a, [2, 4]);
          expect(res.length, 3);
          expect(res[0].toList(), [10, 20]);
          expect(res[1].toList(), [30, 40]);
          expect(res[2].toList(), [50, 60]);

          final resBound = array_split_at(a, [-1, 3, 10]);
          expect(resBound.length, 4);
          expect(resBound[0].toList(), <int>[]);
          expect(resBound[1].toList(), [10, 20, 30]);
          expect(resBound[2].toList(), [40, 50, 60]);
          expect(resBound[3].toList(), <int>[]);
        }),
      );

      test(
        'hsplit, vsplit, dsplit across 1D, 2D, 3D',
        () => NDArray.scope(() {
          final v1d = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            4,
          ], DType.int32);
          final hs1 = hsplit(v1d, 2);
          expect(hs1.length, 2);
          expect(hs1[0].toList(), [1, 2]);
          expect(hs1[1].toList(), [3, 4]);

          final m2d = NDArray.fromList(
            Int32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]),
            [2, 4],
            DType.int32,
          );
          final hs2 = hsplit(m2d, 2);
          expect(hs2.length, 2);
          expect(hs2[0].shape, [2, 2]);
          expect(hs2[0].toList(), [1, 2, 5, 6]);
          expect(hs2[1].toList(), [3, 4, 7, 8]);

          final vs2 = vsplit(m2d, 2);
          expect(vs2.length, 2);
          expect(vs2[0].shape, [1, 4]);
          expect(vs2[0].toList(), [1, 2, 3, 4]);
          expect(vs2[1].toList(), [5, 6, 7, 8]);

          final t3d = NDArray.fromList(
            Int32List.fromList(List.generate(16, (i) => i)),
            [2, 2, 4],
            DType.int32,
          );
          final ds = dsplit(t3d, 2);
          expect(ds.length, 2);
          expect(ds[0].shape, [2, 2, 2]);
          expect(ds[1].shape, [2, 2, 2]);

          expect(() => vsplit(v1d, 2), throwsArgumentError);
          expect(() => dsplit(m2d, 2), throwsArgumentError);
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 11: Tensor Contractions (tensordot, einsum, inner, vdot, kron)
    // ------------------------------------------------------------------------
    group('11. tensor_contractions.dart operations', () {
      test(
        'tensordot across single and multi axes',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([5, 6, 7, 8]), [
            2,
            2,
          ], DType.int32);

          final td1 = tensordot(a, b, axes: 1);
          expect(td1.shape, [2, 2]);
          expect(td1.toList(), [19, 22, 43, 50]);

          final td2 = tensordot(a, b, axes: 2);
          expect(td2.shape, <int>[]);
          expect(td2.scalar, equals(70));

          // tensordot with explicit axes lists
          final tdCustom = tensordot(
            a,
            b,
            axes: TensordotAxes.explicit([1], [0]),
          );
          expect(tdCustom.shape, [2, 2]);
          expect(tdCustom.toList(), [19, 22, 43, 50]);
        }),
      );

      test(
        'inner, vdot, kron',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 2, 3]), [
            3,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([4, 5, 6]), [
            3,
          ], DType.int32);
          final inRes = inner(a, b);
          expect(inRes.scalar, equals(32));

          final c1 = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [2],
            DType.complex128,
          );
          final c2 = NDArray.fromList(
            [Complex(5.0, 6.0), Complex(7.0, 8.0)],
            [2],
            DType.complex128,
          );
          final vd = vdot(c1, c2);
          expect(vd.scalar.real, closeTo(70.0, 1e-6));
          expect(vd.scalar.imag, closeTo(-8.0, 1e-6));

          final k1 = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final k2 = NDArray.fromList(Int32List.fromList([0, 5, 6, 7]), [
            2,
            2,
          ], DType.int32);
          final kr = kron(k1, k2);
          expect(kr.shape, [4, 4]);
          expect(kr.getCell([0, 0]), equals(0));
          expect(kr.getCell([0, 1]), equals(5));
          expect(kr.getCell([0, 2]), equals(0));
          expect(kr.getCell([0, 3]), equals(10));
        }),
      );

      test(
        'einsum with matrix multiplication and trace subscripts',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([5, 6, 7, 8]), [
            2,
            2,
          ], DType.int32);

          final tr = einsum(EinsumSubscripts.parse('ii->'), [a]);
          expect(tr.scalar, equals(5));

          final mm = einsum(EinsumSubscripts.parse('ij,jk->ik'), [a, b]);
          expect(mm.shape, [2, 2]);
          expect(mm.toList(), [19, 22, 43, 50]);
        }),
      );
    });

    // ------------------------------------------------------------------------
    // Group 12: Additional Edge Cases, Recycler Buffers, & Error Handling
    // ------------------------------------------------------------------------
    group('12. Edge cases, recycler buffers, and error paths', () {
      test(
        'det and slogdet on 0x0 and 1x1 matrices and batch 0x0',
        () => NDArray.scope(() {
          // 0x0 matrix
          final empty2d = NDArray.zeros([0, 0], DType.float64);
          final dEmpty = det(empty2d);
          expect(dEmpty.scalar, equals(1.0));

          final sEmpty = slogdet(empty2d);
          expect(sEmpty.sign.scalar, equals(1.0));
          expect(sEmpty.logabsdet.scalar, equals(0.0));
          sEmpty.dispose();

          // 1x1 matrix
          final one2d = NDArray.fromList(Float64List.fromList([42.0]), [
            1,
            1,
          ], DType.float64);
          final dOne = det(one2d);
          expect(dOne.scalar, closeTo(42.0, 1e-6));

          final sOne = slogdet(one2d);
          expect(sOne.sign.scalar, closeTo(1.0, 1e-6));
          expect(sOne.logabsdet.scalar, closeTo(math.log(42.0), 1e-6));
          sOne.dispose();

          // 3D batch of 0x0 matrices: [2, 0, 0]
          final emptyBatch = NDArray.zeros([2, 0, 0], DType.float64);
          final dEmptyBatch = det(emptyBatch);
          expect(dEmptyBatch.shape, [2]);
          expect(dEmptyBatch.toList(), [1.0, 1.0]);

          final sEmptyBatch = slogdet(emptyBatch);
          expect(sEmptyBatch.sign.shape, [2]);
          expect(sEmptyBatch.logabsdet.shape, [2]);
          expect(sEmptyBatch.sign.toList(), [1.0, 1.0]);
          expect(sEmptyBatch.logabsdet.toList(), [0.0, 0.0]);
          sEmptyBatch.dispose();
        }),
      );

      test(
        'eig and eigh 3D batch and out parameter buffers',
        () => NDArray.scope(() {
          // 3D batch eig
          final batch = NDArray.fromList(
            Float64List.fromList([2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0]),
            [2, 2, 2],
            DType.float64,
          );
          final outVals = NDArray.zeros([2, 2], DType.complex128);
          final outVecs = NDArray.zeros([2, 2, 2], DType.complex128);
          final resEig = eig(
            batch,
            out: (eigenvalues: outVals, eigenvectors: outVecs),
          );
          expect(identical(resEig.eigenvalues, outVals), isTrue);
          expect(identical(resEig.eigenvectors, outVecs), isTrue);
          expect(resEig.eigenvalues.getCell([0, 0]).real, closeTo(2.0, 1e-5));
          expect(resEig.eigenvalues.getCell([0, 1]).real, closeTo(3.0, 1e-5));
          expect(resEig.eigenvalues.getCell([1, 0]).real, closeTo(4.0, 1e-5));
          expect(resEig.eigenvalues.getCell([1, 1]).real, closeTo(5.0, 1e-5));
          resEig.dispose();

          // 3D batch eigh
          final outValsH = NDArray.zeros([2, 2], DType.float64);
          final outVecsH = NDArray.zeros([2, 2, 2], DType.float64);
          final resEigh = eigh(
            batch,
            outEigenvalues: outValsH,
            outEigenvectors: outVecsH,
          );
          expect(identical(resEigh.eigenvalues, outValsH), isTrue);
          expect(identical(resEigh.eigenvectors, outVecsH), isTrue);
          expect(resEigh.eigenvalues.getCell([0, 0]), closeTo(2.0, 1e-5));
          expect(resEigh.eigenvalues.getCell([0, 1]), closeTo(3.0, 1e-5));
          resEigh.dispose();
        }),
      );

      test(
        'schur and hessenberg out parameter buffers and 1x1 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([5.0]), [
            1,
            1,
          ], DType.float64);
          final resSchur = schur(a);
          expect(resSchur.t.getCell([0, 0]), closeTo(5.0, 1e-6));
          expect(resSchur.z.getCell([0, 0]), closeTo(1.0, 1e-6));
          resSchur.dispose();

          final resHess = hessenberg(a);
          expect(resHess.h.getCell([0, 0]), closeTo(5.0, 1e-6));
          expect(resHess.q.getCell([0, 0]), closeTo(1.0, 1e-6));
          resHess.dispose();

          // 2x2 with out buffers
          final m = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final outT = NDArray.zeros([2, 2], DType.float64);
          final outZ = NDArray.zeros([2, 2], DType.float64);
          final sBuf = schur(m, outT: outT, outZ: outZ);
          expect(identical(sBuf.t, outT), isTrue);
          expect(identical(sBuf.z, outZ), isTrue);
          sBuf.dispose();

          final outH = NDArray.zeros([2, 2], DType.float64);
          final outQ = NDArray.zeros([2, 2], DType.float64);
          final hBuf = hessenberg(m, outH: outH, outQ: outQ);
          expect(identical(hBuf.h, outH), isTrue);
          expect(identical(hBuf.q, outQ), isTrue);
          hBuf.dispose();
        }),
      );

      test(
        'solve batch 3D systems and out buffer',
        () => NDArray.scope(() {
          final aBatch = NDArray.fromList(
            Float64List.fromList([2.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 4.0]),
            [2, 2, 2],
            DType.float64,
          );
          final bBatch = NDArray.fromList(
            Float64List.fromList([4.0, 6.0, 3.0, 8.0]),
            [2, 2, 1],
            DType.float64,
          );
          final outBuf = NDArray.zeros([2, 2, 1], DType.float64);
          final x = solve(aBatch, bBatch, out: outBuf);
          expect(identical(x, outBuf), isTrue);
          expect(x.shape, [2, 2, 1]);
          // Matrix 0: 2x = 4, 3y = 6 -> x = 2, y = 2
          expect(x.getCell([0, 0, 0]), closeTo(2.0, 1e-6));
          expect(x.getCell([0, 1, 0]), closeTo(2.0, 1e-6));
          // Matrix 1: 1x = 3, 4y = 8 -> x = 3, y = 2
          expect(x.getCell([1, 0, 0]), closeTo(3.0, 1e-6));
          expect(x.getCell([1, 1, 0]), closeTo(2.0, 1e-6));
        }),
      );

      test(
        'norm on 3D and 4D tensors with axis tuples and keepdims',
        () => NDArray.scope(() {
          final t3d = NDArray.fromList(
            Float64List.fromList(List.generate(24, (i) => (i + 1).toDouble())),
            [2, 3, 4],
            DType.float64,
          );

          // Norm along single axis 0
          final nAx0 = norm(t3d, axis: 0);
          expect(nAx0.shape, [3, 4]);

          // Norm along axes [1, 2] (matrix norm across last 2 dims)
          final nMat = norm(t3d, ord: 'fro', axis: [1, 2]);
          expect(nMat.shape, [2]);

          // Norm along axes [-2, -1] with keepdims
          final nKeep = norm(t3d, ord: 1, axis: [-2, -1], keepdims: true);
          expect(nKeep.shape, [2, 1, 1]);

          // Vector norm with p=3.0
          final v = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final nP3 = norm(v, ord: 3);
          // (1 + 8 + 27)^(1/3) = 36^(1/3) ≈ 3.3019
          expect(nP3.scalar, closeTo(math.pow(36.0, 1.0 / 3.0), 1e-5));
        }),
      );

      test(
        'cross with axisa, axisb, axisc and out buffer',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 0.0, 0.0]), [
            3,
          ], DType.float64);
          final b = NDArray.fromList(Float64List.fromList([0.0, 1.0, 0.0]), [
            3,
          ], DType.float64);
          final outCross = NDArray.zeros([3], DType.float64);

          final res = cross(a, b, axisa: 0, axisb: 0, axisc: 0, out: outCross);
          expect(identical(res, outCross), isTrue);
          expect(res.toList(), [0.0, 0.0, 1.0]);
        }),
      );

      test(
        'manipulation roll with axis null (flattened) and 3D tensors',
        () => NDArray.scope(() {
          final m2d = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final rFlat = roll(m2d, 1); // axis is null -> rolls flattened
          expect(rFlat.shape, [2, 2]);
          expect(rFlat.toList(), [4, 1, 2, 3]);

          final t3d = NDArray.fromList(
            Int32List.fromList(List.generate(8, (i) => i)),
            [2, 2, 2],
            DType.int32,
          );
          final r3d = roll(t3d, [1, -1], axis: [0, 2]);
          expect(r3d.shape, [2, 2, 2]);
        }),
      );

      test(
        'diag on rectangular matrices with large k offsets',
        () => NDArray.scope(() {
          final rect = NDArray.fromList(
            Int32List.fromList([1, 2, 3, 4, 5, 6]),
            [2, 3],
            DType.int32,
          );
          // k = 0: [1, 5]
          expect(diag(rect, k: 0).toList(), [1, 5]);
          // k = 1: [2, 6]
          expect(diag(rect, k: 1).toList(), [2, 6]);
          // k = 2: [3]
          expect(diag(rect, k: 2).toList(), [3]);
          // k = -1: [4]
          expect(diag(rect, k: -1).toList(), [4]);
          // k >= cols: empty
          expect(diag(rect, k: 5).toList(), <int>[]);
          // k <= -rows: empty
          expect(diag(rect, k: -5).toList(), <int>[]);
        }),
      );

      test(
        'tril and triu with large k and out buffers',
        () => NDArray.scope(() {
          final m = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final outL = NDArray.zeros([2, 2], DType.int32);
          final lRes = tril(m, k: 10, out: outL); // all elements kept
          expect(identical(lRes, outL), isTrue);
          expect(lRes.toList(), [1, 2, 3, 4]);

          final outU = NDArray.zeros([2, 2], DType.int32);
          final uRes = triu(m, k: -10, out: outU); // all elements kept
          expect(identical(uRes, outU), isTrue);
          expect(uRes.toList(), [1, 2, 3, 4]);
        }),
      );

      test(
        'einsum with outer product and diagonal extraction',
        () => NDArray.scope(() {
          final v1 = NDArray.fromList(Int32List.fromList([1, 2]), [
            2,
          ], DType.int32);
          final v2 = NDArray.fromList(Int32List.fromList([3, 4]), [
            2,
          ], DType.int32);

          // Outer product: 'i,j->ij'
          final outProd = einsum(EinsumSubscripts.parse('i,j->ij'), [v1, v2]);
          expect(outProd.shape, [2, 2]);
          expect(outProd.toList(), [3, 4, 6, 8]);

          // Diagonal extraction: 'ii->i'
          final mat = NDArray.fromList(Int32List.fromList([1, 2, 3, 4]), [
            2,
            2,
          ], DType.int32);
          final dExt = einsum(EinsumSubscripts.parse('ii->i'), [mat]);
          expect(dExt.shape, [2]);
          expect(dExt.toList(), [1, 4]);

          // Transpose: 'ij->ji'
          final tr = einsum(EinsumSubscripts.parse('ij->ji'), [mat]);
          expect(tr.shape, [2, 2]);
          expect(tr.toList(), [1, 3, 2, 4]);
        }),
      );
    });
  });
}
