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
  group('linalg branch coverage', () {
    test('matmul 1D/2D/transposed/dtypes/out/half-precision branches', () {
      NDArray.scope(() {
        // float16 with invalid out shape
        final f16A = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float16,
        );
        final f16BadOut = NDArray.zeros([3, 3], DType.float16);
        expect(() => matmul(f16A, f16A, out: f16BadOut), throwsArgumentError);

        // 1D dot product with broadcast b (strides[0] == 0)
        final v1 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final vScalar = NDArray.fromList([2.0], [1], DType.float64);
        final vBroadcast = broadcastTo(vScalar, [3]);
        expect(matmul(v1, vBroadcast).scalar, closeTo(12.0, 1e-12));

        // 1D complex128 and complex64 dot product with out
        final c128Vec = NDArray.fromList(
          [const Complex(1, 2), const Complex(3, 4)],
          [2],
          DType.complex128,
        );
        final c128Out = NDArray.zeros([], DType.complex128);
        matmul(c128Vec, c128Vec, out: c128Out);
        expect(c128Out.scalar, equals(const Complex(-10, 28)));

        final c64Vec = NDArray.fromList(
          [const Complex(1, 2), const Complex(3, 4)],
          [2],
          DType.complex64,
        );
        final c64Out = NDArray.zeros([], DType.complex64);
        matmul(c64Vec, c64Vec, out: c64Out);
        expect(c64Out.scalar, equals(const Complex(-10, 28)));

        // 1D integer dot product (hits aPromoted && bPromoted in walk)
        final i32Vec = NDArray.fromList([1, 2, 3], [3], DType.int32);
        expect(matmul(i32Vec, i32Vec).scalar, equals(14));

        // 2D-1D (bPromoted) and 1D-2D (aPromoted) with contiguous and transposed matrices
        // for float32, complex128, complex64
        final f32Mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final f32MatT = f32Mat.transpose();
        final f32Vec = NDArray.fromList([1.0, 2.0], [2], DType.float32);
        expect(matmul(f32Mat, f32Vec).toList(), equals([5.0, 11.0]));
        expect(matmul(f32MatT, f32Vec).toList(), equals([7.0, 10.0]));
        expect(matmul(f32Vec, f32Mat).toList(), equals([7.0, 10.0]));
        expect(matmul(f32Vec, f32MatT).toList(), equals([5.0, 11.0]));

        final c128Mat = NDArray.fromList(
          [
            const Complex(1, 0),
            const Complex(2, 1),
            const Complex(3, -1),
            const Complex(4, 0),
          ],
          [2, 2],
          DType.complex128,
        );
        final c128MatT = c128Mat.transpose();
        final c128V = NDArray.fromList(
          [const Complex(1, 1), const Complex(2, 0)],
          [2],
          DType.complex128,
        );
        expect(matmul(c128Mat, c128V).shape, equals([2]));
        expect(matmul(c128MatT, c128V).shape, equals([2]));
        expect(matmul(c128V, c128Mat).shape, equals([2]));
        expect(matmul(c128V, c128MatT).shape, equals([2]));

        final c64Mat = NDArray.fromList(
          [
            const Complex(1, 0),
            const Complex(2, 1),
            const Complex(3, -1),
            const Complex(4, 0),
          ],
          [2, 2],
          DType.complex64,
        );
        final c64MatT = c64Mat.transpose();
        final c64V = NDArray.fromList(
          [const Complex(1, 1), const Complex(2, 0)],
          [2],
          DType.complex64,
        );
        expect(matmul(c64Mat, c64V).shape, equals([2]));
        expect(matmul(c64MatT, c64V).shape, equals([2]));
        expect(matmul(c64V, c64Mat).shape, equals([2]));
        expect(matmul(c64V, c64MatT).shape, equals([2]));
      });
    });

    test('multi_dot validation, promotion, and mixed dtypes', () {
      NDArray.scope(() {
        final a = NDArray.eye(2, DType.float64);
        final disposed = NDArray.eye(2, DType.float64)..dispose();
        expect(() => multi_dot([a, disposed]), throwsStateError);
        expect(() => multi_dot([a, a], out: disposed), throwsStateError);

        final t3d = NDArray.zeros([2, 2, 2], DType.float64);
        expect(() => multi_dot([a, t3d, a]), throwsArgumentError);

        final badLast = NDArray.zeros([3, 2], DType.float64);
        expect(() => multi_dot([a, a, badLast]), throwsArgumentError);

        final badOut = NDArray.zeros([3, 3], DType.float64);
        expect(() => multi_dot([a, a], out: badOut), throwsArgumentError);

        // Integer inputs promoted to float64 via T == Float64 or float64 out
        final i32Mat = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final f64Out = NDArray.zeros([2, 2], DType.float64);
        final resPromoted = multi_dot<Float64>([i32Mat, i32Mat], out: f64Out);
        expect(resPromoted.dtype, equals(DType.float64));
        expect(resPromoted.toList(), equals([7.0, 10.0, 15.0, 22.0]));

        // Mixed dtypes in multi_dot (int32 + float64)
        final resMixed = multi_dot([i32Mat, a]);
        expect(resMixed.dtype, equals(DType.float64));
        expect(resMixed.toList(), equals([1.0, 2.0, 3.0, 4.0]));
      });
    });

    test('promoted out helpers across inv, det, slogdet, solve, qr, svd', () {
      NDArray.scope(() {
        final i32Mat = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32Vec = NDArray.fromList([1, 2], [2], DType.int32);
        final f64MatOut = NDArray.zeros([2, 2], DType.float64);
        final f64VecOut = NDArray.zeros([2], DType.float64);
        final badShapeOut = NDArray.zeros([3, 3], DType.float64);

        // inv with promoted int32 and out
        inv(i32Mat, out: f64MatOut);
        expect(f64MatOut.getCell([0, 0]), closeTo(-2.0, 1e-10));
        expect(() => inv(i32Mat, out: badShapeOut), throwsArgumentError);

        // inv with float16 and float16 out (allowOrig)
        final f16Mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float16,
        );
        final f16MatOut = NDArray.zeros([2, 2], DType.float16);
        inv(f16Mat.asAnySpec, out: f16MatOut.asAnySpec);
        expect(f16MatOut.getCell([0, 0]), closeTo(-2.0, 1e-2));

        // solve with promoted int32 and out
        solve(i32Mat, i32Vec, out: f64VecOut);
        expect(f64VecOut.getCell([0]), closeTo(0.0, 1e-10));
        expect(f64VecOut.getCell([1]), closeTo(0.5, 1e-10));

        // slogdet disposed and promoted out
        final disposed = NDArray.eye(2, DType.float64)..dispose();
        final validScalar = NDArray.zeros([], DType.float64);
        expect(() => slogdet(disposed), throwsStateError);
        expect(() => slogdet(f64MatOut, outSign: disposed), throwsStateError);
        expect(() => slogdet(f64MatOut, outLogdet: disposed), throwsStateError);
        final nonSquare = NDArray.zeros([2, 3], DType.float64);
        expect(() => slogdet(nonSquare), throwsArgumentError);

        final signOut = NDArray.zeros([], DType.float64);
        final logdetOut = NDArray.zeros([], DType.float64);
        slogdet(i32Mat, outSign: signOut, outLogdet: logdetOut);
        expect(signOut.scalar, closeTo(-1.0, 1e-10));
        expect(validScalar.isDisposed, isFalse);

        // qr disposed, aliased out, and promoted out
        final qOut = NDArray.zeros([2, 2], DType.float64);
        final rOut = NDArray.zeros([2, 2], DType.float64);
        expect(() => qr(disposed), throwsStateError);
        expect(
          () => qr(f64MatOut, out: (q: disposed, r: rOut)),
          throwsStateError,
        );
        expect(
          () => qr(f64MatOut, out: (q: qOut, r: qOut)),
          throwsArgumentError,
        );
        qr(i32Mat, out: (q: qOut, r: rOut));
        expect(matmul(qOut, rOut).getCell([0, 0]), closeTo(1.0, 1e-10));

        // svd disposed, aliased out, promoted out, and invalid out.s / out.vh
        final uOut = NDArray.zeros([2, 2], DType.float64);
        final sOut = NDArray.zeros([2], DType.float64);
        final vhOut = NDArray.zeros([2, 2], DType.float64);
        expect(() => svd(disposed), throwsStateError);
        expect(
          () => svd(f64MatOut, out: (u: uOut, s: disposed, vh: vhOut)),
          throwsStateError,
        );
        expect(
          () => svd(f64MatOut, out: (u: uOut, s: sOut, vh: uOut)),
          throwsArgumentError,
        );
        final sViewInVh = NDArray<Float64>.view(
          vhOut,
          shape: [2],
          strides: [1],
          offsetElements: 0,
        );
        expect(
          () => svd(f64MatOut, out: (u: uOut, s: sViewInVh, vh: vhOut)),
          throwsArgumentError,
        );
        svd(i32Mat, out: (u: uOut, s: sOut, vh: vhOut));
        expect(sOut.getCell([0]), greaterThan(5.0));

        final badS = NDArray.zeros([3], DType.float64);
        final badVh = NDArray.zeros([3, 3], DType.float64);
        expect(
          () => svd(f64MatOut, out: (u: uOut, s: badS, vh: vhOut)),
          throwsArgumentError,
        );
        expect(
          () => svd(f64MatOut, out: (u: uOut, s: sOut, vh: badVh)),
          throwsArgumentError,
        );

        // svd on 3D empty matrices [2, 2, 0] and [2, 0, 2] with fullMatrices: true
        final emptyCols = NDArray.zeros([2, 2, 0], DType.float64);
        final svdEmptyCols = svd(emptyCols, fullMatrices: true);
        expect(svdEmptyCols.u.shape, equals([2, 2, 2]));
        final emptyRows = NDArray.zeros([2, 0, 2], DType.float64);
        final svdEmptyRows = svd(emptyRows, fullMatrices: true);
        expect(svdEmptyRows.vh.shape, equals([2, 2, 2]));
      });
    });

    test('inv non-square, LU NaN generation, det/pinv/cholesky/solve errors', () {
      NDArray.scope(() {
        final nonSq = NDArray.zeros([2, 3], DType.float64);
        expect(() => inv(nonSq), throwsArgumentError);

        // Matrices with +Infinity where LU produces Inf - Inf = NaN with info == 0
        final infF64 = NDArray.fromList(
          [1.0, double.infinity, 1.0, double.infinity],
          [2, 2],
          DType.float64,
        );
        expect(inv(infF64).getCell([0, 0]).isNaN, isTrue);

        final infF32 = NDArray.fromList(
          [1.0, double.infinity, 1.0, double.infinity],
          [2, 2],
          DType.float32,
        );
        expect(inv(infF32).getCell([0, 0]).isNaN, isTrue);

        final infC128 = NDArray.fromList(
          [
            const Complex(1, 0),
            const Complex(double.infinity, 0),
            const Complex(1, 0),
            const Complex(double.infinity, 0),
          ],
          [2, 2],
          DType.complex128,
        );
        expect(inv(infC128).getCell([0, 0]).real.isNaN, isTrue);

        final infC64 = NDArray.fromList(
          [
            const Complex(1, 0),
            const Complex(double.infinity, 0),
            const Complex(1, 0),
            const Complex(double.infinity, 0),
          ],
          [2, 2],
          DType.complex64,
        );
        expect(inv(infC64).getCell([0, 0]).real.isNaN, isTrue);

        // det, pinv, cholesky invalid out
        final eye2 = NDArray.eye(2, DType.float64);
        final badOut = NDArray.zeros([5], DType.float64);
        expect(() => det(eye2, out: badOut), throwsArgumentError);
        expect(() => pinv(eye2, out: badOut), throwsArgumentError);
        expect(() => cholesky(eye2, out: badOut), throwsArgumentError);

        // solve error branches & 3D transposed slices
        final bBad2D = NDArray.zeros([3, 2], DType.float64);
        expect(() => solve(eye2, bBad2D), throwsArgumentError);

        final a3D = NDArray.zeros([2, 2, 2], DType.float64);
        a3D.setCell([0, 0, 0], 1.0);
        a3D.setCell([0, 1, 1], 2.0);
        a3D.setCell([1, 0, 0], 3.0);
        a3D.setCell([1, 1, 1], 4.0);
        final bBad1D = NDArray.zeros([3], DType.float64);
        final bBad3D = NDArray.zeros([2, 3, 2], DType.float64);
        expect(() => solve(a3D, bBad1D), throwsArgumentError);
        expect(() => solve(a3D, bBad3D), throwsArgumentError);
        expect(
          () => solve(eye2, NDArray.zeros([2], DType.float64), out: badOut),
          throwsArgumentError,
        );

        final a3DT = a3D.transpose([0, 2, 1]);
        final b3D = NDArray.ones([2, 2, 1], DType.float64);
        final sol3D = solve(a3DT, b3D);
        expect(sol3D.shape, equals([2, 2, 1]));
      });
    });

    test('matrix_power, eig, eigvals validation and aliasing', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final disposed = NDArray.eye(2, DType.float64)..dispose();
        final disposedC128 = NDArray.zeros([2], DType.complex128)..dispose();
        final badOut = NDArray.zeros([3, 3], DType.float64);

        // matrix_power
        expect(() => matrix_power(disposed, 2), throwsStateError);
        expect(() => matrix_power(a, 2, out: disposed), throwsStateError);
        expect(() => matrix_power(a, 2, out: badOut), throwsArgumentError);
        final aCopy = a.copy();
        matrix_power(aCopy, 2, out: aCopy);
        expect(aCopy.toList(), equals([7.0, 10.0, 15.0, 22.0]));

        // eig & eigvals
        final wOut = NDArray.zeros([2], DType.complex128);
        final vrOut = NDArray.zeros([2, 2], DType.complex128);
        final badW = NDArray.zeros([3], DType.complex128);
        final badVr = NDArray.zeros([3, 3], DType.complex128);
        expect(() => eig(disposed), throwsStateError);
        expect(
          () => eig(a, out: (eigenvalues: disposedC128, eigenvectors: vrOut)),
          throwsStateError,
        );
        expect(
          () => eig(a, out: (eigenvalues: badW, eigenvectors: vrOut)),
          throwsArgumentError,
        );
        expect(
          () => eig(a, out: (eigenvalues: wOut, eigenvectors: badVr)),
          throwsArgumentError,
        );
        final wViewInVr = NDArray<Complex128>.view(
          vrOut,
          shape: [2],
          strides: [1],
          offsetElements: 0,
        );
        expect(
          () => eig(a, out: (eigenvalues: wViewInVr, eigenvectors: vrOut)),
          throwsArgumentError,
        );

        expect(() => eigvals(disposed), throwsStateError);
        expect(() => eigvals(a, out: disposedC128), throwsStateError);
        expect(() => eigvals(a, out: badW), throwsArgumentError);
        final boolMat = NDArray.zeros([2, 2], DType.boolean);
        expect(() => eig(boolMat), throwsUnimplementedError);
        expect(() => eigvals(boolMat), throwsUnimplementedError);
      });
    });

    test(
      'eigh, eigvalsh, schur, hessenberg validation, promotion, and non-finite',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [2.0, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final disposed = NDArray.eye(2, DType.float64)..dispose();
          final bad1D = NDArray.zeros([2], DType.float64);
          final badNonSq = NDArray.zeros([2, 3], DType.float64);
          final badVec = NDArray.zeros([3], DType.float64);
          final badMat = NDArray.zeros([3, 3], DType.float64);

          // eigh validation & aliasing
          final wOut = NDArray.zeros([2], DType.float64);
          final vOut = NDArray.zeros([2, 2], DType.float64);
          expect(
            () => eigh(a, outEigenvalues: wOut, outEigenvectors: disposed),
            throwsStateError,
          );
          expect(() => eigh(a, outEigenvalues: badVec), throwsArgumentError);
          expect(() => eigh(a, outEigenvectors: badMat), throwsArgumentError);
          final wViewInV = NDArray<Float64>.view(
            vOut,
            shape: [2],
            strides: [1],
            offsetElements: 0,
          );
          expect(
            () => eigh(a, outEigenvalues: wViewInV, outEigenvectors: vOut),
            throwsArgumentError,
          );

          // eigh with non-contiguous input and outEigenvectors (hits _analyzeNonFinite non-contiguous path)
          final aTrans = a.transpose();
          final vOutContig = NDArray.zeros([2, 2], DType.float64);
          eigh(aTrans, outEigenvectors: vOutContig);

          // eigh with promoted int32 and outEigenvectors (where vOutContig is contiguous and not aliased)
          final i32Sym = NDArray.fromList([2, 1, 1, 2], [2, 2], DType.int32);
          eigh(i32Sym, outEigenvalues: wOut, outEigenvectors: vOutContig);
          expect(wOut.toList(), equals([1.0, 3.0]));

          // eigh with float16 input and float16 out buffers (hits castNDArray branch in needTempVal/needTempVec)
          final f16Sym = NDArray.fromList(
            [2.0, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float16,
          );
          final f16W = NDArray.zeros([2], DType.float16);
          final f16V = NDArray.zeros([2, 2], DType.float16);
          eigh(
            f16Sym.asAnySpec,
            outEigenvalues: f16W.asAnySpec,
            outEigenvectors: f16V.asAnySpec,
          );
          expect(f16W.getCell([0]), closeTo(1.0, 1e-2));

          // eigh with NaN/Inf on float16 (hits L5286 and L5307 after cast)
          final f16Inf = NDArray.fromList(
            [double.infinity, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float16,
          );
          expect(() => eigh(f16Inf.asAnySpec), throwsA(isA<LinAlgException>()));
          expect(
            () => eigh(f16Inf.asAnySpec, outEigenvectors: vOutContig.asAnySpec),
            throwsA(isA<LinAlgException>()),
          );

          // eigvalsh validation, 3D stack, and float16 out
          expect(() => eigvalsh(a, out: badVec), throwsArgumentError);
          eigvalsh(f16Sym.asAnySpec, out: f16W.asAnySpec);
          expect(f16W.getCell([1]), closeTo(3.0, 1e-2));
          final sym3D = NDArray.zeros([2, 2, 2], DType.float64);
          sym3D.setCell([0, 0, 0], 1.0);
          sym3D.setCell([0, 1, 1], 2.0);
          sym3D.setCell([1, 0, 0], 3.0);
          sym3D.setCell([1, 1, 1], 4.0);
          expect(eigvalsh(sym3D).shape, equals([2, 2]));

          // schur validation and float16 out
          expect(() => schur(disposed), throwsStateError);
          expect(() => schur(bad1D), throwsArgumentError);
          expect(() => schur(badNonSq), throwsArgumentError);
          expect(() => schur(a, outT: disposed), throwsStateError);
          expect(() => schur(a, outZ: disposed), throwsStateError);
          expect(() => schur(a, outT: badMat), throwsArgumentError);
          expect(() => schur(a, outZ: badMat), throwsArgumentError);
          final f16T = NDArray.zeros([2, 2], DType.float16);
          final f16Z = NDArray.zeros([2, 2], DType.float16);
          schur(f16Sym.asAnySpec, outT: f16T.asAnySpec, outZ: f16Z.asAnySpec);

          // hessenberg validation and float16 out
          expect(() => hessenberg(disposed), throwsStateError);
          expect(() => hessenberg(bad1D), throwsArgumentError);
          expect(() => hessenberg(badNonSq), throwsArgumentError);
          expect(() => hessenberg(a, outH: disposed), throwsStateError);
          expect(() => hessenberg(a, outQ: disposed), throwsStateError);
          expect(() => hessenberg(a, outH: badMat), throwsArgumentError);
          expect(() => hessenberg(a, outQ: badMat), throwsArgumentError);
          hessenberg(
            f16Sym.asAnySpec,
            outH: f16T.asAnySpec,
            outQ: f16Z.asAnySpec,
          );
        });
      },
    );

    test('outer, cross, matmulAs, outerAs, crossAs branches', () {
      NDArray.scope(() {
        final v = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final badOut = NDArray.zeros([3, 3], DType.float64);
        expect(() => outer(v, v, out: badOut), throwsArgumentError);

        final disposed = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64)
          ..dispose();
        final v3 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        expect(() => cross(disposed, v3), throwsStateError);
        expect(() => cross(v3, v3, axisb: 5), throwsRangeError);
        final v4 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        expect(() => cross(v4, v4), throwsArgumentError);
        expect(() => cross(v, v3), throwsArgumentError);
        expect(() => cross(v3, v3, axisc: 5), throwsRangeError);
        expect(() => cross(v3, v3, out: badOut), throwsArgumentError);

        // float16 cross with and without out
        final f16V3 = NDArray.fromList([1.0, 0.0, 0.0], [3], DType.float16);
        final f16V3B = NDArray.fromList([0.0, 1.0, 0.0], [3], DType.float16);
        final f16Out = NDArray.zeros([3], DType.float16);
        expect(cross(f16V3, f16V3B).getCell([2]), closeTo(1.0, 1e-2));
        cross(f16V3, f16V3B, out: f16Out);
        expect(f16Out.getCell([2]), closeTo(1.0, 1e-2));

        // 3D cross with axisc: 0 on batched 2x3 arrays
        final batchA = NDArray.ones([2, 3], DType.float64);
        final batchB = NDArray.ones([2, 3], DType.float64);
        expect(cross(batchA, batchB, axisc: 0).shape, equals([3, 2]));

        // 2D cross (len == 2) across all dtypes
        expect(
          cross(
            NDArray.fromList([1.0, 2.0], [2], DType.float32),
            NDArray.fromList([3.0, 4.0], [2], DType.float32),
          ).shape,
          equals(<int>[]),
        );
        expect(
          cross(
            NDArray.fromList([1, 2], [2], DType.int64),
            NDArray.fromList([3, 4], [2], DType.int64),
          ).shape,
          equals(<int>[]),
        );
        expect(
          cross(
            NDArray.fromList([1, 2], [2], DType.int32),
            NDArray.fromList([3, 4], [2], DType.int32),
          ).shape,
          equals(<int>[]),
        );
        expect(
          cross(
            NDArray.fromList([1, 2], [2], DType.uint8),
            NDArray.fromList([3, 4], [2], DType.uint8),
          ).shape,
          equals(<int>[]),
        );
        expect(
          cross(
            NDArray.fromList([1, 2], [2], DType.int16),
            NDArray.fromList([3, 4], [2], DType.int16),
          ).shape,
          equals(<int>[]),
        );
        final c128A2 = NDArray.fromList(
          [const Complex(1, 0), const Complex(2, 0)],
          [2],
          DType.complex128,
        );
        expect(cross(c128A2, c128A2).scalar, equals(const Complex(0, 0)));
        final c64A2 = NDArray.fromList(
          [const Complex(1, 0), const Complex(2, 0)],
          [2],
          DType.complex64,
        );
        expect(cross(c64A2, c64A2).scalar, equals(const Complex(0, 0)));
        final boolA2 = NDArray.fromList([true, false], [2], DType.boolean);
        final boolB2 = NDArray.fromList([false, true], [2], DType.boolean);
        expect(cross(boolA2, boolB2).scalar, isTrue);

        // matmulAs, outerAs, crossAs disposed and same-dtype fast-paths
        expect(() => matmulAs(disposed, v3, DType.float64), throwsStateError);
        expect(() => outerAs(disposed, v3, DType.float64), throwsStateError);
        expect(() => crossAs(disposed, v3, DType.float64), throwsStateError);
        expect(matmulAs(v3, v3, DType.float64).scalar, closeTo(14.0, 1e-12));
        expect(outerAs(v, v, DType.float64).shape, equals([2, 2]));
        expect(
          crossAs(v3, v3, DType.float64).toList(),
          equals([0.0, 0.0, 0.0]),
        );
      });
    });

    test('norm axis/ord/dtype/error branches', () {
      NDArray.scope(() {
        final vF64 = NDArray.fromList([3.0, -4.0], [2], DType.float64);
        final vF32 = NDArray.fromList([3.0, -4.0], [2], DType.float32);
        final vC128 = NDArray.fromList(
          [const Complex(3, 0), const Complex(0, -4)],
          [2],
          DType.complex128,
        );
        final vC64 = NDArray.fromList(
          [const Complex(3, 0), const Complex(0, -4)],
          [2],
          DType.complex64,
        );
        final mF64 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final mF32 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final disposed = NDArray.eye(2, DType.float64)..dispose();

        expect(() => norm(disposed), throwsStateError);
        expect(
          () => norm(NDArray.zeros([2, 2, 2], DType.float64)),
          throwsArgumentError,
        );
        expect(() => norm(mF64, axis: 5), throwsRangeError);
        expect(() => norm(mF64, axis: [0, 1, 2]), throwsArgumentError);
        expect(() => norm(mF64, axis: [0, 0]), throwsArgumentError);
        expect(() => norm(mF64, axis: 'bad'), throwsArgumentError);
        expect(
          () => norm(vF64, out: NDArray.zeros([2], DType.float64)),
          throwsArgumentError,
        );

        // Negative axis and float32 axis reduction
        expect(norm(mF32, axis: -1).dtype, equals(DType.float32));

        // Vector norms across NormKind and float32/complex64/complex128/float64
        expect(norm(vF64, ord: NormKind.negL1).scalar, greaterThan(0.0));
        expect(norm(vF64, ord: NormKind.negL2).scalar, greaterThan(0.0));
        for (final vec in <NDArray<AnySpec>>[
          vF64.asAnySpec,
          vF32.asAnySpec,
          vC128.asAnySpec,
          vC64.asAnySpec,
        ]) {
          expect(norm(vec, ord: double.infinity).scalar, closeTo(4.0, 1e-5));
          expect(
            norm(vec, ord: double.negativeInfinity).scalar,
            closeTo(3.0, 1e-5),
          );
          expect(norm(vec, ord: 0).scalar, closeTo(2.0, 1e-5));
          expect(norm(vec, ord: 3).scalar, greaterThan(4.0));
        }
        expect(() => norm(vF64, ord: 'bad'), throwsArgumentError);

        // Matrix norms with NormKind.negL1, negL2, negInfinity, invalid ord, and float16 ord: 2
        expect(norm(mF64, ord: NormKind.negL1).scalar, closeTo(4.0, 1e-10));
        expect(norm(mF64, ord: NormKind.negL2).scalar, greaterThan(0.0));
        expect(
          norm(mF64, ord: NormKind.negInfinity).scalar,
          closeTo(3.0, 1e-10),
        );
        expect(() => norm(mF64, ord: 99), throwsArgumentError);

        final f16Mat = NDArray.fromList(
          [1.0, 0.0, 0.0, 2.0],
          [2, 2],
          DType.float16,
        );
        expect(norm(f16Mat, ord: 2).scalar, closeTo(2.0, 1e-2));
      });
    });

    test(
      'lstsq, cond, svdvals, matrix_rank, tensorsolve, tensorinv branches',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [3, 2],
            DType.float64,
          );
          final b = NDArray.fromList([1.0, 4.0, 3.0], [3], DType.float64);
          final disposed = NDArray.eye(2, DType.float64)..dispose();

          // lstsq errors, non-contiguous b, float16 out, complex128 residuals
          expect(() => lstsq(disposed, b), throwsStateError);
          expect(() => lstsq(a, b, out: disposed), throwsStateError);
          final boolMat = NDArray.fromList(
            [true, false, false, true],
            [2, 2],
            DType.boolean,
          );
          expect(() => lstsq(boolMat, boolMat), throwsArgumentError);
          expect(() => lstsq(b, b), throwsArgumentError);
          expect(
            () => lstsq(a, NDArray.zeros([3, 2, 1], DType.float64)),
            throwsArgumentError,
          );
          expect(
            () => lstsq(a, NDArray.zeros([2], DType.float64)),
            throwsArgumentError,
          );
          expect(
            () => lstsq(a, b, out: NDArray.zeros([5], DType.float64)),
            throwsArgumentError,
          );

          final b2D = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          ).transpose();
          expect(lstsq(a, b2D).x.shape, equals([2, 2]));

          final f16A = NDArray.fromList(
            [1.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float16,
          );
          final f16B = NDArray.fromList([1.0, 4.0], [2], DType.float16);
          final f16XOut = NDArray.zeros([2], DType.float16);
          lstsq(f16A, f16B, out: f16XOut.asAnySpec);
          expect(f16XOut.getCell([1]), closeTo(2.0, 1e-2));

          final c128A = NDArray.fromList(
            [
              const Complex(1, 0),
              const Complex(0, 0),
              const Complex(0, 0),
              const Complex(2, 0),
              const Complex(0, 0),
              const Complex(0, 0),
            ],
            [3, 2],
            DType.complex128,
          );
          final c128B = NDArray.fromList(
            [const Complex(1, 0), const Complex(4, 0), const Complex(3, 4)],
            [3],
            DType.complex128,
          );
          final c128Res = lstsq(c128A, c128B);
          expect(c128Res.residuals.getCell([0]), closeTo(25.0, 1e-10));

          // cond errors
          final eye2 = NDArray.eye(2, DType.float64);
          expect(() => cond(disposed), throwsStateError);
          expect(() => cond(b), throwsArgumentError);
          expect(() => cond(eye2, p: NormKind.nuclear), throwsArgumentError);
          expect(
            () => cond(NDArray.zeros([0, 0], DType.float64)),
            throwsArgumentError,
          );
          expect(
            () => cond(eye2, out: NDArray.zeros([2], DType.float64)),
            throwsArgumentError,
          );

          // svdvals 1D error, promoted int32, invalid out
          expect(() => svdvals(b), throwsArgumentError);
          final i32Mat = NDArray.fromList([3, 0, 0, 4], [2, 2], DType.int32);
          expect(svdvals(i32Mat).toList(), equals([4.0, 3.0]));
          expect(
            () => svdvals(eye2, out: NDArray.zeros([5], DType.float64)),
            throwsArgumentError,
          );

          // matrix_rank disposed, invalid out, 1D complex128, non-square hermitian,
          // empty [0, 0] with out, float32, and aliased out
          expect(() => matrix_rank(disposed), throwsStateError);
          expect(
            () => matrix_rank(
              eye2,
              out: NDArray.zeros([], DType.int64)..dispose(),
            ),
            throwsStateError,
          );
          expect(
            () => matrix_rank(eye2, out: NDArray.zeros([5], DType.int64)),
            throwsArgumentError,
          );
          final c128Vec = NDArray.fromList(
            [const Complex(0, 0), const Complex(0, 2)],
            [2],
            DType.complex128,
          );
          final rankOut = NDArray.zeros([], DType.int64);
          expect(matrix_rank(c128Vec, out: rankOut).scalar, equals(1));
          expect(() => matrix_rank(a, hermitian: true), throwsArgumentError);

          final empty00 = NDArray.zeros([2, 0, 0], DType.int64);
          final emptyRankOut = NDArray<Int64>.view(
            NDArray.zeros([4], DType.int64),
            shape: [2],
            strides: [2],
            offsetElements: 0,
          );
          expect(
            matrix_rank(empty00, out: emptyRankOut).toList(),
            equals([0, 0]),
          );

          final f32Eye = NDArray.eye(2, DType.float32);
          expect(matrix_rank(f32Eye).scalar, equals(2));

          final i64Eye = NDArray.fromList([1, 0, 0, 1], [2, 2], DType.int64);
          final aliasedI64Out = NDArray<Int64>.view(
            i64Eye,
            shape: [],
            strides: [],
            offsetElements: 0,
          );
          expect(matrix_rank(i64Eye, out: aliasedI64Out).scalar, equals(2));

          // tensorsolve & tensorinv validation and promotion
          final tA = NDArray.eye(4, DType.float64).reshape([2, 2, 2, 2]);
          final tB = NDArray.ones([2, 2], DType.float64);
          final tB32 = NDArray.ones([2, 2], DType.float32);
          expect(
            () => tensorsolve(tA.asAnySpec, tB32.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => tensorsolve(tB, NDArray.ones([2, 2, 2], DType.float64)),
            throwsArgumentError,
          );
          expect(
            () => tensorsolve(NDArray.zeros([2, 2, 2, 3], DType.float64), tB),
            throwsArgumentError,
          );
          expect(
            () =>
                tensorsolve(tA, tB, out: NDArray.zeros([3, 3], DType.float64)),
            throwsArgumentError,
          );
          expect(
            () => tensorinv(tA, out: NDArray.zeros([3, 3], DType.float64)),
            throwsArgumentError,
          );

          final i32TA = NDArray.eye(4, DType.int32).reshape([2, 2, 2, 2]);
          final i32TB = NDArray.ones([2, 2], DType.int32);
          expect(tensorsolve(i32TA, i32TB).dtype, equals(DType.float64));
          expect(tensorinv(i32TA).dtype, equals(DType.float64));
        });
      },
    );

    test('remaining edge branches across linalg', () {
      NDArray.scope(() {
        final v2 = NDArray.ones([2], DType.float64);
        final eye2 = NDArray.eye(2, DType.float64);
        final disposed = NDArray.eye(2, DType.float64)..dispose();

        // L343: 1D-1D matmul with wrong out shape
        expect(
          () => matmul(v2, v2, out: NDArray.zeros([1], DType.float64)),
          throwsArgumentError,
        );

        // L589: matmul where a has fewer stack dims than b (2D x 3D)
        final b3D = NDArray.ones([2, 2, 2], DType.float64);
        expect(matmul(eye2, b3D).shape, equals([2, 2, 2]));

        // L662 & L695: 3D transposed matrix x 1D vector and 1D vector x 3D transposed matrix
        final a3DT = b3D.transpose([0, 2, 1]);
        expect(matmul(a3DT, v2).shape, equals([2, 2]));
        expect(matmul(v2, a3DT).shape, equals([2, 2]));

        // L1048..1064: 3D boolean matmul in walk
        final bool3D = NDArray.fromList(
          [true, false, false, true],
          [1, 2, 2],
          DType.boolean,
        );
        expect(
          matmul(bool3D, bool3D).toList(),
          equals([true, false, false, true]),
        );

        // L1521, L1522, L3629, L3630: inv and cholesky with n == 0 (with and without out)
        final empty00 = NDArray.zeros([0, 0], DType.float64);
        final empty00Out = NDArray.zeros([0, 0], DType.float64);
        expect(inv(empty00).shape, equals([0, 0]));
        expect(inv(empty00, out: empty00Out).shape, equals([0, 0]));
        expect(cholesky(empty00).shape, equals([0, 0]));
        expect(cholesky(empty00, out: empty00Out).shape, equals([0, 0]));

        // L1774, L1775: det with aliased out
        final detOut = NDArray<Float64>.view(
          eye2,
          shape: [],
          strides: [],
          offsetElements: 0,
        );
        expect(det(eye2, out: detOut).scalar, closeTo(1.0, 1e-10));

        // L2064..2072: slogdet with aliased outSign and outLogdet
        final signView = NDArray<Float64>.view(
          eye2,
          shape: [],
          strides: [],
          offsetElements: 0,
        );
        final logdetView = NDArray<Float64>.view(
          eye2,
          shape: [],
          strides: [],
          offsetElements: 1,
        );
        slogdet(eye2, outSign: signView, outLogdet: logdetView);
        expect(signView.scalar, closeTo(1.0, 1e-10));
        expect(logdetView.scalar, closeTo(0.0, 1e-10));

        // L2468 & L2481: solve 3D a with 2D b, and mismatched dtypes via asAnySpec
        expect(() => solve(b3D, eye2), throwsArgumentError);
        final f32Vec = NDArray.ones([2], DType.float32);
        expect(
          () => solve(eye2.asAnySpec, f32Vec.asAnySpec),
          throwsArgumentError,
        );

        // L2729 & L3064: eig and eigvals with non-contiguous out
        final wNonContig = NDArray.zeros([
          4,
        ], DType.complex128).slice([Slice(step: 2)]);
        final vrNonContig = NDArray.zeros([
          2,
          4,
        ], DType.complex128).slice([Slice.all(), Slice(step: 2)]);
        eig(eye2, out: (eigenvalues: wNonContig, eigenvectors: vrNonContig));
        eigvals(eye2, out: wNonContig);

        // L3843: qr with aliased out.r
        final qBuf = NDArray.zeros([2, 2], DType.float64);
        final aQr = eye2.copy();
        qr(aQr, out: (q: qBuf, r: aQr));

        // L4230, L4231: svd with aliased out.vh
        final uBuf = NDArray.zeros([2, 2], DType.float64);
        final sBuf = NDArray.zeros([2], DType.float64);
        final aSvd = eye2.copy();
        svd(aSvd, out: (u: uBuf, s: sBuf, vh: aSvd));

        // L4355: svd with m < n and out != null
        final a2x3 = NDArray.ones([2, 3], DType.float64);
        final u2x2 = NDArray.zeros([2, 2], DType.float64);
        final s2 = NDArray.zeros([2], DType.float64);
        final vh2x3 = NDArray.zeros([2, 3], DType.float64);
        svd(a2x3, fullMatrices: false, out: (u: u2x2, s: s2, vh: vh2x3));
        expect(s2.getCell([0]), greaterThan(2.0));

        // L4612: svdvals with m == 0 || n == 0 across real and complex dtypes
        expect(
          svdvals(NDArray.zeros([0, 2], DType.float64)).shape,
          equals([0]),
        );
        expect(
          svdvals(NDArray.zeros([0, 2], DType.complex128)).shape,
          equals([0]),
        );
        expect(
          svdvals(NDArray.zeros([0, 2], DType.complex64)).shape,
          equals([0]),
        );

        // L4942, L5496, L5503, L5808, L5815: eigh, schur, hessenberg with Float16 out via asAnySpec
        final f16Mat = NDArray.fromList(
          [2.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float16,
        );
        final f16VecOut = NDArray.zeros([2], DType.float16);
        final f16MatOut1 = NDArray.zeros([2, 2], DType.float16);
        final f16MatOut2 = NDArray.zeros([2, 2], DType.float16);
        eigh(
          f16Mat.asAnySpec,
          outEigenvalues: f16VecOut.asAnySpec,
          outEigenvectors: f16MatOut1.asAnySpec,
        );
        schur(
          f16Mat.asAnySpec,
          outT: f16MatOut1.asAnySpec,
          outZ: f16MatOut2.asAnySpec,
        );
        hessenberg(
          f16Mat.asAnySpec,
          outH: f16MatOut1.asAnySpec,
          outQ: f16MatOut2.asAnySpec,
        );

        // L6078, L6103, L6104, L6249: outer disposed, aliased out, and 2D b
        expect(() => outer(disposed, v2), throwsStateError);
        expect(() => outer(v2, v2, out: disposed), throwsStateError);
        final outerNonContig = NDArray.zeros([
          2,
          4,
        ], DType.float64).slice([Slice.all(), Slice(step: 2)]);
        outer(v2, v2, out: outerNonContig);
        outer(v2, eye2);

        // L6288, L6376, L6377: cross mismatched dtypes via asAnySpec and aliased out
        final v3F64 = NDArray.fromList([1.0, 0.0, 0.0], [3], DType.float64);
        final v3F32 = NDArray.fromList([0.0, 1.0, 0.0], [3], DType.float32);
        expect(
          () => cross(v3F64.asAnySpec, v3F32.asAnySpec),
          throwsArgumentError,
        );
        final v3Copy = v3F64.copy();
        cross(v3Copy, v3F64, out: v3Copy);
        expect(v3Copy.toList(), equals([0.0, 0.0, 0.0]));

        // L6687, L6729, L6789: matmulAs, outerAs, crossAs with cast + out
        final i32Mat = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32Vec2 = NDArray.fromList([1, 2], [2], DType.int32);
        final i32Vec3 = NDArray.fromList([1, 0, 0], [3], DType.int32);
        final f64Mat2 = NDArray.zeros([2, 2], DType.float64);
        final f64Vec3 = NDArray.zeros([3], DType.float64);
        matmulAs(i32Mat, i32Mat, DType.float64, out: f64Mat2);
        outerAs(i32Vec2, i32Vec2, DType.float64, out: f64Mat2);
        crossAs(i32Vec3, i32Vec3, DType.float64, out: f64Vec3);

        // L6957, L6958, L7316, L7327: norm with aliased out and empty [0, 0] matrix with ord: 2 / -2
        final normOut = NDArray<Float64>.view(
          eye2,
          shape: [],
          strides: [],
          offsetElements: 0,
        );
        norm(eye2, out: normOut);
        expect(norm(empty00, ord: 2).scalar, equals(0.0));
        expect(norm(empty00, ord: -2).scalar, equals(0.0));
      });
    });
  });
}
