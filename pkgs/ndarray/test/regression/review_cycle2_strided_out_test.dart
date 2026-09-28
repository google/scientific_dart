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

void expectArrayClose(NDArray actual, NDArray expected, {double tol = 1e-5}) {
  expect(actual.shape, equals(expected.shape));
  final aList = actual.toList();
  final eList = expected.toList();
  for (var i = 0; i < aList.length; i++) {
    final av = aList[i];
    final ev = eList[i];
    if (av is Complex && ev is Complex) {
      expect(av.real, closeTo(ev.real, tol), reason: 'at index $i real');
      expect(av.imag, closeTo(ev.imag, tol), reason: 'at index $i imag');
    } else if (av is num && ev is num) {
      expect(av.toDouble(), closeTo(ev.toDouble(), tol), reason: 'at index $i');
    } else {
      fail('Incompatible element types at $i: $av vs $ev');
    }
  }
}

void main() {
  group('Review Cycle 2: Integer inputs & Strided/Aliased out buffers', () {
    group('Calculus: gradient & gradientArray', () {
      test(
        'supports integer inputs (int32, int64) by promoting to float64',
        () {
          NDArray.scope(() {
            final fInt32 = NDArray.fromList([1, 4, 9, 16], [4], DType.int32);
            final fInt64 = NDArray.fromList([1, 4, 9, 16], [4], DType.int64);
            final fFloat64 = NDArray.fromList(
              [1.0, 4.0, 9.0, 16.0],
              [4],
              DType.float64,
            );

            final expected = gradient(fFloat64);
            final res32 = gradient(fInt32);
            final res64 = gradient(fInt64);

            expect(res32.dtype, equals(DType.float64));
            expect(res64.dtype, equals(DType.float64));
            expectArrayClose(res32, expected);
            expectArrayClose(res64, expected);

            final arr32 = gradientArray(fInt32);
            expect(arr32[0].dtype, equals(DType.float64));
            expectArrayClose(arr32[0], expected);
          });
        },
      );

      test(
        'supports strided and aliased out buffers in gradient and gradientArray',
        () {
          NDArray.scope(() {
            final f = NDArray.fromList(
              [1.0, 4.0, 9.0, 16.0],
              [4],
              DType.float64,
            );
            final expected = gradient(f);

            // Strided out buffer (step 2 slice)
            final fullOut = NDArray.zeros([8], DType.float64);
            final stridedOut = fullOut.slice([
              Slice(start: 0, stop: 8, step: 2),
            ]);
            expect(stridedOut.isContiguous, isFalse);

            final res = gradient(f, out: stridedOut);
            expect(identical(res, stridedOut), isTrue);
            expectArrayClose(stridedOut, expected);

            // Aliased out buffer (in-place)
            final fCopy = f.copy();
            gradient(fCopy, out: fCopy);
            expectArrayClose(fCopy, expected);

            // Multi-axis aliased out protection in gradientArray
            final f2d = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2, 3],
              DType.float64,
            );
            final expected2d = gradientArray(f2d);
            final f2dCopy = f2d.copy();
            final out1 = NDArray.zeros([2, 3], DType.float64);
            final outList = gradientArray(f2dCopy, out: [f2dCopy, out1]);
            expectArrayClose(outList[0], expected2d[0]);
            expectArrayClose(outList[1], expected2d[1]);
          });
        },
      );
    });

    group('FFT operations with strided and aliased out buffers', () {
      test('fft & ifft support strided and aliased out', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 0), Complex(2, -1), Complex(0, -1), Complex(-1, 2)],
            [4],
            DType.complex128,
          );
          final expectedFft = fft(a);
          final expectedIfft = ifft(a);

          // Strided out
          final full = NDArray.zeros([8], DType.complex128);
          final strided = full.slice([Slice(start: 0, stop: 8, step: 2)]);
          expect(strided.isContiguous, isFalse);

          fft(a, out: strided);
          expectArrayClose(strided, expectedFft);

          ifft(a, out: strided);
          expectArrayClose(strided, expectedIfft);

          // Aliased out
          final aCopy1 = a.copy();
          fft(aCopy1, out: aCopy1);
          expectArrayClose(aCopy1, expectedFft);

          final aCopy2 = a.copy();
          ifft(aCopy2, out: aCopy2);
          expectArrayClose(aCopy2, expectedIfft);
        });
      });

      test('rfft & irfft support strided out', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final expectedRfft = rfft(a);

          final fullC = NDArray.zeros([6], DType.complex128);
          final stridedC = fullC.slice([Slice(start: 0, stop: 6, step: 2)]);
          expect(stridedC.isContiguous, isFalse);

          rfft(a, out: stridedC);
          expectArrayClose(stridedC, expectedRfft);

          final expectedIrfft = irfft(expectedRfft, n: 4);
          final fullR = NDArray.zeros([8], DType.float64);
          final stridedR = fullR.slice([Slice(start: 0, stop: 8, step: 2)]);
          expect(stridedR.isContiguous, isFalse);

          irfft(expectedRfft, n: 4, out: stridedR);
          expectArrayClose(stridedR, expectedIrfft);
        });
      });

      test('fftn & ifftn support strided and aliased out', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 0), Complex(2, 0), Complex(3, 0), Complex(4, 0)],
            [2, 2],
            DType.complex128,
          );
          final expectedFftn = fftn(a);
          final expectedIfftn = ifftn(a);

          final full = NDArray.zeros([2, 4], DType.complex128);
          final strided = full.slice([
            Slice.all(),
            Slice(start: 0, stop: 4, step: 2),
          ]);
          expect(strided.isContiguous, isFalse);

          fftn(a, out: strided);
          expectArrayClose(strided, expectedFftn);

          ifftn(a, out: strided);
          expectArrayClose(strided, expectedIfftn);

          final aCopy = a.copy();
          fftn(aCopy, out: aCopy);
          expectArrayClose(aCopy, expectedFftn);
        });
      });
    });

    group('Linear Algebra operations with strided and aliased out buffers', () {
      test('inv, det, slogdet, solve support strided and aliased out', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [4.0, 7.0, 2.0, 6.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList([1.0, 2.0], [2], DType.float64);

          // inv
          final expInv = inv(a);
          final full2D = NDArray.zeros([2, 4], DType.float64);
          final strided2D = full2D.slice([
            Slice.all(),
            Slice(start: 0, stop: 4, step: 2),
          ]);
          expect(strided2D.isContiguous, isFalse);
          inv(a, out: strided2D);
          expectArrayClose(strided2D, expInv);

          final aCopyInv = a.copy();
          inv(aCopyInv, out: aCopyInv);
          expectArrayClose(aCopyInv, expInv);

          // det (batched so out is 1D and can be strided)
          final aBatch = NDArray.fromList(
            [4.0, 7.0, 2.0, 6.0, 1.0, 2.0, 3.0, 4.0],
            [2, 2, 2],
            DType.float64,
          );
          final expDet = det(aBatch);
          final full1D = NDArray.zeros([4], DType.float64);
          final strided1D = full1D.slice([Slice(start: 0, stop: 4, step: 2)]);
          expect(strided1D.isContiguous, isFalse);
          det(aBatch, out: strided1D);
          expectArrayClose(strided1D, expDet);

          // slogdet
          final expSlogdet = slogdet(aBatch);
          final fullSign = NDArray.zeros([4], DType.float64);
          final stridedSign = fullSign.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]);
          final fullLog = NDArray.zeros([4], DType.float64);
          final stridedLog = fullLog.slice([Slice(start: 0, stop: 4, step: 2)]);
          slogdet(aBatch, outSign: stridedSign, outLogdet: stridedLog);
          expectArrayClose(stridedSign, expSlogdet.sign);
          expectArrayClose(stridedLog, expSlogdet.logabsdet);

          // solve
          final expSolve = solve(a, b);
          final fullSolve = NDArray.zeros([4], DType.float64);
          final stridedSolve = fullSolve.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]);
          solve(a, b, out: stridedSolve);
          expectArrayClose(stridedSolve, expSolve);

          final bCopy = b.copy();
          solve(a, bCopy, out: bCopy);
          expectArrayClose(bCopy, expSolve);
        });
      });

      test('eig, eigvals, eigh, eigvalsh support strided and aliased out', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [2.0, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );

          // eig
          final expEig = eig(a);
          final fullValC = NDArray.zeros([4], DType.complex128);
          final stridedValC = fullValC.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]);
          final fullVecC = NDArray.zeros([2, 4], DType.complex128);
          final stridedVecC = fullVecC.slice([
            Slice.all(),
            Slice(start: 0, stop: 4, step: 2),
          ]);
          eig(a, out: (eigenvalues: stridedValC, eigenvectors: stridedVecC));
          expectArrayClose(stridedValC, expEig.eigenvalues);
          expectArrayClose(stridedVecC, expEig.eigenvectors);

          // eigvals
          final expEigvals = eigvals(a);
          final fullEigvalsC = NDArray.zeros([4], DType.complex128);
          final stridedEigvalsC = fullEigvalsC.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]);
          eigvals(a, out: stridedEigvalsC);
          expectArrayClose(stridedEigvalsC, expEigvals);

          // eigh
          final expEigh = eigh(a);
          final fullValR = NDArray.zeros([4], DType.float64);
          final stridedValR = fullValR.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]);
          final fullVecR = NDArray.zeros([2, 4], DType.float64);
          final stridedVecR = fullVecR.slice([
            Slice.all(),
            Slice(start: 0, stop: 4, step: 2),
          ]);
          eigh(a, outEigenvalues: stridedValR, outEigenvectors: stridedVecR);
          expectArrayClose(stridedValR, expEigh.eigenvalues);
          expectArrayClose(stridedVecR, expEigh.eigenvectors);

          // eigvalsh
          final expEigvalsh = eigvalsh(a);
          final fullEigvalshR = NDArray.zeros([4], DType.float64);
          final stridedEigvalshR = fullEigvalshR.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]);
          eigvalsh(a, out: stridedEigvalshR);
          expectArrayClose(stridedEigvalshR, expEigvalsh);
        });
      });

      test(
        'cholesky, qr, svd, pinv, schur, hessenberg support strided and aliased out',
        () {
          NDArray.scope(() {
            final aPosDef = NDArray.fromList(
              [4.0, 2.0, 2.0, 5.0],
              [2, 2],
              DType.float64,
            );

            // cholesky
            final expChol = cholesky(aPosDef);
            final full2D = NDArray.zeros([2, 4], DType.float64);
            final strided2D = full2D.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            cholesky(aPosDef, out: strided2D);
            expectArrayClose(strided2D, expChol);

            final aCholCopy = aPosDef.copy();
            cholesky(aCholCopy, out: aCholCopy);
            expectArrayClose(aCholCopy, expChol);

            // qr
            final expQr = qr(aPosDef);
            final fullQ = NDArray.zeros([2, 4], DType.float64);
            final stridedQ = fullQ.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            final fullR = NDArray.zeros([2, 4], DType.float64);
            final stridedR = fullR.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            qr(aPosDef, out: (q: stridedQ, r: stridedR));
            expectArrayClose(stridedQ, expQr.q);
            expectArrayClose(stridedR, expQr.r);

            // svd
            final expSvd = svd(aPosDef);
            final fullU = NDArray.zeros([2, 4], DType.float64);
            final stridedU = fullU.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            final fullS = NDArray.zeros([4], DType.float64);
            final stridedS = fullS.slice([Slice(start: 0, stop: 4, step: 2)]);
            final fullVh = NDArray.zeros([2, 4], DType.float64);
            final stridedVh = fullVh.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            svd(aPosDef, out: (u: stridedU, s: stridedS, vh: stridedVh));
            expectArrayClose(stridedU, expSvd.u);
            expectArrayClose(stridedS, expSvd.s);
            expectArrayClose(stridedVh, expSvd.vh);

            // pinv
            final expPinv = pinv(aPosDef);
            final fullPinv = NDArray.zeros([2, 4], DType.float64);
            final stridedPinv = fullPinv.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            pinv(aPosDef, out: stridedPinv);
            expectArrayClose(stridedPinv, expPinv);

            // schur
            final expSchur = schur(aPosDef);
            final fullT = NDArray.zeros([2, 4], DType.float64);
            final stridedT = fullT.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            final fullZ = NDArray.zeros([2, 4], DType.float64);
            final stridedZ = fullZ.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            schur(aPosDef, outT: stridedT, outZ: stridedZ);
            expectArrayClose(stridedT, expSchur.t);
            expectArrayClose(stridedZ, expSchur.z);

            // hessenberg
            final expHess = hessenberg(aPosDef);
            final fullH = NDArray.zeros([2, 4], DType.float64);
            final stridedH = fullH.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            final fullQh = NDArray.zeros([2, 4], DType.float64);
            final stridedQh = fullQh.slice([
              Slice.all(),
              Slice(start: 0, stop: 4, step: 2),
            ]);
            hessenberg(aPosDef, outH: stridedH, outQ: stridedQh);
            expectArrayClose(stridedH, expHess.h);
            expectArrayClose(stridedQh, expHess.q);
          });
        },
      );
    });
  });
}
