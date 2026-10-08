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
  group('Stream 3 Fixes', () {
    test('Task 1: einsum mixed dtypes and out buffer handling', () {
      NDArray.scope(() {
        // Mixed dtypes in einsum (int32 + float64) throw ArgumentError
        final aInt = NDArray<Int32>.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final bFloat = NDArray<Float64>.fromList(
          [0.5, 1.5, 2.5, 3.5],
          [2, 2],
          DType.float64,
        );
        expect(
          () => einsum(EinsumSubscripts.parse('ij,jk->ik'), [
            aInt.asAnySpec,
            bFloat.asAnySpec,
          ]),
          throwsArgumentError,
        );
        final aFloat = aInt.astype(DType.float64);
        final res = einsum(EinsumSubscripts.parse('ij,jk->ik'), [
          aFloat,
          bFloat,
        ]);
        expect(res.dtype, equals(DType.float64));
        expect(res.shape, equals([2, 2]));
        // [1*0.5 + 2*2.5, 1*1.5 + 2*3.5] = [5.5, 8.5]
        // [3*0.5 + 4*2.5, 3*1.5 + 4*3.5] = [11.5, 18.5]
        expect(res[[0, 0]], closeTo(5.5, 1e-9));
        expect(res[[0, 1]], closeTo(8.5, 1e-9));
        expect(res[[1, 0]], closeTo(11.5, 1e-9));
        expect(res[[1, 1]], closeTo(18.5, 1e-9));

        // User-supplied out buffer in einsum
        final outBuf = NDArray<Float64>.zeros([2, 2], DType.float64);
        final outRes = einsum(EinsumSubscripts.parse('ij,jk->ik'), [
          aFloat,
          bFloat,
        ], out: outBuf);
        expect(identical(outRes, outBuf), isTrue);
        expect(outRes[[0, 0]], closeTo(5.5, 1e-9));
      });
    });

    test('Task 2: norm ord == 0 returns count of non-zero elements', () {
      NDArray.scope(() {
        final v = NDArray<Float64>.fromList(
          [0.0, 3.0, 0.0, -5.0, 2.0],
          [5],
          DType.float64,
        );
        final norm0 = norm(v, ord: 0).scalar;
        expect(norm0, equals(3.0));

        final vAllZero = NDArray<Float64>.zeros([4], DType.float64);
        expect(norm(vAllZero, ord: 0).scalar, equals(0.0));

        final vFloat32 = NDArray<Float32>.fromList(
          [1.0, 0.0, 2.0],
          [3],
          DType.float32,
        );
        expect(norm(vFloat32, ord: 0).scalar, equals(2.0));
      });
    });

    test(
      'Task 3: User-supplied out buffer detachment in multi_dot and convolve and correlate',
      () {
        // multi_dot with out in nested scope
        final m1 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final m2 = NDArray<Float64>.fromList(
          [2.0, 0.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );
        final m3 = NDArray<Float64>.fromList(
          [1.0, 1.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        final outDot = NDArray<Float64>.zeros([2, 2], DType.float64);
        NDArray.scope(() {
          multi_dot<Float64>([m1, m2, m3], out: outDot);
        });
        // outDot should still be valid (not disposed by inner scope)
        expect(outDot.isDisposed, isFalse);
        expect(outDot.shape, equals([2, 2]));

        // convolve with out in nested scope
        final sig = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final kernel = NDArray<Float64>.fromList(
          [0.5, 1.0],
          [2],
          DType.float64,
        );
        final outConv = NDArray<Float64>.zeros([4], DType.float64);
        NDArray.scope(() {
          convolve<Float64>(sig, kernel, mode: ConvMode.full, out: outConv);
        });
        expect(outConv.isDisposed, isFalse);
        expect(outConv.shape, equals([4]));

        // correlate same with out in nested scope
        final outSame = NDArray<Float64>.zeros([3], DType.float64);
        NDArray.scope(() {
          correlate<Float64>(sig, kernel, mode: ConvMode.same, out: outSame);
        });
        expect(outSame.isDisposed, isFalse);
        expect(outSame.shape, equals([3]));

        outDot.dispose();
        outConv.dispose();
        outSame.dispose();
        m1.dispose();
        m2.dispose();
        m3.dispose();
        sig.dispose();
        kernel.dispose();
      },
    );

    test('Task 4: lstsq basic computation and s result', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
          [3, 2],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [6.0, 5.0, 7.0],
          [3],
          DType.float64,
        );
        final res = lstsq<Float64, Float64, Float64>(a, b);
        expect(res.x.shape, equals([2]));
        expect(res.rank, equals(2));
        expect(res.s.shape, equals([2]));
        expect(res.s[[0]], greaterThan(0.0));
      });
    });

    test('Task 5 & 6: angle, unwrap, and correlate same mode', () {
      NDArray.scope(() {
        final sig = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final kernel = NDArray<Float64>.fromList(
          [1.0, 1.0, 1.0],
          [3],
          DType.float64,
        );
        final corrSame = correlate<Float64>(sig, kernel, mode: ConvMode.same);
        expect(corrSame.shape, equals([5]));
        expect(corrSame.isContiguous, isTrue);

        final unwrapped = unwrap(sig);
        expect(unwrapped.shape, equals([5]));
      });
    });

    test('Task 7: schur typed record with .t and .z', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [5.0, 7.0, -2.0, -4.0],
          [2, 2],
          DType.float64,
        );
        final ({NDArray<Float64> t, NDArray<Float64> z}) res =
            schur<Float64, Float64>(a, output: SchurForm.real);
        expect(res.t.shape, equals([2, 2]));
        expect(res.z.shape, equals([2, 2]));
        expect(res.t[[1, 0]], closeTo(0.0, 1e-10));
      });
    });

    test('Task 9: fftfreq and rfftfreq argument validation', () {
      expect(() => fftfreq(0), throwsArgumentError);
      expect(() => fftfreq(-1), throwsArgumentError);
      expect(() => rfftfreq(0), throwsArgumentError);
      expect(() => rfftfreq(-10), throwsArgumentError);

      NDArray.scope(() {
        final f = fftfreq(8, d: 0.1);
        expect(f.shape, equals([8]));
        final rf = rfftfreq(8, d: 0.1);
        expect(rf.shape, equals([5]));
      });
    });

    test('Task 10: batched qr decomposition', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0, 0.0],
          [3, 2, 2],
          DType.float64,
        );

        final res = qr<Float64>(a);
        expect(res.q.shape, equals([3, 2, 2]));
        expect(res.r.shape, equals([3, 2, 2]));

        // Check Q * R = A for each batch
        for (var b = 0; b < 3; b++) {
          final qSlice = res.q
              .slice([
                Slice(start: b, stop: b + 1),
                const Slice(),
                const Slice(),
              ])
              .reshape([2, 2]);
          final rSlice = res.r
              .slice([
                Slice(start: b, stop: b + 1),
                const Slice(),
                const Slice(),
              ])
              .reshape([2, 2]);
          final aSlice = a
              .slice([
                Slice(start: b, stop: b + 1),
                const Slice(),
                const Slice(),
              ])
              .reshape([2, 2]);

          final recon = matmul(qSlice, rSlice);
          for (var r = 0; r < 2; r++) {
            for (var c = 0; c < 2; c++) {
              expect(recon[[r, c]], closeTo(aSlice[[r, c]], 1e-10));
            }
          }
        }
      });
    });

    test(
      "Task 11: matmul with non-contiguous 1D out buffers and shape demotion copy",
      () {
        NDArray.scope(() {
          // Matrix (2x3) * Vector (3) -> Vector (2), using a non-contiguous 1D out buffer
          final mat = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final vec = NDArray<Float64>.fromList(
            [1.0, 1.0, 1.0],
            [3],
            DType.float64,
          );

          final largerBuf = NDArray<Float64>.zeros([4], DType.float64);
          final nonContigOut = largerBuf.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]);
          expect(nonContigOut.isContiguous, isFalse);
          expect(nonContigOut.shape, equals([2]));

          final res = matmul(mat, vec, out: nonContigOut);
          expect(identical(res, nonContigOut), isTrue);
          expect(res[[0]], closeTo(6.0, 1e-9));
          expect(res[[1]], closeTo(15.0, 1e-9));
          expect(largerBuf[[0]], closeTo(6.0, 1e-9));
          expect(largerBuf[[2]], closeTo(15.0, 1e-9));

          // Vector dot product (1D * 1D -> 0D scalar) with 0D out buffer
          final out0D = NDArray<Float64>.zeros([], DType.float64);
          final dotRes = matmul(vec, vec, out: out0D);
          expect(identical(dotRes, out0D), isTrue);
          expect(dotRes.scalar, closeTo(3.0, 1e-9));
        });
      },
    );

    test("Task 12: rfft and irfft odd-length fallback with out buffer", () {
      NDArray.scope(() {
        final input = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final outRfft = NDArray.zeros([3], DType.complex128);
        final resRfft = rfft(input, n: 5, out: outRfft);
        expect(identical(resRfft, outRfft), isTrue);
        expect(resRfft.isDisposed, isFalse);

        final outIrfft = NDArray<Float64>.zeros([5], DType.float64);
        final resIrfft = irfft(resRfft, n: 5, out: outIrfft);
        expect(identical(resIrfft, outIrfft), isTrue);
        expect(resIrfft.isDisposed, isFalse);
        expect(resIrfft[[0]], closeTo(1.0, 1e-6));
      });
    });
    test('Cycle 11 Task 1: eigh named record return and EighRecordDispose', () {
      final a = NDArray<Float64>.fromList(
        [2.0, 1.0, 1.0, 2.0],
        [2, 2],
        DType.float64,
      );
      final res = eigh(a);
      expect(res.eigenvalues.shape, equals([2]));
      expect(res.eigenvectors.shape, equals([2, 2]));
      expect(res.eigenvalues[[0]], closeTo(1.0, 1e-9));
      expect(res.eigenvalues[[1]], closeTo(3.0, 1e-9));
      expect(res.eigenvalues.isDisposed, isFalse);
      expect(res.eigenvectors.isDisposed, isFalse);
      res.dispose();
      expect(res.eigenvalues.isDisposed, isTrue);
      expect(res.eigenvectors.isDisposed, isTrue);
      a.dispose();
    });

    test(
      'Cycle 11 Task 2: Standardized lowercase fields and generic RecordDispose extensions',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final qrRes = qr<Float64>(a);
          expect(qrRes.q.shape, equals([2, 2]));
          expect(qrRes.r.shape, equals([2, 2]));
          qrRes.dispose();
          expect(qrRes.q.isDisposed, isTrue);
          expect(qrRes.r.isDisposed, isTrue);

          final svdRes = svd(a);
          expect(svdRes.u.shape, equals([2, 2]));
          expect(svdRes.s.shape, equals([2]));
          expect(svdRes.vh.shape, equals([2, 2]));
          svdRes.dispose();
          expect(svdRes.u.isDisposed, isTrue);
          expect(svdRes.s.isDisposed, isTrue);
          expect(svdRes.vh.isDisposed, isTrue);

          final hessRes = hessenberg(a);
          expect(hessRes.h.shape, equals([2, 2]));
          expect(hessRes.q.shape, equals([2, 2]));
          hessRes.dispose();
          expect(hessRes.h.isDisposed, isTrue);
          expect(hessRes.q.isDisposed, isTrue);
        });
      },
    );

    test(
      'Cycle 11 Task 3: fftn out of bounds axis check order when s is null',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(() => fftn(a, axes: [5]), throwsRangeError);
          expect(() => fftn(a, axes: [-5]), throwsRangeError);
        });
      },
    );
    test(
      'Cycle 12 Stream 3 Task 1: Zero-Sized Matrix (M=0 or N=0) Fast Paths',
      () {
        NDArray.scope(() {
          final mat00 = NDArray<Float64>.zeros([0, 0], DType.float64);
          final mat20 = NDArray<Float64>.zeros([2, 0], DType.float64);
          final mat03 = NDArray<Float64>.zeros([0, 3], DType.float64);

          // inv
          final inv00 = inv(mat00);
          expect(inv00.shape, equals([0, 0]));

          // det
          final det00 = det(mat00);
          expect(det00.scalar, equals(1.0));

          // slogdet
          final slog00 = slogdet(mat00);
          expect(slog00.sign.scalar, equals(1.0));
          expect(slog00.logabsdet.scalar, equals(0.0));

          // cholesky
          final chol00 = cholesky(mat00);
          expect(chol00.shape, equals([0, 0]));

          // schur
          final schur00 = schur(mat00);
          expect(schur00.t.shape, equals([0, 0]));
          expect(schur00.z.shape, equals([0, 0]));

          // pinv
          final pinv20 = pinv(mat20);
          expect(pinv20.shape, equals([0, 2]));

          // svd
          final svd20 = svd(mat20);
          expect(svd20.u.shape, equals([2, 2]));
          expect(svd20.s.shape, equals([0]));
          expect(svd20.vh.shape, equals([0, 0]));
          expect(svd20.u[[0, 0]], equals(1.0));
          expect(svd20.u[[0, 1]], equals(0.0));

          // qr
          final qr03 = qr(mat03);
          expect(qr03.q.shape, equals([0, 0]));
          expect(qr03.r.shape, equals([0, 3]));

          // lstsq
          final lstsq20 = lstsq(
            mat20,
            NDArray<Float64>.zeros([2], DType.float64),
          );
          expect(lstsq20.x.shape, equals([0]));
          expect(lstsq20.residuals.shape, equals([0]));
          expect(lstsq20.rank, equals(0));

          // eigh
          final eigh00 = eigh(mat00);
          expect(eigh00.eigenvalues.shape, equals([0]));
          expect(eigh00.eigenvectors.shape, equals([0, 0]));

          // hessenberg
          final hess00 = hessenberg(mat00);
          expect(hess00.h.shape, equals([0, 0]));
          expect(hess00.q.shape, equals([0, 0]));
        });
      },
    );

    test(
      'Cycle 12 Stream 3 Task 2: Generic <T> Alignment on Records & Dispose Extensions',
      () {
        NDArray.scope(() {
          final mat = NDArray<Float64>.fromList(
            [2.0, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );

          // eigh return generic <T> check
          final ({NDArray<AnySpec> eigenvalues, NDArray<Float64> eigenvectors})
          resEigh = eigh<Float64, Float64>(mat);
          expect(resEigh.eigenvectors.dtype, equals(DType.float64));

          // hessenberg return generic <T> check
          final ({NDArray<Float64> h, NDArray<Float64> q}) resHess = hessenberg(
            mat,
          );
          expect(resHess.h.dtype, equals(DType.float64));
          expect(resHess.q.dtype, equals(DType.float64));

          // SchurRecordDispose<T> generic check
          final ({NDArray<Float64> t, NDArray<Float64> z}) resSchur =
              schur<Float64, Float64>(mat);
          resSchur.dispose();
          expect(resSchur.t.isDisposed, isTrue);
          expect(resSchur.z.isDisposed, isTrue);
        });
      },
    );

    test(
      'Cycle 12 Stream 3 Task 3: Scoping Invariants in Factorizations (schur & lstsq)',
      () {
        NDArray.scope(() {
          final mat = NDArray<Float64>.fromList(
            [4.0, -1.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final vec = NDArray<Float64>.fromList([5.0, 3.0], [2], DType.float64);

          final res = schur<Float64, Float64>(mat);
          // Ensure t and z survive schur's internal NDArray.scope
          expect(res.t.isDisposed, isFalse);
          expect(res.z.isDisposed, isFalse);
          expect(res.t.shape, equals([2, 2]));

          final lstsqRes = lstsq<Float64, Float64, Float64>(mat, vec);
          // Ensure x, residuals, and s survive lstsq's internal NDArray.scope
          expect(lstsqRes.x.isDisposed, isFalse);
          expect(lstsqRes.residuals.isDisposed, isFalse);
          expect(lstsqRes.s.isDisposed, isFalse);
          expect(lstsqRes.x.shape, equals([2]));
        });
      },
    );
  });
}
