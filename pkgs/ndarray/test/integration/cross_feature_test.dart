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

import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;

import 'package:ndarray/ndarray.dart';
import 'package:openblas/openblas.dart';
import 'package:pocketfft/pocketfft.dart';
import 'package:test/test.dart';

const bool _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

/// Root for files written on dart2wasm, where `dart:io` is unavailable.
///
/// `tool/build_wasm.dart` passes a per-target directory and deletes it after
/// the run; the default only applies when a test is run some other way.
const String _wasmTempRoot = String.fromEnvironment(
  'NDARRAY_TEST_TMPDIR',
  defaultValue: '/tmp',
);

void main() {
  Directory? tempDir;
  late String tempDirPath;

  group('Cross-feature interactions', () {
    setUpAll(() {
      if (_isWasm) {
        // save()/savez() create missing parent directories natively.
        tempDirPath =
            '$_wasmTempRoot/ndarray_cross_feature_test_${DateTime.now().microsecondsSinceEpoch}';
      } else {
        tempDir = Directory.systemTemp.createTempSync(
          'ndarray_cross_feature_test_',
        );
        tempDirPath = tempDir!.path;
      }
    });

    tearDownAll(() {
      if (!_isWasm && tempDir != null && tempDir!.existsSync()) {
        tempDir!.deleteSync(recursive: true);
      }
    });

    test('PocketFFTPlanCache and high-level fft/rfft plan cache lifecycle', () {
      NDArray.scope(() {
        clearFFTPlanCache();
        expect(PocketFFTPlanCache.instance.size, equals(0));

        final x = linspace<Float64>(0.0, 7.0, 8, dtype: DType.float64);
        final spec = rfft<Complex128>(x);
        expect(PocketFFTPlanCache.instance.size, greaterThan(0));

        final rec = irfft<Float64>(spec, n: 8);
        expect(allClose(rec, x, atol: 1e-12), isTrue);

        // Raw pocketfft plan lookup for the same length reuses the cached plan
        final cachedRealFwd = getCachedKissFFTRPlan(8, isInverse: false);
        expect(cachedRealFwd.address, isNot(0));

        clearFFTPlanCache();
        expect(PocketFFTPlanCache.instance.size, equals(0));
      });
    });

    test(
      'Raw CBLAS cblas_dgemm on NDArray pointers matches high-level matmul & solve',
      () {
        NDArray.scope(() {
          const cblasRowMajor = 101;
          const cblasNoTrans = 111;
          final a = NDArray.fromList(
            [4.0, 1.0, 2.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final cRaw = NDArray.zeros([2, 2], DType.float64);

          cblas_dgemm(
            cblasRowMajor,
            cblasNoTrans,
            cblasNoTrans,
            2,
            2,
            2,
            1.0,
            a.pointer.cast<ffi.Double>(),
            2,
            b.pointer.cast<ffi.Double>(),
            2,
            0.0,
            cRaw.pointer.cast<ffi.Double>(),
            2,
          );

          final cHigh = matmul(a, b);
          expect(allClose(cRaw, cHigh, atol: 1e-12), isTrue);

          // Verify solve(a, cHigh) recovers b
          final recoveredB = solve(a, cHigh);
          expect(allClose(recoveredB, b, atol: 1e-12), isTrue);
        });
      },
    );

    test(
      'NDIter.broadcast2 with ScratchArena temporary buffer inside nested scope',
      () {
        NDArray.scope(() {
          final col = NDArray.fromList([1.0, 2.0, 3.0], [3, 1], DType.float64);
          final row = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [1, 4],
            DType.float64,
          );

          final result = NDArray.scope(() {
            final marker = ScratchArena.marker;
            try {
              final scratch = ScratchArena.allocate<ffi.Double>(
                12 * ffi.sizeOf<ffi.Double>(),
              );
              final it = NDIter.broadcast2(col, row);
              var k = 0;
              while (it.moveNext()) {
                scratch[k++] =
                    col.getCellRaw(it.getIndex(0)) +
                    row.getCellRaw(it.getIndex(1));
              }
              final out = NDArray.create([3, 4], DType.float64);
              for (var i = 0; i < 12; i++) {
                out.setCellRaw(i, scratch[i]);
              }
              return out.detachToParentScope();
            } finally {
              ScratchArena.reset(marker);
            }
          });

          expect(result.shape, equals([3, 4]));
          expect(allClose(result, add<Float64>(col, row)), isTrue);
        });
      },
    );

    test(
      'Non-contiguous transposed and strided views in matmul, svd, qr, and solve',
      () {
        NDArray.scope(() {
          final raw = NDArray.fromList(
            [4.0, 0.0, 1.0, 0.0, 0.0, 9.0, 0.0, 9.0, 2.0, 0.0, 5.0, 0.0],
            [3, 4],
            DType.float64,
          );
          // Extract 2x2 non-contiguous submatrix [[4, 1], [2, 5]] via step-2 slicing
          final sub = raw.slice([
            const Slice(start: 0, stop: 3, step: 2),
            const Slice(start: 0, stop: 4, step: 2),
          ]);
          expect(sub.isContiguous, isFalse);
          expect(sub.shape, equals([2, 2]));

          final subT = sub.transposed;
          final qrRes = qr(subT);
          expect(allClose(matmul(qrRes.q, qrRes.r), subT, atol: 1e-12), isTrue);

          final svdRes = svd<Float64, Float64>(sub);
          final recon = matmul(matmul(svdRes.u, diag(svdRes.s)), svdRes.vh);
          expect(allClose(recon, sub, atol: 1e-12), isTrue);
        });
      },
    );

    test('FFT and RFFT on negative-stride reversed and transposed views', () {
      NDArray.scope(() {
        final base = linspace<Float64>(
          1.0,
          16.0,
          16,
          dtype: DType.float64,
        ).reshape([4, 4]);
        final revRows = base.slice([const Slice(step: -1), const Slice.all()]);
        expect(revRows.isContiguous, isFalse);

        final spec2D = fft2<Complex128>(revRows);
        final rec2D = real(ifft2<Complex128>(spec2D));
        expect(allClose(rec2D, revRows, atol: 1e-11), isTrue);

        final rSpec2D = rfft2<Complex128>(revRows.transposed);
        final rRec2D = irfft2<Float64>(rSpec2D, s: [4, 4]);
        expect(allClose(rRec2D, revRows.transposed, atol: 1e-11), isTrue);
      });
    });

    test(
      'SendableNDArray copy and borrow views into sorting and statistical reductions',
      () {
        NDArray.scope(() {
          final base = NDArray.fromList(
            [50.0, 10.0, 40.0, 20.0, 30.0],
            [5],
            DType.float64,
          );
          final sendCopy = SendableNDArray.fromCopy(base);
          final matCopy = sendCopy.materialize();

          expect(median<Float64>(matCopy).scalar, closeTo(30.0, 1e-12));
          expect(
            sort(matCopy).toList(),
            equals([10.0, 20.0, 30.0, 40.0, 50.0]),
          );

          final sendBorrow = SendableNDArray.unsafeBorrow(base);
          final viewBorrow = sendBorrow.materializeView();
          expect(mean(viewBorrow).scalar, closeTo(30.0, 1e-12));
          expect(argmax(viewBorrow).scalar, equals(0));
        });
      },
    );

    test(
      'NDArray.returning with compressed .npz roundtrip and scope escape',
      () {
        NDArray.scope(() {
          final escaped = NDArray.returning(() {
            final a = linspace<Float64>(0.0, 10.0, 11, dtype: DType.float64);
            final b = square(a);
            final path = '$tempDirPath/returning_roundtrip.npz';
            savez(path, {'a': a, 'b': b}, compressed: true);
            final loaded = loadz(path);
            return loaded['b']! as NDArray<Float64>;
          });
          expect(escaped.isDisposed, isFalse);
          expect(escaped[[10]], closeTo(100.0, 1e-12));
        });
      },
    );

    test(
      'Standard normal CDF via erf matches empirical quantiles and correlation',
      () {
        NDArray.scope(() {
          // Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
          final z = NDArray.fromList(
            [-1.95996398, 0.0, 1.95996398],
            [3],
            DType.float64,
          );
          final NDArray<Float64> scaled = divide(
            z,
            NDArray.scalar(math.sqrt2, dtype: DType.float64),
          );
          final cdf = multiply<Float64>(
            add<Float64>(
              NDArray.scalar(1.0, dtype: DType.float64),
              erf(scaled),
            ),
            NDArray.scalar(0.5, dtype: DType.float64),
          );
          expect(cdf[[0]], closeTo(0.025, 1e-6));
          expect(cdf[[1]], closeTo(0.500, 1e-6));
          expect(cdf[[2]], closeTo(0.975, 1e-6));
        });
      },
    );

    test('Broadcasted binaryUfunc with where: mask and padded output buffer', () {
      NDArray.scope(() {
        final core = NDArray.ones([2, 2], DType.float64);
        final padded = pad<Float64>(
          core,
          PadWidth.all(1),
          mode: PaddingMode.constant,
          constantValues: PadValues<Float64>.all(0.0),
        );
        expect(padded.shape, equals([4, 4]));

        final rowScale = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4, 1],
          DType.float64,
        );
        final mask = greater(padded, NDArray.scalar(0.0, dtype: DType.float64));
        final out = NDArray.full([4, 4], -1.0, dtype: DType.float64);

        binaryUfunc<Float64, Float64>(
          padded,
          broadcastTo(rowScale, [4, 4]),
          op: BinaryOp.add,
          where: mask,
          out: out,
        );
        // Border elements where mask is false remain -1.0; interior [1,1] is 1 + 20 = 21.0
        expect(out[[0, 0]], equals(-1.0));
        expect(out[[1, 1]], equals(21.0));
        expect(out[[2, 2]], equals(31.0));
      });
    });

    test(
      'Windowed sinusoid RFFT side-lobe suppression (Hanning vs Rectangular)',
      () {
        NDArray.scope(() {
          const n = 64;
          final t = NDArray.arange(0.0, n.toDouble(), dtype: DType.float64);
          // Non-integer bin frequency k = 10.5 causes spectral leakage
          final phase = multiply<Float64>(
            t,
            NDArray.scalar(2.0 * math.pi * 10.5 / n, dtype: DType.float64),
          );
          final sig = cos<Float64>(phase);
          final win = hanning<Float64>(n, dtype: DType.float64);
          final windowedSig = multiply<Float64>(sig, win);

          final magRect = abs<Float64>(rfft<Complex128>(sig));
          final magHann = abs<Float64>(rfft<Complex128>(windowedSig));

          // Distant bin k = 30 side-lobe energy relative to peak is much lower with Hanning window
          final rectRatio = magRect[[30]] / max(magRect).scalar;
          final hannRatio = magHann[[30]] / max(magHann).scalar;
          expect(hannRatio, lessThan(rectRatio * 0.1));
        });
      },
    );

    test(
      'Instantaneous frequency of chirp signal via angle, unwrap, gradient, and trapz',
      () {
        NDArray.scope(() {
          const n = 201;
          final t = linspace<Float64>(0.0, 2.0, n, dtype: DType.float64);
          const dt = 2.0 / (n - 1);
          // Phase φ(t) = 3*t + 2*t^2 -> instantaneous angular freq dφ/dt = 3 + 4*t
          final truePhase = add<Float64>(
            multiply<Float64>(t, NDArray.scalar(3.0, dtype: DType.float64)),
            multiply<Float64>(
              square(t),
              NDArray.scalar(2.0, dtype: DType.float64),
            ),
          );
          // Construct complex signal z(t) = cos(φ) + i*sin(φ)
          final cCos = cos<Float64>(truePhase).astype(DType.complex128);
          final cSin = multiply<Complex128>(
            sin<Float64>(truePhase).astype(DType.complex128),
            NDArray.scalar(Complex(0.0, 1.0), dtype: DType.complex128),
          );
          final z = add<Complex128>(cCos, cSin);

          final recoveredPhase = unwrap<Float64>(angle<Float64>(z));
          final instFreq = gradient<Float64>(
            recoveredPhase,
            spacing: const Spacing.step(dt),
          );
          // At midpoint t = 1.0 (index 100), dφ/dt == 3 + 4(1) = 7.0
          expect(instFreq[[100]], closeTo(7.0, 1e-3));

          // Integrating instFreq over [0, 2] via trapz recovers φ(2) - φ(0) = 6 + 8 = 14.0
          expect(
            trapz<Float64>(instFreq, spacing: const Spacing.step(dt)).scalar,
            closeTo(14.0, 1e-2),
          );
        });
      },
    );

    test(
      '2D row-wise argsort with take_along_axis and inverse put_along_axis',
      () {
        NDArray.scope(() {
          final m = NDArray.fromList(
            [30.0, 10.0, 20.0, 5.0, 25.0, 15.0],
            [2, 3],
            DType.float64,
          );
          final order = argsort(m, axis: 1);
          final sortedRows = take_along_axis(m, order, 1);
          expect(
            sortedRows.toList(),
            equals([10.0, 20.0, 30.0, 5.0, 15.0, 25.0]),
          );

          // Scatter sorted rows back into original positions using put_along_axis
          final restored = NDArray.zeros([2, 3], DType.float64);
          put_along_axis(restored, order, sortedRows, 1);
          expect(allClose(restored, m), isTrue);
        });
      },
    );

    test(
      'Sample covariance matrix via cov matches centered matmul and SVD singular values',
      () {
        NDArray.scope(() {
          // 4 observations in rows, 3 features in columns -> cov expects variables in rows (3x4)
          final xObs = NDArray.fromList(
            [1.0, 2.0, 0.5, 2.0, 3.0, 1.5, 3.0, 5.0, 2.0, 4.0, 6.0, 3.5],
            [4, 3],
            DType.float64,
          );
          final NDArray<Float64> colMeans = mean(xObs, axis: 0, keepdims: true);
          final centered = subtract<Float64>(xObs, colMeans);

          // Manual covariance: (X_c^T * X_c) / (N - 1)
          final covManual = divide(
            matmul(centered.transposed, centered),
            NDArray.scalar(3.0, dtype: DType.float64),
          );
          final covBuiltIn = cov(xObs.transposed);
          expect(allClose(covBuiltIn, covManual, atol: 1e-12), isTrue);

          // Eigenvalues of covManual equal s^2 / (N - 1) from SVD of centered
          final eigVals = sort(eigvalsh(covBuiltIn));
          final svdS = svd<Float64, Float64>(centered).s;
          final svdVar = sort(
            divide(square(svdS), NDArray.scalar(3.0, dtype: DType.float64)),
          );
          expect(allClose(eigVals, svdVar, atol: 1e-11), isTrue);
        });
      },
    );

    test(
      'RandomGenerator.randint empirical histogram matches bincount and uniqueAll',
      () {
        NDArray.scope(() {
          final rng = RandomGenerator(2026);
          final draws = rng.randint([500], low: 0, high: 5);
          final bc = bincount<Int64>(draws, minlength: 5);
          final u = uniqueAll(draws);

          expect(sum(bc).scalar, equals(500));
          expect(u.values.toList(), equals([0, 1, 2, 3, 4]));
          expect(u.counts.toList(), equals(bc.toList()));
        });
      },
    );

    test(
      'Kronecker product trace/determinant identities via einsum and det',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [2.0, 1.0, 1.0, 3.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [4.0, -1.0, 2.0, 5.0],
            [2, 2],
            DType.float64,
          );
          final k = kron(a, b);
          expect(k.shape, equals([4, 4]));

          // trace(A ⊗ B) == trace(A) * trace(B) via einsum('ii->')
          final trSub = EinsumSubscripts.parse('ii->');
          final trK = einsum(trSub, [k]).scalar;
          final trA = einsum(trSub, [a]).scalar;
          final trB = einsum(trSub, [b]).scalar;
          expect(trK, closeTo(trA * trB, 1e-12));

          // For 2x2 A and B: det(A ⊗ B) == det(A)^2 * det(B)^2
          final detA = det(a).scalar;
          final detB = det(b).scalar;
          expect(det(k).scalar, closeTo(detA * detA * detB * detB, 1e-10));
        });
      },
    );

    test(
      'Symmetric-padded valid convolution preserves constant signal boundaries',
      () {
        NDArray.scope(() {
          final sig = NDArray.full([8], 5.0, dtype: DType.float64);
          final padded = pad(sig, PadWidth.all(1), mode: PaddingMode.symmetric);
          final boxKernel = NDArray.full([3], 1.0 / 3.0, dtype: DType.float64);
          final filtered = convolve(padded, boxKernel, mode: ConvMode.valid);

          expect(filtered.shape, equals([8]));
          expect(allClose(filtered, sig, atol: 1e-12), isTrue);
          expect(max(abs<Float64>(diff(filtered))).scalar, closeTo(0.0, 1e-12));
        });
      },
    );

    test(
      'Circulant matrix multiplication via matmul matches circular convolution via FFT',
      () {
        NDArray.scope(() {
          // First column of 4x4 circulant matrix C: c = [4, 1, 2, 3]
          final c = NDArray.fromList([4.0, 1.0, 2.0, 3.0], [4], DType.float64);
          final C = NDArray.fromList(
            [
              4.0,
              3.0,
              2.0,
              1.0,
              1.0,
              4.0,
              3.0,
              2.0,
              2.0,
              1.0,
              4.0,
              3.0,
              3.0,
              2.0,
              1.0,
              4.0,
            ],
            [4, 4],
            DType.float64,
          );
          final x = NDArray.fromList([1.0, -1.0, 2.0, 0.5], [4], DType.float64);

          final yMatmul = matmul(C, x);
          final yFft = real(
            ifft<Complex128>(
              multiply<Complex128>(fft<Complex128>(c), fft<Complex128>(x)),
            ),
          );
          expect(allClose(yMatmul, yFft, atol: 1e-12), isTrue);
        });
      },
    );

    test(
      'Frobenius companion matrix eigvals match roots(p), and Vandermonde lstsq matches polyfit',
      () {
        NDArray.scope(() {
          // P(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
          final p = NDArray.fromList(
            [1.0, -6.0, 11.0, -6.0],
            [4],
            DType.float64,
          );
          final comp = NDArray.fromList(
            [6.0, -11.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [3, 3],
            DType.float64,
          );
          final ev = sort(real(eigvals(comp)));
          final rts = sort(real(roots(p) as NDArray<Complex128>));
          expect(allClose(ev, rts, atol: 1e-11), isTrue);
          expect(
            allClose(
              rts,
              NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64),
              atol: 1e-11,
            ),
            isTrue,
          );

          // Vandermonde lstsq vs polyfit
          final x = linspace<Float64>(0.0, 3.0, 6, dtype: DType.float64);
          final y = polyval<Float64, Float64, Float64>(p, x);
          final x2 = square(x);
          final x3 = multiply<Float64>(x2, x);
          final onesCol = NDArray.ones([6], DType.float64);
          final V = stack([x3, x2, x, onesCol], axis: 1);
          final lsCoeffs = lstsq<Float64, Float64, Float64>(V, y).x;
          final pfCoeffs = polyfit<Float64, Float64, Float64, Float64>(x, y, 3);
          expect(allClose(lsCoeffs, pfCoeffs, atol: 1e-10), isTrue);
        });
      },
    );

    test(
      'Random Gaussian matrix QR decomposition yields orthogonal Q with |det(Q)| == 1',
      () {
        NDArray.scope(() {
          final rng = RandomGenerator(31415);
          final g = rng.normal<Float64>([5, 5], dtype: DType.float64);
          final qrRes = qr(g);
          final qtq = matmul(qrRes.q.transposed, qrRes.q);
          expect(
            allClose(qtq, NDArray<Float64>.eye(5, DType.float64), atol: 1e-11),
            isTrue,
          );
          final sd = slogdet<Float64, Float64>(qrRes.q);
          expect(sd.sign.scalar.abs(), closeTo(1.0, 1e-11));
          expect(sd.logabsdet.scalar, closeTo(0.0, 1e-11));
        });
      },
    );

    test(
      'Linear convolution via zero-padded RFFT/IRFFT matches time-domain convolve',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, -2.0, 3.0, 4.0, 0.5],
            [5],
            DType.float64,
          );
          final b = NDArray.fromList([0.5, 1.5, -1.0, 2.0], [4], DType.float64);
          const fullLen = 5 + 4 - 1; // 8

          final timeConv = convolve(a, b, mode: ConvMode.full);
          final freqConv = irfft<Float64>(
            multiply<Complex128>(
              rfft<Complex128>(a, n: fullLen),
              rfft<Complex128>(b, n: fullLen),
            ),
            n: fullLen,
          );
          expect(allClose(freqConv, timeConv, atol: 1e-12), isTrue);
        });
      },
    );

    test(
      'Fourier spectral derivative matches analytical derivative and central gradient',
      () {
        NDArray.scope(() {
          const n = 64;
          const length = 2.0 * math.pi;
          const dx = length / n;
          final x = linspace<Float64>(
            0.0,
            length,
            n,
            endpoint: false,
            dtype: DType.float64,
          );
          // f(x) = sin(x) -> f'(x) = cos(x)
          final y = sin<Float64>(x);
          final exactDeriv = cos<Float64>(x);

          // Spectral derivative: ifft(i * k * fft(y))
          final k = fftfreq(n, d: dx / (2.0 * math.pi));
          final ik = multiply<Complex128>(
            k.astype(DType.complex128),
            NDArray.scalar(Complex(0.0, 1.0), dtype: DType.complex128),
          );
          final specDeriv = real(
            ifft<Complex128>(multiply<Complex128>(ik, fft<Complex128>(y))),
          );
          expect(allClose(specDeriv, exactDeriv, atol: 1e-11), isTrue);

          // Interior finite-difference gradient also matches cos(x) to O(dx^2)
          final fdGrad = gradient<Float64>(y, spacing: const Spacing.step(dx));
          expect(fdGrad[[16]], closeTo(exactDeriv[[16]], 1e-2));
        });
      },
    );

    test('Polynomial regression on noisy samples and linear interpolation', () {
      NDArray.scope(() {
        final rng = RandomGenerator(888);
        final x = linspace<Float64>(-1.0, 1.0, 41, dtype: DType.float64);
        final clean = exp<Float64>(x);
        final noise = rng.normal<Float64>(
          [41],
          loc: 0.0,
          scale: 1e-3,
          dtype: DType.float64,
        );
        final noisy = add<Float64>(clean, noise);

        final coeffs = polyfit<Float64, Float64, Float64, Float64>(x, noisy, 6);
        final fitted = polyval<Float64, Float64, Float64>(coeffs, x);
        expect(
          max(abs<Float64>(subtract<Float64>(fitted, clean))).scalar,
          lessThan(5e-3),
        );

        // Linear interpolation on dense grid recovers exp(x)
        final xGrid = linspace<Float64>(-1.0, 1.0, 81, dtype: DType.float64);
        final yGrid = exp<Float64>(xGrid);
        final yInterp = interp<Float64>(
          x,
          xGrid,
          yGrid,
          method: InterpolationMethod.linear,
        );
        expect(
          max(abs<Float64>(subtract<Float64>(yInterp, clean))).scalar,
          lessThan(5e-4),
        );
      });
    });

    test(
      'SVD factors saved to compressed .npz and reconstructed via matmul',
      () {
        NDArray.scope(() {
          final m = NDArray.fromList(
            [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
            [4, 3],
            DType.float64,
          );
          final svdRes = svd<Float64, Float64>(m);
          final uThin = svdRes.u.slice([
            const Slice.all(),
            const Slice(start: 0, stop: 3),
          ]);
          final path = '$tempDirPath/svd_factors.npz';
          savez(path, {
            'u': uThin,
            's': svdRes.s,
            'vh': svdRes.vh,
          }, compressed: true);

          final loaded = loadz(path);
          final u = loaded['u']! as NDArray<Float64>;
          final s = loaded['s']! as NDArray<Float64>;
          final vh = loaded['vh']! as NDArray<Float64>;

          final recon = matmul(matmul(u, diag(s)), vh);
          expect(allClose(recon, m, atol: 1e-11), isTrue);
        });
      },
    );

    test(
      'Multi-dtype promotion pipeline (uint8 -> int32 -> float16 -> float64 -> complex128)',
      () {
        NDArray.scope(() {
          final u8 = NDArray.fromList([1, 2, 3, 4], [4], DType.uint8);
          final i32 = u8.astype(DType.int32);
          final sqI32 = square(i32); // [1, 4, 9, 16]
          final f16 = sqI32.astype(DType.float16);
          final f64 = f16.astype(DType.float64);
          expect(cumsum(f64).toList(), equals([1.0, 5.0, 14.0, 30.0]));

          final c128 = f64.astype(DType.complex128);
          expect(sum(c128).scalar, equals(Complex(30.0, 0.0)));
        });
      },
    );
  });
}
