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
  group('Real-world scenarios', () {
    test(
      'Principal Component Analysis (PCA) via SVD and Eigendecomposition with Low-Rank Reconstruction',
      () {
        NDArray.scope(() {
          final rng = RandomGenerator(202601);
          const nSamples = 120;
          const nFeatures = 4;

          // Construct synthetic 4D dataset with 2 dominant latent factors
          final latent = rng.normal<Float64>(
            [nSamples, 2],
            loc: 0.0,
            scale: 1.0,
            dtype: DType.float64,
          );
          // Mixing matrix (2 x 4): strong variance along first two directions
          final mixing = NDArray.fromList(
            [3.0, 2.0, -1.0, 0.5, -0.5, 1.5, 2.5, -2.0],
            [2, nFeatures],
            DType.float64,
          );
          final noise = rng.normal<Float64>(
            [nSamples, nFeatures],
            loc: 0.0,
            scale: 0.05,
            dtype: DType.float64,
          );
          final rawData = add<Float64>(matmul(latent, mixing), noise);

          // 1. Center features to zero mean
          final NDArray<Float64> featureMeans = mean(
            rawData,
            axis: 0,
            keepdims: true,
          );
          final centered = subtract<Float64>(rawData, featureMeans);
          expect(
            max(abs<Float64>(mean(centered, axis: 0))).scalar,
            lessThan(1e-12),
          );

          // 2. Compute sample covariance matrix (4 x 4)
          final sampleCov = cov(centered.transposed);
          expect(sampleCov.shape, equals([nFeatures, nFeatures]));

          // 3. Eigendecomposition of symmetric covariance matrix (eigh returns ascending eigenvalues)
          final eighRes = eigh<Float64, Float64>(sampleCov);
          final eigValsDesc = flip(eighRes.eigenvalues);
          final eigVecsDesc = fliplr(eighRes.eigenvectors);

          // 4. Compare against SVD of centered data: s_i^2 / (N - 1) == lambda_i
          final svdRes = svd<Float64, Float64>(centered);
          final svdVariances = divide(
            square(svdRes.s),
            NDArray.scalar((nSamples - 1).toDouble(), dtype: DType.float64),
          );
          expect(allClose(svdVariances, eigValsDesc, atol: 1e-10), isTrue);

          // 5. Explained variance ratio: top 2 components capture > 99.5% of total variance
          final totalVar = sum(eigValsDesc).scalar;
          final NDArray<Float64> explainedRatios = divide(
            eigValsDesc,
            NDArray.scalar(totalVar, dtype: DType.float64),
          );
          final cumExplained = cumsum(explainedRatios);
          expect(cumExplained[[1]], greaterThan(0.995));
          expect(cumExplained[[3]], closeTo(1.0, 1e-12));

          // 6. Project onto top-2 principal components and reconstruct
          final top2Components = eigVecsDesc.slice([
            const Slice.all(),
            const Slice(start: 0, stop: 2),
          ]);
          final scores = matmul(centered, top2Components);
          expect(scores.shape, equals([nSamples, 2]));

          final reconstructedCentered = matmul(
            scores,
            top2Components.transposed,
          );
          final residual = subtract<Float64>(centered, reconstructedCentered);

          // Eckart-Young-Mirsky theorem: Frobenius norm squared of rank-2 residual
          // equals (N - 1) * (lambda_3 + lambda_4)
          final frobErrSq = square(
            norm<Float64>(residual, ord: NormKind.frobenius),
          ).scalar;
          final tailVarSum =
              (nSamples - 1) * (eigValsDesc[[2]] + eigValsDesc[[3]]);
          expect(frobErrSq, closeTo(tailVarSum, 1e-9));
        });
      },
    );
    test(
      'Multi-Tone Spectral Estimation, Windowed-Sinc FIR Low-Pass Design & SNR Verification',
      () {
        NDArray.scope(() {
          const fs = 256.0; // Sampling rate (Hz)
          const n = 256; // 1.0 second duration
          final t = linspace<Float64>(
            0.0,
            1.0,
            n,
            endpoint: false,
            dtype: DType.float64,
          );

          // Desired low-frequency signal: 8 Hz tone (amplitude 2.0)
          final desired = multiply<Float64>(
            sin<Float64>(
              multiply<Float64>(
                t,
                NDArray.scalar(2.0 * math.pi * 8.0, dtype: DType.float64),
              ),
            ),
            NDArray.scalar(2.0, dtype: DType.float64),
          );
          // High-frequency interference: 72 Hz tone (amplitude 1.5)
          final interference = multiply<Float64>(
            cos<Float64>(
              multiply<Float64>(
                t,
                NDArray.scalar(2.0 * math.pi * 72.0, dtype: DType.float64),
              ),
            ),
            NDArray.scalar(1.5, dtype: DType.float64),
          );
          final composite = add<Float64>(desired, interference);

          // 1. Verify raw spectrum via RFFT and rfftfreq
          final freqs = rfftfreq(n, d: 1.0 / fs);
          final rawMag = abs<Float64>(rfft<Complex128>(composite));
          // Peak at 8 Hz (bin 8) and 72 Hz (bin 72)
          expect(argmax(rawMag).scalar, equals(8));
          expect(freqs[[8]], closeTo(8.0, 1e-12));
          expect(rawMag[[72]], closeTo(1.5 * n / 2.0, 1e-9));

          // 2. Design a 33-tap Hamming-windowed sinc low-pass FIR filter with cutoff fc = 24 Hz
          const taps = 33;
          const half = (taps - 1) ~/ 2; // 16
          const fcNorm = 24.0 / (fs / 2.0); // Normalized to Nyquist = 0.1875
          final nIdx = linspace<Float64>(
            -half.toDouble(),
            half.toDouble(),
            taps,
            dtype: DType.float64,
          );
          // Ideal low-pass impulse response: fcNorm * sinc(fcNorm * n)
          final idealLp = multiply<Float64>(
            sinc<Float64>(
              multiply<Float64>(
                nIdx,
                NDArray.scalar(fcNorm, dtype: DType.float64),
              ),
            ),
            NDArray.scalar(fcNorm, dtype: DType.float64),
          );
          final win = hamming<Float64>(taps, dtype: DType.float64);
          final rawKernel = multiply<Float64>(idealLp, win);
          // Normalize DC gain to 1.0
          final NDArray<Float64> firKernel = divide(
            rawKernel,
            NDArray.scalar(sum(rawKernel).scalar, dtype: DType.float64),
          );

          // 3. Apply FIR filter in time domain via convolve(mode: ConvMode.same)
          final filtered = convolve(composite, firKernel, mode: ConvMode.same);
          expect(filtered.shape, equals([n]));

          // 4. Compare interior steady-state region [half .. n - half] against desired 8 Hz signal
          final interiorSlice = [const Slice(start: half, stop: n - half)];
          final desiredInterior = desired.slice(interiorSlice);
          final compositeInterior = composite.slice(interiorSlice);
          final filteredInterior = filtered.slice(interiorSlice);

          final signalPower = mean(square(desiredInterior)).scalar;
          final preFilterErrorPower = mean(
            square(subtract<Float64>(compositeInterior, desiredInterior)),
          ).scalar;
          final postFilterErrorPower = mean(
            square(subtract<Float64>(filteredInterior, desiredInterior)),
          ).scalar;

          final snrBeforeDb =
              10.0 * math.log(signalPower / preFilterErrorPower) / math.ln10;
          final snrAfterDb =
              10.0 * math.log(signalPower / postFilterErrorPower) / math.ln10;

          // Before filtering, 72 Hz interference has power 1.5^2/2 = 1.125 vs desired 2.0^2/2 = 2.0 (~2.5 dB)
          // After FIR low-pass filtering, 72 Hz is attenuated by > 35 dB!
          expect(snrBeforeDb, closeTo(2.498, 0.1));
          expect(snrAfterDb, greaterThan(35.0));
        });
      },
    );
    test('Finite-Difference Poisson Solver & Heat Diffusion Conservation Laws', () {
      NDArray.scope(() {
        // Part A: 1D Poisson BVP -u''(x) = pi^2 * sin(pi * x) on (0, 1) with u(0)=u(1)=0
        // Exact analytical solution: u(x) = sin(pi * x)
        const nInterior = 31;
        const h = 1.0 / (nInterior + 1);
        final xInt = linspace<Float64>(
          h,
          1.0 - h,
          nInterior,
          dtype: DType.float64,
        );

        // Build tridiagonal second-difference matrix A = (1/h^2) * tridiag(-1, 2, -1)
        final mainDiag = diag(
          NDArray.full([nInterior], 2.0 / (h * h), dtype: DType.float64),
        );
        final offDiag = diag(
          NDArray.full([nInterior - 1], -1.0 / (h * h), dtype: DType.float64),
          k: 1,
        );
        final laplacian = add<Float64>(
          add<Float64>(mainDiag, offDiag),
          offDiag.transposed,
        );

        final rhs = multiply<Float64>(
          sin<Float64>(
            multiply<Float64>(
              xInt,
              NDArray.scalar(math.pi, dtype: DType.float64),
            ),
          ),
          NDArray.scalar(math.pi * math.pi, dtype: DType.float64),
        );
        final uNum = solve(laplacian, rhs);
        final uExact = sin<Float64>(
          multiply<Float64>(
            xInt,
            NDArray.scalar(math.pi, dtype: DType.float64),
          ),
        );

        // Second-order O(h^2) convergence check: max error < 1e-3 for h = 1/32
        final maxBvpErr = max(
          abs<Float64>(subtract<Float64>(uNum, uExact)),
        ).scalar;
        expect(maxBvpErr, lessThan(1e-3));

        // Part B: 1D Periodic Heat Equation u_t = alpha * u_xx solved via Spectral FFT
        // Initial condition u(x, 0) = 1.0 + cos(x) on [0, 2*pi) -> u(x, t) = 1.0 + e^{-alpha * t} cos(x)
        const nGrid = 32;
        const dx = 2.0 * math.pi / nGrid;
        const alpha = 0.5;
        const tFinal = 0.8;
        final xGrid = linspace<Float64>(
          0.0,
          2.0 * math.pi,
          nGrid,
          endpoint: false,
          dtype: DType.float64,
        );
        final u0 = add<Float64>(
          NDArray.scalar(1.0, dtype: DType.float64),
          cos<Float64>(xGrid),
        );

        final kFreq = fftfreq(nGrid, d: dx / (2.0 * math.pi));
        final decay = exp<Float64>(
          multiply<Float64>(
            square(kFreq),
            NDArray.scalar(-alpha * tFinal, dtype: DType.float64),
          ),
        );
        final uFinal = real(
          ifft<Complex128>(
            multiply<Complex128>(
              fft<Complex128>(u0),
              decay.astype(DType.complex128),
            ),
          ),
        );
        final uFinalExact = add<Float64>(
          NDArray.scalar(1.0, dtype: DType.float64),
          multiply<Float64>(
            cos<Float64>(xGrid),
            NDArray.scalar(math.exp(-alpha * tFinal), dtype: DType.float64),
          ),
        );
        expect(allClose(uFinal, uFinalExact, atol: 1e-12), isTrue);

        // Mass conservation: integral of u(x, t) matches integral of u(x, 0)
        expect(sum(uFinal).scalar * dx, closeTo(sum(u0).scalar * dx, 1e-12));
      });
    });
    test(
      'Tikhonov Ridge Regression, Vandermonde Least-Squares & Numerical Integration',
      () {
        NDArray.scope(() {
          // Sample y(x) = 1.0 + 2.0*x - 0.5*x^2 + 0.25*x^3 on [-2, 2]
          final x = linspace<Float64>(-2.0, 2.0, 41, dtype: DType.float64);
          final trueCoeffsDesc = NDArray.fromList(
            [0.25, -0.5, 2.0, 1.0],
            [4],
            DType.float64,
          );
          final y = polyval<Float64, Float64, Float64>(trueCoeffsDesc, x);

          // 1. Design matrix with decreasing powers [x^3, x^2, x, 1] to match polyval
          final x2 = square(x);
          final x3 = multiply<Float64>(x2, x);
          final onesCol = NDArray.ones([41], DType.float64);
          final X = stack([x3, x2, x, onesCol], axis: 1);

          // 2. Ridge-regularized normal equations: (X^T X + lambda I) w = X^T y with small lambda
          const lambda = 1e-8;
          final xtx = matmul(X.transposed, X);
          final regMatrix = add<Float64>(
            xtx,
            multiply<Float64>(
              NDArray<Float64>.eye(4, DType.float64),
              NDArray.scalar(lambda, dtype: DType.float64),
            ),
          );
          final xty = matmul(X.transposed, y);
          final wRidge = solve(regMatrix, xty);
          expect(allClose(wRidge, trueCoeffsDesc, atol: 1e-6), isTrue);

          // 3. Compare with lstsq and polyfit
          final wLstsq = lstsq<Float64, Float64, Float64>(X, y).x;
          final wPolyfit = polyfit<Float64, Float64, Float64, Float64>(x, y, 3);
          expect(allClose(wLstsq, trueCoeffsDesc, atol: 1e-11), isTrue);
          expect(allClose(wPolyfit, trueCoeffsDesc, atol: 1e-11), isTrue);

          // 4. Analytical antiderivative P_int(x) = 0.0625*x^4 - (0.5/3)*x^3 + x^2 + x + 0
          final pIntCoeffs = NDArray.fromList(
            [0.25 / 4.0, -0.5 / 3.0, 1.0, 1.0, 0.0],
            [5],
            DType.float64,
          );
          final exactIntegral =
              polyval<Float64, Float64, Float64>(
                pIntCoeffs,
                NDArray.fromList([2.0], [1], DType.float64),
              )[[0]] -
              polyval<Float64, Float64, Float64>(
                pIntCoeffs,
                NDArray.fromList([-2.0], [1], DType.float64),
              )[[0]];
          final numIntegral = trapz<Float64>(
            y,
            spacing: const Spacing.step(4.0 / 40.0),
          ).scalar;
          // Integral of 1 + 2x - 0.5x^2 + 0.25x^3 on [-2, 2] is 4/3
          expect(exactIntegral, closeTo(4.0 / 3.0, 1e-11));
          expect(numIntegral, closeTo(exactIntegral, 1e-2));
        });
      },
    );
    test(
      'Cholesky-Correlated Asset Simulation, Portfolio VaR/CVaR & Markowitz Minimum-Variance Weights',
      () {
        NDArray.scope(() {
          final rng = RandomGenerator(90210);
          // 3 assets with expected annual returns mu and covariance matrix Sigma
          final mu = NDArray.fromList([0.08, 0.12, 0.15], [3], DType.float64);
          final sigma = NDArray.fromList(
            [0.04, 0.01, 0.005, 0.01, 0.09, 0.02, 0.005, 0.02, 0.16],
            [3, 3],
            DType.float64,
          );

          // 1. Analytical global minimum-variance portfolio weights: w = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)
          final onesVec = NDArray.ones([3], DType.float64);
          final invSigmaOnes = solve(sigma, onesVec);
          final denom = inner(onesVec, invSigmaOnes).scalar;
          final NDArray<Float64> minVarWeights = divide(
            invSigmaOnes,
            NDArray.scalar(denom, dtype: DType.float64),
          );
          expect(sum(minVarWeights).scalar, closeTo(1.0, 1e-12));
          // Asset 0 has lowest variance (0.04), so it receives the largest weight
          expect(argmax(minVarWeights).scalar, equals(0));

          // 2. Simulate 4000 correlated asset return scenarios via Cholesky factorization
          const nScenarios = 4000;
          final cholL = cholesky(sigma);
          final zIndep = rng.normal<Float64>([
            nScenarios,
            3,
          ], dtype: DType.float64);
          final correlatedReturns = add<Float64>(
            matmul(zIndep, cholL.transposed),
            broadcastTo(mu.reshape([1, 3]), [nScenarios, 3]),
          );

          // 3. Compute simulated portfolio returns r_p = R * w
          final portReturns = matmul(correlatedReturns, minVarWeights);
          final theoreticalMean = inner(mu, minVarWeights).scalar;
          final theoreticalVar = inner(
            minVarWeights,
            matmul(sigma, minVarWeights),
          ).scalar;

          expect(mean(portReturns).scalar, closeTo(theoreticalMean, 0.01));
          expect(
            variance<Float64>(portReturns, ddof: 1).scalar,
            closeTo(theoreticalVar, 0.005),
          );

          // 4. Compute 5% Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall)
          final q5 = percentile(portReturns, 5.0).scalar;
          final tailMask = lessEqual(
            portReturns,
            NDArray.scalar(q5, dtype: DType.float64),
          );
          final tailLosses = portReturns.applyMask(tailMask);
          final cvar5 = mean(tailLosses).scalar;
          // CVaR (expected return in worst 5% tail) must be strictly less than 5% VaR quantile
          expect(cvar5, lessThan(q5));

          // 5. Verify vectorized financial TVM helpers (npv / irr consistency)
          final cashFlows = NDArray.fromList(
            [-1000.0, 400.0, 400.0, 400.0],
            [4],
            DType.float64,
          );
          final internalRate = irr(cashFlows).scalar;
          final netPresentVal = npv(
            NDArray.scalar(internalRate, dtype: DType.float64),
            cashFlows,
          ).scalar;
          expect(netPresentVal, closeTo(0.0, 1e-8));
        });
      },
    );
    test(
      '2D Spatial Field SVD Low-Rank Compression, Gradient Magnitude & 2D FFT Shift Registration',
      () {
        NDArray.scope(() {
          const rows = 16;
          const cols = 16;
          final xAxis = linspace<Float64>(
            -1.0,
            1.0,
            cols,
            dtype: DType.float64,
          );
          final yAxis = linspace<Float64>(
            -1.0,
            1.0,
            rows,
            dtype: DType.float64,
          );
          final X = broadcastTo(xAxis.reshape([1, cols]), [rows, cols]);
          final Y = broadcastTo(yAxis.reshape([rows, 1]), [rows, cols]);

          // 1. Construct separable rank-2 spatial field: F(x, y) = exp(-x^2)*cos(pi*y) + 0.5*sin(pi*x)*exp(-2*y^2)
          final term1 = multiply<Float64>(
            exp<Float64>(negative(square(X))),
            cos<Float64>(
              multiply<Float64>(
                Y,
                NDArray.scalar(math.pi, dtype: DType.float64),
              ),
            ),
          );
          final term2 = multiply<Float64>(
            multiply<Float64>(
              sin<Float64>(
                multiply<Float64>(
                  X,
                  NDArray.scalar(math.pi, dtype: DType.float64),
                ),
              ),
              exp<Float64>(
                multiply<Float64>(
                  square(Y),
                  NDArray.scalar(-2.0, dtype: DType.float64),
                ),
              ),
            ),
            NDArray.scalar(0.5, dtype: DType.float64),
          );
          final field = add<Float64>(term1, term2);

          // 2. SVD rank-2 compression recovers a rank-2 separable sum to machine precision!
          final svdRes = svd<Float64, Float64>(field);
          expect(svdRes.s[[0]], greaterThan(1.0));
          expect(svdRes.s[[1]], greaterThan(0.5));
          expect(svdRes.s[[2]], lessThan(1e-12));

          final u2 = svdRes.u.slice([
            const Slice.all(),
            const Slice(start: 0, stop: 2),
          ]);
          final s2 = diag(svdRes.s.slice([const Slice(start: 0, stop: 2)]));
          final vh2 = svdRes.vh.slice([
            const Slice(start: 0, stop: 2),
            const Slice.all(),
          ]);
          final rank2Recon = matmul(matmul(u2, s2), vh2);
          expect(allClose(rank2Recon, field, atol: 1e-11), isTrue);

          // 3. Compute 2D spatial gradient and gradient magnitude ||∇F|| = hypot(dF/dy, dF/dx)
          final grads = gradientArray<Float64>(field);
          final gradMag = hypot(grads[0], grads[1]);
          expect(gradMag.shape, equals([rows, cols]));
          expect(min(gradMag).scalar, greaterThanOrEqualTo(0.0));

          // 4. 2D FFT Phase Correlation to detect integer cyclic shift (shiftRow = 3, shiftCol = 5)
          const shiftRow = 3;
          const shiftCol = 5;
          // Create unique asymmetric pulse image to register
          final pulse = exp<Float64>(
            negative(
              add<Float64>(
                multiply<Float64>(
                  square(
                    subtract<Float64>(
                      X,
                      NDArray.scalar(0.2, dtype: DType.float64),
                    ),
                  ),
                  NDArray.scalar(8.0, dtype: DType.float64),
                ),
                multiply<Float64>(
                  square(
                    add<Float64>(Y, NDArray.scalar(0.1, dtype: DType.float64)),
                  ),
                  NDArray.scalar(12.0, dtype: DType.float64),
                ),
              ),
            ),
          );
          final shiftedPulse = roll(
            roll(pulse, shiftRow, axis: 0),
            shiftCol,
            axis: 1,
          );

          // Cross-power spectrum: R = F(shifted) * conj(F(ref))
          final fRef = fft2<Complex128>(pulse);
          final fShift = fft2<Complex128>(shiftedPulse);
          final crossCorr = real(
            ifft2<Complex128>(multiply<Complex128>(fShift, conj(fRef))),
          );
          final peakFlatIdx = argmax(crossCorr);
          final peakCoords = unravel_index(peakFlatIdx, [rows, cols]);
          expect(peakCoords[0].scalar, equals(shiftRow));
          expect(peakCoords[1].scalar, equals(shiftCol));
        });
      },
    );
  });
}
