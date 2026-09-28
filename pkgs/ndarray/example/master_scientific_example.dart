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

void main() {
  print('=== NDArray Master Scientific & ML Signal Denoising Tutorial ===\n');

  NDArray.scope(() {
    // 1. Generate pure time-domain signal using vectorized operations
    // Sample a 10 Hz sine wave at 128 Hz for 1 second (128 points)
    const samplingRate = 128.0;
    const numPoints = 128;

    final time = NDArray<Float64>.arange(
      0.0,
      numPoints / samplingRate,
      step: 1.0 / samplingRate,
      dtype: DType.float64,
    );

    // 10 Hz sine wave with amplitude 5.0: y(t) = 5.0 * sin(2 * pi * 10.0 * t)
    final angle = time * (2.0 * math.pi * 10.0);
    final NDArray<Float64> pureSignal = multiply(
      sin(angle),
      NDArray.scalar(5.0, dtype: DType.float64),
    );
    print(
      '1. Generated pure 10 Hz sine wave signal (size: $numPoints points).',
    );

    // 2. Inject RNG Gaussian Noise to simulate real-world measurement sensors
    final noise = normal<Float64>(
      [numPoints],
      loc: 0.0,
      scale: 0.5,
      dtype: DType.float64,
      seed: 42,
    );
    final NDArray<Float64> noisySignal = add(pureSignal, noise);
    print(
      '2. Injected RNG Gaussian noise (loc = 0.0, scale = 0.5) element-wise.',
    );

    // 3. Execute FFI-Accelerated Real FFT (rfft) to map signal to frequency space
    final fftCoeffs = rfft(noisySignal);
    final freqs = rfftfreq(numPoints, d: 1.0 / samplingRate);
    print(
      '3. Executed mixed-radix FFI Real FFT (rfft) to transform signal to frequency space.',
    );

    // 4. Vectorized Low-pass Filter: Zero out high frequencies (noise) above 15 Hz!
    final highFreqMask = freqs > 15.0;
    fftCoeffs.setByMaskScalar(highFreqMask, Complex(0.0, 0.0));
    print(
      '4. Low-pass filter applied: zeroed out high-frequency noise bins (> 15 Hz) via boolean mask.',
    );

    // 5. Restoration: Inverse Real Fourier Transform (irfft) back to time domain!
    final reconstructed = irfft(fftCoeffs, n: numPoints);
    print(
      '5. Executed FFI Inverse Real FFT (irfft) to restore time-domain signal.',
    );

    // 6. Compare reconstructed signal against original pure signal!
    final isApproximatelyClose = allClose(
      pureSignal,
      reconstructed,
      rtol: 0.3,
      atol: 0.5,
    );
    print('\n=== Verification Results ===');
    print(
      'Is restored signal approximately close to pure signal? $isApproximatelyClose',
    );

    // Print first 5 samples comparison
    print('\nFirst 5 samples comparison:');
    print('Index | Noisy Signal | Restored Signal | Pure Signal');
    for (var i = 0; i < 5; i++) {
      final noisyVal = noisySignal.getCell([i]);
      final restoredVal = reconstructed.getCell([i]);
      final pureVal = pureSignal.getCell([i]);
      print(
        '  $i   |  ${noisyVal.toStringAsFixed(2).padLeft(11)} |  ${restoredVal.toStringAsFixed(2).padLeft(14)} |  ${pureVal.toStringAsFixed(2).padLeft(10)}',
      );
    }
  });
}
