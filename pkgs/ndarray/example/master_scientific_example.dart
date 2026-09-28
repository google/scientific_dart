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

  // 1. Generate pure time-domain signal
  // Sample a 10 Hz sine wave at 128 Hz for 1 second (128 points)
  const samplingRate = 128.0;
  const numPoints = 128;

  final time = NDArray.create([numPoints], DType.float64);
  final pureSignal = NDArray.create([numPoints], DType.float64);
  addTearDown(() => time.dispose());
  addTearDown(() => pureSignal.dispose());

  final timeData = time.toList();
  final pureData = pureSignal.toList();

  for (var i = 0; i < numPoints; i++) {
    final t = i / samplingRate;
    timeData[i] = Float64(t);
    pureData[i] = Float64(
      5.0 * math.sin(2.0 * math.pi * 10.0 * t),
    ); // 10Hz sine wave, amplitude 5
  }
  print('1. Generated pure 10 Hz sine wave signal (size: 128 points).');

  // 2. Inject RNG Gaussian Noise to simulate real-world measurement sensors
  // Noise distribution: loc = 0.0, scale = 0.5
  final noise = normal([numPoints], loc: 0.0, scale: 0.5, seed: 42);
  addTearDown(() => noise.dispose());

  // Combine pure signal + noise element-wise!
  final noisySignal = add(pureSignal, noise);
  addTearDown(() => noisySignal.dispose());
  print(
    '2. Injected RNG Gaussian noise (loc = 0.0, scale = 1.5) element-wise.',
  );

  // 3. Execute FFI-Accelerated mixed-radix FFT to map signal to frequency space
  final fftCoeffs = fft(noisySignal);
  addTearDown(() => fftCoeffs.dispose);
  print(
    '3. Executed mixed-radix FFI FFT to transform signal to frequency space.',
  );

  // 4. Low-pass Filter: Filter out high frequencies (noise) above 15 Hz!
  // We zero out any frequency bin corresponding to > 15 Hz
  final halfLen = numPoints ~/ 2;
  var filteredBinsCount = 0;

  for (var i = 0; i <= halfLen; i++) {
    final freq = i * samplingRate / numPoints;
    if (freq > 15.0) {
      // Symmetrically zero out positive and corresponding negative high frequency bins!
      fftCoeffs[i] = Complex(0.0, 0.0);
      if (i > 0 && i < halfLen) {
        fftCoeffs[numPoints - i] = Complex(0.0, 0.0);
      }
      filteredBinsCount++;
    }
  }
  print(
    '4. Low-pass filter applied: zeroed out $filteredBinsCount high-frequency noise bins (> 15 Hz).',
  );

  // 5. Restoration: Inverse Fourier Transform (IFFT) back to time domain!
  final restoredComplex = ifft(fftCoeffs);
  addTearDown(() => restoredComplex.dispose());

  // Extract real component of the reconstructed signal
  final reconstructed = NDArray.create([numPoints], DType.float64);
  addTearDown(() => reconstructed.dispose());
  final List<double> reconstructedData = reconstructed.toList();
  for (var i = 0; i < numPoints; i++) {
    reconstructedData[i] = restoredComplex[i].real;
  }
  print('5. Executed FFI IFFT to restore signal back to time-domain.');

  // 6. Compare reconstructed signal against original pure signal!
  // Check if they are approximately equal within rtol = 0.3 tolerance (since filtering keeps primary wave)
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

  // Print first few restored coordinates compared to raw noise
  print('\nFirst 5 samples comparison:');
  print('Index | Noisy Signal | Restored Signal | Pure Signal');
  for (var i = 0; i < 5; i++) {
    print(
      '  $i   |  ${noisySignal[i].toStringAsFixed(2).padLeft(11)} |  ${reconstructed[i].toStringAsFixed(2).padLeft(14)} |  ${pureSignal[i].toStringAsFixed(2).padLeft(10)}',
    );
  }
}

// Simple tear-down helper
final List<void Function()> _teardowns = [];
void addTearDown(void Function() f) => _teardowns.add(f);
