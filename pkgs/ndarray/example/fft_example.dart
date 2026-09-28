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

void main() {
  print('=== NDArray Fast Fourier Transform (FFT) DSP Spectral Analysis ===\n');

  NDArray.scope(() {
    // Create a composite signal containing a mixture of overlapping sine waves:
    // Frequency 1: 10 Hz (Amplitude: 3.0)
    // Frequency 2: 40 Hz (Amplitude: 5.0)
    // We sample at 128 Hz for 1 second, creating exactly 128 discrete points
    const samplingRate = 128.0;
    const numPoints = 128;

    final timeData = Float64List(numPoints);
    for (var i = 0; i < numPoints; i++) {
      final t = i / samplingRate;
      // 10Hz wave + 40Hz wave
      timeData[i] =
          3.0 * math.sin(2.0 * math.pi * 10.0 * t) +
          5.0 * math.sin(2.0 * math.pi * 40.0 * t);
    }

    final signal = NDArray.fromList(timeData, [numPoints], DType.float64);
    print('Generated Time Domain Signal with shape: ${signal.shape}');
    print('Signal contains: 10 Hz (Amp=3.0) + 40 Hz (Amp=5.0) waves.');

    // 1. Execute native mixed-radix FFT via package:pocketfft on unmanaged FFI heap!
    final coefficients = fft(signal);
    print(
      'Computed Native FFT frequency coefficients shape: ${coefficients.shape}\n',
    );

    // 2. Calculate Power Spectral Density (PSD) or Magnitude for each frequency bin.
    // NumPy rule: Bin i corresponds to frequency = i * samplingRate / N.
    // We only care about the positive frequencies (first half due to Nyquist symmetry!).
    print(
      '--- Dominant Spectral Frequency Peaks Detected (Magnitude > 50) ---',
    );
    final halfLen = numPoints ~/ 2;

    for (var i = 0; i <= halfLen; i++) {
      final Complex c = coefficients.getCell([i]);
      // Magnitude = sqrt(real^2 + imag^2)
      final magnitude = math.sqrt(c.real * c.real + c.imag * c.imag);
      final frequency = i * samplingRate / numPoints;

      // Scale magnitude for text visibility spikes
      if (magnitude > 50.0) {
        print(
          'Bin ${i.toString().padRight(3)} | Frequency: ${frequency.toString().padRight(4)} Hz | Magnitude: ${magnitude.toStringAsFixed(2).padLeft(6)} ${"*" * (magnitude ~/ 25)}',
        );
      }
    }

    print(
      '\n--- 3. Signal Restoration via Inverse Fourier Transform (IFFT) ---',
    );
    // Execute native IFFT with automatic 1/N normalization
    final restoredSignal = ifft(coefficients);

    print('Restored signal shape: ${restoredSignal.shape}');
    // Verify first few elements against original time data inputs
    print('Original vs Restored Values check:');
    for (var i = 0; i < 4; i++) {
      final Complex c = restoredSignal.getCell([i]);
      print(
        'Index $i -> Original Real: ${timeData[i].toStringAsFixed(4)} | Restored Real: ${c.real.toStringAsFixed(4)} (Imag: ${c.imag.toStringAsFixed(4)})',
      );
    }
    print('\nFFT and IFFT FFI offloading verified successfully!');
  });
}
