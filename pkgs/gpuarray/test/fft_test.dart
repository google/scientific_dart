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

import 'package:gpuarray/gpuarray.dart';
import 'package:test/test.dart';

void main() {
  group('Fast Fourier Transform (F9)', () {
    test('fft and ifft round-trip with FftNorm modes and out: parameter', () {
      final signal = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0],
        [4],
        DType.float64,
      );
      final preallocatedSpectrum = GpuArray.zeros([4], DType.complex128);
      try {
        for (final normMode in FftNorm.values) {
          final spectrum = fft(
            signal,
            norm: normMode,
            out: preallocatedSpectrum,
          );
          expect(identical(spectrum, preallocatedSpectrum), isTrue);
          final reconstructed = ifft(spectrum, norm: normMode);
          try {
            final values = reconstructed.toList().cast<Complex>();
            for (var i = 0; i < 4; i++) {
              expect(values[i].real, closeTo(i + 1.0, 1e-10));
              expect(values[i].imag, closeTo(0.0, 1e-10));
            }
          } finally {
            reconstructed.dispose();
          }
        }
      } finally {
        preallocatedSpectrum.dispose();
        signal.dispose();
      }
    });

    test('rfft and irfft round-trip real signals with out: parameter', () {
      final signal = GpuArray.fromList(
        <double>[1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
        [6],
        DType.float64,
      );
      final outReal = GpuArray.zeros([6], DType.float64);
      try {
        final spectrum = rfft(signal);
        try {
          expect(spectrum.shape, equals(<int>[4]));
          final recovered = irfft(spectrum, n: 6, out: outReal);
          expect(identical(recovered, outReal), isTrue);
          final values = recovered.toList().cast<double>();
          final expected = <double>[1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
          for (var i = 0; i < expected.length; i++) {
            expect(values[i], closeTo(expected[i], 1e-10));
          }
        } finally {
          spectrum.dispose();
        }
      } finally {
        outReal.dispose();
        signal.dispose();
      }
    });

    test('fft2 and ifft2 round-trip 2-D arrays', () {
      final image = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        DType.float64,
      );
      try {
        final spectrum = fft2(image, norm: FftNorm.ortho);
        final recovered = ifft2(spectrum, norm: FftNorm.ortho);
        try {
          expect(recovered.shape, equals(<int>[2, 3]));
          final values = recovered.toList().cast<Complex>();
          for (var i = 0; i < 6; i++) {
            expect(values[i].real, closeTo(i + 1.0, 1e-10));
            expect(values[i].imag, closeTo(0.0, 1e-10));
          }
        } finally {
          spectrum.dispose();
          recovered.dispose();
        }
      } finally {
        image.dispose();
      }
    });

    test('fftfreq and rfftfreq generate expected frequency bins', () {
      final freqs = fftfreq(4, d: 0.5);
      final rfreqs = rfftfreq(4, d: 0.5);
      try {
        expect(freqs.toList(), equals(<double>[0.0, 0.5, -1.0, -0.5]));
        expect(rfreqs.toList(), equals(<double>[0.0, 0.5, 1.0]));
      } finally {
        freqs.dispose();
        rfreqs.dispose();
      }
    });

    test('fftshift and ifftshift invert each other', () {
      final array = GpuArray.fromList(
        <double>[0.0, 1.0, 2.0, -2.0, -1.0],
        [5],
        DType.float64,
      );
      final outShifted = GpuArray.zeros([5], DType.float64);
      try {
        final shifted = fftshift(array, out: outShifted);
        expect(identical(shifted, outShifted), isTrue);
        expect(shifted.toList(), equals(<double>[-2.0, -1.0, 0.0, 1.0, 2.0]));

        final unshifted = ifftshift(shifted);
        try {
          expect(
            unshifted.toList(),
            equals(<double>[0.0, 1.0, 2.0, -2.0, -1.0]),
          );
        } finally {
          unshifted.dispose();
        }
      } finally {
        outShifted.dispose();
        array.dispose();
      }
    });

    test('validates arguments and disposed state', () {
      expect(() => fftfreq(0), throwsArgumentError);
      expect(() => rfftfreq(-1), throwsArgumentError);
      expect(() => fftfreq(4, d: 0.0), throwsArgumentError);

      final array = GpuArray.fromList(<double>[1.0, 2.0], [2], DType.float64);
      array.dispose();
      expect(() => fft(array), throwsStateError);
    });
  });
}
