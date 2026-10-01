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
import 'package:gpuarray/src/gpu_array.dart' show ResourceScope;
import 'package:test/test.dart';

void main() {
  group('Fast Fourier Transform (F17)', () {
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

    test('non-power-of-2 direct DFT and Bluestein Chirp-Z round-trip', () {
      ResourceScope.scope(() {
        final primeSignal = GpuArray.fromList(
          <double>[1.0, -2.0, 0.5, 4.0, -1.5, 2.25, 3.0],
          [7],
          DType.float64,
        );
        final primeSpectrum = fft(primeSignal);
        final primeRecovered = ifft(primeSpectrum);
        final primeValues = primeRecovered.toList().cast<Complex>();
        final expectedPrime = <double>[1.0, -2.0, 0.5, 4.0, -1.5, 2.25, 3.0];
        for (var i = 0; i < expectedPrime.length; i++) {
          expect(primeValues[i].real, closeTo(expectedPrime[i], 1e-9));
          expect(primeValues[i].imag, closeTo(0.0, 1e-9));
        }

        // Length 300 > 256 exercises the Bluestein Chirp-Z GPU pipeline.
        final bluesteinData = List<double>.generate(
          300,
          (i) => ((i % 11) - 5).toDouble(),
        );
        final bluesteinSignal = GpuArray.fromList(bluesteinData, [
          300,
        ], DType.float64);
        final bluesteinSpectrum = fft(bluesteinSignal);
        final bluesteinRecovered = ifft(bluesteinSpectrum);
        final bluesteinValues = bluesteinRecovered.toList().cast<Complex>();
        for (var i = 0; i < 300; i++) {
          expect(bluesteinValues[i].real, closeTo(bluesteinData[i], 1e-7));
          expect(bluesteinValues[i].imag, closeTo(0.0, 1e-7));
        }
      });
    });

    test('rfft2, irfft2, fftn, ifftn, rfftn, and irfftn round-trip', () {
      ResourceScope.scope(() {
        final grid2d = GpuArray.fromList(
          List<double>.generate(24, (i) => (i + 1).toDouble()),
          [4, 6],
          DType.float64,
        );
        final rspec2d = rfft2(grid2d);
        expect(rspec2d.shape, equals(<int>[4, 4]));
        final irec2d = irfft2(rspec2d, s: <int>[4, 6]);
        expect(irec2d.shape, equals(<int>[4, 6]));
        final rec2dVals = irec2d.toList().cast<double>();
        for (var i = 0; i < 24; i++) {
          expect(rec2dVals[i], closeTo(i + 1.0, 1e-9));
        }

        final grid3d = GpuArray.fromList(
          List<double>.generate(16, (i) => (i * 0.5) - 3.0),
          [2, 2, 4],
          DType.float64,
        );
        final spec3d = fftn(grid3d, norm: FftNorm.ortho);
        final rec3d = ifftn(spec3d, norm: FftNorm.ortho);
        final rec3dVals = rec3d.toList().cast<Complex>();
        for (var i = 0; i < 16; i++) {
          expect(rec3dVals[i].real, closeTo((i * 0.5) - 3.0, 1e-9));
          expect(rec3dVals[i].imag, closeTo(0.0, 1e-9));
        }

        final outRfftn = GpuArray.zeros([2, 2, 4], DType.float64);
        final rspec3d = rfftn(grid3d);
        expect(rspec3d.shape, equals(<int>[2, 2, 3]));
        final rrec3d = irfftn(rspec3d, s: <int>[2, 2, 4], out: outRfftn);
        expect(identical(rrec3d, outRfftn), isTrue);
        final rrec3dVals = rrec3d.toList().cast<double>();
        for (var i = 0; i < 16; i++) {
          expect(rrec3dVals[i], closeTo((i * 0.5) - 3.0, 1e-9));
        }
      });
    });

    test('hfft and ihfft round-trip real spectrum and Hermitian signal', () {
      ResourceScope.scope(() {
        final realSpectrum = GpuArray.fromList(
          <double>[1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
          [6],
          DType.float64,
        );
        final hermSignal = ihfft(realSpectrum);
        expect(hermSignal.shape, equals(<int>[4]));
        final outSpectrum = GpuArray.zeros([6], DType.float64);
        final recovered = hfft(hermSignal, n: 6, out: outSpectrum);
        expect(identical(recovered, outSpectrum), isTrue);
        final values = recovered.toList().cast<double>();
        final expected = <double>[1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
        for (var i = 0; i < expected.length; i++) {
          expect(values[i], closeTo(expected[i], 1e-9));
        }
      });
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

    test(
      'strided out: view preserves untouched elements and validates contracts',
      () {
        ResourceScope.scope(() {
          final fullFreqOut = GpuArray.filled([8], -99.0, DType.float64);
          final stridedFreqOut = fullFreqOut.slice([const Slice(0, 8, 2)]);
          final res = fftfreq(4, d: 0.5, out: stridedFreqOut);
          expect(identical(res, stridedFreqOut), isTrue);
          expect(
            fullFreqOut.toList(),
            equals(<double>[0.0, -99.0, 0.5, -99.0, -1.0, -99.0, -0.5, -99.0]),
          );

          final base = GpuArray.fromList(<double>[1.0], [1], DType.float64);
          final broadcastedOut = base.broadcastTo([4]);
          expect(() => fftfreq(4, out: broadcastedOut), throwsUnsupportedError);
        });
      },
    );

    test('validates arguments and disposed state', () {
      expect(() => fftfreq(0), throwsArgumentError);
      expect(() => rfftfreq(-1), throwsArgumentError);
      expect(() => fftfreq(4, d: 0.0), throwsArgumentError);

      final array = GpuArray.fromList(<double>[1.0, 2.0], [2], DType.float64);
      array.dispose();
      expect(() => fft(array), throwsStateError);

      final valid = GpuArray.fromList(<double>[1.0, 2.0], [2], DType.float64);
      final disposedOut = GpuArray.zeros([4], DType.complex128)..dispose();
      try {
        expect(() => fft(valid, n: 0, out: disposedOut), throwsStateError);
      } finally {
        valid.dispose();
      }
    });
  });
}
