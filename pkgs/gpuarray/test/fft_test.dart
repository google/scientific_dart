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
import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/fft.dart' as fft;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Fast Fourier Transforms (gpuarray.fft)', () {
    test(
      '1D Complex FFT and IFFT roundtrip on power-of-2 lengths (fft, ifft)',
      () {
        ResourceScope.scope(() {
          final signal = GpuArray.fromList(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [8],
            DType.float64,
          );

          final spectrum = fft.fft(signal);
          expect(spectrum.shape, equals([8]));
          expect(spectrum.dtype, equals(DType.complex128));

          // Invert back
          final recovered = fft.ifft(spectrum);
          expect(recovered.shape, equals([8]));

          final recFlat = recovered.toNDArray();
          final recData = recFlat.toList().cast<Complex>();
          final origData = signal.toList().cast<double>();

          for (var i = 0; i < 8; i++) {
            expect(recData[i].real, closeTo(origData[i], 1e-4));
            expect(recData[i].imag, closeTo(0.0, 1e-4));
          }
          recFlat.dispose();
        });
      },
    );

    test(
      'Bluestein Chirp-Z transform on non-power-of-2 lengths (N=3, 5, 7, 11)',
      () {
        ResourceScope.scope(() {
          final testLengths = [3, 5, 6, 7, 9, 11, 13, 25, 50];

          for (final n in testLengths) {
            final rawData = List<double>.generate(
              n,
              (i) =>
                  math.sin(2.0 * math.pi * i / n) +
                  math.cos(4.0 * math.pi * i / n),
            );
            final signal = GpuArray.fromList(rawData, [n], DType.float64);

            // Compute FFT via Bluestein
            final spectrum = fft.fft(signal);
            expect(spectrum.shape, equals([n]));

            // Compare against direct analytical DFT formula
            final specFlat = spectrum.toNDArray();
            final specData = specFlat.toList().cast<Complex>();

            for (var k = 0; k < n; k++) {
              var expectedR = 0.0;
              var expectedI = 0.0;
              for (var t = 0; t < n; t++) {
                final angle = -2.0 * math.pi * k * t / n;
                expectedR += rawData[t] * math.cos(angle);
                expectedI += rawData[t] * math.sin(angle);
              }
              expect(
                specData[k].real,
                closeTo(expectedR, 1e-4),
                reason: "Real mismatch at n=$n, k=$k",
              );
              expect(
                specData[k].imag,
                closeTo(expectedI, 1e-4),
                reason: "Imag mismatch at n=$n, k=$k",
              );
            }
            specFlat.dispose();

            // Invert back via Bluestein IFFT
            final recovered = fft.ifft(spectrum);
            expect(recovered.shape, equals([n]));
            final recFlat = recovered.toNDArray();
            final recData = recFlat.toList().cast<Complex>();

            for (var i = 0; i < n; i++) {
              expect(
                recData[i].real,
                closeTo(rawData[i], 1e-4),
                reason: "IFFT Real mismatch at n=$n, i=$i",
              );
              expect(
                recData[i].imag,
                closeTo(0.0, 1e-4),
                reason: "IFFT Imag mismatch at n=$n, i=$i",
              );
            }
            recFlat.dispose();
          }
        });
      },
    );

    test('Parseval theorem energy conservation with norm=ortho', () {
      ResourceScope.scope(() {
        final signal = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
          [7],
          DType.float64,
        );
        final spectrum = fft.fft(signal, norm: 'ortho');
        final specFlat = spectrum.toNDArray();
        final specData = specFlat.toList().cast<Complex>();
        final origData = signal.toList().cast<double>();

        var timeEnergy = 0.0;
        for (final v in origData) {
          timeEnergy += v * v;
        }

        var freqEnergy = 0.0;
        for (final c in specData) {
          freqEnergy += c.real * c.real + c.imag * c.imag;
        }

        expect(freqEnergy, closeTo(timeEnergy, 1e-4));
        specFlat.dispose();
      });
    });

    test(
      '1D Real FFT and IRFFT (rfft, irfft) on power-of-2 and non-power-of-2 sizes',
      () {
        ResourceScope.scope(() {
          for (final n in [6, 7, 8, 9]) {
            final rawData = List<double>.generate(n, (i) => (i + 1).toDouble());
            final signal = GpuArray.fromList(rawData, [n], DType.float64);

            final rspec = fft.rfft(signal);
            expect(rspec.shape, equals([n ~/ 2 + 1]));

            final recovered = fft.irfft(rspec, n: n);
            expect(recovered.shape, equals([n]));

            final recData = recovered.toList().cast<double>();
            for (var i = 0; i < n; i++) {
              expect(
                recData[i],
                closeTo(rawData[i], 1e-4),
                reason: "IRFFT mismatch at n=$n, i=$i",
              );
            }
          }
        });
      },
    );

    test('2D FFT and IFFT (fft2, ifft2) on non-power-of-2 shapes', () {
      ResourceScope.scope(() {
        final matrix = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );

        final spec2d = fft.fft2(matrix);
        expect(spec2d.shape, equals([2, 3]));

        final rec2d = fft.ifft2(spec2d);
        expect(rec2d.shape, equals([2, 3]));

        final recFlat = rec2d.toNDArray();
        final recData = recFlat.toList().cast<Complex>();
        final origData = matrix.toList().cast<double>();

        for (var i = 0; i < 6; i++) {
          expect(recData[i].real, closeTo(origData[i], 1e-4));
          expect(recData[i].imag, closeTo(0.0, 1e-4));
        }
        recFlat.dispose();
      });
    });

    test(
      'Frequency utilities (fftfreq, rfftfreq, fftshift, ifftshift with int and List<int> axes)',
      () {
        ResourceScope.scope(() {
          final freqs = fft.fftfreq(8, d: 0.1);
          expect(freqs.shape, equals([8]));
          final fList = freqs.toList().cast<double>();
          expect(fList[0], equals(0.0));
          expect(fList[1], closeTo(1.25, 1e-4));

          final rfreqs = fft.rfftfreq(8, d: 0.1);
          expect(rfreqs.shape, equals([5]));

          // 1D fftshift and ifftshift
          final x = GpuArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [8],
            DType.float64,
          );
          final shifted = fft.fftshift(x);
          final unshifted = fft.ifftshift(shifted);
          expect(unshifted.toList(), equals(x.toList()));

          // Odd length 1D fftshift
          final xOdd = GpuArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );
          final shiftedOdd = fft.fftshift(xOdd, axes: 0);
          final unshiftedOdd = fft.ifftshift(shiftedOdd, axes: 0);
          expect(unshiftedOdd.toList(), equals(xOdd.toList()));

          // 2D fftshift with List<int> axes
          final x2d = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 4],
            DType.float64,
          );

          final shifted2d = fft.fftshift(x2d, axes: [0, 1]);
          final unshifted2d = fft.ifftshift(shifted2d, axes: [0, 1]);
          expect(unshifted2d.toList(), equals(x2d.toList()));
        });
      },
    );

    test(
      'IRFFT with arbitrary n (zero-padding and truncation in frequency domain)',
      () {
        ResourceScope.scope(() {
          // Length 5 complex input spectrum (corresponding standard outLen = 2 * (5 - 1) = 8)
          final rawSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
          final sig = GpuArray.fromList(rawSignal, [8], DType.float64);
          final rspec = fft.rfft(sig);
          expect(rspec.shape, equals([5]));

          // Standard length n = 8
          final rec8 = fft.irfft(rspec);
          expect(rec8.shape, equals([8]));
          final rec8List = rec8.toList().cast<double>();
          for (var i = 0; i < 8; i++) {
            expect(rec8List[i], closeTo(rawSignal[i], 1e-4));
          }

          // Explicit n = 16 (zero-padding in frequency domain, n > 2*(inLen - 1))
          final rec16 = fft.irfft(rspec, n: 16);
          expect(rec16.shape, equals([16]));
          expect(rec16.dtype, equals(DType.float64));

          // Explicit n = 6 (truncation in frequency domain, n < 2*(inLen - 1))
          final rec6 = fft.irfft(rspec, n: 6);
          expect(rec6.shape, equals([6]));
          expect(rec6.dtype, equals(DType.float64));

          // Direct synthetic complex spectrum of length 5
          final synthSpec = GpuArray.fromList(
            [
              Complex(10.0, 0.0),
              Complex(2.0, -1.0),
              Complex(1.0, 0.5),
              Complex(-0.5, 0.2),
              Complex(0.1, 0.0),
            ],
            [5],
            DType.complex128,
          );

          final padded = fft.irfft(synthSpec, n: 16);
          expect(padded.shape, equals([16]));

          final truncated = fft.irfft(synthSpec, n: 6);
          expect(truncated.shape, equals([6]));
        });
      },
    );
  });
}
