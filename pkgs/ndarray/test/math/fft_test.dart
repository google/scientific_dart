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
import 'dart:typed_data';

void main() {
  group('NDArray Fast Fourier Transform (FFT & IFFT) Tensor Tests', () {
    test(
      'Kronecker Delta Spike 1D FFT yields perfectly uniform flat spectrum coefficients',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 0.0, 0.0, 0.0]), [
          4,
        ], DType.float64);
        final freq = fft(a);

        expect(freq.shape, [4]);
        expect(freq.dtype, DType.complex128);

        for (var i = 0; i < 4; i++) {
          final Complex c = freq.getCell([i]);
          expect(c.real, closeTo(1.0, 1e-9));
          expect(c.imag, closeTo(0.0, 1e-9));
        }
      }),
    );

    test(
      'Flat Constant DC Signal 1D FFT yields single spectral spike at zero frequency index',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([2.0, 2.0, 2.0, 2.0]), [
          4,
        ], DType.float64);
        final freq = fft(a);

        final Complex dcComponent = freq.getCell([0]);
        expect(dcComponent.real, closeTo(8.0, 1e-9));
        expect(dcComponent.imag, closeTo(0.0, 1e-9));

        for (var i = 1; i < 4; i++) {
          final Complex c = freq.getCell([i]);
          expect(c.real, closeTo(0.0, 1e-9));
          expect(c.imag, closeTo(0.0, 1e-9));
        }
      }),
    );

    test(
      'Verify complete FFT and IFFT full round-trip signal restoration (Power of Two)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([1.5, -2.0, 3.5, 4.0, 0.5, -1.0, 2.0, -0.5]),
          [8],
          DType.float64,
        );

        final freq = fft(a);
        final restored = ifft(freq);

        expect(restored.shape, [8]);
        expect(restored.dtype, DType.complex128);

        for (var i = 0; i < 8; i++) {
          final Complex c = restored.getCell([i]);
          expect(c.real, closeTo(a.getCell([i]), 1e-9));
          expect(c.imag, closeTo(0.0, 1e-9));
        }
      }),
    );

    test(
      'Verify Mixed-Radix core support for non-power-of-two lengths (N=6 and N=5) without crashes!',
      () => NDArray.scope(() {
        final a6 = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [6],
          DType.float64,
        );
        final freq6 = fft(a6);
        expect(freq6.shape, [6]);

        final restored6 = ifft(freq6);
        for (var i = 0; i < 6; i++) {
          expect(restored6.getCell([i]).real, closeTo(a6.getCell([i]), 1e-9));
        }

        final a5 = NDArray.fromList(
          Float64List.fromList([1.0, 0.0, 1.0, 0.0, 1.0]),
          [5],
          DType.float64,
        );
        final freq5 = fft(a5);
        expect(freq5.shape, [5]);

        final restored5 = ifft(freq5);
        expect(restored5.shape, [5]);
        expect(restored5.getCell([4]).real, closeTo(1.0, 1e-9));
      }),
    );

    test(
      'Verify standard parameter [n] zero-coefficient padding and truncation behavior',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        final paddedFreq = fft(a, n: 4);
        expect(paddedFreq.shape, [4]);

        final restored = ifft(paddedFreq);
        expect(restored.shape, [4]);
        expect(restored.getCell([0]).real, closeTo(1.0, 1e-9));
        expect(restored.getCell([1]).real, closeTo(2.0, 1e-9));
        expect(restored.getCell([2]).real, closeTo(0.0, 1e-9));
        expect(restored.getCell([3]).real, closeTo(0.0, 1e-9));
      }),
    );

    test(
      'Verify Float32/Complex64 precision branches, and IFFT with real input fallback',
      () => NDArray.scope(() {
        final a32 = NDArray.fromList(
          Float32List.fromList([1.0, 0.0, 0.0, 0.0]),
          [4],
          DType.float32,
        );
        final freq32 = fft(a32);
        expect(freq32.dtype, DType.complex64);
        expect(freq32.shape, [4]);

        final restored32 = ifft(freq32);
        expect(restored32.dtype, DType.complex64);
        expect(restored32.shape, [4]);
        expect(restored32.getCell([0]).real, closeTo(1.0, 1e-5));

        final comp64 = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [4],
          DType.complex64,
        );
        final freq64 = fft(comp64);
        expect(freq64.dtype, DType.complex64);
        expect(freq64.shape, [4]);

        final restored64 = ifft(freq64);
        expect(restored64.dtype, DType.complex64);
        expect(restored64.shape, [4]);
        expect(restored64.getCell([0]).real, closeTo(1.0, 1e-5));
        expect(restored64.getCell([0]).imag, closeTo(2.0, 1e-5));

        final realSignal = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
          DType.float64,
        );
        final restoredFromReal = ifft(realSignal);
        expect(restoredFromReal.dtype, DType.complex128);
        expect(restoredFromReal.shape, [4]);
      }),
    );

    test(
      'Verify fft() and ifft() on non-contiguous transposed strided views',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final view = parent.transposed;
        expect(view.isContiguous, false);

        final freq = fft(view);
        expect(freq.shape, [2, 2]);

        final contiguous = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final freqContig = fft(contiguous);

        final flatFreq = freq.ravel();
        final flatFreqContig = freqContig.ravel();
        for (var i = 0; i < 4; i++) {
          expect(
            flatFreq.getCell([i]).real,
            closeTo(flatFreqContig.getCell([i]).real, 1e-5),
          );
          expect(
            flatFreq.getCell([i]).imag,
            closeTo(flatFreqContig.getCell([i]).imag, 1e-5),
          );
        }

        final restored = ifft(freq);
        expect(restored.shape, [2, 2]);
        final restoredContig = ifft(freqContig);

        final flatRestored = restored.ravel();
        final flatRestoredContig = restoredContig.ravel();
        for (var i = 0; i < 4; i++) {
          expect(
            flatRestored.getCell([i]).real,
            closeTo(flatRestoredContig.getCell([i]).real, 1e-5),
          );
          expect(
            flatRestored.getCell([i]).imag,
            closeTo(flatRestoredContig.getCell([i]).imag, 1e-5),
          );
        }
      }),
    );

    test(
      'Verify fft() and ifft() throws StateError on native plan allocation failure',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);

        final hugeN = 10000000000000;

        expect(() => fft(a, n: hugeN), throwsA(anything));
        expect(() => ifft(a, n: hugeN), throwsA(anything));
      }),
    );

    group('Multi-dimensional FFT with axis parameter', () {
      test(
        '2D FFT along axis 0 (columns)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final freq = fft(a, axis: 0).copy();
          expect(freq.shape, [2, 2]);
          expect(freq.dtype, DType.complex128);
          final flat = freq.ravel();
          expect(flat.getCell([0]).real, closeTo(4.0, 1e-9));
          expect(flat.getCell([1]).real, closeTo(6.0, 1e-9));
          expect(flat.getCell([2]).real, closeTo(-2.0, 1e-9));
          expect(flat.getCell([3]).real, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        '2D FFT along axis 1 (rows)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final freq = fft(a, axis: 1).copy();
          expect(freq.shape, [2, 2]);
          expect(freq.dtype, DType.complex128);
          final flat = freq.ravel();
          expect(flat.getCell([0]).real, closeTo(3.0, 1e-9));
          expect(flat.getCell([1]).real, closeTo(-1.0, 1e-9));
          expect(flat.getCell([2]).real, closeTo(7.0, 1e-9));
          expect(flat.getCell([3]).real, closeTo(-1.0, 1e-9));
        }),
      );

      test(
        '2D IFFT along axis 0',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(4.0, 0.0),
              Complex(6.0, 0.0),
              Complex(-2.0, 0.0),
              Complex(-2.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final restored = ifft(a, axis: 0).copy();
          expect(restored.shape, [2, 2]);
          final flat = restored.ravel();
          expect(flat.getCell([0]).real, closeTo(1.0, 1e-9));
          expect(flat.getCell([1]).real, closeTo(2.0, 1e-9));
          expect(flat.getCell([2]).real, closeTo(3.0, 1e-9));
          expect(flat.getCell([3]).real, closeTo(4.0, 1e-9));
        }),
      );
    });

    group('fftshift and ifftshift spectrum shifting tests', () {
      test(
        '1D Odd Length (N=5) fftshift & ifftshift round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );

          final shifted = fftshift(a);
          expect(shifted.toList(), [3.0, 4.0, 0.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);
        }),
      );

      test(
        '1D Even Length (N=6) fftshift & ifftshift round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [6],
            DType.float64,
          );

          final shifted = fftshift(a);
          expect(shifted.toList(), [3.0, 4.0, 5.0, 0.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        }),
      );

      test(
        '2D Matrix shifting along all axes',
        () => NDArray.scope(() {
          final grid = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final shifted = fftshift(grid);
          expect(shifted.toList(), [6.0, 4.0, 5.0, 3.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }),
      );

      test(
        '2D Matrix shifting along specific axis (axis 0)',
        () => NDArray.scope(() {
          final grid = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final shifted = fftshift(grid, axes: 0);
          expect(shifted.toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        }),
      );

      test(
        'fftshift / ifftshift extension methods on NDArray',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );

          final shifted = fftshift(a);
          expect(shifted.toList(), [3.0, 4.0, 0.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);
        }),
      );

      test(
        'Preconditions & error handling',
        () => NDArray.scope(() {
          final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);

          expect(() => fftshift(a, axes: 2), throwsRangeError);
          expect(() => fftshift(a, axes: -3), throwsRangeError);
          expect(() => fftshift(a, axes: [0, 0]), throwsArgumentError);
          expect(() => fftshift(a, axes: 'invalid'), throwsArgumentError);

          a.dispose();
          expect(() => fftshift(a), throwsStateError);
        }),
      );
    });

    group('Frequency Generators', () {
      test('fftfreq even', () {
        final freqs = fftfreq(4, d: 2.0);
        expect(freqs.shape, [4]);
        expect(freqs.dtype, DType.float64);
        expect(freqs.getCell([0]), closeTo(0.0, 1e-9));
        expect(freqs.getCell([1]), closeTo(0.125, 1e-9));
        expect(freqs.getCell([2]), closeTo(-0.25, 1e-9));
        expect(freqs.getCell([3]), closeTo(-0.125, 1e-9));
      });

      test('fftfreq odd', () {
        final freqs = fftfreq(5, d: 1.0);
        expect(freqs.shape, [5]);
        expect(freqs.getCell([0]), closeTo(0.0, 1e-9));
        expect(freqs.getCell([1]), closeTo(0.2, 1e-9));
        expect(freqs.getCell([2]), closeTo(0.4, 1e-9));
        expect(freqs.getCell([3]), closeTo(-0.4, 1e-9));
        expect(freqs.getCell([4]), closeTo(-0.2, 1e-9));
      });

      test('rfftfreq even', () {
        final freqs = rfftfreq(4, d: 1.0);
        expect(freqs.shape, [3]);
        expect(freqs.getCell([0]), closeTo(0.0, 1e-9));
        expect(freqs.getCell([1]), closeTo(0.25, 1e-9));
        expect(freqs.getCell([2]), closeTo(0.5, 1e-9));
      });

      test('rfftfreq odd', () {
        final freqs = rfftfreq(5, d: 2.0);
        expect(freqs.shape, [3]);
        expect(freqs.getCell([0]), closeTo(0.0, 1e-9));
        expect(freqs.getCell([1]), closeTo(0.1, 1e-9));
        expect(freqs.getCell([2]), closeTo(0.2, 1e-9));
      });
    });

    group('Real 1D FFTs (rfft & irfft)', () {
      test(
        'rfft even length (N=4) float64',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final freq = rfft(a);
          expect(freq.shape, [3]);
          expect(freq.dtype, DType.complex128);
          expect(freq.getCell([0]).real, closeTo(10.0, 1e-9));
          expect(freq.getCell([0]).imag, closeTo(0.0, 1e-9));
          expect(freq.getCell([1]).real, closeTo(-2.0, 1e-9));
          expect(freq.getCell([1]).imag, closeTo(2.0, 1e-9));
          expect(freq.getCell([2]).real, closeTo(-2.0, 1e-9));
          expect(freq.getCell([2]).imag, closeTo(0.0, 1e-9));
        }),
      );

      test(
        'rfft odd length (N=3) float64 fallback',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final freq = rfft(a);
          expect(freq.shape, [2]);
          expect(freq.dtype, DType.complex128);
          expect(freq.getCell([0]).real, closeTo(6.0, 1e-9));
          expect(freq.getCell([0]).imag, closeTo(0.0, 1e-9));
          expect(freq.getCell([1]).real, closeTo(-1.5, 1e-9));
          expect(freq.getCell([1]).imag, closeTo(0.86602540378, 1e-9));
        }),
      );

      test(
        'irfft even length (N=4) float64 roundtrip',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final freq = rfft(a);
          final restored = irfft(freq, n: 4);
          expect(restored.shape, [4]);
          expect(restored.dtype, DType.float64);
          for (var i = 0; i < 4; i++) {
            expect(restored.getCell([i]), closeTo(a.getCell([i]), 1e-9));
          }
        }),
      );

      test(
        'irfft odd length (N=3) float64 fallback roundtrip',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final freq = rfft(a);
          final restored = irfft(freq, n: 3);
          expect(restored.shape, [3]);
          expect(restored.dtype, DType.float64);
          for (var i = 0; i < 3; i++) {
            expect(restored.getCell([i]), closeTo(a.getCell([i]), 1e-9));
          }
        }),
      );

      test(
        'rfft & irfft float32 precision',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
          final freq = rfft(a);
          expect(freq.dtype, DType.complex64);
          final restored = irfft(freq, n: 4);
          expect(restored.dtype, DType.float32);
          for (var i = 0; i < 4; i++) {
            expect(restored.getCell([i]), closeTo(a.getCell([i]), 1e-5));
          }
        }),
      );

      test(
        'rfft with out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final out = NDArray<Complex128>.zeros([3], DType.complex128);
          rfft(a, out: out);
          expect(out.getCell([0]).real, closeTo(10.0, 1e-9));
        }),
      );

      test(
        'irfft with out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final freq = rfft(a);
          final out = NDArray<Float64>.zeros([4], DType.float64);
          irfft(freq, n: 4, out: out);
          for (var i = 0; i < 4; i++) {
            expect(out.getCell([i]), closeTo(a.getCell([i]), 1e-9));
          }
        }),
      );
    });

    group('Multi-dimensional FFTs (fftn, ifftn, fft2, ifft2)', () {
      test(
        'fftn 2D Kronecker Delta float64',
        () => NDArray.scope(() {
          final list = List<double>.filled(16, 0.0);
          list[0] = 1.0;
          final a2 = NDArray.fromList(list, [4, 4], DType.float64);

          final freq = fftn(a2);
          expect(freq.shape, [4, 4]);
          expect(freq.dtype, DType.complex128);

          final flatFreq = freq.ravel();
          for (var i = 0; i < 16; i++) {
            expect(flatFreq.getCell([i]).real, closeTo(1.0, 1e-9));
            expect(flatFreq.getCell([i]).imag, closeTo(0.0, 1e-9));
          }
        }),
      );

      test(
        'fftn 2D roundtrip',
        () => NDArray.scope(() {
          final list = List<double>.generate(16, (i) => i.toDouble());
          final a = NDArray.fromList(list, [4, 4], DType.float64);
          final freq = fftn(a);
          final restored = ifftn(freq);
          expect(restored.shape, [4, 4]);
          final flatRestored = restored.ravel();
          final flatA = a.ravel();
          for (var i = 0; i < 16; i++) {
            expect(
              flatRestored.getCell([i]).real,
              closeTo(flatA.getCell([i]), 1e-9),
            );
            expect(flatRestored.getCell([i]).imag, closeTo(0.0, 1e-9));
          }
        }),
      );

      test(
        'fftn 2D with padding and truncation',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final freq = fftn(a, s: [3, 1], axes: [0, 1]);
          expect(freq.shape, [3, 1]);

          final flat = freq.ravel();
          expect(flat.getCell([0]).real, closeTo(4.0, 1e-9));
          expect(flat.getCell([1]).real, closeTo(-0.5, 1e-9));
          expect(flat.getCell([1]).imag, closeTo(-2.59807621135, 1e-9));
          expect(flat.getCell([2]).real, closeTo(-0.5, 1e-9));
          expect(flat.getCell([2]).imag, closeTo(2.59807621135, 1e-9));
        }),
      );

      test(
        'fft2 and ifft2 wrappers',
        () => NDArray.scope(() {
          final list = List<double>.generate(16, (i) => i.toDouble());
          final a = NDArray.fromList(list, [4, 4], DType.float64);
          final freq = fft2(a);
          final restored = ifft2(freq);
          expect(restored.shape, [4, 4]);
          final flatRestored = restored.ravel();
          final flatA = a.ravel();
          for (var i = 0; i < 16; i++) {
            expect(
              flatRestored.getCell([i]).real,
              closeTo(flatA.getCell([i]), 1e-9),
            );
          }
        }),
      );
    });
    test(
      'fft() and ifft() empty/0D shape validation checks',
      () => NDArray.scope(() {
        final emptyArr = NDArray.zeros([], DType.float64);

        expect(() => fft(emptyArr), throwsArgumentError);
        expect(() => ifft(emptyArr), throwsArgumentError);
      }),
    );

    test(
      'fft() and ifft() invalid transform length n <= 0 checks',
      () => NDArray.scope(() {
        final signal = NDArray.zeros([5], DType.float64);

        expect(() => fft(signal, n: 0), throwsArgumentError);
        expect(() => fft(signal, n: -5), throwsArgumentError);
        expect(() => ifft(signal, n: 0), throwsArgumentError);
        expect(() => ifft(signal, n: -1), throwsArgumentError);
      }),
    );

    test(
      'fft() and ifft() processing complex inputs directly',
      () => NDArray.scope(() {
        final complexSignal = NDArray.fromList(
          [Complex(1.0, 1.0), Complex(2.0, 2.0)],
          [2],
          DType.complex128,
        );

        final freq = fft(complexSignal);
        expect(freq.dtype, DType.complex128);
        expect(freq.shape, [2]);

        final time = ifft(freq);
        expect(time.dtype, DType.complex128);
        expect(time.shape, [2]);

        // The inverse transform must recover the original complex signal values (within floating precision)
        expect(time.getCell([0]).real, closeTo(1.0, 1e-9));
        expect(time.getCell([0]).imag, closeTo(1.0, 1e-9));
        expect(time.getCell([1]).real, closeTo(2.0, 1e-9));
        expect(time.getCell([1]).imag, closeTo(2.0, 1e-9));
      }),
    );

    test(
      'fft() and ifft() with real inputs and zero padding padding checks',
      () {
        final realSignal = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        // 1. fft with zero-padding (n = 4)
        final freqPadded = fft(realSignal, n: 4);
        expect(freqPadded.shape, [4]);

        // 2. ifft with real input and zero padding (n = 4)
        final timePadded = ifft(realSignal, n: 4);
        expect(timePadded.shape, [4]);
      },
    );

    test(
      'ifft() with non-contiguous strided view input',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );

        // Transposed view is non-contiguous
        final transposed = parent.transposed;
        expect(transposed.isContiguous, false);

        // Should automatically copy transposed inside ifft
        final result = ifft(transposed);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);
        final resultList = result.toList();
        expect(resultList[0].real, closeTo(2.0, 1e-9));
        expect(resultList[0].imag, closeTo(2.0, 1e-9));
        expect(resultList[1].real, closeTo(-1.0, 1e-9));
        expect(resultList[1].imag, closeTo(-1.0, 1e-9));
        expect(resultList[2].real, closeTo(3.0, 1e-9));
        expect(resultList[2].imag, closeTo(3.0, 1e-9));
        expect(resultList[3].real, closeTo(-1.0, 1e-9));
        expect(resultList[3].imag, closeTo(-1.0, 1e-9));
      }),
    );

    test(
      'FFT and IFFT zero-padding with ComplexList inputs and outputs',
      () => NDArray.scope(() {
        final a = NDArray.create([4], DType.complex128);
        a.setCell([0], Complex(1.0, 0.0));
        a.setCell([1], Complex(2.0, 0.0));
        a.setCell([2], Complex(3.0, 0.0));
        a.setCell([3], Complex(4.0, 0.0));

        // Trigger zero-padding block for ComplexList input
        final res = fft(a, n: 8);
        expect(res.shape, [8]);
        expect(res.dtype, DType.complex128);

        // Trigger IFFT zero-padding block for ComplexList input
        final invRes = ifft(res, n: 12);
        expect(invRes.shape, [12]);
        expect(invRes.dtype, DType.complex128);
      }),
    );

    test(
      'Zero-copy FFT and IFFT contiguous Float64 complex128 correctness',
      () {
        // 1. 1D Complex Vector Zero-Copy FFT and IFFT
        final a = NDArray.fromList(
          [
            Complex(1.0, 0.0),
            Complex(2.0, 0.0),
            Complex(3.0, 0.0),
            Complex(4.0, 0.0),
          ],
          [4],
          DType.complex128,
        );

        final resFFT = fft(a);
        expect(resFFT.dtype, DType.complex128);
        expect(resFFT.shape, [4]);

        // Mathematical FFT outputs for [1, 2, 3, 4]:
        // F(0) = 10 + 0i
        // F(1) = -2 + 2i
        // F(2) = -2 + 0i
        // F(3) = -2 - 2i
        expect(resFFT.getCell([0]), Complex(10.0, 0.0));
        expect(resFFT.getCell([1]).real, closeTo(-2.0, 1e-10));
        expect(resFFT.getCell([1]).imag, closeTo(2.0, 1e-10));
        expect(resFFT.getCell([2]).real, closeTo(-2.0, 1e-10));
        expect(resFFT.getCell([2]).imag, closeTo(0.0, 1e-10));
        expect(resFFT.getCell([3]).real, closeTo(-2.0, 1e-10));
        expect(resFFT.getCell([3]).imag, closeTo(-2.0, 1e-10));

        // Round-trip back with zero-copy IFFT
        final resIFFT = ifft(resFFT);
        expect(resIFFT.dtype, DType.complex128);
        expect(resIFFT.shape, [4]);
        expect(resIFFT.getCell([0]).real, closeTo(1.0, 1e-10));
        expect(resIFFT.getCell([0]).imag, closeTo(0.0, 1e-10));
        expect(resIFFT.getCell([1]).real, closeTo(2.0, 1e-10));
        expect(resIFFT.getCell([1]).imag, closeTo(0.0, 1e-10));
        expect(resIFFT.getCell([2]).real, closeTo(3.0, 1e-10));
        expect(resIFFT.getCell([2]).imag, closeTo(0.0, 1e-10));
        expect(resIFFT.getCell([3]).real, closeTo(4.0, 1e-10));
        expect(resIFFT.getCell([3]).imag, closeTo(0.0, 1e-10));

        // 2. High-Dimensional Stacked 2D Complex Matrix Zero-Copy FFT and IFFT
        final mat = NDArray.fromList(
          [
            Complex(1.0, 0.0),
            Complex(2.0, 0.0),
            Complex(3.0, 0.0),
            Complex(4.0, 0.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final matFFT = fft(mat);
        expect(matFFT.shape, [2, 2]);
        expect(matFFT.dtype, DType.complex128);

        // Row 0: [1, 2] -> [3, -1]
        // Row 1: [3, 4] -> [7, -1]
        expect(matFFT.getCell([0, 0]).real, closeTo(3.0, 1e-10));
        expect(matFFT.getCell([0, 1]).real, closeTo(-1.0, 1e-10));
        expect(matFFT.getCell([1, 0]).real, closeTo(7.0, 1e-10));
        expect(matFFT.getCell([1, 1]).real, closeTo(-1.0, 1e-10));

        final matIFFT = ifft(matFFT);
        expect(matIFFT.shape, [2, 2]);
        expect(matIFFT.getCell([0, 0]).real, closeTo(1.0, 1e-10));
        expect(matIFFT.getCell([0, 1]).real, closeTo(2.0, 1e-10));
        expect(matIFFT.getCell([1, 0]).real, closeTo(3.0, 1e-10));
        expect(matIFFT.getCell([1, 1]).real, closeTo(4.0, 1e-10));
      },
    );

    test('Multi-dimensional axis support inside fft() and ifft()', () {
      final mat = NDArray.fromList(
        [
          Complex(1.0, 0.0),
          Complex(1.0, 0.0),
          Complex(2.0, 0.0),
          Complex(2.0, 0.0),
        ],
        [2, 2],
        DType.complex128,
      );

      final fLast = fft(mat, axis: 1);
      expect(fLast.shape, [2, 2]);
      expect(fLast.getCell([0, 0]), Complex(2.0, 0.0));
      expect(fLast.getCell([0, 1]), Complex(0.0, 0.0));
      expect(fLast.getCell([1, 0]), Complex(4.0, 0.0));
      expect(fLast.getCell([1, 1]), Complex(0.0, 0.0));

      final fSwapped = fft(mat, axis: 0).copy();
      expect(fSwapped.shape, [2, 2]);
      expect(fSwapped.getCell([0, 0]), Complex(3.0, 0.0));
      expect(fSwapped.getCell([0, 1]), Complex(3.0, 0.0));
      expect(fSwapped.getCell([1, 0]), Complex(-1.0, 0.0));
      expect(fSwapped.getCell([1, 1]), Complex(-1.0, 0.0));

      final roundtrip = ifft(fSwapped, axis: 0).copy();
      expect(roundtrip.shape, [2, 2]);
      expect(roundtrip.getCell([0, 0]).real, closeTo(1.0, 1e-9));
      expect(roundtrip.getCell([1, 0]).real, closeTo(2.0, 1e-9));

      expect(() => fft(mat, axis: 2), throwsRangeError);
      expect(() => fft(mat, axis: -3), throwsRangeError);
    });

    group('Integer Input FFT Tests', () {
      test(
        'int32 input FFT promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 0, 0, 0], [4], DType.int32);
          final freq = fft(a);
          expect(freq.dtype, DType.complex128);
          expect(freq.shape, [4]);
          for (var i = 0; i < 4; i++) {
            final c = freq.getCell([i]);
            expect(c.real, closeTo(1.0, 1e-9));
            expect(c.imag, closeTo(0.0, 1e-9));
          }
        }),
      );

      test(
        'int64 input FFT promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 0, 0, 0], [4], DType.int64);
          final freq = fft(a);
          expect(freq.dtype, DType.complex128);
          expect(freq.shape, [4]);
          for (var i = 0; i < 4; i++) {
            final c = freq.getCell([i]);
            expect(c.real, closeTo(1.0, 1e-9));
            expect(c.imag, closeTo(0.0, 1e-9));
          }
        }),
      );

      test(
        'int16 input FFT promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 0, 0, 0], [4], DType.int16);
          final freq = fft(a);
          expect(freq.dtype, DType.complex128);
          expect(freq.shape, [4]);
          for (var i = 0; i < 4; i++) {
            final c = freq.getCell([i]);
            expect(c.real, closeTo(1.0, 1e-9));
            expect(c.imag, closeTo(0.0, 1e-9));
          }
        }),
      );

      test(
        'uint8 input FFT promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 0, 0, 0], [4], DType.uint8);
          final freq = fft(a);
          expect(freq.dtype, DType.complex128);
          expect(freq.shape, [4]);
          for (var i = 0; i < 4; i++) {
            final c = freq.getCell([i]);
            expect(c.real, closeTo(1.0, 1e-9));
            expect(c.imag, closeTo(0.0, 1e-9));
          }
        }),
      );

      test(
        'int32 input fftn promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.int32,
          );
          final freq = fftn(a);
          expect(freq.dtype, DType.complex128);
          expect(freq.shape, [2, 4]);
          final flat = freq.ravel();
          for (var i = 0; i < 8; i++) {
            expect(flat.getCell([i]).real, closeTo(1.0, 1e-9));
            expect(flat.getCell([i]).imag, closeTo(0.0, 1e-9));
          }
        }),
      );

      test(
        'int64 input fftn promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.int64,
          );
          final freq = fftn(a);
          expect(freq.dtype, DType.complex128);
          final flat = freq.ravel();
          for (var i = 0; i < 8; i++) {
            expect(flat.getCell([i]).real, closeTo(1.0, 1e-9));
          }
        }),
      );

      test(
        'int16 input fftn promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.int16,
          );
          final freq = fftn(a);
          expect(freq.dtype, DType.complex128);
          final flat = freq.ravel();
          for (var i = 0; i < 8; i++) {
            expect(flat.getCell([i]).real, closeTo(1.0, 1e-9));
          }
        }),
      );

      test(
        'uint8 input fftn promotes to complex128 and is correct',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.uint8,
          );
          final freq = fftn(a);
          expect(freq.dtype, DType.complex128);
          final flat = freq.ravel();
          for (var i = 0; i < 8; i++) {
            expect(flat.getCell([i]).real, closeTo(1.0, 1e-9));
          }
        }),
      );

      test(
        'int32 input fftn with complex64 out promotes to complex64 and calls _copyIntToFloatCpx',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.int32,
          );
          final out = NDArray<Complex64>.zeros([2, 4], DType.complex64);
          final freq = fftn(a, out: out);
          expect(freq.dtype, DType.complex64);
          final flat = freq.ravel();
          for (var i = 0; i < 8; i++) {
            expect(flat.getCell([i]).real, closeTo(1.0, 1e-5));
          }
        }),
      );

      test(
        'int64 input fftn with complex64 out promotes to complex64',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.int64,
          );
          final out = NDArray<Complex64>.zeros([2, 4], DType.complex64);
          final freq = fftn(a, out: out);
          expect(freq.dtype, DType.complex64);
        }),
      );

      test(
        'int16 input fftn with complex64 out promotes to complex64',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.int16,
          );
          final out = NDArray<Complex64>.zeros([2, 4], DType.complex64);
          final freq = fftn(a, out: out);
          expect(freq.dtype, DType.complex64);
        }),
      );

      test(
        'uint8 input fftn with complex64 out promotes to complex64',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 4],
            DType.uint8,
          );
          final out = NDArray<Complex64>.zeros([2, 4], DType.complex64);
          final freq = fftn(a, out: out);
          expect(freq.dtype, DType.complex64);
        }),
      );
    });

    group('FFT Preconditions and Error Handling', () {
      test(
        'disposed input throws StateError',
        () => NDArray.scope(() {
          final a = NDArray.zeros([8], DType.float64);
          a.dispose();

          expect(() => fft(a), throwsStateError);
          expect(() => ifft(a), throwsStateError);
          expect(() => rfft(a), throwsStateError);
          expect(() => irfft(a), throwsStateError);
          expect(() => fftn(a), throwsStateError);
          expect(() => ifftn(a), throwsStateError);

          final a2D = NDArray.zeros([2, 2], DType.float64);
          a2D.dispose();
          expect(() => fft2(a2D), throwsStateError);
          expect(() => ifft2(a2D), throwsStateError);

          expect(() => fftshift(a), throwsStateError);
          expect(() => ifftshift(a), throwsStateError);
        }),
      );

      test(
        'disposed out buffer throws StateError',
        () => NDArray.scope(() {
          final a = NDArray.zeros([8], DType.float64);
          final complexInput = NDArray.zeros([5], DType.complex128);
          final outComplex = NDArray<Complex128>.zeros([8], DType.complex128);
          outComplex.dispose();

          expect(() => fft(a, out: outComplex), throwsStateError);
          expect(() => ifft(complexInput, out: outComplex), throwsStateError);

          final outReal = NDArray.zeros([8], DType.float64);
          outReal.dispose();
          final rfftOut = NDArray<Complex128>.zeros([5], DType.complex128);
          rfftOut.dispose();
          expect(() => rfft(a, out: rfftOut), throwsStateError);

          final complexInputForIrfft = NDArray.zeros([5], DType.complex128);
          expect(
            () => irfft(complexInputForIrfft, n: 8, out: outReal),
            throwsStateError,
          );
        }),
      );

      test(
        'incompatible out buffer shape throws ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.zeros([8], DType.float64);

          final outWrongShape = NDArray<Complex128>.zeros([
            9,
          ], DType.complex128);
          expect(() => fft(a, out: outWrongShape), throwsArgumentError);
        }),
      );

      test(
        'incompatible out buffer dtype throws ArgumentError across FFT functions',
        () => NDArray.scope(() {
          final a64 = NDArray.zeros([8], DType.float64);
          final outC64 = NDArray<Complex64>.zeros([8], DType.complex64);
          expect(() => fft(a64, out: outC64), throwsArgumentError);

          final aC128 = NDArray<Complex128>.zeros([8], DType.complex128);
          expect(() => ifft(aC128, out: outC64), throwsArgumentError);

          final outRfftWrong = NDArray<Complex64>.zeros([5], DType.complex64);
          expect(() => rfft(a64, out: outRfftWrong), throwsArgumentError);

          final outIrfftWrong = NDArray.zeros([8], DType.float32);
          final rfftIn = NDArray<Complex128>.zeros([5], DType.complex128);
          expect(
            () => irfft(rfftIn, n: 8, out: outIrfftWrong),
            throwsArgumentError,
          );

          final outFftnWrong = NDArray<Complex64>.zeros([
            2,
            2,
          ], DType.complex64);
          final a2D = NDArray.zeros([2, 2], DType.float64);
          expect(() => fftn(a2D, out: outFftnWrong), throwsArgumentError);
        }),
      );
    });

    group('PocketFFT Plan Caching in NDArray', () {
      test(
        'Repeated transforms reuse plan cache and clearFFTPlanCache() succeeds',
        () {
          clearFFTPlanCache();

          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [4],
            DType.float64,
          );
          final f1 = fft(a);
          final f2 = fft(a);
          expect(f1.toList(), equals(f2.toList()));
          f1.dispose();
          f2.dispose();

          final r1 = rfft(a);
          final r2 = rfft(a);
          expect(r1.toList(), equals(r2.toList()));
          r1.dispose();
          r2.dispose();

          final a2D = NDArray.zeros([4, 4], DType.float64);
          final nd1 = fft2(a2D);
          final nd2 = fft2(a2D);
          expect(nd1.toList(), equals(nd2.toList()));
          nd1.dispose();
          nd2.dispose();
          a2D.dispose();
          a.dispose();

          clearFFTPlanCache();
        },
      );
    });

    group('PocketFFT Prime Lengths, Parity & Extended Real/Hermitian FFTs', () {
      test(
        'Prime-length 1D fft/ifft and rfft/irfft (n = 997 and n = 10007) accuracy and performance',
        () => NDArray.scope(() {
          for (final n in [997, 10007]) {
            final signal = NDArray.fromList(
              Float64List.fromList(
                List<double>.generate(
                  n,
                  (i) => math.sin(0.11 * i) - 0.5 * math.cos(0.07 * i),
                ),
              ),
              [n],
              DType.float64,
            );

            // Warm up plans
            fft(signal);
            rfft(signal);

            final sw = Stopwatch()..start();
            final spec = fft(signal);
            final rec = ifft(spec);
            final rspec = rfft(signal);
            final rrec = irfft(rspec, n: n);
            sw.stop();

            expect(
              sw.elapsedMilliseconds,
              lessThan(50),
              reason: 'Prime FFT n=$n took ${sw.elapsedMilliseconds}ms',
            );

            for (var i = 0; i < n; i++) {
              final orig = signal.getCell([i]);
              expect((rec.getCell([i]).real - orig).abs(), lessThan(1e-10));
              expect(rec.getCell([i]).imag.abs(), lessThan(1e-10));
              expect((rrec.getCell([i]) - orig).abs(), lessThan(1e-10));
            }
          }
        }),
      );

      test(
        'irfft for odd n truncates input.shape[axis] > n ~/ 2 + 1 to n ~/ 2 + 1 bins',
        () => NDArray.scope(() {
          final fullBins = NDArray.fromList(
            [
              Complex(15.0, 0.0),
              Complex(-2.5, 3.4409548011779334),
              Complex(-2.5, 0.8122992405822658),
              Complex(999.0, -888.0),
              Complex(-777.0, 666.0),
            ],
            [5],
            DType.complex128,
          );
          final exactBins = fullBins.slice([const Slice(start: 0, stop: 3)]);

          final resFromOverlong = irfft(fullBins, n: 5);
          final resFromExact = irfft(exactBins, n: 5);

          expect(resFromOverlong.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(
              resFromOverlong.getCell([i]),
              closeTo(resFromExact.getCell([i]), 1e-12),
            );
            expect(resFromOverlong.getCell([i]), closeTo(i + 1.0, 1e-10));
          }
        }),
      );

      test(
        'rfft, rfft2, rfftn, and ihfft reject complex inputs for both even and odd n',
        () => NDArray.scope(() {
          final c128 = NDArray.fromList(
            [
              Complex(1.0, 2.0),
              Complex(3.0, 4.0),
              Complex(5.0, 6.0),
              Complex(7.0, 8.0),
              Complex(9.0, 10.0),
            ],
            [5],
            DType.complex128,
          );
          final c64 = c128.astype(DType.complex64);

          expect(() => rfft(c128 as dynamic, n: 4), throwsArgumentError);
          expect(() => rfft(c128 as dynamic, n: 5), throwsArgumentError);
          expect(() => rfft(c64 as dynamic, n: 4), throwsArgumentError);
          expect(() => rfft(c64 as dynamic, n: 5), throwsArgumentError);

          expect(() => ihfft(c128 as dynamic, n: 4), throwsArgumentError);
          expect(() => ihfft(c128 as dynamic, n: 5), throwsArgumentError);
          expect(() => ihfft(c64 as dynamic, n: 4), throwsArgumentError);
          expect(() => ihfft(c64 as dynamic, n: 5), throwsArgumentError);

          final c128Mat = c128.slice([const Slice(start: 0, stop: 4)]).reshape([
            2,
            2,
          ]);
          expect(() => rfft2(c128Mat as dynamic), throwsArgumentError);
          expect(() => rfftn(c128Mat as dynamic), throwsArgumentError);
        }),
      );

      test(
        'rfft2, irfft2, rfftn, irfftn roundtrip for even and odd dimensions',
        () => NDArray.scope(() {
          // 2D odd shape [3, 5]
          final mat = NDArray.fromList(
            Float64List.fromList(
              List<double>.generate(15, (i) => (i + 1) * 0.75),
            ),
            [3, 5],
            DType.float64,
          );
          final f2 = rfft2(mat);
          expect(f2.shape, [3, 3]); // 5 ~/ 2 + 1 = 3
          expect(f2.dtype, DType.complex128);

          final rec2 = irfft2(f2, s: [3, 5]);
          expect(rec2.shape, [3, 5]);
          expect(rec2.dtype, DType.float64);
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 5; j++) {
              expect(rec2.getCell([i, j]), closeTo(mat.getCell([i, j]), 1e-10));
            }
          }

          // 3D shape [2, 3, 7]
          final t3 = NDArray.fromList(
            Float64List.fromList(
              List<double>.generate(42, (i) => math.cos(0.2 * i)),
            ),
            [2, 3, 7],
            DType.float64,
          );
          final fn = rfftn(t3);
          expect(fn.shape, [2, 3, 4]); // 7 ~/ 2 + 1 = 4
          final recn = irfftn(fn, s: [2, 3, 7]);
          expect(recn.shape, [2, 3, 7]);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 3; j++) {
              for (var k = 0; k < 7; k++) {
                expect(
                  recn.getCell([i, j, k]),
                  closeTo(t3.getCell([i, j, k]), 1e-10),
                );
              }
            }
          }
        }),
      );

      test(
        'hfft and ihfft roundtrip for even and odd n including out parameter',
        () => NDArray.scope(() {
          for (final n in [4, 5, 8, 9]) {
            final realSpec = NDArray.fromList(
              Float64List.fromList(
                List<double>.generate(
                  n,
                  (i) => (i + 1.0) * (i.isEven ? 1 : -0.5),
                ),
              ),
              [n],
              DType.float64,
            );
            final herm = ihfft(realSpec, n: n);
            expect(herm.shape, [n ~/ 2 + 1]);
            expect(herm.dtype, DType.complex128);

            final out = NDArray<Float64>.zeros([n], DType.float64);
            final rec = hfft(herm, n: n, out: out);
            expect(identical(rec, out), isTrue);
            for (var i = 0; i < n; i++) {
              expect(rec.getCell([i]), closeTo(realSpec.getCell([i]), 1e-10));
            }
          }
        }),
      );
    });
  });
}
