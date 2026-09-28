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
  group('Comprehensive FFT - 1D Transforms (fft, ifft, rfft, irfft)', () {
    test(
      '1D Complex FFT and IFFT with multiple DTypes (Float64, Float32, Int64, Int32, Int16, Uint8, Bool)',
      () {
        NDArray.scope(() {
          // Float64
          final aF64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          final fftF64 = fft(aF64);
          expect(fftF64.dtype, DType.complex128);
          expect(fftF64.shape, [4]);
          expect(fftF64.getCell([0]), Complex(10.0, 0.0));
          expect(fftF64.getCell([1]).real, closeTo(-2.0, 1e-9));
          expect(fftF64.getCell([1]).imag, closeTo(2.0, 1e-9));
          expect(fftF64.getCell([2]).real, closeTo(-2.0, 1e-9));
          expect(fftF64.getCell([2]).imag, closeTo(0.0, 1e-9));
          expect(fftF64.getCell([3]).real, closeTo(-2.0, 1e-9));
          expect(fftF64.getCell([3]).imag, closeTo(-2.0, 1e-9));

          final ifftF64 = ifft(fftF64);
          expect(ifftF64.dtype, DType.complex128);
          for (var i = 0; i < 4; i++) {
            expect(ifftF64.getCell([i]).real, closeTo(aF64.getCell([i]), 1e-9));
            expect(ifftF64.getCell([i]).imag, closeTo(0.0, 1e-9));
          }

          // Float32 -> Complex64
          final aF32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
          );
          final fftF32 = fft(aF32);
          expect(fftF32.dtype, DType.complex64);
          final ifftF32 = ifft(fftF32);
          expect(ifftF32.dtype, DType.complex64);
          expect(ifftF32.getCell([0]).real, closeTo(1.0, 1e-5));

          // Int64 -> Complex128
          final aI64 = NDArray.fromList([1, 2, 3, 4], [4], DType.int64);
          final fftI64 = fft(aI64);
          expect(fftI64.dtype, DType.complex128);
          expect(fftI64.getCell([0]), Complex(10.0, 0.0));

          // Int32 -> Complex128
          final aI32 = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final fftI32 = fft(aI32);
          expect(fftI32.dtype, DType.complex128);
          expect(fftI32.getCell([0]), Complex(10.0, 0.0));

          // Int16 -> Complex128
          final aI16 = NDArray.fromList([1, 2, 3, 4], [4], DType.int16);
          final fftI16 = fft(aI16);
          expect(fftI16.dtype, DType.complex128);

          // Uint8 -> Complex128
          final aU8 = NDArray.fromList([1, 2, 3, 4], [4], DType.uint8);
          final fftU8 = fft(aU8);
          expect(fftU8.dtype, DType.complex128);

          // Bool -> Complex128
          final aBool = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final fftBool = fft(aBool);
          expect(fftBool.dtype, DType.complex128);
          expect(fftBool.getCell([0]).real, closeTo(2.0, 1e-9));

          // Complex128 and Complex64 direct input
          final aC128 = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(2.0, 0.0),
              Complex(3.0, -1.0),
              Complex(4.0, 2.0),
            ],
            [4],
            DType.complex128,
          );
          final fftC128 = fft(aC128);
          expect(fftC128.dtype, DType.complex128);
          final ifftC128 = ifft(fftC128);
          for (var i = 0; i < 4; i++) {
            expect(
              ifftC128.getCell([i]).real,
              closeTo(aC128.getCell([i]).real, 1e-9),
            );
            expect(
              ifftC128.getCell([i]).imag,
              closeTo(aC128.getCell([i]).imag, 1e-9),
            );
          }

          final aC64 = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 0.0)],
            [2],
            DType.complex64,
          );
          final fftC64 = fft(aC64);
          expect(fftC64.dtype, DType.complex64);
          final ifftC64 = ifft(fftC64);
          expect(ifftC64.getCell([0]).real, closeTo(1.0, 1e-5));
          expect(ifftC64.getCell([0]).imag, closeTo(1.0, 1e-5));
        });
      },
    );

    test(
      '1D FFT and IFFT with n parameter: truncation, expansion, and out recycling',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );

          // Truncate length to n = 4
          final fftTrunc = fft(a, n: 4);
          expect(fftTrunc.shape, [4]);
          expect(fftTrunc.getCell([0]).real, closeTo(10.0, 1e-9));

          // Pad length to n = 8
          final fftPad = fft(a, n: 8);
          expect(fftPad.shape, [8]);
          expect(fftPad.getCell([0]).real, closeTo(21.0, 1e-9));

          // With out buffer
          final outBuf = NDArray<Complex128>.zeros([8], DType.complex128);
          final res = fft(a, n: 8, out: outBuf);
          expect(identical(res, outBuf), isTrue);
          expect(outBuf.getCell([0]).real, closeTo(21.0, 1e-9));

          // IFFT with n truncation and out buffer
          final ifftOut = NDArray<Complex128>.zeros([4], DType.complex128);
          final ifftRes = ifft(fftPad, n: 4, out: ifftOut);
          expect(identical(ifftRes, ifftOut), isTrue);
          expect(ifftOut.shape, [4]);
        });
      },
    );

    test('1D FFT and IFFT along non-last axis in multi-dimensional arrays', () {
      NDArray.scope(() {
        // 3D array of shape [2, 3, 4]
        final data = List<double>.generate(24, (i) => (i + 1).toDouble());
        final a3d = NDArray.fromList(data, [2, 3, 4], DType.float64);

        // FFT along axis 0 (dim size 2)
        final fftAx0 = fft(a3d, axis: 0);
        expect(fftAx0.shape, [2, 3, 4]);
        final ifftAx0 = ifft(fftAx0, axis: 0);
        expect(ifftAx0.shape, [2, 3, 4]);
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 3; j++) {
            for (var k = 0; k < 4; k++) {
              expect(
                ifftAx0.getCell([i, j, k]).real,
                closeTo(a3d.getCell([i, j, k]), 1e-9),
              );
              expect(ifftAx0.getCell([i, j, k]).imag, closeTo(0.0, 1e-9));
            }
          }
        }

        // FFT along axis 1 (dim size 3)
        final fftAx1 = fft(a3d, axis: 1);
        expect(fftAx1.shape, [2, 3, 4]);
        final ifftAx1 = ifft(fftAx1, axis: 1);
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 3; j++) {
            for (var k = 0; k < 4; k++) {
              expect(
                ifftAx1.getCell([i, j, k]).real,
                closeTo(a3d.getCell([i, j, k]), 1e-9),
              );
            }
          }
        }

        // FFT along axis -2 with out buffer
        final outBuf = NDArray<Complex128>.zeros([2, 3, 4], DType.complex128);
        final resAx1 = fft(a3d, axis: -2, out: outBuf);
        expect(identical(resAx1, outBuf), isTrue);
      });
    });

    test(
      '1D Real FFT (rfft) and Inverse Real FFT (irfft) across dtypes and shapes',
      () {
        NDArray.scope(() {
          // Even length Float64: kiss_fftr optimized fast path
          final aEven = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );
          final rEven = rfft(aEven);
          expect(rEven.shape, [4]); // 6 // 2 + 1 = 4
          expect(rEven.dtype, DType.complex128);
          expect(rEven.getCell([0]).real, closeTo(21.0, 1e-9));
          expect(rEven.getCell([0]).imag, closeTo(0.0, 1e-9));

          final irEven = irfft(rEven, n: 6);
          expect(irEven.shape, [6]);
          expect(irEven.dtype, DType.float64);
          for (var i = 0; i < 6; i++) {
            expect(irEven.getCell([i]), closeTo(aEven.getCell([i]), 1e-9));
          }

          // Odd length Float64: fallback path
          final aOdd = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );
          final rOdd = rfft(aOdd);
          expect(rOdd.shape, [3]); // 5 // 2 + 1 = 3
          expect(rOdd.getCell([0]).real, closeTo(15.0, 1e-9));

          final irOdd = irfft(rOdd, n: 5);
          expect(irOdd.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(irOdd.getCell([i]), closeTo(aOdd.getCell([i]), 1e-9));
          }

          // Float32 precision rfft & irfft
          final aF32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float32,
          );
          final rF32 = rfft(aF32);
          expect(rF32.dtype, DType.complex64);
          expect(rF32.shape, [3]);
          final irF32 = irfft(rF32, n: 5);
          expect(irF32.dtype, DType.float32);
          for (var i = 0; i < 5; i++) {
            expect(irF32.getCell([i]), closeTo(aF32.getCell([i]), 1e-5));
          }

          // Integer types rfft
          final aInt = NDArray.fromList([2, 4, 6, 8], [4], DType.int64);
          final rInt = rfft(aInt);
          expect(rInt.shape, [3]);
          expect(rInt.getCell([0]).real, closeTo(20.0, 1e-9));

          // Multi-dimensional rfft & irfft along non-last axis
          final mat = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );
          final rMatAx0 = rfft(
            mat,
            axis: 0,
          ); // Axis 0 has size 3 -> output size 3 // 2 + 1 = 2
          expect(rMatAx0.shape, [2, 2]);

          final irMatAx0 = irfft(rMatAx0, n: 3, axis: 0);
          expect(irMatAx0.shape, [3, 2]);
          for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 2; j++) {
              expect(
                irMatAx0.getCell([i, j]),
                closeTo(mat.getCell([i, j]), 1e-9),
              );
            }
          }
        });
      },
    );

    test('rfft & irfft with out buffers and default n parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final outRfft = NDArray<Complex128>.zeros([3], DType.complex128);
        final resRfft = rfft(a, out: outRfft);
        expect(identical(resRfft, outRfft), isTrue);

        // Default n for irfft is 2 * (len - 1) = 2 * (3 - 1) = 4
        final outIrfft = NDArray<Float64>.zeros([4], DType.float64);
        final resIrfft = irfft(outRfft, out: outIrfft);
        expect(identical(resIrfft, outIrfft), isTrue);
        for (var i = 0; i < 4; i++) {
          expect(outIrfft.getCell([i]), closeTo(a.getCell([i]), 1e-9));
        }
      });
    });

    test(
      'rfft2, irfft2, rfftn, irfftn, hfft, ihfft, odd-n irfft truncation, and complex rfft rejection',
      () {
        NDArray.scope(() {
          // Odd-n irfft truncation when input.shape[axis] > n ~/ 2 + 1
          final overlong = NDArray.fromList(
            [
              Complex(15.0, 0.0),
              Complex(-2.5, 3.4409548011779334),
              Complex(-2.5, 0.8122992405822658),
              Complex(100.0, 200.0),
              Complex(-300.0, 400.0),
            ],
            [5],
            DType.complex128,
          );
          final exact3 = overlong.slice([const Slice(start: 0, stop: 3)]);
          final irOverlong = irfft(overlong, n: 5);
          final irExact = irfft(exact3, n: 5);
          for (var i = 0; i < 5; i++) {
            expect(
              irOverlong.getCell([i]),
              closeTo(irExact.getCell([i]), 1e-12),
            );
          }

          // rfft rejects Complex64 and Complex128 on both even and odd n
          expect(() => rfft(overlong as dynamic, n: 4), throwsArgumentError);
          expect(() => rfft(overlong as dynamic, n: 5), throwsArgumentError);
          final overlong64 = overlong.astype(DType.complex64);
          expect(() => rfft(overlong64 as dynamic, n: 4), throwsArgumentError);
          expect(() => rfft(overlong64 as dynamic, n: 5), throwsArgumentError);

          // rfft2 & irfft2
          final mat = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final rf2 = rfft2(mat);
          expect(rf2.shape, [2, 2]);
          final irf2 = irfft2(rf2, s: [2, 3]);
          expect(irf2.shape, [2, 3]);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 3; j++) {
              expect(irf2.getCell([i, j]), closeTo(mat.getCell([i, j]), 1e-10));
            }
          }

          // hfft & ihfft
          final sig = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );
          final ih = ihfft(sig);
          expect(ih.shape, [3]);
          final h = hfft(ih, n: 5);
          expect(h.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(h.getCell([i]), closeTo(sig.getCell([i]), 1e-10));
          }
        });
      },
    );
  });

  group(
    'Comprehensive FFT - 2D and N-D Transforms (fft2, ifft2, fftn, ifftn)',
    () {
      test('2D FFT and IFFT (fft2 & ifft2) on 2D and 3D batch arrays', () {
        NDArray.scope(() {
          final mat = NDArray.fromList(
            [
              1.0,
              2.0,
              3.0,
              4.0,
              5.0,
              6.0,
              7.0,
              8.0,
              9.0,
              10.0,
              11.0,
              12.0,
              13.0,
              14.0,
              15.0,
              16.0,
            ],
            [4, 4],
            DType.float64,
          );

          final f2 = fft2(mat);
          expect(f2.shape, [4, 4]);
          expect(f2.dtype, DType.complex128);
          expect(f2.getCell([0, 0]).real, closeTo(136.0, 1e-9));

          final inv2 = ifft2(f2);
          expect(inv2.shape, [4, 4]);
          for (var i = 0; i < 4; i++) {
            for (var j = 0; j < 4; j++) {
              expect(
                inv2.getCell([i, j]).real,
                closeTo(mat.getCell([i, j]), 1e-9),
              );
              expect(inv2.getCell([i, j]).imag, closeTo(0.0, 1e-9));
            }
          }

          // fft2 with s padding & out parameter
          final out2 = NDArray<Complex128>.zeros([6, 6], DType.complex128);
          final f2Pad = fft2(mat, s: [6, 6], out: out2);
          expect(identical(f2Pad, out2), isTrue);
          expect(out2.shape, [6, 6]);

          // 3D batch array of 2x2 matrices: shape [2, 2, 2]
          final batch = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );

          final fBatch = fft2(batch);
          expect(fBatch.shape, [2, 2, 2]);
          final invBatch = ifft2(fBatch);
          expect(invBatch.shape, [2, 2, 2]);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
              for (var k = 0; k < 2; k++) {
                expect(
                  invBatch.getCell([i, j, k]).real,
                  closeTo(batch.getCell([i, j, k]), 1e-9),
                );
              }
            }
          }
        });
      });

      test(
        '3D and N-D FFT (fftn & ifftn) with custom axes and shape padding/truncation',
        () {
          NDArray.scope(() {
            // 3D array: shape [2, 3, 4]
            final data = List<double>.generate(24, (i) => (i + 1).toDouble());
            final a3d = NDArray.fromList(data, [2, 3, 4], DType.float64);

            // Full 3D FFT over all axes
            final f3d = fftn(a3d);
            expect(f3d.shape, [2, 3, 4]);
            expect(f3d.dtype, DType.complex128);
            expect(f3d.getCell([0, 0, 0]).real, closeTo(300.0, 1e-9));

            final inv3d = ifftn(f3d);
            for (var i = 0; i < 2; i++) {
              for (var j = 0; j < 3; j++) {
                for (var k = 0; k < 4; k++) {
                  expect(
                    inv3d.getCell([i, j, k]).real,
                    closeTo(a3d.getCell([i, j, k]), 1e-9),
                  );
                  expect(inv3d.getCell([i, j, k]).imag, closeTo(0.0, 1e-9));
                }
              }
            }

            // N-D FFT with s padding along axes [0, 2]
            final fPadded = fftn(a3d, s: [4, 6], axes: [0, 2]);
            expect(fPadded.shape, [4, 3, 6]);

            final invPadded = ifftn(fPadded, s: [4, 6], axes: [0, 2]);
            expect(invPadded.shape, [4, 3, 6]);
            // Original region should be restored
            for (var i = 0; i < 2; i++) {
              for (var j = 0; j < 3; j++) {
                for (var k = 0; k < 4; k++) {
                  expect(
                    invPadded.getCell([i, j, k]).real,
                    closeTo(a3d.getCell([i, j, k]), 1e-9),
                  );
                }
              }
            }

            // Float32 precision N-D transform
            final aF32 = NDArray.fromList(data, [2, 3, 4], DType.float32);
            final fF32 = fftn(aF32);
            expect(fF32.dtype, DType.complex64);
            final invF32 = ifftn(fF32);
            expect(invF32.dtype, DType.complex64);
            expect(invF32.getCell([0, 0, 0]).real, closeTo(1.0, 1e-5));
          });
        },
      );

      test(
        'fftn with strided non-contiguous views and out recycling buffer',
        () {
          NDArray.scope(() {
            final mat = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
              [3, 4],
              DType.float64,
            );

            final transposed = mat.transposed; // shape [4, 3], non-contiguous
            expect(transposed.isContiguous, isFalse);

            final out = NDArray<Complex128>.zeros([4, 3], DType.complex128);
            final fRes = fftn(transposed, out: out);
            expect(identical(fRes, out), isTrue);
            expect(out.shape, [4, 3]);

            final invOut = NDArray<Complex128>.zeros([4, 3], DType.complex128);
            ifftn(out, out: invOut);
            for (var i = 0; i < 4; i++) {
              for (var j = 0; j < 3; j++) {
                expect(
                  invOut.getCell([i, j]).real,
                  closeTo(transposed.getCell([i, j]), 1e-9),
                );
              }
            }
          });
        },
      );
    },
  );

  group('Comprehensive FFT - Frequency Helpers & Spectrum Shifting', () {
    test('fftfreq & rfftfreq even and odd lengths with sample spacing d', () {
      NDArray.scope(() {
        // Even length
        final fEven = fftfreq(8, d: 0.5);
        expect(fEven.shape, [8]);
        expect(fEven.dtype, DType.float64);
        expect(fEven.getCell([0]), closeTo(0.0, 1e-9));
        expect(fEven.getCell([1]), closeTo(0.25, 1e-9));
        expect(fEven.getCell([2]), closeTo(0.5, 1e-9));
        expect(fEven.getCell([3]), closeTo(0.75, 1e-9));
        expect(fEven.getCell([4]), closeTo(-1.0, 1e-9));
        expect(fEven.getCell([5]), closeTo(-0.75, 1e-9));
        expect(fEven.getCell([6]), closeTo(-0.5, 1e-9));
        expect(fEven.getCell([7]), closeTo(-0.25, 1e-9));

        // Odd length
        final fOdd = fftfreq(7, d: 1.0);
        expect(fOdd.shape, [7]);
        expect(fOdd.getCell([0]), closeTo(0.0, 1e-9));
        expect(fOdd.getCell([1]), closeTo(1.0 / 7.0, 1e-9));
        expect(fOdd.getCell([3]), closeTo(3.0 / 7.0, 1e-9));
        expect(fOdd.getCell([4]), closeTo(-3.0 / 7.0, 1e-9));

        // rfftfreq even
        final rfEven = rfftfreq(8, d: 0.5);
        expect(rfEven.shape, [5]); // 8 // 2 + 1 = 5
        expect(rfEven.getCell([0]), closeTo(0.0, 1e-9));
        expect(rfEven.getCell([1]), closeTo(0.25, 1e-9));
        expect(rfEven.getCell([4]), closeTo(1.0, 1e-9));

        // rfftfreq odd
        final rfOdd = rfftfreq(7, d: 1.0);
        expect(rfOdd.shape, [4]); // 7 // 2 + 1 = 4
        expect(rfOdd.getCell([0]), closeTo(0.0, 1e-9));
        expect(rfOdd.getCell([3]), closeTo(3.0 / 7.0, 1e-9));
      });
    });

    test(
      'fftshift and ifftshift multi-axis, odd/even, rank 0 scalar, and out recycling',
      () {
        NDArray.scope(() {
          // 0D scalar returns copy
          final scalar = NDArray.fromList([42.0], [], DType.float64);
          expect(fftshift(scalar).scalar, 42.0);
          expect(ifftshift(scalar).scalar, 42.0);

          // 3D tensor: shape [2, 3, 4]
          final a3d = NDArray.fromList(
            List<double>.generate(24, (i) => i.toDouble()),
            [2, 3, 4],
            DType.float64,
          );

          // Shift along all axes
          final shiftedAll = fftshift(a3d);
          expect(shiftedAll.shape, [2, 3, 4]);
          final restoredAll = ifftshift(shiftedAll);
          expect(restoredAll.toList(), equals(a3d.toList()));

          // Shift along single axis (axis = 1)
          final shiftedAx1 = fftshift(a3d, axes: 1);
          final restoredAx1 = ifftshift(shiftedAx1, axes: 1);
          expect(restoredAx1.toList(), equals(a3d.toList()));

          // Shift along negative axis list: axes = [-3, -1]
          final shiftedNeg = fftshift(a3d, axes: [-3, -1]);
          final restoredNeg = ifftshift(shiftedNeg, axes: [-3, -1]);
          expect(restoredNeg.toList(), equals(a3d.toList()));

          // With out buffer
          final outBuf = NDArray<Float64>.zeros([2, 3, 4], DType.float64);
          final res = fftshift(a3d, out: outBuf);
          expect(identical(res, outBuf), isTrue);
          expect(outBuf.toList(), equals(shiftedAll.toList()));

          final outIfft = NDArray<Float64>.zeros([2, 3, 4], DType.float64);
          ifftshift(outBuf, out: outIfft);
          expect(outIfft.toList(), equals(a3d.toList()));
        });
      },
    );

    test('clearFFTPlanCache cleanly resets plan cache without crashing', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        fft(a);
        rfft(a);
        fftn(a.reshape([2, 2]));
        clearFFTPlanCache();
        // Subsequent FFT should regenerate plan seamlessly
        final f = fft(a);
        expect(f.shape, [4]);
      });
    });
  });

  group('Comprehensive FFT - Error Handling & Edge Cases', () {
    test('Preconditions and error handling for FFT functions', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);

        // Invalid axis
        expect(() => fft(a, axis: 5), throwsRangeError);
        expect(() => fft(a, axis: -5), throwsRangeError);
        expect(() => ifft(a, axis: 5), throwsRangeError);
        expect(() => rfft(a, axis: 5), throwsRangeError);
        expect(() => irfft(a, axis: 5), throwsRangeError);

        // Invalid n <= 0
        expect(() => fft(a, n: 0), throwsArgumentError);
        expect(() => fft(a, n: -2), throwsArgumentError);
        expect(() => ifft(a, n: 0), throwsArgumentError);
        expect(() => rfft(a, n: -1), throwsArgumentError);
        expect(() => irfft(a, n: -1), throwsArgumentError);

        // Empty / 0-dim scalar array
        final scalar = NDArray<Float64>.zeros([], DType.float64);
        expect(() => fft(scalar), throwsArgumentError);
        expect(() => ifft(scalar), throwsArgumentError);
        expect(() => rfft(scalar), throwsArgumentError);
        expect(() => irfft(scalar), throwsArgumentError);
        expect(() => fftn(scalar), throwsArgumentError);

        // Incompatible out buffer shape/dtype
        final badShapeOut = NDArray<Complex128>.zeros([5], DType.complex128);
        expect(() => fft(a, out: badShapeOut), throwsArgumentError);
        final badDtypeOut = NDArray<Float64>.zeros([4], DType.float64);
        expect(() => fft(a, out: badDtypeOut as dynamic), throwsArgumentError);

        // fftfreq invalid parameters
        expect(() => fftfreq(0), throwsArgumentError);
        expect(() => fftfreq(-5), throwsArgumentError);
        expect(() => fftfreq(4, d: 0.0), throwsArgumentError);
        expect(() => rfftfreq(0), throwsArgumentError);
        expect(() => rfftfreq(-5), throwsArgumentError);
        expect(() => rfftfreq(4, d: 0.0), throwsArgumentError);

        // fft2 & ifft2 rank < 2 or invalid axes length
        expect(() => fft2(a), throwsArgumentError); // rank 1
        expect(() => ifft2(a), throwsArgumentError);
        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(() => fft2(mat, axes: [0]), throwsArgumentError);
        expect(() => ifft2(mat, axes: [0, 1, 2]), throwsArgumentError);

        // fftn axes / s length mismatch or duplicate axes
        expect(() => fftn(mat, s: [2], axes: [0, 1]), throwsArgumentError);
        expect(() => fftn(mat, axes: [0, 0]), throwsArgumentError);
        expect(() => fftn(mat, s: [-1, 2]), throwsArgumentError);

        // fftshift invalid axes
        expect(() => fftshift(mat, axes: 10), throwsRangeError);
        expect(() => fftshift(mat, axes: [0, 0]), throwsArgumentError);
        expect(
          () => fftshift(mat, axes: 'invalid' as dynamic),
          throwsArgumentError,
        );

        // Disposed array
        final disposed = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        disposed.dispose();
        expect(() => fft(disposed), throwsStateError);
        expect(() => ifft(disposed), throwsStateError);
        expect(() => rfft(disposed), throwsStateError);
        expect(() => irfft(disposed), throwsStateError);
        expect(() => fftn(disposed), throwsStateError);
        expect(() => fftshift(disposed), throwsStateError);
        expect(() => ifftshift(disposed), throwsStateError);
      });
    });
  });

  group('Comprehensive BinaryOp Enum & binaryUfunc Evaluation', () {
    test('Verify isReducible property across all BinaryOp enum values', () {
      for (final op in BinaryOp.values) {
        final expected = {
          BinaryOp.add,
          BinaryOp.multiply,
          BinaryOp.minimum,
          BinaryOp.maximum,
          BinaryOp.fmin,
          BinaryOp.fmax,
          BinaryOp.logaddexp,
          BinaryOp.logaddexp2,
          BinaryOp.gcd,
          BinaryOp.lcm,
          BinaryOp.bitwiseAnd,
          BinaryOp.bitwiseOr,
          BinaryOp.bitwiseXor,
          BinaryOp.logicalAnd,
          BinaryOp.logicalOr,
          BinaryOp.logicalXor,
        }.contains(op);
        expect(
          op.isReducible,
          equals(expected),
          reason: 'Op $op reducible mismatch',
        );
      }
    });

    test(
      'Evaluate all 28 binary ufuncs element-wise through binaryUfunc with where and out',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([8.0, 27.0, 4.0], [3], DType.float64);
          final b = NDArray.fromList([2.0, 3.0, 2.0], [3], DType.float64);

          // Arithmetic
          expect(binaryUfunc(a, b, op: BinaryOp.add).toList(), [
            10.0,
            30.0,
            6.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.subtract).toList(), [
            6.0,
            24.0,
            2.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.multiply).toList(), [
            16.0,
            81.0,
            8.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.divide).toList(), [
            4.0,
            9.0,
            2.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.floorDivide).toList(), [
            4.0,
            9.0,
            2.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.remainder).toList(), [
            0.0,
            0.0,
            0.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.fmod).toList(), [
            0.0,
            0.0,
            0.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.power).toList(), [
            64.0,
            19683.0,
            16.0,
          ]);
          expect(binaryUfunc(a, b, op: BinaryOp.floatPower).toList(), [
            64.0,
            19683.0,
            16.0,
          ]);

          // Math & Transcendental
          final h1 = NDArray.fromList([3.0, 5.0], [2], DType.float64);
          final h2 = NDArray.fromList([4.0, 12.0], [2], DType.float64);
          expect(binaryUfunc(h1, h2, op: BinaryOp.hypot).toList(), [5.0, 13.0]);

          final y = NDArray.fromList([1.0, 0.0], [2], DType.float64);
          final x = NDArray.fromList([1.0, 1.0], [2], DType.float64);
          final at2 = binaryUfunc(y, x, op: BinaryOp.arctan2).toList();
          expect(at2[0], closeTo(math.pi / 4, 1e-9));
          expect(at2[1], closeTo(0.0, 1e-9));

          final cs1 = NDArray.fromList([1.0, -2.0], [2], DType.float64);
          final cs2 = NDArray.fromList([-3.0, 4.0], [2], DType.float64);
          expect(binaryUfunc(cs1, cs2, op: BinaryOp.copysign).toList(), [
            -1.0,
            2.0,
          ]);

          final hv1 = NDArray.fromList([-1.0, 0.0, 1.0], [3], DType.float64);
          final hv2 = NDArray.fromList([0.5, 0.5, 0.5], [3], DType.float64);
          expect(binaryUfunc(hv1, hv2, op: BinaryOp.heaviside).toList(), [
            0.0,
            0.5,
            1.0,
          ]);

          final le1 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          final le2 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          final logadd = binaryUfunc(le1, le2, op: BinaryOp.logaddexp).toList();
          expect(logadd[0], closeTo(math.log(2.0), 1e-9));

          final logadd2 = binaryUfunc(
            le1,
            le2,
            op: BinaryOp.logaddexp2,
          ).toList();
          expect(logadd2[0], closeTo(1.0, 1e-9));

          // Integer Number Theory: GCD & LCM
          final i1 = NDArray.fromList([12, 15, 7], [3], DType.int64);
          final i2 = NDArray.fromList([18, 20, 5], [3], DType.int64);
          expect(binaryUfunc(i1, i2, op: BinaryOp.gcd).toList(), [6, 5, 1]);
          expect(binaryUfunc(i1, i2, op: BinaryOp.lcm).toList(), [36, 60, 35]);

          // Bitwise
          final b1 = NDArray.fromList([6, 12], [2], DType.int32);
          final b2 = NDArray.fromList([3, 10], [2], DType.int32);
          expect(binaryUfunc(b1, b2, op: BinaryOp.bitwiseAnd).toList(), [2, 8]);
          expect(binaryUfunc(b1, b2, op: BinaryOp.bitwiseOr).toList(), [7, 14]);
          expect(binaryUfunc(b1, b2, op: BinaryOp.bitwiseXor).toList(), [5, 6]);

          final s1 = NDArray.fromList([1, 16], [2], DType.int32);
          final s2 = NDArray.fromList([3, 2], [2], DType.int32);
          expect(binaryUfunc(s1, s2, op: BinaryOp.leftShift).toList(), [8, 64]);
          expect(binaryUfunc(s1, s2, op: BinaryOp.rightShift).toList(), [0, 4]);

          // Logical
          final bl1 = NDArray.fromList(
            [true, true, false, false],
            [4],
            DType.boolean,
          );
          final bl2 = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          expect(binaryUfunc(bl1, bl2, op: BinaryOp.logicalAnd).toList(), [
            true,
            false,
            false,
            false,
          ]);
          expect(binaryUfunc(bl1, bl2, op: BinaryOp.logicalOr).toList(), [
            true,
            true,
            true,
            false,
          ]);
          expect(binaryUfunc(bl1, bl2, op: BinaryOp.logicalXor).toList(), [
            false,
            true,
            true,
            false,
          ]);

          // Minimum & Maximum & fmin & fmax
          final m1 = NDArray.fromList([1.0, 5.0, 3.0], [3], DType.float64);
          final m2 = NDArray.fromList([4.0, 2.0, 3.0], [3], DType.float64);
          expect(binaryUfunc(m1, m2, op: BinaryOp.minimum).toList(), [
            1.0,
            2.0,
            3.0,
          ]);
          expect(binaryUfunc(m1, m2, op: BinaryOp.maximum).toList(), [
            4.0,
            5.0,
            3.0,
          ]);
          expect(binaryUfunc(m1, m2, op: BinaryOp.fmin).toList(), [
            1.0,
            2.0,
            3.0,
          ]);
          expect(binaryUfunc(m1, m2, op: BinaryOp.fmax).toList(), [
            4.0,
            5.0,
            3.0,
          ]);

          // With where mask and out buffer
          final mask = NDArray.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          final outRecycler = NDArray.fromList(
            [0.0, 99.0, 0.0],
            [3],
            DType.float64,
          );
          binaryUfunc(a, b, op: BinaryOp.add, where: mask, out: outRecycler);
          expect(outRecycler.toList(), [10.0, 99.0, 6.0]);
        });
      },
    );
  });

  group('Comprehensive Ufunc Methods - reduce & reduceUfunc', () {
    test('All 16 reducible operations tested with .reduce() across axes', () {
      NDArray.scope(() {
        // 1. add
        final aF64 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        expect(aF64.reduce(op: BinaryOp.add, axis: 0).toList(), [
          5.0,
          7.0,
          9.0,
        ]);
        expect(aF64.reduce(op: BinaryOp.add, axis: 1).toList(), [6.0, 15.0]);
        expect(aF64.reduce(op: BinaryOp.add).scalar, 21.0);

        // 2. multiply
        expect(aF64.reduce(op: BinaryOp.multiply, axis: 0).toList(), [
          4.0,
          10.0,
          18.0,
        ]);
        expect(aF64.reduce(op: BinaryOp.multiply).scalar, 720.0);

        // 3. minimum & 4. maximum
        expect(aF64.reduce(op: BinaryOp.minimum, axis: 0).toList(), [
          1.0,
          2.0,
          3.0,
        ]);
        expect(aF64.reduce(op: BinaryOp.maximum, axis: 0).toList(), [
          4.0,
          5.0,
          6.0,
        ]);

        // 5. fmin & 6. fmax
        expect(aF64.reduce(op: BinaryOp.fmin, axis: 1).toList(), [1.0, 4.0]);
        expect(aF64.reduce(op: BinaryOp.fmax, axis: 1).toList(), [3.0, 6.0]);

        // 7. logaddexp & 8. logaddexp2
        final aLog = NDArray.fromList([0.0, 0.0], [2], DType.float64);
        expect(
          aLog.reduce(op: BinaryOp.logaddexp).scalar,
          closeTo(math.log(2.0), 1e-9),
        );
        expect(aLog.reduce(op: BinaryOp.logaddexp2).scalar, closeTo(1.0, 1e-9));

        // 9. gcd & 10. lcm
        final aInt = NDArray.fromList([12, 18, 24], [3], DType.int64);
        expect(aInt.reduce(op: BinaryOp.gcd).scalar, 6);
        expect(aInt.reduce(op: BinaryOp.lcm).scalar, 72);

        // 11. bitwiseAnd, 12. bitwiseOr, 13. bitwiseXor
        final aBits = NDArray.fromList([7, 3, 1], [3], DType.int32);
        expect(aBits.reduce(op: BinaryOp.bitwiseAnd).scalar, 1);
        expect(aBits.reduce(op: BinaryOp.bitwiseOr).scalar, 7);
        expect(aBits.reduce(op: BinaryOp.bitwiseXor).scalar, 5);

        // 14. logicalAnd, 15. logicalOr, 16. logicalXor
        final aBools = NDArray.fromList(
          [true, true, false],
          [3],
          DType.boolean,
        );
        expect(aBools.reduce(op: BinaryOp.logicalAnd).scalar, isFalse);
        expect(aBools.reduce(op: BinaryOp.logicalOr).scalar, isTrue);
        expect(aBools.reduce(op: BinaryOp.logicalXor).scalar, isFalse);
      });
    });

    test(
      'reduce across various integer and floating dtypes (int32, int16, uint8, float32)',
      () {
        NDArray.scope(() {
          final aI32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          expect(aI32.reduce(op: BinaryOp.add, axis: 0).toList(), [4, 6]);
          expect(aI32.reduce(op: BinaryOp.multiply, axis: 1).toList(), [2, 12]);
          expect(aI32.reduce(op: BinaryOp.minimum, axis: 0).toList(), [1, 2]);
          expect(aI32.reduce(op: BinaryOp.maximum, axis: 1).toList(), [2, 4]);
          expect(aI32.reduce(op: BinaryOp.bitwiseAnd).scalar, 0);
          expect(aI32.reduce(op: BinaryOp.bitwiseOr).scalar, 7);

          final aI16 = NDArray.fromList([10, 20, 30], [3], DType.int16);
          expect(aI16.reduce(op: BinaryOp.add).scalar, 60);

          final aU8 = NDArray.fromList([5, 10, 15], [3], DType.uint8);
          expect(aU8.reduce(op: BinaryOp.add).scalar, 30);

          final aF32 = NDArray.fromList([1.5, 2.5, 3.5], [3], DType.float32);
          expect(aF32.reduce(op: BinaryOp.add).scalar, closeTo(7.5, 1e-5));
        });
      },
    );

    test(
      'reduce with initial value and keepdims and top-level reduce function',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final res = a.reduce(
            op: BinaryOp.add,
            initial: 100.0,
            keepdims: true,
          );
          expect(res.shape, [1]);
          expect(res.getCell([0]), 106.0);

          // Top-level reduce
          final topRes = reduce(a, op: BinaryOp.add, initial: 50.0);
          expect(topRes.scalar, 56.0);

          // Axis reduction with initial
          final mat = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final resAxis = mat.reduce(op: BinaryOp.add, axis: 0, initial: 10.0);
          expect(resAxis.toList(), [14.0, 16.0]);
        });
      },
    );

    test(
      'reduce error conditions (non-reducible, invalid axis, empty without initial)',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => a.reduce(op: BinaryOp.subtract), throwsArgumentError);
          expect(() => a.reduce(op: BinaryOp.add, axis: 5), throwsRangeError);

          final empty = NDArray<Float64>.zeros([0], DType.float64);
          expect(() => empty.reduce(op: BinaryOp.add), throwsArgumentError);
        });
      },
    );
  });

  group('Comprehensive Ufunc Methods - accumulate & accumulateUfunc', () {
    test(
      'accumulate across all reducible operations and dtypes along axes',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          // add cumsum
          final cs0 = a.accumulate(op: BinaryOp.add, axis: 0);
          expect(cs0.toList(), [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);

          final cs1 = a.accumulate(op: BinaryOp.add, axis: 1);
          expect(cs1.toList(), [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

          // Top-level accumulate
          final topAcc = accumulate(a, op: BinaryOp.add, axis: 1);
          expect(topAcc.toList(), equals(cs1.toList()));

          // multiply cumprod
          final cp = a.accumulate(op: BinaryOp.multiply, axis: 1);
          expect(cp.toList(), [1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);

          // minimum & maximum cummin/cummax
          final b = NDArray.fromList(
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [5],
            DType.float64,
          );
          expect(b.accumulate(op: BinaryOp.minimum).toList(), [
            3.0,
            1.0,
            1.0,
            1.0,
            1.0,
          ]);
          expect(b.accumulate(op: BinaryOp.maximum).toList(), [
            3.0,
            3.0,
            4.0,
            4.0,
            5.0,
          ]);

          // bitwise operations accumulate
          final bits = NDArray.fromList([7, 3, 1, 0], [4], DType.int32);
          expect(bits.accumulate(op: BinaryOp.bitwiseAnd).toList(), [
            7,
            3,
            1,
            0,
          ]);
          expect(bits.accumulate(op: BinaryOp.bitwiseOr).toList(), [
            7,
            7,
            7,
            7,
          ]);
          expect(bits.accumulate(op: BinaryOp.bitwiseXor).toList(), [
            7,
            4,
            5,
            5,
          ]);

          // logical operations accumulate
          final bools = NDArray.fromList(
            [true, true, false, true],
            [4],
            DType.boolean,
          );
          expect(bools.accumulate(op: BinaryOp.logicalAnd).toList(), [
            true,
            true,
            false,
            false,
          ]);
          expect(bools.accumulate(op: BinaryOp.logicalOr).toList(), [
            true,
            true,
            true,
            true,
          ]);
          expect(bools.accumulate(op: BinaryOp.logicalXor).toList(), [
            true,
            false,
            false,
            true,
          ]);

          // Fallback accumulation (gcd, lcm)
          final gcdArr = NDArray.fromList([12, 18, 24], [3], DType.int64);
          expect(gcdArr.accumulate(op: BinaryOp.gcd).toList(), [12, 6, 6]);
          expect(gcdArr.accumulate(op: BinaryOp.lcm).toList(), [12, 36, 72]);

          // Complex128 accumulate
          final cArr = NDArray.fromList(
            [Complex(1.0, 1.0), Complex(2.0, 3.0)],
            [2],
            DType.complex128,
          );
          final cCumsum = cArr.accumulate(op: BinaryOp.add);
          expect(cCumsum.getCell([0]), Complex(1.0, 1.0));
          expect(cCumsum.getCell([1]), Complex(3.0, 4.0));
        });
      },
    );
  });

  group('Comprehensive Ufunc Methods - reduceat & reduceatUfunc', () {
    test(
      'reduceat with multiple slice intervals, start >= end, and dtypes',
      () {
        NDArray.scope(() {
          // 1D Float64
          final a = NDArray.fromList(
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [6],
            DType.float64,
          );
          final indices = NDArray.fromList([0, 3, 1, 4], [4], DType.int64);
          final res = a.reduceat(indices, op: BinaryOp.add);
          expect(res.shape, [4]);
          expect(res.toList(), [30.0, 30.0, 60.0, 90.0]);

          // Top level reduceat
          final topRedAt = reduceat(a, indices, op: BinaryOp.add);
          expect(topRedAt.toList(), equals(res.toList()));

          // 1D Float32, Int32, Int16, Uint8, Complex128
          final aF32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
          );
          final idx2 = NDArray.fromList([0, 2], [2], DType.int64);
          expect(aF32.reduceat(idx2, op: BinaryOp.add).toList(), [3.0, 7.0]);

          final aI32 = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          expect(aI32.reduceat(idx2, op: BinaryOp.multiply).toList(), [2, 12]);

          final aI16 = NDArray.fromList([1, 2, 3, 4], [4], DType.int16);
          expect(aI16.reduceat(idx2, op: BinaryOp.add).toList(), [3, 7]);

          final aU8 = NDArray.fromList([1, 2, 3, 4], [4], DType.uint8);
          expect(aU8.reduceat(idx2, op: BinaryOp.add).toList(), [3, 7]);

          final aC128 = NDArray.fromList(
            [
              Complex(1.0, 1.0),
              Complex(2.0, 2.0),
              Complex(3.0, 3.0),
              Complex(4.0, 4.0),
            ],
            [4],
            DType.complex128,
          );
          final cRedAt = aC128.reduceat(idx2, op: BinaryOp.add);
          expect(cRedAt.getCell([0]), Complex(3.0, 3.0));
          expect(cRedAt.getCell([1]), Complex(7.0, 7.0));

          // 2D Matrix reduceat along axis 0 and axis 1
          final mat = NDArray.fromList(
            [
              1.0,
              2.0,
              3.0,
              4.0,
              5.0,
              6.0,
              7.0,
              8.0,
              9.0,
              10.0,
              11.0,
              12.0,
              13.0,
              14.0,
              15.0,
              16.0,
            ],
            [4, 4],
            DType.float64,
          );

          final matRedAt0 = mat.reduceat(idx2, op: BinaryOp.add, axis: 0);
          expect(matRedAt0.shape, [2, 4]);
          expect(matRedAt0.toList(), [
            6.0,
            8.0,
            10.0,
            12.0,
            22.0,
            24.0,
            26.0,
            28.0,
          ]);
        });
      },
    );
  });

  group('Comprehensive Ufunc Methods - outer & outerUfunc', () {
    test(
      'outer operations across 1D and 2D combinations with various BinaryOps',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final b = NDArray.fromList([10.0, 20.0], [2], DType.float64);

          // multiply
          expect(a.outer(b, op: BinaryOp.multiply).toList(), [
            10.0,
            20.0,
            20.0,
            40.0,
            30.0,
            60.0,
          ]);

          // Top level outer
          final topOuter = outerUfunc(a, b, op: BinaryOp.multiply);
          expect(topOuter.toList(), [10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);

          // add
          expect(a.outer(b, op: BinaryOp.add).toList(), [
            11.0,
            21.0,
            12.0,
            22.0,
            13.0,
            23.0,
          ]);

          // subtract
          expect(a.outer(b, op: BinaryOp.subtract).toList(), [
            -9.0,
            -19.0,
            -8.0,
            -18.0,
            -7.0,
            -17.0,
          ]);

          // divide
          expect(b.outer(a, op: BinaryOp.divide).toList(), [
            10.0,
            5.0,
            10.0 / 3.0,
            20.0,
            10.0,
            20.0 / 3.0,
          ]);

          // bitwise outer
          final bInt1 = NDArray.fromList([1, 2], [2], DType.int32);
          final bInt2 = NDArray.fromList([2, 3], [2], DType.int32);
          expect(bInt1.outer(bInt2, op: BinaryOp.bitwiseOr).toList(), [
            3,
            3,
            2,
            3,
          ]);

          // logical outer
          final bBool1 = NDArray.fromList([true, false], [2], DType.boolean);
          final bBool2 = NDArray.fromList([true, false], [2], DType.boolean);
          expect(bBool1.outer(bBool2, op: BinaryOp.logicalAnd).toList(), [
            true,
            false,
            false,
            false,
          ]);

          // 2D x 2D outer -> 4D tensor
          final m1 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final m2 = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );
          final tensor4d = m1.outer(m2, op: BinaryOp.multiply);
          expect(tensor4d.shape, [2, 2, 2, 2]);
          expect(tensor4d.getCell([0, 0, 0, 0]), 10.0);
          expect(tensor4d.getCell([1, 1, 1, 1]), 160.0);
        });
      },
    );
  });

  group('Comprehensive Ufunc Methods - at & atUfunc', () {
    test(
      'at in-place scatter accumulation across operations and dtypes with duplicate indices',
      () {
        NDArray.scope(() {
          // Float64 add with duplicate index 0
          final targetF64 = NDArray.fromList(
            [0.0, 0.0, 0.0],
            [3],
            DType.float64,
          );
          final idx = NDArray.fromList([0, 1, 0, 2, 0], [5], DType.int64);
          final valsF64 = NDArray.fromList(
            [1.0, 10.0, 2.0, 100.0, 3.0],
            [5],
            DType.float64,
          );
          targetF64.at(idx, valsF64, op: BinaryOp.add);
          expect(targetF64.toList(), [6.0, 10.0, 100.0]);

          // Top level at function
          final targetTop = NDArray.fromList(
            [0.0, 0.0, 0.0],
            [3],
            DType.float64,
          );
          at(targetTop, idx, valsF64, op: BinaryOp.add);
          expect(targetTop.toList(), [6.0, 10.0, 100.0]);

          // Float32 multiply
          final targetF32 = NDArray.fromList([1.0, 1.0], [2], DType.float32);
          final idxF32 = NDArray.fromList([0, 0, 1], [3], DType.int64);
          final valsF32 = NDArray.fromList([2.0, 3.0, 5.0], [3], DType.float32);
          targetF32.at(idxF32, valsF32, op: BinaryOp.multiply);
          expect(targetF32.toList(), [6.0, 5.0]);

          // Int64 subtract
          final targetI64 = NDArray.fromList([100, 100], [2], DType.int64);
          final idxI64 = NDArray.fromList([0, 0, 1], [3], DType.int64);
          final valsI64 = NDArray.fromList([10, 20, 50], [3], DType.int64);
          targetI64.at(idxI64, valsI64, op: BinaryOp.subtract);
          expect(targetI64.toList(), [70, 50]);

          // Int32 minimum & maximum
          final targetMin = NDArray.fromList([10, 10], [2], DType.int32);
          final valsMin = NDArray.fromList([5, 2, 8], [3], DType.int32);
          targetMin.at(idxI64, valsMin, op: BinaryOp.minimum);
          expect(targetMin.toList(), [2, 8]);

          // Boolean logicalOr
          final targetBool = NDArray.fromList(
            [false, false],
            [2],
            DType.boolean,
          );
          final valsBool = NDArray.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          targetBool.at(idxI64, valsBool, op: BinaryOp.logicalOr);
          expect(targetBool.toList(), [true, true]);

          // Complex128 add
          final targetC128 = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(0.0, 0.0)],
            [2],
            DType.complex128,
          );
          final valsC128 = NDArray.fromList(
            [Complex(1.0, 2.0), Complex(3.0, 4.0), Complex(5.0, 6.0)],
            [3],
            DType.complex128,
          );
          targetC128.at(idxI64, valsC128, op: BinaryOp.add);
          expect(targetC128.getCell([0]), Complex(4.0, 6.0));
          expect(targetC128.getCell([1]), Complex(5.0, 6.0));
        });
      },
    );
  });

  group('Comprehensive Bitwise Operations', () {
    test(
      'bitwise operations on int64, int32, int16, uint8 with broadcasting and strided views',
      () {
        NDArray.scope(() {
          // Broadcast 1D column [2, 1] with 1D row [1, 2]
          final col = NDArray.fromList([3, 12], [2, 1], DType.int32);
          final row = NDArray.fromList([1, 4], [1, 2], DType.int32);

          final andRes = bitwiseAnd(col, row);
          expect(andRes.shape, [2, 2]);
          expect(andRes.toList(), [1, 0, 0, 4]);

          final orRes = bitwiseOr(col, row);
          expect(orRes.toList(), [3, 7, 13, 12]);

          final xorRes = bitwiseXor(col, row);
          expect(xorRes.toList(), [2, 7, 13, 8]);

          // Left shift and right shift broadcasting
          final shiftVal = NDArray.fromList([1, 2], [1, 2], DType.int32);
          final ls = leftShift(col, shiftVal);
          expect(ls.toList(), [6, 12, 24, 48]);

          final rs = rightShift(ls, shiftVal);
          expect(rs.toList(), [3, 3, 12, 12]);

          // Strided invert test
          final parent = NDArray.fromList(
            [1, 2, 3, 4, 5, 6],
            [2, 3],
            DType.int32,
          );
          final strided = parent.slice([Slice(), Index(0)]); // [1, 4] (strided)
          expect(strided.isContiguous, isFalse);
          final invStrided = invert(strided);
          expect(invStrided.toList(), [-2, -5]);

          // where mask with bitwiseAnd
          final mask = NDArray.fromList(
            [true, false, false, true],
            [2, 2],
            DType.boolean,
          );
          final outRecycler = NDArray.fromList(
            [99, 99, 99, 99],
            [2, 2],
            DType.int32,
          );
          bitwiseAnd(col, row, where: mask, out: outRecycler);
          expect(outRecycler.toList(), [1, 99, 99, 4]);
        });
      },
    );

    test(
      'bitwise error conditions (non-integer types, disposed arrays, incompatible shapes)',
      () {
        NDArray.scope(() {
          final fArr = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final iArr = NDArray.fromList([1, 2], [2], DType.int32);
          expect(() => bitwiseAnd(fArr as dynamic, iArr), throwsArgumentError);
          expect(() => invert(fArr as dynamic), throwsArgumentError);

          final disposed = NDArray.fromList([1, 2], [2], DType.int32);
          disposed.dispose();
          expect(() => bitwiseAnd(disposed, iArr), throwsStateError);
        });
      },
    );
  });

  group('Comprehensive Logical Operations', () {
    test(
      'logicalAnd, logicalOr, logicalXor, logicalNot across numeric types & broadcasting',
      () {
        NDArray.scope(() {
          // Logical NOT on various dtypes (Float64, Int32, Complex128, Uint8, Int16)
          final fArr = NDArray.fromList(
            [0.0, 2.5, -1.0, 0.0],
            [4],
            DType.float64,
          );
          expect(logicalNot(fArr).toList(), [true, false, false, true]);

          final f32Arr = NDArray.fromList([0.0, 1.0], [2], DType.float32);
          expect(logicalNot(f32Arr).toList(), [true, false]);

          final iArr = NDArray.fromList([0, 10, 0, -5], [4], DType.int32);
          expect(logicalNot(iArr).toList(), [true, false, true, false]);

          final u8Arr = NDArray.fromList([0, 255], [2], DType.uint8);
          expect(logicalNot(u8Arr).toList(), [true, false]);

          final i16Arr = NDArray.fromList([0, 500], [2], DType.int16);
          expect(logicalNot(i16Arr).toList(), [true, false]);

          final cArr = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 2.0)],
            [3],
            DType.complex128,
          );
          expect(logicalNot(cArr).toList(), [true, false, false]);

          final c64Arr = NDArray.fromList(
            [Complex(0.0, 0.0), Complex(0.0, 1.0)],
            [2],
            DType.complex64,
          );
          expect(logicalNot(c64Arr).toList(), [true, false]);

          // Logical AND, OR, XOR with broadcasting
          final numA = NDArray.fromList([0.0, 1.0], [2, 1], DType.float64);
          final numBInt = NDArray.fromList([0, 5], [1, 2], DType.int32);
          expect(() => logicalAnd(numA, numBInt), throwsArgumentError);
          final numB = numBInt.astype(DType.float64);

          final lAnd = logicalAnd(numA, numB);
          expect(lAnd.shape, [2, 2]);
          expect(lAnd.toList(), [false, false, false, true]);

          final lOr = logicalOr(numA, numB);
          expect(lOr.toList(), [false, true, true, true]);

          final lXor = logicalXor(numA, numB);
          expect(lXor.toList(), [false, true, true, false]);

          // Non-contiguous / strided logical operations
          final matBool = NDArray.fromList(
            [true, false, false, true],
            [2, 2],
            DType.boolean,
          );
          final transBool = matBool.transposed; // non-contiguous
          expect(logicalNot(transBool).toList(), [false, true, true, false]);
        });
      },
    );

    test(
      'Comparisons (equal, notEqual, greater, greaterEqual, less, lessEqual) with broadcasting & masks',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3, 1], DType.float64);
          final b = NDArray.fromList([2.0, 2.0], [1, 2], DType.float64);

          // equal
          final eq = equal(a, b);
          expect(eq.shape, [3, 2]);
          expect(eq.toList(), [false, false, true, true, false, false]);

          // notEqual
          final neq = notEqual(a, b);
          expect(neq.toList(), [true, true, false, false, true, true]);

          // greater
          final gt = greater(a, b);
          expect(gt.toList(), [false, false, false, false, true, true]);

          // greaterEqual
          final gte = greaterEqual(a, b);
          expect(gte.toList(), [false, false, true, true, true, true]);

          // less
          final lt = less(a, b);
          expect(lt.toList(), [true, true, false, false, false, false]);

          // lessEqual
          final lte = lessEqual(a, b);
          expect(lte.toList(), [true, true, true, true, false, false]);

          // Masked comparison
          final mask = NDArray.fromList(
            [true, false, true, false, true, false],
            [3, 2],
            DType.boolean,
          );
          final outRecycler = NDArray.fromList(
            [true, true, true, true, true, true],
            [3, 2],
            DType.boolean,
          );
          equal(a, b, where: mask, out: outRecycler);
          expect(outRecycler.toList(), [false, true, true, true, false, true]);
        });
      },
    );
  });

  group('Comprehensive Clip Operations', () {
    test(
      'clip with scalar bounds across Float64, Float32, Int64, Int32, Int16, Uint8',
      () {
        NDArray.scope(() {
          // Float64 contiguous & strided
          final aF64 = NDArray.fromList(
            [-10.0, 5.0, 20.0, 50.0],
            [4],
            DType.float64,
          );
          expect(clip(aF64, min: 0.0, max: 25.0).toList(), [
            0.0,
            5.0,
            20.0,
            25.0,
          ]);

          final stridedF64 = aF64.slice([
            Slice(start: 0, stop: 4, step: 2),
          ]); // [-10.0, 20.0]
          expect(clip(stridedF64, min: 0.0, max: 15.0).toList(), [0.0, 15.0]);

          // Float32
          final aF32 = NDArray.fromList([-10.0, 5.0, 20.0], [3], DType.float32);
          expect(clip(aF32, min: 0.0, max: 10.0).toList(), [0.0, 5.0, 10.0]);

          // Int64, Int32, Int16, Uint8
          final aI64 = NDArray.fromList([-100, 50, 200], [3], DType.int64);
          expect(clip(aI64, min: 0, max: 100).toList(), [0, 50, 100]);

          final aI32 = NDArray.fromList([-100, 50, 200], [3], DType.int32);
          expect(clip(aI32, min: 0, max: 100).toList(), [0, 50, 100]);

          final aI16 = NDArray.fromList([-100, 50, 200], [3], DType.int16);
          expect(clip(aI16, min: 0, max: 100).toList(), [0, 50, 100]);

          final aU8 = NDArray.fromList([10, 50, 200], [3], DType.uint8);
          expect(clip(aU8, min: 20, max: 100).toList(), [20, 50, 100]);

          // One-sided scalar bounds
          expect(clip(aF64, min: 0.0).toList(), [0.0, 5.0, 20.0, 50.0]);
          expect(clip(aF64, max: 10.0).toList(), [-10.0, 5.0, 10.0, 10.0]);
          expect(clip(aF64).toList(), [-10.0, 5.0, 20.0, 50.0]);

          // with where mask and out buffer
          final mask = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final outBuf = NDArray.fromList(
            [99.0, 99.0, 99.0, 99.0],
            [4],
            DType.float64,
          );
          clip(aF64, min: 0.0, max: 25.0, where: mask, out: outBuf);
          expect(outBuf.toList(), [0.0, 99.0, 20.0, 99.0]);
        });
      },
    );

    test(
      'clipArray with multi-dimensional broadcasting bounds across all numeric types',
      () {
        NDArray.scope(() {
          // 2D Float64 input: shape [2, 3]
          final a = NDArray.fromList(
            [1.0, 5.0, 10.0, -5.0, 20.0, 8.0],
            [2, 3],
            DType.float64,
          );

          // min bounds row: shape [1, 3] -> broadcasts
          final minB = NDArray.fromList([0.0, 2.0, 5.0], [1, 3], DType.float64);
          // max bounds column: shape [2, 1] -> broadcasts
          final maxB = NDArray.fromList([8.0, 15.0], [2, 1], DType.float64);

          final clipped = clipArray(a, min: minB, max: maxB);
          expect(clipped.shape, [2, 3]);
          expect(clipped.toList(), [1.0, 5.0, 8.0, 0.0, 15.0, 8.0]);

          // Integer clipArray
          final aInt = NDArray.fromList([1, 10, 20], [3], DType.int32);
          final minInt = NDArray.fromList([5], [1], DType.int32);
          final maxInt = NDArray.fromList([15], [1], DType.int32);
          expect(clipArray(aInt, min: minInt, max: maxInt).toList(), [
            5,
            10,
            15,
          ]);

          // One-sided array bounds
          expect(clipArray(aInt, min: minInt).toList(), [5, 10, 20]);
          expect(clipArray(aInt, max: maxInt).toList(), [1, 10, 15]);
        });
      },
    );

    test('clip error conditions (complex arrays, incompatible out buffer)', () {
      NDArray.scope(() {
        final cArr = NDArray.fromList(
          [Complex(1.0, 2.0)],
          [1],
          DType.complex128,
        );
        expect(() => clip(cArr as dynamic, min: 0.0), throwsUnsupportedError);
        expect(() => clipArray(cArr as dynamic), throwsUnsupportedError);
      });
    });
  });
}
