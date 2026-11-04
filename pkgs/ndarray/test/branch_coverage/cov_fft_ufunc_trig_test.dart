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

import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('fft branch coverage', () {
    test(
      'fft, ifft, rfft, irfft, hfft, ihfft across all 15 dtypes & strided',
      () {
        NDArray.scope(() {
          final allArrays = <NDArray<DTypeTag>>[
            NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64),
            NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32),
            NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float16),
            NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.bfloat16),
            NDArray.fromList([1, 2, 3, 4], [4], DType.int64),
            NDArray.fromList([1, 2, 3, 4], [4], DType.int32),
            NDArray.fromList([1, 2, 3, 4], [4], DType.int16),
            NDArray.fromList([1, 2, 3, 4], [4], DType.int8),
            NDArray.fromList([1, 2, 3, 4], [4], DType.uint64),
            NDArray.fromList([1, 2, 3, 4], [4], DType.uint32),
            NDArray.fromList([1, 2, 3, 4], [4], DType.uint16),
            NDArray.fromList([1, 2, 3, 4], [4], DType.uint8),
            NDArray.fromList([true, false, true, false], [4], DType.boolean),
          ];

          for (final arr in allArrays) {
            final f = fft(arr.asAnySpec);
            expect(f.shape, [4]);
            final inv = ifft(arr.asAnySpec);
            expect(inv.shape, [4]);
            final rf = rfft(arr.asAnySpec, n: 6);
            expect(rf.shape, [4]);
            final ihf = ihfft(arr.asAnySpec, n: 6);
            expect(ihf.shape, [4]);
            final irf = irfft(arr.asAnySpec, n: 4);
            expect(irf.shape, [4]);
            final hf = hfft(arr.asAnySpec, n: 4);
            expect(hf.shape, [4]);

            // Strided 1D view (elemStride = 2)
            final strided = arr.slice([Slice(start: 0, stop: 4, step: 2)]);
            expect(fft(strided.asAnySpec, n: 4).shape, [4]);
          }

          final c128 = NDArray.fromList(
            [
              const Complex(1, 0),
              const Complex(2, 1),
              const Complex(3, -1),
              const Complex(4, 0),
            ],
            [4],
            DType.complex128,
          );
          final c64 = NDArray.fromList(
            [
              const Complex(1, 0),
              const Complex(2, 1),
              const Complex(3, -1),
              const Complex(4, 0),
            ],
            [4],
            DType.complex64,
          );
          final c128Strided = c128.slice([Slice(start: 0, stop: 4, step: 2)]);
          final c64Strided = c64.slice([Slice(start: 0, stop: 4, step: 2)]);
          final outC128Strided = NDArray.zeros([
            8,
          ], DType.complex128).slice([Slice(start: 0, stop: 8, step: 2)]);
          final outC64Strided = NDArray.zeros([
            8,
          ], DType.complex64).slice([Slice(start: 0, stop: 8, step: 2)]);
          fft(c128Strided, n: 4, out: outC128Strided);
          fft(c64Strided, n: 4, out: outC64Strided);
          ifft(c128Strided, n: 4, out: outC128Strided);
          ifft(c64Strided, n: 4, out: outC64Strided);

          // sharesMemory(a, out) in fft & ifft
          fft(c128, out: c128);
          ifft(c128, out: c128);

          // Empty array in fft, ifft, rfft, irfft with and without out
          final emptyC128 = NDArray.zeros([0, 4], DType.complex128);
          final emptyOutC128 = NDArray.zeros([0, 4], DType.complex128);
          expect(fft(emptyC128).shape, [0, 4]);
          expect(
            identical(fft(emptyC128, out: emptyOutC128), emptyOutC128),
            isTrue,
          );
          expect(ifft(emptyC128).shape, [0, 4]);
          expect(
            identical(ifft(emptyC128, out: emptyOutC128), emptyOutC128),
            isTrue,
          );

          final emptyF64 = NDArray.zeros([0, 4], DType.float64);
          final emptyRfftOut = NDArray.zeros([0, 3], DType.complex128);
          expect(rfft(emptyF64).shape, [0, 3]);
          expect(
            identical(rfft(emptyF64, out: emptyRfftOut), emptyRfftOut),
            isTrue,
          );
          expect(irfft(emptyRfftOut, n: 4).shape, [0, 4]);
          expect(
            identical(irfft(emptyRfftOut, n: 4, out: emptyF64), emptyF64),
            isTrue,
          );

          // ihfft zero-copy fast path (float64 -> complex128, targetLen == lastAxisDim)
          // and float32 path (float32 -> complex64)
          final f64 = allArrays[0] as NDArray<Float64>;
          final f32 = allArrays[1] as NDArray<Float32>;
          expect(ihfft(f64).shape, [3]);
          expect(ihfft(f32).shape, [3]);
          expect(ihfft(f32, n: 6).shape, [4]);
        });
      },
    );

    test('rfft, irfft, hfft non-last axis, out validation, and shift/freq', () {
      NDArray.scope(() {
        final i32 = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final outC64 = NDArray.zeros([3], DType.complex64);
        expect(
          identical(rfft(i32.asAnySpec, out: outC64.asAnySpec), outC64),
          isTrue,
        );

        final badOut = NDArray.zeros([3], DType.int32);
        expect(
          () => fft(i32.asAnySpec, out: badOut.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => ifft(i32.asAnySpec, out: badOut.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => rfft(i32.asAnySpec, out: badOut.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => ifft(
            NDArray.zeros([4], DType.complex128),
            out: NDArray.zeros([5], DType.complex128),
          ),
          throwsArgumentError,
        );
        expect(
          () => rfft(
            NDArray.zeros([4], DType.float64),
            out: NDArray.zeros([5], DType.complex128),
          ),
          throwsArgumentError,
        );

        // rfft non-contiguous out
        final outC128Base = NDArray.zeros([6], DType.complex128);
        final outC128Strided = outC128Base.slice([
          Slice(start: 0, stop: 6, step: 2),
        ]);
        expect(
          identical(rfft(i32, out: outC128Strided), outC128Strided),
          isTrue,
        );

        // rfft & irfft along axis 0 on 2D array
        final a2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [4, 2],
          DType.float64,
        );
        final rAxis0 = rfft(a2d, axis: 0);
        final outAxis0 = NDArray.zeros([3, 2], DType.complex128);
        rfft(a2d, axis: 0, out: outAxis0);
        irfft(rAxis0, axis: 0);
        final outIrAxis0 = NDArray.zeros([4, 2], DType.float64);
        irfft(rAxis0, axis: 0, out: outIrAxis0);

        // irfft out validation & non-contiguous out
        final outF32 = NDArray.zeros([6], DType.float32);
        irfft(i32.asAnySpec, out: outF32.asAnySpec);
        expect(
          () => irfft(
            i32.asAnySpec,
            out: NDArray.zeros([6], DType.int32).asAnySpec,
          ),
          throwsArgumentError,
        );
        expect(
          () => irfft(rAxis0, out: NDArray.zeros([5, 5], DType.float64)),
          throwsArgumentError,
        );
        final outF64Strided = NDArray.zeros([
          12,
        ], DType.float64).slice([Slice(start: 0, stop: 12, step: 2)]);
        irfft(i32, out: outF64Strided);

        // hfft with Complex64 (1D and 2D)
        final c64_1d = NDArray.fromList(
          [const Complex(1, 0), const Complex(2, -1), const Complex(3, 0)],
          [3],
          DType.complex64,
        );
        expect(hfft(c64_1d).dtype, DType.float32);
        final c64_2d = c64_1d.reshape([3, 1]);
        expect(hfft(c64_2d, axis: 0).shape, [4, 1]);
        final outHAxis0 = NDArray.zeros([4, 1], DType.float32);
        hfft(c64_2d, axis: 0, out: outHAxis0);

        // fftshift & ifftshift
        expect(fftshift(a2d).shape, [4, 2]);
        expect(ifftshift(a2d).shape, [4, 2]);
        expect(fftshift(a2d, axes: 0).shape, [4, 2]);
        expect(ifftshift(a2d, axes: [0, 1]).shape, [4, 2]);
        expect(() => fftshift(a2d, axes: 5), throwsRangeError);
        expect(() => fftshift(a2d, axes: [5]), throwsRangeError);
        expect(() => ifftshift(a2d, axes: 5), throwsRangeError);
        expect(() => ifftshift(a2d, axes: [5]), throwsRangeError);

        // fftfreq & rfftfreq
        expect(fftfreq(4).shape, [4]);
        expect(fftfreq(5).shape, [5]);
        expect(rfftfreq(4).shape, [3]);
        expect(rfftfreq(5).shape, [3]);
        final freqOut = NDArray.zeros([4], DType.float64);
        final freqOutStrided = NDArray.zeros([
          8,
        ], DType.float64).slice([Slice(start: 0, stop: 8, step: 2)]);
        fftfreq(4, out: freqOut);
        fftfreq(4, out: freqOutStrided);
        final rfreqOut = NDArray.zeros([3], DType.float64);
        final rfreqOutStrided = NDArray.zeros([
          6,
        ], DType.float64).slice([Slice(start: 0, stop: 6, step: 2)]);
        rfftfreq(4, out: rfreqOut);
        rfftfreq(4, out: rfreqOutStrided);
        expect(
          () => fftfreq(4, out: NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => rfftfreq(4, out: NDArray.zeros([4], DType.float64)),
          throwsArgumentError,
        );
        final disp = NDArray.zeros([4], DType.float64)..dispose();
        expect(() => fftfreq(4, out: disp), throwsStateError);
        expect(() => rfftfreq(4, out: disp), throwsStateError);
      });
    });

    test('fftn, ifftn, rfftn, irfftn, rfft2, irfft2 edge cases', () {
      NDArray.scope(() {
        final i32_2d = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final f64_2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 4],
          DType.float64,
        );
        final disp = NDArray.zeros([2, 2], DType.float64)..dispose();
        final dispC128 = NDArray.zeros([2, 4], DType.complex128)..dispose();

        expect(() => fftn(i32_2d, s: [2], axes: [0, 1]), throwsArgumentError);
        expect(() => fftn(i32_2d, s: [2, 0]), throwsArgumentError);
        expect(() => fftn(i32_2d, axes: [5]), throwsRangeError);
        expect(() => fftn(i32_2d, axes: [0, 0]), throwsArgumentError);
        expect(() => fftn(f64_2d, out: dispC128), throwsStateError);

        final outC64 = NDArray.zeros([2, 2], DType.complex64);
        fftn(i32_2d.asAnySpec, out: outC64.asAnySpec);

        final badOut = NDArray.zeros([2, 2], DType.int32);
        expect(
          () => fftn(i32_2d.asAnySpec, out: badOut.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => fftn(f64_2d, out: NDArray.zeros([3, 3], DType.complex128)),
          throwsArgumentError,
        );

        // fftn non-contiguous out and empty array
        final outFftnStrided = NDArray.zeros(
          [2, 8],
          DType.complex128,
        ).slice([Slice.all(), Slice(start: 0, stop: 8, step: 2)]);
        fftn(f64_2d, out: outFftnStrided);
        final empty2D = NDArray.zeros([0, 4], DType.complex128);
        expect(fftn(empty2D, axes: [1]).shape, [0, 4]);
        expect(
          identical(fftn(empty2D, axes: [1], out: empty2D), empty2D),
          isTrue,
        );

        final outFftn = NDArray.zeros([2, 4], DType.complex128);
        fftn(f64_2d, out: outFftn);
        expect(ifftn(f64_2d).shape, [2, 4]);
        expect(fft2(f64_2d).shape, [2, 4]);
        expect(ifft2(outFftn).shape, [2, 4]);

        // fftn with non-identity axes permutation and out != null
        final c128_2d = outFftn.copy();
        final outPerm = NDArray.zeros([2, 4], DType.complex128);
        fftn(c128_2d, axes: [1, 0], out: outPerm);

        // rfftn validation and edge cases
        expect(() => rfftn(disp), throwsStateError);
        expect(() => rfftn(f64_2d, out: dispC128), throwsStateError);
        expect(() => rfftn(f64_2d, s: []), throwsArgumentError);
        expect(() => rfftn(f64_2d, axes: []), throwsArgumentError);
        expect(() => rfftn(f64_2d, axes: [0, 0]), throwsArgumentError);
        expect(() => rfftn(f64_2d, s: [2], axes: [0, 1]), throwsArgumentError);
        expect(() => rfftn(f64_2d, s: [2, 0]), throwsArgumentError);

        final outRfftnC64 = NDArray.zeros([2, 2], DType.complex64);
        rfftn(i32_2d.asAnySpec, out: outRfftnC64.asAnySpec);
        expect(
          () => rfftn(i32_2d.asAnySpec, out: badOut.asAnySpec),
          throwsArgumentError,
        );

        final outRfftnStrided = NDArray.zeros(
          [2, 6],
          DType.complex128,
        ).slice([Slice.all(), Slice(start: 0, stop: 6, step: 2)]);
        rfftn(f64_2d, out: outRfftnStrided);

        final emptyF64 = NDArray.zeros([0, 4], DType.float64);
        final emptyRfftnOut = NDArray.zeros([0, 3], DType.complex128);
        expect(rfftn(emptyF64, axes: [1]).shape, [0, 3]);
        rfftn(emptyF64, axes: [1], out: emptyRfftnOut);

        // irfftn validation and edge cases
        final scalarF64 = NDArray.fromList([1.0], [], DType.float64);
        expect(() => irfftn(disp), throwsStateError);
        expect(() => irfftn(f64_2d, out: disp), throwsStateError);
        expect(() => irfftn(scalarF64), throwsArgumentError);
        expect(() => irfftn(f64_2d, s: []), throwsArgumentError);
        expect(() => irfftn(f64_2d, axes: []), throwsArgumentError);
        expect(() => irfftn(f64_2d, axes: [0, 0]), throwsArgumentError);
        expect(() => irfftn(f64_2d, axes: [5]), throwsRangeError);
        expect(() => irfftn(f64_2d, s: [2], axes: [0, 1]), throwsArgumentError);
        expect(() => irfftn(f64_2d, s: [2, 0]), throwsArgumentError);

        final outIrfftnF32 = NDArray.zeros([2, 2], DType.float32);
        irfftn(i32_2d.asAnySpec, out: outIrfftnF32.asAnySpec);
        expect(
          () => irfftn(i32_2d.asAnySpec, out: badOut.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => irfftn(
            outRfftnC64.asAnySpec,
            out: NDArray.zeros([2, 2], DType.float64).asAnySpec,
          ),
          throwsArgumentError,
        );
        expect(
          () => irfftn(outRfftnC64, out: NDArray.zeros([3, 3], DType.float32)),
          throwsArgumentError,
        );

        final outIrfftnStrided = NDArray.zeros(
          [2, 4],
          DType.float32,
        ).slice([Slice.all(), Slice(start: 0, stop: 4, step: 2)]);
        irfftn(outRfftnC64, s: [2, 2], out: outIrfftnStrided);

        final emptyC128 = NDArray.zeros([0, 3], DType.complex128);
        final emptyIrfftnOut = NDArray.zeros([0, 4], DType.float64);
        expect(irfftn(emptyC128, axes: [1]).shape, [0, 4]);
        irfftn(emptyC128, axes: [1], out: emptyIrfftnOut);
        expect(irfftn(outRfftnC64, axes: [1]).shape, [2, 2]);

        expect(rfft2(f64_2d, axes: null).shape, [2, 3]);
        expect(() => rfft2(f64_2d, axes: [0]), throwsArgumentError);
        expect(irfft2(outRfftnC64, axes: null).shape, [2, 2]);
        expect(() => irfft2(outRfftnC64, axes: [0]), throwsArgumentError);
      });
    });
  });

  group('bitwise branch coverage', () {
    test(
      'all integer/boolean dtypes, contiguous & strided, out aliasing, errors',
      () {
        NDArray.scope(() {
          final intDTypes = <DType<DTypeTag>>[
            DType.int64,
            DType.int32,
            DType.int16,
            DType.int8,
            DType.uint64,
            DType.uint32,
            DType.uint16,
            DType.uint8,
          ];
          for (final dt in intDTypes) {
            final a = NDArray.fromList([3, 5, 7, 9], [4], dt).asIntegerDType;
            final b = NDArray.fromList([1, 2, 1, 2], [4], dt).asIntegerDType;
            bitwiseAnd(a, b);
            bitwiseOr(a, b);
            bitwiseXor(a, b);
            leftShift(a, b);
            rightShift(a, b);
            invert(a);

            final aS = a.slice([Slice(start: 0, stop: 4, step: 2)]);
            final bS = b.slice([Slice(start: 0, stop: 4, step: 2)]);
            bitwiseAnd(aS, bS);
            bitwiseOr(aS, bS);
            bitwiseXor(aS, bS);
            leftShift(aS, bS);
            rightShift(aS, bS);
            invert(aS);
          }

          final ba = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final bb = NDArray.fromList(
            [true, true, false, false],
            [4],
            DType.boolean,
          );
          bitwiseAnd(ba, bb);
          bitwiseOr(ba, bb);
          bitwiseXor(ba, bb);
          invert(ba);
          final baS = ba.slice([Slice(start: 0, stop: 4, step: 2)]);
          final bbS = bb.slice([Slice(start: 0, stop: 4, step: 2)]);
          bitwiseAnd(baS, bbS);
          bitwiseOr(baS, bbS);
          bitwiseXor(baS, bbS);
          invert(baS);

          // Aliasing out (with and without where)
          final aI32 = NDArray.fromList([3, 5], [2], DType.int32);
          final bI32 = NDArray.fromList([1, 2], [2], DType.int32);
          final mask = NDArray.fromList([true, false], [2], DType.boolean);
          final outAlias = aI32.copy();
          bitwiseAnd(outAlias, bI32, where: mask, out: outAlias);
          bitwiseOr(outAlias, bI32, where: mask, out: outAlias);
          bitwiseXor(outAlias, bI32, where: mask, out: outAlias);
          leftShift(outAlias, bI32, where: mask, out: outAlias);
          rightShift(outAlias, bI32, where: mask, out: outAlias);
          invert(outAlias, where: mask, out: outAlias);
          invert(outAlias, out: outAlias);

          // _rightShiftScalar edge cases (negative and >= bitwidth shifts)
          for (final dt in [
            DType.int8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
          ]) {
            final aShift = NDArray.fromList(
              [
                -16,
                0,
                16,
                0,
                16,
                0,
                16,
                0,
              ].map((x) => dt == DType.int8 ? x : x.abs()).toList(),
              [8],
              dt,
            ).asIntegerDType;
            final bShift = NDArray.fromList(
              dt == DType.int8
                  ? [-1, 0, 65, 0, 2, 0, 0, 0]
                  : [65, 0, 65, 0, 2, 0, 0, 0],
              [8],
              dt,
            ).asIntegerDType;
            if (dt == DType.uint64) {
              bShift.setCellFlat(0, -1);
            }
            rightShift(
              aShift.slice([Slice(start: 0, stop: 8, step: 2)]),
              bShift.slice([Slice(start: 0, stop: 8, step: 2)]),
            );
          }

          // *As functions and error branches
          final outI32 = NDArray.zeros([2], DType.int32);
          final outI64 = NDArray.zeros([2], DType.int64);
          bitwiseAndAs(aI32, bI32, DType.int32, out: outI32);
          bitwiseAndAs(aI32, bI32, DType.int64, out: outI64);
          bitwiseOrAs(aI32, bI32, DType.int32, out: outI32);
          bitwiseOrAs(aI32, bI32, DType.int64, out: outI64);
          bitwiseXorAs(aI32, bI32, DType.int32, out: outI32);
          bitwiseXorAs(aI32, bI32, DType.int64, out: outI64);
          leftShiftAs(aI32, bI32, DType.int32, out: outI32);
          leftShiftAs(aI32, bI32, DType.int64, out: outI64);
          rightShiftAs(aI32, bI32, DType.int32, out: outI32);
          rightShiftAs(aI32, bI32, DType.int64, out: outI64);

          final dispMask = NDArray.fromList([true, false], [2], DType.boolean)
            ..dispose();
          expect(
            () => bitwiseAnd(aI32, bI32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => bitwiseOr(aI32, bI32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => bitwiseXor(aI32, bI32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => leftShift(aI32, bI32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => rightShift(aI32, bI32, where: dispMask),
            throwsStateError,
          );
          expect(() => invert(aI32, where: dispMask), throwsStateError);
          expect(
            () => bitwiseAndAs(aI32, bI32, DType.int32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => bitwiseOrAs(aI32, bI32, DType.int32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => bitwiseXorAs(aI32, bI32, DType.int32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => leftShiftAs(aI32, bI32, DType.int32, where: dispMask),
            throwsStateError,
          );
          expect(
            () => rightShiftAs(aI32, bI32, DType.int32, where: dispMask),
            throwsStateError,
          );

          final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(
            () => bitwiseAndAs(aI32, f64, DType.int32),
            throwsArgumentError,
          );
          expect(
            () => bitwiseAndAs(aI32, bI32, DType.float64),
            throwsArgumentError,
          );
          expect(
            () => leftShiftAs(f64, bI32, DType.int32),
            throwsArgumentError,
          );
          expect(
            () => invert(aI32, out: NDArray.zeros([3], DType.int32)),
            throwsArgumentError,
          );
          expect(
            () => bitwiseAnd(aI32, bI32, out: NDArray.zeros([3], DType.int32)),
            throwsArgumentError,
          );
        });
      },
    );
  });

  group('manipulation branch coverage', () {
    test(
      'concatenate, stack, expand_dims, squeeze, flip, rot90, vstack, hstack',
      () {
        NDArray.scope(() {
          final a2d = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final b2d = NDArray.fromList([5, 6, 7, 8], [2, 2], DType.int32);
          final disp = NDArray.zeros([2, 2], DType.int32)..dispose();

          // concatenate along axis 0 and axis 1, strided input, aliasing out, empty
          final a2dStrided = NDArray.zeros(
            [2, 4],
            DType.int32,
          ).slice([Slice.all(), Slice(start: 0, stop: 4, step: 2)]);
          final outCat1 = NDArray.zeros([2, 4], DType.int32);
          concatenate([a2dStrided, b2d], axis: 1, out: outCat1);
          final outCatStrided = NDArray.zeros(
            [2, 8],
            DType.int32,
          ).slice([Slice.all(), Slice(start: 0, stop: 8, step: 2)]);
          concatenate([a2d, b2d], axis: 1, out: outCatStrided);
          final emptyCat = NDArray.zeros([0, 2], DType.int32);
          expect(concatenate([emptyCat, emptyCat], axis: 0).shape, [0, 2]);
          concatenate([emptyCat, emptyCat], axis: 0, out: emptyCat);
          expect(() => concatenate([disp, a2d]), throwsStateError);
          expect(() => concatenate([a2d, b2d], out: disp), throwsStateError);
          expect(
            () => concatenate([
              a2d,
              b2d,
            ], out: NDArray.zeros([3, 3], DType.int32)),
            throwsArgumentError,
          );

          // stack
          final outStack = NDArray.zeros([2, 2, 2], DType.int32);
          stack([a2d, b2d], axis: 1, out: outStack);
          final outStackStrided = NDArray.zeros([2, 2, 4], DType.int32).slice([
            Slice.all(),
            Slice.all(),
            Slice(start: 0, stop: 4, step: 2),
          ]);
          stack([a2d, b2d], axis: 0, out: outStackStrided);
          final emptyStackOut = NDArray.zeros([0, 2, 2], DType.int32);
          expect(stack([emptyCat, emptyCat], axis: 1).shape, [0, 2, 2]);
          stack([emptyCat, emptyCat], axis: 1, out: emptyStackOut);
          expect(() => stack([a2d, b2d], out: disp), throwsStateError);
          expect(
            () => stack([a2d, b2d], out: NDArray.zeros([3, 3], DType.int32)),
            throwsArgumentError,
          );

          // copy, expand_dims, squeeze, slidingWindowView
          expect(() => copy(disp), throwsStateError);
          expect(() => expand_dims(disp, 0), throwsStateError);
          expect(() => squeeze(disp), throwsStateError);
          expect(
            () => squeeze(a2d.reshape([1, 2, 2]), axis: [5]),
            throwsRangeError,
          );
          expect(() => slidingWindowView(disp, [1, 1]), throwsStateError);
          expect(
            () => slidingWindowView(a2d, [1], axis: [5]),
            throwsRangeError,
          );
          expect(
            () => slidingWindowView(a2d, [1, 1], axis: [0, 0]),
            throwsArgumentError,
          );

          // flip, fliplr, flipud, rot90
          expect(() => flip(disp), throwsStateError);
          expect(flip(NDArray.fromList([1], [], DType.int32)).shape, <int>[]);
          expect(() => flip(a2d, axis: 5), throwsRangeError);
          expect(() => flip(a2d, axis: [0, 0]), throwsArgumentError);
          expect(() => fliplr(disp), throwsStateError);
          expect(() => flipud(disp), throwsStateError);
          expect(fliplr(a2d).shape, [2, 2]);
          expect(flipud(a2d).shape, [2, 2]);

          expect(() => rot90(disp), throwsStateError);
          expect(() => rot90(a2d, 1, [0]), throwsArgumentError);
          expect(() => rot90(a2d, 1, [0, 0]), throwsArgumentError);
          expect(rot90(a2d, 0).shape, [2, 2]);
          expect(rot90(a2d, 1).shape, [2, 2]);
          expect(rot90(a2d, 2).shape, [2, 2]);
          expect(rot90(a2d, 3).shape, [2, 2]);
          expect(rot90(a2d, -1).shape, [2, 2]);

          // vstack & hstack
          expect(() => vstack(<NDArray<Int32>>[]), throwsArgumentError);
          expect(() => hstack(<NDArray<Int32>>[]), throwsArgumentError);
          final v1 = NDArray.fromList([1, 2], [2], DType.int32);
          final v2 = NDArray.fromList([1, 2, 3], [1, 3], DType.int32);
          expect(() => vstack([v1, v2]), throwsArgumentError);
          expect(vstack([v1, v1]).shape, [4]);
          expect(hstack([v1, v1]).shape, [4]);
          expect(hstack([a2d, b2d]).shape, [2, 4]);
        });
      },
    );

    test('diag, tril, triu, diff, roll, astype', () {
      NDArray.scope(() {
        final disp = NDArray.zeros([2, 2], DType.int32)..dispose();
        final v1d = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final m2d = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);

        // diag 1D and 2D
        expect(() => diag(disp), throwsStateError);
        expect(
          () => diag(NDArray.zeros([1, 1, 1], DType.int32)),
          throwsArgumentError,
        );
        expect(diag(m2d, k: 0).toList(), [1, 5]);
        expect(diag(m2d, k: 1).toList(), [2, 6]);
        expect(diag(m2d, k: -1).toList(), [4]);
        expect(diag(m2d, k: 5).shape, [0]);
        final emptyDiagOut = NDArray.zeros([0], DType.int32);
        diag(m2d, k: 5, out: emptyDiagOut);
        expect(
          () => diag(m2d, k: 5, out: NDArray.zeros([1], DType.int32)),
          throwsArgumentError,
        );
        final diag2dOut = NDArray.zeros([2], DType.int32);
        diag(m2d, k: 0, out: diag2dOut);

        expect(diag(v1d, k: 0).shape, [3, 3]);
        expect(diag(v1d, k: 1).shape, [4, 4]);
        expect(diag(v1d, k: -1).shape, [4, 4]);
        final diag1dOut = NDArray.zeros([3, 3], DType.int32);
        diag(v1d, k: 0, out: diag1dOut);
        final diag1dStridedOut = NDArray.zeros([
          3,
          6,
        ], DType.int32).slice([Slice.all(), Slice(start: 0, stop: 6, step: 2)]);
        diag(v1d, k: 0, out: diag1dStridedOut);
        expect(
          () => diag(v1d, out: NDArray.zeros([2, 2], DType.int32)),
          throwsArgumentError,
        );

        // tril & triu
        expect(() => tril(disp), throwsStateError);
        expect(() => triu(disp), throwsStateError);
        expect(() => tril(v1d), throwsArgumentError);
        expect(() => triu(v1d), throwsArgumentError);
        expect(
          () => tril(m2d, out: NDArray.zeros([3, 3], DType.int32)),
          throwsArgumentError,
        );
        expect(
          () => triu(m2d, out: NDArray.zeros([3, 3], DType.int32)),
          throwsArgumentError,
        );
        for (final dt in [
          DType.float64,
          DType.float32,
          DType.float16,
          DType.int32,
        ]) {
          final arr = NDArray.fromList(List.generate(12, (i) => i + 1), [
            2,
            2,
            3,
          ], dt);
          tril(arr, k: 0);
          triu(arr, k: 0);
        }
        final m2dStrided = NDArray.zeros([
          2,
          6,
        ], DType.int32).slice([Slice.all(), Slice(start: 0, stop: 6, step: 2)]);
        tril(m2d, out: m2dStrided);
        triu(m2d, out: m2dStrided);

        // diff across all 15 dtypes, n=0,1,2, n>=shape, aliasing out
        expect(() => diff(disp), throwsStateError);
        expect(() => diff(v1d, n: -1), throwsArgumentError);
        expect(() => diff(v1d, axis: 5), throwsRangeError);
        expect(diff(v1d, n: 0).toList(), [1, 2, 3]);
        expect(diff(v1d, n: 2).toList(), [0]);
        expect(diff(v1d, n: 5).shape, [0]);
        final emptyDiffOut = NDArray.zeros([0], DType.int32);
        diff(v1d, n: 5, out: emptyDiffOut);
        expect(
          () => diff(v1d, n: 5, out: NDArray.zeros([1], DType.int32)),
          throwsArgumentError,
        );
        expect(
          () => diff(v1d, n: 1, out: NDArray.zeros([5], DType.int32)),
          throwsArgumentError,
        );
        final diffOutStrided = NDArray.zeros([
          4,
        ], DType.int32).slice([Slice(start: 0, stop: 4, step: 2)]);
        diff(v1d, n: 1, out: diffOutStrided);

        for (final dt in DType.values) {
          final list = dt.isComplex
              ? [const Complex(1, 0), const Complex(3, 1), const Complex(6, 2)]
              : (dt == DType.boolean ? [true, false, true] : [1, 3, 6]);
          final arr = NDArray.fromList(list, [3], dt);
          expect(diff(arr.asAnySpec).shape, [2]);
        }

        // roll & astype
        expect(() => roll(disp, 1), throwsStateError);
        expect(() => roll(m2d, [1, 2]), throwsArgumentError);
        expect(() => roll(m2d, [1, 2], axis: 0), throwsArgumentError);
        expect(roll(NDArray.fromList([1], [], DType.int32), 1).shape, <int>[]);
        expect(roll(m2d, 1, axis: <int>[]).shape, [2, 3]);
        expect(roll(m2d, 0, axis: 0).shape, [2, 3]);
        expect(roll(v1d, 0).toList(), [1, 2, 3]);
        expect(roll(m2d, 1).shape, [2, 3]);
        expect(roll(m2dStrided, 1).shape, [2, 3]);
        final m2dStridedOut = NDArray.zeros([
          2,
          6,
        ], DType.int32).slice([Slice.all(), Slice(start: 0, stop: 6, step: 2)]);
        roll(m2d, 1, out: m2dStridedOut);
        roll(m2d, [1, 1], axis: [0, 1], out: m2dStridedOut);
        roll(v1d.slice([Slice(start: 0, stop: 3, step: 2)]), 1);
        roll(v1d, 1, axis: 0);
        roll(m2dStrided, 1, axis: 0);

        expect(identical(astype(v1d, DType.int32, copy: false), v1d), isTrue);
        expect(astype(v1d, DType.float64).dtype, DType.float64);
      });
    });
  });

  group('trigonometric branch coverage', () {
    test('all 13 unary trig functions across dtypes, strided, where, out', () {
      NDArray.scope(() {
        final f64 = NDArray.fromList([0.1, 0.2, 0.3, 0.4], [4], DType.float64);
        final f32 = NDArray.fromList([0.1, 0.2, 0.3, 0.4], [4], DType.float32);
        final i32 = NDArray.fromList([0, 1, 0, 1], [4], DType.int32);
        final c128 = NDArray.fromList(
          [
            const Complex(0.1, 0.2),
            const Complex(0.3, 0.4),
            const Complex(0.5, 0.6),
            const Complex(0.7, 0.8),
          ],
          [4],
          DType.complex128,
        );
        final c64 = NDArray.fromList(
          [
            const Complex(0.1, 0.2),
            const Complex(0.3, 0.4),
            const Complex(0.5, 0.6),
            const Complex(0.7, 0.8),
          ],
          [4],
          DType.complex64,
        );
        final mask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final dispMask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        )..dispose();
        final badOut = NDArray.zeros([3], DType.float64);

        final ops =
            <
              NDArray<DTypeTag> Function(
                NDArray<AnySpec>, {
                NDArray<DTypeTag>? where,
                NDArray<AnySpec>? out,
              })
            >[
              (a, {where, out}) => sin(a, where: where, out: out),
              (a, {where, out}) => sinc(a, where: where, out: out),
              (a, {where, out}) => cos(a, where: where, out: out),
              (a, {where, out}) => tan(a, where: where, out: out),
              (a, {where, out}) => asin(a, where: where, out: out),
              (a, {where, out}) => acos(a, where: where, out: out),
              (a, {where, out}) => atan(a, where: where, out: out),
              (a, {where, out}) => sinh(a, where: where, out: out),
              (a, {where, out}) => cosh(a, where: where, out: out),
              (a, {where, out}) => tanh(a, where: where, out: out),
              (a, {where, out}) => asinh(a, where: where, out: out),
              (a, {where, out}) => acosh(a, where: where, out: out),
              (a, {where, out}) => atanh(a, where: where, out: out),
            ];

        for (final op in ops) {
          // Contiguous across dtypes
          for (final arr in <NDArray<DTypeTag>>[f64, f32, i32, c128, c64]) {
            op(arr.asAnySpec);
            // Strided across dtypes
            final strided = arr.slice([Slice(start: 0, stop: 4, step: 2)]);
            op(strided.asAnySpec);
          }
          // Aliasing out with where
          final outAlias = f64.copy();
          op(outAlias.asAnySpec, where: mask, out: outAlias.asAnySpec);
          // Error paths: disposed where and bad out shape
          expect(() => op(f64.asAnySpec, where: dispMask), throwsStateError);
          expect(
            () => op(f64.asAnySpec, out: badOut.asAnySpec),
            throwsArgumentError,
          );
        }
      });
    });

    test('atan2, atan2As, hypot, hypotAs, deg2rad, rad2deg', () {
      NDArray.scope(() {
        final f64a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final f64b = NDArray.fromList([2.0, 3.0, 4.0, 5.0], [4], DType.float64);
        final f32a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
        final f32b = NDArray.fromList([2.0, 3.0, 4.0, 5.0], [4], DType.float32);
        final i32a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final i32b = NDArray.fromList([2, 3, 4, 5], [4], DType.int32);
        final u64a = NDArray.fromList([3, 5], [2], DType.uint64);
        final u64b = NDArray.fromList([4, 12], [2], DType.uint64);
        final bArr = NDArray.fromList([true, false], [2], DType.boolean);
        final c128 = NDArray.zeros([2], DType.complex128);
        final mask4 = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final dispMask = NDArray.fromList([true, false], [2], DType.boolean)
          ..dispose();

        // Contiguous and strided atan2 & hypot for float64, float32, int32
        atan2(f64a, f64b);
        atan2(f32a, f32b);
        atan2(i32a, i32b);
        atan2(
          f64a.slice([Slice(start: 0, stop: 4, step: 2)]),
          f64b.slice([Slice(start: 0, stop: 4, step: 2)]),
        );
        atan2(
          f32a.slice([Slice(start: 0, stop: 4, step: 2)]),
          f32b.slice([Slice(start: 0, stop: 4, step: 2)]),
        );
        final alias64 = f64a.copy();
        atan2(alias64, f64b, where: mask4, out: alias64);

        hypot(f64a, f64b);
        hypot(f32a, f32b);
        hypot(i32a, i32b);
        hypot(u64a, u64b);
        hypot(bArr, bArr);
        final infArr = NDArray.fromList(
          [double.infinity, 1.0, 2.0, 3.0],
          [4],
          DType.float64,
        );
        hypot(
          infArr.slice([Slice(start: 0, stop: 4, step: 2)]),
          f64b.slice([Slice(start: 0, stop: 4, step: 2)]),
        );
        hypot(
          f32a.slice([Slice(start: 0, stop: 4, step: 2)]),
          f32b.slice([Slice(start: 0, stop: 4, step: 2)]),
        );
        hypot(alias64, f64b, where: mask4, out: alias64);

        // atan2As & hypotAs
        final w2 = NDArray.fromList([true, false], [2], DType.boolean);
        final outF16 = NDArray.zeros([2], DType.float16);
        atan2As(u64a, u64b, DType.float32);
        atan2As(u64a, u64b, DType.float64);
        atan2As(u64a, u64b, DType.float16, where: w2, out: outF16);
        hypotAs(u64a, u64b, DType.float32);
        hypotAs(u64a, u64b, DType.float64);
        hypotAs(u64a, u64b, DType.float16, where: w2, out: outF16);

        // Errors in atan2, atan2As, hypot, hypotAs
        expect(() => atan2(f64a, f64b, where: dispMask), throwsStateError);
        expect(
          () => atan2(f64a.asAnySpec, f32a.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => atan2(c128.asAnySpec, c128.asAnySpec),
          throwsUnsupportedError,
        );
        expect(
          () => atan2(f64a, f64b, out: NDArray.zeros([2], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => atan2As(u64a, u64b, DType.float64, where: dispMask),
          throwsStateError,
        );
        expect(
          () => atan2As(c128, c128, DType.float64),
          throwsUnsupportedError,
        );
        expect(
          () => atan2As(
            u64a,
            u64b,
            DType.float64,
            out: NDArray.zeros([3], DType.float64),
          ),
          throwsArgumentError,
        );

        expect(() => hypot(f64a, f64b, where: dispMask), throwsStateError);
        expect(
          () => hypot(f64a.asAnySpec, f32a.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => hypot(f64a, f64b, out: NDArray.zeros([2], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => hypotAs(u64a, u64b, DType.float64, where: dispMask),
          throwsStateError,
        );
        expect(
          () => hypotAs(c128, c128, DType.float64),
          throwsUnsupportedError,
        );
        expect(
          () => hypotAs(
            u64a,
            u64b,
            DType.float64,
            out: NDArray.zeros([3], DType.float64),
          ),
          throwsArgumentError,
        );

        // deg2rad & rad2deg
        deg2rad(f64a);
        deg2rad(f32a);
        deg2rad(i32a);
        deg2rad(alias64, where: mask4, out: alias64);
        rad2deg(f64a);
        rad2deg(f32a);
        rad2deg(i32a);
        rad2deg(alias64, where: mask4, out: alias64);
        expect(
          () => deg2rad(f64a, out: NDArray.zeros([2], DType.float64)),
          throwsArgumentError,
        );
        expect(
          () => rad2deg(f64a, out: NDArray.zeros([2], DType.float64)),
          throwsArgumentError,
        );
      });
    });
  });

  group('ufunc_methods branch coverage', () {
    test(
      'binaryUfunc and minimum/maximum/fmin/fmax across all dtypes & where',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray.fromList([2.0, 1.0], [2], DType.float64);
          final dispMask = NDArray.fromList([true, false], [2], DType.boolean)
            ..dispose();

          // _withView when a is NDArray<AnySpec> and when fn(v) throws
          final outDynBool = NDArray.zeros([2], DType.boolean).asAnySpec;
          binaryUfunc(a, b, op: BinaryOp.equal, out: outDynBool);
          final outDynBoolWrongShape = NDArray.zeros([
            3,
          ], DType.boolean).asAnySpec;
          expect(
            () => binaryUfunc(
              a,
              b,
              op: BinaryOp.equal,
              out: outDynBoolWrongShape,
            ),
            throwsArgumentError,
          );
          final outDynInt = NDArray.zeros([2], DType.int32).asAnySpec;
          expect(
            () => binaryUfunc(a, b, op: BinaryOp.equal, out: outDynInt),
            throwsArgumentError,
          );

          // _coerceOwned when R is narrower than AnySpec
          expect(
            binaryUfunc<Float64, Float64>(a, b, op: BinaryOp.add).toList(),
            [3.0, 3.0],
          );

          // Comparison ops in binaryUfunc
          for (final op in [
            BinaryOp.notEqual,
            BinaryOp.greater,
            BinaryOp.greaterEqual,
            BinaryOp.less,
            BinaryOp.lessEqual,
          ]) {
            binaryUfunc(a, b, op: op);
          }
          expect(
            () => binaryUfunc(a, b, op: BinaryOp.add, where: dispMask),
            throwsStateError,
          );
          expect(
            () => binaryUfunc(
              a.asAnySpec,
              NDArray.zeros([2], DType.float32).asAnySpec,
              op: BinaryOp.add,
            ),
            throwsArgumentError,
          );
          expect(
            () => binaryUfunc(a, b, op: BinaryOp.gcd),
            throwsUnsupportedError,
          );
          expect(
            () => binaryUfunc(a, b, op: BinaryOp.lcm),
            throwsUnsupportedError,
          );

          // minimum, maximum, fmin, fmax across all 15 dtypes
          for (final dt in DType.values) {
            final list1 = dt.isComplex
                ? [const Complex(1, 2), const Complex(3, 1)]
                : (dt == DType.boolean ? [true, false] : [1, 3]);
            final list2 = dt.isComplex
                ? [const Complex(1, 3), const Complex(2, 5)]
                : (dt == DType.boolean ? [true, true] : [2, 2]);
            final x1 = NDArray.fromList(list1, [2], dt).asAnySpec;
            final x2 = NDArray.fromList(list2, [2], dt).asAnySpec;
            minimum(x1, x2);
            maximum(x1, x2);
            fmin(x1, x2);
            fmax(x1, x2);
          }

          // _elementwiseMinMax with where across boolean, uint64, complex128 (with NaN), float16 (with NaN)
          final mask2 = NDArray.fromList([true, false], [2], DType.boolean);
          final b1 = NDArray.fromList([true, false], [2], DType.boolean);
          final b2 = NDArray.fromList([false, true], [2], DType.boolean);
          minimum(b1, b2, where: mask2);
          maximum(b1, b2, where: mask2);
          final u64_1 = NDArray.fromList([1, 5], [2], DType.uint64);
          final u64_2 = NDArray.fromList([3, 2], [2], DType.uint64);
          minimum(u64_1, u64_2, where: mask2);
          final c128_1 = NDArray.fromList(
            [const Complex(double.nan, 0), const Complex(1, 2)],
            [2],
            DType.complex128,
          );
          final c128_2 = NDArray.fromList(
            [const Complex(1, 0), const Complex(1, 3)],
            [2],
            DType.complex128,
          );
          minimum(c128_1, c128_2, where: mask2);
          fmax(c128_1, c128_2, where: mask2);

          final f16a = NDArray.fromList(
            [double.nan, 1.0, double.nan, 4.0],
            [4],
            DType.float16,
          );
          final f16b = NDArray.fromList(
            [double.nan, double.nan, 3.0, 2.0],
            [4],
            DType.float16,
          );
          final maskI32 = NDArray.fromList([1, 1, 1, 0], [4], DType.int32);
          final outF16 = f16a.copy();
          minimum(f16a, f16b, where: maskI32, out: outF16);
          maximum(f16a, f16b, where: maskI32, out: outF16);
          fmin(f16a, f16b, where: maskI32, out: outF16);
          fmax(f16a, f16b, where: maskI32, out: outF16);

          // Broadcast & empty in _nativeMinMax, plus aliasing out
          final a2x1 = NDArray.fromList([1.0, 2.0], [2, 1], DType.float64);
          final b1x2 = NDArray.fromList([2.0, 1.0], [1, 2], DType.float64);
          expect(minimum(a2x1, b1x2).shape, [2, 2]);
          final emptyF64 = NDArray.zeros([0], DType.float64);
          expect(minimum(emptyF64, emptyF64).shape, [0]);
          final outAlias = a.copy();
          minimum(a, b, out: outAlias);

          // *As functions (both same dtype and cast dtype) and errors
          minimumAs(a, b, DType.float64);
          minimumAs(a, b, DType.float32);
          maximumAs(a, b, DType.float64);
          maximumAs(a, b, DType.float32);
          fminAs(a, b, DType.float64);
          fminAs(a, b, DType.float32);
          fmaxAs(a, b, DType.float64);
          fmaxAs(a, b, DType.float32);

          final f32 = NDArray.zeros([2], DType.float32);
          final badShape = NDArray.zeros([3], DType.float64);
          expect(() => minimum(a, b, where: dispMask), throwsStateError);
          expect(() => maximum(a, b, where: dispMask), throwsStateError);
          expect(() => fmin(a, b, where: dispMask), throwsStateError);
          expect(() => fmax(a, b, where: dispMask), throwsStateError);
          expect(
            () => minimumAs(a, b, DType.float64, where: dispMask),
            throwsStateError,
          );
          expect(
            () => maximumAs(a, b, DType.float64, where: dispMask),
            throwsStateError,
          );
          expect(
            () => fminAs(a, b, DType.float64, where: dispMask),
            throwsStateError,
          );
          expect(
            () => fmaxAs(a, b, DType.float64, where: dispMask),
            throwsStateError,
          );
          expect(
            () => minimum(a.asAnySpec, f32.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => maximum(a.asAnySpec, f32.asAnySpec),
            throwsArgumentError,
          );
          expect(() => fmin(a.asAnySpec, f32.asAnySpec), throwsArgumentError);
          expect(() => fmax(a.asAnySpec, f32.asAnySpec), throwsArgumentError);
          expect(
            () => minimum(a.asAnySpec, b.asAnySpec, out: f32.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => maximum(a.asAnySpec, b.asAnySpec, out: f32.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => fmin(a.asAnySpec, b.asAnySpec, out: f32.asAnySpec),
            throwsArgumentError,
          );
          expect(
            () => fmax(a.asAnySpec, b.asAnySpec, out: f32.asAnySpec),
            throwsArgumentError,
          );
          expect(() => minimum(a, b, out: badShape), throwsArgumentError);
          expect(
            () => minimum(a, b, where: mask2, out: badShape),
            throwsArgumentError,
          );
        });
      },
    );

    test('reduceUfunc, accumulateUfunc, reduceatUfunc, outerUfunc, atUfunc', () {
      NDArray.scope(() {
        final dtypes = <DType<DTypeTag>>[
          DType.float64,
          DType.float32,
          DType.int64,
          DType.int32,
          DType.int16,
          DType.uint8,
          DType.float16,
          DType.complex128,
          DType.complex64,
          DType.int8,
        ];
        for (final dt in dtypes) {
          final list = dt.isComplex
              ? [const Complex(1, 0), const Complex(2, 1), const Complex(3, 2)]
              : [1, 2, 3];
          final arr = NDArray.fromList(list, [3], dt).asAnySpec;
          for (final op in [
            BinaryOp.add,
            BinaryOp.multiply,
            BinaryOp.minimum,
            BinaryOp.maximum,
          ]) {
            reduceUfunc(arr, op: op);
            accumulateUfunc(arr, op: op);
          }
          if (!dt.isComplex) {
            for (final op in [BinaryOp.fmin, BinaryOp.fmax]) {
              reduceUfunc(arr, op: op);
              accumulateUfunc(arr, op: op);
            }
          }
          if (dt.isInteger) {
            for (final op in [
              BinaryOp.bitwiseAnd,
              BinaryOp.bitwiseOr,
              BinaryOp.bitwiseXor,
            ]) {
              reduceUfunc(arr, op: op);
              accumulateUfunc(arr, op: op);
            }
          }
        }

        // Boolean & float64 logical/logaddexp in reduceUfunc & accumulateUfunc
        final bArr = NDArray.fromList([true, false, true], [3], DType.boolean);
        final f64Arr = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        for (final op in [
          BinaryOp.bitwiseAnd,
          BinaryOp.bitwiseOr,
          BinaryOp.bitwiseXor,
        ]) {
          reduceUfunc(bArr, op: op);
          expect(() => reduceUfunc(f64Arr, op: op), throwsArgumentError);
          accumulateUfunc(bArr, op: op);
          expect(() => accumulateUfunc(f64Arr, op: op), throwsArgumentError);
        }
        for (final op in [
          BinaryOp.logicalAnd,
          BinaryOp.logicalOr,
          BinaryOp.logicalXor,
        ]) {
          reduceUfunc(bArr, op: op);
          reduceUfunc(f64Arr, op: op);
          accumulateUfunc(bArr, op: op);
          accumulateUfunc(f64Arr, op: op);
        }
        for (final op in [BinaryOp.logaddexp, BinaryOp.logaddexp2]) {
          reduceUfunc(f64Arr, op: op);
          accumulateUfunc(f64Arr, op: op);
        }

        // reduceUfunc & accumulateUfunc out aliasing, empty, errors
        expect(
          () => reduceUfunc(bArr, op: BinaryOp.subtract),
          throwsUnsupportedError,
        );
        expect(
          () => accumulateUfunc(bArr, op: BinaryOp.subtract),
          throwsUnsupportedError,
        );
        expect(
          () => reduceUfunc(
            f64Arr,
            op: BinaryOp.add,
            out: NDArray.zeros([2], DType.float64),
          ),
          throwsArgumentError,
        );
        expect(
          () => accumulateUfunc(
            f64Arr,
            op: BinaryOp.add,
            out: NDArray.zeros([2], DType.float64),
          ),
          throwsArgumentError,
        );
        final f64_2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final redAlias = f64_2d.slice([const Index(0), Slice.all()]);
        reduceUfunc(f64_2d, op: BinaryOp.add, axis: 0, out: redAlias);
        final accAlias = f64Arr.copy();
        accumulateUfunc(accAlias, op: BinaryOp.add, out: accAlias);
        final empty2d = NDArray.zeros([2, 0], DType.float64);
        expect(reduceUfunc(empty2d, op: BinaryOp.add, axis: 0).shape, [0]);
        expect(accumulateUfunc(empty2d, op: BinaryOp.add, axis: 0).shape, [
          2,
          0,
        ]);

        // reduceatUfunc across 1D and 2D for all native dtypes + negative indices + errors
        final idx1d = NDArray.fromList([0, -1], [2], DType.int64);
        final idx1dI32 = NDArray.fromList([0, -1], [2], DType.int32);
        reduceatUfunc(f64Arr, idx1d, op: BinaryOp.add);
        reduceatUfunc(f64Arr, idx1dI32, op: BinaryOp.add);
        for (final dt in [
          DType.float64,
          DType.float32,
          DType.int64,
          DType.int32,
          DType.int16,
          DType.uint8,
          DType.boolean,
          DType.complex128,
          DType.complex64,
          DType.float16,
          DType.int8,
        ]) {
          final list = dt.isComplex
              ? [
                  const Complex(1, 0),
                  const Complex(2, 1),
                  const Complex(3, 0),
                  const Complex(4, 1),
                ]
              : (dt == DType.boolean
                    ? [true, false, true, false]
                    : [1, 2, 3, 4]);
          final a1 = NDArray.fromList(list, [4], dt).asAnySpec;
          final a2 = a1.reshape([2, 2]);
          final op = dt == DType.boolean ? BinaryOp.bitwiseXor : BinaryOp.add;
          reduceatUfunc(a1, NDArray.fromList([0, 2], [2], DType.int64), op: op);
          reduceatUfunc(
            a2,
            NDArray.fromList([0, 1], [2], DType.int64),
            op: op,
            axis: 0,
          );
        }
        // int8 fallback with start < end (j = start + 1; j < end)
        final aI8_4 = NDArray.fromList([1, 2, 3, 4], [4], DType.int8);
        reduceatUfunc(
          aI8_4,
          NDArray.fromList([0, 3], [2], DType.int64),
          op: BinaryOp.minimum,
        );
        final ratAlias = f64Arr.slice([Slice(start: 0, stop: 2)]);
        reduceatUfunc(f64Arr, idx1d, op: BinaryOp.add, out: ratAlias);
        expect(
          () => reduceatUfunc(bArr, idx1d, op: BinaryOp.subtract),
          throwsUnsupportedError,
        );
        expect(
          () => reduceatUfunc(
            f64Arr,
            idx1d,
            op: BinaryOp.add,
            out: NDArray.zeros([5], DType.float64),
          ),
          throwsArgumentError,
        );
        expect(
          () => reduceatUfunc(
            NDArray.zeros([0], DType.float64),
            idx1d,
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );
        expect(
          () => reduceatUfunc(
            f64Arr,
            NDArray.fromList([10], [1], DType.int64),
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );
        expect(
          () => reduceatUfunc(
            f64Arr,
            NDArray.fromList([10], [1], DType.int32),
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );

        // atUfunc across dtypes, empty indices, negative indices, aliasing, division error
        for (final dt in [
          DType.float64,
          DType.float32,
          DType.int64,
          DType.int32,
          DType.int16,
          DType.uint8,
          DType.complex128,
          DType.complex64,
          DType.boolean,
          DType.float16,
          DType.int8,
        ]) {
          final list = dt.isComplex
              ? [const Complex(1, 0), const Complex(2, 1), const Complex(3, 0)]
              : (dt == DType.boolean ? [true, false, true] : [10, 20, 30]);
          final bList = dt.isComplex
              ? [const Complex(1, 1), const Complex(2, 0)]
              : (dt == DType.boolean ? [false, true] : [2, 5]);
          final aAt = NDArray.fromList(list, [3], dt).asAnySpec;
          final bAt = NDArray.fromList(bList, [2], dt).asAnySpec;
          final op = dt == DType.boolean ? BinaryOp.bitwiseXor : BinaryOp.add;
          atUfunc(aAt, idx1d, bAt, op: op);
        }
        atUfunc(
          f64Arr,
          idx1dI32,
          NDArray.fromList([1.0, 2.0], [2], DType.float64),
          op: BinaryOp.add,
        );
        atUfunc(
          f64Arr,
          NDArray.zeros([0], DType.int64),
          NDArray.zeros([0], DType.float64),
          op: BinaryOp.add,
        );
        atUfunc(
          f64Arr,
          idx1d,
          f64Arr.slice([Slice(start: 0, stop: 2)]),
          op: BinaryOp.add,
        );

        final i32At = NDArray.fromList([10, 20], [2], DType.int32);
        expect(
          () => atUfunc(
            i32At,
            NDArray.fromList([0], [1], DType.int64),
            NDArray.fromList([0], [1], DType.int32),
            op: BinaryOp.floorDivide,
          ),
          throwsUnsupportedError,
        );
        expect(
          () => atUfunc(
            NDArray.fromList([1.0], [], DType.float64),
            idx1d,
            f64Arr,
            op: BinaryOp.add,
          ),
          throwsArgumentError,
        );
        expect(
          () => atUfunc(bArr, idx1d, bArr, op: BinaryOp.subtract),
          throwsUnsupportedError,
        );
        expect(
          () => atUfunc(f64Arr, idx1d, f64Arr, op: BinaryOp.bitwiseAnd),
          throwsUnsupportedError,
        );
        expect(
          () => atUfunc(
            NDArray.zeros([0], DType.float64),
            idx1d,
            f64Arr,
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );
        expect(
          () => atUfunc(
            f64Arr,
            NDArray.fromList([10], [1], DType.int64),
            f64Arr,
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );
        expect(
          () => atUfunc(
            f64Arr,
            NDArray.fromList([10], [1], DType.int32),
            f64Arr,
            op: BinaryOp.add,
          ),
          throwsRangeError,
        );
      });
    });

    test('unaryUfunc across all UnaryOps', () {
      NDArray.scope(() {
        final f64 = NDArray.fromList([0.2, 0.5], [2], DType.float64);
        final f32 = NDArray.fromList(
          [0.0, 0.5, double.nan, double.infinity],
          [4],
          DType.float32,
        );
        final i32 = NDArray.fromList([-2, 3], [2], DType.int32);
        final u8 = NDArray.fromList([1, 2], [2], DType.uint8);
        final c128 = NDArray.fromList(
          [const Complex(1, 2), const Complex(3, 4)],
          [2],
          DType.complex128,
        );
        final mask4 = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final dispMask = NDArray.fromList([true, false], [2], DType.boolean)
          ..dispose();

        for (final op in UnaryOp.values) {
          if (op == UnaryOp.invert || op == UnaryOp.bitwiseNot) {
            unaryUfunc(i32, op: op);
          } else {
            unaryUfunc(f64, op: op);
          }
        }
        // signbit on uint8, int32, complex (error)
        unaryUfunc(u8, op: UnaryOp.signbit);
        unaryUfunc(i32, op: UnaryOp.signbit);
        expect(
          () => unaryUfunc(c128, op: UnaryOp.signbit),
          throwsUnsupportedError,
        );
        expect(
          () => unaryUfunc(c128, op: UnaryOp.cbrt),
          throwsUnsupportedError,
        );
        expect(
          () => unaryUfunc(c128, op: UnaryOp.spacing),
          throwsUnsupportedError,
        );
        expect(
          () => unaryUfunc(f64, op: UnaryOp.sin, where: dispMask),
          throwsStateError,
        );
        expect(
          () => unaryUfunc(
            f64,
            op: UnaryOp.spacing,
            out: NDArray.zeros([3], DType.float64).asAnySpec,
          ),
          throwsArgumentError,
        );
        final outSpF32 = NDArray.zeros([4], DType.float32);
        unaryUfunc(
          f32,
          op: UnaryOp.spacing,
          where: mask4,
          out: outSpF32.asAnySpec,
        );

        // Additional 2D reduceUfunc, binaryUfunc ops, outerUfunc, extension methods
        for (final dt in [
          DType.float64,
          DType.float32,
          DType.int64,
          DType.int32,
          DType.int16,
          DType.uint8,
        ]) {
          final a2d = NDArray.fromList([1, 2, 3, 4], [2, 2], dt).asAnySpec;
          for (final op in [
            BinaryOp.add,
            BinaryOp.multiply,
            BinaryOp.minimum,
            BinaryOp.maximum,
            BinaryOp.fmin,
            BinaryOp.fmax,
          ]) {
            reduceUfunc(a2d, op: op, axis: 0);
          }
          if (dt.isInteger) {
            for (final op in [
              BinaryOp.bitwiseAnd,
              BinaryOp.bitwiseOr,
              BinaryOp.bitwiseXor,
            ]) {
              reduceUfunc(a2d, op: op, axis: 0);
            }
          }
        }
        final b2d = NDArray.fromList(
          [true, false, true, true],
          [2, 2],
          DType.boolean,
        );
        for (final op in [
          BinaryOp.logicalAnd,
          BinaryOp.logicalOr,
          BinaryOp.logicalXor,
        ]) {
          reduceUfunc(b2d, op: op, axis: 0);
        }
        reduceUfunc(
          NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int8),
          op: BinaryOp.add,
          axis: 0,
          keepdims: true,
          initial: 5,
        );
        reduceUfunc(
          NDArray.zeros([0, 2], DType.float64),
          op: BinaryOp.add,
          axis: 0,
          initial: 0.0,
        );
        expect(
          () =>
              reduceUfunc(NDArray.zeros([0], DType.float64), op: BinaryOp.add),
          throwsArgumentError,
        );

        // Remaining BinaryOps in binaryUfunc
        for (final op in [
          BinaryOp.subtract,
          BinaryOp.multiply,
          BinaryOp.divide,
          BinaryOp.floorDivide,
          BinaryOp.remainder,
          BinaryOp.fmod,
          BinaryOp.heaviside,
          BinaryOp.power,
          BinaryOp.floatPower,
          BinaryOp.arctan2,
          BinaryOp.hypot,
          BinaryOp.copysign,
        ]) {
          binaryUfunc(f64, f64, op: op);
        }
        for (final op in [
          BinaryOp.gcd,
          BinaryOp.lcm,
          BinaryOp.leftShift,
          BinaryOp.rightShift,
        ]) {
          binaryUfunc(i32, i32, op: op);
        }

        // Extension methods and outerUfunc + atUfunc int8 divide/minimum
        f64.reduce(op: BinaryOp.add);
        f64.accumulate(op: BinaryOp.add);
        f64.reduceat(
          NDArray.fromList([0, 1], [2], DType.int64),
          op: BinaryOp.add,
        );
        f64.at(
          NDArray.fromList([0], [1], DType.int64),
          NDArray.fromList([1.0], [1], DType.float64),
          op: BinaryOp.add,
        );
        final outOuter = NDArray.zeros([2, 2], DType.float64);
        outerUfunc(
          outOuter.slice([const Index(0), Slice.all()]),
          f64,
          op: BinaryOp.add,
          where: b2d,
          out: outOuter.asAnySpec,
        );
        final aAtI8 = NDArray.fromList([10, 20], [2], DType.int8);
        final idxAt1 = NDArray.fromList([0], [1], DType.int64);
        atUfunc(
          aAtI8,
          idxAt1,
          NDArray.fromList([2], [1], DType.int8),
          op: BinaryOp.divide,
        );
        atUfunc(
          aAtI8,
          idxAt1,
          NDArray.fromList([3], [1], DType.int8),
          op: BinaryOp.minimum,
        );

        // Additional fft & manipulation coverage
        final bigC128 = NDArray.zeros([256], DType.complex128);
        ifft(bigC128);
        final a3dC128 = NDArray.zeros([2, 2, 2], DType.complex128);
        fft(a3dC128, axis: 0, out: NDArray.zeros([2, 2, 2], DType.complex128));
        ifft(a3dC128, axis: 0, out: NDArray.zeros([2, 2, 2], DType.complex128));
        fftn(a3dC128, s: [1, 3, 2]);
        fftshift(a3dC128, out: NDArray.zeros([2, 2, 2], DType.complex128));
        ifftshift(a3dC128, out: NDArray.zeros([2, 2, 2], DType.complex128));
        expect(() => fftfreq(0), throwsArgumentError);
        expect(() => fftfreq(4, d: 0.0), throwsArgumentError);
        expect(() => rfftfreq(0), throwsArgumentError);
        expect(() => rfftfreq(4, d: 0.0), throwsArgumentError);

        squeeze(NDArray.zeros([1, 2, 1], DType.int32));
        expect(
          () => squeeze(NDArray.zeros([2, 2], DType.int32), axis: [0]),
          throwsArgumentError,
        );
        slidingWindowView(NDArray.zeros([3, 3], DType.int32), [2, 2]);
        expect(
          () => slidingWindowView(NDArray.zeros([3, 3], DType.int32), [2]),
          throwsArgumentError,
        );
        expect(
          () => slidingWindowView(NDArray.zeros([3, 3], DType.int32), [0, 2]),
          throwsArgumentError,
        );
        expect(
          () => slidingWindowView(NDArray.zeros([3, 3], DType.int32), [5, 2]),
          throwsArgumentError,
        );
        flip(NDArray.zeros([2, 2], DType.int32));
        flip(NDArray.zeros([2, 2], DType.int32), axis: [0, 1]);
        final rollAlias = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        roll(rollAlias, 1, axis: [0, 1], out: rollAlias);

        // Additional validation branches in fft.dart and manipulation.dart
        final dispC = NDArray.zeros([4], DType.complex128)..dispose();
        final dispF = NDArray.zeros([4], DType.float64)..dispose();
        final scalarF = NDArray.fromList([1.0], [], DType.float64);
        final a4F = NDArray.zeros([4], DType.float64);
        final a4C = NDArray.zeros([4], DType.complex128);
        final a4C64 = NDArray.zeros([4], DType.complex64);

        expect(() => fft(dispC), throwsStateError);
        expect(() => fft(a4C, out: dispC), throwsStateError);
        expect(() => fft(scalarF), throwsArgumentError);
        expect(() => fft(a4C, axis: 5), throwsRangeError);
        expect(() => fft(a4C, n: 0), throwsArgumentError);
        expect(
          () => fft(a4C.asAnySpec, out: a4C64.asAnySpec),
          throwsArgumentError,
        );
        expect(
          () => fft(a4C, out: NDArray.zeros([5], DType.complex128)),
          throwsArgumentError,
        );

        expect(() => ifft(dispC), throwsStateError);
        expect(() => ifft(a4C, out: dispC), throwsStateError);
        expect(() => ifft(scalarF), throwsArgumentError);
        expect(() => ifft(a4C, axis: 5), throwsRangeError);
        expect(() => ifft(a4C, n: 0), throwsArgumentError);
        expect(
          () => ifft(a4C.asAnySpec, out: a4C64.asAnySpec),
          throwsArgumentError,
        );

        expect(() => rfft(dispF), throwsStateError);
        expect(() => rfft(a4F, out: dispC), throwsStateError);
        expect(() => rfft(scalarF), throwsArgumentError);
        expect(() => rfft(a4F, axis: 5), throwsRangeError);
        expect(() => rfft(a4F, n: 0), throwsArgumentError);
        expect(
          () => rfft(a4F.asAnySpec, out: a4C64.asAnySpec),
          throwsArgumentError,
        );

        expect(() => irfft(dispC), throwsStateError);
        expect(() => irfft(a4C, out: dispF), throwsStateError);
        expect(() => irfft(scalarF), throwsArgumentError);
        expect(() => irfft(a4C, axis: 5), throwsRangeError);
        expect(() => irfft(a4C, n: 0), throwsArgumentError);
        expect(
          () => irfft(
            a4C.asAnySpec,
            out: NDArray.zeros([6], DType.float32).asAnySpec,
          ),
          throwsArgumentError,
        );

        expect(() => fftshift(dispC), throwsStateError);
        expect(() => ifftshift(dispC), throwsStateError);
        expect(fftshift(scalarF).shape, <int>[]);
        expect(ifftshift(scalarF).shape, <int>[]);
        expect(fftshift(a4C, axes: <int>[]).shape, [4]);
        expect(ifftshift(a4C, axes: <int>[]).shape, [4]);
        expect(() => fftshift(a4C, axes: [0, 0]), throwsArgumentError);
        expect(() => ifftshift(a4C, axes: [0, 0]), throwsArgumentError);

        expect(() => concatenate([a4F], axis: 5), throwsRangeError);
        expect(
          () => concatenate([a4F.asAnySpec, a4C.asAnySpec]),
          throwsArgumentError,
        );
        expect(
          () => concatenate([
            a4F,
            NDArray.zeros([2, 2], DType.float64),
          ]),
          throwsArgumentError,
        );
        expect(
          () => concatenate([
            NDArray.zeros([2, 2], DType.float64),
            NDArray.zeros([2, 3], DType.float64),
          ], axis: 0),
          throwsArgumentError,
        );

        expect(() => stack(<NDArray<Float64>>[]), throwsArgumentError);
        expect(() => stack([dispF]), throwsStateError);
        expect(
          () => stack([a4F.asAnySpec, a4C.asAnySpec]),
          throwsArgumentError,
        );
        expect(
          () => stack([
            a4F,
            NDArray.zeros([5], DType.float64),
          ]),
          throwsArgumentError,
        );
        expect(() => stack([a4F], axis: 5), throwsRangeError);
        final a4FStrided = a4F.slice([Slice(start: 0, stop: 4, step: 2)]);
        stack([a4FStrided, a4FStrided], axis: 0);
        stack([a4FStrided, a4FStrided], axis: 1);

        expect(() => expand_dims(a4F, 5), throwsRangeError);
        expect(() => fliplr(a4F), throwsArgumentError);
        expect(() => flipud(scalarF), throwsArgumentError);
        expect(() => rot90(a4F), throwsArgumentError);
        expect(
          () => rot90(NDArray.zeros([2, 2], DType.float64), 1, [0, 5]),
          throwsRangeError,
        );

        expect(() => roll(a4F, 'bad'), throwsArgumentError);
        expect(() => roll(a4F, 1, axis: 'bad'), throwsArgumentError);
        expect(() => roll(a4F, [1, 2], axis: [0]), throwsArgumentError);
        expect(() => roll(a4F, 1, axis: 5), throwsRangeError);
        expect(
          () => roll(a4F.asAnySpec, 1, out: a4C.asAnySpec),
          throwsArgumentError,
        );
      });
    });
  });
}
