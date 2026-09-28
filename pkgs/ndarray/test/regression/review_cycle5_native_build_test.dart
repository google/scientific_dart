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

import 'dart:ffi' as ffi;
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_bindings.dart' as bindings;
import 'package:ndarray/src/ndarray_extensions_bindings.dart' as ext_bindings;
import 'package:test/test.dart';

void main() {
  group('Review Cycle 5 Native & Build Remediation Tests', () {
    test(
      'Complex64 and Complex128 division extreme magnitudes across lengths 1, 8, 9, 17',
      () {
        for (final length in [1, 8, 9, 17]) {
          NDArray.scope(() {
            final num64 = NDArray<Complex64>.fromList(
              List.filled(length, Complex(1.0, 0.0)),
              [length],
              DType.complex64,
            );
            final den64 = NDArray<Complex64>.fromList(
              List.filled(length, Complex(1e-25, 0.0)),
              [length],
              DType.complex64,
            );
            final res64 = divide<Complex64, Complex64, Complex64>(num64, den64);
            for (var i = 0; i < length; i++) {
              final val = res64.getCell([i]);
              expect(val.real.isNaN, isFalse, reason: 'len=$length idx=$i');
              expect(val.imag.isNaN, isFalse, reason: 'len=$length idx=$i');
              expect(
                val.real,
                closeTo(1e25, 1e18),
                reason: 'len=$length idx=$i',
              );
              expect(
                val.imag,
                closeTo(0.0, 1e-5),
                reason: 'len=$length idx=$i',
              );
            }

            // Also test direct native v_div_complex64 call
            final nativeRes64 = NDArray<Complex64>.create([
              length,
            ], DType.complex64);
            bindings.v_div_complex64(
              num64.pointer.cast(),
              den64.pointer.cast(),
              nativeRes64.pointer.cast(),
              length,
              ffi.nullptr,
            );
            for (var i = 0; i < length; i++) {
              final val = nativeRes64.getCell([i]);
              expect(
                val.real.isNaN,
                isFalse,
                reason: 'native len=$length i=$i',
              );
              expect(val.real, closeTo(1e25, 1e18));
            }

            final num128 = NDArray<Complex128>.fromList(
              List.filled(length, Complex(1.0, 0.0)),
              [length],
              DType.complex128,
            );
            final den128 = NDArray<Complex128>.fromList(
              List.filled(length, Complex(1e-200, 0.0)),
              [length],
              DType.complex128,
            );
            final res128 = divide<Complex128, Complex128, Complex128>(
              num128,
              den128,
            );
            for (var i = 0; i < length; i++) {
              final val = res128.getCell([i]);
              expect(val.real.isNaN, isFalse, reason: 'len=$length idx=$i');
              expect(val.imag.isNaN, isFalse, reason: 'len=$length idx=$i');
              expect(
                val.real,
                closeTo(1e200, 1e186),
                reason: 'len=$length idx=$i',
              );
              expect(
                val.imag,
                closeTo(0.0, 1e-12),
                reason: 'len=$length idx=$i',
              );
            }

            // Also test direct native v_div_complex call
            final nativeRes128 = NDArray<Complex128>.create([
              length,
            ], DType.complex128);
            bindings.v_div_complex(
              num128.pointer.cast(),
              den128.pointer.cast(),
              nativeRes128.pointer.cast(),
              length,
              ffi.nullptr,
            );
            for (var i = 0; i < length; i++) {
              final val = nativeRes128.getCell([i]);
              expect(
                val.real.isNaN,
                isFalse,
                reason: 'native len=$length i=$i',
              );
              expect(val.real, closeTo(1e200, 1e186));
            }
          });
        }
      },
    );

    test(
      'Contiguous overlapping slices for add/div on Complex128/Complex64',
      () {
        NDArray.scope(() {
          // Complex128 add native overlap
          final base128 = NDArray<Complex128>.fromList(
            List.generate(
              17,
              (i) => Complex((i + 1).toDouble(), (i + 1) * 2.0),
            ),
            [17],
            DType.complex128,
          );
          final a128 = base128.slice([const Slice(start: 0, stop: 16)]);
          final out128 = base128.slice([const Slice(start: 1, stop: 17)]);
          final expectedAdd128 = List.generate(
            16,
            (i) => Complex((i + 1) * 2.0, (i + 1) * 4.0),
          );

          bindings.v_add_complex(
            a128.pointer.cast(),
            a128.pointer.cast(),
            out128.pointer.cast(),
            16,
            ffi.nullptr,
          );
          for (var i = 0; i < 16; i++) {
            expect(
              out128.getCell([i]).real,
              closeTo(expectedAdd128[i].real, 1e-12),
            );
            expect(
              out128.getCell([i]).imag,
              closeTo(expectedAdd128[i].imag, 1e-12),
            );
          }

          // Complex128 div native overlap
          final baseDiv128 = NDArray<Complex128>.fromList(
            List.generate(17, (i) => Complex((i + 2).toDouble(), 0.0)),
            [17],
            DType.complex128,
          );
          final aDiv128 = baseDiv128.slice([const Slice(start: 0, stop: 16)]);
          final outDiv128 = baseDiv128.slice([const Slice(start: 1, stop: 17)]);
          final constTwo128 = NDArray<Complex128>.fromList(
            List.filled(16, Complex(2.0, 0.0)),
            [16],
            DType.complex128,
          );
          bindings.v_div_complex(
            aDiv128.pointer.cast(),
            constTwo128.pointer.cast(),
            outDiv128.pointer.cast(),
            16,
            ffi.nullptr,
          );
          for (var i = 0; i < 16; i++) {
            expect(outDiv128.getCell([i]).real, closeTo((i + 2) / 2.0, 1e-12));
          }

          // Complex64 add & div native overlap
          final base64 = NDArray<Complex64>.fromList(
            List.generate(
              17,
              (i) => Complex((i + 1).toDouble(), (i + 1).toDouble()),
            ),
            [17],
            DType.complex64,
          );
          final a64 = base64.slice([const Slice(start: 0, stop: 16)]);
          final out64 = base64.slice([const Slice(start: 1, stop: 17)]);
          bindings.v_add_complex64(
            a64.pointer.cast(),
            a64.pointer.cast(),
            out64.pointer.cast(),
            16,
            ffi.nullptr,
          );
          for (var i = 0; i < 16; i++) {
            expect(out64.getCell([i]).real, closeTo((i + 1) * 2.0, 1e-5));
          }
        });
      },
    );

    test(
      'Contiguous overlapping slices for sin/atan2 on Float32 and clip on Float64/Float32',
      () {
        NDArray.scope(() {
          // Float32 sin native overlap
          final baseSin = NDArray<Float32>.fromList(
            List.generate(17, (i) => (i + 1) * 0.1),
            [17],
            DType.float32,
          );
          final aSin = baseSin.slice([const Slice(start: 0, stop: 16)]);
          final outSin = baseSin.slice([const Slice(start: 1, stop: 17)]);
          final expectedSin = List.generate(16, (i) => math.sin((i + 1) * 0.1));
          bindings.v_sin_float(
            aSin.pointer.cast(),
            outSin.pointer.cast(),
            16,
            ffi.nullptr,
          );
          for (var i = 0; i < 16; i++) {
            expect(outSin.getCell([i]), closeTo(expectedSin[i], 1e-5));
          }

          // Float32 atan2 native overlap
          final baseAtan2 = NDArray<Float32>.fromList(
            List.generate(17, (i) => (i + 1).toDouble()),
            [17],
            DType.float32,
          );
          final xAtan2 = NDArray<Float32>.fromList(List.filled(16, 2.0), [
            16,
          ], DType.float32);
          final ySlice = baseAtan2.slice([const Slice(start: 0, stop: 16)]);
          final outAtan2 = baseAtan2.slice([const Slice(start: 1, stop: 17)]);
          final expectedAtan2 = List.generate(
            16,
            (i) => math.atan2((i + 1).toDouble(), 2.0),
          );
          bindings.v_atan2_float(
            ySlice.pointer.cast(),
            xAtan2.pointer.cast(),
            outAtan2.pointer.cast(),
            16,
            ffi.nullptr,
          );
          for (var i = 0; i < 16; i++) {
            expect(outAtan2.getCell([i]), closeTo(expectedAtan2[i], 1e-5));
          }

          // Float64 clip native overlap
          final baseClip64 = NDArray<Float64>.fromList(
            List.generate(17, (i) => i.toDouble()),
            [17],
            DType.float64,
          );
          final srcClip64 = baseClip64.slice([const Slice(start: 0, stop: 16)]);
          final outClip64 = baseClip64.slice([const Slice(start: 1, stop: 17)]);
          bindings.v_clip_double(
            srcClip64.pointer.cast(),
            outClip64.pointer.cast(),
            3.0,
            12.0,
            16,
            ffi.nullptr,
          );
          for (var i = 0; i < 16; i++) {
            final expected = i.toDouble().clamp(3.0, 12.0);
            expect(outClip64.getCell([i]), closeTo(expected, 1e-12));
          }

          // Float32 clip native overlap
          final baseClip32 = NDArray<Float32>.fromList(
            List.generate(17, (i) => i.toDouble()),
            [17],
            DType.float32,
          );
          final srcClip32 = baseClip32.slice([const Slice(start: 0, stop: 16)]);
          final outClip32 = baseClip32.slice([const Slice(start: 1, stop: 17)]);
          bindings.v_clip_float(
            srcClip32.pointer.cast(),
            outClip32.pointer.cast(),
            3.0,
            12.0,
            16,
            ffi.nullptr,
          );
          for (var i = 0; i < 16; i++) {
            final expected = i.toDouble().clamp(3.0, 12.0);
            expect(outClip32.getCell([i]), closeTo(expected, 1e-5));
          }
        });
      },
    );

    test(
      'In-place roll on non-contiguous strided slice and overlapping contiguous slices',
      () {
        NDArray.scope(() {
          // Non-contiguous strided slice: v = base.slice([Slice(start: 0, stop: 6, step: 2)]) -> [10, 30, 50]
          final base = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [6],
            DType.float64,
          );
          final v = base.slice([const Slice(start: 0, stop: 6, step: 2)]);
          roll<Float64>(v, 1, axis: 0, out: v);
          // Expected rolled v: [50.0, 10.0, 30.0]
          expect(v.getCell([0]), closeTo(50.0, 1e-12));
          expect(v.getCell([1]), closeTo(10.0, 1e-12));
          expect(v.getCell([2]), closeTo(30.0, 1e-12));

          // Also test direct native_roll_nd on strided view in-place
          final baseNative = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [6],
            DType.float64,
          );
          final vNative = baseNative.slice([
            const Slice(start: 0, stop: 6, step: 2),
          ]);
          final marker = ScratchArena.marker;
          try {
            final cShape = ScratchArena.allocate<ffi.Int64>(
              ffi.sizeOf<ffi.Int64>(),
            );
            final cStrides = ScratchArena.allocate<ffi.Int64>(
              ffi.sizeOf<ffi.Int64>(),
            );
            cShape[0] = 3;
            cStrides[0] = vNative.strides[0];
            ext_bindings.native_roll_nd(
              DType.float64.index,
              vNative.pointer.cast(),
              cShape,
              cStrides,
              1,
              1,
              0,
              vNative.pointer.cast(),
              cStrides,
            );
          } finally {
            ScratchArena.reset(marker);
          }
          expect(vNative.getCell([0]), closeTo(50.0, 1e-12));
          expect(vNative.getCell([1]), closeTo(10.0, 1e-12));
          expect(vNative.getCell([2]), closeTo(30.0, 1e-12));

          // Overlapping contiguous slices: roll(base.slice([Slice(start: 0, stop: 5)]), 2, out: base.slice([Slice(start: 1, stop: 6)]))
          final baseOverlap = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );
          final srcSlice = baseOverlap.slice([const Slice(start: 0, stop: 5)]);
          final dstSlice = baseOverlap.slice([const Slice(start: 1, stop: 6)]);
          roll<Float64>(srcSlice, 2, out: dstSlice);
          // srcSlice was [1, 2, 3, 4, 5], rolled by 2 is [4, 5, 1, 2, 3]
          expect(dstSlice.getCell([0]), closeTo(4.0, 1e-12));
          expect(dstSlice.getCell([1]), closeTo(5.0, 1e-12));
          expect(dstSlice.getCell([2]), closeTo(1.0, 1e-12));
          expect(dstSlice.getCell([3]), closeTo(2.0, 1e-12));
          expect(dstSlice.getCell([4]), closeTo(3.0, 1e-12));

          // Direct native_roll_1d overlapping contiguous call
          final baseOverlapNative = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
            DType.float64,
          );
          final srcNative = baseOverlapNative.slice([
            const Slice(start: 0, stop: 5),
          ]);
          final dstNative = baseOverlapNative.slice([
            const Slice(start: 1, stop: 6),
          ]);
          ext_bindings.native_roll_1d(
            DType.float64.index,
            srcNative.pointer.cast(),
            5,
            2,
            dstNative.pointer.cast(),
          );
          expect(dstNative.getCell([0]), closeTo(4.0, 1e-12));
          expect(dstNative.getCell([1]), closeTo(5.0, 1e-12));
          expect(dstNative.getCell([2]), closeTo(1.0, 1e-12));
          expect(dstNative.getCell([3]), closeTo(2.0, 1e-12));
          expect(dstNative.getCell([4]), closeTo(3.0, 1e-12));
        });
      },
    );

    test(
      'Strided clipArray(a, min, max, out: a.transpose()) on a square matrix',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [
              1.0, 5.0, 9.0, //
              2.0, 6.0, 10.0, //
              3.0, 7.0, 11.0,
            ],
            [3, 3],
            DType.float64,
          );
          final minArr = NDArray<Float64>.fromList([3.0], [1], DType.float64);
          final maxArr = NDArray<Float64>.fromList([8.0], [1], DType.float64);

          final expected = [
            [3.0, 3.0, 3.0],
            [5.0, 6.0, 7.0],
            [8.0, 8.0, 8.0],
          ];

          clipArray<Float64>(a, min: minArr, max: maxArr, out: a.transpose());

          for (var r = 0; r < 3; r++) {
            for (var c = 0; c < 3; c++) {
              expect(
                a.getCell([r, c]),
                closeTo(expected[r][c], 1e-12),
                reason: 'high-level clipArray at ($r, $c)',
              );
            }
          }

          // Also test direct native s_clip_double with out = a.transpose()
          final aNative = NDArray<Float64>.fromList(
            [
              1.0, 5.0, 9.0, //
              2.0, 6.0, 10.0, //
              3.0, 7.0, 11.0,
            ],
            [3, 3],
            DType.float64,
          );
          final aTrans = aNative.transpose();
          final marker = ScratchArena.marker;
          try {
            final cShape = ScratchArena.allocate<ffi.Int>(
              2 * ffi.sizeOf<ffi.Int>(),
            );
            final cStridesA = ScratchArena.allocate<ffi.Int>(
              2 * ffi.sizeOf<ffi.Int>(),
            );
            final cStridesMin = ScratchArena.allocate<ffi.Int>(
              2 * ffi.sizeOf<ffi.Int>(),
            );
            final cStridesMax = ScratchArena.allocate<ffi.Int>(
              2 * ffi.sizeOf<ffi.Int>(),
            );
            final cStridesRes = ScratchArena.allocate<ffi.Int>(
              2 * ffi.sizeOf<ffi.Int>(),
            );
            cShape[0] = 3;
            cShape[1] = 3;
            cStridesA[0] = aNative.strides[0];
            cStridesA[1] = aNative.strides[1];
            cStridesMin[0] = 0;
            cStridesMin[1] = 0;
            cStridesMax[0] = 0;
            cStridesMax[1] = 0;
            cStridesRes[0] = aTrans.strides[0];
            cStridesRes[1] = aTrans.strides[1];

            bindings.s_clip_double(
              aNative.pointer.cast(),
              cStridesA,
              minArr.pointer.cast(),
              cStridesMin,
              maxArr.pointer.cast(),
              cStridesMax,
              aTrans.pointer.cast(),
              cStridesRes,
              cShape,
              2,
              ffi.nullptr,
            );
          } finally {
            ScratchArena.reset(marker);
          }

          for (var r = 0; r < 3; r++) {
            for (var c = 0; c < 3; c++) {
              expect(
                aNative.getCell([r, c]),
                closeTo(expected[r][c], 1e-12),
                reason: 'native s_clip_double at ($r, $c)',
              );
            }
          }
        });
      },
    );
  });
}
