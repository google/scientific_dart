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
  group('Same-DType Enforcement Across Binary Operations', () {
    test('rejects mixed-dtype NDArray operands with ArgumentError', () {
      NDArray.scope(() {
        final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final f32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
        final i32 = NDArray.fromList([1, 2], [2], DType.int32);
        final i64 = NDArray.fromList([1, 2], [2], DType.int64);
        final b1 = NDArray.fromList([true, false], [2], DType.boolean);
        final f64Vec3 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final f32Vec3 = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float32);

        expect(() => add(f64, f32), throwsArgumentError);
        expect(() => subtract(f64, f32), throwsArgumentError);
        expect(() => multiply(f64, f32), throwsArgumentError);
        expect(() => divide<AnySpec, DTypeTag>(f64, f32), throwsArgumentError);
        expect(() => floorDivide(f64, f32), throwsArgumentError);
        expect(() => remainder(f64, f32), throwsArgumentError);
        expect(() => mod(f64, f32), throwsArgumentError);
        expect(() => fmod(f64, f32), throwsArgumentError);
        expect(() => divmod(f64, f32), throwsArgumentError);
        expect(() => power(f64, f32), throwsArgumentError);
        expect(
          () => floatPower<AnySpec, DTypeTag>(f64, f32),
          throwsArgumentError,
        );
        expect(() => minimum(f64, f32), throwsArgumentError);
        expect(() => maximum(f64, f32), throwsArgumentError);
        expect(() => fmin(f64, f32), throwsArgumentError);
        expect(() => fmax(f64, f32), throwsArgumentError);
        expect(() => heaviside(f64, f32), throwsArgumentError);
        expect(
          () => logaddexp<AnySpec, DTypeTag>(f64, f32),
          throwsArgumentError,
        );
        expect(
          () => logaddexp2<AnySpec, DTypeTag>(f64, f32),
          throwsArgumentError,
        );
        expect(
          () => gcd<IntegerDType>(i32.asIntegerDType, i64.asIntegerDType),
          throwsArgumentError,
        );
        expect(
          () => lcm<IntegerDType>(i32.asIntegerDType, i64.asIntegerDType),
          throwsArgumentError,
        );
        expect(() => copysign(f64, f32), throwsArgumentError);
        expect(
          () => binaryUfunc(f64, f32, op: BinaryOp.floatPower),
          throwsArgumentError,
        );
        expect(() => atan2<AnySpec, DTypeTag>(f64, f32), throwsArgumentError);
        expect(() => hypot<AnySpec, DTypeTag>(f64, f32), throwsArgumentError);
        expect(() => equal(f64, f32), throwsArgumentError);
        expect(() => notEqual(f64, f32), throwsArgumentError);
        expect(() => greater(f64, f32), throwsArgumentError);
        expect(() => greaterEqual(f64, f32), throwsArgumentError);
        expect(() => less(f64, f32), throwsArgumentError);
        expect(() => lessEqual(f64, f32), throwsArgumentError);
        expect(() => logicalAnd(b1, i32), throwsArgumentError);
        expect(() => logicalOr(b1, i32), throwsArgumentError);
        expect(() => logicalXor(b1, i32), throwsArgumentError);
        expect(
          () =>
              bitwiseAnd<BitwiseDType>(i32.asBitwiseDType, i64.asBitwiseDType),
          throwsArgumentError,
        );
        expect(
          () => bitwiseOr<BitwiseDType>(i32.asBitwiseDType, i64.asBitwiseDType),
          throwsArgumentError,
        );
        expect(
          () =>
              bitwiseXor<BitwiseDType>(i32.asBitwiseDType, i64.asBitwiseDType),
          throwsArgumentError,
        );
        expect(
          () => leftShift<IntegerDType>(i32.asIntegerDType, i64.asIntegerDType),
          throwsArgumentError,
        );
        expect(
          () =>
              rightShift<IntegerDType>(i32.asIntegerDType, i64.asIntegerDType),
          throwsArgumentError,
        );
        expect(
          () => binaryUfunc(f64, f32, op: BinaryOp.add),
          throwsArgumentError,
        );
        expect(
          () => outerUfunc(f64, f32, op: BinaryOp.add),
          throwsArgumentError,
        );
        expect(() => matmul(f64, f32), throwsArgumentError);
        expect(() => dot(f64, f32), throwsArgumentError);
        expect(() => tensordot(f64, f32, axes: 1), throwsArgumentError);
        expect(() => inner(f64, f32), throwsArgumentError);
        expect(() => vdot(f64, f32), throwsArgumentError);
        expect(() => kron(f64, f32), throwsArgumentError);
        expect(() => outer(f64, f32), throwsArgumentError);
        expect(() => cross(f64Vec3, f32Vec3), throwsArgumentError);

        // *As variants explicitly support mixed input dtypes into target dtype:
        expect(addAs(f64, f32, DType.float64).toList(), [2.0, 4.0]);
        expect(subtractAs(f64, f32, DType.float64).toList(), [0.0, 0.0]);
        expect(multiplyAs(f64, f32, DType.float64).toList(), [1.0, 4.0]);
        expect(divideAs(f64, f32, DType.float64).toList(), [1.0, 1.0]);
        expect(floorDivideAs(f64, i32, DType.float64).toList(), [1.0, 1.0]);
        expect(remainderAs(f64, i32, DType.float64).toList(), [0.0, 0.0]);
        expect(modAs(f64, i32, DType.float64).toList(), [0.0, 0.0]);
        expect(fmodAs(f64, i32, DType.float64).toList(), [0.0, 0.0]);
        final dm = divmodAs(f64, i32, DType.float64);
        expect(dm.quotient.toList(), [1.0, 1.0]);
        expect(dm.remainder.toList(), [0.0, 0.0]);
        expect(powerAs(f64, i32, DType.float64).toList(), [1.0, 4.0]);
        expect(floatPowerAs(i32, i64, DType.float64).toList(), [1.0, 4.0]);
        expect(minimumAs(f64, i32, DType.float64).toList(), [1.0, 2.0]);
        expect(maximumAs(f64, i32, DType.float64).toList(), [1.0, 2.0]);
        expect(fminAs(f64, i32, DType.float64).toList(), [1.0, 2.0]);
        expect(fmaxAs(f64, i32, DType.float64).toList(), [1.0, 2.0]);
        expect(heavisideAs(f64, i32, DType.float64).toList(), [1.0, 1.0]);
        expect(copysignAs(f64, i32, DType.float64).toList(), [1.0, 2.0]);
        expect(atan2As(f64, f32, DType.float64).dtype, DType.float64);
        expect(hypotAs(f64, f32, DType.float64).dtype, DType.float64);
        expect(logaddexpAs(f64, f32, DType.float64).dtype, DType.float64);
        expect(logaddexp2As(f64, f32, DType.float64).dtype, DType.float64);
        expect(gcdAs(i32, i64, DType.int64).toList(), [1, 2]);
        expect(lcmAs(i32, i64, DType.int64).toList(), [1, 2]);
        expect(bitwiseAndAs(i32, i64, DType.int64).toList(), [1, 2]);
        expect(bitwiseOrAs(i32, i64, DType.int64).toList(), [1, 2]);
        expect(bitwiseXorAs(i32, i64, DType.int64).toList(), [0, 0]);
        expect(leftShiftAs(i32, i64, DType.int64).toList(), [2, 8]);
        expect(rightShiftAs(i32, i64, DType.int64).toList(), [0, 0]);
        expect(matmulAs(f64, f32, DType.float64).scalar, 5.0);
        expect(dotAs(f64, f32, DType.float64).scalar, 5.0);
        expect(tensordotAs(f64, f32, DType.float64, axes: 1).scalar, 5.0);
        expect(innerAs(f64, f32, DType.float64).scalar, 5.0);
        expect(vdotAs(f64, f32, DType.float64).scalar, 5.0);
        expect(kronAs(f64, f32, DType.float64).toList(), [1.0, 2.0, 2.0, 4.0]);
        expect(outerAs(f64, f32, DType.float64).toList(), [1.0, 2.0, 2.0, 4.0]);
        expect(crossAs(f64Vec3, f32Vec3, DType.float64).toList(), [
          -3.0,
          6.0,
          -3.0,
        ]);
      });
    });
  });

  group('P3: round() Half-to-Even (Banker Rounding)', () {
    test('rounds halfway cases to nearest even integer', () {
      NDArray.scope(() {
        for (final dtype in [DType.float64, DType.float32]) {
          final a = NDArray.fromList(
            [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5],
            [8],
            dtype,
          );
          final res = round(a);
          expect(res.toList(), [0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0, -4.0]);

          // Strided view
          final strided = a.slice([Slice(start: 0, stop: 8, step: 2)]);
          final resStrided = round(strided);
          expect(resStrided.toList(), [0.0, 2.0, 0.0, -2.0]);
        }

        final dec = NDArray.fromList(
          [0.05, 0.15, 0.25, 0.35],
          [4],
          DType.float64,
        );
        final resDec = round(dec, decimals: 1);
        expect(resDec.getCell([0]), closeTo(0.0, 1e-12));
        expect(resDec.getCell([2]), closeTo(0.2, 1e-12));
      });
    });
  });

  group('P4 & P5: Float floorDivide Quotient and Division by Zero', () {
    test(
      'returns divmod-consistent floor quotient and IEEE-754 signed Inf/NaN on zero divisor',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 5.5, -5.5], [3], DType.float64);
          final b = NDArray.fromList([0.1, 2.0, 2.0], [3], DType.float64);
          final q = floorDivide(a, b);
          expect(q.toList(), [9.0, 2.0, -3.0]);

          for (final dtype in [DType.float64, DType.float32]) {
            final numArr = NDArray.fromList(
              [1.0, -1.0, 0.0, double.nan],
              [4],
              dtype,
            );
            final zeroArr = NDArray.fromList([0.0, 0.0, 0.0, 0.0], [4], dtype);
            final divZero = floorDivide(numArr, zeroArr);
            expect(divZero.getCell([0]), double.infinity);
            expect(divZero.getCell([1]), double.negativeInfinity);
            expect(divZero.getCell([2]).isNaN, isTrue);
            expect(divZero.getCell([3]).isNaN, isTrue);
          }
        });
      },
    );
  });

  group('P7: Saturating Float-to-Int Casts and NaN -> 0', () {
    test(
      'saturates out-of-range floats and maps NaN to 0 across all integer dtypes',
      () {
        NDArray.scope(() {
          final src = NDArray.fromList(
            [double.nan, double.infinity, double.negativeInfinity, 1e30, -1e30],
            [5],
            DType.float64,
          );

          expect(src.astype(DType.int8).toList(), [0, 127, -128, 127, -128]);
          expect(src.astype(DType.uint8).toList(), [0, 255, 0, 255, 0]);
          expect(src.astype(DType.int16).toList(), [
            0,
            32767,
            -32768,
            32767,
            -32768,
          ]);
          expect(src.astype(DType.uint16).toList(), [0, 65535, 0, 65535, 0]);
          expect(src.astype(DType.int32).toList(), [
            0,
            2147483647,
            -2147483648,
            2147483647,
            -2147483648,
          ]);
          expect(src.astype(DType.uint32).toList(), [
            0,
            4294967295,
            0,
            4294967295,
            0,
          ]);
          expect(src.astype(DType.int64).toList(), [
            0,
            9223372036854775807,
            -9223372036854775808,
            9223372036854775807,
            -9223372036854775808,
          ]);
          // uint64 max is 0xFFFFFFFFFFFFFFFF (-1 in signed 64-bit representation)
          expect(src.astype(DType.uint64).toList(), [0, -1, 0, -1, 0]);

          // Also test strided view
          final strided = src.slice([Slice(start: 0, stop: 5, step: 2)]);
          expect(strided.astype(DType.int8).toList(), [0, -128, -128]);
          expect(strided.astype(DType.uint8).toList(), [0, 0, 0]);
        });
      },
    );
  });

  group(
    'B1 (H7) & M1: Exact Modular Integer Arithmetic Without Float64 Clamping',
    () {
      test(
        'uint64, uint32, uint16, and int8 preserve exact modular wrap-around',
        () {
          NDArray.scope(() {
            // uint64: 2^63 + 2^63 == 0 (mod 2^64), and large integers > 2^53 preserve exact bits
            const int twoPow63 = -9223372036854775808; // 0x8000000000000000
            const int largeExact = 4611686018427387905; // (1 << 62) + 1
            final u64A = NDArray.fromList(
              [twoPow63, 0, largeExact],
              [3],
              DType.uint64,
            );
            final u64B = NDArray.fromList([twoPow63, 1, 2], [3], DType.uint64);
            expect(add(u64A, u64B).toList(), [0, 1, largeExact + 2]);
            expect(subtract(u64A, u64B).toList(), [0, -1, largeExact - 2]);

            // uint32: 4294967295 * 4294967295 == 1 (mod 2^32)
            final u32A = NDArray.fromList([4294967295, 0], [2], DType.uint32);
            final u32B = NDArray.fromList([4294967295, 1], [2], DType.uint32);
            expect(multiply(u32A, u32B).toList(), [1, 0]);
            expect(subtract(u32A, u32B).toList(), [0, 4294967295]);

            // int8: 127 + 1 == -128, -128 - 1 == 127, abs(-128) == -128
            final i8A = NDArray.fromList([127, -128], [2], DType.int8);
            final i8B = NDArray.fromList([1, 1], [2], DType.int8);
            expect(add(i8A, i8B).toList(), [-128, -127]);
            expect(subtract(i8A, i8B).toList(), [126, 127]);
            expect(abs(i8A).toList(), [127, -128]);
          });
        },
      );
    },
  );

  group('B5 (M13): Float16 and BFloat16 IEEE-754 Round-to-Nearest-Even', () {
    test('rounds tie cases to even and preserves NaN', () {
      NDArray.scope(() {
        // Float16 tie at 1.0 + 2^-11 (ULP at 1.0 is 2^-10)
        final tieEven =
            1.0 + (1.0 / 2048.0); // bit 0 is 0 -> rounds down to 1.0
        final tieOdd =
            1.0 + (3.0 / 2048.0); // bit 0 is 1 -> rounds up to 1.0 + 2^-9
        final f16 = NDArray.fromList(
          [tieEven, tieOdd, double.nan],
          [3],
          DType.float16,
        );
        expect(f16.getCell([0]), 1.0);
        expect(f16.getCell([1]), 1.0 + (1.0 / 512.0));
        expect(f16.getCell([2]).isNaN, isTrue);

        // BFloat16 tie at 1.0 + 2^-8 (ULP at 1.0 is 2^-7)
        final bfTieEven = 1.0 + (1.0 / 256.0);
        final bfTieOdd = 1.0 + (3.0 / 256.0);
        final bf16 = NDArray.fromList(
          [bfTieEven, bfTieOdd, double.nan],
          [3],
          DType.bfloat16,
        );
        expect(bf16.getCell([0]), 1.0);
        expect(bf16.getCell([1]), 1.0 + (1.0 / 64.0));
        expect(bf16.getCell([2]).isNaN, isTrue);
      });
    });
  });

  group('B7: Overflow/Underflow-Safe Complex abs() via hypot', () {
    test(
      'avoids intermediate overflow/underflow on large and small complex magnitudes',
      () {
        NDArray.scope(() {
          final c128 = NDArray.fromList(
            [Complex(3e200, 4e200), Complex(3e-200, 4e-200)],
            [2],
            DType.complex128,
          );
          final mag128 = abs(c128);
          expect(mag128.getCell([0]), closeTo(5e200, 1e186));
          expect(mag128.getCell([1]), closeTo(5e-200, 1e-214));

          final c64 = NDArray.fromList(
            [Complex(3e25, 4e25), Complex(3e-25, 4e-25)],
            [2],
            DType.complex64,
          );
          final mag64 = abs(c64);
          expect(mag64.getCell([0]), closeTo(5e25, 1e19));
          expect(mag64.getCell([1]), closeTo(5e-25, 1e-31));
        });
      },
    );
  });

  group('B13: Bitwise Shifts With Negative or >= BitWidth Shift Counts', () {
    test('produces defined results without C++ undefined behavior', () {
      NDArray.scope(() {
        final i32A = NDArray.fromList([16, -16, 16, -16], [4], DType.int32);
        final i32B = NDArray.fromList([-1, -1, 32, 32], [4], DType.int32);
        expect(leftShift(i32A, i32B).toList(), [0, 0, 0, 0]);
        expect(rightShift(i32A, i32B).toList(), [0, 0, 0, -1]);

        final i8A = NDArray.fromList([8, -8, 8, -8], [4], DType.int8);
        final i8B = NDArray.fromList([-1, -1, 8, 8], [4], DType.int8);
        expect(leftShift(i8A, i8B).toList(), [0, 0, 0, 0]);
        expect(rightShift(i8A, i8B).toList(), [0, 0, 0, -1]);
      });
    });
  });

  group('B14 & B15: Boolean add, multiply, kron, matmul, and invert', () {
    test(
      'boolean operations return NDArray<Boolean> with logical OR/AND/NOT semantics',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [true, true, false, false],
            [2, 2],
            DType.boolean,
          );
          final b = NDArray.fromList(
            [true, false, true, false],
            [2, 2],
            DType.boolean,
          );

          final resAdd = add(a, b);
          expect(resAdd.dtype, DType.boolean);
          expect(resAdd.toList(), [true, true, true, false]);

          final resMul = multiply(a, b);
          expect(resMul.dtype, DType.boolean);
          expect(resMul.toList(), [true, false, false, false]);

          final resInv = invert(a);
          expect(resInv.dtype, DType.boolean);
          expect(resInv.toList(), [false, false, true, true]);

          final resKron = kron(
            NDArray.fromList([true, false], [2], DType.boolean),
            NDArray.fromList([true, true], [2], DType.boolean),
          );
          expect(resKron.dtype, DType.boolean);
          expect(resKron.toList(), [true, true, false, false]);

          final resMatmul = matmul(a, b);
          expect(resMatmul.dtype, DType.boolean);
          expect(resMatmul.toList(), [true, false, false, false]);
        });
      },
    );
  });

  group('S6 & S7: clip() NaN Propagation and hypot() Complex Rejection', () {
    test('clip propagates NaN in array or bounds', () {
      NDArray.scope(() {
        for (final dtype in [DType.float64, DType.float32]) {
          final a = NDArray.fromList([double.nan, -5.0, 0.0, 5.0], [4], dtype);
          final res = clip(a, min: -1.0, max: 1.0);
          expect(res.getCell([0]).isNaN, isTrue);
          expect(res.getCell([1]), -1.0);
          expect(res.getCell([2]), 0.0);
          expect(res.getCell([3]), 1.0);

          final resNanMin = clip(a, min: double.nan, max: 1.0);
          expect(resNanMin.getCell([2]).isNaN, isTrue);
        }
      });
    });

    test('hypot rejects complex inputs with UnsupportedError', () {
      NDArray.scope(() {
        final c128 = NDArray.fromList([Complex(3, 4)], [1], DType.complex128);
        final c64 = NDArray.fromList([Complex(3, 4)], [1], DType.complex64);
        expect(() => hypot(c128, c128), throwsUnsupportedError);
        expect(() => hypot(c64, c64), throwsUnsupportedError);
      });
    });
  });

  group('DivideTag, BitwiseTag, and ShiftTag Static & Runtime Contracts', () {
    test(
      'DivideTag: f16 / f16 and bf16 / bf16 preserve dtype; int/bool promote to Float64',
      () {
        NDArray.scope(() {
          final f16a = NDArray.fromList([6.0, 9.0], [2], DType.float16);
          final f16b = NDArray.fromList([2.0, 3.0], [2], DType.float16);
          final NDArray<Float16> q16 = f16a / f16b;
          final NDArray<Float16> q16Fn = divide(f16a, f16b);
          expect(q16.dtype, DType.float16);
          expect(q16.toList(), [3.0, 3.0]);
          expect(q16Fn.toList(), [3.0, 3.0]);

          final bf16a = NDArray.fromList([8.0, 12.0], [2], DType.bfloat16);
          final bf16b = NDArray.fromList([2.0, 4.0], [2], DType.bfloat16);
          final NDArray<BFloat16> qbf16 = bf16a / bf16b;
          final NDArray<BFloat16> qbf16Fn = divide(bf16a, bf16b);
          expect(qbf16.dtype, DType.bfloat16);
          expect(qbf16.toList(), [4.0, 3.0]);
          expect(qbf16Fn.toList(), [4.0, 3.0]);

          final i32a = NDArray.fromList([3, 6], [2], DType.int32);
          final i32b = NDArray.fromList([2, 4], [2], DType.int32);
          final NDArray<Float64> qi32 = i32a / i32b;
          final NDArray<Float64> qi32Fn = divide(i32a, i32b);
          expect(qi32.dtype, DType.float64);
          expect(qi32.toList(), [1.5, 1.5]);
          expect(qi32Fn.toList(), [1.5, 1.5]);
        });
      },
    );

    test(
      'BitwiseTag and ShiftTag: boolean & | ^ ~ work and invalid widened ops throw UnsupportedError',
      () {
        NDArray.scope(() {
          final b1 = NDArray.fromList(
            [true, true, false, false],
            [4],
            DType.boolean,
          );
          final b2 = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );

          final NDArray<Boolean> resAnd = b1 & b2;
          final NDArray<Boolean> resOr = b1 | b2;
          final NDArray<Boolean> resXor = b1 ^ b2;
          final NDArray<Boolean> resNot = ~b1;
          expect(resAnd.toList(), [true, false, false, false]);
          expect(resOr.toList(), [true, true, true, false]);
          expect(resXor.toList(), [false, true, true, false]);
          expect(resNot.toList(), [false, false, true, true]);

          expect(bitwiseAnd(b1, b2).toList(), [true, false, false, false]);
          expect(bitwiseOr(b1, b2).toList(), [true, true, true, false]);
          expect(bitwiseXor(b1, b2).toList(), [false, true, true, false]);

          final i32 = NDArray.fromList([2, 4], [2], DType.int32);
          final NDArray<Int32> shl = i32 << 1;
          final NDArray<Int32> shr = i32 >> 1;
          expect(shl.toList(), [4, 8]);
          expect(shr.toList(), [1, 2]);

          final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(
            () => bitwiseAnd<BitwiseDType>(
              f64.asBitwiseDType,
              f64.asBitwiseDType,
            ),
            throwsArgumentError,
          );
          expect(
            () => invert<BitwiseDType>(f64.asBitwiseDType),
            throwsArgumentError,
          );
          expect(
            () => leftShift<IntegerDType>(b1.asIntegerDType, b1.asIntegerDType),
            throwsArgumentError,
          );
        });
      },
    );
  });
}
