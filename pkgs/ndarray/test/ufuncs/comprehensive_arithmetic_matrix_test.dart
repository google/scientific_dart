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

import "dart:math" as math;
import "dart:ffi" as ffi;
import "package:ndarray/ndarray.dart";
import "package:ndarray/src/operations/helpers.dart";
import "package:test/test.dart";

void main() {
  const allDTypes = [
    DType.float64,
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint64,
    DType.uint32,
    DType.uint16,
    DType.uint8,
    DType.complex128,
    DType.complex64,
    DType.boolean,
  ];

  dynamic sampleValue(DType dtype, int seed) {
    switch (dtype) {
      case DType.complex128:
      case DType.complex64:
        return Complex(seed.toDouble() + 1.0, (seed % 3).toDouble() + 1.0);
      case DType.float64:
      case DType.float32:
      case DType.float16:
      case DType.bfloat16:
        return seed.toDouble() + 1.5;
      case DType.boolean:
        return seed.isOdd;
      default:
        return (seed % 10) + 1;
    }
  }

  NDArray<AnySpec> createArray(
    List<int> shape,
    DType dtype, {
    int seedOffset = 0,
  }) {
    final size = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final list = List.generate(size, (i) => sampleValue(dtype, i + seedOffset));
    return switch (dtype) {
      DType.float64 => NDArray<Float64>.fromList(list, shape, DType.float64),
      DType.float32 => NDArray<Float32>.fromList(list, shape, DType.float32),
      DType.float16 => NDArray<Float16>.fromList(list, shape, DType.float16),
      DType.bfloat16 => NDArray<BFloat16>.fromList(list, shape, DType.bfloat16),
      DType.int64 => NDArray<Int64>.fromList(list, shape, DType.int64),
      DType.int32 => NDArray<Int32>.fromList(list, shape, DType.int32),
      DType.int16 => NDArray<Int16>.fromList(list, shape, DType.int16),
      DType.int8 => NDArray<Int8>.fromList(list, shape, DType.int8),
      DType.uint64 => NDArray<Uint64>.fromList(list, shape, DType.uint64),
      DType.uint32 => NDArray<Uint32>.fromList(list, shape, DType.uint32),
      DType.uint16 => NDArray<Uint16>.fromList(list, shape, DType.uint16),
      DType.uint8 => NDArray<Uint8>.fromList(list, shape, DType.uint8),
      DType.complex128 => NDArray<Complex128>.fromList(
        list,
        shape,
        DType.complex128,
      ),
      DType.complex64 => NDArray<Complex64>.fromList(
        list,
        shape,
        DType.complex64,
      ),
      DType.boolean => NDArray<Boolean>.fromList(list, shape, DType.boolean),
    };
  }

  group("Workstream 1: Parametrized Binary Arithmetic across all 15 DTypes", () {
    test(
      "Homogeneous binary arithmetic (add, subtract, multiply, divide) on all 15 DTypes",
      () {
        NDArray.scope(() {
          for (final dtype in allDTypes) {
            final a = createArray([2, 3], dtype, seedOffset: 1);
            final b = createArray([2, 3], dtype, seedOffset: 2);

            // add
            final sum = add(a, b);
            expect(sum.shape, [2, 3]);
            expect(sum.dtype, dtype);

            // subtract
            if (dtype == DType.boolean) {
              expect(() => subtract(a, b), throwsUnsupportedError);
            } else {
              final diff = subtract(a, b);
              expect(diff.shape, [2, 3]);
              expect(diff.dtype, resolveDType(dtype, dtype));
            }

            // multiply
            final prod = multiply(a, b);
            expect(prod.shape, [2, 3]);
            expect(prod.dtype, dtype);

            // divide
            final quot = divide(a, b);
            expect(quot.shape, [2, 3]);
            if (dtype.isComplex) {
              expect(quot.dtype.isComplex, true);
            } else {
              expect(quot.dtype.isFloating, true);
            }
          }
        });
      },
    );

    test(
      "Cross-DType binary arithmetic combinations for add, subtract, multiply, divide",
      () {
        NDArray.scope(() {
          final keyDTypes = [
            DType.float64,
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int64,
            DType.int32,
            DType.int16,
            DType.int8,
            DType.uint64,
            DType.uint32,
            DType.uint16,
            DType.uint8,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (var i = 0; i < keyDTypes.length; i++) {
            for (var j = 0; j < keyDTypes.length; j++) {
              final dtA = keyDTypes[i];
              final dtB = keyDTypes[j];

              final a = createArray([2, 2], dtA, seedOffset: 1);
              final b = createArray([2, 2], dtB, seedOffset: 2);

              if (dtA != dtB) {
                expect(() => add<AnySpec>(a, b), throwsArgumentError);
                expect(() => subtract<AnySpec>(a, b), throwsArgumentError);
                expect(() => multiply<AnySpec>(a, b), throwsArgumentError);
                expect(
                  () => divide<AnySpec, DTypeTag>(a, b),
                  throwsArgumentError,
                );
                continue;
              }

              final sum = add<AnySpec>(a, b);
              expect(sum.shape, [2, 2]);
              expect(sum.dtype, dtA);

              if (dtA == DType.boolean) {
                expect(() => subtract<AnySpec>(a, b), throwsUnsupportedError);
              } else {
                final diff = subtract<AnySpec>(a, b);
                expect(diff.shape, [2, 2]);
                expect(diff.dtype, resolveDType(dtA, dtB));
              }

              final prod = multiply<AnySpec>(a, b);
              expect(prod.shape, [2, 2]);
              expect(prod.dtype, dtA);

              final quot = divide<AnySpec, DTypeTag>(a, b);
              expect(quot.shape, [2, 2]);
            }
          }
        });
      },
    );

    test(
      "Contiguous vs Non-contiguous / Strided Transposed Views for arithmetic",
      () {
        NDArray.scope(() {
          final testTypes = [
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.int16,
            DType.uint8,
            DType.complex128,
            DType.complex64,
            DType.float16,
          ];

          for (final dtype in testTypes) {
            final a = createArray([3, 4], dtype, seedOffset: 0);
            final b = createArray([4, 3], dtype, seedOffset: 5);

            // bTransposed is non-contiguous with shape [3, 4]
            final bT = b.transpose([1, 0]);
            expect(bT.isContiguous, false);
            expect(bT.shape, [3, 4]);

            final sum = add(a, bT);
            expect(sum.shape, [3, 4]);

            final diff = subtract(a, bT);
            expect(diff.shape, [3, 4]);

            final prod = multiply(a, bT);
            expect(prod.shape, [3, 4]);

            final quot = divide(a, bT);
            expect(quot.shape, [3, 4]);

            // Both non-contiguous
            final aT = a.transpose([1, 0]);
            final sumTT = add(aT, b);
            expect(sumTT.shape, [4, 3]);
          }
        });
      },
    );

    test("Multidimensional broadcasting and 0-D scalar arrays", () {
      NDArray.scope(() {
        // 0-D scalar arrays
        final s1 = NDArray.scalar(10.0, dtype: DType.float64);
        final s2 = NDArray.scalar(2.0, dtype: DType.float64);
        final a2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final sumScalar = add(s1, s2);
        expect(sumScalar.shape, <int>[]);
        expect(sumScalar.toList(), [12.0]);

        final sumBroad = add(a2d, s1);
        expect(sumBroad.shape, [2, 2]);
        expect(sumBroad.toList(), [11.0, 12.0, 13.0, 14.0]);

        final diffBroad = subtract(s1, a2d);
        expect(diffBroad.shape, [2, 2]);
        expect(diffBroad.toList(), [9.0, 8.0, 7.0, 6.0]);

        // 3D broadcasting: [2, 1, 3] with [1, 4, 1] -> [2, 4, 3]
        final a3d = NDArray.fromList(
          List.generate(6, (i) => (i + 1).toDouble()),
          [2, 1, 3],
          DType.float64,
        );
        final b3d = NDArray.fromList(
          List.generate(4, (i) => ((i + 1) * 10).toDouble()),
          [1, 4, 1],
          DType.float64,
        );

        final sum3d = add(a3d, b3d);
        expect(sum3d.shape, [2, 4, 3]);
        expect(sum3d.size, 24);

        final mul3d = multiply(a3d, b3d);
        expect(mul3d.shape, [2, 4, 3]);
      });
    });

    test(
      "In-place out destination parameter buffer reuse and compatibility checks",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final outBuffer = NDArray<Float64>.create([2, 2], DType.float64);

          final res = add<Float64>(a, b, out: outBuffer);
          expect(identical(res, outBuffer), true);
          expect(outBuffer.toList(), [11.0, 22.0, 33.0, 44.0]);

          // Strided out buffer
          final fullOut = NDArray<Float64>.zeros([2, 4], DType.float64);
          final stridedOut = fullOut.slice([Slice.all(), Slice(step: 2)]);
          expect(stridedOut.isContiguous, false);
          expect(stridedOut.shape, [2, 2]);

          final resStrided = add<Float64>(a, b, out: stridedOut);
          expect(resStrided.toList(), [11.0, 22.0, 33.0, 44.0]);

          // Incompatible shape throws ArgumentError
          final invalidShapeOut = NDArray<Float64>.create([
            3,
            2,
          ], DType.float64);
          expect(
            () => add<Float64>(a, b, out: invalidShapeOut),
            throwsArgumentError,
          );

          // Incompatible dtype throws ArgumentError
          final invalidDTypeOut = NDArray<Int32>.create([2, 2], DType.int32);
          expect(
            () => add<AnySpec>(
              a.asAnySpec,
              b.asAnySpec,
              out: invalidDTypeOut.asAnySpec,
            ),
            throwsArgumentError,
          );
        });
      },
    );

    test("Optional where boolean and uint8 mask parameters for arithmetic", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
        );
        final b = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final whereMask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final out = NDArray.fromList(
          [100.0, 200.0, 300.0, 400.0],
          [4],
          DType.float64,
        );

        final res = add(a, b, where: whereMask, out: out);
        expect(res.toList(), [11.0, 200.0, 33.0, 400.0]);

        // uint8 where mask
        final uint8Mask = NDArray.fromList([1, 0, 1, 0], [4], DType.uint8);
        final out2 = NDArray.fromList(
          [100.0, 200.0, 300.0, 400.0],
          [4],
          DType.float64,
        );
        final res2 = subtract(a, b, where: uint8Mask, out: out2);
        expect(res2.toList(), [9.0, 200.0, 27.0, 400.0]);

        // Broadcasted where mask
        final maskBroad = NDArray.fromList(
          [true, false],
          [2, 1],
          DType.boolean,
        );
        final a2d = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b2d = NDArray.fromList(
          [10.0, 10.0, 10.0, 10.0],
          [2, 2],
          DType.float64,
        );
        final outBroad = NDArray.zeros([2, 2], DType.float64);
        final resBroad = add(a2d, b2d, where: maskBroad, out: outBroad);
        expect(resBroad.toList(), [11.0, 12.0, 0.0, 0.0]);

        // Invalid where dtype throws ArgumentError
        final invalidWhere = NDArray.fromList(
          [1.0, 0.0, 1.0, 0.0],
          [4],
          DType.float64,
        );
        expect(() => add(a, b, where: invalidWhere), throwsArgumentError);
      });
    });

    test("Disposed inputs error validation across operations", () {
      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
      final c = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final w = NDArray.fromList([true, false], [2], DType.boolean);
      a.dispose();

      expect(() => add(a, b), throwsStateError);
      expect(() => subtract(a, b), throwsStateError);
      expect(() => multiply(a, b), throwsStateError);
      expect(() => divide(a, b), throwsStateError);
      expect(() => sqrt(a), throwsStateError);
      expect(() => expm1(a), throwsStateError);
      expect(() => log1p(a), throwsStateError);
      expect(() => isnan(a), throwsStateError);
      expect(() => isinf(a), throwsStateError);
      expect(() => isfinite(a), throwsStateError);
      expect(() => copysign(a, b), throwsStateError);
      expect(() => copysign(b, a), throwsStateError);

      b.dispose();
      c.dispose();
      w.dispose();
    });
  });

  group("Workstream 1: Specialized Binary Mathematical Functions", () {
    test("floorDivide (and BinaryOp.floorDivide) across numeric DTypes", () {
      NDArray.scope(() {
        final intTypes = <DType<AnyIntegerSpec>>[
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ];

        for (final dtype in intTypes) {
          final x1 = NDArray.fromList([7, 15, 20, 25], [2, 2], dtype);
          final x2 = NDArray.fromList([3, 4, 6, 7], [2, 2], dtype);
          final res = floorDivide(x1, x2);
          expect(res.shape, [2, 2]);
          expect(res.toList(), [2, 3, 3, 3]);

          // Strided
          final x1T = x1.transpose([1, 0]);
          final resT = floorDivide(x1T, x2);
          expect(resT.shape, [2, 2]);

          // out and where
          final outBuf = NDArray.zeros([2, 2], dtype);
          final mask = NDArray.fromList(
            [true, false, true, false],
            [2, 2],
            DType.boolean,
          );
          floorDivide(x1, x2, where: mask, out: outBuf);
          expect(outBuf.toList(), [2, 0, 3, 0]);
        }

        // Float floorDivide
        final f64_1 = NDArray.fromList(
          [7.5, -7.5, 8.0, -8.0],
          [4],
          DType.float64,
        );
        final f64_2 = NDArray.fromList(
          [2.0, 2.0, 3.0, 3.0],
          [4],
          DType.float64,
        );
        final f64Res = floorDivide(f64_1, f64_2);
        expect(f64Res.toList(), [3.0, -4.0, 2.0, -3.0]);

        final f32_1 = NDArray.fromList([7.5, -7.5], [2], DType.float32);
        final f32_2 = NDArray.fromList([2.0, 2.0], [2], DType.float32);
        final f32Res = floorDivide(f32_1, f32_2);
        expect(f32Res.toList(), [3.0, -4.0]);

        // Complex floorDivide throws UnsupportedError
        final cpx = NDArray.fromList([Complex(1, 2)], [1], DType.complex128);
        expect(() => floorDivide(cpx, cpx), throwsUnsupportedError);
      });
    });

    test("remainder, mod, and fmod operations", () {
      NDArray.scope(() {
        final x1 = NDArray.fromList(
          [10.0, -10.0, 10.0, -10.0],
          [4],
          DType.float64,
        );
        final x2 = NDArray.fromList([3.0, 3.0, -3.0, -3.0], [4], DType.float64);

        // remainder: Python style % (sign of divisor)
        final remRes = remainder(x1, x2);
        expect(remRes.toList(), [1.0, 2.0, -2.0, -1.0]);

        final modRes = mod(x1, x2);
        expect(modRes.toList(), [1.0, 2.0, -2.0, -1.0]);

        // fmod: C style fmod (sign of dividend)
        final fmodRes = fmod(x1, x2);
        expect(fmodRes.toList(), [1.0, -1.0, 1.0, -1.0]);

        // Float32 fmod & remainder
        final x1f = NDArray.fromList([10.0, -10.0], [2], DType.float32);
        final x2f = NDArray.fromList([3.0, 3.0], [2], DType.float32);
        expect(fmod(x1f, x2f).toList(), [1.0, -1.0]);
        expect(remainder(x1f, x2f).toList(), [1.0, 2.0]);

        // Integer remainder & fmod across dtypes
        final i32_1 = NDArray.fromList([10, -10, 10, -10], [4], DType.int32);
        final i32_2 = NDArray.fromList([3, 3, -3, -3], [4], DType.int32);
        expect(remainder(i32_1, i32_2).toList(), [1, 2, -2, -1]);
        expect(fmod(i32_1, i32_2).toList(), [1, -1, 1, -1]);

        final u8_1 = NDArray.fromList([10, 20], [2], DType.uint8);
        final u8_2 = NDArray.fromList([3, 6], [2], DType.uint8);
        expect(remainder(u8_1, u8_2).toList(), [1, 2]);
        expect(fmod(u8_1, u8_2).toList(), [1, 2]);

        // Complex throws UnsupportedError
        final cpx = NDArray.fromList([Complex(1, 2)], [1], DType.complex128);
        expect(() => remainder(cpx, cpx), throwsUnsupportedError);
        expect(() => fmod(cpx, cpx), throwsUnsupportedError);
      });
    });

    test(
      "power operation across floating, integer, boolean, and complex DTypes",
      () {
        NDArray.scope(() {
          // Float power
          final baseF = NDArray.fromList(
            [2.0, 3.0, 4.0, 5.0],
            [4],
            DType.float64,
          );
          final expF = NDArray.fromList(
            [3.0, 2.0, 0.5, 0.0],
            [4],
            DType.float64,
          );
          final powF = power(baseF, expF);
          expect(powF.toList(), [8.0, 9.0, 2.0, 1.0]);

          // Float32 power
          final baseF32 = NDArray.fromList([2.0, 3.0], [2], DType.float32);
          final expF32 = NDArray.fromList([3.0, 2.0], [2], DType.float32);
          expect(power(baseF32, expF32).toList(), [8.0, 9.0]);

          // Integer power
          final baseI = NDArray.fromList([2, 3, 4], [3], DType.int32);
          final expI = NDArray.fromList([3, 2, 0], [3], DType.int32);
          expect(power(baseI, expI).toList(), [8, 9, 1]);

          final baseI64 = NDArray.fromList([2, 3], [2], DType.int64);
          final expI64 = NDArray.fromList([4, 3], [2], DType.int64);
          expect(power(baseI64, expI64).toList(), [16, 27]);

          final baseU8 = NDArray.fromList([2, 3], [2], DType.uint8);
          final expU8 = NDArray.fromList([3, 2], [2], DType.uint8);
          expect(power(baseU8, expU8).toList(), [8, 9]);

          // Complex power
          final c1 = NDArray.fromList(
            [Complex(0, 1), Complex(1, 1)],
            [2],
            DType.complex128,
          );
          final c2 = NDArray.fromList(
            [Complex(2, 0), Complex(2, 0)],
            [2],
            DType.complex128,
          );
          final cpow = power(c1, c2);
          expect(cpow.dtype, DType.complex128);
          final cList = cpow.toList();
          expect(cList[0].real, closeTo(-1.0, 1e-6));
          expect(cList[0].imag, closeTo(0.0, 1e-6));
          expect(cList[1].real, closeTo(0.0, 1e-6));
          expect(cList[1].imag, closeTo(2.0, 1e-6));

          // Complex64 power
          final c1_64 = NDArray.fromList([Complex(0, 1)], [1], DType.complex64);
          final c2_64 = NDArray.fromList([Complex(2, 0)], [1], DType.complex64);
          final cpow64 = power(c1_64, c2_64);
          expect(cpow64.dtype, DType.complex64);
        });
      },
    );

    test("heaviside step function across DTypes and edge cases", () {
      NDArray.scope(() {
        final x1 = NDArray.fromList([-1.5, 0.0, 2.5], [3], DType.float64);
        final x2 = NDArray.fromList([0.5, 0.5, 0.5], [3], DType.float64);

        final h = heaviside(x1, x2);
        expect(h.toList(), [0.0, 0.5, 1.0]);

        // Float32 heaviside
        final x1f = NDArray.fromList([-2.0, 0.0, 3.0], [3], DType.float32);
        final x2f = NDArray.fromList([0.75, 0.75, 0.75], [3], DType.float32);
        expect(heaviside(x1f, x2f).toList(), [0.0, 0.75, 1.0]);

        // Int32 / Int64 heaviside
        final x1i = NDArray.fromList([-5, 0, 5], [3], DType.int32);
        final x2i = NDArray.fromList([2, 2, 2], [3], DType.int32);
        expect(heaviside(x1i, x2i).toList(), [0, 2, 1]);

        // Strided, where mask, and out buffer
        final outH = NDArray.zeros([3], DType.float64);
        final mask = NDArray.fromList([true, true, false], [3], DType.boolean);
        heaviside(x1, x2, where: mask, out: outH);
        expect(outH.toList(), [0.0, 0.5, 0.0]);

        // Complex throws UnsupportedError
        final cpx = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
        expect(() => heaviside(cpx, cpx), throwsUnsupportedError);
      });
    });

    test("copysign across floating, integer, and strided arrays", () {
      NDArray.scope(() {
        final x1 = NDArray.fromList([1.0, -2.0, 3.0, -4.0], [4], DType.float64);
        final x2 = NDArray.fromList([-1.0, 1.0, 0.0, -0.0], [4], DType.float64);

        final res = copysign(x1, x2);
        expect(res.toList(), [-1.0, 2.0, 3.0, -4.0]);

        // Float32 copysign
        final x1f = NDArray.fromList([5.0, -5.0], [2], DType.float32);
        final x2f = NDArray.fromList([-1.0, 1.0], [2], DType.float32);
        expect(copysign(x1f, x2f).toList(), [-5.0, 5.0]);

        // Integer copysign
        final x1i = NDArray.fromList([10, -20], [2], DType.int32);
        final x2i = NDArray.fromList([-1, 1], [2], DType.int32);
        expect(copysign(x1i, x2i).toList(), [-10, 20]);

        // Strided copysign
        final x1s = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        ).transpose([1, 0]);
        final x2s = NDArray.fromList(
          [-1.0, -1.0, 1.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final resStrided = copysign(x1s, x2s);
        expect(resStrided.shape, [2, 2]);

        // Complex throws UnsupportedError
        final cpx = NDArray.fromList([Complex(1, 2)], [1], DType.complex128);
        expect(() => copysign(cpx, cpx), throwsUnsupportedError);
      });
    });

    test("gcd and lcm on integer DTypes and edge cases", () {
      NDArray.scope(() {
        final intDTypes = <DType<AnyIntegerSpec>>[
          DType.int64,
          DType.int32,
          DType.int16,
          DType.int8,
          DType.uint64,
          DType.uint32,
          DType.uint16,
          DType.uint8,
        ];

        for (final dtype in intDTypes) {
          final x1 = NDArray.fromList([12, 20, 0, 7], [4], dtype);
          final x2 = NDArray.fromList([18, 24, 5, 0], [4], dtype);

          final g = gcd(x1, x2);
          expect(g.toList(), [6, 4, 5, 7]);

          final l = lcm(x1, x2);
          expect(l.toList(), [36, 120, 0, 0]);

          // Strided gcd/lcm
          final x1_2d = NDArray.fromList(
            [12, 20, 8, 14],
            [2, 2],
            dtype,
          ).transpose([1, 0]);
          final x2_2d = NDArray.fromList([18, 24, 12, 21], [2, 2], dtype);
          final gT = gcd(x1_2d, x2_2d);
          expect(gT.shape, [2, 2]);
        }

        // Float & Complex gcd/lcm are rejected by IntegerDType / asIntegerDType
        final fArr = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(
          () => gcd(fArr.asIntegerDType, fArr.asIntegerDType),
          throwsArgumentError,
        );
        expect(
          () => lcm(fArr.asIntegerDType, fArr.asIntegerDType),
          throwsArgumentError,
        );

        final cArr = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
        expect(
          () => gcd(cArr.asIntegerDType, cArr.asIntegerDType),
          throwsArgumentError,
        );
        expect(
          () => lcm(cArr.asIntegerDType, cArr.asIntegerDType),
          throwsArgumentError,
        );
      });
    });

    test("logaddexp and logaddexp2 mathematical evaluations and limits", () {
      NDArray.scope(() {
        final x1 = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
        final x2 = NDArray.fromList([0.0, 2.0, 1.0], [3], DType.float64);

        final lae = logaddexp(x1, x2);
        expect(lae.dtype, DType.float64);
        expect(lae.toList()[0], closeTo(math.log(2.0), 1e-6));
        expect(
          lae.toList()[1],
          closeTo(math.log(math.exp(1.0) + math.exp(2.0)), 1e-6),
        );

        final lae2 = logaddexp2(x1, x2);
        expect(lae2.dtype, DType.float64);
        expect(lae2.toList()[0], closeTo(1.0, 1e-6)); // log2(2^0 + 2^0) = 1

        // -infinity edge cases
        final negInf = NDArray.fromList(
          [double.negativeInfinity],
          [1],
          DType.float64,
        );
        final laeInf = logaddexp(negInf, negInf);
        expect(laeInf.toList()[0], double.negativeInfinity);

        final lae2Inf = logaddexp2(negInf, negInf);
        expect(lae2Inf.toList()[0], double.negativeInfinity);

        // Float32 logaddexp & logaddexp2
        final x1f = NDArray.fromList([0.0, 1.0], [2], DType.float32);
        final x2f = NDArray.fromList([0.0, 1.0], [2], DType.float32);
        expect(logaddexp(x1f, x2f).dtype, DType.float32);
        expect(logaddexp2(x1f, x2f).dtype, DType.float32);

        // Integer inputs promote to float
        final x1i = NDArray.fromList([0, 1], [2], DType.int32);
        final x2i = NDArray.fromList([0, 1], [2], DType.int32);
        expect(logaddexp(x1i, x2i).dtype, DType.float64);
        expect(logaddexp2(x1i, x2i).dtype, DType.float64);

        // Complex throws UnsupportedError
        final cArr = NDArray.fromList([Complex(1, 0)], [1], DType.complex128);
        expect(() => logaddexp(cArr, cArr), throwsUnsupportedError);
        expect(() => logaddexp2(cArr, cArr), throwsUnsupportedError);
      });
    });

    test("hypot and atan2 trigonometric operations across DTypes", () {
      NDArray.scope(() {
        // hypot: sqrt(x^2 + y^2)
        final x = NDArray.fromList([3.0, 5.0, 8.0], [3], DType.float64);
        final y = NDArray.fromList([4.0, 12.0, 15.0], [3], DType.float64);

        final h = hypot(x, y);
        expect(h.toList(), [5.0, 13.0, 17.0]);

        // Float32 hypot
        final xf = NDArray.fromList([3.0, 5.0], [2], DType.float32);
        final yf = NDArray.fromList([4.0, 12.0], [2], DType.float32);
        expect(hypot(xf, yf).toList(), [5.0, 13.0]);

        // Complex hypot throws UnsupportedError (matching NumPy)
        final c1 = NDArray.fromList([Complex(3, 4)], [1], DType.complex128);
        final c2 = NDArray.fromList([Complex(0, 0)], [1], DType.complex128);
        expect(() => hypot(c1, c2), throwsUnsupportedError);

        // atan2
        final at2 = atan2(y, x);
        expect(at2.dtype, DType.float64);
        expect(at2.toList()[0], closeTo(math.atan2(4.0, 3.0), 1e-6));

        // Complex atan2 throws UnsupportedError
        expect(() => atan2(c1, c2), throwsUnsupportedError);
      });
    });
  });

  group("Workstream 1: Unary Arithmetic Functions", () {
    test("sqrt, expm1, log1p across floats, integers, and complex", () {
      NDArray.scope(() {
        // sqrt
        final aF = NDArray.fromList([4.0, 9.0, 16.0, 25.0], [4], DType.float64);
        expect(sqrt(aF).toList(), [2.0, 3.0, 4.0, 5.0]);

        final aI = NDArray.fromList([4, 9, 16, 25], [4], DType.int32);
        expect(sqrt(aI).toList(), [2.0, 3.0, 4.0, 5.0]);

        final aC = NDArray.fromList(
          [Complex(-1, 0), Complex(0, 4)],
          [2],
          DType.complex128,
        );
        final sqrtC = sqrt(aC);
        expect(sqrtC.dtype, DType.complex128);
        expect(sqrtC.toList()[0].real, closeTo(0.0, 1e-6));
        expect(sqrtC.toList()[0].imag, closeTo(1.0, 1e-6));

        // expm1: exp(x) - 1
        final eArr = NDArray.fromList([0.0, 1.0, 1e-7], [3], DType.float64);
        final expm1Res = expm1(eArr);
        expect(expm1Res.toList()[0], closeTo(0.0, 1e-6));
        expect(expm1Res.toList()[1], closeTo(math.exp(1.0) - 1.0, 1e-6));
        expect(expm1Res.toList()[2], closeTo(1e-7, 1e-12));

        // log1p: ln(1 + x)
        final lArr = NDArray.fromList(
          [0.0, math.e - 1.0, 1e-7],
          [3],
          DType.float64,
        );
        final log1pRes = log1p(lArr);
        expect(log1pRes.toList()[0], closeTo(0.0, 1e-6));
        expect(log1pRes.toList()[1], closeTo(1.0, 1e-6));
        expect(log1pRes.toList()[2], closeTo(1e-7, 1e-12));

        // Complex expm1 & log1p
        final cZero = NDArray.fromList([Complex(0, 0)], [1], DType.complex128);
        expect(expm1(cZero).toList()[0].real, closeTo(0.0, 1e-6));
        expect(expm1(cZero).toList()[0].imag, closeTo(0.0, 1e-6));
        expect(log1p(cZero).toList()[0].real, closeTo(0.0, 1e-6));
        expect(log1p(cZero).toList()[0].imag, closeTo(0.0, 1e-6));
      });
    });

    test("rint, trunc, fix, square, reciprocal, positive, negative", () {
      NDArray.scope(() {
        final nums = NDArray.fromList(
          [-2.7, -2.5, -2.3, 2.3, 2.5, 2.7],
          [6],
          DType.float64,
        );

        // rint: round to nearest integer
        expect(rint(nums).toList(), [-3.0, -2.0, -2.0, 2.0, 2.0, 3.0]);

        // trunc: truncate towards zero
        expect(trunc(nums).toList(), [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]);

        // fix: round towards zero (same as trunc)
        expect(fix(nums).toList(), [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]);

        // square
        final sqArr = NDArray.fromList([2.0, 3.0, -4.0], [3], DType.float64);
        expect(square(sqArr).toList(), [4.0, 9.0, 16.0]);

        // reciprocal: 1 / x
        final recArr = NDArray.fromList([2.0, 4.0, 8.0], [3], DType.float64);
        expect(reciprocal(recArr).toList(), [0.5, 0.25, 0.125]);

        // positive: +x
        expect(positive(nums).toList(), nums.toList());

        // negative: -x
        expect(negative(sqArr).toList(), [-2.0, -3.0, 4.0]);
      });
    });

    test("abs, sign, ceil, floor, round across DTypes", () {
      NDArray.scope(() {
        final fArr = NDArray.fromList([-3.7, 0.0, 4.2], [3], DType.float64);

        expect(abs(fArr).toList(), [3.7, 0.0, 4.2]);
        expect(sign(fArr).toList(), [-1.0, 0.0, 1.0]);
        expect(ceil(fArr).toList(), [-3.0, 0.0, 5.0]);
        expect(floor(fArr).toList(), [-4.0, 0.0, 4.0]);
        expect(round(fArr).toList(), [-4.0, 0.0, 4.0]);

        // Complex abs & sign
        final cArr = NDArray.fromList(
          [Complex(3, 4), Complex(-3, 4)],
          [2],
          DType.complex128,
        );
        expect(abs(cArr).toList(), [5.0, 5.0]);
        final cSign = sign(cArr);
        expect(cSign.toList()[0].real, closeTo(0.6, 1e-6));
        expect(cSign.toList()[0].imag, closeTo(0.8, 1e-6));

        // Integer abs & sign
        final iArr = NDArray.fromList([-10, 0, 10], [3], DType.int32);
        expect(abs(iArr).toList(), [10, 0, 10]);
        expect(sign(iArr).toList(), [-1, 0, 1]);
      });
    });
  });

  group("Workstream 1: Floating Point Classification and Comparisons", () {
    test(
      "isnan, isinf, isfinite across float64, float32, float16, complex, and int dtypes",
      () {
        NDArray.scope(() {
          final floatVals = [
            1.0,
            double.nan,
            double.infinity,
            double.negativeInfinity,
            0.0,
            -0.0,
          ];

          // Float64
          final f64 = NDArray.fromList(floatVals, [6], DType.float64);
          expect(isnan(f64).toList(), [
            false,
            true,
            false,
            false,
            false,
            false,
          ]);
          expect(isinf(f64).toList(), [false, false, true, true, false, false]);
          expect(isfinite(f64).toList(), [
            true,
            false,
            false,
            false,
            true,
            true,
          ]);

          // Float32
          final f32 = NDArray.fromList(floatVals, [6], DType.float32);
          expect(isnan(f32).toList(), [
            false,
            true,
            false,
            false,
            false,
            false,
          ]);
          expect(isinf(f32).toList(), [false, false, true, true, false, false]);
          expect(isfinite(f32).toList(), [
            true,
            false,
            false,
            false,
            true,
            true,
          ]);

          // Float16 & BFloat16
          final f16 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float16);
          expect(isfinite(f16).toList(), [true, true, true]);

          final bf16 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.bfloat16);
          expect(isfinite(bf16).toList(), [true, true, true]);

          // Complex128 & Complex64
          final cVals = [
            Complex(1.0, 2.0),
            Complex(double.nan, 0.0),
            Complex(0.0, double.infinity),
            Complex(double.negativeInfinity, double.nan),
          ];
          final c128 = NDArray.fromList(cVals, [4], DType.complex128);
          expect(isnan(c128).toList(), [false, true, false, true]);
          expect(isinf(c128).toList(), [false, false, true, true]);
          expect(isfinite(c128).toList(), [true, false, false, false]);

          final c64 = NDArray.fromList(cVals, [4], DType.complex64);
          expect(isnan(c64).toList(), [false, true, false, true]);
          expect(isinf(c64).toList(), [false, false, true, true]);
          expect(isfinite(c64).toList(), [true, false, false, false]);

          // Integer arrays are always not nan, not inf, and finite
          final i32 = NDArray.fromList([10, 20, -30], [3], DType.int32);
          expect(isnan(i32).toList(), [false, false, false]);
          expect(isinf(i32).toList(), [false, false, false]);
          expect(isfinite(i32).toList(), [true, true, true]);

          final u8 = NDArray.fromList([0, 100, 255], [3], DType.uint8);
          expect(isnan(u8).toList(), [false, false, false]);
          expect(isinf(u8).toList(), [false, false, false]);
          expect(isfinite(u8).toList(), [true, true, true]);

          // Strided views
          final f64_2d = NDArray.fromList(floatVals, [
            2,
            3,
          ], DType.float64).transpose([1, 0]);
          expect(f64_2d.isContiguous, false);
          expect(isnan(f64_2d).shape, [3, 2]);
          expect(isinf(f64_2d).shape, [3, 2]);
          expect(isfinite(f64_2d).shape, [3, 2]);

          // out parameter & where mask
          final outBool = NDArray.zeros([6], DType.boolean);
          final mask = NDArray.fromList(
            [true, true, false, false, true, true],
            [6],
            DType.boolean,
          );
          isnan(f64, where: mask, out: outBool);
          expect(outBool.toList(), [false, true, false, false, false, false]);
        });
      },
    );

    test("isClose and allClose approximate equality evaluations", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 1.00001, double.nan, double.infinity],
          [4],
          DType.float64,
        );
        final b = NDArray.fromList(
          [1.0, 1.00002, double.nan, double.infinity],
          [4],
          DType.float64,
        );

        final closeDefault = isClose(a, b);
        expect(closeDefault.toList(), [true, true, false, true]);

        final closeEqualNan = isClose(a, b, equalNan: true);
        expect(closeEqualNan.toList(), [true, true, true, true]);

        expect(allClose(a, b, equalNan: true), true);
        expect(allClose(a, b, equalNan: false), false);

        // Complex isClose
        final ca = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        final cb = NDArray.fromList(
          [Complex(1.000001, 2.000001), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        expect(isClose(ca, cb, atol: 1e-4).toList(), [true, true]);
        expect(allClose(ca, cb, atol: 1e-4), true);

        // Mixed types (real and complex)
        final ra = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final rc = NDArray.fromList(
          [Complex(1.0, 0.0), Complex(2.0, 0.0)],
          [2],
          DType.complex128,
        );
        expect(isClose(ra, rc).toList(), [true, true]);
        expect(allClose(ra, rc), true);

        // Broadcasting in isClose
        final broadA = NDArray.fromList([1.0, 2.0], [2, 1], DType.float64);
        final broadB = NDArray.fromList([1.0, 1.0], [1, 2], DType.float64);
        final resBroad = isClose(broadA, broadB);
        expect(resBroad.shape, [2, 2]);
        expect(resBroad.toList(), [true, true, false, false]);
      });
    });
  });

  group("Workstream 1: Helpers Module Full Coverage (helpers.dart)", () {
    test("resolveDType comprehensive 15x15 combination matrix", () {
      for (final a in allDTypes) {
        for (final b in allDTypes) {
          final res = resolveDType(a, b);
          expect(res, isA<DType>());

          // Commutativity: resolveDType(a, b) should match resolveDType(b, a)
          final resCommutative = resolveDType(b, a);
          expect(
            res,
            resCommutative,
            reason: "resolveDType should be commutative for $a and $b",
          );
        }
      }

      // Specific promotion assertions
      expect(resolveDType(DType.boolean, DType.boolean), DType.uint8);
      expect(resolveDType(DType.boolean, DType.float64), DType.float64);
      expect(resolveDType(DType.float32, DType.int64), DType.float64);
      expect(resolveDType(DType.float16, DType.bfloat16), DType.float32);
      expect(resolveDType(DType.int32, DType.uint32), DType.int64);
      expect(resolveDType(DType.int16, DType.uint16), DType.int32);
      expect(resolveDType(DType.int8, DType.uint8), DType.int16);
      expect(resolveDType(DType.complex64, DType.float64), DType.complex128);
      expect(resolveDType(DType.complex64, DType.float32), DType.complex64);
    });

    test("normalizeScalar across all 15 DTypes with num, Complex, bool", () {
      for (final dtype in allDTypes) {
        final normNum = normalizeScalar(42, dtype);
        final normDouble = normalizeScalar(42.5, dtype);
        final normComplex = normalizeScalar(Complex(1.0, 2.0), dtype);
        final normBool = normalizeScalar(true, dtype);

        expect(normNum, isNotNull);
        expect(normDouble, isNotNull);
        expect(normComplex, isNotNull);
        expect(normBool, isNotNull);
      }
    });

    test("toNDArray conversion from scalar and existing NDArray", () {
      NDArray.scope(() {
        final arr = NDArray.fromList([1, 2, 3], [3], DType.int32);

        // Same dtype returns existing
        final same = toNDArray(arr, DType.int32);
        expect(identical(same, arr), true);

        // Different dtype casts
        final casted = toNDArray(arr, DType.float64);
        expect(casted.dtype, DType.float64);
        expect(casted.toList(), [1.0, 2.0, 3.0]);

        // Scalar conversion
        final scalarArr = toNDArray(5.5, DType.float64);
        expect(scalarArr.shape, <int>[]);
        expect(scalarArr.toList(), [5.5]);

        // Disposed throws StateError
        final disposedArr = NDArray.fromList([1], [1], DType.int32);
        disposedArr.dispose();
        expect(() => toNDArray(disposedArr, DType.int32), throwsStateError);
      });
    });

    test(
      "linspaceInternal across numeric/complex DTypes, endpoint, 0 samples, and boolean error",
      () {
        NDArray.scope(() {
          // 0 samples
          final zeroRes = linspaceInternal(0.0, 10.0, 0, dtype: DType.float64);
          expect(zeroRes.samples.shape, [0]);
          expect(zeroRes.step.isNaN, true);

          // Float64 endpoint true/false
          final lsF64 = linspaceInternal(
            0.0,
            10.0,
            5,
            endpoint: true,
            dtype: DType.float64,
          );
          expect(lsF64.samples.toList(), [0.0, 2.5, 5.0, 7.5, 10.0]);
          expect(lsF64.step, 2.5);

          final lsF64NoEnd = linspaceInternal(
            0.0,
            10.0,
            5,
            endpoint: false,
            dtype: DType.float64,
          );
          expect(lsF64NoEnd.samples.toList(), [0.0, 2.0, 4.0, 6.0, 8.0]);
          expect(lsF64NoEnd.step, 2.0);

          // Float32
          final lsF32 = linspaceInternal(0.0, 4.0, 5, dtype: DType.float32);
          expect(lsF32.samples.dtype, DType.float32);
          expect(lsF32.samples.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);

          // Complex128 & Complex64
          final lsC128 = linspaceInternal(
            Complex(0, 0),
            Complex(4, 8),
            5,
            dtype: DType.complex128,
          );
          expect(lsC128.samples.dtype, DType.complex128);
          expect(lsC128.samples.toList()[4].real, 4.0);
          expect(lsC128.samples.toList()[4].imag, 8.0);

          final lsC64 = linspaceInternal(
            Complex(0, 0),
            Complex(2, 4),
            3,
            dtype: DType.complex64,
          );
          expect(lsC64.samples.dtype, DType.complex64);

          // Integers: int64, int32, int16, uint8, float16, bfloat16, int8, uint64, uint32, uint16
          for (final dtype in [
            DType.int64,
            DType.int32,
            DType.int16,
            DType.uint8,
            DType.float16,
            DType.bfloat16,
            DType.int8,
            DType.uint64,
            DType.uint32,
            DType.uint16,
          ]) {
            final ls = linspaceInternal(0, 10, 5, dtype: dtype);
            expect(ls.samples.shape, [5]);
          }

          // Out buffer
          final outBuf = NDArray<Float64>.create([5], DType.float64);
          final lsOut = linspaceInternal(
            0.0,
            10.0,
            5,
            dtype: DType.float64,
            out: outBuf,
          );
          expect(identical(lsOut.samples, outBuf), true);

          // Disposed out buffer throws StateError
          final disposedOut = NDArray<Float64>.create([5], DType.float64);
          disposedOut.dispose();
          expect(
            () => linspaceInternal(
              0.0,
              10.0,
              5,
              dtype: DType.float64,
              out: disposedOut,
            ),
            throwsStateError,
          );

          // Negative numSamples throws ArgumentError
          expect(
            () => linspaceInternal(0.0, 10.0, -1, dtype: DType.float64),
            throwsArgumentError,
          );

          // Boolean throws UnsupportedError
          expect(
            () => linspaceInternal(false, true, 5, dtype: DType.boolean),
            throwsUnsupportedError,
          );
        });
      },
    );

    test("cumOpFFI for sum, prod, min, max across all 15 DTypes", () {
      NDArray.scope(() {
        for (final dtype in allDTypes) {
          final a = createArray([2, 3], dtype, seedOffset: 1);
          final resSum = NDArray.create([2, 3], dtype);
          cumOpFFI(a, 1, resSum, CumOpType.sum);
          expect(resSum.shape, [2, 3]);

          final resProd = NDArray.create([2, 3], dtype);
          cumOpFFI(a, 0, resProd, CumOpType.prod);
          expect(resProd.shape, [2, 3]);

          if (!dtype.isComplex) {
            final resMin = NDArray.create([2, 3], dtype);
            cumOpFFI(a, 1, resMin, CumOpType.min);
            expect(resMin.shape, [2, 3]);

            final resMax = NDArray.create([2, 3], dtype);
            cumOpFFI(a, 0, resMax, CumOpType.max);
            expect(resMax.shape, [2, 3]);
          } else {
            // Complex min and max throw ArgumentError
            final resMin = NDArray.create([2, 3], dtype);
            expect(
              () => cumOpFFI(a, 1, resMin, CumOpType.min),
              throwsArgumentError,
            );
            expect(
              () => cumOpFFI(a, 0, resMin, CumOpType.max),
              throwsArgumentError,
            );
          }
        }
      });
    });

    test("castNDArray full conversion matrix across DTypes", () {
      NDArray.scope(() {
        for (final srcType in allDTypes) {
          final src = createArray([2, 2], srcType, seedOffset: 1);
          for (final dstType in allDTypes) {
            final converted = castNDArray(src, dstType);
            expect(converted.dtype, dstType);
            expect(converted.shape, [2, 2]);
          }
        }
      });
    });

    test("MaskHolder and prepareMask validation and memory management", () {
      NDArray.scope(() {
        // null mask
        final nullHolder = prepareMask(null, [2, 3]);
        expect(nullHolder.pointer, ffi.nullptr);
        nullHolder.dispose();

        // boolean contiguous mask
        final boolMask = NDArray.fromList(
          [true, false, true, false],
          [4],
          DType.boolean,
        );
        final boolHolder = prepareMask(boolMask, [4]);
        expect(boolHolder.pointer, isNot(ffi.nullptr));
        boolHolder.dispose();

        // uint8 mask
        final u8Mask = NDArray.fromList([1, 0, 1, 0], [4], DType.uint8);
        final u8Holder = prepareMask(u8Mask, [4]);
        expect(u8Holder.pointer, isNot(ffi.nullptr));
        u8Holder.dispose();

        // strided mask needing broadcast or copy
        final stridedMask = NDArray.fromList(
          [true, false, true, false],
          [2, 2],
          DType.boolean,
        ).transpose([1, 0]);
        final stridedHolder = prepareMask(stridedMask, [2, 2]);
        expect(stridedHolder.pointer, isNot(ffi.nullptr));
        stridedHolder.dispose();

        // invalid mask dtype throws ArgumentError
        final invalidMask = NDArray.fromList([1.0, 0.0], [2], DType.float64);
        expect(() => prepareMask(invalidMask, [2]), throwsArgumentError);

        // disposed mask throws StateError
        final disposedMask = NDArray.fromList([true], [1], DType.boolean);
        disposedMask.dispose();
        expect(() => prepareMask(disposedMask, [1]), throwsStateError);
      });
    });

    test("encodeDType, mapSortKind, promoteToDouble, promoteToComplex", () {
      NDArray.scope(() {
        for (final dtype in allDTypes) {
          final code = encodeDType(dtype);
          expect(code, inInclusiveRange(0, 14));
        }

        expect(mapSortKind(SortKind.quicksort), 0);
        expect(mapSortKind(SortKind.mergesort), 1);
        expect(mapSortKind(SortKind.stable), 1);
        expect(mapSortKind(SortKind.heapsort), 2);

        final iArr = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final dbl = promoteToDouble(iArr);
        expect(dbl.dtype, DType.float64);
        expect(dbl.toList(), [1.0, 2.0, 3.0, 4.0]);

        final cpx = promoteToComplex(iArr);
        expect(cpx.dtype, DType.complex128);
        expect(cpx.toList()[0], Complex(1.0, 0.0));

        // Disposed checks
        final disp = NDArray.fromList([1], [1], DType.int32);
        disp.dispose();
        expect(() => promoteToDouble(disp), throwsStateError);
        expect(() => promoteToComplex(disp), throwsStateError);
      });
    });

    test(
      "broadcastStackShapes, broadcast3Shapes, broadcastStrides error branches",
      () {
        // broadcastStackShapes
        expect(broadcastStackShapes([2, 1], [1, 3]), [2, 3]);
        expect(broadcastStackShapes([5], [2, 5]), [2, 5]);
        expect(() => broadcastStackShapes([2], [3]), throwsArgumentError);

        // broadcast3Shapes
        expect(broadcast3Shapes([2, 1, 1], [1, 3, 1], [1, 1, 4]), [2, 3, 4]);
        expect(() => broadcast3Shapes([2], [3], [2]), throwsArgumentError);

        // broadcastStrides
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2, 1], DType.int32);
          final strides = broadcastStrides(a, [2, 3]);
          expect(strides, [1, 0]);
          expect(() => broadcastStrides(a, [3, 3]), throwsArgumentError);
        });
      },
    );

    test("isTrueHelper and castValue functions", () {
      expect(isTrueHelper(true), true);
      expect(isTrueHelper(false), false);
      expect(isTrueHelper(1), true);
      expect(isTrueHelper(0), false);
      expect(isTrueHelper(Complex(1, 0)), true);
      expect(isTrueHelper(Complex(0, 0)), false);
      expect(isTrueHelper("unknown"), false);

      for (final dtype in allDTypes) {
        expect(castValue(10, dtype), isNotNull);
        expect(castValue(Complex(1, 2), dtype), isNotNull);
        expect(castValue(true, dtype), isNotNull);
      }
    });

    test("elementWiseOp, unaryOp, ternaryOp, and recursive reductions", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final c = NDArray.fromList(
          [100.0, 200.0, 300.0, 400.0],
          [2, 2],
          DType.float64,
        );
        final res = NDArray<Float64>.zeros([2, 2], DType.float64);

        // unaryOp
        unaryOp<DTypeTag, DTypeTag>(
          res,
          a,
          [2, 2],
          a.strides,
          res.strides,
          0,
          a.offsetElements,
          res.offsetElements,
          (x) => (x as double) * 2.0,
        );
        expect(res.toList(), [2.0, 4.0, 6.0, 8.0]);

        // elementWiseOp
        elementWiseOp<DTypeTag, DTypeTag, DTypeTag>(
          res,
          a,
          b,
          [2, 2],
          a.strides,
          b.strides,
          res.strides,
          0,
          a.offsetElements,
          b.offsetElements,
          res.offsetElements,
          (x, y) => x + y,
        );
        expect(res.toList(), [11.0, 22.0, 33.0, 44.0]);

        // ternaryOp
        ternaryOp<DTypeTag, DTypeTag, DTypeTag, DTypeTag>(
          res,
          a,
          b,
          c,
          [2, 2],
          a.strides,
          b.strides,
          c.strides,
          res.strides,
          0,
          a.offsetElements,
          b.offsetElements,
          c.offsetElements,
          res.offsetElements,
          (x, y, z) => x + y + z,
        );
        expect(res.toList(), [111.0, 222.0, 333.0, 444.0]);

        // walkStackCoords
        final visited = <List<int>>[];
        walkStackCoords([2, 2], List.filled(2, 0), 0, (coords) {
          visited.add(List.from(coords));
        });
        expect(visited.length, 4);
        expect(visited, [
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ]);

        // reduceRecursive
        final src = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final dest = NDArray<Float64>.zeros([2], DType.float64);
        reduceRecursive<DTypeTag, DTypeTag>(
          src,
          dest,
          List.filled(2, 0),
          List.filled(1, 0),
          1,
          0,
          (acc, val) => acc + val,
        );
        expect(dest.toList(), [3.0, 7.0]);

        // nanReduceRecursive
        final nanSrc = NDArray.fromList(
          [1.0, double.nan, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final nanDest = NDArray<Float64>.zeros([2], DType.float64);
        final counts = NDArray<Int64>.zeros([2], DType.int64);
        nanReduceRecursive<DTypeTag>(
          nanSrc,
          nanDest,
          counts,
          List.filled(2, 0),
          List.filled(1, 0),
          1,
          0,
        );
        expect(nanDest.toList(), [1.0, 7.0]);
        expect(counts.toList(), [1, 2]);
      });
    });
  });
}
