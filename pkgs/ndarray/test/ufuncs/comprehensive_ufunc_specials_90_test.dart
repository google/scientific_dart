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
  group('Comprehensive Ufunc Methods, Bitwise, Logic & Specials Coverage (90%+)', () {
    // =========================================================================
    // 1. Universal Function Methods: BinaryOp Universal Methods
    // =========================================================================
    group('BinaryOp Universal Methods (reduce, accumulate, reduceat, outer, at)', () {
      test(
        'BinaryOp.reduce on all reducible operations across multiple DTypes',
        () {
          NDArray.scope(() {
            // minimum and maximum
            final f64 = NDArray<Float64>.fromList(
              [5.0, 2.0, 8.0, 1.0, 9.0],
              [5],
              DType.float64,
            );
            expect(f64.reduce(op: BinaryOp.minimum).scalar, equals(1.0));
            expect(f64.reduce(op: BinaryOp.maximum).scalar, equals(9.0));
            expect(f64.reduce(op: BinaryOp.fmin).scalar, equals(1.0));
            expect(f64.reduce(op: BinaryOp.fmax).scalar, equals(9.0));

            // Float32
            final f32 = NDArray<Float32>.fromList(
              [5.0, 2.0, 8.0, 1.0, 9.0],
              [5],
              DType.float32,
            );
            expect(f32.reduce(op: BinaryOp.minimum).scalar, equals(1.0));
            expect(f32.reduce(op: BinaryOp.maximum).scalar, equals(9.0));

            // Int64 and Int32
            final i64 = NDArray<Int64>.fromList(
              [15, 3, 27, 9],
              [4],
              DType.int64,
            );
            expect(i64.reduce(op: BinaryOp.minimum).scalar, equals(3));
            expect(i64.reduce(op: BinaryOp.maximum).scalar, equals(27));
            expect(i64.reduce(op: BinaryOp.gcd).scalar, equals(3));
            expect(i64.reduce(op: BinaryOp.lcm).scalar, equals(135));

            final i32 = NDArray<Int32>.fromList([12, 18, 24], [3], DType.int32);
            expect(i32.reduce(op: BinaryOp.gcd).scalar, equals(6));
            expect(i32.reduce(op: BinaryOp.lcm).scalar, equals(72));

            // Int16 and Uint8
            final i16 = NDArray<Int16>.fromList([10, 20, 5], [3], DType.int16);
            expect(i16.reduce(op: BinaryOp.minimum).scalar, equals(5));
            expect(i16.reduce(op: BinaryOp.maximum).scalar, equals(20));

            final u8 = NDArray<Uint8>.fromList(
              [100, 250, 50],
              [3],
              DType.uint8,
            );
            expect(u8.reduce(op: BinaryOp.minimum).scalar, equals(50));
            expect(u8.reduce(op: BinaryOp.maximum).scalar, equals(250));

            // Bitwise Reductions
            final bitI64 = NDArray<Int64>.fromList(
              [0xFF, 0x0F, 0x3F],
              [3],
              DType.int64,
            );
            expect(bitI64.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(0x0F));
            expect(bitI64.reduce(op: BinaryOp.bitwiseOr).scalar, equals(0xFF));
            expect(bitI64.reduce(op: BinaryOp.bitwiseXor).scalar, equals(0xCF));

            final bitI32 = NDArray<Int32>.fromList(
              [0xAA, 0x55],
              [2],
              DType.int32,
            );
            expect(bitI32.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(0));
            expect(bitI32.reduce(op: BinaryOp.bitwiseOr).scalar, equals(0xFF));
            expect(bitI32.reduce(op: BinaryOp.bitwiseXor).scalar, equals(0xFF));

            final bitI16 = NDArray<Int16>.fromList(
              [0xF0, 0x0F],
              [2],
              DType.int16,
            );
            expect(bitI16.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(0));
            expect(bitI16.reduce(op: BinaryOp.bitwiseOr).scalar, equals(0xFF));
            expect(bitI16.reduce(op: BinaryOp.bitwiseXor).scalar, equals(0xFF));

            final bitU8 = NDArray<Uint8>.fromList(
              [0xF0, 0x0F],
              [2],
              DType.uint8,
            );
            expect(bitU8.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(0));
            expect(bitU8.reduce(op: BinaryOp.bitwiseOr).scalar, equals(0xFF));
            expect(bitU8.reduce(op: BinaryOp.bitwiseXor).scalar, equals(0xFF));

            // Logical Reductions
            final boolsT = NDArray<Boolean>.fromList(
              [true, true, true],
              [3],
              DType.boolean,
            );
            expect(boolsT.reduce(op: BinaryOp.logicalAnd).scalar, isTrue);
            expect(boolsT.reduce(op: BinaryOp.logicalOr).scalar, isTrue);
            expect(boolsT.reduce(op: BinaryOp.logicalXor).scalar, isTrue);

            final boolsM = NDArray<Boolean>.fromList(
              [true, false, false],
              [3],
              DType.boolean,
            );
            expect(boolsM.reduce(op: BinaryOp.logicalAnd).scalar, isFalse);
            expect(boolsM.reduce(op: BinaryOp.logicalOr).scalar, isTrue);
            expect(boolsM.reduce(op: BinaryOp.logicalXor).scalar, isTrue);

            // logaddexp & logaddexp2
            final logA = NDArray<Float64>.fromList(
              [0.0, 1.0],
              [2],
              DType.float64,
            );
            expect(
              logA.reduce(op: BinaryOp.logaddexp).scalar,
              closeTo(math.log(1.0 + math.exp(1.0)), 1e-6),
            );
            expect(
              logA.reduce(op: BinaryOp.logaddexp2).scalar,
              closeTo(math.log(1.0 + 2.0) / math.ln2, 1e-6),
            );

            // Reduce with initial value and keepdims
            final redInit = f64.reduce(
              op: BinaryOp.minimum,
              initial: 0.5,
              keepdims: true,
            );
            expect(redInit.shape, equals([1]));
            expect(redInit.getCell([0]), equals(0.5));

            // Reduce on empty array with initial
            final emptyArr = NDArray<Float64>.zeros([0], DType.float64);
            expect(
              emptyArr.reduce(op: BinaryOp.add, initial: 42.0).scalar,
              equals(42.0),
            );
            expect(
              () => emptyArr.reduce(op: BinaryOp.add),
              throwsArgumentError,
            );

            // Disposed array throws StateError
            final dispArr = NDArray<Float64>.zeros([3], DType.float64);
            dispArr.dispose();
            expect(() => dispArr.reduce(op: BinaryOp.add), throwsStateError);

            // Non-reducible op throws ArgumentError
            final normArr = NDArray<Float64>.zeros([3], DType.float64);
            expect(
              () => normArr.reduce(op: BinaryOp.subtract),
              throwsArgumentError,
            );
          });
        },
      );

      test(
        'BinaryOp.reduce along axes on 2D and 3D tensors for all reducible ops',
        () {
          NDArray.scope(() {
            // 2D Matrix bitwise reductions along axis 0 and 1
            final matInt = NDArray<Int32>.fromList(
              [0xF0, 0x0F, 0xFF, 0xAA, 0x55, 0x00],
              [2, 3],
              DType.int32,
            );

            final andAxis0 = matInt.reduce(op: BinaryOp.bitwiseAnd, axis: 0);
            expect(andAxis0.shape, equals([3]));
            expect(andAxis0.toList(), equals([0xF0 & 0xAA, 0x0F & 0x55, 0]));

            final orAxis1 = matInt.reduce(op: BinaryOp.bitwiseOr, axis: 1);
            expect(orAxis1.shape, equals([2]));
            expect(orAxis1.toList(), equals([0xFF, 0xFF]));

            final xorAxis0 = matInt.reduce(op: BinaryOp.bitwiseXor, axis: 0);
            expect(xorAxis0.toList(), equals([0xF0 ^ 0xAA, 0x0F ^ 0x55, 0xFF]));

            // Int64 2D axis reductions
            final matI64 = NDArray<Int64>.fromList(
              [10, 20, 30, 5, 15, 25],
              [2, 3],
              DType.int64,
            );
            expect(
              matI64.reduce(op: BinaryOp.minimum, axis: 0).toList(),
              equals([5, 15, 25]),
            );
            expect(
              matI64.reduce(op: BinaryOp.maximum, axis: 1).toList(),
              equals([30, 25]),
            );
            expect(
              matI64.reduce(op: BinaryOp.bitwiseAnd, axis: 0).shape,
              equals([3]),
            );
            expect(
              matI64.reduce(op: BinaryOp.bitwiseOr, axis: 1).shape,
              equals([2]),
            );
            expect(
              matI64.reduce(op: BinaryOp.bitwiseXor, axis: 0).shape,
              equals([3]),
            );

            // Int16 2D axis reductions
            final matI16 = NDArray<Int16>.fromList(
              [10, 20, 5, 15],
              [2, 2],
              DType.int16,
            );
            expect(
              matI16.reduce(op: BinaryOp.minimum, axis: 0).toList(),
              equals([5, 15]),
            );
            expect(
              matI16.reduce(op: BinaryOp.maximum, axis: 1).toList(),
              equals([20, 15]),
            );
            expect(
              matI16.reduce(op: BinaryOp.bitwiseAnd, axis: 0).toList(),
              equals([0, 4]),
            );
            expect(
              matI16.reduce(op: BinaryOp.bitwiseOr, axis: 1).toList(),
              equals([30, 15]),
            );
            expect(
              matI16.reduce(op: BinaryOp.bitwiseXor, axis: 0).toList(),
              equals([15, 27]),
            );

            // Uint8 2D axis reductions
            final matU8 = NDArray<Uint8>.fromList(
              [10, 20, 5, 15],
              [2, 2],
              DType.uint8,
            );
            expect(
              matU8.reduce(op: BinaryOp.minimum, axis: 0).toList(),
              equals([5, 15]),
            );
            expect(
              matU8.reduce(op: BinaryOp.maximum, axis: 1).toList(),
              equals([20, 15]),
            );
            expect(
              matU8.reduce(op: BinaryOp.bitwiseAnd, axis: 0).toList(),
              equals([0, 4]),
            );
            expect(
              matU8.reduce(op: BinaryOp.bitwiseOr, axis: 1).toList(),
              equals([30, 15]),
            );
            expect(
              matU8.reduce(op: BinaryOp.bitwiseXor, axis: 0).toList(),
              equals([15, 27]),
            );

            // Float32 2D axis reductions
            final matF32 = NDArray<Float32>.fromList(
              [10.0, 20.0, 5.0, 15.0],
              [2, 2],
              DType.float32,
            );
            expect(
              matF32.reduce(op: BinaryOp.minimum, axis: 0).toList(),
              equals([5.0, 15.0]),
            );
            expect(
              matF32.reduce(op: BinaryOp.maximum, axis: 1).toList(),
              equals([20.0, 15.0]),
            );

            // Boolean 2D axis reductions
            final matBool = NDArray<Boolean>.fromList(
              [true, false, true, true],
              [2, 2],
              DType.boolean,
            );
            expect(
              matBool.reduce(op: BinaryOp.logicalAnd, axis: 0).toList(),
              equals([true, false]),
            );
            expect(
              matBool.reduce(op: BinaryOp.logicalOr, axis: 1).toList(),
              equals([true, true]),
            );
            expect(
              matBool.reduce(op: BinaryOp.logicalXor, axis: 0).toList(),
              equals([false, true]),
            );

            // Complex 2D axis reductions
            final matC128 = NDArray<Complex128>.fromList(
              [
                Complex(1.0, 1.0),
                Complex(2.0, 0.0),
                Complex(0.0, 1.0),
                Complex(3.0, 2.0),
              ],
              [2, 2],
              DType.complex128,
            );
            final cProd = matC128.reduce(op: BinaryOp.multiply, axis: 0);
            expect(cProd.shape, equals([2]));
            expect(
              cProd.getCell([0]),
              equals(Complex(1.0, 1.0) * Complex(0.0, 1.0)),
            );

            final matC64 = NDArray<Complex64>.fromList(
              [
                Complex(1.0, 1.0),
                Complex(2.0, 0.0),
                Complex(0.0, 1.0),
                Complex(3.0, 2.0),
              ],
              [2, 2],
              DType.complex64,
            );
            final c64Prod = matC64.reduce(op: BinaryOp.multiply, axis: 0);
            expect(c64Prod.shape, equals([2]));

            // Out buffer reuse in axis reduction
            final outRed = NDArray<Float64>.zeros([2], DType.float64);
            final matF64 = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final resRed = matF64.reduce(
              op: BinaryOp.add,
              axis: 1,
              out: outRed,
            );
            expect(identical(resRed, outRed), isTrue);
            expect(outRed.toList(), equals([3.0, 7.0]));
          });
        },
      );

      test(
        'BinaryOp.accumulate across all reducible ops and multiple axes',
        () {
          NDArray.scope(() {
            // 1D accumulates
            final aF64 = NDArray<Float64>.fromList(
              [4.0, 2.0, 5.0, 1.0],
              [4],
              DType.float64,
            );
            expect(
              aF64.accumulate(op: BinaryOp.minimum).toList(),
              equals([4.0, 2.0, 2.0, 1.0]),
            );
            expect(
              aF64.accumulate(op: BinaryOp.maximum).toList(),
              equals([4.0, 4.0, 5.0, 5.0]),
            );

            final aF32 = NDArray<Float32>.fromList(
              [4.0, 2.0, 5.0, 1.0],
              [4],
              DType.float32,
            );
            expect(
              aF32.accumulate(op: BinaryOp.minimum).toList(),
              equals([4.0, 2.0, 2.0, 1.0]),
            );
            expect(
              aF32.accumulate(op: BinaryOp.maximum).toList(),
              equals([4.0, 4.0, 5.0, 5.0]),
            );

            final aI64 = NDArray<Int64>.fromList(
              [4, 2, 5, 1],
              [4],
              DType.int64,
            );
            expect(
              aI64.accumulate(op: BinaryOp.minimum).toList(),
              equals([4, 2, 2, 1]),
            );
            expect(
              aI64.accumulate(op: BinaryOp.maximum).toList(),
              equals([4, 4, 5, 5]),
            );
            expect(
              aI64.accumulate(op: BinaryOp.bitwiseAnd).toList(),
              equals([4, 0, 0, 0]),
            );
            expect(
              aI64.accumulate(op: BinaryOp.bitwiseOr).toList(),
              equals([4, 6, 7, 7]),
            );
            expect(
              aI64.accumulate(op: BinaryOp.bitwiseXor).toList(),
              equals([4, 6, 3, 2]),
            );

            final aI32 = NDArray<Int32>.fromList(
              [4, 2, 5, 1],
              [4],
              DType.int32,
            );
            expect(
              aI32.accumulate(op: BinaryOp.minimum).toList(),
              equals([4, 2, 2, 1]),
            );
            expect(
              aI32.accumulate(op: BinaryOp.maximum).toList(),
              equals([4, 4, 5, 5]),
            );
            expect(
              aI32.accumulate(op: BinaryOp.bitwiseAnd).toList(),
              equals([4, 0, 0, 0]),
            );
            expect(
              aI32.accumulate(op: BinaryOp.bitwiseOr).toList(),
              equals([4, 6, 7, 7]),
            );
            expect(
              aI32.accumulate(op: BinaryOp.bitwiseXor).toList(),
              equals([4, 6, 3, 2]),
            );

            final aI16 = NDArray<Int16>.fromList(
              [0xFF, 0x0F, 0x30],
              [3],
              DType.int16,
            );
            expect(
              aI16.accumulate(op: BinaryOp.bitwiseAnd).toList(),
              equals([0xFF, 0x0F, 0x00]),
            );
            expect(
              aI16.accumulate(op: BinaryOp.bitwiseOr).toList(),
              equals([0xFF, 0xFF, 0xFF]),
            );
            expect(
              aI16.accumulate(op: BinaryOp.bitwiseXor).toList(),
              equals([0xFF, 0xF0, 0xC0]),
            );

            final aU8 = NDArray<Uint8>.fromList(
              [0xFF, 0x0F, 0x30],
              [3],
              DType.uint8,
            );
            expect(
              aU8.accumulate(op: BinaryOp.bitwiseAnd).toList(),
              equals([0xFF, 0x0F, 0x00]),
            );
            expect(
              aU8.accumulate(op: BinaryOp.bitwiseOr).toList(),
              equals([0xFF, 0xFF, 0xFF]),
            );
            expect(
              aU8.accumulate(op: BinaryOp.bitwiseXor).toList(),
              equals([0xFF, 0xF0, 0xC0]),
            );

            final aBool = NDArray<Boolean>.fromList(
              [true, true, false, true],
              [4],
              DType.boolean,
            );
            expect(
              aBool.accumulate(op: BinaryOp.logicalAnd).toList(),
              equals([true, true, false, false]),
            );
            expect(
              aBool.accumulate(op: BinaryOp.logicalOr).toList(),
              equals([true, true, true, true]),
            );
            expect(
              aBool.accumulate(op: BinaryOp.logicalXor).toList(),
              equals([true, false, false, true]),
            );

            // 2D Matrix accumulates along axis 0 and 1
            final matF64 = NDArray<Float64>.fromList(
              [10.0, 5.0, 2.0, 8.0],
              [2, 2],
              DType.float64,
            );
            final cummin0 = matF64.accumulate(op: BinaryOp.minimum, axis: 0);
            expect(cummin0.toList(), equals([10.0, 5.0, 2.0, 5.0]));
            final cummax1 = matF64.accumulate(op: BinaryOp.maximum, axis: 1);
            expect(cummax1.toList(), equals([10.0, 10.0, 2.0, 8.0]));

            // Complex accumulates
            final c128 = NDArray<Complex128>.fromList(
              [Complex(1.0, 2.0), Complex(2.0, 0.0)],
              [2],
              DType.complex128,
            );
            final cProd = c128.accumulate(op: BinaryOp.multiply);
            expect(cProd.getCell([0]), equals(Complex(1.0, 2.0)));
            expect(cProd.getCell([1]), equals(Complex(2.0, 4.0)));

            final c64 = NDArray<Complex64>.fromList(
              [Complex(1.0, 2.0), Complex(2.0, 0.0)],
              [2],
              DType.complex64,
            );
            final c64Prod = c64.accumulate(op: BinaryOp.multiply);
            expect(c64Prod.getCell([0]), equals(Complex(1.0, 2.0)));
            expect(c64Prod.getCell([1]), equals(Complex(2.0, 4.0)));

            // Out buffer reuse in accumulate
            final outAcc = NDArray<Float64>.zeros([2, 2], DType.float64);
            final resAcc = matF64.accumulate(
              op: BinaryOp.add,
              axis: 1,
              out: outAcc,
            );
            expect(identical(resAcc, outAcc), isTrue);
            expect(outAcc.toList(), equals([10.0, 15.0, 2.0, 10.0]));
          });
        },
      );

      test(
        'BinaryOp.outer across all 15 DTypes combinations and BinaryOp variants',
        () {
          NDArray.scope(() {
            final allDTypes = [
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
              DType.boolean,
              DType.complex128,
              DType.complex64,
            ];

            for (final dt in allDTypes) {
              if (dt == DType.boolean) {
                final a = NDArray.fromList([true, false], [2], dt);
                final b = NDArray.fromList([false, true], [2], dt);
                final out = a.outer(b, op: BinaryOp.logicalAnd);
                expect(out.shape, equals([2, 2]));
                expect(out.toList(), equals([false, true, false, false]));
              } else if (dt.isComplex) {
                final a = NDArray.fromList(
                  [Complex(1.0, 0.0), Complex(2.0, 0.0)],
                  [2],
                  dt,
                );
                final b = NDArray.fromList(
                  [Complex(3.0, 0.0), Complex(4.0, 0.0)],
                  [2],
                  dt,
                );
                final out = a.outer(b, op: BinaryOp.multiply);
                expect(out.shape, equals([2, 2]));
                expect(out.getCell([0, 0]), equals(Complex(3.0, 0.0)));
                expect(out.getCell([1, 1]), equals(Complex(8.0, 0.0)));
              } else if (dt.isFloating) {
                final a = NDArray.fromList([1.0, 2.0], [2], dt);
                final b = NDArray.fromList([3.0, 4.0], [2], dt);
                final out = a.outer(b, op: BinaryOp.add);
                expect(out.shape, equals([2, 2]));
                expect(
                  (out.getCell([0, 0]) as num).toDouble(),
                  closeTo(4.0, 1e-4),
                );
                expect(
                  (out.getCell([1, 1]) as num).toDouble(),
                  closeTo(6.0, 1e-4),
                );
              } else {
                final a = NDArray.fromList([1, 2], [2], dt);
                final b = NDArray.fromList([3, 4], [2], dt);
                final out = a.outer(b, op: BinaryOp.add);
                expect(out.shape, equals([2, 2]));
                expect((out.getCell([0, 0]) as num).toInt(), equals(4));
                expect((out.getCell([1, 1]) as num).toInt(), equals(6));
              }
            }

            // Test BinaryOp variants in outer
            final vecA = NDArray<Float64>.fromList(
              [10.0, 20.0],
              [2],
              DType.float64,
            );
            final vecB = NDArray<Float64>.fromList(
              [2.0, 5.0],
              [2],
              DType.float64,
            );

            expect(
              vecA.outer(vecB, op: BinaryOp.subtract).toList(),
              equals([8.0, 5.0, 18.0, 15.0]),
            );
            expect(
              vecA.outer(vecB, op: BinaryOp.divide).toList(),
              equals([5.0, 2.0, 10.0, 4.0]),
            );
            expect(
              vecA.outer(vecB, op: BinaryOp.floorDivide).toList(),
              equals([5.0, 2.0, 10.0, 4.0]),
            );
            expect(
              vecA.outer(vecB, op: BinaryOp.remainder).toList(),
              equals([0.0, 0.0, 0.0, 0.0]),
            );
            expect(
              vecA.outer(vecB, op: BinaryOp.fmod).toList(),
              equals([0.0, 0.0, 0.0, 0.0]),
            );
            expect(
              vecA.outer(vecB, op: BinaryOp.minimum).toList(),
              equals([2.0, 5.0, 2.0, 5.0]),
            );
            expect(
              vecA.outer(vecB, op: BinaryOp.maximum).toList(),
              equals([10.0, 10.0, 20.0, 20.0]),
            );
            expect(vecA.outer(vecB, op: BinaryOp.hypot).shape, equals([2, 2]));
            expect(
              vecA.outer(vecB, op: BinaryOp.arctan2).shape,
              equals([2, 2]),
            );
            expect(
              vecA.outer(vecB, op: BinaryOp.copysign).toList(),
              equals([10.0, 10.0, 20.0, 20.0]),
            );

            // Outer with where mask
            final whereMask = NDArray<Boolean>.fromList(
              [true, false, false, true],
              [2, 2],
              DType.boolean,
            );
            final outMasked = vecA.outer(
              vecB,
              op: BinaryOp.add,
              where: whereMask,
            );
            expect(outMasked.shape, equals([2, 2]));
            expect(outMasked.getCell([0, 0]), equals(12.0));
            expect(outMasked.getCell([1, 1]), equals(25.0));
          });
        },
      );

      test(
        'BinaryOp.reduceat with out-of-order indices, step indices, and multi-dimensional slices',
        () {
          NDArray.scope(() {
            // Out of order and step indices on 1D
            final a1D = NDArray<Float64>.fromList(
              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
              [8],
              DType.float64,
            );
            final indices1D = NDArray.fromList(
              [0, 4, 1, 5, 2],
              [5],
              DType.int64,
            );
            final res1D = a1D.reduceat(indices1D, op: BinaryOp.add);
            expect(res1D.shape, equals([5]));
            expect(res1D.toList(), equals([6.0, 4.0, 10.0, 5.0, 27.0]));

            // Reduceat across multiple DTypes: Float32, Int64, Int32, Int16, Uint8, Bool, Complex128, Complex64
            final aF32 = NDArray<Float32>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float32,
            );
            final idx = NDArray.fromList([0, 2], [2], DType.int64);
            expect(
              aF32.reduceat(idx, op: BinaryOp.add).toList(),
              equals([3.0, 7.0]),
            );

            final aI64 = NDArray<Int64>.fromList(
              [1, 2, 3, 4],
              [4],
              DType.int64,
            );
            expect(
              aI64.reduceat(idx, op: BinaryOp.add).toList(),
              equals([3, 7]),
            );

            final aI32 = NDArray<Int32>.fromList(
              [1, 2, 3, 4],
              [4],
              DType.int32,
            );
            expect(
              aI32.reduceat(idx, op: BinaryOp.add).toList(),
              equals([3, 7]),
            );

            final aI16 = NDArray<Int16>.fromList(
              [1, 2, 3, 4],
              [4],
              DType.int16,
            );
            expect(
              aI16.reduceat(idx, op: BinaryOp.add).toList(),
              equals([3, 7]),
            );

            final aU8 = NDArray<Uint8>.fromList([1, 2, 3, 4], [4], DType.uint8);
            expect(
              aU8.reduceat(idx, op: BinaryOp.add).toList(),
              equals([3, 7]),
            );

            final aBool = NDArray<Boolean>.fromList(
              [true, true, false, true],
              [4],
              DType.boolean,
            );
            expect(
              aBool.reduceat(idx, op: BinaryOp.logicalAnd).toList(),
              equals([true, false]),
            );

            final aC128 = NDArray<Complex128>.fromList(
              [
                Complex(1.0, 0.0),
                Complex(2.0, 0.0),
                Complex(3.0, 0.0),
                Complex(4.0, 0.0),
              ],
              [4],
              DType.complex128,
            );
            final cRedAt = aC128.reduceat(idx, op: BinaryOp.add);
            expect(cRedAt.getCell([0]), equals(Complex(3.0, 0.0)));
            expect(cRedAt.getCell([1]), equals(Complex(7.0, 0.0)));

            final aC64 = NDArray<Complex64>.fromList(
              [
                Complex(1.0, 0.0),
                Complex(2.0, 0.0),
                Complex(3.0, 0.0),
                Complex(4.0, 0.0),
              ],
              [4],
              DType.complex64,
            );
            final c64RedAt = aC64.reduceat(idx, op: BinaryOp.add);
            expect(c64RedAt.getCell([0]), equals(Complex(3.0, 0.0)));
            expect(c64RedAt.getCell([1]), equals(Complex(7.0, 0.0)));

            // Fallback DTypes (Float16, BFloat16, Int8, Uint64, Uint32, Uint16)
            final aF16 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float16,
            );
            expect(aF16.reduceat(idx, op: BinaryOp.add).shape, equals([2]));

            final aBf16 = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.bfloat16,
            );
            expect(aBf16.reduceat(idx, op: BinaryOp.add).shape, equals([2]));

            final aI8 = NDArray.fromList([1, 2, 3, 4], [4], DType.int8);
            expect(aI8.reduceat(idx, op: BinaryOp.add).shape, equals([2]));

            final aU64 = NDArray.fromList([1, 2, 3, 4], [4], DType.uint64);
            expect(aU64.reduceat(idx, op: BinaryOp.add).shape, equals([2]));

            final aU32 = NDArray.fromList([1, 2, 3, 4], [4], DType.uint32);
            expect(aU32.reduceat(idx, op: BinaryOp.add).shape, equals([2]));

            final aU16 = NDArray.fromList([1, 2, 3, 4], [4], DType.uint16);
            expect(aU16.reduceat(idx, op: BinaryOp.add).shape, equals([2]));

            // 2D Matrix reduceat along axis 0 and axis 1
            final mat2D = NDArray<Float64>.fromList(
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

            final resAx0 = mat2D.reduceat(idx, op: BinaryOp.add, axis: 0);
            expect(resAx0.shape, equals([2, 4]));
            expect(
              resAx0.toList(),
              equals([6.0, 8.0, 10.0, 12.0, 22.0, 24.0, 26.0, 28.0]),
            );

            final resAx1 = mat2D.reduceat(idx, op: BinaryOp.add, axis: 1);
            expect(resAx1.shape, equals([4, 2]));
          });
        },
      );

      test(
        'BinaryOp.at indexed accumulation across all DTypes and strided views',
        () {
          NDArray.scope(() {
            // Float64
            final targetF64 = NDArray<Float64>.zeros([5], DType.float64);
            final indices = NDArray.fromList([0, 1, 0, 2, 1], [5], DType.int64);
            final valsF64 = NDArray<Float64>.fromList(
              [1.0, 10.0, 2.0, 100.0, 20.0],
              [5],
              DType.float64,
            );
            targetF64.at(indices, valsF64, op: BinaryOp.add);
            expect(targetF64.toList(), equals([3.0, 30.0, 100.0, 0.0, 0.0]));

            // Float32
            final targetF32 = NDArray<Float32>.zeros([3], DType.float32);
            final idx3 = NDArray.fromList([0, 0, 1], [3], DType.int64);
            final valsF32 = NDArray<Float32>.fromList(
              [2.0, 3.0, 5.0],
              [3],
              DType.float32,
            );
            targetF32.at(idx3, valsF32, op: BinaryOp.multiply);
            expect(targetF32.toList(), equals([0.0, 0.0, 0.0]));

            // Int64 and Int32 with various operations
            final targetI64 = NDArray<Int64>.fromList(
              [10, 20, 30],
              [3],
              DType.int64,
            );
            final valsI64 = NDArray<Int64>.fromList([2, 5], [2], DType.int64);
            final idx2 = NDArray.fromList([0, 1], [2], DType.int64);
            targetI64.at(idx2, valsI64, op: BinaryOp.subtract);
            expect(targetI64.toList(), equals([8, 15, 30]));

            final targetI32 = NDArray<Int32>.fromList(
              [10, 20, 30],
              [3],
              DType.int32,
            );
            final valsI32 = NDArray<Int32>.fromList([2, 4], [2], DType.int32);
            targetI32.at(idx2, valsI32, op: BinaryOp.divide);
            expect(targetI32.toList(), equals([5, 5, 30]));

            // Int16 and Uint8
            final targetI16 = NDArray<Int16>.fromList(
              [1, 2, 3],
              [3],
              DType.int16,
            );
            final valsI16 = NDArray<Int16>.fromList([10, 20], [2], DType.int16);
            targetI16.at(idx2, valsI16, op: BinaryOp.add);
            expect(targetI16.toList(), equals([11, 22, 3]));

            final targetU8 = NDArray<Uint8>.fromList(
              [1, 2, 3],
              [3],
              DType.uint8,
            );
            final valsU8 = NDArray<Uint8>.fromList([10, 20], [2], DType.uint8);
            targetU8.at(idx2, valsU8, op: BinaryOp.add);
            expect(targetU8.toList(), equals([11, 22, 3]));

            // Complex128 and Complex64
            final targetC128 = NDArray<Complex128>.zeros([3], DType.complex128);
            final valsC128 = NDArray<Complex128>.fromList(
              [Complex(1.0, 1.0), Complex(2.0, 3.0)],
              [2],
              DType.complex128,
            );
            targetC128.at(idx2, valsC128, op: BinaryOp.add);
            expect(targetC128.getCell([0]), equals(Complex(1.0, 1.0)));
            expect(targetC128.getCell([1]), equals(Complex(2.0, 3.0)));

            final targetC64 = NDArray<Complex64>.zeros([3], DType.complex64);
            final valsC64 = NDArray<Complex64>.fromList(
              [Complex(1.0, 1.0), Complex(2.0, 3.0)],
              [2],
              DType.complex64,
            );
            targetC64.at(idx2, valsC64, op: BinaryOp.add);
            expect(targetC64.getCell([0]), equals(Complex(1.0, 1.0)));

            // Boolean at
            final targetBool = NDArray<Boolean>.fromList(
              [false, false, false],
              [3],
              DType.boolean,
            );
            final valsBool = NDArray<Boolean>.fromList(
              [true, true],
              [2],
              DType.boolean,
            );
            targetBool.at(idx2, valsBool, op: BinaryOp.logicalOr);
            expect(targetBool.toList(), equals([true, true, false]));

            // Fallback DTypes (Float16, BFloat16, Int8, Uint64, Uint32, Uint16)
            final targetF16 = NDArray.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float16,
            );
            final valsF16 = NDArray.fromList([5.0, 6.0], [2], DType.float16);
            targetF16.at(idx2, valsF16, op: BinaryOp.add);
            expect(targetF16.shape, equals([3]));

            final targetBf16 = NDArray.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.bfloat16,
            );
            final valsBf16 = NDArray.fromList([5.0, 6.0], [2], DType.bfloat16);
            targetBf16.at(idx2, valsBf16, op: BinaryOp.add);
            expect(targetBf16.shape, equals([3]));

            final targetI8 = NDArray.fromList([1, 2, 3], [3], DType.int8);
            final valsI8 = NDArray.fromList([5, 6], [2], DType.int8);
            targetI8.at(idx2, valsI8, op: BinaryOp.add);
            expect(targetI8.shape, equals([3]));

            final targetU64 = NDArray.fromList([1, 2, 3], [3], DType.uint64);
            final valsU64 = NDArray.fromList([5, 6], [2], DType.uint64);
            targetU64.at(idx2, valsU64, op: BinaryOp.add);
            expect(targetU64.shape, equals([3]));

            final targetU32 = NDArray.fromList([1, 2, 3], [3], DType.uint32);
            final valsU32 = NDArray.fromList([5, 6], [2], DType.uint32);
            targetU32.at(idx2, valsU32, op: BinaryOp.add);
            expect(targetU32.shape, equals([3]));

            final targetU16 = NDArray.fromList([1, 2, 3], [3], DType.uint16);
            final valsU16 = NDArray.fromList([5, 6], [2], DType.uint16);
            targetU16.at(idx2, valsU16, op: BinaryOp.add);
            expect(targetU16.shape, equals([3]));
          });
        },
      );

      test(
        'Direct top-level universal function methods: binaryUfunc, reduce, accumulate, reduceat, at',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0],
              [3],
              DType.float64,
            );
            final b = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );

            final added = binaryUfunc(a, b, op: BinaryOp.add);
            expect(added.toList(), equals([11.0, 22.0, 33.0]));

            final reduced = reduce(a, op: BinaryOp.add);
            expect(reduced.scalar, equals(60.0));

            final accumulated = accumulate(a, op: BinaryOp.add);
            expect(accumulated.toList(), equals([10.0, 30.0, 60.0]));

            final idx = NDArray.fromList([0, 1], [2], DType.int64);
            final redAt = reduceat(a, idx, op: BinaryOp.add);
            expect(redAt.toList(), equals([10.0, 50.0]));

            final target = NDArray<Float64>.zeros([3], DType.float64);
            at(
              target,
              idx,
              b.slice([Slice(start: 0, stop: 2)]),
              op: BinaryOp.add,
            );
            expect(target.toList(), equals([1.0, 2.0, 0.0]));
          });
        },
      );
    });

    // =========================================================================
    // 2. Bitwise & Logical Operations
    // =========================================================================
    group('Bitwise & Logical Operations Coverage', () {
      test(
        'bitwiseAnd, bitwiseOr, bitwiseXor, invert, leftShift, rightShift on integer types and error handling',
        () {
          NDArray.scope(() {
            final supportedIntDTypes = <DType<AnyIntegerSpec>>[
              DType.int64,
              DType.int32,
              DType.int16,
              DType.int8,
              DType.uint64,
              DType.uint32,
              DType.uint16,
              DType.uint8,
            ];

            for (final dt in supportedIntDTypes) {
              final a = NDArray.fromList([12, 10, 7], [3], dt);
              final b = NDArray.fromList([6, 3, 2], [3], dt);

              // bitwiseAnd: 12&6=4, 10&3=2, 7&2=2
              final resAnd = bitwiseAnd(a, b);
              expect(resAnd.toList(), equals([4, 2, 2]));

              // bitwiseOr: 12|6=14, 10|3=11, 7|2=7
              final resOr = bitwiseOr(a, b);
              expect(resOr.toList(), equals([14, 11, 7]));

              // bitwiseXor: 12^6=10, 10^3=9, 7^2=5
              final resXor = bitwiseXor(a, b);
              expect(resXor.toList(), equals([10, 9, 5]));

              // invert (NOT)
              final resInv = invert(a);
              expect(resInv.shape, equals([3]));

              // leftShift: 12<<1=24, 10<<2=40, 7<<1=14
              final shifts = NDArray.fromList([1, 2, 1], [3], dt);
              final resLShift = leftShift(a, shifts);
              expect(resLShift.toList(), equals([24, 40, 14]));

              // rightShift: 12>>1=6, 10>>2=2, 7>>1=3
              final resRShift = rightShift(a, shifts);
              expect(resRShift.toList(), equals([6, 2, 3]));
            }
          });
        },
      );

      test(
        'Bitwise operations with non-contiguous strided views, scalar broadcasting, where mask, and out buffer',
        () {
          NDArray.scope(() {
            final parentA = NDArray<Int32>.fromList(
              [1, 2, 3, 4, 5, 6, 7, 8],
              [2, 4],
              DType.int32,
            );
            final viewA = parentA.slice([
              Slice(),
              Slice(start: 0, stop: 4, step: 2),
            ]); // shape [2, 2] -> [[1, 3], [5, 7]]
            expect(viewA.isContiguous, isFalse);

            final scalarB = NDArray<Int32>.fromList(
              [3],
              [1],
              DType.int32,
            ); // shape [1] broadcasts to [2, 2]

            final resAnd = bitwiseAnd(viewA, scalarB);
            expect(resAnd.shape, equals([2, 2]));
            expect(resAnd.toList(), equals([1 & 3, 3 & 3, 5 & 3, 7 & 3]));

            // Invert on strided view
            final resInv = invert(viewA);
            expect(resInv.shape, equals([2, 2]));

            // With where mask and out buffer
            final outBuf = NDArray<Int32>.zeros([2, 2], DType.int32);
            final mask = NDArray<Boolean>.fromList(
              [true, false, true, false],
              [2, 2],
              DType.boolean,
            );
            bitwiseOr(viewA, scalarB, where: mask, out: outBuf);
            expect(outBuf.getCell([0, 0]), equals(1 | 3));
            expect(outBuf.getCell([0, 1]), equals(0)); // masked out
            expect(outBuf.getCell([1, 0]), equals(5 | 3));

            // Error handling
            final floatArr = NDArray<Float64>.fromList(
              [1.0],
              [1],
              DType.float64,
            );
            expect(
              () => bitwiseAnd(floatArr as dynamic, scalarB),
              throwsArgumentError,
            );
          });
        },
      );

      test(
        'logicalAnd, logicalOr, logicalXor, logicalNot across numeric, float, complex, and boolean arrays with broadcasting',
        () {
          NDArray.scope(() {
            // Numeric truthy/falsy
            final numA = NDArray<Float64>.fromList(
              [0.0, 2.5, 0.0, -1.0],
              [4],
              DType.float64,
            );
            final numB = NDArray<Float64>.fromList(
              [1.0, 0.0, 0.0, 3.0],
              [4],
              DType.float64,
            );

            expect(
              logicalNot(numA).toList(),
              equals([true, false, true, false]),
            );
            expect(
              logicalAnd(numA, numB).toList(),
              equals([false, false, false, true]),
            );
            expect(
              logicalOr(numA, numB).toList(),
              equals([true, true, false, true]),
            );
            expect(
              logicalXor(numA, numB).toList(),
              equals([true, true, false, false]),
            );

            // Complex truthy/falsy
            final cA = NDArray<Complex128>.fromList(
              [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 1.0)],
              [3],
              DType.complex128,
            );
            final cB = NDArray<Complex128>.fromList(
              [Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(2.0, 2.0)],
              [3],
              DType.complex128,
            );

            expect(logicalNot(cA).toList(), equals([true, false, false]));
            expect(logicalAnd(cA, cB).toList(), equals([false, false, true]));
            expect(logicalOr(cA, cB).toList(), equals([false, true, true]));
            expect(logicalXor(cA, cB).toList(), equals([false, true, false]));

            // Non-contiguous strided logical operations with where mask and out buffer
            final matA = NDArray<Int32>.fromList(
              [0, 1, 2, 0],
              [2, 2],
              DType.int32,
            );
            final matB = NDArray<Int32>.fromList(
              [1, 1, 0, 0],
              [2, 2],
              DType.int32,
            );
            final outLog = NDArray<Boolean>.zeros([2, 2], DType.boolean);
            final whereMask = NDArray<Boolean>.fromList(
              [true, true, false, true],
              [2, 2],
              DType.boolean,
            );

            logicalAnd(matA, matB, where: whereMask, out: outLog);
            expect(outLog.getCell([0, 0]), isFalse);
            expect(outLog.getCell([0, 1]), isTrue);
            expect(outLog.getCell([1, 0]), isFalse); // masked out
            expect(outLog.getCell([1, 1]), isFalse);
          });
        },
      );

      test(
        'equal, notEqual, greater, greaterEqual, less, lessEqual comparison operations',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
            final b = NDArray<Float64>.fromList(
              [2.0, 2.0, 2.0, 2.0],
              [4],
              DType.float64,
            );

            expect(equal(a, b).toList(), equals([false, true, false, false]));
            expect(notEqual(a, b).toList(), equals([true, false, true, true]));
            expect(greater(a, b).toList(), equals([false, false, true, true]));
            expect(
              greaterEqual(a, b).toList(),
              equals([false, true, true, true]),
            );
            expect(less(a, b).toList(), equals([true, false, false, false]));
            expect(
              lessEqual(a, b).toList(),
              equals([true, true, false, false]),
            );

            // Complex comparisons
            final c1 = NDArray<Complex128>.fromList(
              [Complex(1.0, 2.0)],
              [1],
              DType.complex128,
            );
            final c2 = NDArray<Complex128>.fromList(
              [Complex(1.0, 2.0)],
              [1],
              DType.complex128,
            );
            expect(equal(c1, c2).toList(), equals([true]));
            expect(notEqual(c1, c2).toList(), equals([false]));
            expect(() => greater(c1, c2), throwsUnsupportedError);
            expect(() => greaterEqual(c1, c2), throwsUnsupportedError);
            expect(() => less(c1, c2), throwsUnsupportedError);
            expect(() => lessEqual(c1, c2), throwsUnsupportedError);
          });
        },
      );
    });

    // =========================================================================
    // 3. Floating Point & Special Functions
    // =========================================================================
    group('Floating Point & Special Functions Coverage', () {
      test(
        'isnan, isinf, isfinite on subnormals, infinities, NaNs, zeros across all floating and complex DTypes',
        () {
          NDArray.scope(() {
            final f64Vals = [
              0.0,
              -0.0,
              5e-324,
              1.0,
              -1.0,
              double.nan,
              double.infinity,
              double.negativeInfinity,
            ];
            final aF64 = NDArray<Float64>.fromList(f64Vals, [8], DType.float64);

            expect(
              isnan(aF64).toList(),
              equals([false, false, false, false, false, true, false, false]),
            );
            expect(
              isinf(aF64).toList(),
              equals([false, false, false, false, false, false, true, true]),
            );
            expect(
              isfinite(aF64).toList(),
              equals([true, true, true, true, true, false, false, false]),
            );

            // Float32
            final aF32 = NDArray<Float32>.fromList(
              [0.0, 1e-40, 1.0, double.nan, double.infinity],
              [5],
              DType.float32,
            );
            expect(
              isnan(aF32).toList(),
              equals([false, false, false, true, false]),
            );
            expect(
              isinf(aF32).toList(),
              equals([false, false, false, false, true]),
            );
            expect(
              isfinite(aF32).toList(),
              equals([true, true, true, false, false]),
            );

            // Complex128 and Complex64
            final aC128 = NDArray<Complex128>.fromList(
              [
                Complex(0.0, 0.0),
                Complex(double.nan, 1.0),
                Complex(1.0, double.infinity),
                Complex(3.0, 4.0),
              ],
              [4],
              DType.complex128,
            );
            expect(isnan(aC128).toList(), equals([false, true, false, false]));
            expect(isinf(aC128).toList(), equals([false, false, true, false]));
            expect(
              isfinite(aC128).toList(),
              equals([true, false, false, true]),
            );

            final aC64 = NDArray<Complex64>.fromList(
              [
                Complex(0.0, 0.0),
                Complex(double.nan, 1.0),
                Complex(1.0, double.infinity),
                Complex(3.0, 4.0),
              ],
              [4],
              DType.complex64,
            );
            expect(isnan(aC64).toList(), equals([false, true, false, false]));
            expect(isinf(aC64).toList(), equals([false, false, true, false]));
            expect(isfinite(aC64).toList(), equals([true, false, false, true]));

            // Non-contiguous strided views and where masks
            final matF64 = NDArray<Float64>.fromList(
              [0.0, double.nan, double.infinity, 1.0],
              [2, 2],
              DType.float64,
            );
            final viewF64 = matF64.slice([
              Slice(),
              Index(0),
            ]); // [0.0, infinity]
            expect(viewF64.isContiguous, isFalse);
            expect(isinf(viewF64).toList(), equals([false, true]));
          });
        },
      );

      test(
        'copysign on subnormals, infinities, NaNs, and zeros with sign preservation',
        () {
          NDArray.scope(() {
            final x1 = NDArray<Float64>.fromList(
              [1.0, -1.0, 5.0, 0.0, double.infinity],
              [5],
              DType.float64,
            );
            final x2 = NDArray<Float64>.fromList(
              [-1.0, 1.0, -0.0, -5.0, -double.infinity],
              [5],
              DType.float64,
            );

            final res = copysign(x1, x2);
            expect(res.toList()[0], equals(-1.0));
            expect(res.toList()[1], equals(1.0));
            expect(res.toList()[2], equals(-5.0));
            expect(res.toList()[3].isNegative, isTrue);
            expect(res.toList()[4], equals(double.negativeInfinity));

            // Strided view and broadcasting
            final a2D = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final b1D = NDArray<Float64>.fromList([-1.0], [1], DType.float64);
            final resBroadcast = copysign(a2D, b1D);
            expect(resBroadcast.shape, equals([2, 2]));
            expect(resBroadcast.toList(), equals([-1.0, -2.0, -3.0, -4.0]));
          });
        },
      );

      test(
        'Special math functions: sinc, i0, gamma, erf across real, complex, integer, and boolean DTypes',
        () {
          NDArray.scope(() {
            // sinc
            final sincVals = NDArray<Float64>.fromList(
              [0.0, 1.0, -1.0, 0.5],
              [4],
              DType.float64,
            );
            final resSinc = sinc(sincVals);
            expect(resSinc.getCell([0]), closeTo(1.0, 1e-9)); // sinc(0) = 1
            expect(resSinc.getCell([1]), closeTo(0.0, 1e-9)); // sinc(1) = 0
            expect(resSinc.getCell([2]), closeTo(0.0, 1e-9)); // sinc(-1) = 0
            expect(
              resSinc.getCell([3]),
              closeTo(2.0 / math.pi, 1e-9),
            ); // sinc(0.5) = 2/pi

            // sinc on complex
            final sincC128 = NDArray<Complex128>.fromList(
              [Complex(0.0, 0.0)],
              [1],
              DType.complex128,
            );
            expect(sinc(sincC128).getCell([0]), equals(Complex(1.0, 0.0)));

            // i0 (zeroth order modified Bessel function)
            final i0Vals = NDArray<Float64>.fromList(
              [0.0, 1.0, 2.0, 5.0],
              [4],
              DType.float64,
            );
            final resI0 = i0(i0Vals);
            expect(resI0.getCell([0]), closeTo(1.0, 1e-6));
            expect(resI0.getCell([1]), closeTo(1.2660658777, 1e-6));
            expect(resI0.getCell([2]), closeTo(2.2795853023, 1e-6));
            expect(
              resI0.getCell([3]),
              closeTo(27.239871825, 1e-4),
            ); // |x| > 3.75 branch

            // i0 on complex (|z| <= 15 and |z| > 15 across quadrants)
            final i0Complex = NDArray<Complex128>.fromList(
              [
                Complex(0.0, 0.0),
                Complex(1.0, 1.0),
                Complex(20.0, 10.0), // |z| > 15 Q1
                Complex(-20.0, 10.0), // |z| > 15 Q2
                Complex(-20.0, -10.0), // |z| > 15 Q3
                Complex(20.0, -10.0), // |z| > 15 Q4
              ],
              [6],
              DType.complex128,
            );
            final resI0C = i0(i0Complex);
            expect(resI0C.getCell([0]), equals(Complex(1.0, 0.0)));
            expect(resI0C.shape, equals([6]));

            // gamma function
            final gammaVals = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
              [6],
              DType.float64,
            );
            final resGamma = gamma(gammaVals);
            expect(resGamma.getCell([0]), closeTo(1.0, 1e-9)); // gamma(1) = 1
            expect(resGamma.getCell([1]), closeTo(1.0, 1e-9)); // gamma(2) = 1
            expect(resGamma.getCell([2]), closeTo(2.0, 1e-9)); // gamma(3) = 2
            expect(resGamma.getCell([3]), closeTo(6.0, 1e-9)); // gamma(4) = 6
            expect(resGamma.getCell([4]), closeTo(24.0, 1e-9)); // gamma(5) = 24
            expect(
              resGamma.getCell([5]),
              closeTo(math.sqrt(math.pi), 1e-6),
            ); // gamma(0.5) = sqrt(pi)

            // erf (error function)
            final erfVals = NDArray<Float64>.fromList(
              [0.0, 1.0, -1.0, 5.0, -5.0],
              [5],
              DType.float64,
            );
            final resErf = erf(erfVals);
            expect(resErf.getCell([0]), closeTo(0.0, 1e-9));
            expect(resErf.getCell([1]), closeTo(0.8427007929, 1e-6));
            expect(resErf.getCell([2]), closeTo(-0.8427007929, 1e-6));
            expect(resErf.getCell([3]), closeTo(1.0, 1e-6));
            expect(resErf.getCell([4]), closeTo(-1.0, 1e-6));

            // Promotions on integer and boolean inputs
            final intArr = NDArray<Int32>.fromList([1, 2, 3], [3], DType.int32);
            expect(gamma(intArr).toList(), equals([1.0, 1.0, 2.0]));
            expect(erf(intArr).shape, equals([3]));
            expect(i0(intArr).shape, equals([3]));
            expect(sinc(intArr).shape, equals([3]));
          });
        },
      );

      test(
        'Window functions: hanning and hamming across lengths and dtypes',
        () {
          NDArray.scope(() {
            // Length M = 0, 1, 2, 8, 512
            expect(hanning(0).shape, equals([0]));
            expect(hanning(1).toList(), equals([1.0]));

            final h2 = hanning(2);
            expect(h2.shape, equals([2]));
            expect(h2.getCell([0]), closeTo(0.0, 1e-9));
            expect(h2.getCell([1]), closeTo(0.0, 1e-9));

            final h8 = hanning(8);
            expect(h8.shape, equals([8]));
            expect(h8.getCell([0]), closeTo(0.0, 1e-9));
            expect(h8.getCell([7]), closeTo(0.0, 1e-9));

            // Hamming window
            expect(hamming(0).shape, equals([0]));
            expect(hamming(1).toList(), equals([1.0]));

            final ham2 = hamming(2);
            expect(ham2.shape, equals([2]));
            expect(ham2.getCell([0]), closeTo(0.08, 1e-4));
            expect(ham2.getCell([1]), closeTo(0.08, 1e-4));

            final ham8 = hamming(8);
            expect(ham8.shape, equals([8]));
            expect(ham8.getCell([0]), closeTo(0.08, 1e-4));
            expect(ham8.getCell([7]), closeTo(0.08, 1e-4));

            // Float32 precision
            final hF32 = hanning(8, dtype: DType.float32);
            expect(hF32.dtype, DType.float32);

            final hamF32 = hamming(8, dtype: DType.float32);
            expect(hamF32.dtype, DType.float32);

            // Out buffer recycling
            final outBuf = NDArray<Float64>.zeros([8], DType.float64);
            final res = hanning(8, out: outBuf);
            expect(identical(res, outBuf), isTrue);

            final outHam = NDArray<Float64>.zeros([8], DType.float64);
            final resHam = hamming(8, out: outHam);
            expect(identical(resHam, outHam), isTrue);
          });
        },
      );

      test(
        'Complex manipulation: real, imag, conj, conjugate, angle across real and complex arrays',
        () {
          NDArray.scope(() {
            // Complex arrays
            final cArr = NDArray<Complex128>.fromList(
              [Complex(3.0, 4.0), Complex(-5.0, 12.0), Complex(0.0, -1.0)],
              [3],
              DType.complex128,
            );

            final r = real(cArr);
            expect(r.dtype, DType.float64);
            expect(r.toList(), equals([3.0, -5.0, 0.0]));

            final im = imag(cArr);
            expect(im.dtype, DType.float64);
            expect(im.toList(), equals([4.0, 12.0, -1.0]));

            final cj = conj(cArr);
            expect(cj.dtype, DType.complex128);
            expect(cj.getCell([0]), equals(Complex(3.0, -4.0)));
            expect(cj.getCell([1]), equals(Complex(-5.0, -12.0)));
            expect(cj.getCell([2]), equals(Complex(0.0, 1.0)));

            final cjAlias = conjugate(cArr);
            expect(cjAlias.getCell([0]), equals(Complex(3.0, -4.0)));

            // Real arrays (zero-copy view and zero imag)
            final realArr = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0],
              [3],
              DType.float64,
            );
            final rView = real(realArr);
            expect(rView.toList(), equals([1.0, 2.0, 3.0]));

            final imZero = imag(realArr);
            expect(imZero.toList(), equals([0.0, 0.0, 0.0]));

            final cjReal = conj(realArr);
            expect(cjReal.toList(), equals([1.0, 2.0, 3.0]));

            // Angle
            final ang = angle(cArr);
            expect(ang.dtype, DType.float64);
            expect(ang.getCell([0]), closeTo(math.atan2(4.0, 3.0), 1e-9));
            expect(ang.getCell([1]), closeTo(math.atan2(12.0, -5.0), 1e-9));
            expect(ang.getCell([2]), closeTo(math.atan2(-1.0, 0.0), 1e-9));

            // Angle out buffer
            final outAng = NDArray<Float64>.zeros([3], DType.float64);
            final resAng = angle(cArr, out: outAng);
            expect(identical(resAng, outAng), isTrue);
            expect(outAng.getCell([0]), closeTo(math.atan2(4.0, 3.0), 1e-9));
          });
        },
      );
    });
  });

  group('Negative window lengths', () {
    test('hanning/hamming with M < 0 return an empty window', () {
      NDArray.scope(() {
        for (final fn in [hanning<Float64>, hamming<Float64>]) {
          final wNeg = fn(-3, dtype: DType.float64);
          expect(wNeg.shape, equals([0]));
        }
      });
    });
  });
}
