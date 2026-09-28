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
  group('Comprehensive Ufunc Methods Tests', () {
    group('BinaryOp.reduce - Global Reductions across all DTypes', () {
      test('Float64 and Float32 global reductions', () {
        NDArray.scope(() {
          final f64 = NDArray<Float64>.fromList(
            [2.0, 3.0, 4.0, 5.0],
            [4],
            DType.float64,
          );
          expect(f64.reduce(op: BinaryOp.add).scalar, equals(14.0));
          expect(f64.reduce(op: BinaryOp.multiply).scalar, equals(120.0));
          expect(f64.reduce(op: BinaryOp.minimum).scalar, equals(2.0));
          expect(f64.reduce(op: BinaryOp.maximum).scalar, equals(5.0));
          expect(f64.reduce(op: BinaryOp.fmin).scalar, equals(2.0));
          expect(f64.reduce(op: BinaryOp.fmax).scalar, equals(5.0));

          final f32 = NDArray<Float32>.fromList(
            [2.0, 3.0, 4.0, 5.0],
            [4],
            DType.float32,
          );
          expect(f32.reduce(op: BinaryOp.add).scalar, equals(14.0));
          expect(f32.reduce(op: BinaryOp.multiply).scalar, equals(120.0));
          expect(f32.reduce(op: BinaryOp.minimum).scalar, equals(2.0));
          expect(f32.reduce(op: BinaryOp.maximum).scalar, equals(5.0));
        });
      });

      test('Float16 and BFloat16 fallback reductions', () {
        NDArray.scope(() {
          final f16 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float16,
          );
          final sum16 = f16.reduce(op: BinaryOp.add);
          expect(sum16.scalar, closeTo(10.0, 1e-2));

          final bf16 = NDArray.fromList([2.0, 3.0, 4.0], [3], DType.bfloat16);
          final prodBf16 = bf16.reduce(op: BinaryOp.multiply);
          expect(prodBf16.scalar, closeTo(24.0, 1e-1));
        });
      });

      test('Integer DTypes global reductions (int64, int32, int16, uint8)', () {
        NDArray.scope(() {
          final i64 = NDArray<Int64>.fromList([10, 20, 30], [3], DType.int64);
          expect(i64.reduce(op: BinaryOp.add).scalar, equals(60));
          expect(i64.reduce(op: BinaryOp.multiply).scalar, equals(6000));
          expect(i64.reduce(op: BinaryOp.minimum).scalar, equals(10));
          expect(i64.reduce(op: BinaryOp.maximum).scalar, equals(30));
          expect(i64.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(0));
          expect(i64.reduce(op: BinaryOp.bitwiseOr).scalar, equals(30));
          expect(i64.reduce(op: BinaryOp.bitwiseXor).scalar, equals(0));

          final i32 = NDArray<Int32>.fromList([12, 18, 24], [3], DType.int32);
          expect(i32.reduce(op: BinaryOp.add).scalar, equals(54));
          expect(i32.reduce(op: BinaryOp.multiply).scalar, equals(5184));
          expect(i32.reduce(op: BinaryOp.minimum).scalar, equals(12));
          expect(i32.reduce(op: BinaryOp.maximum).scalar, equals(24));
          expect(i32.reduce(op: BinaryOp.gcd).scalar, equals(6));
          expect(i32.reduce(op: BinaryOp.lcm).scalar, equals(72));

          final i16 = NDArray<Int16>.fromList([5, 15, 25], [3], DType.int16);
          expect(i16.reduce(op: BinaryOp.add).scalar, equals(45));
          expect(i16.reduce(op: BinaryOp.multiply).scalar, equals(1875));
          expect(i16.reduce(op: BinaryOp.minimum).scalar, equals(5));
          expect(i16.reduce(op: BinaryOp.maximum).scalar, equals(25));
          expect(i16.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(1));
          expect(i16.reduce(op: BinaryOp.bitwiseOr).scalar, equals(31));
          expect(i16.reduce(op: BinaryOp.bitwiseXor).scalar, equals(19));

          final u8 = NDArray<Uint8>.fromList(
            [0xFF, 0x0F, 0x3F],
            [3],
            DType.uint8,
          );
          expect(u8.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(0x0F));
          expect(u8.reduce(op: BinaryOp.bitwiseOr).scalar, equals(0xFF));
          expect(u8.reduce(op: BinaryOp.minimum).scalar, equals(0x0F));
          expect(u8.reduce(op: BinaryOp.maximum).scalar, equals(0xFF));
        });
      });

      test(
        'Extended Integer DTypes reductions (int8, uint64, uint32, uint16)',
        () {
          NDArray.scope(() {
            final i8 = NDArray.fromList([2, 4, 6], [3], DType.int8);
            expect(i8.reduce(op: BinaryOp.add).scalar, equals(12));

            final u64 = NDArray.fromList([10, 20, 30], [3], DType.uint64);
            expect(u64.reduce(op: BinaryOp.add).scalar, equals(60));

            final u32 = NDArray.fromList([3, 6, 9], [3], DType.uint32);
            expect(u32.reduce(op: BinaryOp.add).scalar, equals(18));

            final u16 = NDArray.fromList([100, 200], [2], DType.uint16);
            expect(u16.reduce(op: BinaryOp.add).scalar, equals(300));
          });
        },
      );

      test('Boolean global reductions', () {
        NDArray.scope(() {
          final allTrue = NDArray<bool>.fromList(
            [true, true, true],
            [3],
            DType.boolean,
          );
          expect(allTrue.reduce(op: BinaryOp.logicalAnd).scalar, isTrue);
          expect(allTrue.reduce(op: BinaryOp.logicalOr).scalar, isTrue);
          expect(allTrue.reduce(op: BinaryOp.logicalXor).scalar, isTrue);

          final mixed = NDArray<bool>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          expect(mixed.reduce(op: BinaryOp.logicalAnd).scalar, isFalse);
          expect(mixed.reduce(op: BinaryOp.logicalOr).scalar, isTrue);
          expect(mixed.reduce(op: BinaryOp.logicalXor).scalar, isFalse);
        });
      });

      test('Complex global reductions', () {
        NDArray.scope(() {
          final c128 = NDArray<Complex128>.fromList(
            [Complex128(1.0, 2.0), Complex128(3.0, -1.0), Complex128(2.0, 4.0)],
            [3],
            DType.complex128,
          );
          final sum = c128.reduce(op: BinaryOp.add);
          expect(sum.scalar, equals(Complex(6.0, 5.0)));

          final prod = c128.reduce(op: BinaryOp.multiply);
          expect(prod.scalar, equals(Complex(-10.0, 30.0)));

          final c64 = NDArray<Complex64>.fromList(
            [Complex64(1.0, 1.0), Complex64(2.0, 2.0)],
            [2],
            DType.complex64,
          );
          final sum64 = c64.reduce(op: BinaryOp.add);
          expect(sum64.scalar, equals(Complex(3.0, 3.0)));
        });
      });

      test('Transcendental reductions: logaddexp, logaddexp2', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final logadd = a.reduce(op: BinaryOp.logaddexp);
          final expected = math.log(
            math.exp(1.0) + math.exp(2.0) + math.exp(3.0),
          );
          expect(logadd.scalar, closeTo(expected, 1e-6));

          final logadd2 = a.reduce(op: BinaryOp.logaddexp2);
          final expected2 =
              math.log(
                math.pow(2.0, 1.0) + math.pow(2.0, 2.0) + math.pow(2.0, 3.0),
              ) /
              math.ln2;
          expect(logadd2.scalar, closeTo(expected2, 1e-6));
        });
      });
    });

    group('BinaryOp.reduce - Multi-Axis & Strided Reductions', () {
      test('2D Matrix reductions along axis 0, 1, and negative axes', () {
        NDArray.scope(() {
          final mat = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final axis0 = mat.reduce(op: BinaryOp.add, axis: 0);
          expect(axis0.shape, equals([3]));
          expect(axis0.toList(), equals([5.0, 7.0, 9.0]));

          final axisNeg2 = mat.reduce(op: BinaryOp.add, axis: -2);
          expect(axisNeg2.toList(), equals([5.0, 7.0, 9.0]));

          final axis1 = mat.reduce(op: BinaryOp.add, axis: 1);
          expect(axis1.shape, equals([2]));
          expect(axis1.toList(), equals([6.0, 15.0]));

          final axisNeg1 = mat.reduce(op: BinaryOp.add, axis: -1);
          expect(axisNeg1.toList(), equals([6.0, 15.0]));
        });
      });

      test('3D Tensor reductions across axes with keepdims', () {
        NDArray.scope(() {
          final tensor = NDArray<Int32>.fromList(
            List.generate(24, (i) => i + 1),
            [2, 3, 4],
            DType.int32,
          );

          final r0 = tensor.reduce(op: BinaryOp.add, axis: 0, keepdims: true);
          expect(r0.shape, equals([1, 3, 4]));

          final r1 = tensor.reduce(op: BinaryOp.add, axis: 1, keepdims: true);
          expect(r1.shape, equals([2, 1, 4]));

          final r2 = tensor.reduce(op: BinaryOp.add, axis: 2, keepdims: true);
          expect(r2.shape, equals([2, 3, 1]));

          final r0NoKeep = tensor.reduce(
            op: BinaryOp.add,
            axis: 0,
            keepdims: false,
          );
          expect(r0NoKeep.shape, equals([3, 4]));
        });
      });

      test(
        'Strided non-contiguous reductions (transposition and stepping)',
        () {
          NDArray.scope(() {
            final mat = NDArray<Float64>.fromList(
              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
              [4, 2],
              DType.float64,
            );
            final transposed = mat.transpose();
            expect(transposed.isContiguous, isFalse);

            final sumTrans0 = transposed.reduce(op: BinaryOp.add, axis: 0);
            expect(sumTrans0.shape, equals([4]));
            expect(sumTrans0.toList(), equals([3.0, 7.0, 11.0, 15.0]));

            final stepped = mat.slice([Slice(step: 2), Slice()]);
            expect(stepped.shape, equals([2, 2]));
            final sumStepped = stepped.reduce(op: BinaryOp.add, axis: 0);
            expect(sumStepped.toList(), equals([6.0, 8.0]));
          });
        },
      );

      test('Reductions with initial parameter', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final sumWithInit = a.reduce(
            op: BinaryOp.add,
            initial: Float64(10.0),
          );
          expect(sumWithInit.scalar, equals(16.0));

          final prodWithInit = a.reduce(
            op: BinaryOp.multiply,
            initial: Float64(2.0),
          );
          expect(prodWithInit.scalar, equals(12.0));

          final mat = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final axisSumInit = mat.reduce(
            op: BinaryOp.add,
            axis: 0,
            initial: Float64(100.0),
          );
          expect(axisSumInit.toList(), equals([104.0, 106.0]));
        });
      });

      test('Reductions with out buffer validation', () {
        NDArray.scope(() {
          final a = NDArray<Int64>.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          final validOut = NDArray<Int64>.zeros([2], DType.int64);
          final res = a.reduce(op: BinaryOp.add, axis: 0, out: validOut);
          expect(identical(res, validOut), isTrue);
          expect(validOut.toList(), equals([4, 6]));

          // Incompatible shape
          final wrongShapeOut = NDArray<Int64>.zeros([3], DType.int64);
          expect(
            () => a.reduce(op: BinaryOp.add, axis: 0, out: wrongShapeOut),
            throwsArgumentError,
          );
        });
      });

      test('Empty array reductions', () {
        NDArray.scope(() {
          final empty1D = NDArray<Float64>.zeros([0], DType.float64);
          expect(() => empty1D.reduce(op: BinaryOp.add), throwsArgumentError);

          final emptyWithInit = empty1D.reduce(
            op: BinaryOp.add,
            initial: Float64(99.0),
          );
          expect(emptyWithInit.scalar, equals(99.0));

          final empty2D = NDArray<Float64>.zeros([0, 5], DType.float64);
          expect(
            () => empty2D.reduce(op: BinaryOp.add, axis: 0),
            throwsArgumentError,
          );

          final empty2DWithInit = empty2D.reduce(
            op: BinaryOp.add,
            axis: 0,
            initial: Float64(10.0),
          );
          expect(empty2DWithInit.shape, equals([5]));
          expect(empty2DWithInit.toList(), equals(List.filled(5, 10.0)));
        });
      });
    });

    group('BinaryOp.accumulate Tests', () {
      test('Accumulate across multiple DTypes (Float, Int, Complex, Bool)', () {
        NDArray.scope(() {
          final f32 = NDArray<Float32>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float32,
          );
          expect(
            f32.accumulate(op: BinaryOp.add).toList(),
            equals([1.0, 3.0, 6.0]),
          );
          expect(
            f32.accumulate(op: BinaryOp.multiply).toList(),
            equals([1.0, 2.0, 6.0]),
          );

          final i32 = NDArray<Int32>.fromList([10, 5, 2], [3], DType.int32);
          expect(
            i32.accumulate(op: BinaryOp.add).toList(),
            equals([10, 15, 17]),
          );
          expect(
            i32.accumulate(op: BinaryOp.minimum).toList(),
            equals([10, 5, 2]),
          );
          expect(
            i32.accumulate(op: BinaryOp.maximum).toList(),
            equals([10, 10, 10]),
          );
          expect(
            i32.accumulate(op: BinaryOp.bitwiseAnd).toList(),
            equals([10, 0, 0]),
          );
          expect(
            i32.accumulate(op: BinaryOp.bitwiseOr).toList(),
            equals([10, 15, 15]),
          );

          final i16 = NDArray<Int16>.fromList([7, 3, 1], [3], DType.int16);
          expect(
            i16.accumulate(op: BinaryOp.bitwiseXor).toList(),
            equals([7, 4, 5]),
          );

          final u8 = NDArray<Uint8>.fromList([2, 3, 4], [3], DType.uint8);
          expect(u8.accumulate(op: BinaryOp.add).toList(), equals([2, 5, 9]));

          final c128 = NDArray<Complex128>.fromList(
            [Complex128(1.0, 1.0), Complex128(2.0, -1.0)],
            [2],
            DType.complex128,
          );
          final cCumsum = c128.accumulate(op: BinaryOp.add);
          expect(cCumsum.getCell([0]), equals(Complex(1.0, 1.0)));
          expect(cCumsum.getCell([1]), equals(Complex(3.0, 0.0)));

          final cCumprod = c128.accumulate(op: BinaryOp.multiply);
          expect(cCumprod.getCell([0]), equals(Complex(1.0, 1.0)));
          expect(cCumprod.getCell([1]), equals(Complex(3.0, 1.0)));

          final bools = NDArray<bool>.fromList(
            [true, true, false, true],
            [4],
            DType.boolean,
          );
          expect(
            bools.accumulate(op: BinaryOp.logicalAnd).toList(),
            equals([true, true, false, false]),
          );
          expect(
            bools.accumulate(op: BinaryOp.logicalOr).toList(),
            equals([true, true, true, true]),
          );
          expect(
            bools.accumulate(op: BinaryOp.logicalXor).toList(),
            equals([true, false, false, true]),
          );
        });
      });

      test('Accumulate 2D and 3D arrays along various axes with striding', () {
        NDArray.scope(() {
          final mat = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final acc0 = mat.accumulate(op: BinaryOp.add, axis: 0);
          expect(acc0.shape, equals([2, 3]));
          expect(acc0.toList(), equals([1.0, 2.0, 3.0, 5.0, 7.0, 9.0]));

          final acc1 = mat.accumulate(op: BinaryOp.add, axis: 1);
          expect(acc1.shape, equals([2, 3]));
          expect(acc1.toList(), equals([1.0, 3.0, 6.0, 4.0, 9.0, 15.0]));

          final accNeg1 = mat.accumulate(op: BinaryOp.add, axis: -1);
          expect(accNeg1.toList(), equals([1.0, 3.0, 6.0, 4.0, 9.0, 15.0]));

          final transposed = mat.transpose();
          final accTrans = transposed.accumulate(op: BinaryOp.add, axis: 0);
          expect(accTrans.shape, equals([3, 2]));
          expect(accTrans.toList(), equals([1.0, 4.0, 3.0, 9.0, 6.0, 15.0]));
        });
      });

      test('Accumulate fallback with non-standard DTypes and out buffer', () {
        NDArray.scope(() {
          final f16 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float16);
          final accF16 = f16.accumulate(op: BinaryOp.add);
          expect(accF16.shape, equals([3]));
          expect(accF16.getCell([0]), closeTo(1.0, 1e-2));
          expect(accF16.getCell([1]), closeTo(3.0, 1e-2));
          expect(accF16.getCell([2]), closeTo(6.0, 1e-2));

          final out = NDArray<Float64>.zeros([3], DType.float64);
          final src = NDArray<Float64>.fromList(
            [2.0, 3.0, 4.0],
            [3],
            DType.float64,
          );
          final res = src.accumulate(op: BinaryOp.add, out: out);
          expect(identical(res, out), isTrue);
          expect(out.toList(), equals([2.0, 5.0, 9.0]));
        });
      });
    });

    group('BinaryOp.reduceat Tests', () {
      test('1D reduceat across various DTypes (Float64, Float32, Int64, Int32, Int16, Uint8, Complex)', () {
        NDArray.scope(() {
          final f64 = NDArray<Float64>.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [8],
            DType.float64,
          );
          final idx64 = NDArray<int>.fromList([0, 4, 1, 5], [4], DType.int64);
          final res64 = f64.reduceat(idx64, op: BinaryOp.add);
          expect(res64.toList(), equals([6.0, 4.0, 10.0, 18.0]));

          final f32 = NDArray<Float32>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
          );
          final idx32 = NDArray<int>.fromList([0, 2], [2], DType.int32);
          final res32 = f32.reduceat(idx32, op: BinaryOp.multiply);
          expect(res32.toList(), equals([2.0, 12.0]));

          final i32Arr = NDArray<Int32>.fromList(
            [10, 20, 30, 40],
            [4],
            DType.int32,
          );
          final idxI32 = NDArray<int>.fromList([0, 2], [2], DType.int64);
          expect(
            i32Arr.reduceat(idxI32, op: BinaryOp.add).toList(),
            equals([30, 70]),
          );

          final i16Arr = NDArray<Int16>.fromList(
            [1, 2, 3, 4],
            [4],
            DType.int16,
          );
          expect(
            i16Arr.reduceat(idxI32, op: BinaryOp.add).toList(),
            equals([3, 7]),
          );

          final u8Arr = NDArray<Uint8>.fromList([1, 2, 3, 4], [4], DType.uint8);
          expect(
            u8Arr.reduceat(idxI32, op: BinaryOp.add).toList(),
            equals([3, 7]),
          );

          final c128Arr = NDArray<Complex128>.fromList(
            [
              Complex128(1.0, 1.0),
              Complex128(2.0, 2.0),
              Complex128(3.0, 3.0),
              Complex128(4.0, 4.0),
            ],
            [4],
            DType.complex128,
          );
          final cRes = c128Arr.reduceat(idxI32, op: BinaryOp.add);
          expect(cRes.getCell([0]), equals(Complex(3.0, 3.0)));
          expect(cRes.getCell([1]), equals(Complex(7.0, 7.0)));

          final c64Arr = NDArray<Complex64>.fromList(
            [Complex64(1.0, 1.0), Complex64(2.0, 2.0)],
            [2],
            DType.complex64,
          );
          final idx1 = NDArray<int>.fromList([0], [1], DType.int64);
          expect(
            c64Arr.reduceat(idx1, op: BinaryOp.add).getCell([0]),
            equals(Complex(3.0, 3.0)),
          );
        });
      });

      test('2D reduceat along axis 0 and axis 1', () {
        NDArray.scope(() {
          final mat = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [4, 3],
            DType.float64,
          );

          final indices0 = NDArray<int>.fromList([0, 2], [2], DType.int64);
          final red0 = mat.reduceat(indices0, op: BinaryOp.add, axis: 0);
          expect(red0.shape, equals([2, 3]));
          expect(red0.toList(), equals([5.0, 7.0, 9.0, 17.0, 19.0, 21.0]));

          final indices1 = NDArray<int>.fromList([0, 1], [2], DType.int64);
          final red1 = mat.reduceat(indices1, op: BinaryOp.add, axis: 1);
          expect(red1.shape, equals([4, 2]));
        });
      });

      test('reduceat with extended DTypes (Float16, Int8, Uint64)', () {
        NDArray.scope(() {
          final f16 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float16,
          );
          final idx = NDArray<int>.fromList([0, 2], [2], DType.int64);
          final res = f16.reduceat(idx, op: BinaryOp.add);
          expect(res.shape, equals([2]));
          expect(res.getCell([0]), closeTo(3.0, 1e-2));
          expect(res.getCell([1]), closeTo(7.0, 1e-2));

          final i8 = NDArray.fromList([2, 3, 4, 5], [4], DType.int8);
          final resI8 = i8.reduceat(idx, op: BinaryOp.add);
          expect(resI8.toList(), equals([5, 9]));
        });
      });
    });

    group('BinaryOp.outer Tests', () {
      test('Outer product, addition, subtraction, division across multiple dimensions', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray<Float64>.fromList(
            [3.0, 4.0, 5.0],
            [3],
            DType.float64,
          );

          final mul = a.outer(b, op: BinaryOp.multiply);
          expect(mul.shape, equals([2, 3]));
          expect(mul.toList(), equals([3.0, 4.0, 5.0, 6.0, 8.0, 10.0]));

          final add = a.outer(b, op: BinaryOp.add);
          expect(add.toList(), equals([4.0, 5.0, 6.0, 5.0, 6.0, 7.0]));

          final sub = a.outer(b, op: BinaryOp.subtract);
          expect(sub.toList(), equals([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0]));

          final div = a.outer(b, op: BinaryOp.divide);
          expect(div.shape, equals([2, 3]));
          expect(div.getCell([0, 0]), closeTo(1.0 / 3.0, 1e-6));
        });
      });

      test('Outer operations with 2D x 1D and 2D x 2D arrays', () {
        NDArray.scope(() {
          final a2d = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b1d = NDArray<Float64>.fromList(
            [10.0, 20.0],
            [2],
            DType.float64,
          );

          final res2d1d = a2d.outer(b1d, op: BinaryOp.multiply);
          expect(res2d1d.shape, equals([2, 2, 2]));

          final b2d = NDArray<Float64>.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final res2d2d = a2d.outer(b2d, op: BinaryOp.multiply);
          expect(res2d2d.shape, equals([2, 2, 2, 2]));
        });
      });

      test('Outer with integer, bitwise, and logical ops', () {
        NDArray.scope(() {
          final i1 = NDArray<Int64>.fromList([1, 2], [2], DType.int64);
          final i2 = NDArray<Int64>.fromList([3, 4], [2], DType.int64);

          expect(
            i1.outer(i2, op: BinaryOp.bitwiseAnd).toList(),
            equals([1, 0, 2, 0]),
          );
          expect(
            i1.outer(i2, op: BinaryOp.bitwiseOr).toList(),
            equals([3, 5, 3, 6]),
          );
          expect(
            i1.outer(i2, op: BinaryOp.bitwiseXor).toList(),
            equals([2, 5, 1, 6]),
          );

          final b1 = NDArray<bool>.fromList([true, false], [2], DType.boolean);
          final b2 = NDArray<bool>.fromList([true, false], [2], DType.boolean);
          expect(
            b1.outer(b2, op: BinaryOp.logicalAnd).toList(),
            equals([true, false, false, false]),
          );
          expect(
            b1.outer(b2, op: BinaryOp.logicalOr).toList(),
            equals([true, true, true, false]),
          );
          expect(
            b1.outer(b2, op: BinaryOp.logicalXor).toList(),
            equals([false, true, true, false]),
          );
        });
      });

      test('Outer with where mask and out buffer', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray<Float64>.fromList([10.0, 20.0], [2], DType.float64);
          final out = NDArray<Float64>.zeros([2, 2], DType.float64);
          final mask = NDArray<bool>.fromList(
            [true, false, false, true],
            [2, 2],
            DType.boolean,
          );

          a.outer(b, op: BinaryOp.multiply, where: mask, out: out);
          expect(out.getCell([0, 0]), equals(10.0));
          expect(out.getCell([0, 1]), equals(0.0));
          expect(out.getCell([1, 0]), equals(0.0));
          expect(out.getCell([1, 1]), equals(40.0));
        });
      });
    });

    group('BinaryOp.at Tests', () {
      test('at in-place scatter update across multiple DTypes', () {
        NDArray.scope(() {
          // Float64
          final f64 = NDArray<Float64>.fromList(
            [0.0, 0.0, 0.0],
            [3],
            DType.float64,
          );
          final idx = NDArray<int>.fromList([0, 1, 0], [3], DType.int64);
          final valF64 = NDArray<Float64>.fromList(
            [5.0, 10.0, 2.0],
            [3],
            DType.float64,
          );
          f64.at(idx, valF64, op: BinaryOp.add);
          expect(f64.toList(), equals([7.0, 10.0, 0.0]));

          // Float32
          final f32 = NDArray<Float32>.fromList(
            [10.0, 20.0],
            [2],
            DType.float32,
          );
          final idxF32 = NDArray<int>.fromList([0, 1], [2], DType.int32);
          final valF32 = NDArray<Float32>.fromList(
            [3.0, 5.0],
            [2],
            DType.float32,
          );
          f32.at(idxF32, valF32, op: BinaryOp.subtract);
          expect(f32.toList(), equals([7.0, 15.0]));

          // Int64
          final i64 = NDArray<Int64>.fromList([1, 1, 1], [3], DType.int64);
          final valI64 = NDArray<Int64>.fromList([2, 3, 4], [3], DType.int64);
          i64.at(idx, valI64, op: BinaryOp.multiply);
          expect(i64.toList(), equals([8, 3, 1]));

          // Uint8
          final u8 = NDArray<Uint8>.fromList([0x0F, 0xF0], [2], DType.uint8);
          final idxU8 = NDArray<int>.fromList([0, 1], [2], DType.int64);
          final valU8 = NDArray<Uint8>.fromList([0x01, 0x10], [2], DType.uint8);
          u8.at(idxU8, valU8, op: BinaryOp.bitwiseOr);
          expect(u8.toList(), equals([0x0F, 0xF0]));

          // Int16
          final i16 = NDArray<Int16>.fromList([100, 200], [2], DType.int16);
          final valI16 = NDArray<Int16>.fromList([10, 20], [2], DType.int16);
          i16.at(idxU8, valI16, op: BinaryOp.remainder);
          expect(i16.toList(), equals([0, 0]));

          // Complex128
          final c128 = NDArray<Complex128>.fromList(
            [Complex128(0.0, 0.0), Complex128(1.0, 1.0)],
            [2],
            DType.complex128,
          );
          final valC128 = NDArray<Complex128>.fromList(
            [Complex128(2.0, 3.0), Complex128(1.0, -1.0)],
            [2],
            DType.complex128,
          );
          c128.at(idxU8, valC128, op: BinaryOp.add);
          expect(c128.getCell([0]), equals(Complex(2.0, 3.0)));
          expect(c128.getCell([1]), equals(Complex(2.0, 0.0)));

          // Boolean
          final bools = NDArray<bool>.fromList(
            [false, true],
            [2],
            DType.boolean,
          );
          final valBools = NDArray<bool>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );
          bools.at(idxU8, valBools, op: BinaryOp.logicalOr);
          expect(bools.toList(), equals([true, true]));
        });
      });

      test('at with extended DTypes (Float16, Int8, Uint64)', () {
        NDArray.scope(() {
          final f16 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float16);
          final idx = NDArray<int>.fromList([0, 1], [2], DType.int64);
          final val = NDArray.fromList([10.0, 20.0], [2], DType.float16);
          f16.at(idx, val, op: BinaryOp.add);
          expect(f16.getCell([0]), closeTo(11.0, 1e-1));
          expect(f16.getCell([1]), closeTo(22.0, 1e-1));
          expect(f16.getCell([2]), closeTo(3.0, 1e-1));
        });
      });

      test('at scatter operations with power, min, max', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [2.0, 5.0, 10.0],
            [3],
            DType.float64,
          );
          final idx = NDArray<int>.fromList([0, 1, 2], [3], DType.int64);
          final bPower = NDArray<Float64>.fromList(
            [3.0, 2.0, 0.5],
            [3],
            DType.float64,
          );

          a.at(idx, bPower, op: BinaryOp.power);
          expect(a.getCell([0]), closeTo(8.0, 1e-6));
          expect(a.getCell([1]), closeTo(25.0, 1e-6));
          expect(a.getCell([2]), closeTo(math.sqrt(10.0), 1e-6));

          final bMin = NDArray<Float64>.fromList(
            [5.0, 30.0, 1.0],
            [3],
            DType.float64,
          );
          a.at(idx, bMin, op: BinaryOp.minimum);
          expect(a.getCell([0]), closeTo(5.0, 1e-6));
          expect(a.getCell([1]), closeTo(25.0, 1e-6));
          expect(a.getCell([2]), closeTo(1.0, 1e-6));

          final bMax = NDArray<Float64>.fromList(
            [10.0, 20.0, 100.0],
            [3],
            DType.float64,
          );
          a.at(idx, bMax, op: BinaryOp.maximum);
          expect(a.getCell([0]), closeTo(10.0, 1e-6));
          expect(a.getCell([1]), closeTo(25.0, 1e-6));
          expect(a.getCell([2]), closeTo(100.0, 1e-6));
        });
      });
    });

    group('binaryUfunc Switch Matrix Coverage', () {
      test(
        'binaryUfunc dispatches correctly across all BinaryOp enum values',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList([6.0, 8.0], [2], DType.float64);
            final b = NDArray<Float64>.fromList([2.0, 4.0], [2], DType.float64);

            expect(
              binaryUfunc(a, b, op: BinaryOp.add).toList(),
              equals([8.0, 12.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.subtract).toList(),
              equals([4.0, 4.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.multiply).toList(),
              equals([12.0, 32.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.divide).toList(),
              equals([3.0, 2.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.floorDivide).toList(),
              equals([3.0, 2.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.remainder).toList(),
              equals([0.0, 0.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.fmod).toList(),
              equals([0.0, 0.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.power).toList(),
              equals([36.0, 4096.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.floatPower).toList(),
              equals([36.0, 4096.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.heaviside).toList(),
              equals([1.0, 1.0]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.hypot).toList(),
              equals([math.sqrt(40.0), math.sqrt(80.0)]),
            );
            expect(
              binaryUfunc(a, b, op: BinaryOp.copysign).toList(),
              equals([6.0, 8.0]),
            );
            expect(binaryUfunc(a, b, op: BinaryOp.arctan2).shape, equals([2]));

            final iA = NDArray<Int64>.fromList([12, 15], [2], DType.int64);
            final iB = NDArray<Int64>.fromList([18, 20], [2], DType.int64);
            expect(
              binaryUfunc(iA, iB, op: BinaryOp.gcd).toList(),
              equals([6, 5]),
            );
            expect(
              binaryUfunc(iA, iB, op: BinaryOp.lcm).toList(),
              equals([36, 60]),
            );
            expect(
              binaryUfunc(iA, iB, op: BinaryOp.bitwiseAnd).toList(),
              equals([12 & 18, 15 & 20]),
            );
            expect(
              binaryUfunc(iA, iB, op: BinaryOp.bitwiseOr).toList(),
              equals([12 | 18, 15 | 20]),
            );
            expect(
              binaryUfunc(iA, iB, op: BinaryOp.bitwiseXor).toList(),
              equals([12 ^ 18, 15 ^ 20]),
            );
            expect(
              binaryUfunc(iA, iB, op: BinaryOp.leftShift).shape,
              equals([2]),
            );
            expect(
              binaryUfunc(iA, iB, op: BinaryOp.rightShift).shape,
              equals([2]),
            );

            final bA = NDArray<bool>.fromList(
              [true, false],
              [2],
              DType.boolean,
            );
            final bB = NDArray<bool>.fromList(
              [false, true],
              [2],
              DType.boolean,
            );
            expect(
              binaryUfunc(bA, bB, op: BinaryOp.logicalAnd).toList(),
              equals([false, false]),
            );
            expect(
              binaryUfunc(bA, bB, op: BinaryOp.logicalOr).toList(),
              equals([true, true]),
            );
            expect(
              binaryUfunc(bA, bB, op: BinaryOp.logicalXor).toList(),
              equals([true, true]),
            );
          });
        },
      );
    });

    group('Error Handling across Ufunc Methods', () {
      test('Disposed array checks', () {
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        final indices = NDArray<int>.fromList([0], [1], DType.int64);
        final b = NDArray<Float64>.fromList([3.0], [1], DType.float64);
        final out = NDArray<Float64>.zeros([2], DType.float64);

        a.dispose();
        expect(() => reduce(a, op: BinaryOp.add), throwsStateError);
        expect(() => accumulate(a, op: BinaryOp.add), throwsStateError);
        expect(() => reduceat(a, indices, op: BinaryOp.add), throwsStateError);
        expect(() => outer(a, b), throwsStateError);
        expect(() => at(a, indices, b, op: BinaryOp.add), throwsStateError);

        indices.dispose();
        b.dispose();
        out.dispose();
      });

      test(
        'Non-reducible op on reduce, accumulate, reduceat throws ArgumentError',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
            final indices = NDArray<int>.fromList([0], [1], DType.int64);

            expect(() => a.reduce(op: BinaryOp.subtract), throwsArgumentError);
            expect(
              () => a.accumulate(op: BinaryOp.divide),
              throwsArgumentError,
            );
            expect(
              () => a.reduceat(indices, op: BinaryOp.power),
              throwsArgumentError,
            );
          });
        },
      );

      test('Axis out of range throws RangeError', () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final indices = NDArray<int>.fromList([0], [1], DType.int64);

          expect(() => a.reduce(op: BinaryOp.add, axis: 5), throwsRangeError);
          expect(() => a.reduce(op: BinaryOp.add, axis: -5), throwsRangeError);
          expect(
            () => a.accumulate(op: BinaryOp.add, axis: 3),
            throwsRangeError,
          );
          expect(
            () => a.accumulate(op: BinaryOp.add, axis: -3),
            throwsRangeError,
          );
          expect(
            () => a.reduceat(indices, op: BinaryOp.add, axis: 2),
            throwsRangeError,
          );
        });
      });
    });
  });
}
