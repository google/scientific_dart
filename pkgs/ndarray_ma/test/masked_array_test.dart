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

import 'package:ndarray_ma/ndarray_ma.dart';
import 'package:test/test.dart';

void main() {
  group('MaskedArray Initialization', () {
    test('default constructor and properties', () {
      final data = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final mask = NDArray.fromList(
        [false, true, false, false],
        [2, 2],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      expect(marr.shape, [2, 2]);
      expect(marr.rank, 2);
      expect(marr.size, 4);
      expect(marr.dtype, DType.float64);
      expect(marr.fillValue, 1e20);
      expect(getdata(marr), same(data));
      expect(getmaskarray(marr), same(mask));
    });

    test('constructor shape mismatch', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      final mask = NDArray.fromList(
        [false, true, false, false],
        [2, 2],
        DType.boolean,
      );
      expect(() => MaskedArray(data, mask), throwsArgumentError);
    });

    test('constructor mask type mismatch', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      final mask = NDArray.fromList([0, 1, 0, 0], [4], DType.int32);
      expect(
        () => MaskedArray(data, mask as dynamic),
        throwsA(isA<TypeError>()),
      );
    });

    test('zeros and ones factories', () {
      final mZeros = MaskedArray<Float64>.zeros([2, 3], DType.float64);
      expect(mZeros.shape, [2, 3]);
      expect(mZeros.data.toList(), List.filled(6, 0.0));
      expect(mZeros.mask.toList(), List.filled(6, false));

      final mOnes = MaskedArray<Float64>.ones([2, 3], DType.float64);
      expect(mOnes.shape, [2, 3]);
      expect(mOnes.data.toList(), List.filled(6, 1.0));
      expect(mOnes.mask.toList(), List.filled(6, false));
    });

    test('maskedInvalid factory', () {
      final data = NDArray.fromList(
        [1.0, double.nan, 3.0, double.infinity],
        [4],
        DType.float64,
      );
      final marr = maskedInvalid(data);
      expect(marr.mask.toList(), [false, true, false, true]);
    });

    test('maskedEqual and maskedNotEqual factories', () {
      final data = NDArray.fromList([1, 2, 3, 2], [4], DType.int32);
      final marr = maskedEqual(data, 2);
      expect(marr.mask.toList(), [false, true, false, true]);

      final mNotEq = maskedNotEqual(data, 2);
      expect(mNotEq.mask.toList(), [true, false, true, false]);
    });

    test('maskedGreater/GreaterEqual/Less/LessEqual/Inside/Outside/Where', () {
      final data = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);

      final mGT = maskedGreater(data, 2);
      expect(mGT.mask.toList(), [false, false, true, true]);

      final mGTE = maskedGreaterEqual(data, 2);
      expect(mGTE.mask.toList(), [false, true, true, true]);

      final mLT = maskedLess(data, 3);
      expect(mLT.mask.toList(), [true, true, false, false]);

      final mLTE = maskedLessEqual(data, 3);
      expect(mLTE.mask.toList(), [true, true, true, false]);

      final mIn = maskedInside(data, 2, 3);
      expect(mIn.mask.toList(), [false, true, true, false]);

      final mOut = maskedOutside(data, 2, 3);
      expect(mOut.mask.toList(), [true, false, false, true]);

      final cond = NDArray.fromList(
        [true, false, true, false],
        [4],
        DType.boolean,
      );
      final mWhere = maskedWhere(cond, data);
      expect(mWhere.mask.toList(), [true, false, true, false]);
    });
  });

  group('Element Access & Slicing', () {
    test('coordinate access 1D and toList', () {
      final data = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final mask = NDArray.fromList([false, true, false], [3], DType.boolean);
      final marr = MaskedArray(data, mask);

      expect(marr[0], 1);
      expect(marr[1], null);
      expect(marr[2], 3);
      expect(marr.toList(), <int?>[1, null, 3]);
    });

    test('coordinate access 2D', () {
      final data = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, false],
        [2, 2],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      expect(marr[[0, 0]], 1);
      expect(marr[[0, 1]], null);
      expect(marr[[1, 0]], 3);
      expect(marr[[1, 1]], 4);
    });

    test('slicing 1D', () {
      final data = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, true],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final slice = marr[Slice(start: 1, stop: 3)];
      expect(slice.shape, [2]);
      expect(slice[0], null);
      expect(slice[1], 3);
    });

    test('slicing 2D', () {
      final data = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, false, false, true],
        [2, 3],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final row0 = marr[0];
      expect(row0.shape, [3]);
      expect(row0[0], 1);
      expect(row0[1], null);
      expect(row0[2], 3);

      final colSlice = marr[[Slice.all(), Slice(start: 1, stop: 3)]];
      expect(colSlice.shape, [2, 2]);
      expect(colSlice[[0, 0]], null);
      expect(colSlice[[0, 1]], 3);
      expect(colSlice[[1, 0]], 5);
      expect(colSlice[[1, 1]], null);
    });
  });

  group('Assignment', () {
    test('coordinate assignment', () {
      final data = NDArray.zeros([3], DType.int32);
      final mask = NDArray.fromList([false, false, false], [3], DType.boolean);
      final marr = MaskedArray(data, mask);

      marr[1] = null;
      expect(marr[1], null);
      expect(marr.mask.toList(), [false, true, false]);

      marr[1] = 42;
      expect(marr[1], 42);
      expect(marr.data.toList(), [0, 42, 0]);
      expect(marr.mask.toList(), [false, false, false]);
    });

    test('slice assignment from scalar', () {
      final data = NDArray.zeros([4], DType.int32);
      final mask = NDArray.fromList(
        [false, false, false, false],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      marr[Slice(start: 1, stop: 3)] = 42;
      expect(marr.data.toList(), [0, 42, 42, 0]);
      expect(marr.mask.toList(), [false, false, false, false]);

      marr[Slice(start: 2, stop: 4)] = null;
      expect(marr.mask.toList(), [false, false, true, true]);
    });

    test('slice assignment from NDArray', () {
      final data = NDArray.zeros([4], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, true],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final val = NDArray.fromList([10, 20], [2], DType.int32);
      marr[Slice(start: 1, stop: 3)] = val;

      expect(marr.data.toList(), [0, 10, 20, 0]);
      expect(marr.mask.toList(), [false, false, false, true]);
    });

    test('slice assignment from MaskedArray', () {
      final data = NDArray.zeros([4], DType.int32);
      final mask = NDArray.fromList(
        [false, false, false, false],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final valData = NDArray.fromList([10, 20], [2], DType.int32);
      final valMask = NDArray.fromList([false, true], [2], DType.boolean);
      final valMarr = MaskedArray(valData, valMask);

      marr[Slice(start: 1, stop: 3)] = valMarr;

      expect(marr.data.toList(), [0, 10, 20, 0]);
      expect(marr.mask.toList(), [false, false, true, false]);
    });
  });

  group('Arithmetic, Bitwise, Shift, & Math Operations', () {
    test('addition and out: destination buffer', () {
      final a = MaskedArray(
        NDArray.fromList([1, 2, 3], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );
      final b = MaskedArray(
        NDArray.fromList([10, 20, 30], [3], DType.int32),
        NDArray.fromList([false, false, true], [3], DType.boolean),
      );

      final res = a.add(b);
      expect(res.shape, [3]);
      expect(res.data.toList(), [11, 22, 33]);
      expect(res.mask.toList(), [false, true, true]);

      final out = MaskedArray<Int32>.zeros([3], DType.int32);
      final outRes = add(a, b, out: out);
      expect(outRes, same(out));
      expect(out.toList(), [11, null, null]);
    });

    test('division and zero masking', () {
      final a = MaskedArray(
        NDArray.fromList([4.0, 4.0, 4.0], [3], DType.float64),
        NDArray.fromList([false, false, false], [3], DType.boolean),
      );
      final b = MaskedArray(
        NDArray.fromList([2.0, 0.0, 2.0], [3], DType.float64),
        NDArray.fromList([false, false, false], [3], DType.boolean),
      );

      final res = a.divide(b);
      expect(res.shape, [3]);
      expect(res.mask.toList(), [false, true, false]);
      expect(res[0], 2.0);
      expect(res[1], null);
      expect(res[2], 2.0);
    });

    test('division bypass for masked elements', () {
      final a = MaskedArray(
        NDArray.fromList([4, 4, 4], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );
      final b = MaskedArray(
        NDArray.fromList([2, 0, 2], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );

      expect(() => a.divide(b), returnsNormally);
      final res = a.divide(b);
      expect(res.mask.toList(), [false, true, false]);
      expect(res[0], 2.0);
      expect(res[1], null);
      expect(res[2], 2.0);
    });

    test('bitwise and shift operations', () {
      final a = MaskedArray(
        NDArray.fromList([6, 7, 8], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );
      final b = MaskedArray(
        NDArray.fromList([3, 3, 1], [3], DType.int32),
        NDArray.fromList([false, false, false], [3], DType.boolean),
      );

      expect(bitwiseAnd(a, b).toList(), [2, null, 0]);
      expect((a & b).toList(), [2, null, 0]);
      expect(bitwiseOr(a, b).toList(), [7, null, 9]);
      expect((a | b).toList(), [7, null, 9]);
      expect(bitwiseXor(a, b).toList(), [5, null, 9]);
      expect((a ^ b).toList(), [5, null, 9]);
      expect(invert(a).toList(), [~6, null, ~8]);
      expect((~a).toList(), [~6, null, ~8]);
      expect(leftShift(a, b).toList(), [6 << 3, null, 8 << 1]);
      expect((a << 1).toList(), [12, null, 16]);
      expect(rightShift(a, b).toList(), [6 >> 3, null, 8 >> 1]);
      expect((a >> 1).toList(), [3, null, 4]);
    });

    test('unary and binary math functions, where, concatenate, and clip', () {
      final a = MaskedArray(
        NDArray.fromList([0.0, 1.0, 4.0], [3], DType.float64),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );
      final b = MaskedArray(
        NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64),
        NDArray.fromList([false, false, false], [3], DType.boolean),
      );

      expect(sqrt(a).toList(), [0.0, null, 2.0]);
      expect(sin(a)[0], closeTo(0.0, 1e-12));
      expect(cos(a)[0], closeTo(1.0, 1e-12));
      expect(tan(a)[0], closeTo(0.0, 1e-12));
      expect(exp(a)[0], closeTo(1.0, 1e-12));
      expect(log(b)[0], closeTo(0.0, 1e-12));
      expect(abs(-b).toList(), [1.0, 2.0, 3.0]);
      expect(power(b, b).toList(), [1.0, 4.0, 27.0]);
      expect(hypot(a, b)[0], closeTo(1.0, 1e-12));
      expect(atan2(a, b)[0], closeTo(0.0, 1e-12));
      expect(minimum(a, b).toList(), [0.0, null, 3.0]);
      expect(maximum(a, b).toList(), [1.0, null, 4.0]);
      expect(clip(b, aMin: 1.5, aMax: 2.5).toList(), [1.5, 2.0, 2.5]);

      final cond = NDArray.fromList([true, false, false], [3], DType.boolean);
      expect(where(cond, a, b).toList(), [0.0, 2.0, 3.0]);
      expect(concatenate([a, b]).toList(), [0.0, null, 4.0, 1.0, 2.0, 3.0]);
    });
  });

  group('Reductions & Statistics', () {
    test('sum with and without out:', () {
      final data = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, false],
        [2, 2],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final s = marr.sum();
      expect(s.rank, 0);
      expect(s.scalar, 8);
      expect(s.mask.scalar, false);

      final out0 = MaskedArray<Int64>.zeros([2], DType.int64);
      final s0 = marr.sum(axis: 0, out: out0);
      expect(s0, same(out0));
      expect(s0.shape, [2]);
      expect(s0.data.toList(), [4, 4]);
      expect(s0.mask.toList(), [false, false]);

      final mask2 = NDArray.fromList(
        [false, true, false, true],
        [2, 2],
        DType.boolean,
      );
      final marr2 = MaskedArray(data, mask2);
      final s0_2 = marr2.sum(axis: 0);
      expect(s0_2.mask.toList(), [false, true]);
    });

    test('mean', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      final mask = NDArray.fromList(
        [false, true, false, false],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final m = marr.mean();
      expect(m.scalar, closeTo(8.0 / 3.0, 1e-9));
    });

    test('variance and std', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      final mask = NDArray.fromList(
        [false, true, false, false],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final v = marr.variance();
      expect(v.scalar, closeTo(14.0 / 9.0, 1e-9));

      final s = marr.std();
      expect(s.scalar, closeTo(math.sqrt(14.0 / 9.0), 1e-9));
    });
  });

  group('Compression & Shape Manipulations', () {
    test('count', () {
      final data = NDArray.zeros([2, 3], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, true, true, false],
        [2, 3],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      expect(marr.count().scalar, 3);

      final c0 = marr.count(axis: 0);
      expect(c0.toList(), [1, 0, 2]);
    });

    test('compressed 1D and 2D', () {
      final data = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, true],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final comp1 = compressed(marr);
      expect(comp1.shape, [2]);
      expect(comp1.toList(), [1, 3]);

      final data2 = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
      final mask2 = NDArray.fromList(
        [false, true, false, true, false, false],
        [2, 3],
        DType.boolean,
      );
      final marr2 = MaskedArray(data2, mask2);

      final comp2 = marr2.compressed();
      expect(comp2.shape, [4]);
      expect(comp2.toList(), [1, 3, 5, 6]);
    });

    test('reshape, transpose, transposed, squeeze, and flatten', () {
      final marr = MaskedArray<Float64>.zeros([2, 1, 3], DType.float64);
      marr[[0, 0, 1]] = null;

      final squeezed = marr.squeeze(1);
      expect(squeezed.shape, [2, 3]);
      expect(squeezed[[0, 1]], null);

      final reshaped = squeezed.reshape([3, 2]);
      expect(reshaped.shape, [3, 2]);
      expect(reshaped[[0, 1]], null);

      final transposed = squeezed.transposed;
      expect(transposed.shape, [3, 2]);
      expect(transposed[[1, 0]], null);

      final flat = squeezed.flatten();
      expect(flat.shape, [6]);
      expect(flat[1], null);
    });
  });

  group('Conversions, Copy, Equals & Mapper', () {
    test('filled and copy', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final mask = NDArray.fromList([false, true, false], [3], DType.boolean);
      final marr = MaskedArray(data, mask, fillValue: -99.0);

      final filledDefault = filled(marr);
      expect(filledDefault.toList(), [1.0, -99.0, 3.0]);

      final filledCustom = marr.filled(fillValue: 999.0);
      expect(filledCustom.toList(), [1.0, 999.0, 3.0]);

      final copyArr = marr.copy();
      expect(copyArr.equals(marr), isTrue);
    });

    test('mapUnary', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final mask = NDArray.fromList([false, true, false], [3], DType.boolean);
      final marr = MaskedArray(data, mask);

      final res = marr.mapUnary((x) => -x);
      expect(res[0], -1.0);
      expect(res[1], null);
      expect(res[2], -3.0);
    });
  });

  group('Runtime Type Preservation & Static Projections', () {
    test('runtime type preservation after operations', () {
      final a = MaskedArray<Float64>.zeros([3], DType.float64);
      final b = MaskedArray<Float64>.zeros([3], DType.float64);

      expect(a.add(b), isA<MaskedArray<Float64>>());
      expect(a.divide(b), isA<MaskedArray<Float64>>());
      expect(a.sum(), isA<MaskedArray<Float64>>());
      expect(a.prod(), isA<MaskedArray<Float64>>());
      expect(a.mean(), isA<MaskedArray<Float64>>());
      expect(a.variance(), isA<MaskedArray<Float64>>());
      expect(a.std(), isA<MaskedArray<Float64>>());
      expect(a.reshape([3, 1]), isA<MaskedArray<Float64>>());
    });

    test('mixed dtypes throw ArgumentError on same-dtype binary ops', () {
      final a = MaskedArray<Float64>.zeros([3], DType.float64);
      final b = MaskedArray<Int32>.zeros([3], DType.int32);

      expect(() => (a as MaskedArray<DTypeTag>).add(b), throwsArgumentError);
      expect(
        () => (b as MaskedArray<DTypeTag>).subtract(a),
        throwsArgumentError,
      );
      expect(() => a.asAnySpec + b, throwsArgumentError);
      expect(() => a.asAnySpec / b, throwsArgumentError);
      expect(() => add(a.asAnySpec, b.asAnySpec), throwsArgumentError);
    });

    test('asBitwiseDType and asIntegerDType escape hatches', () {
      final i32 = MaskedArray<Int32>.ones([2], DType.int32);
      final b = MaskedArray<Boolean>.ones([2], DType.boolean);
      final f64 = MaskedArray<Float64>.ones([2], DType.float64);

      expect(
        (i32 as MaskedArray<DTypeTag>).asBitwiseDType,
        isA<MaskedArray<AnyBitwiseSpec>>(),
      );
      expect(
        (b as MaskedArray<DTypeTag>).asBitwiseDType,
        isA<MaskedArray<AnyBitwiseSpec>>(),
      );
      expect(
        () => (f64 as MaskedArray<DTypeTag>).asBitwiseDType,
        throwsArgumentError,
      );

      expect(
        (i32 as MaskedArray<DTypeTag>).asIntegerDType,
        isA<MaskedArray<AnyIntegerSpec>>(),
      );
      expect(
        () => (b as MaskedArray<DTypeTag>).asIntegerDType,
        throwsArgumentError,
      );
      expect(
        () => (f64 as MaskedArray<DTypeTag>).asIntegerDType,
        throwsArgumentError,
      );
    });

    test('AccumulatorTag, DivideTag, and Int64 count projections', () {
      final i32 = MaskedArray(
        NDArray.fromList([2, 4, 6], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );
      final sumI32 = i32.sum();
      expect(sumI32, isA<MaskedArray<Int64>>());
      expect(sumI32.scalar, 8);

      final prodI32 = i32.prod();
      expect(prodI32, isA<MaskedArray<Int64>>());
      expect(prodI32.scalar, 12);

      final divI32 = i32 / 2;
      expect(divI32, isA<MaskedArray<Float64>>());
      expect(divI32[0], 1.0);
      expect(divI32[1], isNull);
      expect(divI32[2], 3.0);

      final f16 = MaskedArray(
        NDArray.fromList([4.0, 6.0], [2], DType.float16),
        NDArray.fromList([false, false], [2], DType.boolean),
      );
      final divF16 = f16 / 2.0;
      expect(divF16, isA<MaskedArray<Float16>>());
      expect(divF16[0], 2.0);
      expect(divF16[1], 3.0);

      final cnt = i32.count();
      expect(cnt, isA<NDArray<Int64>>());
      expect(cnt.dtype, DType.int64);
      expect(cnt.scalar, 2);
    });

    test('arithmetic and comparison operators on MaskedArray', () {
      final a = MaskedArray(
        NDArray.fromList([10, 20, 30], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );
      final b = MaskedArray(
        NDArray.fromList([3, 4, 5], [3], DType.int32),
        NDArray.fromList([false, false, false], [3], DType.boolean),
      );

      expect((a + b)[0], 13);
      expect((a + b)[1], isNull);
      expect((a - b)[2], 25);
      expect((a * 2)[0], 20);
      expect((a ~/ b)[0], 3);
      expect((a % b)[0], 1);
      expect((-a)[0], -10);

      final lt = a < 25;
      expect(lt, isA<MaskedArray<Boolean>>());
      expect(lt[0], true);
      expect(lt[1], isNull);
      expect(lt[2], false);

      expect((a <= 10)[0], true);
      expect((a > b)[2], true);
      expect((a >= 30)[2], true);
      expect(equal(a, b).toList(), [false, null, false]);
      expect(notEqual(a, b).toList(), [true, null, true]);
      expect(less(a, b).toList(), [false, null, false]);
      expect(lessEqual(a, b).toList(), [false, null, false]);
      expect(greater(a, b).toList(), [true, null, true]);
      expect(greaterEqual(a, b).toList(), [true, null, true]);
    });
  });

  group('Top-level Operations', () {
    test('top-level arithmetic and instance NDArray operand', () {
      final a = MaskedArray(
        NDArray.fromList([1, 2], [2], DType.int32),
        NDArray.fromList([false, true], [2], DType.boolean),
      );
      final b = MaskedArray.unmasked(
        NDArray.fromList([10, 20], [2], DType.int32),
      );

      final res = add(a, b);
      expect(res[0], 11);
      expect(res[1], null);

      final rawB = NDArray.fromList([10, 20], [2], DType.int32);
      final resInstance = a.add(rawB);
      expect(resInstance[0], 11);
      expect(resInstance[1], null);
    });

    test('top-level reductions', () {
      final a = MaskedArray(
        NDArray.fromList([1, 2, 3], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
      );
      expect(sum(a).scalar, 4);
      expect(mean(a).scalar, 2.0);
    });
  });

  group('Type Safety & Extra Runtime Types', () {
    test('mean returns Float64 runtime type for Integer input', () {
      final a = MaskedArray<Int32>.ones([3], DType.int32);
      final m = a.mean();
      expect(m, isA<MaskedArray<Float64>>());
      expect(m.data, isA<NDArray<Float64>>());
    });

    test('std returns Float64 runtime type', () {
      final a = MaskedArray<Float64>.ones([3], DType.float64);
      final s = a.std();
      expect(s, isA<MaskedArray<Float64>>());
      expect(s.data, isA<NDArray<Float64>>());
    });

    test('preserve and coerce custom fill value on division promotion', () {
      final a = MaskedArray(
        NDArray.fromList([4, 6, 8], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
        fillValue: 42,
      );
      final b = MaskedArray(
        NDArray.fromList([2, 2, 2], [3], DType.int32),
        NDArray.fromList([false, false, true], [3], DType.boolean),
      );

      final res = a.divide(b);
      expect(res.fillValue, 42.0);
      expect(res, isA<MaskedArray<Float64>>());
    });
  });

  group('Memory Management', () {
    test('scope auto-dispose and MaskedArray.dispose()', () {
      late NDArray rawData;
      late NDArray rawMask;

      final ma = NDArray.scope(() {
        final data = NDArray<Float64>.ones([2, 2], DType.float64);
        final mask = NDArray<Boolean>.zeros([2, 2], DType.boolean);
        rawData = data;
        rawMask = mask;
        return MaskedArray<Float64>(data, mask).detachToParentScope();
      });

      expect(ma.isDisposed, false);
      expect(rawData.isDisposed, false);
      expect(rawMask.isDisposed, false);

      ma.detachFromScope();
      ma.dispose();
      expect(ma.isDisposed, true);
    });
  });
}
