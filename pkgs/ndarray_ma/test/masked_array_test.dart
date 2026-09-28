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

import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart'
    hide
        add,
        subtract,
        multiply,
        divide,
        sum,
        prod,
        min,
        max,
        mean,
        variance,
        std;
import 'package:ndarray_ma/ndarray_ma.dart';
import 'dart:math' as math;

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
      final marr = MaskedArray.maskedInvalid(data);
      expect(marr.mask.toList(), [false, true, false, true]);
    });

    test('maskedEqual factory', () {
      final data = NDArray.fromList([1, 2, 3, 2], [4], DType.int32);
      final marr = MaskedArray.maskedEqual(data, 2);
      expect(marr.mask.toList(), [false, true, false, true]);
    });

    test('maskedGreater/GreaterEqual/Less/LessEqual factories', () {
      final data = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);

      final mGT = MaskedArray.maskedGreater(data, 2);
      expect(mGT.mask.toList(), [false, false, true, true]);

      final mGTE = MaskedArray.maskedGreaterEqual(data, 2);
      expect(mGTE.mask.toList(), [false, true, true, true]);

      final mLT = MaskedArray.maskedLess(data, 3);
      expect(mLT.mask.toList(), [true, true, false, false]);

      final mLTE = MaskedArray.maskedLessEqual(data, 3);
      expect(mLTE.mask.toList(), [true, true, true, false]);
    });
  });

  group('Element Access & Slicing', () {
    test('coordinate access 1D', () {
      final data = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final mask = NDArray.fromList([false, true, false], [3], DType.boolean);
      final marr = MaskedArray(data, mask);

      expect(marr[0], 1);
      expect(marr[1], null);
      expect(marr[2], 3);
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
      expect(slice[0], null); // originally index 1
      expect(slice[1], 3); // originally index 2
    });

    test('slicing 2D', () {
      final data = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, false, false, true],
        [2, 3],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      // Select first row
      final row0 = marr[0];
      expect(row0.shape, [3]);
      expect(row0[0], 1);
      expect(row0[1], null);
      expect(row0[2], 3);

      // Select slice of columns
      final colSlice = marr[[Slice.all(), Slice(start: 1, stop: 3)]];
      expect(colSlice.shape, [2, 2]);
      expect(colSlice[[0, 0]], null); // originally [0, 1] (masked)
      expect(colSlice[[0, 1]], 3); // originally [0, 2]
      expect(colSlice[[1, 0]], 5); // originally [1, 1]
      expect(colSlice[[1, 1]], null); // originally [1, 2] (masked)
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
      expect(marr.mask.toList(), [
        false,
        false,
        false,
        true,
      ]); // index 1 is now unmasked
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
      expect(marr.mask.toList(), [
        false,
        false,
        true,
        false,
      ]); // index 2 (slice index 1) is now masked
    });
  });

  group('Arithmetic Methods', () {
    test('addition', () {
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
      expect(res.data.toList(), [11, 22, 33]); // 1+10, 2+20, 3+30
      expect(res.mask.toList(), [false, true, true]); // OR of masks
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
      expect(res.mask.toList(), [
        false,
        true,
        false,
      ]); // index 1 masked due to divisor 0
      expect(res[0], 2.0);
      expect(res[1], null);
      expect(res[2], 2.0);
    });

    test('division bypass for masked elements', () {
      // dividend is masked at index 1, divisor is 0 at index 1 but also masked.
      // It should NOT throw for integer division.
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
  });

  group('Reductions & Statistics', () {
    test('sum', () {
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

      final s0 = marr.sum(axis: 0);
      expect(s0.shape, [2]);
      expect(s0.data.toList(), [4, 4]);
      expect(s0.mask.toList(), [false, false]);

      // If a column is fully masked:
      final mask2 = NDArray.fromList(
        [false, true, false, true],
        [2, 2],
        DType.boolean,
      );
      final marr2 = MaskedArray(data, mask2);
      final s0_2 = marr2.sum(axis: 0);
      expect(s0_2.mask.toList(), [false, true]); // column 1 is fully masked
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

      final std = marr.std();
      expect(std.scalar, closeTo(math.sqrt(14.0 / 9.0), 1e-9));
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

      expect(marr.count().scalar, 3); // 3 unmasked elements

      final c0 = marr.count(axis: 0);
      expect(c0.toList(), [1, 0, 2]);
    });

    test('compressed 1D', () {
      final data = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, true],
        [4],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final compressed = marr.compressed();
      expect(compressed.shape, [2]);
      expect(compressed.toList(), [1, 3]);
    });

    test('compressed 2D', () {
      final data = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
      final mask = NDArray.fromList(
        [false, true, false, true, false, false],
        [2, 3],
        DType.boolean,
      );
      final marr = MaskedArray(data, mask);

      final compressed = marr.compressed();
      expect(compressed.shape, [4]);
      expect(compressed.toList(), [1, 3, 5, 6]);
    });

    test('reshape and transpose', () {
      final marr = MaskedArray<Float64>.zeros([2, 3], DType.float64);
      marr[[0, 1]] = null;

      final reshaped = marr.reshape([3, 2]);
      expect(reshaped.shape, [3, 2]);
      expect(reshaped[[0, 1]], null);

      final transposed = marr.transpose();
      expect(transposed.shape, [3, 2]);
      expect(transposed[[1, 0]], null);
    });
  });

  group('Conversions & Mapper', () {
    test('filled', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final mask = NDArray.fromList([false, true, false], [3], DType.boolean);
      final marr = MaskedArray(data, mask, fillValue: -99.0);

      final filledDefault = marr.filled();
      expect(filledDefault.toList(), [1.0, -99.0, 3.0]);

      final filledCustom = marr.filled(fillValue: 999.0);
      expect(filledCustom.toList(), [1.0, 999.0, 3.0]);
    });

    test('mapUnary', () {
      final data = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final mask = NDArray.fromList([false, true, false], [3], DType.boolean);
      final marr = MaskedArray(data, mask);

      final res = marr.mapUnary((x) => negative(x));
      expect(res[0], -1.0);
      expect(res[1], null);
      expect(res[2], -3.0);
    });
  });

  group('Runtime Type Preservation', () {
    test('runtime type preservation after operations', () {
      final a = MaskedArray<Float64>.zeros([3], DType.float64);
      final b = MaskedArray<Float64>.zeros([3], DType.float64);

      final res = a.add(b);
      expect(res, isA<MaskedArray<Float64>>());

      final resDiv = a.divide(b);
      expect(resDiv, isA<MaskedArray<Float64>>());

      final resSum = a.sum();
      expect(resSum, isA<MaskedArray<Float64>>());

      final resMean = a.mean();
      expect(resMean, isA<MaskedArray<Float64>>());

      final resVar = a.variance();
      expect(resVar, isA<MaskedArray<Float64>>());

      final resStd = a.std();
      expect(resStd, isA<MaskedArray<Float64>>());

      final resReshape = a.reshape([3, 1]);
      expect(resReshape, isA<MaskedArray<Float64>>());
    });

    test('runtime type preservation with mixed types', () {
      final a = MaskedArray<Float64>.zeros([3], DType.float64);
      final b = MaskedArray<Int32>.zeros([3], DType.int32);

      // Float64 + Int32 -> Float64
      final res = a.add(b);
      expect(res, isA<MaskedArray<Float64>>());

      // Int32 + Float64 -> Float64
      final res2 = b.add(a);
      expect(res2, isA<MaskedArray<Float64>>());
    });
  });

  group('Top-level Operations', () {
    test('top-level arithmetic', () {
      final a = MaskedArray(
        NDArray.fromList([1, 2], [2], DType.int32),
        NDArray.fromList([false, true], [2], DType.boolean),
      );
      final b = NDArray.fromList([10, 20], [2], DType.int32);

      final res = add(a, b);
      expect(res[0], 11);
      expect(res[1], null);
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

    test('preserve and coerce custom fill value', () {
      final a = MaskedArray(
        NDArray.fromList([1, 2, 3], [3], DType.int32),
        NDArray.fromList([false, true, false], [3], DType.boolean),
        fillValue: 42,
      );
      final b = MaskedArray(
        NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64),
        NDArray.fromList([false, false, true], [3], DType.boolean),
      );

      final res = a.add(b);
      expect(res.fillValue, 42.0);
      expect(res, isA<MaskedArray<Float64>>());
    });
  });

  group('Memory Management', () {
    test('scope auto-dispose', () {
      late NDArray rawData;
      late NDArray rawMask;

      final ma = NDArray.scope(() {
        final data = NDArray<Float64>.ones([2, 2], DType.float64);
        final mask = NDArray<Boolean>.zeros([2, 2], DType.boolean);
        rawData = data;
        rawMask = mask;
        return MaskedArray<Float64>(data, mask).detachToParentScope();
      });

      expect(rawData.isDisposed, false);
      expect(rawMask.isDisposed, false);

      ma.detachFromScope();
      ma.data.dispose();
      ma.mask.dispose();
    });
  });
}
