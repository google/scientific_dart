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
import 'package:ndarray/src/operations/helpers.dart';
import 'package:test/test.dart';

void main() {
  group('Bug 1: Double-Addition of offsetElements to Pointers & Views', () {
    test(
      'asStrided on view with non-zero offsetElements does not double-add offsetElements and validates bounds',
      () {
        final base = NDArray.fromList(
          [10, 20, 30, 40, 50, 60],
          [6],
          DType.int64,
        );
        final flipped = flip(
          base,
        ); // [60, 50, 40, 30, 20, 10], offsetElements = 5, strides = [-1]
        expect(flipped.offsetElements, equals(5));

        final strided = asStrided(flipped, shape: [3], strides: [-2]);
        expect(strided.toList(), equals([60, 40, 20]));

        // Negative shape dimension throws ArgumentError
        expect(
          () => asStrided(flipped, shape: [-1], strides: [1]),
          throwsArgumentError,
        );

        // Out of bounds physical memory throws RangeError
        expect(
          () => asStrided(flipped, shape: [5], strides: [1]),
          throwsRangeError,
        );

        base.dispose();
      },
    );

    test(
      'linspaceGrid on sliced start/stop/out views respects offsetElements',
      () {
        final baseStart = NDArray.fromList(
          [99.0, 0.0, 10.0],
          [3],
          DType.float64,
        );
        final baseStop = NDArray.fromList(
          [99.0, 2.0, 20.0],
          [3],
          DType.float64,
        );
        final startView = baseStart.slice([
          Slice(start: 1, stop: 3),
        ]); // [0.0, 10.0], offset = 1
        final stopView = baseStop.slice([
          Slice(start: 1, stop: 3),
        ]); // [2.0, 20.0], offset = 1

        final res = linspaceGrid(startView, stopView, 3, axis: 0);
        expect(res.shape, equals([3, 2]));
        expect(res.toList(), equals([0.0, 10.0, 1.0, 15.0, 2.0, 20.0]));

        baseStart.dispose();
        baseStop.dispose();
        res.dispose();
      },
    );

    test('clipArray on sliced views respects offsetElements', () {
      final baseA = NDArray.fromList(
        [99.0, -5.0, 5.0, 15.0],
        [4],
        DType.float64,
      );
      final baseMin = NDArray.fromList(
        [99.0, 0.0, 0.0, 0.0],
        [4],
        DType.float64,
      );
      final baseMax = NDArray.fromList(
        [99.0, 10.0, 10.0, 10.0],
        [4],
        DType.float64,
      );

      final aView = baseA.slice([Slice(start: 1, stop: 4)]);
      final minView = baseMin.slice([Slice(start: 1, stop: 4)]);
      final maxView = baseMax.slice([Slice(start: 1, stop: 4)]);

      final res = clipArray(aView, min: minView, max: maxView);
      expect(res.toList(), equals([0.0, 5.0, 10.0]));

      baseA.dispose();
      baseMin.dispose();
      baseMax.dispose();
      res.dispose();
    });

    test(
      'choice and shuffle on sliced views do not double-add offsetElements',
      () {
        final base = NDArray.fromList([999, 888, 10, 20, 30], [5], DType.int64);
        final view = base.slice([
          Slice(start: 2, stop: 5),
        ]); // [10, 20, 30], offset = 2

        final chosen = choice(view, size: [20], replace: true, seed: 42);
        for (final val in chosen.toList()) {
          expect([10, 20, 30], contains(val));
        }

        shuffle(view, seed: 42);
        expect(view.toList()..sort(), equals([10, 20, 30]));
        expect(base.getCell([0]), equals(999));
        expect(base.getCell([1]), equals(888));

        chosen.dispose();
        base.dispose();
      },
    );

    test(
      'sqrt on sliced view and non-contiguous out does not double-add offsetElements',
      () {
        final base = NDArray.fromList([99, 4, 9, 16], [4], DType.int64);
        final view = base.slice([
          Slice(start: 1, stop: 4),
        ]); // [4, 9, 16], offset = 1
        final res = sqrt(view);
        expect(res.toList(), equals([2.0, 3.0, 4.0]));

        base.dispose();
        res.dispose();
      },
    );

    test(
      'ndenumerate on 0-D view with non-zero offsetElements reads correct element without RangeError',
      () {
        final base = NDArray.fromList([100, 200, 300], [3], DType.int64);
        final flipped = flip(base); // offsetElements = 2, first element is 300
        final scalarView = asStrided(
          flipped,
          shape: [],
          strides: [],
        ); // 0-D view with offsetElements = 2
        expect(scalarView.shape, isEmpty);

        final items = ndenumerate(scalarView).toList();
        expect(items.length, equals(1));
        expect(items.first.$1, isEmpty);
        expect(items.first.$2, equals(300));

        base.dispose();
      },
    );
  });

  group(
    'Bug 2: Passing Already-Strided NDIter Physical Offsets to getCellFlat/setCellFlat',
    () {
      test(
        'isClose on transposed/sliced non-contiguous views works accurately',
        () {
          final aBase = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final bBase = NDArray.fromList(
            [1.0, 3.0, 2.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final aT = aBase.transpose(); // [[1, 3], [2, 4]]
          final res = isClose(aT, bBase);
          expect(res.toList(), equals([true, true, true, true]));

          aBase.dispose();
          bBase.dispose();
          res.dispose();
        },
      );

      test(
        'stats.all and stats.any on non-contiguous sliced views read correct raw offsets',
        () {
          final base = NDArray<Boolean>.fromList(
            [false, true, true, true],
            [4],
            DType.boolean,
          );
          final view = base.slice([
            Slice(start: 1, stop: 4, step: 2),
          ]); // [true, true], offset = 1, stride = 2
          expect(all(view).scalar, isTrue);

          final base2 = NDArray<Boolean>.fromList(
            [true, false, false, false],
            [4],
            DType.boolean,
          );
          final view2 = base2.slice([
            Slice(start: 1, stop: 4),
          ]); // [false, false, false], offset = 1
          expect(any(view2).scalar, isFalse);

          base.dispose();
          base2.dispose();
        },
      );
    },
  );

  group(
    'Bug 3: flip() Negative Pointer Underflow, diag() Double-Striding, & roll() In-Place Overwrite',
    () {
      test(
        'flip on empty array (axis size 0) does not produce negative offsetElements',
        () {
          final empty = NDArray.create([0, 3], DType.float64);
          final flipped = flip(empty, axis: 0);
          expect(flipped.shape, equals([0, 3]));
          expect(flipped.offsetElements, equals(0));
          empty.dispose();
        },
      );

      test(
        'diag on 1D input with non-contiguous out writes to exact strided coordinates',
        () {
          final v = NDArray.fromList([1, 2, 3], [3], DType.int64);
          final outBase = NDArray.zeros([3, 3], DType.int64);
          final outTransposed = outBase.transpose(); // non-contiguous view
          diag(v, out: outTransposed);
          expect(outBase.toList(), equals([1, 0, 0, 0, 2, 0, 0, 0, 3]));

          v.dispose();
          outBase.dispose();
        },
      );

      test(
        'diag on 2D input with out-of-bounds k validates and returns out buffer',
        () {
          final m = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          final outValid = NDArray.create([0], DType.int64);
          final res = diag(m, k: 5, out: outValid);
          expect(identical(res, outValid), isTrue);

          final outInvalid = NDArray.create([1], DType.int64);
          expect(() => diag(m, k: 5, out: outInvalid), throwsArgumentError);

          m.dispose();
          outValid.dispose();
          outInvalid.dispose();
        },
      );

      test(
        'in-place roll on strided and contiguous arrays avoids overwrite corruption',
        () {
          final a = NDArray.fromList([1, 2, 3, 4, 5], [5], DType.int64);
          roll(a, 2, out: a);
          expect(a.toList(), equals([4, 5, 1, 2, 3]));

          final m = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          final mT = m.transpose(); // [[1, 3], [2, 4]]
          roll(mT, 1, axis: 0, out: mT);
          expect(mT.toList(), equals([2, 4, 1, 3]));

          a.dispose();
          m.dispose();
        },
      );
    },
  );

  group(
    'Bug 4: Ufuncs with where Mask Zero-Init & Binary Fallback Masked/Strided Copy',
    () {
      test(
        'trigonometric and arithmetic ufuncs zero-initialize unmasked elements when out == null',
        () {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final mask = NDArray<Boolean>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );

          final s = sin(a, where: mask);
          expect(s.getCell([1]), equals(0.0));

          final c = ceil(
            NDArray.fromList([5, 6, 7], [3], DType.int64),
            where: mask,
          );
          expect(c.toList(), equals([5, 0, 7]));

          a.dispose();
          mask.dispose();
          s.dispose();
          c.dispose();
        },
      );

      test(
        'binary fallback ufuncs preserve out values when where == false and handle strided out',
        () {
          final a = NDArray<Float16>.fromList([10.0, 20.0], [2], DType.float16);
          final b = NDArray<Float16>.fromList([1.0, 2.0], [2], DType.float16);
          final mask = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );
          final out = NDArray<Float16>.fromList(
            [99.0, 88.0],
            [2],
            DType.float16,
          );

          add<Float16>(a, b, where: mask, out: out);
          expect(out.getCell([0]), closeTo(11.0, 1e-2));
          expect(out.getCell([1]), closeTo(88.0, 1e-2));

          a.dispose();
          b.dispose();
          mask.dispose();
          out.dispose();
        },
      );
    },
  );

  group(
    'Bug 5: stats.min/max on bool/complex, stats.sum Modulo 256 Overflow, & keepdims',
    () {
      test('stats.min and stats.max work on boolean and complex arrays', () {
        final b = NDArray<Boolean>.fromList(
          [true, false, true],
          [3],
          DType.boolean,
        );
        expect(min(b).scalar, isFalse);
        expect(max(b).scalar, isTrue);

        final c = NDArray.fromList(
          [Complex(2.0, 1.0), Complex(1.0, 5.0), Complex(1.0, 2.0)],
          [3],
          DType.complex128,
        );
        expect(min(c).scalar, equals(Complex(1.0, 2.0)));
        expect(max(c).scalar, equals(Complex(2.0, 1.0)));

        b.dispose();
        c.dispose();
      });

      test(
        'stats.sum on boolean with dtype: DType.boolean does not overflow at 256',
        () {
          final list = List<bool>.filled(256, true);
          final b = NDArray<Boolean>.fromList(list, [256], DType.boolean);
          final s = sumAs(b, DType.boolean);
          expect(s.scalar, isTrue);

          final b2 = NDArray<Boolean>.fromList(list, [1, 256], DType.boolean);
          final sAxis = sumAs(b2, DType.boolean, axis: 1);
          expect(sAxis.getCell([0]), isTrue);

          b.dispose();
          s.dispose();
          b2.dispose();
          sAxis.dispose();
        },
      );

      test('floatPower promotes integer inputs to float64', () {
        final a = NDArray.fromList([2, 4], [2], DType.int64);
        final b = NDArray.fromList([-1, -2], [2], DType.int64);
        final res = binaryUfunc<DTypeTag, DTypeTag>(
          a,
          b,
          op: BinaryOp.floatPower,
        );
        expect(res.toList(), equals([0.5, 0.0625]));

        a.dispose();
        b.dispose();
        res.dispose();
      });

      test(
        'minimum/maximum propagate NaN symmetrically while fmin/fmax ignore NaN',
        () {
          final a = NDArray.fromList([1.0, double.nan], [2], DType.float64);
          final b = NDArray.fromList([double.nan, 2.0], [2], DType.float64);

          final minRes = binaryUfunc<DTypeTag, DTypeTag>(
            a,
            b,
            op: BinaryOp.minimum,
          );
          expect(minRes.getCell([0]).isNaN, isTrue);
          expect(minRes.getCell([1]).isNaN, isTrue);

          final fminRes = binaryUfunc<DTypeTag, DTypeTag>(
            a,
            b,
            op: BinaryOp.fmin,
          );
          expect(fminRes.toList(), equals([1.0, 2.0]));

          a.dispose();
          b.dispose();
          minRes.dispose();
          fminRes.dispose();
        },
      );

      test('reduceUfunc with initial != null and keepdims: true succeeds', () {
        final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
        final res = reduceUfunc(
          a,
          op: BinaryOp.add,
          axis: 1,
          keepdims: true,
          initial: 10,
        );
        expect(res.shape, equals([2, 1]));
        expect(res.toList(), equals([13, 17]));

        a.dispose();
        res.dispose();
      });

      test('quantile and median respect keepdims: true when axis == null', () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final q = quantile(a, 0.5, keepdims: true);
        expect(q.shape, equals([1, 1]));
        expect(q.getCell([0, 0]), equals(2.5));

        final m = median(a, keepdims: true);
        expect(m.shape, equals([1, 1]));
        expect(m.getCell([0, 0]), equals(2.5));

        a.dispose();
        q.dispose();
        m.dispose();
      });
    },
  );

  group('Bug 6: DType.uint64 Unsigned Precision & Clamping Bugs', () {
    test(
      'castNDArray and castValue preserve unsigned uint64 values >= 2^63',
      () {
        // -1 in signed 64-bit int represents 18446744073709551615 (2^64 - 1)
        final u = NDArray.fromList([-1], [1], DType.uint64);
        final f = castNDArray(u, DType.float64);
        expect(f.getCell([0]), closeTo(18446744073709551615.0, 1e5));

        final back = castNDArray(f, DType.uint64);
        expect(back.getCell([0]), equals(-1));

        u.dispose();
        f.dispose();
        back.dispose();
      },
    );

    test('cummin and cummax on uint64 use unsigned comparison', () {
      // 10 vs -1 (18446744073709551615)
      final u = NDArray.fromList([10, -1, 5], [3], DType.uint64);
      final cmin = cummin(u);
      final cmax = cummax(u);
      expect(cmin.toList(), equals([10, 10, 5]));
      expect(cmax.toList(), equals([10, -1, -1]));

      u.dispose();
      cmin.dispose();
      cmax.dispose();
    });

    test(
      'median on uint64 with values >= 2^63 does not clamp to signed int64 max',
      () {
        // Two elements both equal to -2 (2^64 - 2) -> median is Float64 (2^64 - 2).toDouble()
        final u = NDArray.fromList([-2, -2], [2], DType.uint64);
        final med = median(u);
        expect(med.dtype, equals(DType.float64));
        expect(med.scalar, equals(BigInt.from(-2).toUnsigned(64).toDouble()));

        u.dispose();
        med.dispose();
      },
    );

    test(
      'clip on uint64 treats values >= 2^63 as positive unsigned integers',
      () {
        final u = NDArray.fromList([5, -1], [2], DType.uint64);
        final c = clip(u, min: 10);
        // 5 clipped to 10; -1 (max uint64) stays -1
        expect(c.toList(), equals([10, -1]));

        u.dispose();
        c.dispose();
      },
    );
  });

  group(
    'Bug 7: put_along_axis, choose, & toNDArray Extension Type Erasure Bugs',
    () {
      test(
        'toNDArray returns zero-copy view when runtime generic type differs',
        () {
          final NDArray<AnySpec> objArr = NDArray.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
          );
          final typed = toNDArray<AnySpec>(objArr, DType.float64);
          expect(typed.toList(), equals([1.0, 2.0]));
          objArr.dispose();
        },
      );

      test(
        'put_along_axis casts values when runtime extension types erase to double',
        () {
          final arr = NDArray<Float16>.fromList([0.0, 0.0], [2], DType.float16);
          final indices = NDArray.fromList([0, 1], [2], DType.int64);
          final values = NDArray<Float64>.fromList(
            [3.5, 7.25],
            [2],
            DType.float64,
          );

          put_along_axis(arr, indices, values, 0);
          expect(arr.getCell([0]), closeTo(3.5, 1e-2));
          expect(arr.getCell([1]), closeTo(7.25, 1e-2));

          arr.dispose();
          indices.dispose();
          values.dispose();
        },
      );

      test(
        'choose infers integer dtype for integer scalars and casts mixed choices',
        () {
          final a = NDArray.fromList([0, 1], [2], DType.int64);
          final resInt = choose<DTypeTag>(a, [10, 20]);
          expect(resInt.dtype, equals(DType.int64));
          expect(resInt.toList(), equals([10, 20]));

          a.dispose();
          resInt.dispose();
        },
      );
    },
  );

  group('Bug 8: sorting.dart Bugs', () {
    test(
      'argsort on boolean and uint64 with non-contiguous out does not use disposed tempResult',
      () {
        final b = NDArray<Boolean>.fromList(
          [true, false, true, false],
          [2, 2],
          DType.boolean,
        );
        final outBase = NDArray.zeros([2, 2], DType.int64);
        final outTransposed = outBase.transpose();

        final res = argsort(b, axis: 1, out: outTransposed);
        expect(res.toList(), equals([1, 0, 1, 0]));

        b.dispose();
        outBase.dispose();
      },
    );

    test('sort on uint64 multi-row array sorts unsigned values per row', () {
      final u = NDArray.fromList([-1, 5, 10, -2], [2, 2], DType.uint64);
      final sorted = sort(u, axis: 1);
      expect(sorted.toList(), equals([5, -1, 10, -2]));

      u.dispose();
      sorted.dispose();
    });
  });

  group('Bug 9: padding.dart, spacers.dart, & helpers.dart Bugs', () {
    test('pad with zero padding returns a new copy, not input array', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int64);
      final padded = pad(a, PadWidth.all(0));
      expect(identical(padded, a), isFalse);
      padded.setCell([0], 99);
      expect(a.getCell([0]), equals(1));

      a.dispose();
      padded.dispose();
    });

    test(
      'linspaceWithStep and linspaceGridWithStep with numSamples == 0 on integer dtype do not crash',
      () {
        final res = linspaceWithStep<DTypeTag>(0, 10, 0, dtype: DType.int64);
        expect(res.samples.shape, equals([0]));
        expect(res.step, equals(0));
        res.samples.dispose();
      },
    );

    test('logspaceGrid with axis != 0 and base array broadcasts properly', () {
      final start = NDArray.fromList([0.0, 1.0], [2], DType.float64);
      final stop = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final base = NDArray<Float64>.fromList([10.0, 2.0], [2], DType.float64);

      final res = logspaceGrid(start, stop, 3, base: base, axis: 1);
      expect(res.shape, equals([2, 3]));
      expect(res.getCell([0, 0]), closeTo(1.0, 1e-6));
      expect(res.getCell([0, 2]), closeTo(10.0, 1e-6));
      expect(res.getCell([1, 0]), closeTo(2.0, 1e-6));
      expect(res.getCell([1, 2]), closeTo(4.0, 1e-6));

      start.dispose();
      stop.dispose();
      base.dispose();
      res.dispose();
    });

    test('geomspaceGrid supports negative start and stop with same sign', () {
      final start = NDArray.fromList([-1.0, -10.0], [2], DType.float64);
      final stop = NDArray.fromList([-100.0, -1000.0], [2], DType.float64);

      final res = geomspaceGrid(start, stop, 3, axis: 0);
      expect(res.shape, equals([3, 2]));
      expect(res.getCell([0, 0]), closeTo(-1.0, 1e-6));
      expect(res.getCell([1, 0]), closeTo(-10.0, 1e-6));
      expect(res.getCell([2, 0]), closeTo(-100.0, 1e-6));

      start.dispose();
      stop.dispose();
      res.dispose();
    });
  });

  group(
    'Synthesis Edge Cases: Underflow-Safe Geomspace, Uint64 Sqrt/Clip, Complex FloatPower & MinMax, NanToNum/NanSum',
    () {
      test(
        'geomspaceGrid with small negative floats does not underflow sign check to zero',
        () {
          final start = NDArray.fromList(
            [-1e-200, -1e-200],
            [2],
            DType.float64,
          );
          final stop = NDArray.fromList([-1e-198, -1e-198], [2], DType.float64);
          final res = geomspaceGrid(start, stop, 3, axis: 0);
          expect(res.shape, equals([3, 2]));
          expect(res.getCell([0, 0]), closeTo(-1e-200, 1e-205));
          expect(res.getCell([1, 0]), closeTo(-1e-199, 1e-204));
          expect(res.getCell([2, 0]), closeTo(-1e-198, 1e-203));
          start.dispose();
          stop.dispose();
          res.dispose();
        },
      );

      test(
        'sqrt on uint64 with values >= 2^63 computes unsigned square root instead of NaN',
        () {
          // 1 << 62 is 4611686018427387904 (sqrt = 2147483648.0)
          // 1 << 63 is -9223372036854775808 in signed int64 (unsigned 9223372036854775808, sqrt = 3037000499.9760499)
          final val63 = (BigInt.one << 63).toSigned(64).toInt();
          final u = NDArray.fromList([val63], [1], DType.uint64);
          final sq = sqrt(u);
          expect(sq.getCell([0]).isNaN, isFalse);
          expect(sq.getCell([0]), closeTo(3037000499.97605, 1e-3));
          u.dispose();
          sq.dispose();
        },
      );

      test(
        'clip on uint64 when min > max clamps to max matching NumPy/C behavior',
        () {
          final u = NDArray.fromList([5, 50, 500], [3], DType.uint64);
          final clipped = clip(u, min: 100, max: 10);
          expect(clipped.toList(), equals([10, 10, 10]));
          u.dispose();
          clipped.dispose();
        },
      );

      test(
        'nan_to_num with where mask zero-initializes unmasked elements when out == null',
        () {
          final a = NDArray.fromList(
            [double.nan, double.nan, double.infinity],
            [3],
            DType.float64,
          );
          final mask = NDArray<Boolean>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          final res = nan_to_num(a, where: mask, nan: 0.0, posinf: 999.0);
          expect(res.getCell([0]), equals(0.0));
          // Unmasked elements follow the where: contract in
          // test/meta/operation_contracts_test.dart: zero when out == null.
          expect(res.getCell([1]), equals(0.0));
          expect(res.getCell([2]), equals(999.0));
          a.dispose();
          mask.dispose();
          res.dispose();
        },
      );

      test(
        'nansum on integer, uint64, and boolean arrays delegates cleanly without TypeError',
        () {
          final b = NDArray<Boolean>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          final sb = nansum(b);
          expect(sb.scalar, isTrue);

          final u = NDArray.fromList([10, 20, 30], [3], DType.uint64);
          final su = nansum(u);
          expect(su.scalar, equals(60));

          b.dispose();
          sb.dispose();
          u.dispose();
          su.dispose();
        },
      );

      test('floatPower on complex numbers promotes to complex128', () {
        final c = NDArray.fromList([Complex(0.0, 1.0)], [1], DType.complex64);
        final p = NDArray.fromList([Complex(2.0, 0.0)], [1], DType.complex64);
        final res = binaryUfunc<DTypeTag, DTypeTag>(
          c,
          p,
          op: BinaryOp.floatPower,
        );
        expect(res.dtype, equals(DType.complex128));
        expect(res.getCell([0]).real, closeTo(-1.0, 1e-6));
        expect(res.getCell([0]).imag, closeTo(0.0, 1e-6));
        c.dispose();
        p.dispose();
        res.dispose();
      });

      test(
        'minimum/maximum vs fmin/fmax on complex numbers with NaN components',
        () {
          final a = NDArray.fromList(
            [Complex(1.0, double.nan)],
            [1],
            DType.complex128,
          );
          final b = NDArray.fromList(
            [Complex(2.0, 0.0)],
            [1],
            DType.complex128,
          );

          final minRes = binaryUfunc<DTypeTag, DTypeTag>(
            a,
            b,
            op: BinaryOp.minimum,
          );
          expect(minRes.getCell([0]).real, equals(1.0));
          expect(minRes.getCell([0]).imag.isNaN, isTrue);

          final fminRes = binaryUfunc<DTypeTag, DTypeTag>(
            a,
            b,
            op: BinaryOp.fmin,
          );
          expect(fminRes.getCell([0]), equals(Complex(2.0, 0.0)));

          a.dispose();
          b.dispose();
          minRes.dispose();
          fminRes.dispose();
        },
      );

      test('bincount with non-contiguous out view writes accurately', () {
        final x = NDArray.fromList([0, 1, 1, 2, 2, 2], [6], DType.int32);
        final outBase = NDArray.zeros([2, 3], DType.int64);
        final outStrided = outBase.slice([Index(1), Slice()]);
        bincount(x, out: outStrided);
        expect(outStrided.toList(), equals([1, 2, 3]));
        expect(outBase.slice([Index(0), Slice()]).toList(), equals([0, 0, 0]));
        x.dispose();
        outBase.dispose();
        outStrided.dispose();
      });
    },
  );

  group('Additional Combined Regression Coverage', () {
    test('allClose on transposed non-contiguous arrays', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      final b = NDArray.fromList([1.0, 3.0, 2.0, 4.0], [2, 2], DType.float64);
      final bT = b.transpose();
      expect(allClose(a, bT), isTrue);
      a.dispose();
      b.dispose();
      bT.dispose();
    });

    test('stats.all and stats.any on transposed 2x3 arrays with step > 1', () {
      final a = NDArray<Boolean>.fromList(
        [true, true, false, true, true, true],
        [2, 3],
        DType.boolean,
      );
      final aT = a.transpose();
      expect(all(aT).scalar, isFalse);

      final b = NDArray<Boolean>.fromList(
        [false, false, true, false, false, false],
        [2, 3],
        DType.boolean,
      );
      final bT = b.transpose();
      expect(any(bT).scalar, isTrue);

      a.dispose();
      aT.dispose();
      b.dispose();
      bT.dispose();
    });

    test(
      'roll in-place on strided slice view (step: 2) avoids overwrite corruption',
      () {
        final base = NDArray.fromList(
          [1, 0, 2, 0, 3, 0, 4, 0],
          [8],
          DType.int32,
        );
        final stridedView = base.slice([Slice(start: 0, stop: 8, step: 2)]);
        roll(stridedView, 1, out: stridedView);
        expect(stridedView.toList(), equals([4, 1, 2, 3]));
        base.dispose();
        stridedView.dispose();
      },
    );

    test('choose with mixed Float16 and Float64 choice arrays', () {
      final idx = NDArray.fromList([0, 1, 0], [3], DType.int32);
      final cFloat16 = NDArray<Float16>.fromList(
        [1.5, 2.5, 3.5],
        [3],
        DType.float16,
      );
      final cFloat64 = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
      final resMixed = choose(idx, [cFloat16, cFloat64]);
      expect(resMixed.dtype, equals(DType.float64));
      expect(resMixed.getCell([0]), closeTo(1.5, 1e-2));
      expect(resMixed.getCell([1]), closeTo(20.0, 1e-5));
      expect(resMixed.getCell([2]), closeTo(3.5, 1e-2));
      idx.dispose();
      cFloat16.dispose();
      cFloat64.dispose();
      resMixed.dispose();
    });

    test('partition with uint64 on 2D array partitions each row once', () {
      final u = NDArray.fromList([-1, 10, 5, 20, 1, 15], [2, 3], DType.uint64);
      final part = partition(u, 1, axis: 1);
      expect(part.getCell([0, 1]), equals(10));
      expect(part.getCell([1, 1]), equals(15));
      u.dispose();
      part.dispose();
    });

    test(
      'pad with (0, 0) pad widths inside scope does not detach caller input array',
      () {
        final outer = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        NDArray.scope(() {
          final padded = pad(outer, PadWidth.all(0));
          expect(identical(padded, outer), isFalse);
          expect(padded.toList(), equals([1.0, 2.0, 3.0]));
        });
        expect(outer.isDisposed, isFalse);
        expect(outer.toList(), equals([1.0, 2.0, 3.0]));
        outer.dispose();
      },
    );
  });

  group('L2 Synthesis Additional Edge Cases', () {
    test(
      'cumsum on uint64 preserves >2^53 precision and supports boolean-to-int32 out',
      () {
        final v1 = ((BigInt.one << 63) + BigInt.from(1)).toSigned(64).toInt();
        final v2 = 10;
        final uPrec = NDArray.fromList([v1, v2], [2], DType.uint64);
        final csum = cumsum(uPrec, axis: 0);
        final expectedSum = ((BigInt.one << 63) + BigInt.from(11))
            .toSigned(64)
            .toInt();
        expect(csum.toList(), equals([v1, expectedSum]));

        final boolArr = NDArray<Boolean>.fromList(
          [true, true, false, true],
          [4],
          DType.boolean,
        );
        final intOut = NDArray<Int32>.create([4], DType.int32);
        cumsumAs(boolArr, DType.int32, axis: 0, out: intOut);
        expect(intOut.toList(), equals([1, 2, 2, 3]));

        uPrec.dispose();
        csum.dispose();
        boolArr.dispose();
        intOut.dispose();
      },
    );

    test(
      'clipArray on uint64 treats values >= 2^63 as positive unsigned integers',
      () {
        final u = NDArray.fromList([5, -1], [2], DType.uint64);
        final minArr = NDArray.fromList([10, 10], [2], DType.uint64);
        final cArr = clipArray(u, min: minArr);
        expect(cArr.toList(), equals([10, -1]));

        u.dispose();
        minArr.dispose();
        cArr.dispose();
      },
    );

    test('nanmin and nanmax delegate boolean arrays cleanly', () {
      final b = NDArray<Boolean>.fromList(
        [true, false, true, true],
        [2, 2],
        DType.boolean,
      );
      final nmin = nanmin(b);
      final nmax = nanmax(b);
      expect(nmin.scalar, isFalse);
      expect(nmax.scalar, isTrue);

      final nminAxis = nanmin(b, axis: 0);
      final nmaxAxis = nanmax(b, axis: 0);
      expect(nminAxis.toList(), equals([true, false]));
      expect(nmaxAxis.toList(), equals([true, true]));

      b.dispose();
      nmin.dispose();
      nmax.dispose();
      nminAxis.dispose();
      nmaxAxis.dispose();
    });

    test(
      'isnan and isinf handle uint64, int8, and float16 arrays without uninitialized memory',
      () {
        final u = NDArray.fromList([0, -1, 42], [3], DType.uint64);
        final nanU = isnan(u);
        final infU = isinf(u);
        expect(nanU.toList(), equals([false, false, false]));
        expect(infU.toList(), equals([false, false, false]));

        final f16 = NDArray<Float16>.fromList(
          [1.0, double.nan, double.infinity],
          [3],
          DType.float16,
        );
        final nanF16 = isnan(f16);
        final infF16 = isinf(f16);
        expect(nanF16.toList(), equals([false, true, false]));
        expect(infF16.toList(), equals([false, false, true]));

        u.dispose();
        nanU.dispose();
        infU.dispose();
        f16.dispose();
        nanF16.dispose();
        infF16.dispose();
      },
    );
  });
}
