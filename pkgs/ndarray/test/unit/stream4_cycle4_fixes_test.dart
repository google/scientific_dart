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
  group('Stream 4 Cycle 4 Fixes', () {
    group('random choice multi-dimensional size', () {
      test('choice with multi-dimensional shape 2D', () {
        final a = NDArray<Int32>.fromList(
          [10, 20, 30, 40, 50].map((e) => Int32(e)).toList(),
          [5],
          DType.int32,
        );

        final sampled = choice(a, size: [2, 3], replace: true, seed: 42);
        expect(sampled.shape, [2, 3]);
        expect(sampled.size, 6);
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 3; j++) {
            expect(
              [10, 20, 30, 40, 50].contains(sampled.getCell([i, j]).toInt()),
              true,
            );
          }
        }
      });

      test('choice with multi-dimensional shape 3D without replacement', () {
        final a = NDArray<Int32>.arange(0, 24, dtype: DType.int32);
        final sampled = choice(a, size: [2, 3, 2], replace: false, seed: 123);
        expect(sampled.shape, [2, 3, 2]);
        expect(sampled.size, 12);

        final seen = <int>{};
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 3; j++) {
            for (var k = 0; k < 2; k++) {
              final v = sampled.getCell([i, j, k]).toInt();
              expect(seen.contains(v), false);
              seen.add(v);
            }
          }
        }
        expect(seen.length, 12);
      });

      test('choice with multi-dimensional shape and p array', () {
        final a = NDArray<Int32>.fromList(
          [1, 2, 3].map((e) => Int32(e)).toList(),
          [3],
          DType.int32,
        );
        final p = NDArray<Float64>.fromList(
          [0.1, 0.7, 0.2].map((e) => Float64(e)).toList(),
          [3],
          DType.float64,
        );

        final sampled = choice(a, size: [4, 5], p: p, replace: true, seed: 99);
        expect(sampled.shape, [4, 5]);
        expect(sampled.size, 20);
      });

      test('choice scalar (empty size)', () {
        final a = NDArray<Int32>.fromList(
          [5, 10, 15].map((e) => Int32(e)).toList(),
          [3],
          DType.int32,
        );
        final sampled = choice(a, size: [], seed: 7);
        expect(sampled.rank, 0);
        expect([5, 10, 15].contains(sampled.scalar.toInt()), true);
      });
    });

    group('manipulation data access & copy', () {
      test('diag with 1D input and out buffer', () {
        final v = NDArray<Int32>.fromList(
          [1, 2, 3].map((e) => Int32(e)).toList(),
          [3],
          DType.int32,
        );
        final out = NDArray<Int32>.ones([3, 3], DType.int32);
        final d = diag(v, out: out);
        expect(identical(d, out), true);
        expect(d.getCell([0, 0]).toInt(), 1);
        expect(d.getCell([1, 1]).toInt(), 2);
        expect(d.getCell([2, 2]).toInt(), 3);
        expect(d.getCell([0, 1]).toInt(), 0);
      });

      test('tril and triu non-contiguous', () {
        final base = NDArray<Int32>.arange(
          0,
          16,
          dtype: DType.int32,
        ).reshape([4, 4]);
        final view = base.slice([
          Slice(step: 2),
          Slice(step: 2),
        ]); // 2x2 non-contiguous
        expect(view.isContiguous, false);

        final lower = tril(view);
        expect(lower.getCell([0, 0]).toInt(), 0);
        expect(lower.getCell([0, 1]).toInt(), 0);
        expect(lower.getCell([1, 0]).toInt(), 8);
        expect(lower.getCell([1, 1]).toInt(), 10);

        final upper = triu(view);
        expect(upper.getCell([0, 0]).toInt(), 0);
        expect(upper.getCell([0, 1]).toInt(), 2);
        expect(upper.getCell([1, 0]).toInt(), 0);
        expect(upper.getCell([1, 1]).toInt(), 10);
      });

      test('diff with boolean and uint8 dtypes', () {
        final b = NDArray<bool>.fromList(
          [true, false, true, true],
          [4],
          DType.boolean,
        );
        final diffB = diff(b);
        expect(diffB.shape, [3]);

        final u = NDArray<Uint8>.fromList(
          [10, 25, 5, 40].map((e) => Uint8(e)).toList(),
          [4],
          DType.uint8,
        );
        final diffU = diff(u);
        expect(diffU.shape, [3]);
        expect(diffU.getCell([0]).toInt(), 15);
      });
    });

    group('stats ScratchArena and Float64 return types', () {
      test('std and variance Float64 typing', () {
        final a = NDArray<Int32>.fromList(
          [1, 2, 3, 4].map((e) => Int32(e)).toList(),
          [4],
          DType.int32,
        );

        final NDArray<Float64> v = variance(a);
        expect(v.scalar.toDouble(), closeTo(1.25, 1e-9));

        final NDArray<Float64> s = std(a);
        expect(s.scalar.toDouble(), closeTo(math.sqrt(1.25), 1e-9));

        final NDArray<Float64> nv = nanvar(a);
        expect(nv.scalar.toDouble(), closeTo(1.25, 1e-9));

        final NDArray<Float64> ns = nanstd(a);
        expect(ns.scalar.toDouble(), closeTo(math.sqrt(1.25), 1e-9));

        final out = NDArray<Float64>.create([], DType.float64);
        variance(a, out: out);
        expect(out.scalar.toDouble(), closeTo(1.25, 1e-9));
      });

      test('sum, mean, quantile, median ScratchArena reset verification', () {
        final a = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0].map((e) => Float64(e)).toList(),
          [2, 2],
          DType.float64,
        );

        for (var i = 0; i < 100; i++) {
          final s = sum(a, axis: 0);
          final m = mean(a, axis: 1);
          final q = quantile(a, 0.5, axis: 0);
          final med = median(a, axis: 1);
          expect(s.shape, [2]);
          expect(m.shape, [2]);
          expect(q.shape, [2]);
          expect(med.shape, [2]);
        }
      });
    });

    group('sorting argsort ScratchArena and out types', () {
      test('argsort int32 and int64 out', () {
        final a = NDArray<Float64>.fromList(
          [3.0, 1.0, 4.0, 2.0].map((e) => Float64(e)).toList(),
          [4],
          DType.float64,
        );

        final out32 = NDArray<Int32>.create([4], DType.int32);
        final res32 = argsort(a, out: out32);
        expect(identical(res32, out32), true);
        expect(out32.toList().map((e) => e.toInt()).toList(), [1, 3, 0, 2]);

        final out64 = NDArray<Int64>.create([4], DType.int64);
        final res64 = argsort(a, out: out64);
        expect(identical(res64, out64), true);
        expect(out64.toList().map((e) => e.toInt()).toList(), [1, 3, 0, 2]);
      });
    });

    group('indexing ScratchArena try/finally', () {
      test('take_along_axis and put_along_axis', () {
        final a = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0, 50.0, 60.0].map((e) => Float64(e)).toList(),
          [2, 3],
          DType.float64,
        );
        final idx = NDArray<Int32>.fromList(
          [2, 0, 1, 1].map((e) => Int32(e)).toList(),
          [2, 2],
          DType.int32,
        );

        final taken = take_along_axis(a, idx, 1);
        expect(taken.shape, [2, 2]);
        expect(taken.getCell([0, 0]).toDouble(), 30.0);
        expect(taken.getCell([0, 1]).toDouble(), 10.0);

        final out = a.copy();
        final vals = NDArray<Float64>.fromList(
          [99.0, 88.0, 77.0, 66.0].map((e) => Float64(e)).toList(),
          [2, 2],
          DType.float64,
        );
        put_along_axis(out, idx, vals, 1);
        expect(out.getCell([0, 2]).toDouble(), 99.0);
        expect(out.getCell([0, 0]).toDouble(), 88.0);
      });

      test('choose and select', () {
        final a = NDArray<Int32>.fromList(
          [0, 1, 0].map((e) => Int32(e)).toList(),
          [3],
          DType.int32,
        );
        final c0 = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0].map((e) => Float64(e)).toList(),
          [3],
          DType.float64,
        );
        final c1 = NDArray<Float64>.fromList(
          [100.0, 200.0, 300.0].map((e) => Float64(e)).toList(),
          [3],
          DType.float64,
        );

        final chosen = choose(a, [c0, c1]);
        expect(chosen.toList().map((e) => (e as Float64).toDouble()).toList(), [
          10.0,
          200.0,
          30.0,
        ]);

        final cond1 = NDArray<bool>.fromList(
          [true, false, false],
          [3],
          DType.boolean,
        );
        final cond2 = NDArray<bool>.fromList(
          [false, true, false],
          [3],
          DType.boolean,
        );
        final selected = select(
          [cond1, cond2],
          [c0, c1],
          defaultValue: Float64(999.0),
        );
        expect(selected.toList().map((e) => (e as num).toDouble()).toList(), [
          10.0,
          200.0,
          999.0,
        ]);
      });
    });

    group('calculus spacingArray cleanup in trapz and gradient', () {
      test('trapz with List<double> spacing', () {
        final y = NDArray<Float64>.fromList(
          [1.0, 4.0, 9.0, 16.0].map((e) => Float64(e)).toList(),
          [4],
          DType.float64,
        );
        final result = trapz(
          y,
          spacing: Spacing.coordinates([0.0, 1.0, 2.0, 3.0]),
        );
        expect(result.scalar.toDouble(), closeTo(21.5, 1e-9));
      });

      test('gradient with List<double> spacing', () {
        final f = NDArray<Float64>.fromList(
          [1.0, 2.0, 4.0, 7.0, 11.0, 16.0].map((e) => Float64(e)).toList(),
          [6],
          DType.float64,
        );
        final grad = gradient(
          f,
          spacing: Spacing.coordinates([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        expect(grad.shape, [6]);
      });
    });
  });
}
