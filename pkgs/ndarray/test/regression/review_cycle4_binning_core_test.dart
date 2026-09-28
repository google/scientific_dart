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
  group('Review Cycle 4: Binning, Lstsq & Core Regressions', () {
    test(
      'bincount with strided out slice (step: 2) and negative-stride flipped out',
      () {
        NDArray.scope(() {
          final x = NDArray<Int32>.fromList(
            [0, 1, 1, 3, 2, 1, 3],
            [7],
            DType.int32,
          );
          // Expected bincount: [1, 3, 1, 2]

          // 1. Strided out slice (step: 2) with Int64 (direct kernel or copy)
          final buf64 = NDArray<Int64>.fromList(List<int>.filled(8, 99), [
            8,
          ], DType.int64);
          final stridedOut64 = buf64.slice([Slice(step: 2)]);
          final res64 = bincount<Int64>(x, out: stridedOut64);
          expect(identical(res64, stridedOut64), isTrue);
          expect(stridedOut64.toList(), equals([1, 3, 1, 2]));
          expect(buf64.toList(), equals([1, 99, 3, 99, 1, 99, 2, 99]));

          // 2. Strided out slice (step: 2) with Int32 (triggers _fastCopyAndCast)
          final buf32 = NDArray<Int32>.fromList(List<int>.filled(8, 77), [
            8,
          ], DType.int32);
          final stridedOut32 = buf32.slice([Slice(step: 2)]);
          final res32 = bincount<Int32>(x, out: stridedOut32);
          expect(identical(res32, stridedOut32), isTrue);
          expect(stridedOut32.toList(), equals([1, 3, 1, 2]));
          expect(buf32.toList(), equals([1, 77, 3, 77, 1, 77, 2, 77]));

          // 3. Negative-stride flipped out with Int64
          final flipBuf64 = NDArray<Int64>.fromList(
            [0, 0, 0, 0],
            [4],
            DType.int64,
          );
          final flippedOut64 = flip(flipBuf64);
          bincount<Int64>(x, out: flippedOut64);
          expect(flippedOut64.toList(), equals([1, 3, 1, 2]));
          expect(flipBuf64.toList(), equals([2, 1, 3, 1]));

          // 4. Negative-stride flipped out with Int32 (triggers _fastCopyAndCast)
          final flipBuf32 = NDArray<Int32>.fromList(
            [0, 0, 0, 0],
            [4],
            DType.int32,
          );
          final flippedOut32 = flip(flipBuf32);
          bincount<Int32>(x, out: flippedOut32);
          expect(flippedOut32.toList(), equals([1, 3, 1, 2]));
          expect(flipBuf32.toList(), equals([2, 1, 3, 1]));
        });
      },
    );

    test(
      'bincount memory aliasing protection: bincount(x, out: x) and bincount(x, weights: w, out: w)',
      () {
        NDArray.scope(() {
          // 1. bincount(x, out: x) where x is Int64
          final x64 = NDArray<Int64>.fromList([0, 1, 1, 2], [4], DType.int64);
          final resX64 = bincount<Int64>(x64, out: x64);
          expect(identical(resX64, x64), isTrue);
          expect(x64.toList(), equals([1, 2, 1, 0]));

          // 2. bincount(x, out: x) where x is Int32
          final x32 = NDArray<Int32>.fromList([0, 1, 1, 2], [4], DType.int32);
          final resX32 = bincount<Int32>(x32, out: x32);
          expect(identical(resX32, x32), isTrue);
          expect(x32.toList(), equals([1, 2, 1, 0]));

          // 3. bincount(x, weights: w, out: w) where w is Float64
          final xForW = NDArray<Int32>.fromList([0, 1, 1, 2], [4], DType.int32);
          final w64 = NDArray<Float64>.fromList(
            [0.5, 1.5, 2.5, 3.0],
            [4],
            DType.float64,
          );
          final resW64 = bincount<Float64>(xForW, weights: w64, out: w64);
          expect(identical(resW64, w64), isTrue);
          expect(w64.toList(), equals([0.5, 4.0, 3.0, 0.0]));

          // 4. bincount(x, weights: w, out: w) where w is Float32
          final w32 = NDArray<Float32>.fromList(
            [0.5, 1.5, 2.5, 3.0],
            [4],
            DType.float32,
          );
          final resW32 = bincount<Float32>(xForW, weights: w32, out: w32);
          expect(identical(resW32, w32), isTrue);
          expect(w32.toList(), equals([0.5, 4.0, 3.0, 0.0]));
        });
      },
    );

    test(
      'digitize with negative-stride flipped out (both Int32 and Int64)',
      () {
        NDArray.scope(() {
          final x = NDArray<Float64>.fromList(
            [0.2, 6.4, 3.0, 1.6],
            [4],
            DType.float64,
          );
          final bins = NDArray<Float64>.fromList(
            [0.0, 1.0, 2.5, 4.0, 10.0],
            [5],
            DType.float64,
          );
          // Expected digitize indices: [1, 4, 3, 2]

          // 1. Int32 flipped out via digitizeAs
          final buf32 = NDArray<Int32>.fromList([0, 0, 0, 0], [4], DType.int32);
          final flippedOut32 = flip(buf32);
          final res32 = digitizeAs(x, bins, DType.int32, out: flippedOut32);
          expect(identical(res32, flippedOut32), isTrue);
          expect(flippedOut32.toList(), equals([1, 4, 3, 2]));
          expect(buf32.toList(), equals([2, 3, 4, 1]));

          // 2. Int64 flipped out (default digitize)
          final buf64 = NDArray<Int64>.fromList([0, 0, 0, 0], [4], DType.int64);
          final flippedOut64 = flip(buf64);
          final res64 = digitize(x, bins, out: flippedOut64);
          expect(identical(res64, flippedOut64), isTrue);
          expect(flippedOut64.toList(), equals([1, 4, 3, 2]));
          expect(buf64.toList(), equals([2, 3, 4, 1]));
        });
      },
    );

    test(
      'histogram(a, bins: bins) does not steal or dispose caller bins array',
      () {
        final bins = NDArray<Float64>.fromList(
          [0.0, 1.0, 2.0, 3.0],
          [4],
          DType.float64,
        );
        final a = NDArray<Float64>.fromList(
          [0.5, 1.5, 1.8, 2.2],
          [4],
          DType.float64,
        );

        final res = histogram(a, bins: bins);
        expect(res.hist.toList(), equals([1, 2, 1]));
        expect(res.binEdges.toList(), equals([0.0, 1.0, 2.0, 3.0]));
        expect(identical(res.binEdges, bins), isFalse);

        // Disposing binEdges must leave original bins undisposed
        res.binEdges.dispose();
        res.hist.dispose();
        expect(bins.isDisposed, isFalse);
        expect(bins.toList(), equals([0.0, 1.0, 2.0, 3.0]));

        a.dispose();
        bins.dispose();
      },
    );

    test(
      'lstsq return record types have residuals and s typed as NDArray<Float64>',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            [3, 2],
            DType.float64,
          );
          final b = NDArray<Float64>.fromList(
            [1.0, 2.0, 2.0],
            [3],
            DType.float64,
          );

          final LstsqResult<Float64> res = lstsq<Float64, Float64, Float64>(
            a,
            b,
          );
          final NDArray<DTypeTag> residuals = res.residuals;
          final NDArray<DTypeTag> s = res.s;

          expect(residuals, isA<NDArray<Float64>>());
          expect(s, isA<NDArray<Float64>>());
          expect(residuals.size, equals(1));
          expect(s.size, equals(2));
        });
      },
    );
  });
}
