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
  group('Review Cycle 8 Manipulation Aliasing & Non-Contiguous out Tests', () {
    test('pad with out aliasing src slice across modes', () {
      NDArray.scope(() {
        // PadMode.constant
        final buf1 = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0, 99.0, 99.0],
          [6],
          DType.float64,
        );
        final src1 = buf1.slice([Slice(start: 0, stop: 4)]);
        pad(src1, PadWidth.all(1), mode: PadMode.constant, out: buf1);
        expect(buf1.toList(), equals([0.0, 10.0, 20.0, 30.0, 40.0, 0.0]));

        // PadMode.edge
        final buf2 = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0, 99.0, 99.0],
          [6],
          DType.float64,
        );
        final src2 = buf2.slice([Slice(start: 0, stop: 4)]);
        pad(src2, PadWidth.all(1), mode: PadMode.edge, out: buf2);
        expect(buf2.toList(), equals([10.0, 10.0, 20.0, 30.0, 40.0, 40.0]));

        // PadMode.reflect
        final buf3 = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0, 99.0, 99.0],
          [6],
          DType.float64,
        );
        final src3 = buf3.slice([Slice(start: 0, stop: 4)]);
        pad(src3, PadWidth.all(1), mode: PadMode.reflect, out: buf3);
        expect(buf3.toList(), equals([20.0, 10.0, 20.0, 30.0, 40.0, 30.0]));

        // PadMode.mean (exercises _padAxisByAxis fallback path)
        final buf4 = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0, 99.0, 99.0],
          [6],
          DType.float64,
        );
        final src4 = buf4.slice([Slice(start: 0, stop: 4)]);
        pad(src4, PadWidth.all(1), mode: PadMode.mean, out: buf4);
        expect(buf4.toList(), equals([25.0, 10.0, 20.0, 30.0, 40.0, 25.0]));
      });
    });

    test('concatenate and stack with inputs aliasing out', () {
      NDArray.scope(() {
        final buf = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final firstHalf = buf.slice([Slice(start: 0, stop: 2)]);
        final secondHalf = buf.slice([Slice(start: 2, stop: 4)]);
        concatenate([secondHalf, firstHalf], out: buf);
        expect(buf.toList(), equals([3.0, 4.0, 1.0, 2.0]));

        final mat = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final row0 = mat.slice([Index(0), Slice.all()]);
        final row1 = mat.slice([Index(1), Slice.all()]);
        stack([row1, row0], out: mat);
        expect(mat.shape, equals([2, 2]));
        expect(mat.toList(), equals([3.0, 4.0, 1.0, 2.0]));
      });
    });

    test('diag(d, out: m) where d = diag(m) is a view of m', () {
      NDArray.scope(() {
        final m = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final d = diag(m);
        diag(d, out: m);
        expect(m.shape, equals([2, 2]));
        expect(m.toList(), equals([1.0, 0.0, 0.0, 4.0]));
      });
    });

    test('tril(m.transpose(), out: m) and triu(m.transpose(), out: m)', () {
      NDArray.scope(() {
        final m1 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        tril(m1.transpose(), out: m1);
        expect(m1.shape, equals([2, 2]));
        expect(m1.toList(), equals([1.0, 0.0, 2.0, 4.0]));

        final m2 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        triu(m2.transpose(), out: m2);
        expect(m2.shape, equals([2, 2]));
        expect(m2.toList(), equals([1.0, 3.0, 0.0, 4.0]));
      });
    });

    test(
      'repeat (int and List<int>) and tile with out: flip(buf) and out: buf aliasing src',
      () {
        NDArray.scope(() {
          // repeat with int repeats and negative-stride out: flip(buf)
          final src = NDArray<Int32>.fromList([10, 20], [2], DType.int32);
          final buf1 = NDArray<Int32>.zeros([4], DType.int32);
          final flippedOut1 = flip(buf1);
          repeat(src, 2, out: flippedOut1);
          expect(flippedOut1.toList(), equals([10, 10, 20, 20]));
          expect(buf1.toList(), equals([20, 20, 10, 10]));

          // repeat with List<int> repeats and negative-stride out: flip(buf)
          final buf2 = NDArray<Int32>.zeros([5], DType.int32);
          final flippedOut2 = flip(buf2);
          repeat(src, [2, 3], out: flippedOut2);
          expect(flippedOut2.toList(), equals([10, 10, 20, 20, 20]));
          expect(buf2.toList(), equals([20, 20, 20, 10, 10]));

          // repeat with out: buf where src is a slice of buf (both int and List<int>)
          final buf3 = NDArray<Int32>.fromList(
            [10, 20, 99, 99],
            [4],
            DType.int32,
          );
          final srcSlice3 = buf3.slice([Slice(start: 0, stop: 2)]);
          repeat(srcSlice3, 2, out: buf3);
          expect(buf3.toList(), equals([10, 10, 20, 20]));

          final buf4 = NDArray<Int32>.fromList(
            [10, 20, 99, 99, 99],
            [5],
            DType.int32,
          );
          final srcSlice4 = buf4.slice([Slice(start: 0, stop: 2)]);
          repeat(srcSlice4, [2, 3], out: buf4);
          expect(buf4.toList(), equals([10, 10, 20, 20, 20]));

          // tile with negative-stride out: flip(buf)
          final buf5 = NDArray<Int32>.zeros([4], DType.int32);
          final flippedOut5 = flip(buf5);
          tile(src, [2], out: flippedOut5);
          expect(flippedOut5.toList(), equals([10, 20, 10, 20]));
          expect(buf5.toList(), equals([20, 10, 20, 10]));

          // tile with out: buf where src is a slice of buf
          final buf6 = NDArray<Int32>.fromList(
            [10, 20, 99, 99],
            [4],
            DType.int32,
          );
          final srcSlice6 = buf6.slice([Slice(start: 0, stop: 2)]);
          tile(srcSlice6, [2], out: buf6);
          expect(buf6.toList(), equals([10, 20, 10, 20]));
        });
      },
    );

    test('diff with out sharing memory with a', () {
      NDArray.scope(() {
        final buf = NDArray<Float64>.fromList(
          [1.0, 3.0, 6.0, 10.0, 15.0],
          [5],
          DType.float64,
        );
        // out overlaps with the trailing elements of buf (indices 1..4)
        final outSlice = buf.slice([Slice(start: 1, stop: 5)]);
        diff(buf, n: 1, out: outSlice);
        expect(outSlice.toList(), equals([2.0, 3.0, 4.0, 5.0]));
        expect(buf.toList(), equals([1.0, 2.0, 3.0, 4.0, 5.0]));
      });
    });
  });
}
