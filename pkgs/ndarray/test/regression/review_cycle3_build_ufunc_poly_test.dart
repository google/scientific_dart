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

import 'dart:io';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Review Cycle 3: Build Hooks, Ufunc, & Polynomial Fixes', () {
    test(
      '1. Build hooks include -arch flag for macOS/iOS cross-compilation',
      () {
        final ndarrayBuild = File('hook/build.dart').readAsStringSync();
        expect(
          ndarrayBuild,
          contains("arch == Architecture.arm64 ? 'arm64' : 'x86_64'"),
        );

        final openblasBuildFile = File('../openblas/hook/build.dart');
        if (openblasBuildFile.existsSync()) {
          final openblasBuild = openblasBuildFile.readAsStringSync();
          expect(
            openblasBuild,
            contains("arch == Architecture.arm64 ? 'arm64' : 'x86_64'"),
          );
        }

        final pocketfftBuildFile = File('../pocketfft/hook/build.dart');
        if (pocketfftBuildFile.existsSync()) {
          final pocketfftBuild = pocketfftBuildFile.readAsStringSync();
          expect(
            pocketfftBuild,
            contains("arch == Architecture.arm64 ? 'arm64' : 'x86_64'"),
          );
        }
      },
      skip: const bool.fromEnvironment('dart.tool.dart2wasm')
          ? 'Host source file inspection is not supported on Wasm'
          : false,
    );

    test(
      '2a. reduceatUfunc handles non-contiguous strided int64 and int32 indices',
      () {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [6],
          DType.float64,
        );

        // Strided int64 indices: [0, 99, 2, 99, 4] with step 2 -> [0, 2, 4]
        final rawIdx64 = NDArray.fromList([0, 99, 2, 99, 4], [5], DType.int64);
        final stridedIdx64 = rawIdx64.slice([
          const Slice(start: 0, stop: 5, step: 2),
        ]);
        expect(stridedIdx64.isContiguous, isFalse);

        final res64 = reduceatUfunc(a, stridedIdx64, op: BinaryOp.add);
        // [0..2): 1+2=3, [2..4): 3+4=7, [4..end): 5+6=11
        expect(res64.toList(), equals([3.0, 7.0, 11.0]));

        // Strided int32 indices: [1, 88, 3, 88] with step 2 -> [1, 3]
        final rawIdx32 = NDArray.fromList([1, 88, 3, 88], [4], DType.int32);
        final stridedIdx32 = rawIdx32.slice([
          const Slice(start: 0, stop: 4, step: 2),
        ]);
        expect(stridedIdx32.isContiguous, isFalse);

        final res32 = reduceatUfunc(a, stridedIdx32, op: BinaryOp.add);
        // [1..3): 2+3=5, [3..end): 4+5+6=15
        expect(res32.toList(), equals([5.0, 15.0]));

        a.dispose();
        rawIdx64.dispose();
        stridedIdx64.dispose();
        res64.dispose();
        rawIdx32.dispose();
        stridedIdx32.dispose();
        res32.dispose();
      },
    );

    test(
      '2b. accumulateUfunc handles memory aliasing between input and out',
      () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        // Reversed view shares memory with a
        final outReversed = a.slice([const Slice(step: -1)]);
        accumulateUfunc(a, op: BinaryOp.add, out: outReversed);
        // Cumulative sum of [1, 2, 3, 4] is [1, 3, 6, 10]
        expect(outReversed.toList(), equals([1.0, 3.0, 6.0, 10.0]));
        expect(a.toList(), equals([10.0, 6.0, 3.0, 1.0]));

        a.dispose();
        outReversed.dispose();
      },
    );

    test('3a. Orthogonal polynomials check isDisposed at the very start', () {
      final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final x2d = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      final out = NDArray<Float64>.zeros([2, 2], DType.float64);

      c.dispose();
      expect(
        () => chebval(x2d, c),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Cannot access a disposed NDArray.',
          ),
        ),
      );
      expect(
        () => legval(x2d, c),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Cannot access a disposed NDArray.',
          ),
        ),
      );
      expect(
        () => hermval(x2d, c),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Cannot access a disposed NDArray.',
          ),
        ),
      );
      expect(
        () => lagval(x2d, c),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Cannot access a disposed NDArray.',
          ),
        ),
      );

      final cValid = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      out.dispose();
      expect(
        () => chebval(x2d, cValid, out: out),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Cannot access a disposed NDArray.',
          ),
        ),
      );

      x2d.dispose();
      cValid.dispose();
    });

    test(
      '3b. Orthogonal polynomials handle memory aliasing in Dart and C++ strided paths',
      () {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final xRef = NDArray.fromList([0.5, 1.5, 2.5], [3], DType.float64);
        final expectedCheb = chebval(xRef, c);

        // Case 1: out aliases c
        final cAlias = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        chebval(xRef, cAlias, out: cAlias);
        expect(cAlias.toList(), equals(expectedCheb.toList()));

        // Case 2: out is reversed view of x (strided aliasing)
        final xBuf = NDArray.fromList([0.5, 1.5, 2.5], [3], DType.float64);
        final outRev = xBuf.slice([const Slice(step: -1)]);
        chebval(xBuf, c, out: outRev);
        expect(outRev.toList(), equals(expectedCheb.toList()));

        // Case 3: strided x and overlapping strided out in 2D
        final mat = NDArray.fromList(
          [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
          [2, 3],
          DType.float64,
        );
        final xStrided = mat.slice([
          const Slice.all(),
          const Slice(start: 0, stop: 2),
        ]); // shape [2, 2]
        final expectedStrided = legval(xStrided, c);
        final outStrided = mat.slice([
          const Slice.all(),
          const Slice(start: 1, stop: 3),
        ]); // overlaps xStrided
        legval(xStrided, c, out: outStrided);
        expect(outStrided.toList(), equals(expectedStrided.toList()));

        c.dispose();
        xRef.dispose();
        expectedCheb.dispose();
        cAlias.dispose();
        xBuf.dispose();
        outRev.dispose();
        mat.dispose();
        xStrided.dispose();
        expectedStrided.dispose();
        outStrided.dispose();
      },
    );

    test(
      '3c. C++ strided_cum_op_impl and strided_diff_op_impl handle overlapping strided buffers',
      () {
        // Cumsum with strided overlapping buffer
        final buf = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final srcCum = buf.slice([
          const Slice.all(),
          const Slice(start: 0, stop: 2),
        ]); // [[1, 2], [4, 5]]
        final expectedCum = cumsum(srcCum, axis: 1);
        final outCum = buf.slice([
          const Slice.all(),
          const Slice(start: 1, stop: 3),
        ]); // overlaps srcCum
        cumsum(srcCum, axis: 1, out: outCum);
        expect(outCum.toList(), equals(expectedCum.toList()));

        // Diff with strided overlapping buffer
        final bufDiff = NDArray.fromList(
          [10.0, 25.0, 45.0, 70.0, 100.0],
          [5],
          DType.float64,
        );
        // Non-contiguous or overlapping slice for diff
        final expectedDiff = diff(bufDiff);
        final outDiff = bufDiff.slice([
          const Slice(start: 1, stop: 5),
        ]); // overlaps input at offset +1
        diff(bufDiff, out: outDiff);
        expect(outDiff.toList(), equals(expectedDiff.toList()));

        buf.dispose();
        srcCum.dispose();
        expectedCum.dispose();
        outCum.dispose();
        bufDiff.dispose();
        expectedDiff.dispose();
        outDiff.dispose();
      },
    );

    test(
      '4. AGENTS.md Float64 type compliance for norm, SVDRecordDispose, cond, logaddexp, logaddexp2, atan2',
      () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final outScalar = NDArray<Float64>.zeros([], DType.float64);

        final NDArray<Float64> nRes = norm(a, out: outScalar);
        expect(nRes.scalar, closeTo(5.477225575, 1e-6));

        final svdRes = svd(a);
        final NDArray<AnySpec> sArr = svdRes.s;
        expect(sArr.shape, equals([2]));
        svdRes.dispose();

        final NDArray<Float64> cRes = cond(a, out: outScalar);
        expect(cRes.scalar, greaterThan(1.0));

        final x1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final x2 = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final outVec = NDArray<Float64>.zeros([2], DType.float64);

        final NDArray<DTypeTag> lae = logaddexp(x1, x2, out: outVec);
        expect(lae.shape, equals([2]));

        final NDArray<DTypeTag> lae2 = logaddexp2(x1, x2, out: outVec);
        expect(lae2.shape, equals([2]));

        final NDArray<DTypeTag> at2 = atan2(x1, x2, out: outVec);
        expect(at2.shape, equals([2]));

        a.dispose();
        outScalar.dispose();
        x1.dispose();
        x2.dispose();
        outVec.dispose();
      },
    );
  });
}
