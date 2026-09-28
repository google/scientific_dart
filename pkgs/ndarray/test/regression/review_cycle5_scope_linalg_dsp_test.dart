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
  group('Review Cycle 5: Scope, Linalg, Manipulation & DSP regressions', () {
    test(
      'roll with empty axes does not steal/dispose input and copies to out',
      () {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final out = NDArray<Float64>.zeros([2, 2], DType.float64);

        NDArray.scope(() {
          final res1 = roll(a, <int>[], axis: <int>[]);
          expect(res1.toList(), equals([1.0, 2.0, 3.0, 4.0]));
          expect(identical(res1, a), isFalse);

          final res2 = roll(a, <int>[], axis: <int>[], out: out);
          expect(identical(res2, out), isTrue);
          expect(out.toList(), equals([1.0, 2.0, 3.0, 4.0]));
        });

        // Verify `a` was not stolen into the inner scope and disposed
        expect(a.isDisposed, isFalse);
        expect(a.toList(), equals([1.0, 2.0, 3.0, 4.0]));
        expect(out.isDisposed, isFalse);
        expect(out.toList(), equals([1.0, 2.0, 3.0, 4.0]));

        a.dispose();
        out.dispose();
      },
    );

    test(
      'correlate and convolve handle out sharing memory with in1 or in2',
      () {
        // Test correlate with out sharing memory with in1 (valid mode)
        final a1 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final v1 = NDArray<Float64>.fromList(
          [1.0, 2.0, 1.0],
          [3],
          DType.float64,
        );
        final expectedCorrValid = correlate(a1, v1, mode: ConvMode.valid);

        final a1Slice = a1.slice([Slice(start: 1, stop: 4)]);
        correlate(a1, v1, mode: ConvMode.valid, out: a1Slice);
        expect(a1Slice.toList(), equals(expectedCorrValid.toList()));

        a1Slice.dispose();
        expectedCorrValid.dispose();
        a1.dispose();
        v1.dispose();

        // Test correlate with out sharing memory with in2 (same mode)
        final a2 = NDArray<Float64>.fromList(
          [2.0, 4.0, 6.0],
          [3],
          DType.float64,
        );
        final v2 = NDArray<Float64>.fromList(
          [1.0, 0.5, 0.25],
          [3],
          DType.float64,
        );
        final expectedCorrSame = correlate(a2, v2, mode: ConvMode.same);

        correlate(a2, v2, mode: ConvMode.same, out: v2);
        expect(v2.toList(), equals(expectedCorrSame.toList()));

        expectedCorrSame.dispose();
        a2.dispose();
        v2.dispose();

        // Test convolve with out sharing memory with in1 (same mode)
        final a3 = NDArray<Float64>.fromList(
          [1.0, 3.0, 5.0, 7.0],
          [4],
          DType.float64,
        );
        final v3 = NDArray<Float64>.fromList([1.0, -1.0], [2], DType.float64);
        final expectedConvSame = convolve(a3, v3, mode: ConvMode.same);

        convolve(a3, v3, mode: ConvMode.same, out: a3);
        expect(a3.toList(), equals(expectedConvSame.toList()));

        expectedConvSame.dispose();
        a3.dispose();
        v3.dispose();

        // Test convolve with out sharing memory with in2 (valid mode)
        final a4 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final v4 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final expectedConvValid = convolve(a4, v4, mode: ConvMode.valid);

        final v4Slice = v4.slice([Slice(start: 0, stop: 2)]);
        convolve(a4, v4, mode: ConvMode.valid, out: v4Slice);
        expect(v4Slice.toList(), equals(expectedConvValid.toList()));

        v4Slice.dispose();
        expectedConvValid.dispose();
        a4.dispose();
        v4.dispose();
      },
    );

    test(
      'nested NDArray.scope with views of outer arrays and StateError on view detach',
      () {
        late NDArray<Float64> outerArr;
        late NDArray<Float64> view;

        NDArray.scope(() {
          outerArr = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [4],
            DType.float64,
          );

          NDArray.scope(() {
            view = outerArr.slice([Slice(start: 1, stop: 3)]);
            // Views do not own memory; calling detachToParentScope or detachFromScope
            // on a view throws StateError.
            expect(() => view.detachToParentScope(), throwsStateError);
            expect(() => view.detachFromScope(), throwsStateError);
            expect(view.isDisposed, isFalse);
          });

          // Views of outer arrays are not tracked by the inner scope and remain valid
          // after the inner scope exits without needing detachToParentScope().
          expect(outerArr.isDisposed, isFalse);
          expect(view.isDisposed, isFalse);
          expect(view.toList(), equals([20.0, 30.0]));

          // Detaching outerArr from the outer scope removes its tracking entry.
          outerArr.detachFromScope();
        });

        // Because outerArr was detached from outer scope, both outerArr and view remain alive.
        expect(outerArr.isDisposed, isFalse);
        expect(view.isDisposed, isFalse);
        expect(outerArr.toList(), equals([10.0, 20.0, 30.0, 40.0]));
        outerArr.dispose();
        expect(outerArr.isDisposed, isTrue);
        expect(view.isDisposed, isTrue);
      },
    );

    test('fft on integer input promotes via native castNDArray cleanly', () {
      final a = NDArray<Int64>.fromList([1, 2, 3, 4], [4], DType.int64);
      final res = fft(a);
      expect(res.dtype, equals(DType.complex128));
      expect(res.shape, equals([4]));
      expect(res.toList()[0].real, closeTo(10.0, 1e-9));
      a.dispose();
      res.dispose();
    });
  });
}
