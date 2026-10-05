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

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Review Cycle 2: Missing isDisposed Validation (StateError)', () {
    test('ravel() throws StateError on disposed array', () {
      final a = NDArray<Float64>.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      a.dispose();
      expect(() => a.ravel(), throwsStateError);
    });

    test(
      'scalar getter throws StateError on disposed array (even before rank check)',
      () {
        final a0 = NDArray<Float64>.scalar(42.0, dtype: DType.float64);
        a0.dispose();
        expect(() => a0.scalar, throwsStateError);

        final a1 = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        a1.dispose();
        expect(() => a1.scalar, throwsStateError);
      },
    );

    test(
      'getCell and setCell throw StateError on disposed array before coord validation',
      () {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        a.dispose();
        expect(() => a.getCell([0, 0]), throwsStateError);
        expect(() => a.getCell([0]), throwsStateError);
        expect(() => a.setCell([0, 0], 10.0), throwsStateError);
        expect(() => a.setCell([0], 10.0), throwsStateError);
      },
    );

    test(
      'setByMask throws StateError if target, mask, or values is disposed',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
          final mask = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );
          final vals = NDArray<Float64>.fromList([99.0], [1], DType.float64);

          final disposedA = a.copy()..dispose();
          expect(() => disposedA.setByMask(mask, vals), throwsStateError);

          final disposedMask = mask.copy()..dispose();
          expect(() => a.setByMask(disposedMask, vals), throwsStateError);

          final disposedVals = vals.copy()..dispose();
          expect(() => a.setByMask(mask, disposedVals), throwsStateError);
        });
      },
    );

    test('setByMaskScalar throws StateError if target or mask is disposed', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        final mask = NDArray<Boolean>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );

        final disposedA = a.copy()..dispose();
        expect(() => disposedA.setByMaskScalar(mask, 99.0), throwsStateError);

        final disposedMask = mask.copy()..dispose();
        expect(() => a.setByMaskScalar(disposedMask, 99.0), throwsStateError);
      });
    });

    test(
      'setIndicesScalar throws StateError if target or indices is disposed',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final idx = NDArray.fromList([0, 2], [2], DType.int32);

          final disposedA = a.copy()..dispose();
          expect(() => disposedA.setIndicesScalar(idx, 99.0), throwsStateError);

          final disposedIdx = idx.copy()..dispose();
          expect(() => a.setIndicesScalar(disposedIdx, 99.0), throwsStateError);
        });
      },
    );

    test(
      'setIndices throws StateError if target, indices, or values is disposed',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final idx = NDArray.fromList([0, 2], [2], DType.int32);
          final vals = NDArray<Float64>.fromList(
            [10.0, 20.0],
            [2],
            DType.float64,
          );

          final disposedA = a.copy()..dispose();
          expect(() => disposedA.setIndices(idx, vals), throwsStateError);

          final disposedIdx = idx.copy()..dispose();
          expect(() => a.setIndices(disposedIdx, vals), throwsStateError);

          final disposedVals = vals.copy()..dispose();
          expect(() => a.setIndices(idx, disposedVals), throwsStateError);
        });
      },
    );

    test(
      'applyMask throws StateError if target or mask is disposed before shape/size checks',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.create([0], DType.float64);
          final maskWrongShape = NDArray<Boolean>.fromList(
            [true, false],
            [2],
            DType.boolean,
          );

          final disposedA = a.copy()..dispose();
          // Should throw StateError even though size is 0 and shape mismatches
          expect(() => disposedA.applyMask(maskWrongShape), throwsStateError);

          final disposedMask = maskWrongShape.copy()..dispose();
          expect(() => a.applyMask(disposedMask), throwsStateError);
        });
      },
    );

    test(
      'expandDims throws StateError if disposed before axis range validation',
      () {
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        // Out-of-bounds axis 99 would throw RangeError if checked before isDisposed
        expect(() => a.expandDims(99), throwsStateError);
      },
    );

    test('squeeze throws StateError if disposed before axis validation', () {
      final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
      a.dispose();
      // Axis 0 has size 2 (would throw ArgumentError) and axis 99 is out of bounds (RangeError)
      expect(() => a.squeeze(axis: 0), throwsStateError);
      expect(() => a.squeeze(axis: 99), throwsStateError);
    });
  });

  group('Review Cycle 2: Memory Aliasing Protection (sharesMemory)', () {
    test(
      'NDArray.copy(out: ...) handles overlapping memory with distinct root parents',
      () {
        final ptr = malloc<ffi.Double>(4);
        try {
          ptr[0] = 1.0;
          ptr[1] = 2.0;
          ptr[2] = 3.0;
          ptr[3] = 4.0;

          // Two distinct root wrappers around the same memory buffer, one strided (transposed)
          final src = NDArray<Float64>.fromPointer(
            ptr.cast(),
            [2, 2],
            DType.float64,
            strides: [1, 2],
          );
          final dst = NDArray<Float64>.fromPointer(
            ptr.cast(),
            [2, 2],
            DType.float64,
            strides: [2, 1],
          );

          src.copy(out: dst);
          expect(dst.getCell([0, 0]), equals(1.0));
          expect(dst.getCell([0, 1]), equals(3.0));
          expect(dst.getCell([1, 0]), equals(2.0));
          expect(dst.getCell([1, 1]), equals(4.0));
        } finally {
          malloc.free(ptr);
        }
      },
    );

    test(
      'interp protects against aliasing when out shares memory with x, xp, or fp',
      () {
        NDArray.scope(() {
          final xp = NDArray<Float64>.fromList(
            [0.0, 1.0, 2.0, 3.0],
            [4],
            DType.float64,
          );
          final fp = NDArray<Float64>.fromList(
            [0.0, 10.0, 20.0, 30.0],
            [4],
            DType.float64,
          );
          final x = NDArray<Float64>.fromList(
            [0.5, 1.5, 2.5, 0.25],
            [4],
            DType.float64,
          );
          final expected = interp(x, xp, fp);

          // Aliasing out with x
          final xCopy = x.copy();
          interp(xCopy, xp, fp, out: xCopy);
          for (var i = 0; i < 4; i++) {
            expect(xCopy.getCell([i]), closeTo(expected.getCell([i]), 1e-12));
          }

          // Aliasing out with fp
          final fpCopy = fp.copy();
          interp(x, xp, fpCopy, out: fpCopy);
          for (var i = 0; i < 4; i++) {
            expect(fpCopy.getCell([i]), closeTo(expected.getCell([i]), 1e-12));
          }
        });
      },
    );

    test(
      'polyval protects against aliasing when out shares memory with c or x',
      () {
        NDArray.scope(() {
          final c = NDArray<Float64>.fromList(
            [2.0, -3.0, 1.0],
            [3],
            DType.float64,
          );
          final x = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final expected = polyval<Float64>(c, x);

          // Aliasing out with x
          final xCopy = x.copy();
          polyval<Float64>(c, xCopy, out: xCopy);
          for (var i = 0; i < 3; i++) {
            expect(xCopy.getCell([i]), closeTo(expected.getCell([i]), 1e-12));
          }

          // Aliasing out with c
          final cCopy = c.copy();
          polyval<Float64>(cCopy, x, out: cCopy);
          for (var i = 0; i < 3; i++) {
            expect(cCopy.getCell([i]), closeTo(expected.getCell([i]), 1e-12));
          }
        });
      },
    );

    test(
      'take_along_axis protects against aliasing when out shares memory with arr',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );
          final indices = NDArray.fromList([1, 0, 1, 0], [2, 2], DType.int32);
          final expected = take_along_axis(a, indices, 1);

          final aCopy = a.copy();
          take_along_axis(aCopy, indices, 1, out: aCopy);
          for (var r = 0; r < 2; r++) {
            for (var c = 0; c < 2; c++) {
              expect(aCopy.getCell([r, c]), equals(expected.getCell([r, c])));
            }
          }
        });
      },
    );

    test(
      'put_along_axis protects against aliasing when out shares memory with values',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final indices = NDArray.fromList([1, 0, 1, 0], [2, 2], DType.int32);

          // Values is a view of out buffer
          final out = a.copy();
          final valuesView = out.slice([Slice.all(), Slice.all()]);
          put_along_axis(a, indices, valuesView, 1, out: out);

          // row 0: idx 1 gets values[0,0]=1.0, idx 0 gets values[0,1]=2.0 => [2.0, 1.0]
          // row 1: idx 1 gets values[1,0]=3.0, idx 0 gets values[1,1]=4.0 => [4.0, 3.0]
          expect(out.getCell([0, 0]), equals(2.0));
          expect(out.getCell([0, 1]), equals(1.0));
          expect(out.getCell([1, 0]), equals(4.0));
          expect(out.getCell([1, 1]), equals(3.0));
        });
      },
    );

    test(
      'choose protects against aliasing when out shares memory with choices',
      () {
        NDArray.scope(() {
          final c0 = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final c1 = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0],
            [3],
            DType.float64,
          );
          final a = NDArray.fromList([1, 0, 1], [3], DType.int32);
          final expected = choose<Float64>(a, [c0, c1]);

          final c0Copy = c0.copy();
          // Reverse view of c0Copy so writing to out[0] mutates c0Copy[2] if unbuffered
          final c0Rev = c0Copy.slice([const Slice(step: -1)]);
          final expectedRev = choose<Float64>(a, [c0Rev, c1]);

          choose<Float64>(a, [c0Rev, c1], out: c0Copy);
          for (var i = 0; i < 3; i++) {
            expect(c0Copy.getCell([i]), equals(expectedRev.getCell([i])));
          }
          expect(expected.getCell([0]), equals(10.0));
        });
      },
    );

    test(
      'select protects against aliasing when out shares memory with choicelist or default',
      () {
        NDArray.scope(() {
          final cond = NDArray<Boolean>.fromList(
            [true, false, true],
            [3],
            DType.boolean,
          );
          final c0 = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0],
            [3],
            DType.float64,
          );
          final def = NDArray<Float64>.fromList(
            [100.0, 200.0, 300.0],
            [3],
            DType.float64,
          );

          final c0Copy = c0.copy();
          final c0Rev = c0Copy.slice([const Slice(step: -1)]);
          final expected = select<Float64>([cond], [c0Rev], defaultValue: def);

          select<Float64>([cond], [c0Rev], defaultValue: def, out: c0Copy);
          for (var i = 0; i < 3; i++) {
            expect(c0Copy.getCell([i]), equals(expected.getCell([i])));
          }
        });
      },
    );
  });

  group(
    'Review Cycle 2: NDArray Selector Disposal and Integer Indexing without toList()',
    () {
      test(
        'operator [] and []= with integer NDArray selector work and check isDisposed',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [10.0, 20.0, 30.0, 40.0],
              [4],
              DType.float64,
            );
            final idx = NDArray.fromList([3, 1], [2], DType.int32);

            final res = a[idx] as NDArray<Float64>;
            expect(res.shape, equals([2]));
            expect(res.getCell([0]), equals(40.0));
            expect(res.getCell([1]), equals(20.0));

            a[idx] = NDArray<Float64>.fromList(
              [99.0, 88.0],
              [2],
              DType.float64,
            );
            expect(a.getCell([3]), equals(99.0));
            expect(a.getCell([1]), equals(88.0));

            final disposedIdx = idx.copy()..dispose();
            expect(() => a[disposedIdx], throwsStateError);
            expect(() => a[disposedIdx] = 0.0, throwsStateError);
            expect(() => a[[disposedIdx]], throwsStateError);
            expect(() => a[[disposedIdx]] = 0.0, throwsStateError);
          });
        },
      );

      test(
        'operator [] and []= with disposed boolean NDArray selector throw StateError',
        () {
          NDArray.scope(() {
            final a = NDArray<Float64>.fromList(
              [10.0, 20.0],
              [2],
              DType.float64,
            );
            final mask = NDArray<Boolean>.fromList(
              [true, false],
              [2],
              DType.boolean,
            );
            mask.dispose();

            expect(() => a[mask], throwsStateError);
            expect(() => a[mask] = 1.0, throwsStateError);
            expect(() => a[[mask]], throwsStateError);
            expect(() => a[[mask]] = 1.0, throwsStateError);
          });
        },
      );
    },
  );
}
