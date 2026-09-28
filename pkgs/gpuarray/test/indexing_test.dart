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

import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  group('GpuArray Indexing & Slicing', () {
    test('1D and 2D Multi-Axis Strided Slicing and Views', () {
      ResourceScope.scope(() {
        final a1 = GpuArray.fromList(
          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
          [6],
          DType.float64,
        );
        final sub1 = a1.slice([const Slice(1, 5, 2)]);
        expect(sub1.shape, equals([2]));
        expect(sub1.toList(), equals([1.0, 3.0]));

        final rev1 = a1.slice([const Slice(null, null, -1)]);
        expect(rev1.shape, equals([6]));
        expect(rev1.toList(), equals([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]));

        final a2 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final row1 = a2.slice([const Index(1), const All()]);
        expect(row1.shape, equals([3]));
        expect(row1.toList(), equals([4.0, 5.0, 6.0]));

        final col2 = a2.slice([const All(), const Index(2)]);
        expect(col2.shape, equals([3]));
        expect(col2.toList(), equals([3.0, 6.0, 9.0]));

        final sub2 = a2.slice([const Slice(0, 2), const Slice(1, 3)]);
        expect(sub2.shape, equals([2, 2]));
        expect(sub2.toList(), equals([2.0, 3.0, 5.0, 6.0]));

        // Strongly typed operator []
        final indexed = a2[[const Index(0), const All()]];
        expect(indexed.shape, equals([3]));
        expect(indexed.toList(), equals([1.0, 2.0, 3.0]));
      });
    });

    test('Ellipsis and NewAxis Slicing', () {
      ResourceScope.scope(() {
        final a3 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float32,
        );

        final sub = a3.slice([const Ellipsis(), const Index(1)]);
        expect(sub.shape, equals([2, 2]));
        expect(sub.toList(), equals([2.0, 4.0, 6.0, 8.0]));

        final exp = a3.slice([
          const NewAxis(),
          const All(),
          const All(),
          const All(),
        ]);
        expect(exp.shape, equals([1, 2, 2, 2]));
      });
    });

    test('where() conditional selection with broadcasting and out:', () {
      ResourceScope.scope(() {
        final cond = GpuArray.fromList(
          [true, false, false, true],
          [2, 2],
          DType.boolean,
        );
        final x = GpuArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final y = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final res = where<Float64>(cond, x, y);
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([10.0, 2.0, 3.0, 40.0]));

        final outBuf = GpuArray.empty([2, 2], DType.float64);
        final resOut = where<Float64>(cond, x, y, out: outBuf);
        expect(identical(resOut, outBuf), isTrue);
        expect(outBuf.toList(), equals([10.0, 2.0, 3.0, 40.0]));

        final coords = nonzero(cond);
        expect(coords.length, equals(2));
        expect(coords[0].toList(), equals([0, 1]));
        expect(coords[1].toList(), equals([0, 1]));
      });
    });

    test('select() and extract()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final cond1 = a.lessThan(2.5);
        final cond2 = a.greaterThan(4.0);

        final choice1 = GpuArray.fromList(
          [10.0, 10.0, 10.0, 10.0, 10.0],
          [5],
          DType.float64,
        );
        final choice2 = GpuArray.fromList(
          [50.0, 50.0, 50.0, 50.0, 50.0],
          [5],
          DType.float64,
        );

        final selected = select<Float64>([cond1, cond2], [choice1, choice2]);
        expect(selected.toList(), equals([10.0, 10.0, 0.0, 0.0, 50.0]));

        final outSelect = GpuArray.empty([5], DType.float64);
        select<Float64>([cond1, cond2], [choice1, choice2], out: outSelect);
        expect(outSelect.toList(), equals([10.0, 10.0, 0.0, 0.0, 50.0]));

        final extracted = extract(cond1, a);
        expect(extracted.shape, equals([2]));
        expect(extracted.toList(), equals([1.0, 2.0]));
      });
    });

    test('take(), put(), takeAlongAxis(), and putAlongAxis()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [10.0, 30.0, 20.0, 60.0, 40.0, 50.0],
          [2, 3],
          DType.float64,
        );

        final flatIdx = GpuArray.fromList([0, 3, 5], [3], DType.int32);
        final takenFlat = take(a, flatIdx);
        expect(takenFlat.shape, equals([3]));
        expect(takenFlat.toList(), equals([10.0, 60.0, 50.0]));

        final colIdx = GpuArray.fromList([2, 0], [2], DType.int32);
        final takenAxis = take(a, colIdx, axis: 1);
        expect(takenAxis.shape, equals([2, 2]));
        expect(takenAxis.toList(), equals([20.0, 10.0, 50.0, 60.0]));

        final indices = GpuArray.fromList(
          [0, 2, 1, 1, 0, 2],
          [2, 3],
          DType.int32,
        );

        final taken = takeAlongAxis(a, indices, 1);
        expect(taken.shape, equals([2, 3]));
        expect(taken.toList(), equals([10.0, 20.0, 30.0, 40.0, 60.0, 50.0]));

        final values = GpuArray.fromList(
          [99.0, 99.0, 99.0, 88.0, 88.0, 88.0],
          [2, 3],
          DType.float64,
        );

        putAlongAxis(a, indices, values, 1);
        expect(a.toList(), equals([99.0, 99.0, 99.0, 88.0, 88.0, 88.0]));

        final putIdx = GpuArray.fromList([0, 5], [2], DType.int32);
        final putVals = GpuArray.fromList([111.0, 222.0], [2], DType.float64);
        put(a, putIdx, putVals);
        expect(a.toList(), equals([111.0, 99.0, 99.0, 88.0, 88.0, 222.0]));
      });
    });

    test('nonzero(), flatnonzero(), argwhere()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final flat = flatnonzero(a);
        expect(flat.toList(), equals([0, 4, 6, 8]));

        final coords = argwhere(a);
        expect(coords.shape, equals([4, 2]));
        expect(coords.toList(), equals([0, 0, 1, 1, 2, 0, 2, 2]));
      });
    });

    test(
      'nonzero(), flatnonzero(), argwhere() with Complex64 and Complex128',
      () {
        ResourceScope.scope(() {
          final c64Data = [
            Complex(0.0, 2.0),
            Complex(0.0, 0.0),
            Complex(3.0, 0.0),
            Complex(4.0, -5.0),
          ];
          final aC64 = GpuArray.fromList(c64Data, [2, 2], DType.complex64);

          final flatC64 = flatnonzero(aC64);
          expect(flatC64.toList(), equals([0, 2, 3]));

          final nzC64 = nonzero(aC64);
          expect(nzC64.length, equals(2));
          expect(nzC64[0].toList(), equals([0, 1, 1]));
          expect(nzC64[1].toList(), equals([0, 0, 1]));

          final coordsC64 = argwhere(aC64);
          expect(coordsC64.shape, equals([3, 2]));
          expect(coordsC64.toList(), equals([0, 0, 1, 0, 1, 1]));

          final c128Data = [
            Complex(0.0, 0.0),
            Complex(0.0, 7.5),
            Complex(-1.0, 0.0),
          ];
          final aC128 = GpuArray.fromList(c128Data, [3], DType.complex128);

          final flatC128 = flatnonzero(aC128);
          expect(flatC128.toList(), equals([1, 2]));

          final nzC128 = nonzero(aC128);
          expect(nzC128.length, equals(1));
          expect(nzC128[0].toList(), equals([1, 2]));

          final coordsC128 = argwhere(aC128);
          expect(coordsC128.shape, equals([2, 1]));
          expect(coordsC128.toList(), equals([1, 2]));
        });
      },
    );

    test('Rejects broadcasted view as out destination', () {
      final base = GpuArray<Float32>.fromList([1.0], [1], DType.float32);
      final broadcasted = base.broadcastTo([4]);

      final cond = GpuArray.fromList(
        [true, false, true, false],
        [4],
        DType.boolean,
      );
      final inA = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
      final inB = GpuArray.fromList(
        [10.0, 20.0, 30.0, 40.0],
        [4],
        DType.float32,
      );
      final idx = GpuArray.fromList([0, 1, 2, 3], [4], DType.int32);

      expect(
        () => where(cond, inA, inB, out: broadcasted),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('Must be writeable and not a broadcasted view.'),
          ),
        ),
      );

      expect(
        () => take(inA, idx, out: broadcasted),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('Must be writeable and not a broadcasted view.'),
          ),
        ),
      );

      expect(
        () => takeAlongAxis(inA, idx, 0, out: broadcasted),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('Must be writeable and not a broadcasted view.'),
          ),
        ),
      );

      base.dispose();
      broadcasted.dispose();
      cond.dispose();
      inA.dispose();
      inB.dispose();
      idx.dispose();
    });
  });
}
