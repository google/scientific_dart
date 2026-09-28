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
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Indexing & Slicing', () {
    test('1D and 2D Multi-Axis Strided Slicing and Views', () {
      ResourceScope.scope(() {
        // 1D Slicing with step
        final a1 = GpuArray.fromList(
          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
          [6],
          DType.float64,
        );
        final sub1 = a1.slice([const Slice(1, 5, 2)]);
        expect(sub1.shape, equals([2]));
        expect(sub1.toList(), equals([1.0, 3.0]));

        // Negative step slicing (reverse)
        final rev1 = a1.slice([const Slice(null, null, -1)]);
        expect(rev1.shape, equals([6]));
        expect(rev1.toList(), equals([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]));

        // 2D Slicing with Index, Slice, All
        final a2 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );

        // Row selection reducing rank
        final row1 = a2.slice([const Index(1), const All()]);
        expect(row1.shape, equals([3]));
        expect(row1.toList(), equals([4.0, 5.0, 6.0]));

        // Column selection reducing rank
        final col2 = a2.slice([const All(), const Index(2)]);
        expect(col2.shape, equals([3]));
        expect(col2.toList(), equals([3.0, 6.0, 9.0]));

        // Submatrix
        final sub2 = a2.slice([const Slice(0, 2), const Slice(1, 3)]);
        expect(sub2.shape, equals([2, 2]));
        expect(sub2.toList(), equals([2.0, 3.0, 5.0, 6.0]));
      });
    });

    test('Ellipsis and NewAxis Slicing', () {
      ResourceScope.scope(() {
        final a3 = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 2, 2],
          DType.float32,
        );

        // Ellipsis in front
        final sub = a3.slice([const Ellipsis(), const Index(1)]);
        expect(sub.shape, equals([2, 2]));
        expect(sub.toList(), equals([2.0, 4.0, 6.0, 8.0]));

        // NewAxis insertion
        final exp = a3.slice([
          const NewAxis(),
          const All(),
          const All(),
          const All(),
        ]);
        expect(exp.shape, equals([1, 2, 2, 2]));
      });
    });

    test('where() conditional selection with broadcasting', () {
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

        final res = where(cond, x, y) as GpuArray;
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals([10.0, 2.0, 3.0, 40.0]));

        // where with nonzero coordinate return
        final coords = where(cond) as List<GpuArray<Int32>>;
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

        final selected = select([cond1, cond2], [choice1, choice2]);
        expect(selected.toList(), equals([10.0, 10.0, 0.0, 0.0, 50.0]));

        // extract elements satisfying condition
        final extracted = extract(cond1, a);
        expect(extracted.shape, equals([2]));
        expect(extracted.toList(), equals([1.0, 2.0]));
      });
    });

    test('take_along_axis() and put_along_axis()', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [10.0, 30.0, 20.0, 60.0, 40.0, 50.0],
          [2, 3],
          DType.float64,
        );

        final indices = GpuArray.fromList(
          [0, 2, 1, 1, 0, 2],
          [2, 3],
          DType.int32,
        );

        final taken = take_along_axis(a, indices, 1);
        expect(taken.shape, equals([2, 3]));
        expect(taken.toList(), equals([10.0, 20.0, 30.0, 40.0, 60.0, 50.0]));

        // put_along_axis in-place
        final values = GpuArray.fromList(
          [99.0, 99.0, 99.0, 88.0, 88.0, 88.0],
          [2, 3],
          DType.float64,
        );

        put_along_axis(a, indices, values, 1);
        expect(a.toList(), equals([99.0, 99.0, 99.0, 88.0, 88.0, 88.0]));
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
          // Complex64 with pure imaginary non-zeros
          final c64Data = [
            Complex(0.0, 2.0), // pure imaginary non-zero -> idx 0 (0, 0)
            Complex(0.0, 0.0), // zero -> idx 1 (0, 1)
            Complex(3.0, 0.0), // pure real non-zero -> idx 2 (1, 0)
            Complex(4.0, -5.0), // mixed non-zero -> idx 3 (1, 1)
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

          // Complex128 with pure imaginary non-zeros
          final c128Data = [
            Complex(0.0, 0.0), // idx 0
            Complex(0.0, 7.5), // pure imag non-zero -> idx 1
            Complex(-1.0, 0.0), // idx 2
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
  });
}
