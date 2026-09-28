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
  group('GpuArray Arithmetic & Ufuncs', () {
    test('Elementwise binary operators (+, -, *, /)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b = GpuArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );

        final sum = a + b;
        expect(
          sum.toNestedList(),
          equals([
            [11.0, 22.0],
            [33.0, 44.0],
          ]),
        );

        final diff = b - a;
        expect(
          diff.toNestedList(),
          equals([
            [9.0, 18.0],
            [27.0, 36.0],
          ]),
        );

        final prod = a * b;
        expect(
          prod.toNestedList(),
          equals([
            [10.0, 40.0],
            [90.0, 160.0],
          ]),
        );

        final quot = b / a;
        expect(
          quot.toNestedList(),
          equals([
            [10.0, 10.0],
            [10.0, 10.0],
          ]),
        );
      });
    });

    test('Scalar operations and broadcasting', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final addedScalar = a + 10.0;
        expect(addedScalar.toNestedList(), equals([11.0, 12.0, 13.0]));

        final scaled = a * 5.0;
        expect(scaled.toNestedList(), equals([5.0, 10.0, 15.0]));

        // Multidimensional broadcasting: [2, 3] + [1, 3]
        final m1 = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );
        final m2 = GpuArray.fromList(
          [
            [10.0, 20.0, 30.0],
          ],
          [1, 3],
          DType.float64,
        );

        final broadcasted = m1 + m2;
        expect(broadcasted.shape, equals([2, 3]));
        expect(
          broadcasted.toNestedList(),
          equals([
            [11.0, 22.0, 33.0],
            [14.0, 25.0, 36.0],
          ]),
        );
      });
    });

    test('Unary operations (sin, cos, exp, log, sqrt, abs, negate)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([-1.0, 0.0, 1.0, 4.0], [4], DType.float64);

        final neg = -a;
        expect(neg.toList(), equals([1.0, -0.0, -1.0, -4.0]));

        final absolute = a.abs();
        expect(absolute.toList(), equals([1.0, 0.0, 1.0, 4.0]));

        final pos = GpuArray.fromList([0.0, 1.0, 4.0, 9.0], [4], DType.float64);
        final sq = pos.sqrt();
        expect(sq.toList(), equals([0.0, 1.0, 2.0, 3.0]));

        final zeros = GpuArray.zeros([2], DType.float64);
        final cosZeros = zeros.cos();
        expect(cosZeros.toList(), equals([1.0, 1.0]));

        final sinZeros = zeros.sin();
        expect(sinZeros.toList(), equals([0.0, 0.0]));
      });
    });

    test('Comparison operations', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 5.0, 10.0], [3], DType.float64);
        final b = GpuArray.fromList([2.0, 5.0, 8.0], [3], DType.float64);

        final gt = a.greater(b);
        expect(gt.dtype, equals(DType.boolean));
        expect(gt.toList(), equals([false, false, true]));

        final eq = a.equal(b);
        expect(eq.toList(), equals([false, true, false]));

        final le = a.lessEqual(b);
        expect(le.toList(), equals([true, true, false]));
      });
    });
  });
}
