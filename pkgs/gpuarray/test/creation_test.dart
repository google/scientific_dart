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
  group('GpuArray Creation & Basic Properties', () {
    test('GpuArray.fromList creates 1D and 2D arrays', () {
      ResourceScope.scope(() {
        final arr1d = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        expect(arr1d.rank, equals(1));
        expect(arr1d.size, equals(4));
        expect(arr1d.shape, equals([4]));
        expect(arr1d.toList(), equals([1.0, 2.0, 3.0, 4.0]));

        final arr2d = GpuArray.fromList(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ],
          [2, 2],
          DType.float32,
        );
        expect(arr2d.rank, equals(2));
        expect(arr2d.size, equals(4));
        expect(arr2d.shape, equals([2, 2]));
        expect(arr2d.isSquare, isTrue);
        expect(
          arr2d.toNestedList(),
          equals([
            [1.0, 2.0],
            [3.0, 4.0],
          ]),
        );
      });
    });

    test('GpuArray.zeros and GpuArray.ones', () {
      ResourceScope.scope(() {
        final zeros = GpuArray.zeros([2, 3], DType.float64);
        expect(zeros.shape, equals([2, 3]));
        expect(
          zeros.toNestedList(),
          equals([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
          ]),
        );

        final ones = GpuArray.ones([2, 2], DType.int32);
        expect(
          ones.toNestedList(),
          equals([
            [1, 1],
            [1, 1],
          ]),
        );
      });
    });

    test('GpuArray.filled', () {
      ResourceScope.scope(() {
        final filled = GpuArray.filled([3], 42.0, DType.float64);
        expect(filled.toNestedList(), equals([42.0, 42.0, 42.0]));
      });
    });

    test('Scalar retrieval from 0D or 1D length-1 tensor', () {
      ResourceScope.scope(() {
        final scalarArr = GpuArray.fromList([99.5], [1], DType.float64);
        expect(scalarArr.scalar, equals(99.5));

        final nonScalar = GpuArray.zeros([2, 2], DType.float64);
        expect(() => nonScalar.scalar, throwsStateError);
      });
    });

    test(
      'Shape manipulations (reshape, transpose, flatten, squeeze, unsqueeze)',
      () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final reshaped = a.reshape([3, 2]);
          expect(reshaped.shape, equals([3, 2]));
          expect(
            reshaped.toNestedList(),
            equals([
              [1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0],
            ]),
          );

          final transposed = a.transpose();
          expect(transposed.shape, equals([3, 2]));
          expect(
            transposed.toNestedList(),
            equals([
              [1.0, 4.0],
              [2.0, 5.0],
              [3.0, 6.0],
            ]),
          );

          final flat = a.flatten();
          expect(flat.shape, equals([6]));
          expect(flat.toNestedList(), equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

          final unsqueezed = a.unsqueeze(0);
          expect(unsqueezed.shape, equals([1, 2, 3]));

          final squeezed = unsqueezed.squeeze(axis: 0);
          expect(squeezed.shape, equals([2, 3]));
        });
      },
    );
  });
}
