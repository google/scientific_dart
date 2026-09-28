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
  group('GpuArray Reductions', () {
    test('Full reductions (sum, mean, min, max, prod)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);

        expect(a.sum().scalar, equals(10.0));
        expect(a.mean().scalar, equals(2.5));
        expect(a.min().scalar, equals(1.0));
        expect(a.max().scalar, equals(4.0));
        expect(a.prod().scalar, equals(24.0));
      });
    });

    test('2D reductions along axis 0 and axis 1', () {
      ResourceScope.scope(() {
        final mat = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );

        // Sum along columns (axis 0) -> [5.0, 7.0, 9.0]
        final sum0 = mat.sum(axis: 0);
        expect(sum0.shape, equals([3]));
        expect(sum0.toList(), equals([5.0, 7.0, 9.0]));

        // Sum along rows (axis 1) -> [6.0, 15.0]
        final sum1 = mat.sum(axis: 1);
        expect(sum1.shape, equals([2]));
        expect(sum1.toList(), equals([6.0, 15.0]));

        // Mean along axis 0
        final mean0 = mat.mean(axis: 0);
        expect(mean0.toList(), equals([2.5, 3.5, 4.5]));

        // Min and Max along axis 1
        expect(mat.min(axis: 1).toList(), equals([1.0, 4.0]));
        expect(mat.max(axis: 1).toList(), equals([3.0, 6.0]));
      });
    });

    test('Reductions with keepDims: true', () {
      ResourceScope.scope(() {
        final mat = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );

        final sumKeep = mat.sum(axis: 1, keepDims: true);
        expect(sumKeep.shape, equals([2, 1]));
        expect(
          sumKeep.toNestedList(),
          equals([
            [6.0],
            [15.0],
          ]),
        );

        final fullSumKeep = mat.sum(keepDims: true);
        expect(fullSumKeep.shape, equals([1, 1]));
        expect(
          fullSumKeep.toNestedList(),
          equals([
            [21.0],
          ]),
        );
      });
    });

    test('Complex128 sum, prod, and axis reductions', () {
      ResourceScope.scope(() {
        // Complex128 1D sum reduction
        final a1 = GpuArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, -4.0),
            Complex(0.0, 5.0),
            Complex(-2.0, 1.0),
          ],
          [4],
          DType.complex128,
        );
        final sum1 = a1.sum();
        expect(sum1.dtype, equals(DType.complex128));
        final sumVal = sum1.scalar as Complex;
        expect(sumVal.real, closeTo(2.0, 1e-5));
        expect(sumVal.imag, closeTo(4.0, 1e-5));

        // Complex128 1D prod reduction
        // (1+i)(2-i) = 3 + i; (3+i)(3i) = -3 + 9i
        final a2 = GpuArray.fromList(
          [Complex(1.0, 1.0), Complex(2.0, -1.0), Complex(0.0, 3.0)],
          [3],
          DType.complex128,
        );
        final prod2 = a2.prod();
        expect(prod2.dtype, equals(DType.complex128));
        final prodVal = prod2.scalar as Complex;
        expect(prodVal.real, closeTo(-3.0, 1e-5));
        expect(prodVal.imag, closeTo(9.0, 1e-5));

        // Complex128 2D axis reduction
        final mat = GpuArray.fromList(
          [
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [Complex(5.0, 6.0), Complex(7.0, 8.0)],
          ],
          [2, 2],
          DType.complex128,
        );

        final sum0 = mat.sum(axis: 0);
        expect(sum0.shape, equals([2]));
        final sum0List = sum0.toList().cast<Complex>();
        expect(sum0List[0].real, closeTo(6.0, 1e-5));
        expect(sum0List[0].imag, closeTo(8.0, 1e-5));
        expect(sum0List[1].real, closeTo(10.0, 1e-5));
        expect(sum0List[1].imag, closeTo(12.0, 1e-5));

        final sum1Axis = mat.sum(axis: 1);
        expect(sum1Axis.shape, equals([2]));
        final sum1List = sum1Axis.toList().cast<Complex>();
        expect(sum1List[0].real, closeTo(4.0, 1e-5));
        expect(sum1List[0].imag, closeTo(6.0, 1e-5));
        expect(sum1List[1].real, closeTo(12.0, 1e-5));
        expect(sum1List[1].imag, closeTo(14.0, 1e-5));
      });
    });
  });
}
