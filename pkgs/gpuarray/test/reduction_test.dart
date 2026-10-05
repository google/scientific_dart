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

    test('Reductions with out: parameter and error validation', () {
      ResourceScope.scope(() {
        final mat = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );
        final outScalar = GpuArray.zeros(const [], DType.float64);
        final s = mat.sum(out: outScalar);
        expect(identical(s, outScalar), isTrue);
        expect(outScalar.scalar, equals(21.0));

        final outAxis = GpuArray.zeros([2], DType.float64);
        final sAxis = mat.sum(axis: 1, out: outAxis);
        expect(identical(sAxis, outAxis), isTrue);
        expect(outAxis.toList(), equals([6.0, 15.0]));

        final outMax = GpuArray.zeros([2], DType.float64);
        final mAxis = mat.max(axis: 1, out: outMax);
        expect(identical(mAxis, outMax), isTrue);
        expect(outMax.toList(), equals([3.0, 6.0]));

        expect(
          () => mat.sum(axis: 5),
          throwsA(isA<GpuAxisOutOfBoundsException>()),
        );
        expect(() => mat.sum(axis: 5), throwsA(isA<RangeError>()));
      });
    });

    test(
      'F2: mean() and nanmean() preserve floating-point dtypes and promote integers to Float64',
      () {
        ResourceScope.scope(() {
          final f32 = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
          );
          final GpuArray<Float32> m32 = f32.mean();
          expect(m32.dtype, equals(DType.float32));
          expect(m32.scalar, closeTo(2.5, 1e-5));

          final f16 = GpuArray.fromList([2.0, 4.0], [2], DType.float16);
          final GpuArray<Float16> m16 = f16.mean();
          expect(m16.dtype, equals(DType.float16));
          expect(m16.scalar, closeTo(3.0, 1e-2));

          final bf16 = GpuArray.fromList([2.0, 6.0], [2], DType.bfloat16);
          final GpuArray<BFloat16> mb16 = bf16.mean();
          expect(mb16.dtype, equals(DType.bfloat16));
          expect(mb16.scalar, closeTo(4.0, 1e-2));

          final i32 = GpuArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final mi32 = i32.mean();
          expect(mi32.dtype, equals(DType.float64));
          expect(mi32.scalar, closeTo(2.5, 1e-6));

          final mi32Custom = i32.mean(dtype: DType.float32);
          expect(mi32Custom.dtype, equals(DType.float32));
          expect(mi32Custom.scalar, closeTo(2.5, 1e-5));
        });
      },
    );

    test('F3: argmin, argmax, and countNonzero return GpuArray<Int64>', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [
            [3.0, 0.0, 5.0],
            [1.0, 8.0, 0.0],
          ],
          [2, 3],
          DType.float32,
        );
        final GpuArray<Int64> amin = a.argmin();
        final GpuArray<Int64> amax = a.argmax();
        final GpuArray<Int64> cnz = a.countNonzero();
        expect(amin.dtype, equals(DType.int64));
        expect(amax.dtype, equals(DType.int64));
        expect(cnz.dtype, equals(DType.int64));
        expect(amin.scalar, equals(1));
        expect(amax.scalar, equals(4));
        expect(cnz.scalar, equals(4));

        final GpuArray<Int64> aminAxis1 = argmin(a, axis: 1);
        final GpuArray<Int64> amaxAxis0 = argmax(a, axis: 0);
        final GpuArray<Int64> cnzAxis1 = countNonzero(a, axis: 1);
        expect(aminAxis1.dtype, equals(DType.int64));
        expect(aminAxis1.toList(), equals([1, 2]));
        expect(amaxAxis0.dtype, equals(DType.int64));
        expect(amaxAxis0.toList(), equals([0, 1, 0]));
        expect(cnzAxis1.dtype, equals(DType.int64));
        expect(cnzAxis1.toList(), equals([2, 2]));
      });
    });

    test(
      'F4: Statistical and NaN-aware reductions (variance, std, ptp, nansum, nanmean, nanmin, nanmax)',
      () {
        ResourceScope.scope(() {
          final f32 = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [4],
            DType.float32,
          );
          final GpuArray<Float32> v32 = f32.variance();
          final GpuArray<Float32> s32 = f32.std();
          final GpuArray<Float32> p32 = f32.ptp();
          expect(v32.dtype, equals(DType.float32));
          expect(s32.dtype, equals(DType.float32));
          expect(p32.dtype, equals(DType.float32));
          expect(v32.scalar, closeTo(1.25, 1e-5));
          expect(s32.scalar, closeTo(1.1180339887, 1e-5));
          expect(p32.scalar, closeTo(3.0, 1e-5));

          // Sample variance with ddof: 1
          expect(variance(f32, ddof: 1).scalar, closeTo(5.0 / 3.0, 1e-5));

          // NaN-aware reductions
          final withNan = GpuArray.fromList(
            [1.0, double.nan, 3.0, 5.0],
            [4],
            DType.float32,
          );
          final GpuArray<Float32> ns = nansum(withNan);
          final GpuArray<Float32> nm = withNan.nanmean();
          final GpuArray<Float32> nmin = nanmin(withNan);
          final GpuArray<Float32> nmax = nanmax(withNan);
          expect(ns.dtype, equals(DType.float32));
          expect(nm.dtype, equals(DType.float32));
          expect(ns.scalar, closeTo(9.0, 1e-5));
          expect(nm.scalar, closeTo(3.0, 1e-5));
          expect(nmin.scalar, closeTo(1.0, 1e-5));
          expect(nmax.scalar, closeTo(5.0, 1e-5));
        });
      },
    );
  });
}
