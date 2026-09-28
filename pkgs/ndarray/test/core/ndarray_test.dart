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

import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('NDArray Tests', () {
    test(
      'NDArray.scalar creation',
      () => NDArray.scope(() {
        final s = NDArray.scalar(42.0, dtype: DType.float64);
        expect(s.shape, <int>[]);
        expect(s.rank, 0);
        expect(s.size, 1);
        expect(s.scalar, 42.0);

        final sInt = NDArray.scalar(7, dtype: DType.int32);
        expect(sInt.shape, <int>[]);
        expect(sInt.scalar, 7);
      }),
    );
    test(
      'Creation and Strides',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4, 5, 6]), [
          2,
          3,
        ], DType.float64);
        expect(a.shape, [2, 3]);
        expect(a.strides, [3, 1]);
      }),
    );

    test(
      'Broadcasting Shape',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2]), [
          2,
          1,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
          1,
          3,
        ], DType.float64);
        final result = broadcast(a, b);
        expect(result.shape, [2, 3]);
        expect(result.stridesA, [1, 0]);
        expect(result.stridesB, [0, 1]);
      }),
    );

    test(
      'Element-wise Addition with Broadcasting',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2]), [
          2,
          1,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
          1,
          3,
        ], DType.float64);
        final c = add(a, b);
        expect(c.shape, [2, 3]);
        expect(c.data, [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
      }),
    );

    test(
      'Matrix Multiplication (OpenBLAS)',
      () => NDArray.scope(() {
        final a = NDArray<double>.fromList(Float64List.fromList([1, 2, 3, 4]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray<double>.fromList(Float64List.fromList([5, 6, 7, 8]), [
          2,
          2,
        ], DType.float64);
        final c = matmul(a, b);
        expect(c.shape, [2, 2]);
        expect(c.data, [19.0, 22.0, 43.0, 50.0]);
      }),
    );

    test(
      'View Shares Memory',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
          2,
          2,
        ], DType.float64);
        final view = NDArray.view(
          a,
          shape: [2],
          strides: [1],
          offsetElements: 1,
        ); // View of [2, 3] if we flatten or just take from offset 1
        // Wait, strides for view of [2] with stride 1 from offset 1 will be elements at index 1 and 2.
        // data at index 1 is 2.0.
        // data at index 2 is 3.0.
        expect(view.data[0], 2.0);
        expect(view.data[1], 3.0);

        // Modify view
        view.data[0] = Float64(99.0);
        // Check original
        expect(a.data[1], 99.0);
      }),
    );

    test(
      'Element-wise Subtraction with Broadcasting',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2]), [
          2,
          1,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
          1,
          3,
        ], DType.float64);
        final c = subtract(a, b);
        expect(c.shape, [2, 3]);
        expect(c.data, [-9.0, -19.0, -29.0, -8.0, -18.0, -28.0]);
      }),
    );

    test(
      'Element-wise Multiplication with Broadcasting',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2]), [
          2,
          1,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
          1,
          3,
        ], DType.float64);
        final c = multiply(a, b);
        expect(c.shape, [2, 3]);
        expect(c.data, [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
      }),
    );

    test(
      'Element-wise Division with Broadcasting',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([10, 20]), [
          2,
          1,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([2, 4, 5]), [
          1,
          3,
        ], DType.float64);
        final c = divide(a, b);
        expect(c.shape, [2, 3]);
        expect(c.data, [5.0, 2.5, 2.0, 10.0, 5.0, 4.0]);
      }),
    );

    test(
      'Zeros Factory',
      () => NDArray.scope(() {
        final a = NDArray<double>.zeros([2, 3], DType.float64);
        expect(a.shape, [2, 3]);
        expect(a.data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
      }),
    );

    test(
      'Ones Factory',
      () => NDArray.scope(() {
        final a = NDArray<double>.ones([2, 3], DType.float64);
        expect(a.shape, [2, 3]);
        expect(a.data, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
      }),
    );

    test(
      'Arange Factory',
      () => NDArray.scope(() {
        final a = NDArray<double>.arange(
          0.0,
          5.0,
          step: 1.0,
          dtype: DType.float64,
        );
        expect(a.shape, [5]);
        expect(a.data, [0.0, 1.0, 2.0, 3.0, 4.0]);
      }),
    );

    test(
      'Linspace Factory',
      () => NDArray.scope(() {
        final a = linspace<double>(0.0, 1.0, 5, dtype: DType.float64);
        expect(a.shape, [5]);
        expect(a.data, [0.0, 0.25, 0.5, 0.75, 1.0]);
      }),
    );

    test(
      'Global Sum',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
          2,
          2,
        ], DType.float64);
        final s = sum(a);
        expect(s.scalar, 10.0);
      }),
    );

    test(
      'Manual Dispose',
      () => NDArray.scope(() {
        final a = NDArray<double>.create([2, 2], DType.float64);
        // Should not throw
        a.dispose();
      }),
    );

    test(
      'Eye Factory',
      () => NDArray.scope(() {
        final a = NDArray<double>.eye(3, DType.float64);
        expect(a.shape, [3, 3]);
        expect(a.data, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
      }),
    );

    test(
      'Uniform Factory',
      () => NDArray.scope(() {
        final a = uniform([2, 3], dtype: DType.float64);
        expect(a.shape, [2, 3]);
        expect(a.data.length, 6);
        for (final value in a.data) {
          expect(value, greaterThanOrEqualTo(0.0));
          expect(value, lessThan(1.0));
        }
      }),
    );

    test(
      'Reshape',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
          4,
        ], DType.float64);
        final b = a.reshape([2, 2]);
        expect(b.shape, [2, 2]);
        expect(b.strides, [2, 1]);
        expect(b.data, [1.0, 2.0, 3.0, 4.0]);

        // Modify view
        b.data[0] = Float64(99.0);
        // Check original
        expect(a.data[0], 99.0);
      }),
    );

    test(
      'Reshape non-contiguous view disposal memory safety',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );
        final viewT = parent.transposed;
        final reshaped = viewT.reshape([6]);

        expect(reshaped.isDisposed, false);
        reshaped.dispose();
        expect(reshaped.isDisposed, true);

        expect(parent.isDisposed, false);
        expect(viewT.isDisposed, false);
      }),
    );

    test(
      'Dart Addition (Fallback)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([10, 20, 30, 40]), [
          2,
          2,
        ], DType.float64);

        // Force Dart path

        final c = add(a, b);
        expect(c.shape, [2, 2]);
        expect(c.data, [11.0, 22.0, 33.0, 44.0]);
      }),
    );

    test(
      'SIMD Addition (Float32)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]),
          [5],
          DType.float32,
        );
        final b = NDArray.fromList(
          Float32List.fromList([10.0, 20.0, 30.0, 40.0, 50.0]),
          [5],
          DType.float32,
        );

        // This should trigger SIMD path because it is float32 and contiguous
        final c = add(a, b);
        expect(c.shape, [5]);
        expect(c.data, [11.0, 22.0, 33.0, 44.0, 55.0]);
      }),
    );

    test(
      'Sum along Axis 0',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final s = sum(a, axis: 0);
        expect(s.shape, [2]);
        expect(s.data, [4.0, 6.0]);
      }),
    );

    test(
      'Sum along Axis 1',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final s = sum(a, axis: 1);
        expect(s.shape, [2]);
        expect(s.data, [3.0, 7.0]);
      }),
    );

    test(
      'NDArray.create supports Int32',
      () => NDArray.scope(() {
        final a = NDArray<Int32>.create([2, 2], DType.int32);
        expect(a.shape, [2, 2]);
        expect(a.dtype, DType.int32);
      }),
    );

    test(
      'NDArray.create supports Int64',
      () => NDArray.scope(() {
        final a = NDArray<Int64>.create([2, 2], DType.int64);
        expect(a.shape, [2, 2]);
        expect(a.dtype, DType.int64);
      }),
    );

    test(
      'Randint Factory',
      () => NDArray.scope(() {
        final a = randint([2, 3], low: 0, high: 10, dtype: DType.int64);
        expect(a.shape, [2, 3]);
        expect(a.data.length, 6);
        for (final value in a.data) {
          expect(value, greaterThanOrEqualTo(0));
          expect(value, lessThan(10));
        }
      }),
    );

    test(
      'Sqrt Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 4.0, 9.0]), [
          3,
        ], DType.float64);
        final b = sqrt(a);
        expect(b.shape, [3]);
        expect(b.data, [1.0, 2.0, 3.0]);
      }),
    );

    test(
      'Sin Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([0.0, math.pi / 2]), [
          2,
        ], DType.float64);
        final b = sin(a);
        expect(b.shape, [2]);
        expect(b.data[0], closeTo(0.0, 1e-10));
        expect(b.data[1], closeTo(1.0, 1e-10));
      }),
    );

    test(
      'Cos Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([0.0, math.pi]), [
          2,
        ], DType.float64);
        final b = cos(a);
        expect(b.shape, [2]);
        expect(b.data[0], closeTo(1.0, 1e-10));
        expect(b.data[1], closeTo(-1.0, 1e-10));
      }),
    );

    test(
      'Exp Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([0.0, 1.0]), [
          2,
        ], DType.float64);
        final b = exp(a);
        expect(b.shape, [2]);
        expect(b.data[0], closeTo(1.0, 1e-10));
        expect(b.data[1], closeTo(math.e, 1e-10));
      }),
    );

    test(
      'Log Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, math.e]), [
          2,
        ], DType.float64);
        final b = log(a);
        expect(b.shape, [2]);
        expect(b.data[0], closeTo(0.0, 1e-10));
        expect(b.data[1], closeTo(1.0, 1e-10));
      }),
    );

    test(
      'Mean Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final m = mean(a);
        expect(m.scalar, 2.5);

        final m0 = mean(a, axis: 0);
        expect(m0.shape, [2]);
        expect(m0.data, [2.0, 3.0]);
      }),
    );

    test(
      'Min Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 4.0, 2.0]), [
          2,
          2,
        ], DType.float64);
        final m = min(a);
        expect(m.scalar, 1.0);

        final m0 = min(a, axis: 0);
        expect(m0.shape, [2]);
        expect(m0.data, [3.0, 1.0]); // Min along rows: min(3,4)=3, min(1,2)=1
      }),
    );

    test(
      'Max Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 4.0, 2.0]), [
          2,
          2,
        ], DType.float64);
        final m = max(a);
        expect(m.scalar, 4.0);

        final m0 = max(a, axis: 0);
        expect(m0.shape, [2]);
        expect(m0.data, [4.0, 2.0]); // Max along rows: max(3,4)=4, max(1,2)=2
      }),
    );

    test(
      'Prod Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final p = prod(a);
        expect(p.scalar, 24.0);

        final p0 = prod(a, axis: 0);
        expect(p0.shape, [2]);
        expect(p0.toList(), [3.0, 8.0]); // Prod along rows: 1*3=3, 2*4=8
      }),
    );

    test(
      'Variance Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final v = variance(a);
        expect(
          v.scalar,
          closeTo(1.25, 1e-10),
        ); // Mean=2.5, SqDiffs=[2.25, 0.25, 0.25, 2.25], Sum=5, Var=1.25

        final a2 = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [2, 2],
          DType.float64,
        );
        final v0 = variance(a2, axis: 0);
        expect(v0.shape, [2]);
        expect(v0.toList(), [
          1.0,
          1.0,
        ]); // Variance along rows: var([1,3])=1, var([2,4])=1
      }),
    );

    test(
      'Std Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final s = std(a);
        expect(s.scalar, closeTo(math.sqrt(1.25), 1e-10));

        final a2 = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [2, 2],
          DType.float64,
        );
        final s0 = std(a2, axis: 0);
        expect(s0.shape, [2]);
        expect(s0.toList(), [1.0, 1.0]);
      }),
    );

    test(
      'Determinant Operation',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final d = det(a);
        expect(d.scalar, closeTo(-2.0, 1e-10)); // 1*4 - 2*3 = -2

        final a3 = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]),
          [3, 3],
          DType.float64,
        );
        final d3 = det(a3);
        expect(
          d3.scalar,
          closeTo(1.0, 1e-10),
        ); // 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
      }),
    );

    test(
      'Determinant of Singular Matrix',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final d = det(a);
        expect(d.scalar, closeTo(0.0, 1e-10));
      }),
    );

    test(
      'Solve Linear System (1D b)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 1.0, 2.0]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
          2,
        ], DType.float64);
        final x = solve(a, b);
        expect(x.shape, [2]);
        expect(x.data[0], closeTo(2.0, 1e-10));
        expect(x.data[1], closeTo(3.0, 1e-10));
      }),
    );

    test(
      'Solve Linear System (2D b)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 1.0, 2.0]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([9.0, 5.0, 8.0, 5.0]), [
          2,
          2,
        ], DType.float64);
        final x = solve(a, b);
        expect(x.shape, [2, 2]);
        expect(x.data[0], closeTo(2.0, 1e-10));
        expect(x.data[1], closeTo(1.0, 1e-10));
        expect(x.data[2], closeTo(3.0, 1e-10));
        expect(x.data[3], closeTo(2.0, 1e-10));
      }),
    );

    test(
      'Solve Singular Matrix throws',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([5.0, 10.0]), [
          2,
        ], DType.float64);
        expect(() => solve(a, b), throwsA(isA<SingularMatrixException>()));
      }),
    );

    test(
      'Solve Float32',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float32List.fromList([3.0, 1.0, 1.0, 2.0]), [
          2,
          2,
        ], DType.float32);
        final b = NDArray.fromList(Float32List.fromList([9.0, 8.0]), [
          2,
        ], DType.float32);
        final x = solve(a, b);
        expect(x.shape, [2]);
        expect(x.dtype, DType.float32);
        expect(x.data[0], closeTo(2.0, 1e-5));
        expect(x.data[1], closeTo(3.0, 1e-5));
      }),
    );

    test(
      'Solve mismatched dtypes throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
        final b = NDArray.fromList([9.0, 8.0], [2], DType.float32);
        expect(() => solve(a, b), throwsArgumentError);
      }),
    );

    test(
      'Solve Complex128',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2, 2], DType.complex128);
        a.data[0] = Complex(3.0, 0.0);
        a.data[1] = Complex(1.0, 0.0);
        a.data[2] = Complex(1.0, 0.0);
        a.data[3] = Complex(2.0, 0.0);

        final b = NDArray<Complex>.create([2], DType.complex128);
        b.data[0] = Complex(9.0, 0.0);
        b.data[1] = Complex(8.0, 0.0);

        final x = solve(a, b);
        expect(x.shape, [2]);
        expect(x.dtype, DType.complex128);
        expect(x.data[0], Complex(2.0, 0.0));
        expect(x.data[1], Complex(3.0, 0.0));
      }),
    );

    test(
      'Eigen Decompositions (Real Matrix)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 1.0]), [
          2,
          2,
        ], DType.float64);
        final (eigenvalues: w, eigenvectors: vr) = eig(a);

        expect(w.shape, [2]);
        expect(vr.shape, [2, 2]);

        final val0 = w.data[0];
        final val1 = w.data[1];

        expect(
          (val0.real - 3.0).abs() < 1e-5 || (val0.real - (-1.0)).abs() < 1e-5,
          true,
        );
        expect(
          (val1.real - 3.0).abs() < 1e-5 || (val1.real - (-1.0)).abs() < 1e-5,
          true,
        );
        expect(val0.imag, closeTo(0.0, 1e-5));
        expect(val1.imag, closeTo(0.0, 1e-5));
      }),
    );

    test(
      'Eigen Decompositions (Complex Matrix)',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2, 2], DType.complex128);
        a.data[0] = Complex(0.0, 1.0);
        a.data[1] = Complex(0.0, 0.0);
        a.data[2] = Complex(0.0, 0.0);
        a.data[3] = Complex(0.0, 1.0);

        final (eigenvalues: w, eigenvectors: _) = eig(a);

        expect(w.shape, [2]);
        expect(w.data[0], Complex(0.0, 1.0));
        expect(w.data[1], Complex(0.0, 1.0));
      }),
    );

    test(
      'Matrix Inversion',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([4.0, 7.0, 2.0, 6.0]), [
          2,
          2,
        ], DType.float64);
        final b = inv(a);
        expect(b.shape, [2, 2]);
        expect(b.data[0], closeTo(0.6, 1e-10));
        expect(b.data[1], closeTo(-0.7, 1e-10));
        expect(b.data[2], closeTo(-0.2, 1e-10));
        expect(b.data[3], closeTo(0.4, 1e-10));
      }),
    );

    test(
      'Multi-dimensional Indexing',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        expect(a[[0, 0]], 1.0);
        expect(a[[0, 1]], 2.0);
        expect(a[[1, 0]], 3.0);
        expect(a[[1, 1]], 4.0);

        a[[0, 0]] = 10.0;
        expect(a[[0, 0]], 10.0);
      }),
    );

    test(
      'Basic Slicing',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final b = a.slice([Slice(start: 1, stop: 3)]);
        expect(b.shape, [2]);
        expect(b.toList(), [2.0, 3.0]);
      }),
    );

    test(
      'Slicing shares memory',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final b = a.slice([Slice(start: 1, stop: 3)]);
        b.data[0] = Float64(20.0);
        expect(a.data[1], 20.0);
      }),
    );

    test(
      'Slicing with step',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final b = a.slice([Slice(start: 0, stop: 4, step: 2)]);
        expect(b.shape, [2]);
        expect(b.toList(), [1.0, 3.0]);
      }),
    );

    test(
      'Slicing reduces rank with int selector',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = a.slice([Index(1)]); // Select row 1
        expect(b.shape, [2]);
        expect(b.toList(), [3.0, 4.0]);
      }),
    );

    test(
      'Integer Array Indexing',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final b = a.slice([
          Indices([0, 2, 3]),
        ]);
        expect(b.shape, [3]);
        expect(b.toList(), [1.0, 3.0, 4.0]);
      }),
    );

    test(
      'Integer Array Indexing (Copy)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final b = a.slice([
          Indices([0, 2]),
        ]);
        b.data[0] = Float64(10.0);
        expect(a.data[0], 1.0); // Original should not change!
      }),
    );

    test(
      'Comparison Operators',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final b = a > 2.0;
        expect(b.toList(), [false, false, true, true]);

        final c = a < 3.0;
        expect(c.toList(), [true, true, false, false]);

        final d = a.eq(2.0);
        expect(d.toList(), [false, true, false, false]);
      }),
    );

    test(
      'Boolean Masking',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final mask = a > 2.0;
        final b = a.slice([Mask(BooleanMask(mask))]);
        expect(b.shape, [2]);
        expect(b.toList(), [3.0, 4.0]);
      }),
    );

    test(
      'Boolean Masking (Copy)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final mask = a > 2.0;
        final b = a.slice([Mask(BooleanMask(mask))]);
        b.data[0] = Float64(10.0);
        expect(a.data[2], 3.0); // Original should not change!
      }),
    );

    test(
      'Take Method',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = a.take([0, 1], axis: 1); // Select columns 0 and 1
        expect(b.shape, [2, 2]);
        expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

        final c = a.take([1], axis: 0); // Select row 1
        expect(c.shape, [1, 2]);
        expect(c.toList(), [3.0, 4.0]);
      }),
    );

    test(
      'ApplyMask Method',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          4,
        ], DType.float64);
        final mask = a > 2.0;
        final b = a.applyMask(mask);
        expect(b.shape, [2]);
        expect(b.toList(), [3.0, 4.0]);
      }),
    );

    test(
      'Flatten',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = a.flatten();
        expect(b.shape, [4]);
        expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

        b.data[0] = Float64(10.0);
        expect(a.data[0], 1.0);
      }),
    );

    test(
      'Flatten contiguous arrays across all DTypes (FFI block copy)',
      () => NDArray.scope(() {
        for (final dtype in DType.values) {
          final NDArray a;
          final List expected;
          if (dtype == DType.complex128 || dtype == DType.complex64) {
            a = NDArray<Complex>.create([2, 2], dtype as dynamic);
            a.data[0] = Complex(1.0, 1.0);
            a.data[1] = Complex(2.0, 2.0);
            a.data[2] = Complex(3.0, 3.0);
            a.data[3] = Complex(4.0, 4.0);
            expected = [
              Complex(1.0, 1.0),
              Complex(2.0, 2.0),
              Complex(3.0, 3.0),
              Complex(4.0, 4.0),
            ];
          } else if (dtype == DType.boolean) {
            a = NDArray<bool>.fromList(
              [true, false, true, false],
              [2, 2],
              dtype as dynamic,
            );
            expected = [true, false, true, false];
          } else if (dtype == DType.int32 || dtype == DType.int64) {
            a = NDArray.fromList(
              (dtype == DType.int32
                      ? [Int32(1), Int32(2), Int32(3), Int32(4)]
                      : [Int64(1), Int64(2), Int64(3), Int64(4)])
                  as dynamic,
              [2, 2],
              dtype as dynamic,
            );
            expected = [1, 2, 3, 4];
          } else {
            a = NDArray.fromList(
              (dtype == DType.float32
                      ? [Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)]
                      : [
                          Float64(1.0),
                          Float64(2.0),
                          Float64(3.0),
                          Float64(4.0),
                        ])
                  as dynamic,
              [2, 2],
              dtype as dynamic,
            );
            expected = [1.0, 2.0, 3.0, 4.0];
          }

          final b = a.flatten();

          expect(b.shape, [4]);
          expect(b.dtype, dtype);
          expect(b.toList(), expected);
        }
      }),
    );

    test(
      'Ravel (Contiguous)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = a.ravel();
        expect(b.shape, [4]);
        expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

        b.data[0] = Float64(10.0);
        expect(a.data[0], 10.0);
      }),
    );

    test(
      'Ravel (Non-Contiguous)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = a.slice([
          Slice(start: 0, stop: 2),
          Index(0),
        ]); // Select column 0
        final c = b.ravel();
        expect(c.shape, [2]);
        expect(c.toList(), [1.0, 3.0]);

        c.data[0] = Float64(10.0);
        expect(b.data[0], 1.0);
      }),
    );

    test(
      'Transpose (Default)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = a.transposed;
        expect(b.shape, [2, 2]);
        expect(b.toList(), [1.0, 3.0, 2.0, 4.0]);

        b.data[0] = Float64(10.0);
        expect(a.data[0], 10.0);
      }),
    );

    test(
      'Transpose with Axes',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = a.transpose([1, 0]);
        expect(b.shape, [2, 2]);
        expect(b.toList(), [1.0, 3.0, 2.0, 4.0]);
      }),
    );

    test(
      'Concatenate (Axis 0)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
          1,
          2,
        ], DType.float64);
        final c = concatenate([a, b], axis: 0);
        expect(c.shape, [2, 2]);
        expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
      }),
    );

    test(
      'Concatenate (Axis 1)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
          1,
          2,
        ], DType.float64);
        final c = concatenate([a, b], axis: 1);
        expect(c.shape, [1, 4]);
        expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
      }),
    );

    test(
      'vstack',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
          1,
          2,
        ], DType.float64);
        final c = vstack([a, b]);
        expect(c.shape, [2, 2]);
        expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
      }),
    );

    test(
      'hstack',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
          1,
          2,
        ], DType.float64);
        final c = hstack([a, b]);
        expect(c.shape, [1, 4]);
        expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
      }),
    );
    test(
      'Complex Array Creation',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2], DType.complex128);
        expect(a.shape, [2]);
        expect(a.dtype, DType.complex128);

        a.data[0] = Complex(1.0, 2.0);
        a.data[1] = Complex(3.0, 4.0);

        expect(a.data[0].real, 1.0);
        expect(a.data[0].imag, 2.0);
        expect(a.data[1].real, 3.0);
        expect(a.data[1].imag, 4.0);
      }),
    );

    test(
      'Complex Array Addition',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2], DType.complex128);
        a.data[0] = Complex(1.0, 2.0);
        a.data[1] = Complex(3.0, 4.0);

        final b = NDArray<Complex>.create([2], DType.complex128);
        b.data[0] = Complex(10.0, 20.0);
        b.data[1] = Complex(30.0, 40.0);

        final c = add(a, b);
        expect(c.shape, [2]);
        expect(c.data[0], Complex(11.0, 22.0));
        expect(c.data[1], Complex(33.0, 44.0));
      }),
    );

    test(
      'Complex and Real Array Interaction',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2], DType.complex128);
        a.data[0] = Complex(1.0, 2.0);
        a.data[1] = Complex(3.0, 4.0);

        final b = NDArray<double>.fromList([10.0, 20.0], [2], DType.float64);

        final c = add(a, b);
        expect(c.shape, [2]);
        expect(c.data[0], Complex(11.0, 2.0));
        expect(c.data[1], Complex(23.0, 4.0));
      }),
    );

    group('NDArray Bounds, Formats, and Error Exceptions Tests', () {
      test(
        'Transpose axes validations',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => a.transpose([0]), throwsArgumentError);
          expect(() => a.transpose([0, -5]), throwsRangeError);
          expect(() => a.transpose([0, 0]), throwsArgumentError);
        }),
      );

      test(
        'getCell and setCell coordinate checks',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => a.getCell([0]), throwsArgumentError);
          expect(() => a.getCell([0, 5]), throwsRangeError);
          expect(() => a.setCell([0], 1.0 as dynamic), throwsArgumentError);
          expect(() => a.setCell([0, 5], 1.0 as dynamic), throwsRangeError);
        }),
      );

      test(
        'setByMask dimensions validation',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          final mask1D = NDArray<bool>.zeros([2], DType.boolean);
          expect(
            () => a.setByMaskScalar(mask1D, 5.0 as dynamic),
            throwsArgumentError,
          );

          final mask2D = NDArray<bool>.zeros([2, 2], DType.boolean);
          expect(
            () => a.setByMaskScalar(mask2D, 5.0 as dynamic),
            throwsArgumentError,
          );
        }),
      );

      test(
        'setIndices and setIndicesScalar bounds checks',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          final indices = NDArray<Int32>.fromList([Int32(0)], [1], DType.int32);
          expect(
            () => a.setIndicesScalar(indices, Float64(1.0), axis: 5),
            throwsRangeError,
          );

          final badIndices = NDArray<Int32>.fromList(
            [Int32(5)],
            [1],
            DType.int32,
          );
          expect(
            () => a.setIndicesScalar(badIndices, Float64(1.0)),
            throwsRangeError,
          );

          final val = NDArray.zeros([1], DType.float64);
          expect(() => a.setIndices(badIndices, val), throwsRangeError);
        }),
      );

      test(
        'setIndices multi-dimensional slice assignment',
        () => NDArray.scope(() {
          final a = NDArray.zeros([3, 3], DType.float64);

          final indices = NDArray.fromList([0, 2], [2], DType.int32);

          final vals = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 0.0, 0.0]),
            [2, 3],
            DType.float64,
          );

          a.setIndices(indices, vals, axis: 0);

          expect(a.toList(), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
        }),
      );

      test(
        'operator [] fancy index parameter checks',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => a[[0]], throwsArgumentError);

          final badMask = NDArray<bool>.zeros([2, 2], DType.boolean);
          expect(() => a[badMask], throwsArgumentError);
        }),
      );

      test(
        'isSquare and hasSameShape checks',
        () => NDArray.scope(() {
          final squareMat = NDArray.zeros([3, 3], DType.float64);
          final nonSquareMat = NDArray.zeros([3, 4], DType.float64);
          final vector = NDArray.zeros([3], DType.float64);
          final scalar = NDArray.zeros([], DType.float64);

          expect(squareMat.isSquare, isTrue);
          expect(nonSquareMat.isSquare, isFalse);
          expect(vector.isSquare, isFalse);
          expect(scalar.isSquare, isFalse);

          final otherSquareMat = NDArray.ones([3, 3], DType.float64);
          final differentSquareMat = NDArray.zeros([4, 4], DType.float64);

          expect(squareMat.hasSameShape(otherSquareMat), isTrue);
          expect(squareMat.hasSameShape(differentSquareMat), isFalse);
          expect(squareMat.hasSameShape(nonSquareMat), isFalse);
          expect(squareMat.hasSameShape(vector), isFalse);
        }),
      );
    });
    test(
      'arange() zero and negative step validation',
      () => NDArray.scope(() {
        expect(() => NDArray.arange(0.0, 5.0, step: 0.0), throwsArgumentError);
        expect(() => NDArray.arange(0.0, 5.0, step: -1.0), throwsArgumentError);
        expect(() => NDArray.arange(5.0, 0.0, step: 1.0), throwsArgumentError);

        // Valid arange with negative step
        final a = NDArray<double>.arange(
          5.0,
          0.0,
          step: -1.0,
          dtype: DType.float64,
        );
        expect(a.toList(), [5.0, 4.0, 3.0, 2.0, 1.0]);
      }),
    );

    test(
      'NDArray Structural Value Equality operator == and hashCode',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final c = NDArray.fromList([1.0, 2.0, 9.0], [3], DType.float64);
        final d = NDArray.fromList([1.0, 2.0, 3.0], [1, 3], DType.float64);

        expect(a == b, true);
        expect(a == c, false);
        expect(a == d, false); // different shape
        expect(a.hashCode == b.hashCode, true);

        // Non-contiguous view comparisons tests (triggering recursive walkers)
        final parent1 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final parent2 = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final viewT1 =
            parent1.transposed; // non-contiguous: [[1.0, 3.0], [2.0, 4.0]]
        expect(viewT1.isContiguous, false);

        // 1. Non-contiguous == contiguous
        expect(viewT1 == parent2, true);
        expect(viewT1 == parent1, false);

        // 2. Non-contiguous hashCode
        expect(viewT1.hashCode == parent2.hashCode, true);
      }),
    );

    test(
      'DType.complex64 array creation, viewing and operations coverage',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2, 2], DType.complex64);
        expect(a.shape, [2, 2]);
        expect(a.dtype, DType.complex64);

        a.setCell([0, 0], Complex(1.0, 2.0));
        a.setCell([0, 1], Complex(3.0, 4.0));
        a.setCell([1, 0], Complex(5.0, 6.0));
        a.setCell([1, 1], Complex(7.0, 8.0));

        expect(a.getCell([0, 0]), Complex(1.0, 2.0));
        expect(a.getCell([1, 1]), Complex(7.0, 8.0));

        // View creation coverage (parent.dtype == DType.complex64)
        final view = a.slice([Index(0)]); // Select first row
        expect(view.shape, [2]);
        expect(view.dtype, DType.complex64);
        expect(view.getCell([0]), Complex(1.0, 2.0));
        expect(view.getCell([1]), Complex(3.0, 4.0));

        // Operation coverage (add complex64 arrays)
        final b = NDArray<Complex>.create([2, 2], DType.complex64);
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 2; j++) {
            b.setCell([i, j], Complex(10.0, 10.0));
          }
        }

        final c = add(a, b);
        expect(c.dtype, DType.complex64);
        expect(c.getCell([0, 0]), Complex(11.0, 12.0));
        expect(c.getCell([1, 1]), Complex(17.0, 18.0));
      }),
    );

    test(
      'linspace() with num == 1 coverage',
      () => NDArray.scope(() {
        final a = linspace<double>(5.0, 10.0, 1, dtype: DType.float64);
        expect(a.shape, [1]);
        expect(a.toList(), [5.0]);
      }),
    );

    test(
      'DType properties getters isComplex, isFloating, isInteger coverage',
      () {
        expect(DType.complex128.isComplex, true);
        expect(DType.float64.isComplex, false);
        expect(DType.float64.isFloating, true);
        expect(DType.int64.isFloating, false);
        expect(DType.int64.isInteger, true);
        expect(DType.float64.isInteger, false);
      },
    );

    test(
      'NDArray.eye() factory with integer dtype coverage',
      () => NDArray.scope(() {
        final a = NDArray.eye(2, DType.int32);
        expect(a.toList(), [1, 0, 0, 1]);
        expect(a.dtype, DType.int32);
      }),
    );

    test(
      'NDArray.view() FFI constructors with float32 and int64 coverage',
      () => NDArray.scope(() {
        final pF32 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
        final vF32 = NDArray.view(
          pF32,
          shape: [2],
          strides: [1],
          offsetElements: 1,
        );
        expect(vF32.toList(), [2.0, 3.0]);
        expect(vF32.dtype, DType.float32);

        final pI64 = NDArray.fromList([10, 20, 30], [3], DType.int64);
        final vI64 = NDArray.view(
          pI64,
          shape: [2],
          strides: [1],
          offsetElements: 1,
        );
        expect(vI64.toList(), [20, 30]);
        expect(vI64.dtype, DType.int64);
      }),
    );

    test(
      'NDArray.ones() factory with complex dtypes coverage',
      () => NDArray.scope(() {
        final a = NDArray.ones([2], DType.complex128);
        expect(a.toList(), [Complex(1.0, 0.0), Complex(1.0, 0.0)]);
        expect(a.dtype, DType.complex128);

        final b = NDArray.ones([2], DType.complex64);
        expect(b.toList(), [Complex(1.0, 0.0), Complex(1.0, 0.0)]);
        expect(b.dtype, DType.complex64);
      }),
    );

    test(
      'NDArray.eye() complex identity matrix type safety validations',
      () => NDArray.scope(() {
        final eye128 = NDArray<Complex>.eye(3, DType.complex128);
        expect(eye128.dtype, DType.complex128);
        expect(eye128.getCell([0, 0]), Complex(1.0, 0.0));
        expect(eye128.getCell([0, 1]), Complex(0.0, 0.0));
        expect(eye128.getCell([1, 1]), Complex(1.0, 0.0));

        final eye64 = NDArray<Complex>.eye(3, DType.complex64);
        expect(eye64.dtype, DType.complex64);
        expect(eye64.getCell([0, 0]), Complex(1.0, 0.0));
        expect(eye64.getCell([1, 1]), Complex(1.0, 0.0));
      }),
    );

    test(
      'arange() and linspace() type safety with complex dtypes',
      () => NDArray.scope(() {
        // 1. arange with complex128
        final a128 = NDArray<Complex>.arange(0, 3, dtype: DType.complex128);
        expect(a128.shape, [3]);
        expect(a128.dtype, DType.complex128);
        expect(a128.toList(), [
          Complex(0.0, 0.0),
          Complex(1.0, 0.0),
          Complex(2.0, 0.0),
        ]);

        // 2. arange with complex64
        final a64 = NDArray<Complex>.arange(1, 3, dtype: DType.complex64);
        expect(a64.shape, [2]);
        expect(a64.dtype, DType.complex64);
        expect(a64.toList(), [Complex(1.0, 0.0), Complex(2.0, 0.0)]);

        // 3. linspace with complex128
        final l128 = linspace<Complex>(
          Complex(1.0, 0.0),
          Complex(2.0, 0.0),
          3,
          dtype: DType.complex128,
        );
        expect(l128.shape, [3]);
        expect(l128.dtype, DType.complex128);
        expect(l128.toList(), [
          Complex(1.0, 0.0),
          Complex(1.5, 0.0),
          Complex(2.0, 0.0),
        ]);

        // 4. linspace with single-element num == 1 and complex64
        final l64 = linspace<Complex>(
          Complex(5.0, 0.0),
          Complex(10.0, 0.0),
          1,
          dtype: DType.complex64,
        );
        expect(l64.shape, [1]);
        expect(l64.dtype, DType.complex64);
        expect(l64.toList(), [Complex(5.0, 0.0)]);
      }),
    );

    test(
      'FFI native C flatten() and hashCode() correctness and invariants across all DTypes',
      () => NDArray.scope(() {
        // 1. Float64
        final f64 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final f64T = f64.transposed;
        final f64Flat = f64T.flatten();
        expect(f64Flat.isContiguous, true);
        expect(f64Flat.toList(), [1.0, 3.0, 2.0, 4.0]);
        final f64Contig = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(f64T == f64Contig, true);
        expect(f64T.hashCode == f64Contig.hashCode, true);

        // 2. Float32
        final f32 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final f32T = f32.transposed;
        final f32Flat = f32T.flatten();
        expect(f32Flat.isContiguous, true);
        expect(f32Flat.toList(), [1.0, 3.0, 2.0, 4.0]);
        final f32Contig = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float32,
        );
        expect(f32T == f32Contig, true);
        expect(f32T.hashCode == f32Contig.hashCode, true);

        // 3. Int64
        final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
        final i64T = i64.transposed;
        final i64Flat = i64T.flatten();
        expect(i64Flat.isContiguous, true);
        expect(i64Flat.toList(), [1, 3, 2, 4]);
        final i64Contig = NDArray.fromList([1, 3, 2, 4], [2, 2], DType.int64);
        expect(i64T == i64Contig, true);
        expect(i64T.hashCode == i64Contig.hashCode, true);

        // 4. Int32
        final i32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32T = i32.transposed;
        final i32Flat = i32T.flatten();
        expect(i32Flat.isContiguous, true);
        expect(i32Flat.toList(), [1, 3, 2, 4]);
        final i32Contig = NDArray.fromList([1, 3, 2, 4], [2, 2], DType.int32);
        expect(i32T == i32Contig, true);
        expect(i32T.hashCode == i32Contig.hashCode, true);

        // 5. Complex128
        final c128 = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );
        final c128T = c128.transposed;
        final c128Flat = c128T.flatten();
        expect(c128Flat.isContiguous, true);
        expect(c128Flat.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);
        final c128Contig = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(3.0, 3.0),
            Complex(2.0, 2.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );
        expect(c128T == c128Contig, true);
        expect(c128T.hashCode == c128Contig.hashCode, true);

        // 6. Complex64
        final c64 = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex64,
        );
        final c64T = c64.transposed;
        final c64Flat = c64T.flatten();
        expect(c64Flat.isContiguous, true);
        expect(c64Flat.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);
        final c64Contig = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(3.0, 3.0),
            Complex(2.0, 2.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex64,
        );
        expect(c64T == c64Contig, true);
        expect(c64T.hashCode == c64Contig.hashCode, true);

        // 7. Boolean
        final b = NDArray.fromList(
          [true, false, true, false],
          [2, 2],
          DType.boolean,
        );
        final bT = b.transposed;
        final bFlat = bT.flatten();
        expect(bFlat.isContiguous, true);
        expect(bFlat.toList(), [true, true, false, false]);
        final bContig = NDArray.fromList(
          [true, true, false, false],
          [2, 2],
          DType.boolean,
        );
        expect(bT == bContig, true);
        expect(bT.hashCode == bContig.hashCode, true);
      }),
    );

    test("Contiguity checking ignores stride mismatches on unit dimensions (shape[i] <= 1)", () {
      NDArray.scope(() {
        // Create an array with a unit dimension and an arbitrary stride on that dimension
        final arr = NDArray.fromList([1, 2, 3], [1, 3], DType.int64);
        // By slicing or creating a view where dimension 0 has size 1 with non-standard stride
        // We can test _checkContiguous via broadcastTo or reshape or creating view
        final view = NDArray.view(arr, shape: [1, 1, 3], strides: [999, 3, 1]);
        expect(view.isContiguous, true);
      });
    });
  });
}
