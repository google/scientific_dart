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
  group('Calculus Solver: trapz() composite trapezoidal integrals', () {
    test(
      '1D Contiguous array constant spacing dx=1.0',
      () => NDArray.scope(() {
        final y = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
        final res = trapz(y); // Default step(1.0)
        expect(res.shape, []);
        expect(
          res.toList()[0],
          closeTo(4.5, 1e-9),
        ); // 0.5*(1+2)*1 + 0.5*(2+4)*1 = 1.5 + 3 = 4.5
      }),
    );

    test(
      '1D Contiguous array custom constant spacing dx=2.0',
      () => NDArray.scope(() {
        final y = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
        final res = trapz(y, spacing: Spacing.step(2.0));
        expect(res.shape, []);
        expect(res.toList()[0], closeTo(9.0, 1e-9));
      }),
    );

    test(
      '1D Contiguous array non-uniform (variable) spacing using coordinates array',
      () => NDArray.scope(() {
        final y = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
        final x = [0.0, 2.0, 3.0];
        final res = trapz(y, spacing: Spacing.coordinates(x));
        expect(res.shape, []);
        expect(
          res.toList()[0],
          closeTo(6.0, 1e-9),
        ); // 0.5*(1+2)*2 + 0.5*(2+4)*1 = 3.0 + 3.0 = 6.0
      }),
    );

    test(
      'Float32 precision consistency',
      () => NDArray.scope(() {
        final yFloat = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float32);
        final resFloat = trapz(yFloat);
        expect(resFloat.dtype, DType.float32);
        expect(resFloat.toList()[0], closeTo(4.5, 1e-7));
      }),
    );

    test(
      'Complex numbers composite integrations',
      () => NDArray.scope(() {
        final y = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(2.0, 3.0), Complex(4.0, 5.0)],
          [3],
          DType.complex128,
        );
        final res = trapz(y);
        expect(res.dtype, DType.complex128);
        expect(res.toList()[0].real, closeTo(4.5, 1e-9));
        expect(
          res.toList()[0].imag,
          closeTo(6.5, 1e-9),
        ); // 0.5*(2+3)*1 + 0.5*(3+5)*1 = 2.5 + 4 = 6.5
      }),
    );

    test(
      '2D multi-dimensional integrations along specified axes',
      () => NDArray.scope(() {
        // 2 rows, 3 columns
        final y = NDArray.fromList(
          [1.0, 2.0, 4.0, 2.0, 4.0, 8.0],
          [2, 3],
          DType.float64,
        );

        // Integrate along columns (axis = 1)
        final resAxis1 = trapz(y, axis: 1);
        expect(resAxis1.shape, [2]);
        expect(resAxis1.toList()[0], closeTo(4.5, 1e-9));
        expect(resAxis1.toList()[1], closeTo(9.0, 1e-9));

        // Integrate along rows (axis = 0)
        final resAxis0 = trapz(y, axis: 0);
        expect(resAxis0.shape, [3]);
        expect(resAxis0.toList()[0], closeTo(1.5, 1e-9)); // 0.5*(1+2)*1 = 1.5
        expect(resAxis0.toList()[1], closeTo(3.0, 1e-9)); // 0.5*(2+4)*1 = 3.0
        expect(resAxis0.toList()[2], closeTo(6.0, 1e-9)); // 0.5*(4+8)*1 = 6.0
      }),
    );

    test(
      'Preconditions & resource cleanup checking',
      () => NDArray.scope(() {
        final y = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final xBad = [0.0, 1.0, 2.0];

        expect(
          () => trapz(y, spacing: Spacing.coordinates(xBad)),
          throwsArgumentError,
        );

        // Fail on booleans, succeed on integers
        final boolArr = NDArray<Boolean>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );
        expect(() => trapz(boolArr as dynamic), throwsArgumentError);

        final intArr = NDArray.fromList([1, 2], [2], DType.int64);
        final intRes = trapz(intArr);
        expect(intRes, isA<NDArray<Float64>>());
        expect(intRes.dtype, DType.float64);
        expect(intRes.scalar, closeTo(1.5, 1e-9));

        y.dispose();
        expect(() => trapz(y), throwsStateError);
      }),
    );
  });

  group('Calculus Solver: gradient() and gradientArray() N-D gradients', () {
    test(
      '1D Constant spacing edgeOrder=1',
      () => NDArray.scope(() {
        final f = NDArray.fromList([1.0, 2.0, 4.0, 7.0], [4], DType.float64);
        final res = gradient(f, spacing: Spacing.step(1.0), edgeOrder: 1);
        expect(res.shape, [4]);
        expect(
          res.toList()[0],
          closeTo(1.0, 1e-9),
        ); // one-sided forward: (2.0 - 1.0) / 1
        expect(res.toList()[1], closeTo(1.5, 1e-9)); // central: (4.0 - 1.0) / 2
        expect(res.toList()[2], closeTo(2.5, 1e-9)); // central: (7.0 - 2.0) / 2
        expect(
          res.toList()[3],
          closeTo(3.0, 1e-9),
        ); // one-sided backward: (7.0 - 4.0) / 1
      }),
    );

    test(
      '1D Constant spacing edgeOrder=2 parabolic boundaries',
      () => NDArray.scope(() {
        final f = NDArray.fromList([1.0, 2.0, 4.0, 7.0], [4], DType.float64);
        final res = gradient(f, spacing: Spacing.step(1.0), edgeOrder: 2);
        expect(res.shape, [4]);
        expect(
          res.toList()[0],
          closeTo(0.5, 1e-9),
        ); // parabolic forward: (-3*1 + 4*2 - 4) / 2 = 0.5
        expect(res.toList()[1], closeTo(1.5, 1e-9)); // central: 1.5
        expect(res.toList()[2], closeTo(2.5, 1e-9)); // central: 2.5
        expect(
          res.toList()[3],
          closeTo(3.5, 1e-9),
        ); // parabolic backward: (3*7 - 4*4 + 2) / 2 = 3.5
      }),
    );

    test(
      '1D Variable (non-uniform) spacing edgeOrder=1',
      () => NDArray.scope(() {
        final f = NDArray.fromList([1.0, 2.0, 4.0, 7.0], [4], DType.float64);
        final res = gradient(
          f,
          spacing: Spacing.coordinates([0.0, 1.0, 3.0, 4.0]),
          edgeOrder: 1,
        );
        expect(res.shape, [4]);
        expect(
          res.toList()[0],
          closeTo(1.0, 1e-9),
        ); // one-sided: (2-1)/(1-0) = 1.0
        expect(
          res.toList()[1],
          closeTo(1.0, 1e-9),
        ); // central: (1^2*4 + 3*2 - 4*1)/6 = 1.0
        expect(res.toList()[2], closeTo(2.333333333, 1e-7)); // central: 7/3
        expect(
          res.toList()[3],
          closeTo(3.0, 1e-9),
        ); // one-sided: (7-4)/(4-3) = 3.0
      }),
    );

    test(
      'Complex numbers gradient component walks',
      () => NDArray.scope(() {
        final f = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(2.0, 3.0), Complex(4.0, 5.0)],
          [3],
          DType.complex128,
        );
        final res = gradient(f, spacing: Spacing.step(1.0));
        expect(res.dtype, DType.complex128);
        expect(res.toList()[0].real, closeTo(1.0, 1e-9));
        expect(res.toList()[0].imag, closeTo(1.0, 1e-9));
        expect(res.toList()[1].real, closeTo(1.5, 1e-9));
        expect(res.toList()[1].imag, closeTo(1.5, 1e-9));
      }),
    );

    test(
      '2D Multi-Dimensional arrays single-axis gradients',
      () => NDArray.scope(() {
        final f = NDArray.fromList(
          [1.0, 2.0, 4.0, 2.0, 4.0, 8.0],
          [2, 3],
          DType.float64,
        );

        // Gradient along columns (axis = 1)
        final resCol = gradient(f, spacing: Spacing.step(1.0), axis: 1);
        expect(resCol.shape, [2, 3]);
        expect(resCol.toList()[0], closeTo(1.0, 1e-9)); // (2-1)/1
        expect(resCol.toList()[1], closeTo(1.5, 1e-9)); // (4-1)/2
        expect(resCol.toList()[2], closeTo(2.0, 1e-9)); // (4-2)/1

        // Gradient along rows (axis = 0)
        final resRow = gradient(f, spacing: Spacing.step(1.0), axis: 0);
        expect(resRow.shape, [2, 3]);
        expect(resRow.toList()[0], closeTo(1.0, 1e-9)); // (2-1)/1
        expect(resRow.toList()[3], closeTo(1.0, 1e-9)); // (2-1)/1 (backward)
      }),
    );

    test(
      'gradientArray() multiple axes calculations',
      () => NDArray.scope(() {
        final f = NDArray.fromList(
          [1.0, 2.0, 4.0, 2.0, 4.0, 8.0],
          [2, 3],
          DType.float64,
        );
        final grads = gradientArray(
          f,
          spacings: [Spacing.step(1.0), Spacing.step(2.0)],
        );

        expect(grads.length, 2);
        // Gradient along row axis (axis = 0), spacing dx0 = 1.0
        expect(grads[0].shape, [2, 3]);
        expect(grads[0].toList()[0], closeTo(1.0, 1e-9));

        // Gradient along column axis (axis = 1), spacing dx1 = 2.0
        expect(grads[1].shape, [2, 3]);
        expect(grads[1].toList()[0], closeTo(0.5, 1e-9)); // (2-1)/2
        expect(grads[1].toList()[1], closeTo(0.75, 1e-9)); // (4-1)/4
      }),
    );

    test(
      'gradientArray() default spacing',
      () => NDArray.scope(() {
        final f = NDArray.fromList(
          [1.0, 2.0, 4.0, 2.0, 4.0, 8.0],
          [2, 3],
          DType.float64,
        );
        final grads = gradientArray(f); // spacing: null

        expect(grads.length, 2);
        expect(grads[0].shape, [2, 3]);
        expect(grads[1].shape, [2, 3]);
        expect(grads[0].toList()[0], closeTo(1.0, 1e-9));
        expect(grads[1].toList()[0], closeTo(1.0, 1e-9));
      }),
    );

    test(
      'Invalid/out-of-bounds errors',
      () => NDArray.scope(() {
        final f = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        expect(
          () => gradient(f, spacing: Spacing.coordinates([1.0, 2.0, 3.0])),
          throwsArgumentError,
        );
        expect(
          () => gradient(f, spacing: Spacing.step(1.0), edgeOrder: 3),
          throwsArgumentError,
        );
        final boolArr = NDArray<Boolean>.fromList(
          [true, false],
          [2],
          DType.boolean,
        );
        expect(() => gradient(boolArr), throwsArgumentError);
        expect(() => gradientArray(boolArr), throwsArgumentError);
        final intArr = NDArray.fromList([1, 3, 6], [3], DType.int64);
        final intGrad = gradient(intArr);
        expect(intGrad.dtype, DType.float64);
        expect(intGrad.toList(), equals([2.0, 2.5, 3.0]));
      }),
    );
  });

  group('Calculus Solver: Complex Spacing support', () {
    test(
      'trapz() contour integration along imaginary axis',
      () => NDArray.scope(() {
        // Integrate f(z) = z from 0 to i
        // Points: 0, 0.5i, i
        final y = NDArray.fromList(
          [Complex(0, 0), Complex(0, 0.5), Complex(0, 1.0)],
          [3],
          DType.complex128,
        );
        // Constant spacing dx = 0.5i
        final res = trapz(y, spacing: Spacing.step(Complex(0, 0.5)));
        expect(res.dtype, DType.complex128);
        expect(res.toList()[0].real, closeTo(-0.5, 1e-9));
        expect(res.toList()[0].imag, closeTo(0.0, 1e-9));
      }),
    );

    test(
      'gradient() with complex coordinates',
      () => NDArray.scope(() {
        // f(z) = z^2. f'(z) = 2z.
        // Points z: 0, i, 2i
        // f(z): 0, -1, -4
        final f = NDArray.fromList(
          [Complex(0, 0), Complex(-1, 0), Complex(-4, 0)],
          [3],
          DType.complex128,
        );
        // Coordinates z: [0, i, 2i]
        final coords = [Complex(0, 0), Complex(0, 1), Complex(0, 2)];
        final grads = gradient(f, spacing: Spacing.coordinates(coords));

        // At z=i, f'(i) = 2i.
        // Central diff: (f(2i) - f(0)) / (2i - 0) = (-4 - 0) / 2i = -2 / i = 2i.
        expect(grads.toList()[1].real, closeTo(0.0, 1e-9));
        expect(grads.toList()[1].imag, closeTo(2.0, 1e-9));
      }),
    );

    test(
      'gradientArray() with complexSpacing',
      () => NDArray.scope(() {
        final f = NDArray.fromList(
          [Complex(0, 0), Complex(-1, 0), Complex(-4, 0)],
          [3],
          DType.complex128,
        );
        final coords = [Complex(0, 0), Complex(0, 1), Complex(0, 2)];
        final grads = gradientArray(f, spacings: [Spacing.coordinates(coords)]);

        expect(grads.length, 1);
        expect(grads[0].toList()[1].imag, closeTo(2.0, 1e-9));
      }),
    );

    test(
      'trapz() contour integration along imaginary axis with complex64',
      () => NDArray.scope(() {
        // Integrate f(z) = z from 0 to i
        // Points: 0, 0.5i, i
        final y = NDArray.fromList(
          [Complex(0, 0), Complex(0, 0.5), Complex(0, 1.0)],
          [3],
          DType.complex64,
        );
        // Constant spacing dx = 0.5i
        final res = trapz(y, spacing: Spacing.step(Complex(0, 0.5)));
        expect(res.dtype, DType.complex64);
        expect(res.toList()[0].real, closeTo(-0.5, 1e-5));
        expect(res.toList()[0].imag, closeTo(0.0, 1e-5));
      }),
    );

    test(
      'trapz() with complex coordinates on complex64',
      () => NDArray.scope(() {
        final y = NDArray.fromList(
          [Complex(0, 0), Complex(0, 0.5), Complex(0, 1.0)],
          [3],
          DType.complex64,
        );
        final coords = [Complex(0, 0), Complex(0, 0.5), Complex(0, 1.0)];
        final res = trapz(y, spacing: Spacing.coordinates(coords));
        expect(res.dtype, DType.complex64);
        expect(res.toList()[0].real, closeTo(-0.5, 1e-5));
        expect(res.toList()[0].imag, closeTo(0.0, 1e-5));
      }),
    );
  });

  group('Integer Spacing Bug Repro', () {
    test('trapz and gradient with integer step spacing', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
        final result = trapz(y, spacing: const Spacing.step(1));
        expect(result.shape, []);
        expect(result.scalar, 4.5);
        final grad = gradient(y, spacing: const Spacing.step(2));
        expect(grad.shape, [3]);
        expect(grad.toList()[0], closeTo(0.5, 1e-9));
        expect(grad.toList()[1], closeTo(0.75, 1e-9));
        expect(grad.toList()[2], closeTo(1.0, 1e-9));
      });
    });
    test(
      'linalg.diff() discrete difference correctness',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 4, 7, 0], [5], DType.int64);

        // 1. Flat diff (n=1)
        final d1 = diff(a);
        expect(d1.toList(), [1, 2, 3, -7]);

        // 2. Flat diff (n=2)
        final d2 = diff(a, n: 2);
        expect(d2.toList(), [1, 1, -10]);

        // 3. 2D Matrix diff along axis=1
        final mat = NDArray.fromList(
          [1.0, 3.0, 9.0, 2.0, 5.0, 15.0],
          [2, 3],
          DType.float64,
        );
        final dAxis1 = diff(mat, axis: 1);
        expect(dAxis1.toList(), [2.0, 6.0, 3.0, 10.0]);

        // 4. Complex diff
        final aComp = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 5.0), Complex(6.0, 10.0)],
          [3],
          DType.complex128,
        );
        final dComp = diff(aComp);
        expect(dComp.getCell([0]), Complex(2.0, 3.0));
        expect(dComp.getCell([1]), Complex(3.0, 5.0));

        // 5. in-place recycler out reuse
        final recycler = NDArray.zeros([2, 2], DType.float64);
        final dRecycled = diff(mat, axis: 1, out: recycler);
        expect(identical(dRecycled, recycler), true);
        expect(dRecycled.toList(), [2.0, 6.0, 3.0, 10.0]);
      }),
    );
  });

  group('Strided Gradient Tests', () {
    test(
      'float64 strided gradient',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          List<double>.generate(10, (i) => i.toDouble() * i.toDouble()),
          [10],
          DType.float64,
        );
        final view = parent.slice([const Slice(start: 0, stop: 10, step: 2)]);
        expect(view.isContiguous, false);

        final grad = gradient(view, spacing: const Spacing.step(2.0));
        expect(grad.dtype, DType.float64);
        expect(grad.shape, [5]);
        expect(grad.toList()[0], closeTo(2.0, 1e-9));
        expect(grad.toList()[1], closeTo(4.0, 1e-9));
        expect(grad.toList()[2], closeTo(8.0, 1e-9));
        expect(grad.toList()[3], closeTo(12.0, 1e-9));
        expect(grad.toList()[4], closeTo(14.0, 1e-9));
      }),
    );

    test(
      'float32 strided gradient',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          List<double>.generate(10, (i) => i.toDouble() * i.toDouble()),
          [10],
          DType.float32,
        );
        final view = parent.slice([const Slice(start: 0, stop: 10, step: 2)]);
        expect(view.isContiguous, false);

        final grad = gradient(view, spacing: const Spacing.step(2.0));
        expect(grad.dtype, DType.float32);
        expect(grad.shape, [5]);
        expect(grad.toList()[0], closeTo(2.0, 1e-5));
        expect(grad.toList()[1], closeTo(4.0, 1e-5));
        expect(grad.toList()[2], closeTo(8.0, 1e-5));
        expect(grad.toList()[3], closeTo(12.0, 1e-5));
        expect(grad.toList()[4], closeTo(14.0, 1e-5));
      }),
    );

    test(
      'complex128 strided gradient',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          List<Complex>.generate(
            10,
            (i) => Complex(i.toDouble() * i.toDouble(), i.toDouble()),
          ),
          [10],
          DType.complex128,
        );
        final view = parent.slice([const Slice(start: 0, stop: 10, step: 2)]);
        expect(view.isContiguous, false);

        final grad = gradient(view, spacing: const Spacing.step(2.0));
        expect(grad.dtype, DType.complex128);
        expect(grad.shape, [5]);

        expect(grad.toList()[0].real, closeTo(2.0, 1e-9));
        expect(grad.toList()[0].imag, closeTo(1.0, 1e-9));
        expect(grad.toList()[1].real, closeTo(4.0, 1e-9));
        expect(grad.toList()[1].imag, closeTo(1.0, 1e-9));
        expect(grad.toList()[2].real, closeTo(8.0, 1e-9));
        expect(grad.toList()[2].imag, closeTo(1.0, 1e-9));
      }),
    );

    test(
      'complex64 strided gradient',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          List<Complex>.generate(
            10,
            (i) => Complex(i.toDouble() * i.toDouble(), i.toDouble()),
          ),
          [10],
          DType.complex64,
        );
        final view = parent.slice([const Slice(start: 0, stop: 10, step: 2)]);
        expect(view.isContiguous, false);

        final grad = gradient(view, spacing: const Spacing.step(2.0));
        expect(grad.dtype, DType.complex64);
        expect(grad.shape, [5]);

        expect(grad.toList()[0].real, closeTo(2.0, 1e-5));
        expect(grad.toList()[0].imag, closeTo(1.0, 1e-5));
        expect(grad.toList()[1].real, closeTo(4.0, 1e-5));
        expect(grad.toList()[1].imag, closeTo(1.0, 1e-5));
        expect(grad.toList()[2].real, closeTo(8.0, 1e-5));
        expect(grad.toList()[2].imag, closeTo(1.0, 1e-5));
      }),
    );

    test(
      'gradientArray on strided 2D view',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          List<double>.generate(16, (i) => i.toDouble()),
          [4, 4],
          DType.float64,
        );
        final view = parent.transposed;
        expect(view.isContiguous, false);

        final grads = gradientArray(view);
        expect(grads.length, 2);
        expect(grads[0].shape, [4, 4]);
        expect(grads[1].shape, [4, 4]);

        expect(grads[0].toList().every((e) => (e - 1.0).abs() < 1e-9), true);
        expect(grads[1].toList().every((e) => (e - 4.0).abs() < 1e-9), true);
      }),
    );
  });

  group('Calculus Preconditions and Error Handling', () {
    test(
      'disposed input throws StateError',
      () => NDArray.scope(() {
        final a = NDArray.zeros([8], DType.float64);
        a.dispose();

        expect(() => trapz(a), throwsStateError);
        expect(() => gradient(a), throwsStateError);
        expect(() => gradientArray(a), throwsStateError);
        expect(() => diff(a), throwsStateError);
      }),
    );

    test(
      'disposed out buffer throws StateError',
      () => NDArray.scope(() {
        final a = NDArray.zeros([8], DType.float64);

        final outTrapz = NDArray.zeros([], DType.float64);
        outTrapz.dispose();
        expect(() => trapz(a, out: outTrapz), throwsStateError);

        final outGrad = NDArray.zeros([8], DType.float64);
        outGrad.dispose();
        expect(() => gradient(a, out: outGrad), throwsStateError);
        expect(() => gradientArray(a, out: [outGrad]), throwsStateError);

        final outDiff = NDArray.zeros([7], DType.float64);
        outDiff.dispose();
        expect(() => diff(a, out: outDiff), throwsStateError);
      }),
    );

    test(
      'incompatible out buffer shape or dtype throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.zeros([8], DType.float64);

        final outGradWrongShape = NDArray.zeros([9], DType.float64);
        expect(() => gradient(a, out: outGradWrongShape), throwsArgumentError);

        final outGradWrongDtype = NDArray.zeros([8], DType.float32);
        expect(() => gradient(a, out: outGradWrongDtype), throwsArgumentError);

        final outDiffWrongShape = NDArray.zeros([8], DType.float64);
        expect(() => diff(a, out: outDiffWrongShape), throwsArgumentError);

        final outDiffWrongDtype = NDArray.zeros([7], DType.float32);
        expect(() => diff(a, out: outDiffWrongDtype), throwsArgumentError);
      }),
    );

    test(
      'out of bounds axis throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.zeros([8], DType.float64);

        expect(() => trapz(a, axis: 1), throwsArgumentError);
        expect(() => trapz(a, axis: -2), throwsArgumentError);

        expect(() => gradient(a, axis: 1), throwsArgumentError);
        expect(() => gradient(a, axis: -2), throwsArgumentError);

        expect(() => diff(a, axis: 1), throwsArgumentError);
        expect(() => diff(a, axis: -2), throwsArgumentError);
      }),
    );
  });
}
