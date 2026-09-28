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
  group('NDArray Complex Components (real, imag) Tests', () {
    test(
      'real() and imag() basic complex128 contiguous checks',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.fromList(
          [Complex(3.5, 4.5), Complex(-1.0, 0.0), Complex(0.0, -2.5)],
          [3],
          DType.complex128,
        );

        final r = real(a);
        expect(r.dtype, DType.float64);
        expect(r.shape, [3]);
        expect(r.toList(), [3.5, -1.0, 0.0]);

        final im = imag(a);
        expect(im.dtype, DType.float64);
        expect(im.shape, [3]);
        expect(im.toList(), [4.5, 0.0, -2.5]);
      }),
    );

    test(
      'real() and imag() support complex64',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.fromList(
          [Complex(1.5, -2.5), Complex(0.0, 3.0)],
          [2],
          DType.complex64,
        );

        final r = real(a);
        expect(r.dtype, DType.float32);
        expect(r.toList(), [1.5, 0.0]);

        final im = imag(a);
        expect(im.dtype, DType.float32);
        expect(im.toList(), [-2.5, 3.0]);
      }),
    );

    test(
      'real() on already real arrays returns zero-copy views',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final r = real(a);

        expect(r.dtype, DType.float64);
        expect(r.toList(), [10.0, 20.0]);

        // Mutating view updates parent zero-copy!
        r.setCell([0], 99.0);
        expect(a.getCell([0]), 99.0);
      }),
    );

    test(
      'imag() on already real arrays returns zero-filled arrays',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final im = imag(a);

        expect(im.dtype, DType.float64);
        expect(im.toList(), [0.0, 0.0]);
      }),
    );

    test(
      'recycler out parameter checks',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );

        final out = NDArray.create([2], DType.float64);
        final res = real(a, out: out);

        expect(identical(res, out), true);
        expect(out.toList(), [1.0, 3.0]);
      }),
    );

    test(
      'real() recycler out parameter checks when a is already real',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final out = NDArray.create([2], DType.float64);
        final res = real(a, out: out);

        expect(identical(res, out), true);
        expect(out.toList(), [10.0, 20.0]);
      }),
    );

    test(
      'imag() recycler out parameter checks when a is already real',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final out = NDArray.create([2], DType.float64);
        final res = imag(a, out: out);

        expect(identical(res, out), true);
        expect(out.toList(), [0.0, 0.0]);
      }),
    );

    test(
      'recycler out shape and dtype mismatch throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2], DType.complex128);
        final wrongShape = NDArray.create([3], DType.float64);
        final wrongDType = NDArray.create([2], DType.int32);

        expect(() => real(a, out: wrongShape), throwsArgumentError);
        expect(() => real(a, out: wrongDType), throwsArgumentError);
        expect(() => imag(a, out: wrongShape), throwsArgumentError);
        expect(() => imag(a, out: wrongDType), throwsArgumentError);

        final realArr = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        expect(() => real(realArr, out: wrongShape), throwsArgumentError);
        expect(() => real(realArr, out: wrongDType), throwsArgumentError);
        expect(() => imag(realArr, out: wrongShape), throwsArgumentError);
        expect(() => imag(realArr, out: wrongDType), throwsArgumentError);
      }),
    );

    test(
      'strided non-contiguous complex128 addition walks native C kernels',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final b = NDArray<Complex>.fromList(
          [
            Complex(10.0, 10.0),
            Complex(20.0, 20.0),
            Complex(30.0, 30.0),
            Complex(40.0, 40.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final viewA = a.transposed;
        final viewB = b.transposed;

        expect(viewA.isContiguous, false);
        expect(viewB.isContiguous, false);

        final res = add(viewA, viewB);

        expect(res.shape, [2, 2]);
        expect(res.dtype, DType.complex128);

        final resList = res.toList();
        expect(resList[0], Complex(11.0, 12.0));
        expect(resList[1], Complex(35.0, 36.0));
        expect(resList[2], Complex(23.0, 24.0));
        expect(resList[3], Complex(47.0, 48.0));
      }),
    );

    test(
      'strided non-contiguous complex128 subtraction fallback elementswise sweeps',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.fromList(
          [
            Complex(10.0, 10.0),
            Complex(20.0, 20.0),
            Complex(30.0, 30.0),
            Complex(40.0, 40.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final b = NDArray<Complex>.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final aT = a.transposed;
        final bT = b.transposed;
        final res1 = subtract(aT, bT);
        expect(res1.shape, [2, 2]);
        expect(res1.dtype, DType.complex128);
        final res1List = res1.toList();
        expect(res1List[0], Complex(9.0, 8.0));
        expect(res1List[1], Complex(25.0, 24.0));
        expect(res1List[2], Complex(17.0, 16.0));
        expect(res1List[3], Complex(33.0, 32.0));

        final realArr = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final realArrT = realArr.transposed;
        final res2 = subtract(aT, realArrT);
        expect(res2.shape, [2, 2]);
        expect(res2.dtype, DType.complex128);
        final res2List = res2.toList();
        expect(res2List[0], Complex(9.0, 10.0));
        expect(res2List[1], Complex(27.0, 30.0));
        expect(res2List[2], Complex(18.0, 20.0));
        expect(res2List[3], Complex(36.0, 40.0));
      }),
    );

    test(
      'disposed arrays throw StateError',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2], DType.complex128);
        a.dispose();
        expect(() => real(a), throwsStateError);
        expect(() => imag(a), throwsStateError);
      }),
    );
  });
}
