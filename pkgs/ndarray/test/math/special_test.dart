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
import 'package:ndarray/ndarray.dart';

void main() {
  group('Special Functions Tests (gamma, erf, i0, sinc asymptotic limits)', () {
    test(
      'gamma() float64 and float32 contiguous and strided',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final g = gamma(a);
        expect(g.dtype, DType.float64);
        expect(g.toList(), [
          closeTo(1.0, 1e-9),
          closeTo(1.0, 1e-9),
          closeTo(2.0, 1e-9),
          closeTo(6.0, 1e-9),
          closeTo(24.0, 1e-9),
        ]);

        final a32 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
        final g32 = gamma(a32);
        expect(g32.dtype, DType.float32);
        expect(g32.toList(), [
          closeTo(1.0, 1e-5),
          closeTo(1.0, 1e-5),
          closeTo(2.0, 1e-5),
          closeTo(6.0, 1e-5),
        ]);

        // Integer promotion
        final aInt = NDArray.fromList([1, 2, 3, 4], [4], DType.int64);
        final gInt = gamma(aInt);
        expect(gInt.dtype, DType.float64);
        expect(gInt.toList(), [
          closeTo(1.0, 1e-9),
          closeTo(1.0, 1e-9),
          closeTo(2.0, 1e-9),
          closeTo(6.0, 1e-9),
        ]);

        // Strided
        final aStrided = a.slice([
          Slice(start: 0, stop: 5, step: 2),
        ]); // [1.0, 3.0, 5.0]
        final gStrided = gamma(aStrided);
        expect(gStrided.toList(), [
          closeTo(1.0, 1e-9),
          closeTo(2.0, 1e-9),
          closeTo(24.0, 1e-9),
        ]);

        // Where mask and out parameter
        final out = NDArray.zeros([5], DType.float64);
        final mask = NDArray.fromList(
          [true, false, true, false, true],
          [5],
          DType.boolean,
        );
        gamma(a, where: mask, out: out);
        expect(out.toList()[0], closeTo(1.0, 1e-9));
        expect(out.toList()[1], 0.0);
        expect(out.toList()[2], closeTo(2.0, 1e-9));
        expect(out.toList()[3], 0.0);
        expect(out.toList()[4], closeTo(24.0, 1e-9));
      }),
    );

    test(
      'erf() float64 and float32 contiguous and strided',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [0.0, double.infinity, -double.infinity],
          [3],
          DType.float64,
        );
        final e = erf(a);
        expect(e.toList()[0], 0.0);
        expect(e.toList()[1], closeTo(1.0, 1e-9));
        expect(e.toList()[2], closeTo(-1.0, 1e-9));

        final a32 = NDArray.fromList([0.0, 1.0], [2], DType.float32);
        final e32 = erf(a32);
        expect(e32.dtype, DType.float32);
        expect(e32.toList()[0], 0.0);
        expect(e32.toList()[1], closeTo(0.8427007929497148, 1e-5));
      }),
    );

    test(
      'i0 asymptotic Inf / -Inf limit returns +Inf instead of NaN',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [double.infinity, -double.infinity, 0.0],
          [3],
          DType.float64,
        );
        final res = i0(a);
        expect(res.toList()[0], double.infinity);
        expect(res.toList()[1], double.infinity);
        expect(res.toList()[2], 1.0);
      }),
    );

    test(
      'sinc asymptotic Inf / -Inf limit returns 0.0',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [double.infinity, -double.infinity, 0.0],
          [3],
          DType.float64,
        );
        final res = sinc(a);
        expect(res.toList()[0], 0.0);
        expect(res.toList()[1], 0.0);
        expect(res.toList()[2], 1.0);

        final c = NDArray.fromList(
          [
            Complex(double.infinity, 1.0),
            Complex(1.0, double.infinity),
            Complex(0.0, 0.0),
          ],
          [3],
          DType.complex128,
        );
        final resC = sinc(c);
        expect(resC.getCell([0]), Complex(0.0, 0.0));
        expect(resC.getCell([1]), Complex(0.0, 0.0));
        expect(resC.getCell([2]), Complex(1.0, 0.0));
      }),
    );

    test(
      'gcd, lcm, fmod, heaviside ufuncs tests',
      () => NDArray.scope(() {
        final x1 = NDArray.fromList([12, -18, 0, 7], [4], DType.int64);
        final x2 = NDArray.fromList([8, 24, 5, 3], [4], DType.int64);
        expect(gcd(x1, x2).toList(), [4, 6, 5, 1]);
        expect(lcm(x1, x2).toList(), [24, 72, 0, 21]);

        final f1 = NDArray.fromList([5.5, -5.5], [2], DType.float64);
        final f2 = NDArray.fromList([2.0, 2.0], [2], DType.float64);
        expect(fmod(f1, f2).toList(), [
          1.5,
          -1.5,
        ]); // C-style remainder matches dividend sign

        final h1 = NDArray.fromList([-2.0, 0.0, 3.0], [3], DType.float64);
        final h2 = NDArray.fromList([0.5, 0.5, 0.5], [3], DType.float64);
        expect(heaviside(h1, h2).toList(), [0.0, 0.5, 1.0]);
      }),
    );
  });
}
