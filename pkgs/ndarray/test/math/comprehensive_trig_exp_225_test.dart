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
  group('Comprehensive Trig & Exp 225 Matrix Cross-DType Suite', () {
    final all15DTypes = [
      DType.float64,
      DType.float32,
      DType.float16,
      DType.bfloat16,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
      DType.complex128,
      DType.complex64,
      DType.boolean,
    ];

    NDArray<AnySpec> makeArr(DType dt, List<int> shape, {int seed = 2}) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<Object>.generate(size, (i) {
        final val = ((i + seed) % 5) + 2;
        if (dt == DType.boolean) return true;
        if (dt == DType.complex128 || dt == DType.complex64) {
          return Complex(val.toDouble(), 1.0);
        }
        return val;
      });

      return NDArray.fromList(rawList, shape, (dt as DType<AnySpec>));
    }

    test(
      'All Cross-DType Pairs for arctan2, hypot, logaddexp, logaddexp2 across Contiguous, Transposed & Broadcast',
      () {
        NDArray.scope(() {
          final mask = NDArray<Boolean>.fromList(
            [true, false, true, false, true, false],
            [2, 3],
            DType.boolean,
          );

          for (final dtA in all15DTypes) {
            for (final dtB in all15DTypes) {
              // Contiguous
              final aContig = makeArr(dtA, [2, 3], seed: 1);
              final bContig = makeArr(dtB, [2, 3], seed: 2);

              // Transposed non-contiguous
              final aBase = makeArr(dtA, [3, 2], seed: 1);
              final bBase = makeArr(dtB, [3, 2], seed: 2);
              final aTrans = aBase.transpose();
              final bTrans = bBase.transpose();

              // Broadcast
              final bBcast = makeArr(dtB, [1, 3], seed: 3);

              final rHypot1 = hypot(aContig, bContig);
              expect(rHypot1.shape, [2, 3]);

              final rHypot2 = hypot(aTrans, bTrans, where: mask);
              expect(rHypot2.shape, [2, 3]);

              final rHypot3 = hypot(aContig, bBcast);
              expect(rHypot3.shape, [2, 3]);

              if (!dtA.isComplex && !dtB.isComplex) {
                final rLog1 = logaddexp(aContig, bContig);
                expect(rLog1.shape, [2, 3]);

                final rLog21 = logaddexp2(aContig, bContig);
                expect(rLog21.shape, [2, 3]);

                final rLog2 = logaddexp(aTrans, bTrans, where: mask);
                expect(rLog2.shape, [2, 3]);

                final rLog22 = logaddexp2(aTrans, bTrans, where: mask);
                expect(rLog22.shape, [2, 3]);

                final rLog3 = logaddexp(aContig, bBcast);
                expect(rLog3.shape, [2, 3]);

                final rLog23 = logaddexp2(aContig, bBcast);
                expect(rLog23.shape, [2, 3]);

                final rAtan21 = atan2(aContig, bContig);
                expect(rAtan21.shape, [2, 3]);

                final rAtan22 = atan2(aTrans, bTrans, where: mask);
                expect(rAtan22.shape, [2, 3]);

                final rAtan23 = atan2(aContig, bBcast);
                expect(rAtan23.shape, [2, 3]);
              }
            }
          }
        });
      },
    );

    test(
      'All Unary Transcendental Functions across Contiguous & Transposed Views',
      () {
        NDArray.scope(() {
          final mask = NDArray<Boolean>.fromList(
            [true, false, true, false, true, false],
            [2, 3],
            DType.boolean,
          );

          for (final dt in all15DTypes) {
            final aContig = makeArr(dt, [2, 3]);
            final aBase = makeArr(dt, [3, 2]);
            final aTrans = aBase.transpose();

            for (final arr in [aContig, aTrans]) {
              final s = sin(arr, where: mask);
              expect(s.shape, [2, 3]);

              final c = cos(arr, where: mask);
              expect(c.shape, [2, 3]);

              final t = tan(arr, where: mask);
              expect(t.shape, [2, 3]);

              final sh = sinh(arr, where: mask);
              expect(sh.shape, [2, 3]);

              final ch = cosh(arr, where: mask);
              expect(ch.shape, [2, 3]);

              final th = tanh(arr, where: mask);
              expect(th.shape, [2, 3]);

              final ex = exp(arr, where: mask);
              expect(ex.shape, [2, 3]);

              final exm = expm1(arr, where: mask);
              expect(exm.shape, [2, 3]);

              final lg = log(arr, where: mask);
              expect(lg.shape, [2, 3]);

              final lg2 = log2(arr, where: mask);
              expect(lg2.shape, [2, 3]);

              final lg10 = log10(arr, where: mask);
              expect(lg10.shape, [2, 3]);

              final lg1 = log1p(arr, where: mask);
              expect(lg1.shape, [2, 3]);

              if (dt != DType.complex128 && dt != DType.complex64) {
                final asin_ = asin(arr, where: mask);
                expect(asin_.shape, [2, 3]);

                final acos_ = acos(arr, where: mask);
                expect(acos_.shape, [2, 3]);

                final atan_ = atan(arr, where: mask);
                expect(atan_.shape, [2, 3]);

                final asinh_ = asinh(arr, where: mask);
                expect(asinh_.shape, [2, 3]);

                final acosh_ = acosh(arr, where: mask);
                expect(acosh_.shape, [2, 3]);

                final atanh_ = atanh(arr, where: mask);
                expect(atanh_.shape, [2, 3]);

                final d2r = deg2rad(arr, where: mask);
                expect(d2r.shape, [2, 3]);

                final r2d = rad2deg(arr, where: mask);
                expect(r2d.shape, [2, 3]);
              }
            }
          }
        });
      },
    );
  });
}
