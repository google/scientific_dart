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
  group('Comprehensive Arithmetic 225 Matrix Cross-DType Suite', () {
    final coreDTypes = [
      DType.float64,
      DType.float32,
      DType.complex128,
      DType.complex64,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.uint8,
      DType.float16,
      DType.bfloat16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.boolean,
    ];

    NDArray<AnySpec> makeArr(DType dt, List<int> shape, {int seed = 2}) {
      final size = shape.reduce((a, b) => a * b);
      final rawList = List<Object>.generate(size, (i) {
        final val = ((i + seed) % 5) + 2; // avoid 0/1 division
        if (dt == DType.boolean) return true;
        if (dt == DType.complex128 || dt == DType.complex64) {
          return Complex(val.toDouble(), 1.0);
        }
        return val;
      });

      return NDArray.fromList(rawList, shape, (dt as DType<AnySpec>));
    }

    test(
      'All Cross-DType Pairs for add, subtract, multiply, divide across Contiguous, Transposed & Broadcast',
      () {
        NDArray.scope(() {
          final mask = NDArray<Boolean>.fromList(
            [true, false, true, false, true, false],
            [2, 3],
            DType.boolean,
          );

          for (final dtA in coreDTypes) {
            for (final dtB in coreDTypes) {
              // Mode 1: Contiguous same shape [2, 3] -> hits v_*
              final aContig = makeArr(dtA, [2, 3], seed: 1);
              final bContig = makeArr(dtB, [2, 3], seed: 2);

              if (dtA != dtB) {
                expect(() => add(aContig, bContig), throwsArgumentError);
                expect(() => subtract(aContig, bContig), throwsArgumentError);
                expect(() => multiply(aContig, bContig), throwsArgumentError);
                expect(() => divide(aContig, bContig), throwsArgumentError);
                continue;
              }

              final rAdd1 = add(aContig, bContig);
              expect(rAdd1.shape, [2, 3]);

              final rSub1 = subtract(aContig, bContig);
              expect(rSub1.shape, [2, 3]);

              final rMul1 = multiply(aContig, bContig);
              expect(rMul1.shape, [2, 3]);

              final rDiv1 = divide(aContig, bContig);
              expect(rDiv1.shape, [2, 3]);

              // Mode 2: Non-contiguous transposed view -> hits s_*
              final aBase = makeArr(dtA, [3, 2], seed: 1);
              final bBase = makeArr(dtB, [3, 2], seed: 2);
              final aTrans = aBase.transpose();
              final bTrans = bBase.transpose();

              final rAdd2 = add(aTrans, bTrans, where: mask);
              expect(rAdd2.shape, [2, 3]);

              final rSub2 = subtract(aTrans, bTrans, where: mask);
              expect(rSub2.shape, [2, 3]);

              final rMul2 = multiply(aTrans, bTrans, where: mask);
              expect(rMul2.shape, [2, 3]);

              final rDiv2 = divide(aTrans, bTrans, where: mask);
              expect(rDiv2.shape, [2, 3]);

              // Mode 3: Broadcasting [2, 3] + [1, 3] -> hits s_* broadcasting
              final bBcast = makeArr(dtB, [1, 3], seed: 3);

              final rAdd3 = add(aContig, bBcast);
              expect(rAdd3.shape, [2, 3]);

              final rSub3 = subtract(aContig, bBcast);
              expect(rSub3.shape, [2, 3]);

              final rMul3 = multiply(aContig, bBcast);
              expect(rMul3.shape, [2, 3]);

              final rDiv3 = divide(aContig, bBcast);
              expect(rDiv3.shape, [2, 3]);
            }
          }
        });
      },
    );

    test('All Non-complex Pairs for floorDivide, remainder, fmod, power', () {
      NDArray.scope(() {
        final mask = NDArray<Boolean>.fromList(
          [true, false, true, false, true, false],
          [2, 3],
          DType.boolean,
        );

        for (final dtA in coreDTypes) {
          for (final dtB in coreDTypes) {
            if (dtA.isComplex || dtB.isComplex) continue;

            final aContig = makeArr(dtA, [2, 3], seed: 1);
            final bContig = makeArr(dtB, [2, 3], seed: 2);

            if (dtA != dtB) {
              expect(() => floorDivide(aContig, bContig), throwsArgumentError);
              expect(() => remainder(aContig, bContig), throwsArgumentError);
              expect(() => fmod(aContig, bContig), throwsArgumentError);
              expect(() => power(aContig, bContig), throwsArgumentError);
              continue;
            }

            final aBase = makeArr(dtA, [3, 2], seed: 1);
            final bBase = makeArr(dtB, [3, 2], seed: 2);
            final aTrans = aBase.transpose();
            final bTrans = bBase.transpose();

            // Contiguous & Transposed floorDivide
            final rFloor1 = floorDivide(aContig, bContig, where: mask);
            expect(rFloor1.shape, [2, 3]);

            final rFloor2 = floorDivide(aTrans, bTrans);
            expect(rFloor2.shape, [2, 3]);

            // Contiguous & Transposed remainder
            final rRem1 = remainder(aContig, bContig, where: mask);
            expect(rRem1.shape, [2, 3]);

            final rRem2 = remainder(aTrans, bTrans);
            expect(rRem2.shape, [2, 3]);

            // Contiguous & Transposed fmod
            final rFmod1 = fmod(aContig, bContig, where: mask);
            expect(rFmod1.shape, [2, 3]);

            final rFmod2 = fmod(aTrans, bTrans);
            expect(rFmod2.shape, [2, 3]);

            // Contiguous & Transposed power
            if (dtA == dtB && dtA != DType.boolean) {
              final rPow1 = power(aContig, bContig, where: mask);
              expect(rPow1.shape, [2, 3]);

              final rPow2 = power(aTrans, bTrans);
              expect(rPow2.shape, [2, 3]);
            }
          }
        }
      });
    });
  });
}
