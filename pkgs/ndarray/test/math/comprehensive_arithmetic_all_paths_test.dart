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
  group('Comprehensive Arithmetic All Paths Suite', () {
    final allDTypes = [
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
    ];

    test('4D Array Arithmetic across DTypes (Contiguous & Strided with out & where)', () {
      NDArray.scope(() {
        for (final dt in allDTypes) {
          final a = NDArray.fromList(
            List.generate(16, (i) {
              final v = (i % 5) + 2;
              if (dt == DType.complex128 || dt == DType.complex64) {
                return Complex(v.toDouble(), 1.0);
              }
              return v;
            }),
            [2, 2, 2, 2],
            dt,
          );

          final b = NDArray.fromList(
            List.generate(16, (i) {
              final v = (i % 3) + 2;
              if (dt == DType.complex128 || dt == DType.complex64) {
                return Complex(v.toDouble(), 1.0);
              }
              return v;
            }),
            [2, 2, 2, 2],
            dt,
          );

          final mask = NDArray<bool>.fromList(
            List.generate(16, (i) => i % 2 == 0),
            [2, 2, 2, 2],
            DType.boolean,
          );

          final rAdd = add(a, b, where: mask);
          expect(rAdd.shape, [2, 2, 2, 2]);

          final rSub = subtract(a, b, where: mask);
          expect(rSub.shape, [2, 2, 2, 2]);

          final rMul = multiply(a, b, where: mask);
          expect(rMul.shape, [2, 2, 2, 2]);

          final rDiv = divide(a, b, where: mask);
          expect(rDiv.shape, [2, 2, 2, 2]);

          final outArr = NDArray.create([2, 2, 2, 2], rAdd.dtype);
          add(a, b, out: outArr, where: mask);
          expect(outArr.shape, [2, 2, 2, 2]);
        }
      });
    });

    test('Scalar broadcasting with 0-D scalar arrays across all DTypes', () {
      NDArray.scope(() {
        for (final dt in allDTypes) {
          final val = (dt == DType.complex128 || dt == DType.complex64)
              ? Complex(3.0, 1.0)
              : 3;
          final scalarArr = NDArray.scalar(val, dtype: dt);
          final arr2d = NDArray.fromList(
            List.generate(6, (i) {
              final v = (i % 4) + 2;
              if (dt == DType.complex128 || dt == DType.complex64) {
                return Complex(v.toDouble(), 0.5);
              }
              return v;
            }),
            [2, 3],
            dt,
          );

          final rAdd = add(arr2d, scalarArr);
          expect(rAdd.shape, [2, 3]);

          final rAddRev = add(scalarArr, arr2d);
          expect(rAddRev.shape, [2, 3]);

          final rMul = multiply(arr2d, scalarArr);
          expect(rMul.shape, [2, 3]);

          final rSub = subtract(arr2d, scalarArr);
          expect(rSub.shape, [2, 3]);

          final rDiv = divide(arr2d, scalarArr);
          expect(rDiv.shape, [2, 3]);
        }
      });
    });

    test('Operator overloads (+, -, *, /, ~/, %) on NDArray', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final b = NDArray.fromList([2.0, 4.0, 5.0, 8.0], [2, 2], DType.float64);

        final addOp = a + b;
        expect(addOp.shape, [2, 2]);

        final subOp = a - b;
        expect(subOp.shape, [2, 2]);

        final mulOp = a * b;
        expect(mulOp.shape, [2, 2]);

        final divOp = a / b;
        expect(divOp.shape, [2, 2]);

        final floorDivOp = a ~/ b;
        expect(floorDivOp.shape, [2, 2]);

        final modOp = a % b;
        expect(modOp.shape, [2, 2]);

        final negOp = -a;
        expect(negOp.shape, [2, 2]);
      });
    });

    test('StateError and ArgumentError exception paths on disposed inputs and incompatible shapes', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

        // Incompatible shape
        expect(() => add(a, b), throwsArgumentError);
        expect(() => subtract(a, b), throwsArgumentError);
        expect(() => multiply(a, b), throwsArgumentError);
        expect(() => divide(a, b), throwsArgumentError);

        final outBad = NDArray.create([4], DType.float64);
        expect(() => add(a, a, out: outBad), throwsArgumentError);

        // Disposed array
        final disposedArr = NDArray.fromList([1.0], [1], DType.float64);
        disposedArr.dispose();

        expect(() => add(disposedArr, a), throwsStateError);
        expect(() => add(a, disposedArr), throwsStateError);
        expect(() => positive(disposedArr), throwsStateError);
        expect(() => negative(disposedArr), throwsStateError);
        expect(() => abs(disposedArr), throwsStateError);
        expect(() => sqrt(disposedArr), throwsStateError);
      });
    });
  });
}
