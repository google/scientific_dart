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
  group('P1-2: minimum, maximum, fmin, fmax', () {
    test('signed zero handling for float64 and float32', () {
      final a = NDArray.fromList([-0.0, 0.0, -0.0, 0.0], [4], DType.float64);
      final b = NDArray.fromList([0.0, -0.0, -0.0, 0.0], [4], DType.float64);

      final minRes = binaryUfunc(a, b, op: BinaryOp.minimum);
      expect(minRes.toList(), [0.0, 0.0, 0.0, 0.0]);
      // Verify sign bit: minimum of -0.0 and 0.0 must be -0.0
      expect((minRes.getCell([0]) as double).isNegative, isTrue);
      expect((minRes.getCell([1]) as double).isNegative, isTrue);
      expect((minRes.getCell([2]) as double).isNegative, isTrue);
      expect((minRes.getCell([3]) as double).isNegative, isFalse);

      final maxRes = binaryUfunc(a, b, op: BinaryOp.maximum);
      expect((maxRes.getCell([0]) as double).isNegative, isFalse);
      expect((maxRes.getCell([1]) as double).isNegative, isFalse);
      expect((maxRes.getCell([2]) as double).isNegative, isTrue);
      expect((maxRes.getCell([3]) as double).isNegative, isFalse);

      final fminRes = binaryUfunc(a, b, op: BinaryOp.fmin);
      expect((fminRes.getCell([0]) as double).isNegative, isTrue);
      expect((fminRes.getCell([1]) as double).isNegative, isTrue);

      final fmaxRes = binaryUfunc(a, b, op: BinaryOp.fmax);
      expect((fmaxRes.getCell([0]) as double).isNegative, isFalse);
      expect((fmaxRes.getCell([1]) as double).isNegative, isFalse);
    });

    test('NaN propagation: minimum/maximum propagate, fmin/fmax ignore', () {
      final a = NDArray.fromList(
        [double.nan, 2.0, double.nan, 10.0],
        [4],
        DType.float64,
      );
      final b = NDArray.fromList(
        [1.0, double.nan, double.nan, 5.0],
        [4],
        DType.float64,
      );

      final minRes = binaryUfunc(a, b, op: BinaryOp.minimum);
      expect((minRes.getCell([0]) as double).isNaN, isTrue);
      expect((minRes.getCell([1]) as double).isNaN, isTrue);
      expect((minRes.getCell([2]) as double).isNaN, isTrue);
      expect(minRes.getCell([3]), equals(5.0));

      final maxRes = binaryUfunc(a, b, op: BinaryOp.maximum);
      expect((maxRes.getCell([0]) as double).isNaN, isTrue);
      expect((maxRes.getCell([1]) as double).isNaN, isTrue);
      expect((maxRes.getCell([2]) as double).isNaN, isTrue);
      expect(maxRes.getCell([3]), equals(10.0));

      final fminRes = binaryUfunc(a, b, op: BinaryOp.fmin);
      expect(fminRes.getCell([0]), equals(1.0));
      expect(fminRes.getCell([1]), equals(2.0));
      expect((fminRes.getCell([2]) as double).isNaN, isTrue);
      expect(fminRes.getCell([3]), equals(5.0));

      final fmaxRes = binaryUfunc(a, b, op: BinaryOp.fmax);
      expect(fmaxRes.getCell([0]), equals(1.0));
      expect(fmaxRes.getCell([1]), equals(2.0));
      expect((fmaxRes.getCell([2]) as double).isNaN, isTrue);
      expect(fmaxRes.getCell([3]), equals(10.0));
    });

    test('complex lexicographical comparison', () {
      final a = NDArray.fromList(
        [Complex(1.0, 2.0), Complex(1.0, 5.0), Complex(3.0, 0.0)],
        [3],
        DType.complex128,
      );
      final b = NDArray.fromList(
        [Complex(2.0, 0.0), Complex(1.0, 3.0), Complex(3.0, 1.0)],
        [3],
        DType.complex128,
      );

      final minRes = binaryUfunc(a, b, op: BinaryOp.minimum);
      expect(minRes.getCell([0]), equals(Complex(1.0, 2.0)));
      expect(minRes.getCell([1]), equals(Complex(1.0, 3.0)));
      expect(minRes.getCell([2]), equals(Complex(3.0, 0.0)));

      final maxRes = binaryUfunc(a, b, op: BinaryOp.maximum);
      expect(maxRes.getCell([0]), equals(Complex(2.0, 0.0)));
      expect(maxRes.getCell([1]), equals(Complex(1.0, 5.0)));
      expect(maxRes.getCell([2]), equals(Complex(3.0, 1.0)));
    });

    test('strided and broadcasting minimum/maximum', () {
      final a = NDArray.fromList([1, 5, 3, 7], [2, 2], DType.int32);
      final b = NDArray.fromList([4, 2], [1, 2], DType.int32);

      final minRes = binaryUfunc(a, b, op: BinaryOp.minimum);
      expect(minRes.toList(), [1, 2, 3, 2]);

      final maxRes = binaryUfunc(a, b, op: BinaryOp.maximum);
      expect(maxRes.toList(), [4, 5, 4, 7]);
    });

    test('where-mask fallback works correctly', () {
      final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
      final b = NDArray.fromList([5.0, 25.0, 15.0], [3], DType.float64);
      final out = NDArray.fromList([0.0, 0.0, 0.0], [3], DType.float64);
      final where = NDArray.fromList([true, false, true], [3], DType.boolean);

      binaryUfunc(a, b, op: BinaryOp.minimum, where: where, out: out);
      expect(out.toList(), [5.0, 0.0, 15.0]);
    });
  });

  group('P1-3: accumulateUfunc and outerUfunc cleanup', () {
    test('accumulateUfunc cumsum and cumprod for int16 and uint8', () {
      final a16 = NDArray.fromList([1, 2, 3, 4], [4], DType.int16);
      final cs16 = accumulateUfunc(a16, op: BinaryOp.add);
      expect(cs16.toList(), [1, 3, 6, 10]);

      final cp16 = accumulateUfunc(a16, op: BinaryOp.multiply);
      expect(cp16.toList(), [1, 2, 6, 24]);

      final u8 = NDArray.fromList([2, 3, 4, 5], [4], DType.uint8);
      final cs8 = accumulateUfunc(u8, op: BinaryOp.add);
      expect(cs8.toList(), [2, 5, 9, 14]);

      final cp8 = accumulateUfunc(u8, op: BinaryOp.multiply);
      expect(cp8.toList(), [2, 6, 24, 120]);
    });

    test('outerUfunc delegates to binaryUfunc for multiply and add', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final b = NDArray.fromList([10.0, 20.0], [2], DType.float64);

      final mulOuter = outerUfunc(a, b, op: BinaryOp.multiply);
      expect(mulOuter.shape, [3, 2]);
      expect(mulOuter.toList(), [10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);

      final addOuter = outerUfunc(a, b, op: BinaryOp.add);
      expect(addOuter.shape, [3, 2]);
      expect(addOuter.toList(), [11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);

      final minOuter = outerUfunc(a, b, op: BinaryOp.minimum);
      expect(minOuter.shape, [3, 2]);
      expect(minOuter.toList(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    });
  });
}
