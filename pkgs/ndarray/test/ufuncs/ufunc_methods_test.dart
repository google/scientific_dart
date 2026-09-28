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
  group('BinaryOp Enum Tests', () {
    test('isReducible property returns correct values', () {
      expect(BinaryOp.add.isReducible, isTrue);
      expect(BinaryOp.multiply.isReducible, isTrue);
      expect(BinaryOp.minimum.isReducible, isTrue);
      expect(BinaryOp.maximum.isReducible, isTrue);
      expect(BinaryOp.fmin.isReducible, isTrue);
      expect(BinaryOp.fmax.isReducible, isTrue);
      expect(BinaryOp.logaddexp.isReducible, isTrue);
      expect(BinaryOp.logaddexp2.isReducible, isTrue);
      expect(BinaryOp.gcd.isReducible, isTrue);
      expect(BinaryOp.lcm.isReducible, isTrue);
      expect(BinaryOp.bitwiseAnd.isReducible, isTrue);
      expect(BinaryOp.bitwiseOr.isReducible, isTrue);
      expect(BinaryOp.bitwiseXor.isReducible, isTrue);
      expect(BinaryOp.logicalAnd.isReducible, isTrue);
      expect(BinaryOp.logicalOr.isReducible, isTrue);
      expect(BinaryOp.logicalXor.isReducible, isTrue);

      expect(BinaryOp.subtract.isReducible, isFalse);
      expect(BinaryOp.divide.isReducible, isFalse);
      expect(BinaryOp.floorDivide.isReducible, isFalse);
      expect(BinaryOp.remainder.isReducible, isFalse);
      expect(BinaryOp.fmod.isReducible, isFalse);
      expect(BinaryOp.power.isReducible, isFalse);
      expect(BinaryOp.floatPower.isReducible, isFalse);
      expect(BinaryOp.arctan2.isReducible, isFalse);
      expect(BinaryOp.hypot.isReducible, isFalse);
      expect(BinaryOp.copysign.isReducible, isFalse);
      expect(BinaryOp.leftShift.isReducible, isFalse);
      expect(BinaryOp.rightShift.isReducible, isFalse);
      expect(BinaryOp.heaviside.isReducible, isFalse);
      expect(BinaryOp.equal.isReducible, isFalse);
      expect(BinaryOp.notEqual.isReducible, isFalse);
      expect(BinaryOp.greater.isReducible, isFalse);
      expect(BinaryOp.greaterEqual.isReducible, isFalse);
      expect(BinaryOp.less.isReducible, isFalse);
      expect(BinaryOp.lessEqual.isReducible, isFalse);
    });
  });

  group('Ufunc reduce Tests', () {
    test('Global reduction on 1D and 2D arrays', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final sumRes = a.reduce(op: BinaryOp.add);
        expect(sumRes.scalar, equals(10.0));
        expect(sumRes.shape, isEmpty);

        final prodRes = a.reduce(op: BinaryOp.multiply);
        expect(prodRes.scalar, equals(24.0));

        final minRes = a.reduce(op: BinaryOp.minimum);
        expect(minRes.scalar, equals(1.0));

        final maxRes = a.reduce(op: BinaryOp.maximum);
        expect(maxRes.scalar, equals(4.0));
      });
    });

    test('Global reduction with keepdims and initial', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final res = a.reduce(op: BinaryOp.add, keepdims: true, initial: 10.0);
        expect(res.shape, equals([1, 1]));
        expect(res.getCell([0, 0]), equals(20.0));
      });
    });

    test('Axis reduction on 2D matrix', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final rowSum = a.reduce(op: BinaryOp.add, axis: 0); // Along columns
        expect(rowSum.shape, equals([3]));
        expect(rowSum.toList(), equals([5.0, 7.0, 9.0]));

        final colSum = a.reduce(op: BinaryOp.add, axis: 1); // Along rows
        expect(colSum.shape, equals([2]));
        expect(colSum.toList(), equals([6.0, 15.0]));
      });
    });

    test('Bitwise and Logical reductions', () {
      NDArray.scope(() {
        final ints = NDArray<Int64>.fromList([7, 3, 1], [3], DType.int64);
        expect(ints.reduce(op: BinaryOp.bitwiseAnd).scalar, equals(1));
        expect(ints.reduce(op: BinaryOp.bitwiseOr).scalar, equals(7));
        expect(ints.reduce(op: BinaryOp.bitwiseXor).scalar, equals(5));

        final bools = NDArray<Boolean>.fromList(
          [true, true, false],
          [3],
          DType.boolean,
        );
        expect(bools.reduce(op: BinaryOp.logicalAnd).scalar, isFalse);
        expect(bools.reduce(op: BinaryOp.logicalOr).scalar, isTrue);
      });
    });

    test('Out buffer recycling', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final out = NDArray<Float64>.zeros([2], DType.float64);
        final res = a.reduce(op: BinaryOp.add, axis: 0, out: out);
        expect(identical(res, out), isTrue);
        expect(out.toList(), equals([4.0, 6.0]));
      });
    });

    test('Non-reducible op throws ArgumentError', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => a.reduce(op: BinaryOp.subtract), throwsArgumentError);
      });
    });
  });

  group('Ufunc accumulate Tests', () {
    test('Accumulate 1D array add, multiply, min, max', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final cumsum = a.accumulate(op: BinaryOp.add);
        expect(cumsum.toList(), equals([1.0, 3.0, 6.0, 10.0]));

        final cumprod = a.accumulate(op: BinaryOp.multiply);
        expect(cumprod.toList(), equals([1.0, 2.0, 6.0, 24.0]));

        final b = NDArray<Float64>.fromList(
          [3.0, 1.0, 4.0, 2.0],
          [4],
          DType.float64,
        );
        expect(
          b.accumulate(op: BinaryOp.minimum).toList(),
          equals([3.0, 1.0, 1.0, 1.0]),
        );
        expect(
          b.accumulate(op: BinaryOp.maximum).toList(),
          equals([3.0, 3.0, 4.0, 4.0]),
        );
      });
    });

    test('Accumulate 2D array along axis', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final cumsum0 = a.accumulate(op: BinaryOp.add, axis: 0);
        expect(cumsum0.shape, equals([2, 3]));
        expect(cumsum0.toList(), equals([1.0, 2.0, 3.0, 5.0, 7.0, 9.0]));

        final cumsum1 = a.accumulate(op: BinaryOp.add, axis: 1);
        expect(cumsum1.toList(), equals([1.0, 3.0, 6.0, 4.0, 9.0, 15.0]));
      });
    });
  });

  group('Ufunc reduceat Tests', () {
    test('reduceat 1D array interval slices', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
          [8],
          DType.float64,
        );
        final indices = NDArray.fromList([0, 4, 1, 5], [4], DType.int64);
        // Step 0: slice [0:4] -> sum(0..3) = 6.0
        // Step 1: slice [4:1] (start >= end) -> element a[4] = 4.0
        // Step 2: slice [1:5] -> sum(1..4) = 10.0
        // Step 3: slice [5:8] (final) -> sum(5..7) = 18.0
        final res = a.reduceat(indices, op: BinaryOp.add);
        expect(res.toList(), equals([6.0, 4.0, 10.0, 18.0]));
      });
    });
  });

  group('Ufunc outer Tests', () {
    test('Outer product of 1D vectors', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList([4.0, 5.0], [2], DType.float64);
        final res = a.outer(b, op: BinaryOp.multiply);
        expect(res.shape, equals([3, 2]));
        expect(res.toList(), equals([4.0, 5.0, 8.0, 10.0, 12.0, 15.0]));
      });
    });

    test('Outer addition and subtraction with out buffer', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList([10.0, 20.0], [2], DType.float64);
        final b = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        final out = NDArray<Float64>.zeros([2, 2], DType.float64);
        final res = a.outer(b, op: BinaryOp.add, out: out);
        expect(identical(res, out), isTrue);
        expect(out.toList(), equals([11.0, 12.0, 21.0, 22.0]));
      });
    });
  });

  group('Ufunc at Tests', () {
    test('at unbuffered scatter addition with duplicate indices', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 0.0, 0.0],
          [3],
          DType.float64,
        );
        final indices = NDArray.fromList([0, 1, 0, 1, 0], [5], DType.int64);
        final b = NDArray<Float64>.fromList(
          [1.0, 10.0, 2.0, 20.0, 3.0],
          [5],
          DType.float64,
        );

        a.at(indices, b, op: BinaryOp.add);
        // index 0 gets 1.0 + 2.0 + 3.0 = 6.0
        // index 1 gets 10.0 + 20.0 = 30.0
        // index 2 untouched = 0.0
        expect(a.toList(), equals([6.0, 30.0, 0.0]));
      });
    });

    test('at scatter multiplication on integer array', () {
      NDArray.scope(() {
        final a = NDArray<Int64>.fromList([1, 1, 1], [3], DType.int64);
        final indices = NDArray.fromList([0, 0, 1], [3], DType.int64);
        final b = NDArray<Int64>.fromList([2, 3, 5], [3], DType.int64);

        a.at(indices, b, op: BinaryOp.multiply);
        // index 0 gets 1 * 2 * 3 = 6
        // index 1 gets 1 * 5 = 5
        expect(a.toList(), equals([6, 5, 1]));
      });
    });
  });

  group('Error Handling & Edge Cases', () {
    test('Disposed array throws StateError for all ufunc methods', () {
      final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
      final indices = NDArray.fromList([0], [1], DType.int64);
      final b = NDArray<Float64>.fromList([5.0], [1], DType.float64);
      a.dispose();

      expect(() => a.reduce(op: BinaryOp.add), throwsStateError);
      expect(() => a.accumulate(op: BinaryOp.add), throwsStateError);
      expect(() => a.reduceat(indices, op: BinaryOp.add), throwsStateError);
      expect(() => a.outer(b), throwsStateError);
      expect(() => a.at(indices, b, op: BinaryOp.add), throwsStateError);

      indices.dispose();
      b.dispose();
    });

    test('Invalid axis throws RangeError', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => a.reduce(op: BinaryOp.add, axis: 5), throwsRangeError);
        expect(
          () => a.accumulate(op: BinaryOp.add, axis: -5),
          throwsRangeError,
        );
        expect(
          () => a.reduceat(
            NDArray.fromList([0], [1], DType.int64),
            op: BinaryOp.add,
            axis: 2,
          ),
          throwsRangeError,
        );
      });
    });

    test('Empty array reduction without initial throws ArgumentError', () {
      NDArray.scope(() {
        final empty = NDArray<Float64>.zeros([0], DType.float64);
        expect(() => empty.reduce(op: BinaryOp.add), throwsArgumentError);
        final initRes = empty.reduce(op: BinaryOp.add, initial: 42.0);
        expect(initRes.scalar, equals(42.0));
      });
    });

    test('Complex number product reduction and accumulation', () {
      NDArray.scope(() {
        final c = NDArray<Complex128>.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );

        final prodRes = c.reduce(op: BinaryOp.multiply);
        // (1 + 2i) * (3 + 4i) = 3 + 4i + 6i - 8 = -5 + 10i
        expect(prodRes.scalar, equals(Complex(-5.0, 10.0)));

        final cumsum = c.accumulate(op: BinaryOp.add);
        expect(cumsum.getCell([0]), equals(Complex(1.0, 2.0)));
        expect(cumsum.getCell([1]), equals(Complex(4.0, 6.0)));
      });
    });

    test(
      'outer operations with non-reducible ops (subtract, divide, copysign)',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList([10.0, 20.0], [2], DType.float64);
          final b = NDArray<Float64>.fromList([2.0, 4.0], [2], DType.float64);

          final subOuter = a.outer(b, op: BinaryOp.subtract);
          expect(subOuter.toList(), equals([8.0, 6.0, 18.0, 16.0]));

          final divOuter = a.outer(b, op: BinaryOp.divide);
          expect(divOuter.toList(), equals([5.0, 2.5, 10.0, 5.0]));
        });
      },
    );
  });
  group('Top-Level Ufunc Function Tests', () {
    test('top-level reduce, accumulate, reduceat, outer, at functions', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final sumRes = reduce(a, op: BinaryOp.add);
        expect(sumRes.scalar, equals(10.0));

        final cumsum = accumulate(a, op: BinaryOp.add);
        expect(cumsum.toList(), equals([1.0, 3.0, 6.0, 10.0]));

        final indices = NDArray.fromList([0, 2], [2], DType.int64);
        final redAt = reduceat(a, indices, op: BinaryOp.add);
        expect(redAt.toList(), equals([3.0, 7.0]));

        final b = NDArray<Float64>.fromList([5.0, 6.0], [2], DType.float64);
        final outRes = a.outer(b, op: BinaryOp.multiply);
        expect(outRes.shape, equals([4, 2]));

        final target = NDArray<Float64>.zeros([3], DType.float64);
        final atIndices = NDArray.fromList([0, 0, 1], [3], DType.int64);
        final valBuffer = NDArray<Float64>.fromList(
          [2.0, 3.0, 5.0],
          [3],
          DType.float64,
        );
        at(target, atIndices, valBuffer, op: BinaryOp.add);
        expect(target.toList(), equals([5.0, 5.0, 0.0]));
      });
    });
  });
}
