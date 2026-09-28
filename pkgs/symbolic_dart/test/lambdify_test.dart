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
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  group('SymbolicLambda vectorized evaluator', () {
    test('scalar evaluation', () {
      final x = Symbol('x');
      final y = Symbol('y');
      final f = (x ^ 2) + (y * 3);
      final lambda = f.lambdify([x, y]);

      expect(lambda.callScalar([2, 4]), closeTo(16.0, 1e-12));
    });

    test('vectorized evaluation over 1D NDArray', () {
      final x = Symbol('x');
      final f = (x ^ 2) + 1;
      final lambda = f.lambdify([x]);

      final arr = NDArray.fromList([0.0, 1.0, 2.0, 3.0], [4], DType.float64);
      final out = lambda.callArray([arr]);

      expect(out.shape, [4]);
      expect(out.getCell([0]), closeTo(1.0, 1e-12));
      expect(out.getCell([1]), closeTo(2.0, 1e-12));
      expect(out.getCell([2]), closeTo(5.0, 1e-12));
      expect(out.getCell([3]), closeTo(10.0, 1e-12));
    });

    test('broadcasting multiple 2D NDArrays', () {
      final x = Symbol('x');
      final y = Symbol('y');
      final f = x * y;
      final lambda = f.lambdify([x, y]);

      final xArr = NDArray.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final yArr = NDArray.fromList([10.0, 20.0], [1, 2], DType.float64);

      final out = lambda.callArray([xArr, yArr]);
      expect(out.shape, [2, 2]);
      expect(out.getCell([0, 0]), closeTo(10.0, 1e-12));
      expect(out.getCell([0, 1]), closeTo(40.0, 1e-12));
      expect(out.getCell([1, 0]), closeTo(30.0, 1e-12));
      expect(out.getCell([1, 1]), closeTo(80.0, 1e-12));
    });
  });
}
