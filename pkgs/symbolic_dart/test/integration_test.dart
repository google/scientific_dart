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

import 'package:ndarray/ndarray.dart' hide exp, sin;
import 'package:symbolic_dart/symbolic_dart.dart';
import 'package:test/test.dart';

void main() {
  group('SymbolicMatrix integration with NDArray', () {
    test('creation, properties, and toNDArray', () {
      final mat = SymbolicMatrix.fromList([
        [1, Symbol('x')],
        [Real(2.5), 4],
      ]);
      expect(mat.rows, 2);
      expect(mat.cols, 2);
      expect(mat.shape, (rows: 2, cols: 2));

      final subMat = mat.subs({Symbol('x'): 10.0});
      final arr = subMat.toNDArray();
      expect(arr.shape, [2, 2]);
      expect(arr.getCell([0, 0]), (1.0));
      expect(arr.getCell([0, 1]), (10.0));
      expect(arr.getCell([1, 0]), (2.5));
      expect(arr.getCell([1, 1]), (4.0));
    });

    test('matrix algebra: det, inv, solve', () {
      // Square matrix A = [[2, 1], [1, 3]]
      final matA = SymbolicMatrix.fromList([
        [2, 1],
        [1, 3],
      ]);
      final detA = matA.det().asDouble;
      expect(detA, closeTo(5.0, 1e-12)); // 2*3 - 1*1 = 5

      // Inverse A^-1
      final aInv = matA.inv();
      final matI = (matA * aInv).toNDArray();
      expect(matI.getCell([0, 0]), closeTo(1.0, 1e-12));
      expect(matI.getCell([0, 1]), closeTo(0.0, 1e-12));
      expect(matI.getCell([1, 0]), closeTo(0.0, 1e-12));
      expect(matI.getCell([1, 1]), closeTo(1.0, 1e-12));

      // Solve A * x = b where b = [[5], [5]]
      final b = SymbolicMatrix.fromVector([5, 5]);
      final x = matA.solve(b); // x1 = 2, x2 = 1
      expect(x.getCell(0, 0).asDouble, closeTo(2.0, 1e-12));
      expect(x.getCell(1, 0).asDouble, closeTo(1.0, 1e-12));
    });

    test('analytical Jacobian matrix generation', () {
      final x = Symbol('x');
      final y = Symbol('y');
      // f1(x, y) = x^2 + y
      // f2(x, y) = 2*x*y
      final fVec = SymbolicMatrix.fromVector([
        (x ^ 2) + y,
        (Integer(2) * x) * y,
      ]);

      final J = fVec.jacobian([x, y]);
      expect(J.rows, 2);
      expect(J.cols, 2);

      // J[0, 0] = df1/dx = 2*x
      // J[0, 1] = df1/dy = 1
      // J[1, 0] = df2/dx = 2*y
      // J[1, 1] = df2/dy = 2*x
      final jAt = J.subs({x: 3.0, y: 4.0});
      expect(jAt.getCell(0, 0).asDouble, closeTo(6.0, 1e-12)); // 2*3
      expect(jAt.getCell(0, 1).asDouble, closeTo(1.0, 1e-12)); // 1
      expect(jAt.getCell(1, 0).asDouble, closeTo(8.0, 1e-12)); // 2*4
      expect(jAt.getCell(1, 1).asDouble, closeTo(6.0, 1e-12)); // 2*3
    });
  });

  group('NDArray symbolic mapSymbolic & evaluateSymbolic', () {
    test('mapSymbolic element-wise evaluation', () {
      final x = Symbol('x');
      final arr = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
      // f(x) = x^2 + 3
      final res = arr.mapSymbolic((x ^ 2) + Integer(3), x);
      expect(res.getCell([0]), closeTo(3.0, 1e-12));
      expect(res.getCell([1]), closeTo(4.0, 1e-12));
      expect(res.getCell([2]), closeTo(7.0, 1e-12));
    });

    test('evaluateSymbolic with named inputs map', () {
      final x = Symbol('x');
      final y = Symbol('y');
      final arrX = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final arrY = NDArray.fromList([10.0, 20.0], [2], DType.float64);

      final res = evaluateSymbolic(
        (x * Integer(2)) + y,
        inputs: {x: arrX, y: arrY},
      );
      expect(res.getCell([0]), closeTo(12.0, 1e-12));
      expect(res.getCell([1]), closeTo(24.0, 1e-12));
    });
  });

  group('SymbolicOptimizer with exact analytical derivatives', () {
    test('solveNewtonRaphson non-linear equation system', () {
      final x = Symbol('x');
      final y = Symbol('y');
      // Circle intersection:
      // f1(x, y) = x^2 + y^2 - 25 = 0
      // f2(x, y) = y - x - 1 = 0
      // Root near (3, 4) since 3^2 + 4^2 = 25 and 4 - 3 - 1 = 0.
      final x0 = NDArray.fromList([2.5, 3.5], [2], DType.float64);

      final result = SymbolicOptimizer.solveNewtonRaphson(
        equations: [(x ^ 2) + (y ^ 2) - Integer(25), y - x - Integer(1)],
        variables: [x, y],
        x0: x0,
      );

      final solX = (result.solution.getCell([0]) as num).toDouble();
      final solY = (result.solution.getCell([1]) as num).toDouble();
      expect(solX, closeTo(3.0, 1e-8));
      expect(solY, closeTo(4.0, 1e-8));
      expect(result.residual, lessThan(1e-8));
    });

    test('minimizeGradientDescent quadratic bowl', () {
      final x = Symbol('x');
      final y = Symbol('y');
      // L(x, y) = (x - 3)^2 + (y + 2)^2
      final loss = ((x - Integer(3)) ^ 2) + ((y + Integer(2)) ^ 2);
      final x0 = NDArray.fromList([0.0, 0.0], [2], DType.float64);

      final result = SymbolicOptimizer.minimizeGradientDescent(
        objective: loss,
        variables: [x, y],
        x0: x0,
        learningRate: 0.1,
        maxIterations: 200,
      );

      final solX = (result.solution.getCell([0]) as num).toDouble();
      final solY = (result.solution.getCell([1]) as num).toDouble();
      expect(solX, closeTo(3.0, 1e-5));
      expect(solY, closeTo(-2.0, 1e-5));
    });

    test(
      'minimizeNewton on Rosenbrock function with exact symbolic Hessian',
      () {
        final x = Symbol('x');
        final y = Symbol('y');
        // Rosenbrock: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2, minimum at (1, 1)
        final rosenbrock =
            ((1.toExpr - x) ^ 2) + 100.toExpr * ((y - (x ^ 2)) ^ 2);
        final x0 = NDArray.fromList([-1.2, 1.0], [2], DType.float64);

        final res = SymbolicOptimizer.minimizeNewton(
          objective: rosenbrock,
          variables: [x, y],
          x0: x0,
          maxIterations: 50,
          gradientTolerance: 1e-8,
        );

        final solX = (res.solution.getCell([0]) as num).toDouble();
        final solY = (res.solution.getCell([1]) as num).toDouble();
        expect(solX, closeTo(1.0, 1e-6));
        expect(solY, closeTo(1.0, 1e-6));
        expect(res.loss, closeTo(0.0, 1e-10));
      },
    );

    test('jacobian throws ArgumentError on non-vector 2x2 matrix', () {
      final x = Symbol('x');
      final mat2x2 = SymbolicMatrix.fromList([
        [x, 1],
        [2, x ^ 2],
      ]);
      expect(() => mat2x2.jacobian([x]), throwsArgumentError);
    });
  });

  group('NDArray.scope scoped memory management with symbolic_dart', () {
    test('automatic disposal of intermediate symbolic NDArray results', () {
      NDArray<Float64>? weakRefA;
      NDArray<Float64>? weakRefB;

      NDArray.scope(() {
        final x = Symbol('x');
        final a = NDArray.ones([50, 50], DType.float64);
        final b = a.mapSymbolic(sin(x) * 2, x);

        weakRefA = a;
        weakRefB = b;

        expect(a.isDisposed, isFalse);
        expect(b.isDisposed, isFalse);
      });

      // After NDArray.scope finishes, both intermediate arrays must be disposed
      expect(weakRefA!.isDisposed, isTrue);
      expect(weakRefB!.isDisposed, isTrue);
    });

    test('returning result from NDArray.returning survives inner scope', () {
      final survived = NDArray.returning(() {
        final x = Symbol('x');
        final y = Symbol('y');
        final mat = SymbolicMatrix.fromList([
          [x ^ 2, y * 3],
          [x + y, Integer(10)],
        ]);
        final evaluated = mat.subs({x: 2.0, y: 5.0});
        return evaluated.toNDArray();
      });

      expect(survived.isDisposed, isFalse);
      expect(survived.getCell([0, 0]), (4.0));
      expect(survived.getCell([0, 1]), (15.0));
      expect(survived.getCell([1, 0]), (7.0));
      expect(survived.getCell([1, 1]), (10.0));

      survived.dispose();
      expect(survived.isDisposed, isTrue);
    });

    test(
      'Expr, SymbolicMatrix, and FlintRationalPoly automatically disposed in scope',
      () {
        Expr? exprRef;
        SymbolicMatrix? matRef;
        FlintRationalPoly? polyRef;

        NDArray.scope(() {
          final x = Symbol('x');
          exprRef = (x ^ 3) + sin(x);
          matRef = SymbolicMatrix.fromList([
            [x, Integer(1)],
            [Integer(0), x * 2],
          ]);
          polyRef = FlintRationalPoly.fromIntCoefficients([5, -2, 7]);

          expect(exprRef!.isDisposed, isFalse);
          expect(matRef!.isDisposed, isFalse);
          expect(polyRef!.isDisposed, isFalse);
        });

        expect(exprRef!.isDisposed, isTrue);
        expect(matRef!.isDisposed, isTrue);
        expect(polyRef!.isDisposed, isTrue);
      },
    );
  });
}
