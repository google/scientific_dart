# symbolic_dart (Experimental)

<img src="doc/images/mascot.png" align="right" width="220" alt="symbolic_dart mascot">

> [!WARNING]
> **Experimental**: This package is experimental and under active development. Its API and native SymEngine/FLINT build hooks may change significantly before reaching a stable release.

A high-performance symbolic computer algebra system (CAS) for Dart, built on top of native **SymEngine** (C++) and **FLINT** (exact number theory & computer algebra).

## Features

- **Symbolic Expressions (`Expr`)**:
  - Expression trees (`Symbol`, `Integer`, `Real`, `Rational`)
  - Arithmetic operators (`+`, `-`, `*`, `/`, `-`, `^`) and natural numeric arithmetic (`2 * x + y - 5` via `SymbolicNumExtension`)
  - Elementary functions (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `log`, `sqrt`, `abs`)
  - Symbolic differentiation (`diff`), algebraic expansion (`expand`), and substitution (`subs`)
  - Analytical gradient vectors (`f.gradient([x, y])`) and Hessian matrices (`f.hessian([x, y])`)
  - Truncated Taylor / Maclaurin series expansion (`f.series(x, order: 6)`)
  - Formatting (`toString`, `toLatex`, `toCCode`, `toJSCode`)
- **Vectorized Array Evaluation (`SymbolicLambda`)**:
  - Compiles symbolic formulas via `.lambdify([x, y])` into functions that evaluate vectorized over multi-dimensional `NDArray<Float64>` arrays from `package:ndarray` with automatic shape broadcasting.
- **Symbolic Matrices & Analytical Jacobians (`SymbolicMatrix`)**:
  - Exact matrix addition, subtraction, multiplication (`A * B`), and transpose (`transpose`).
  - Exact symbolic determinant (`det`) and matrix inverse (`inv`).
  - Exact linear equation solving (`A.solve(b)`) via LU decomposition.
  - Analytical Jacobian matrix generation (`vecF.jacobian(variables)`) for vector systems $\vec{f}(\vec{x})$.
  - Bidirectional conversion to/from 2D numeric arrays: `SymbolicMatrix.fromNDArray(arr)` and `.toNDArray()`.
- **Seamless `NDArray` Integration (`NDArraySymbolicExtension` & `evaluateSymbolic`)**:
  - Univariate element-wise mapping: `arr.mapSymbolic(sin(x) + (x ^ 2), x)`
  - Named broadcasting evaluation: `evaluateSymbolic(f, inputs: {x: xArr, y: yArr})`
- **Auto-Diff Root Solvers & Optimizers (`SymbolicOptimizer`)**:
  - `SymbolicOptimizer.solveNewtonRaphson`: Solves non-linear equations $\vec{f}(\vec{x}) = \vec{0}$ over `NDArray<Float64>` state vectors by pre-compiling analytical Jacobian matrices and solving linear systems via Gaussian elimination with partial pivoting.
  - `SymbolicOptimizer.minimizeGradientDescent`: Minimizes objective functions $L(\vec{x})$ using exact symbolic gradients with backtracking Armijo line search.
  - `SymbolicOptimizer.minimizeNewton`: Second-order minimization of $L(\vec{x})$ using exact analytical gradients and Hessian matrices with Levenberg-Marquardt regularization and backtracking Armijo line search.
- **Exact Polynomial Ring over $\mathbb{Q}[x]$ (`FlintRationalPoly`)**:
  - Exact addition, subtraction, multiplication, and quotient/remainder division (`divmod`)
  - Exact derivative, integral, and polynomial GCD (`gcd`)
  - Conversion to/from symbolic expressions: `FlintRationalPoly.fromExpr(expr, x)` and `.toExpr(x)`
  - **Exact Polynomial Factorization (`factor`)** into monic/primitive irreducible factors and rational scalar content.

> [!IMPORTANT]
> **Operator Precedence for Power (`^`) in Dart**:
> In Dart, the `^` operator is bitwise XOR, which has **lower precedence** than arithmetic operators (`+`, `-`, `*`, `/`).
> For example, `sin(x) + x ^ 2` is parsed by Dart as `(sin(x) + x) ^ 2`.
> Always wrap powers in parentheses (e.g. `(x ^ 2)`) or use `x.pow(2)`.

## Quickstart

```dart
import 'package:ndarray/ndarray.dart';
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  final x = Symbol('x');
  final y = Symbol('y');

  // 1. Natural arithmetic with SymbolicNumExtension & calculus
  // Notice: 2.toExpr * x + y - 5 or (x * 2) + y - 5 works naturally
  final f = sin(x ^ 2) + 2.toExpr * x + y - 5;
  print('f(x, y) = $f');
  print('df/dx   = ${f.diff(x)}');

  // Exact analytical gradient vector and Hessian matrix
  final grad = f.gradient([x, y]);
  final hess = f.hessian([x, y]);
  print('Gradient:\n$grad');
  print('Hessian:\n$hess');

  // Truncated Taylor series expansion
  final taylor = sin(x).series(x, at: 0, order: 5);
  print('Taylor series of sin(x): $taylor');

  // 2. Vectorized evaluation over NDArray
  final wave = sin(x) * exp(Real(-0.1) * y);
  final lambda = wave.lambdify([x, y]);

  final xArr = NDArray.fromList([0.0, 1.57, 3.14], [3], DType.float64);
  final yArr = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
  final zArr = lambda.callArray([xArr, yArr]);
  print('Vectorized output shape: ${zArr.shape}');

  // 3. Exact polynomial factorization & conversion over Q[x]
  final polyFromExpr = FlintRationalPoly.fromExpr((x ^ 2) - 4, x);
  final p2 = FlintRationalPoly.fromIntCoefficients([3, 2]); // 2x + 3
  final poly = polyFromExpr * p2 * p2;

  final fac = poly.factor();
  print('Content = ${fac.content.numerator}/${fac.content.denominator}');
  for (final item in fac.factors) {
    print('  Factor (${item.factor}) ^ ${item.exponent}');
  }

  // 4. Exact equation & linear system solving
  final roots = Expr.solvePoly((x ^ 2) - 9, x);
  print('Roots of x^2 - 9 = 0: $roots');

  final sol = Expr.solveLinearSystem([
    2.toExpr * x + y - 5,
    x + 3.toExpr * y - 5,
  ], [x, y]);
  print('Linear system solution (x, y): $sol');

  // 5. Numerical Optimization with exact analytical derivatives
  final opt = SymbolicOptimizer.minimizeNewton(
    objective: ((x - 3) ^ 2) + ((y + 2) ^ 2),
    variables: [x, y],
    x0: NDArray.zeros([2], DType.float64),
  );
  print('Newton solution: ${opt.solution}, loss: ${opt.loss}');

  // Code Generation
  print('C code: ${wave.toCCode()}');
  print('JS code: ${wave.toJSCode()}');
}
```

## Scoped Memory Management

`package:symbolic_dart` integrates with **`package:resource_scope`** (`ScopedResource`) and `package:ndarray`'s zone-based scopes (`NDArray.scope`).

All symbolic AST nodes (`Expr`), matrices (`SymbolicMatrix`), exact polynomials (`FlintRationalPoly`), and numerical buffers (`NDArray`) created inside a scope block are automatically and deterministically disposed (`.dispose()`) when the block exits:

```dart
import 'package:ndarray/ndarray.dart';
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  // Free all intermediate C++ SymEngine AST nodes and C buffers immediately at scope exit
  final result = NDArray.returning(() {
    final x = Symbol('x');
    final mat = SymbolicMatrix.fromVector([(x ^ 2), sin(x)]);
    final jacobian = mat.jacobian([x]);
    return jacobian.subs({x: 2.0}).toNDArray(); // Promoted out of scope
  });

  print(result);
  result.dispose();
}
```
