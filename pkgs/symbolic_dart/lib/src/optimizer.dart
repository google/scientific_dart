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
import 'expr.dart';
import 'lambdify.dart';
import 'matrix.dart';

/// Numerical optimizers and root solvers powered by exact symbolic analytical
/// derivatives, Jacobians, and Hessians from `symbolic_dart`, operating on
/// `NDArray<Float64>` state vectors.
final class SymbolicOptimizer {
  SymbolicOptimizer._();

  /// Solves a system of non-linear equations `f_1(x) = 0, ..., f_n(x) = 0`
  /// using exact analytical Jacobian matrices via the Newton-Raphson method.
  ///
  /// - [equations]: List of `n` symbolic equations `f_i(x_1, ..., x_n) = 0`.
  /// - [variables]: List of `n` symbolic variables `[x_1, ..., x_n]`.
  /// - [x0]: Initial guess 1D [NDArray<Float64>] of shape `[n]`.
  /// - [maxIterations]: Maximum number of Newton iterations.
  /// - [tolerance]: Infinity-norm convergence threshold on residual `||f(x)||_\infty`.
  ///
  /// Returns the solution vector [NDArray<Float64>], number of iterations taken,
  /// and final residual max-norm.
  ///
  /// It is an error if `equations.length != variables.length`, or if `x0` is not
  /// a 1D array matching `variables.length`.
  static ({NDArray<Float64> solution, int iterations, double residual})
  solveNewtonRaphson({
    required List<Expr> equations,
    required List<Expr> variables,
    required NDArray<Float64> x0,
    int maxIterations = 100,
    double tolerance = 1e-9,
  }) {
    final n = variables.length;
    if (equations.length != n) {
      throw ArgumentError(
        'Expected $n equations to match $n variables, got ${equations.length}',
      );
    }
    if (x0.shape.length != 1 || x0.shape[0] != n) {
      throw ArgumentError(
        'Initial guess x0 must be a 1D array of length $n, got shape ${x0.shape}',
      );
    }

    // Pre-compile f(x) vector and exact analytical Jacobian matrix J(x) ONCE
    final fVec = SymbolicMatrix.fromVector(equations);
    final jacMat = fVec.jacobian(variables);
    final fLambda = fVec.lambdify(variables);
    final jLambda = jacMat.lambdify(variables);

    try {
      final xCurrent = NDArray.zeros([n], DType.float64);
      final scalarBuf = List<double>.filled(n, 0.0);
      for (var i = 0; i < n; i++) {
        final val = (x0.getCell([i]) as num).toDouble();
        scalarBuf[i] = val;
        xCurrent.setCell([i], (val));
      }

      double maxRes = 0.0;

      for (var iter = 0; iter < maxIterations; iter++) {
        final (converged, delta) = NDArray.scope(() {
          final fArr = fLambda.callScalar(scalarBuf);
          final jArr = jLambda.callScalar(scalarBuf);

          maxRes = 0.0;
          for (var i = 0; i < n; i++) {
            final absVal = (fArr.getCell([i, 0]) as num).toDouble().abs();
            if (absVal > maxRes) maxRes = absVal;
          }

          if (maxRes < tolerance) {
            return (true, null);
          }

          // Solve linear system J * Delta x = -f using Gaussian elimination
          // with partial pivoting on double matrices directly in Dart.
          final aMat = List<double>.filled(n * n, 0.0);
          final bVec = List<double>.filled(n, 0.0);
          for (var i = 0; i < n; i++) {
            bVec[i] = -(fArr.getCell([i, 0]) as num).toDouble();
            for (var j = 0; j < n; j++) {
              aMat[i * n + j] = (jArr.getCell([i, j]) as num).toDouble();
            }
          }

          final d = _solveLinearSystem(n, aMat, bVec);
          if (d == null) {
            throw StateError(
              'Jacobian matrix is singular or ill-conditioned at iteration $iter',
            );
          }
          return (false, d);
        });

        if (converged) {
          return (solution: xCurrent, iterations: iter, residual: maxRes);
        }

        // Update x_{k+1} = x_k + Delta x
        for (var i = 0; i < n; i++) {
          scalarBuf[i] += delta![i];
          xCurrent.setCell([i], (scalarBuf[i]));
        }
      }

      return (solution: xCurrent, iterations: maxIterations, residual: maxRes);
    } finally {
      fLambda.dispose();
      jLambda.dispose();
      fVec.dispose();
      jacMat.dispose();
    }
  }

  /// Minimizes a scalar [objective] function `L(x_1, ..., x_n)` using exact
  /// analytical symbolic gradients `\nabla L` via Gradient Descent with
  /// backtracking Armijo line search.
  ///
  /// - [objective]: Scalar symbolic loss function `L`.
  /// - [variables]: Parameter symbols `[x_1, ..., x_n]`.
  /// - [x0]: Initial guess 1D [NDArray<Float64>] of length `n`.
  /// - [learningRate]: Initial step size alpha.
  /// - [maxIterations]: Maximum gradient descent iterations.
  /// - [gradientTolerance]: Convergence threshold on gradient infinity-norm `||\nabla L||_\infty`.
  ///
  /// It is an error if `x0` is not a 1D array matching `variables.length`.
  static ({
    NDArray<Float64> solution,
    int iterations,
    double loss,
    double gradientNorm,
  })
  minimizeGradientDescent({
    required Expr objective,
    required List<Expr> variables,
    required NDArray<Float64> x0,
    double learningRate = 0.1,
    int maxIterations = 200,
    double gradientTolerance = 1e-7,
  }) {
    final n = variables.length;
    if (x0.shape.length != 1 || x0.shape[0] != n) {
      throw ArgumentError(
        'Initial guess x0 must be a 1D array of length $n, got shape ${x0.shape}',
      );
    }

    // Pre-compile objective and gradient vector ONCE before the loop
    final lossLambda = objective.lambdify(variables);
    final gradVec = SymbolicMatrix.fromVector(
      variables.map((v) => objective.diff(v)).toList(),
    );
    final gradLambda = gradVec.lambdify(variables);

    try {
      final xCurrent = NDArray.zeros([n], DType.float64);
      final xBuf = List<double>.filled(n, 0.0);
      for (var i = 0; i < n; i++) {
        final val = (x0.getCell([i]) as num).toDouble();
        xBuf[i] = val;
        xCurrent.setCell([i], (val));
      }

      double currentLoss = 0.0;
      double maxGradNorm = 0.0;

      for (var iter = 0; iter < maxIterations; iter++) {
        currentLoss = lossLambda.callScalar(xBuf);

        final (grad, gradL2Sq, gNorm) = NDArray.scope(() {
          final gradArr = gradLambda.callScalar(xBuf);
          final gList = List<double>.filled(n, 0.0);
          var l2Sq = 0.0;
          var maxG = 0.0;
          for (var i = 0; i < n; i++) {
            final g = (gradArr.getCell([i, 0]) as num).toDouble();
            gList[i] = g;
            l2Sq += g * g;
            final absG = g.abs();
            if (absG > maxG) maxG = absG;
          }
          return (gList, l2Sq, maxG);
        });

        maxGradNorm = gNorm;

        if (maxGradNorm < gradientTolerance) {
          return (
            solution: xCurrent,
            iterations: iter,
            loss: currentLoss,
            gradientNorm: maxGradNorm,
          );
        }

        // Backtracking Armijo line search:
        // Start with step = learningRate. While step > 1e-12: evaluate candidate
        // point x_cand = x_k - step * g_k.
        // If L(x_cand) <= L(x_k) - 1e-4 * step * ||g_k||_2^2, accept and break;
        // otherwise step *= 0.5.
        var step = learningRate;
        final xCand = List<double>.filled(n, 0.0);
        var stepAccepted = false;

        while (step > 1e-12) {
          for (var i = 0; i < n; i++) {
            xCand[i] = xBuf[i] - (step * grad[i]);
          }
          final lossCand = lossLambda.callScalar(xCand);
          if (lossCand <= currentLoss - (1e-4 * step * gradL2Sq)) {
            for (var i = 0; i < n; i++) {
              xBuf[i] = xCand[i];
              xCurrent.setCell([i], (xBuf[i]));
            }
            currentLoss = lossCand;
            stepAccepted = true;
            break;
          }
          step *= 0.5;
        }

        if (!stepAccepted) {
          break;
        }
      }

      NDArray.scope(() {
        final finalGradArr = gradLambda.callScalar(xBuf);
        maxGradNorm = 0.0;
        for (var i = 0; i < n; i++) {
          final absG = (finalGradArr.getCell([i, 0]) as num).toDouble().abs();
          if (absG > maxGradNorm) maxGradNorm = absG;
        }
      });

      return (
        solution: xCurrent,
        iterations: maxIterations,
        loss: currentLoss,
        gradientNorm: maxGradNorm,
      );
    } finally {
      lossLambda.dispose();
      gradLambda.dispose();
      gradVec.dispose();
    }
  }

  /// Minimizes a scalar [objective] function `L(x_1, ..., x_n)` using exact
  /// analytical symbolic gradients `\nabla L` and exact analytical symbolic
  /// Hessian matrices `H = \nabla^2 L` via Newton's method with
  /// Levenberg-Marquardt regularization and backtracking Armijo line search.
  ///
  /// - [objective]: Scalar symbolic loss function `L`.
  /// - [variables]: Parameter symbols `[x_1, ..., x_n]`.
  /// - [x0]: Initial guess 1D [NDArray<Float64>] of length `n`.
  /// - [maxIterations]: Maximum Newton iterations.
  /// - [gradientTolerance]: Convergence threshold on gradient infinity-norm `||\nabla L||_\infty`.
  ///
  /// It is an error if `variables` is empty, or if `x0` is not a 1D array
  /// matching `variables.length`.
  static ({
    NDArray<Float64> solution,
    int iterations,
    double loss,
    double gradientNorm,
  })
  minimizeNewton({
    required Expr objective,
    required List<Expr> variables,
    required NDArray<Float64> x0,
    int maxIterations = 100,
    double gradientTolerance = 1e-8,
  }) {
    final n = variables.length;
    if (n == 0) {
      throw ArgumentError('Variables list must not be empty');
    }
    if (x0.shape.length != 1 || x0.shape[0] != n) {
      throw ArgumentError(
        'Initial guess x0 must be a 1D array of length $n, got shape ${x0.shape}',
      );
    }

    // Pre-compile loss, gradient vector, and Hessian matrix ONCE before loop
    final lossLambda = objective.lambdify(variables);
    final gradVec = SymbolicMatrix.fromVector(
      variables.map((v) => objective.diff(v)).toList(),
    );
    final gradLambda = gradVec.lambdify(variables);
    final hessMat = gradVec.jacobian(variables);
    final hessLambda = hessMat.lambdify(variables);

    try {
      final xCurrent = NDArray.zeros([n], DType.float64);
      final xBuf = List<double>.filled(n, 0.0);
      for (var i = 0; i < n; i++) {
        final val = (x0.getCell([i]) as num).toDouble();
        xBuf[i] = val;
        xCurrent.setCell([i], (val));
      }

      double currentLoss = 0.0;
      double maxGradNorm = 0.0;

      for (var iter = 0; iter < maxIterations; iter++) {
        currentLoss = lossLambda.callScalar(xBuf);

        final (grad, hMat, gNorm) = NDArray.scope(() {
          final gradArr = gradLambda.callScalar(xBuf);
          final hessArr = hessLambda.callScalar(xBuf);

          final gList = List<double>.filled(n, 0.0);
          var maxG = 0.0;
          for (var i = 0; i < n; i++) {
            final g = (gradArr.getCell([i, 0]) as num).toDouble();
            gList[i] = g;
            final absG = g.abs();
            if (absG > maxG) maxG = absG;
          }

          final hList = List<double>.filled(n * n, 0.0);
          for (var i = 0; i < n; i++) {
            for (var j = 0; j < n; j++) {
              hList[i * n + j] = (hessArr.getCell([i, j]) as num).toDouble();
            }
          }

          return (gList, hList, maxG);
        });

        maxGradNorm = gNorm;

        if (maxGradNorm < gradientTolerance) {
          return (
            solution: xCurrent,
            iterations: iter,
            loss: currentLoss,
            gradientNorm: maxGradNorm,
          );
        }

        // Compute Newton search direction d by solving H * d = -g.
        // If H is not positive-definite or if grad^T * d >= 0, regularize with
        // Levenberg-Marquardt damping lambda * I.
        var lambda = 0.0;
        List<double>? d;
        var gradDotD = 0.0;

        for (var attempt = 0; attempt < 12; attempt++) {
          final aMat = List<double>.from(hMat);
          if (lambda > 0.0) {
            for (var i = 0; i < n; i++) {
              aMat[i * n + i] += lambda;
            }
          }

          final bVec = List<double>.generate(n, (i) => -grad[i]);
          final sol = _solveLinearSystem(n, aMat, bVec);
          if (sol != null) {
            var dot = 0.0;
            for (var i = 0; i < n; i++) {
              dot += grad[i] * sol[i];
            }
            if (dot < 0.0) {
              d = sol;
              gradDotD = dot;
              break;
            }
          }

          if (lambda == 0.0) {
            var maxDiag = 0.0;
            for (var i = 0; i < n; i++) {
              final diag = hMat[i * n + i].abs();
              if (diag > maxDiag) maxDiag = diag;
            }
            lambda = maxDiag > 0.0 ? maxDiag * 1e-3 : 1e-3;
          } else {
            lambda *= 10.0;
          }
        }

        if (d == null) {
          // Fallback to steepest descent direction d = -g
          d = List<double>.generate(n, (i) => -grad[i]);
          gradDotD = 0.0;
          for (var i = 0; i < n; i++) {
            gradDotD += grad[i] * d[i];
          }
        }

        // Backtracking Armijo line search along direction d:
        // Start with step = 1.0. While step > 1e-12: evaluate candidate
        // point x_cand = x_k + step * d.
        // If L(x_cand) <= L(x_k) + 1e-4 * step * (grad^T * d), accept and break;
        // otherwise step *= 0.5.
        var step = 1.0;
        final xCand = List<double>.filled(n, 0.0);
        var stepAccepted = false;

        while (step > 1e-12) {
          for (var i = 0; i < n; i++) {
            xCand[i] = xBuf[i] + (step * d[i]);
          }
          final lossCand = lossLambda.callScalar(xCand);
          if (lossCand <= currentLoss + (1e-4 * step * gradDotD)) {
            for (var i = 0; i < n; i++) {
              xBuf[i] = xCand[i];
              xCurrent.setCell([i], (xBuf[i]));
            }
            currentLoss = lossCand;
            stepAccepted = true;
            break;
          }
          step *= 0.5;
        }

        if (!stepAccepted) {
          break;
        }
      }

      NDArray.scope(() {
        final finalGradArr = gradLambda.callScalar(xBuf);
        maxGradNorm = 0.0;
        for (var i = 0; i < n; i++) {
          final absG = (finalGradArr.getCell([i, 0]) as num).toDouble().abs();
          if (absG > maxGradNorm) maxGradNorm = absG;
        }
      });

      return (
        solution: xCurrent,
        iterations: maxIterations,
        loss: currentLoss,
        gradientNorm: maxGradNorm,
      );
    } finally {
      lossLambda.dispose();
      gradLambda.dispose();
      hessLambda.dispose();
      gradVec.dispose();
      hessMat.dispose();
    }
  }

  /// Solves `A * x = b` for an `n x n` system using Gaussian elimination with
  /// partial pivoting. Returns `null` if the matrix is singular or ill-conditioned.
  static List<double>? _solveLinearSystem(
    int n,
    List<double> aMat,
    List<double> bVec,
  ) {
    for (var k = 0; k < n; k++) {
      var pivotRow = k;
      var maxVal = aMat[k * n + k].abs();
      for (var i = k + 1; i < n; i++) {
        final val = aMat[i * n + k].abs();
        if (val > maxVal) {
          maxVal = val;
          pivotRow = i;
        }
      }

      if (maxVal < 1e-15) {
        return null;
      }

      if (pivotRow != k) {
        for (var j = k; j < n; j++) {
          final tmp = aMat[k * n + j];
          aMat[k * n + j] = aMat[pivotRow * n + j];
          aMat[pivotRow * n + j] = tmp;
        }
        final tmpB = bVec[k];
        bVec[k] = bVec[pivotRow];
        bVec[pivotRow] = tmpB;
      }

      final pivot = aMat[k * n + k];
      for (var i = k + 1; i < n; i++) {
        final factor = aMat[i * n + k] / pivot;
        aMat[i * n + k] = 0.0;
        for (var j = k + 1; j < n; j++) {
          aMat[i * n + j] -= factor * aMat[k * n + j];
        }
        bVec[i] -= factor * bVec[k];
      }
    }

    final x = List<double>.filled(n, 0.0);
    for (var i = n - 1; i >= 0; i--) {
      var sum = bVec[i];
      for (var j = i + 1; j < n; j++) {
        sum -= aMat[i * n + j] * x[j];
      }
      x[i] = sum / aMat[i * n + i];
    }
    return x;
  }
}
