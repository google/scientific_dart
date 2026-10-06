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

// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;

import 'package:openblas/openblas.dart';

import '../ndarray.dart';
import '../scratch_arena.dart';
import 'helpers.dart';

/// Method selection for 1D scalar root finding ([root_scalar]).
enum RootMethod {
  /// Brent's algorithm combining bisection, secant, and inverse quadratic interpolation.
  brentq,

  /// Newton-Raphson method (requires derivative [fprime]).
  newton,

  /// Secant method (derivative-free secant iteration).
  secant,
}

void _dcopy(int n, ffi.Pointer<ffi.Double> src, ffi.Pointer<ffi.Double> dest) {
  checkBlasIntDim(n, 'n', 'optimize');
  cblas_dcopy(n, src, 1, dest, 1);
}

void _copyArrayToPointer(
  int n,
  NDArray<Float64> arr,
  ffi.Pointer<ffi.Double> dest,
) {
  checkBlasIntDim(n, 'n', 'optimize');
  if (arr.rank == 1) {
    checkBlasIntStride(arr.strides[0], 'incx', 'optimize');
  }
  if (arr.isContiguous) {
    cblas_dcopy(n, arr.pointer.cast<ffi.Double>(), 1, dest, 1);
  } else if (arr.rank == 1 && arr.strides[0] > 0) {
    cblas_dcopy(n, arr.pointer.cast<ffi.Double>(), arr.strides[0], dest, 1);
  } else if (arr.rank == 1) {
    final stride = arr.strides[0];
    final ptr = arr.pointer.cast<ffi.Double>();
    for (var i = 0; i < n; i++) {
      dest[i] = ptr[i * stride];
    }
  } else {
    final contig = arr.copy();
    try {
      cblas_dcopy(n, contig.pointer.cast<ffi.Double>(), 1, dest, 1);
    } finally {
      contig.dispose();
    }
  }
}

void _copyPointerToArray(
  int n,
  ffi.Pointer<ffi.Double> src,
  NDArray<Float64> dest,
) {
  checkBlasIntDim(n, 'n', 'optimize');
  if (dest.rank == 1) {
    checkBlasIntStride(dest.strides[0], 'incy', 'optimize');
  }
  if (dest.isContiguous) {
    cblas_dcopy(n, src, 1, dest.pointer.cast<ffi.Double>(), 1);
  } else if (dest.rank == 1 && dest.strides[0] > 0) {
    cblas_dcopy(n, src, 1, dest.pointer.cast<ffi.Double>(), dest.strides[0]);
  } else {
    final stride = dest.strides[0];
    final ptr = dest.pointer.cast<ffi.Double>();
    for (var i = 0; i < n; i++) {
      ptr[i * stride] = src[i];
    }
  }
}

double _ddot(int n, ffi.Pointer<ffi.Double> x, ffi.Pointer<ffi.Double> y) {
  checkBlasIntDim(n, 'n', 'optimize');
  return cblas_ddot(n, x, 1, y, 1);
}

void _daxpy(
  int n,
  double alpha,
  ffi.Pointer<ffi.Double> x,
  ffi.Pointer<ffi.Double> y,
) {
  checkBlasIntDim(n, 'n', 'optimize');
  cblas_daxpy(n, alpha, x, 1, y, 1);
}

void _dscal(int n, double alpha, ffi.Pointer<ffi.Double> x) {
  checkBlasIntDim(n, 'n', 'optimize');
  cblas_dscal(n, alpha, x, 1);
}

double _dnrm2(int n, ffi.Pointer<ffi.Double> x) {
  checkBlasIntDim(n, 'n', 'optimize');
  return cblas_dnrm2(n, x, 1);
}

/// Method selection for multivariate scalar function minimization ([minimize]).
enum MinimizeMethod {
  /// Nelder-Mead simplex algorithm (derivative-free).
  nelderMead,

  /// Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton algorithm.
  lbfgs,
}

/// Type alias for 1D root finding results represented as a named record.
///
/// Contains the calculated root location [root], convergence flag [converged],
/// iteration count [iterations], function evaluation count [functionCalls],
/// and diagnostic message [message].
typedef RootScalarResult = ({
  double root,
  bool converged,
  int iterations,
  int functionCalls,
  String message,
});

/// Type alias for multivariate scalar optimization results represented as a named record.
///
/// Contains the optimal vector [x], final objective value [fun], success status [success],
/// iteration count [nit], function evaluation count [nfev], diagnostic message [message],
/// and optional final gradient vector [jac].
///
/// ### Diagnostic Message Format ([message])
/// The [message] field provides detailed human-readable feedback on algorithm termination states:
/// - `'Optimization terminated successfully.'`: Standard convergence when parameters meet `xatol`/`fatol` tolerances.
/// - `'Optimization terminated successfully (gradient norm <= gtol).'`: Gradient norm reached tolerance in L-BFGS.
/// - `'Exact root found at bracket endpoint a.'` / `'Exact root found at bracket endpoint b.'`: Exact zero found at bracket endpoints.
/// - `'Converged to root within requested tolerance.'`: Scalar root iteration satisfied absolute/relative tolerances.
/// - `'Maximum iterations or evaluations reached'`: Iteration limit reached prior to convergence.
/// - `'Line search failed to find sufficient decrease'`: Line search step length could not satisfy the Armijo condition.
/// - `'Derivative evaluated to zero.'`: Newton iteration step failed due to zero derivative.
/// - `'Zero denominator encountered in secant step.'`: Secant step failed due to flat function value.
typedef OptimizeResult = ({
  NDArray<Float64> x,
  double fun,
  bool success,
  int nit,
  int nfev,
  String message,
  NDArray<Float64>? jac,
});

/// Finds a root of a scalar function within a bracketed interval $[a, b]$ using Brent's method.
///
/// Brent's method combines root bracketing, inverse quadratic interpolation, the secant method,
/// and bisection to achieve superlinear convergence while guaranteeing linear convergence in worst cases.
///
/// It is an error if $f(a)$ and $f(b)$ have the same sign ($f(a) \cdot f(b) > 0$).
/// It is an error if [maxiter] is less than or equal to zero.
/// It is an error if [xtol] or [rtol] is negative.
///
/// ### References & Further Reading
/// - [NumPy / SciPy brentq Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
/// - [Wikipedia: Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method)
///
/// {@example /example/optimize_example.dart lang=dart}
RootScalarResult brentq(
  double Function(double) f,
  double a,
  double b, {
  double xtol = 2e-12,
  double rtol = 8.881784197001252e-16,
  int maxiter = 100,
}) {
  if (maxiter <= 0) {
    throw ArgumentError.value(
      maxiter,
      'maxiter',
      'Must be positive (got $maxiter)',
    );
  }
  if (xtol < 0 || rtol < 0) {
    throw ArgumentError.value(
      [xtol, rtol],
      'xtol, rtol',
      'Must be non-negative (tolerances xtol and rtol must be non-negative)',
    );
  }

  var fa = f(a);
  var fb = f(b);
  var nfev = 2;

  if (fa == 0.0) {
    return (
      root: a,
      converged: true,
      iterations: 0,
      functionCalls: nfev,
      message: 'Exact root found at bracket endpoint a.',
    );
  }
  if (fb == 0.0) {
    return (
      root: b,
      converged: true,
      iterations: 0,
      functionCalls: nfev,
      message: 'Exact root found at bracket endpoint b.',
    );
  }

  if (fa.isNaN || fb.isNaN || (fa > 0.0) == (fb > 0.0)) {
    throw ArgumentError.value(
      [fa, fb],
      'a, b',
      'Must have different signs for f(a) and f(b) (got f(a)=$fa, f(b)=$fb)',
    );
  }

  var c = a;
  var fc = fa;
  var d = b - a;
  var e = d;

  for (var iter = 1; iter <= maxiter; iter++) {
    if (fb == 0.0) {
      return (
        root: b,
        converged: true,
        iterations: iter,
        functionCalls: nfev,
        message: 'Converged to root.',
      );
    }

    if ((fb > 0.0) == (fc > 0.0)) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }

    if (fc.abs() < fb.abs()) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }

    final tol1 = 2.0 * rtol * b.abs() + 0.5 * xtol;
    final xm = 0.5 * (c - b);

    if (xm.abs() <= tol1 || fb == 0.0) {
      return (
        root: b,
        converged: true,
        iterations: iter,
        functionCalls: nfev,
        message: 'Converged to root within requested tolerance.',
      );
    }

    if (e.abs() >= tol1 && fa.abs() > fb.abs()) {
      final s = fb / fa;
      var p = 0.0;
      var q = 0.0;
      if (a == c) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        final r = fb / fc;
        p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }

      if (p > 0) {
        q = -q;
      } else {
        p = -p;
      }

      final min1 = 3.0 * xm * q - (tol1 * q).abs();
      final min2 = (e * q).abs();

      if (2.0 * p < (min1 < min2 ? min1 : min2)) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }

    a = b;
    fa = fb;

    if (d.abs() > tol1) {
      b += d;
    } else {
      b += xm >= 0 ? tol1 : -tol1;
    }

    fb = f(b);
    nfev++;
  }

  return (
    root: b,
    converged: false,
    iterations: maxiter,
    functionCalls: nfev,
    message: 'Failed to converge within maximum iterations ($maxiter).',
  );
}

/// Finds a root of a scalar function using Newton-Raphson or Secant iteration.
///
/// If derivative function [fprime] is provided, Newton-Raphson iteration is performed:
/// $$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$
/// If [fprime] is `null`, secant iteration is used starting from initial guess [x0].
///
/// It is an error if [maxiter] is less than or equal to zero.
/// It is an error if [tol] is less than or equal to zero.
///
/// ### References & Further Reading
/// - [SciPy newton Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html)
/// - [Wikipedia: Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
///
/// {@example /example/optimize_example.dart lang=dart}
RootScalarResult newton(
  double Function(double) f,
  double x0, {
  double Function(double)? fprime,
  double tol = 1.48e-8,
  int maxiter = 50,
}) {
  if (maxiter <= 0) {
    throw ArgumentError.value(
      maxiter,
      'maxiter',
      'Must be positive (got $maxiter)',
    );
  }
  if (tol <= 0) {
    throw ArgumentError.value(tol, 'tol', 'Must be positive (got $tol)');
  }

  var p0 = x0;
  var nfev = 0;

  if (fprime != null) {
    for (var iter = 1; iter <= maxiter; iter++) {
      final y = f(p0);
      nfev++;
      if (y == 0.0) {
        return (
          root: p0,
          converged: true,
          iterations: iter - 1,
          functionCalls: nfev,
          message: 'Exact root found.',
        );
      }
      final yprime = fprime(p0);
      if (yprime == 0.0) {
        return (
          root: p0,
          converged: false,
          iterations: iter,
          functionCalls: nfev,
          message: 'Derivative evaluated to zero.',
        );
      }
      final p = p0 - y / yprime;
      if ((p - p0).abs() <= tol) {
        return (
          root: p,
          converged: true,
          iterations: iter,
          functionCalls: nfev,
          message: 'Converged to root within requested tolerance.',
        );
      }
      p0 = p;
    }
    return (
      root: p0,
      converged: false,
      iterations: maxiter,
      functionCalls: nfev,
      message: 'Failed to converge within maximum iterations ($maxiter).',
    );
  } else {
    var p1 = x0 != 0.0 ? x0 * 1.0001 : 1e-4;
    var q0 = f(p0);
    var q1 = f(p1);
    nfev += 2;

    for (var iter = 1; iter <= maxiter; iter++) {
      if (q1 == q0) {
        return (
          root: p1,
          converged: false,
          iterations: iter,
          functionCalls: nfev,
          message: 'Zero denominator encountered in secant step.',
        );
      }
      final p = p1 - q1 * (p1 - p0) / (q1 - q0);
      if ((p - p1).abs() <= tol) {
        return (
          root: p,
          converged: true,
          iterations: iter,
          functionCalls: nfev,
          message: 'Converged to root within requested tolerance.',
        );
      }
      p0 = p1;
      q0 = q1;
      p1 = p;
      q1 = f(p1);
      nfev++;
    }
    return (
      root: p1,
      converged: false,
      iterations: maxiter,
      functionCalls: nfev,
      message: 'Failed to converge within maximum iterations ($maxiter).',
    );
  }
}

/// Unified entry point for 1D scalar root finding algorithms.
///
/// Dispatches to supported root-finding methods defined in [RootMethod]:
/// - [RootMethod.brentq]: Requires bracket endpoints [bracketA] and [bracketB].
/// - [RootMethod.newton]: Requires initial guess [x0] and optional derivative [fprime].
/// - [RootMethod.secant]: Requires initial guess [x0].
///
/// It is an error if [method] is [RootMethod.brentq] and [bracketA] or [bracketB] is omitted.
/// It is an error if [method] is [RootMethod.newton] or [RootMethod.secant] and [x0] is omitted.
///
/// ### References & Further Reading
/// - [SciPy root_scalar Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html)
///
/// {@example /example/optimize_example.dart lang=dart}
RootScalarResult root_scalar(
  double Function(double) f, {
  RootMethod method = RootMethod.brentq,
  double? bracketA,
  double? bracketB,
  double? x0,
  double Function(double)? fprime,
  double xtol = 2e-12,
  double rtol = 8.881784197001252e-16,
  double tol = 1.48e-8,
  int maxiter = 100,
}) {
  switch (method) {
    case RootMethod.brentq:
      if (bracketA == null || bracketB == null) {
        throw ArgumentError.value(
          [bracketA, bracketB],
          'bracketA, bracketB',
          'Must be provided for brentq (brentq requires bracketA and bracketB)',
        );
      }
      return brentq(
        f,
        bracketA,
        bracketB,
        xtol: xtol,
        rtol: rtol,
        maxiter: maxiter,
      );
    case RootMethod.newton:
    case RootMethod.secant:
      if (x0 == null) {
        throw ArgumentError.value(
          x0,
          'x0',
          'Must be provided for $method ($method requires initial guess x0)',
        );
      }
      return newton(
        f,
        x0,
        fprime: method == RootMethod.newton ? fprime : null,
        tol: tol,
        maxiter: maxiter,
      );
  }
}

/// Minimizes a multivariate scalar objective function using the Nelder-Mead simplex algorithm.
///
/// All heavy vector updates and distance calculations are offloaded to C OpenBLAS intrinsics
/// using a fixed pool of pre-allocated scratch buffers.
///
/// If [out] is provided, the optimal parameter vector is written directly into [out] and
/// returned as the `x` field of the [OptimizeResult].
///
/// It is an error if [x0] or [out] is disposed.
/// It is an error if [x0] is not a 1-dimensional array, or if [out] does not match [x0]'s shape and dtype.
/// It is an error if [xatol] or [fatol] is negative.
///
/// ### References & Further Reading
/// - [SciPy minimize(method='Nelder-Mead') Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)
/// - [Wikipedia: Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
///
/// {@example /example/optimize_example.dart lang=dart}
OptimizeResult nelder_mead(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
  double xatol = 1e-4,
  double fatol = 1e-4,
  int? maxiter,
  int? maxfev,
  bool adaptive = false,
  NDArray<Float64>? out,
}) {
  if (x0.isDisposed) {
    throw StateError('Cannot execute nelder_mead on disposed x0 array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write nelder_mead result to a disposed out array.',
    );
  }
  if (x0.shape.length != 1) {
    throw ArgumentError.value(
      x0.shape,
      'x0',
      'Must be a 1D vector for nelder_mead (got shape ${x0.shape})',
    );
  }
  if (x0.size == 0) {
    throw ArgumentError.value(
      x0.size,
      'x0',
      'Must not be empty (initial vector x0 must not be empty)',
    );
  }
  if (out != null) {
    validateOutBuffer(out);
    if (out.shape.length != 1 ||
        out.shape[0] != x0.shape[0] ||
        out.dtype != DType.float64) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be a 1D Float64 array of length ${x0.shape[0]} (got shape ${out.shape}, dtype ${out.dtype})',
      );
    }
  }
  if (xatol < 0 || fatol < 0) {
    throw ArgumentError.value(
      [xatol, fatol],
      'xatol, fatol',
      'Must be non-negative (tolerances xatol and fatol must be non-negative)',
    );
  }
  checkBlasIntDim(x0.shape[0], 'n', 'nelder_mead');
  checkBlasIntStride(x0.strides[0], 'incx', 'nelder_mead');
  if (out != null) {
    checkBlasIntStride(out.strides[0], 'incout', 'nelder_mead');
  }

  return NDArray.scope(() {
    final n = x0.shape[0];
    final limitIter = maxiter ?? n * 200;
    final limitFev = maxfev ?? n * 200;

    final alpha = 1.0;
    final gamma = adaptive ? 1.0 + 2.0 / n : 2.0;
    final beta = adaptive ? 0.75 - 1.0 / (2.0 * n) : 0.5;
    final sigma = adaptive ? 1.0 - 1.0 / n : 0.5;

    final arenaMarker = ScratchArena.marker;
    try {
      final doubleBytes = ffi.sizeOf<ffi.Double>();
      final pSim = List<ffi.Pointer<ffi.Double>>.generate(
        n + 1,
        (_) => ScratchArena.allocate<ffi.Double>(n * doubleBytes),
      );
      final pFSim = ScratchArena.allocate<ffi.Double>((n + 1) * doubleBytes);
      final pXBar = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXR = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXE = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXC = ScratchArena.allocate<ffi.Double>(n * doubleBytes);

      _copyArrayToPointer(n, x0, pSim[0]);

      for (int i = 0; i < n; i++) {
        _dcopy(n, pSim[0], pSim[i + 1]);
        final val = pSim[i + 1][i];
        pSim[i + 1][i] = val != 0.0 ? val * 1.05 : 0.00025;
      }

      int nfev = 0;
      final pArr = NDArray<Float64>.create([n], DType.float64);
      final pArrPtr = pArr.pointer.cast<ffi.Double>();
      double evalPoint(ffi.Pointer<ffi.Double> ptr) {
        _dcopy(n, ptr, pArrPtr);
        final val = NDArray.scope(() => fun(pArr));
        nfev++;
        return val;
      }

      for (int i = 0; i <= n; i++) {
        pFSim[i] = evalPoint(pSim[i]);
      }

      final idx = List<int>.generate(n + 1, (i) => i);
      final tempSim = List<ffi.Pointer<ffi.Double>>.filled(n + 1, ffi.nullptr);
      final tempFSim = List<double>.filled(n + 1, 0.0);

      int nit = 0;
      bool success = false;
      String msg = 'Maximum iterations or evaluations reached';

      while (nit < limitIter && nfev < limitFev) {
        nit++;

        for (int i = 0; i <= n; i++) {
          idx[i] = i;
        }
        idx.sort((a, b) => pFSim[a].compareTo(pFSim[b]));

        for (int i = 0; i <= n; i++) {
          tempSim[i] = pSim[idx[i]];
          tempFSim[i] = pFSim[idx[i]];
        }
        for (int i = 0; i <= n; i++) {
          pSim[i] = tempSim[i];
          pFSim[i] = tempFSim[i];
        }

        double maxDiffF = 0.0;
        for (int i = 1; i <= n; i++) {
          final diff = (pFSim[i] - pFSim[0]).abs();
          if (diff > maxDiffF) maxDiffF = diff;
        }

        double maxDiffX = 0.0;
        for (int i = 1; i <= n; i++) {
          _dcopy(n, pSim[i], pXR);
          _daxpy(n, -1.0, pSim[0], pXR);
          final norm = _dnrm2(n, pXR);
          if (norm > maxDiffX) maxDiffX = norm;
        }

        if (maxDiffF < fatol && maxDiffX < xatol) {
          success = true;
          msg = 'Optimization terminated successfully.';
          break;
        }

        _dscal(n, 0.0, pXBar);
        for (int i = 0; i < n; i++) {
          _daxpy(n, 1.0, pSim[i], pXBar);
        }
        _dscal(n, 1.0 / n, pXBar);

        _dcopy(n, pXBar, pXR);
        _dscal(n, 1.0 + alpha, pXR);
        _daxpy(n, -alpha, pSim[n], pXR);
        final fxr = evalPoint(pXR);

        bool doshrink = false;
        if (fxr < pFSim[0]) {
          _dcopy(n, pXBar, pXE);
          _dscal(n, 1.0 - gamma, pXE);
          _daxpy(n, gamma, pXR, pXE);
          final fxe = evalPoint(pXE);
          if (fxe < fxr) {
            _dcopy(n, pXE, pSim[n]);
            pFSim[n] = fxe;
          } else {
            _dcopy(n, pXR, pSim[n]);
            pFSim[n] = fxr;
          }
        } else if (fxr < pFSim[n - 1]) {
          _dcopy(n, pXR, pSim[n]);
          pFSim[n] = fxr;
        } else {
          if (fxr < pFSim[n]) {
            _dcopy(n, pXBar, pXC);
            _dscal(n, 1.0 - beta, pXC);
            _daxpy(n, beta, pXR, pXC);
            final fxc = evalPoint(pXC);
            if (fxc <= fxr) {
              _dcopy(n, pXC, pSim[n]);
              pFSim[n] = fxc;
            } else {
              doshrink = true;
            }
          } else {
            _dcopy(n, pXBar, pXC);
            _dscal(n, 1.0 - beta, pXC);
            _daxpy(n, beta, pSim[n], pXC);
            final fxc = evalPoint(pXC);
            if (fxc < pFSim[n]) {
              _dcopy(n, pXC, pSim[n]);
              pFSim[n] = fxc;
            } else {
              doshrink = true;
            }
          }
        }

        if (doshrink) {
          for (int i = 1; i <= n; i++) {
            _dscal(n, sigma, pSim[i]);
            _daxpy(n, 1.0 - sigma, pSim[0], pSim[i]);
            pFSim[i] = evalPoint(pSim[i]);
          }
        }
      }

      final NDArray<Float64> resArr;
      if (out != null) {
        _copyPointerToArray(n, pSim[0], out);
        resArr = out;
      } else {
        resArr = NDArray<Float64>.create([n], DType.float64);
        _dcopy(n, pSim[0], resArr.pointer.cast<ffi.Double>());
        resArr.detachToParentScope();
      }

      return (
        x: resArr,
        fun: pFSim[0],
        success: success,
        nit: nit,
        nfev: nfev,
        message: msg,
        jac: null,
      );
    } finally {
      ScratchArena.reset(arenaMarker);
    }
  });
}

/// CamelCase alias for [nelder_mead].
OptimizeResult nelderMead(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
  double xatol = 1e-4,
  double fatol = 1e-4,
  int? maxiter,
  int? maxfev,
  bool adaptive = false,
  NDArray<Float64>? out,
}) => nelder_mead(
  fun,
  x0,
  xatol: xatol,
  fatol: fatol,
  maxiter: maxiter,
  maxfev: maxfev,
  adaptive: adaptive,
  out: out,
);

/// Minimizes a multivariate scalar objective function using the L-BFGS quasi-Newton algorithm.
///
/// All heavy matrix-free two-loop recursion step calculations and vector updates are offloaded
/// to C OpenBLAS intrinsics using a fixed ring-buffer pool of pre-allocated temporaries.
///
/// If [out] or [outJac] is provided, the final parameter vector and/or gradient vector are written
/// directly into those buffers. If [jacInto] is provided, gradients are computed in-place into a
/// pre-allocated gradient buffer without allocating a new [NDArray] per step.
///
/// It is an error if [x0], [out], or [outJac] is disposed.
/// It is an error if [x0] is not a 1-dimensional array.
/// It is an error if [m] is less than or equal to zero.
/// It is an error if [gtol] is less than or equal to zero or [maxiter] is less than or equal to zero.
///
/// ### References & Further Reading
/// - [SciPy minimize(method='L-BFGS-B') Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
/// - [Wikipedia: Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
///
/// {@example /example/optimize_example.dart lang=dart}
OptimizeResult lbfgs(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
  NDArray<Float64> Function(NDArray<Float64>)? jac,
  void Function(NDArray<Float64> x, NDArray<Float64> out)? jacInto,
  (double, NDArray<Float64>) Function(NDArray<Float64>)? funAndGrad,
  int m = 10,
  double gtol = 1e-5,
  int maxiter = 15000,
  NDArray<Float64>? out,
  NDArray<Float64>? outJac,
}) {
  if (x0.isDisposed) {
    throw StateError('Cannot execute lbfgs on disposed x0 array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write lbfgs result to a disposed out array.');
  }
  if (outJac != null && outJac.isDisposed) {
    throw StateError('Cannot write lbfgs gradient to a disposed outJac array.');
  }
  if (x0.shape.length != 1) {
    throw ArgumentError.value(
      x0.shape,
      'x0',
      'Must be a 1D vector for lbfgs (got shape ${x0.shape})',
    );
  }
  if (x0.size == 0) {
    throw ArgumentError.value(
      x0.size,
      'x0',
      'Must not be empty (initial vector x0 must not be empty)',
    );
  }
  if (out != null) {
    validateOutBuffer(out);
    if (out.shape.length != 1 ||
        out.shape[0] != x0.shape[0] ||
        out.dtype != DType.float64) {
      throw ArgumentError.value(
        out,
        'out',
        'Must be a 1D Float64 array of length ${x0.shape[0]} (got shape ${out.shape}, dtype ${out.dtype})',
      );
    }
  }
  if (outJac != null) {
    validateOutBuffer(outJac, 'outJac');
    if (outJac.shape.length != 1 ||
        outJac.shape[0] != x0.shape[0] ||
        outJac.dtype != DType.float64) {
      throw ArgumentError.value(
        outJac,
        'outJac',
        'Must be a 1D Float64 array of length ${x0.shape[0]} (got shape ${outJac.shape}, dtype ${outJac.dtype})',
      );
    }
  }
  if (out != null && outJac != null && sharesMemory(out, outJac)) {
    throw ArgumentError.value(
      outJac,
      'outJac',
      'Must not share memory with out',
    );
  }
  if (m <= 0) {
    throw ArgumentError.value(m, 'm', 'Must be strictly positive (got $m)');
  }
  if (gtol <= 0 || maxiter <= 0) {
    throw ArgumentError.value(
      [gtol, maxiter],
      'gtol, maxiter',
      'Must be positive (gtol and maxiter must be positive, got gtol=$gtol, maxiter=$maxiter)',
    );
  }
  checkBlasIntDim(x0.shape[0], 'n', 'lbfgs');
  checkBlasIntStride(x0.strides[0], 'incx', 'lbfgs');
  if (out != null) {
    checkBlasIntStride(out.strides[0], 'incout', 'lbfgs');
  }
  if (outJac != null) {
    checkBlasIntStride(outJac.strides[0], 'incoutJac', 'lbfgs');
  }

  return NDArray.scope(() {
    final n = x0.shape[0];
    final doubleBytes = ffi.sizeOf<ffi.Double>();
    final arenaMarker = ScratchArena.marker;
    try {
      final pXCurr = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pGCurr = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pQ = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pR = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pP = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXNext = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pGNext = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pS = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pY = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXTemp = ScratchArena.allocate<ffi.Double>(n * doubleBytes);

      final sPool = List<ffi.Pointer<ffi.Double>>.generate(
        m,
        (_) => ScratchArena.allocate<ffi.Double>(n * doubleBytes),
      );
      final yPool = List<ffi.Pointer<ffi.Double>>.generate(
        m,
        (_) => ScratchArena.allocate<ffi.Double>(n * doubleBytes),
      );

      int nfev = 0;
      final xArr = NDArray<Float64>.create([n], DType.float64);
      final xArrPtr = xArr.pointer.cast<ffi.Double>();
      final gBufArr = NDArray<Float64>.create([n], DType.float64);
      final gBufPtr = gBufArr.pointer.cast<ffi.Double>();
      final xPlus = NDArray<Float64>.create([n], DType.float64);
      final xPlusPtr = xPlus.pointer.cast<ffi.Double>();
      final xMinus = NDArray<Float64>.create([n], DType.float64);
      final xMinusPtr = xMinus.pointer.cast<ffi.Double>();

      (double, ffi.Pointer<ffi.Double>) evalFunAndGrad(
        ffi.Pointer<ffi.Double> pX,
        ffi.Pointer<ffi.Double> pGOut,
      ) {
        _dcopy(n, pX, xArrPtr);

        if (funAndGrad != null) {
          return NDArray.scope(() {
            final (fVal, gArr) = funAndGrad(xArr);
            nfev++;
            _copyArrayToPointer(n, gArr, pGOut);
            return (fVal, pGOut);
          });
        } else if (jacInto != null) {
          return NDArray.scope(() {
            final fVal = fun(xArr);
            nfev++;
            jacInto(xArr, gBufArr);
            _dcopy(n, gBufPtr, pGOut);
            return (fVal, pGOut);
          });
        } else if (jac != null) {
          return NDArray.scope(() {
            final fVal = fun(xArr);
            nfev++;
            final gArr = jac(xArr);
            _copyArrayToPointer(n, gArr, pGOut);
            return (fVal, pGOut);
          });
        } else {
          final fVal = NDArray.scope(() => fun(xArr));
          nfev++;
          const h = 1e-8;
          for (int i = 0; i < n; i++) {
            _dcopy(n, pX, pXTemp);
            pXTemp[i] += h;
            _dcopy(n, pXTemp, xPlusPtr);
            final fPlus = NDArray.scope(() => fun(xPlus));

            pXTemp[i] = pX[i] - h;
            _dcopy(n, pXTemp, xMinusPtr);
            final fMinus = NDArray.scope(() => fun(xMinus));

            pGOut[i] = (fPlus - fMinus) / (2.0 * h);
            nfev += 2;
          }
          return (fVal, pGOut);
        }
      }

      _copyArrayToPointer(n, x0, pXCurr);
      var (fCurr, _) = evalFunAndGrad(pXCurr, pGCurr);

      final pSHist = <ffi.Pointer<ffi.Double>>[];
      final pYHist = <ffi.Pointer<ffi.Double>>[];
      final rhoHist = <double>[];
      final alphaArr = List<double>.filled(m, 0.0);

      int nit = 0;
      bool success = false;
      String msg = 'Maximum iterations reached';

      for (int iter = 0; iter < maxiter; iter++) {
        nit++;

        double gNorm = 0.0;
        for (int i = 0; i < n; i++) {
          final absG = pGCurr[i].abs();
          if (absG > gNorm) gNorm = absG;
        }

        if (gNorm <= gtol) {
          success = true;
          msg = 'Optimization terminated successfully (gradient norm <= gtol).';
          break;
        }

        _dcopy(n, pGCurr, pQ);
        final k = pSHist.length;

        for (int i = k - 1; i >= 0; i--) {
          final sq = _ddot(n, pSHist[i], pQ);
          alphaArr[i] = rhoHist[i] * sq;
          _daxpy(n, -alphaArr[i], pYHist[i], pQ);
        }

        double gamma = 1.0;
        if (k > 0) {
          final sy = _ddot(n, pSHist[k - 1], pYHist[k - 1]);
          final yy = _ddot(n, pYHist[k - 1], pYHist[k - 1]);
          if (yy > 0) gamma = sy / yy;
        }

        _dcopy(n, pQ, pR);
        _dscal(n, gamma, pR);

        for (int i = 0; i < k; i++) {
          final yr = _ddot(n, pYHist[i], pR);
          final beta = rhoHist[i] * yr;
          _daxpy(n, alphaArr[i] - beta, pSHist[i], pR);
        }

        _dcopy(n, pR, pP);
        _dscal(n, -1.0, pP);

        double alphaStep = 1.0;
        const c1 = 1e-4;
        final dg = _ddot(n, pGCurr, pP);

        double fNext = fCurr;
        bool lineSearchSuccess = false;

        for (int ls = 0; ls < 25; ls++) {
          _dcopy(n, pXCurr, pXNext);
          _daxpy(n, alphaStep, pP, pXNext);

          final (fTry, _) = evalFunAndGrad(pXNext, pGNext);

          if (fTry <= fCurr + c1 * alphaStep * dg) {
            fNext = fTry;
            lineSearchSuccess = true;
            break;
          }
          alphaStep *= 0.5;
        }

        if (!lineSearchSuccess) {
          msg = 'Line search failed to find sufficient decrease';
          break;
        }

        _dcopy(n, pXNext, pS);
        _daxpy(n, -1.0, pXCurr, pS);

        _dcopy(n, pGNext, pY);
        _daxpy(n, -1.0, pGCurr, pY);

        final ys = _ddot(n, pY, pS);

        if (ys > 1e-10) {
          final ffi.Pointer<ffi.Double> targetS;
          final ffi.Pointer<ffi.Double> targetY;
          if (pSHist.length >= m) {
            targetS = pSHist.removeAt(0);
            targetY = pYHist.removeAt(0);
            rhoHist.removeAt(0);
          } else {
            targetS = sPool[pSHist.length];
            targetY = yPool[pYHist.length];
          }
          _dcopy(n, pS, targetS);
          _dcopy(n, pY, targetY);
          pSHist.add(targetS);
          pYHist.add(targetY);
          rhoHist.add(1.0 / ys);
        }

        _dcopy(n, pXNext, pXCurr);
        _dcopy(n, pGNext, pGCurr);
        fCurr = fNext;
      }

      final NDArray<Float64> resX;
      if (out != null) {
        _copyPointerToArray(n, pXCurr, out);
        resX = out;
      } else {
        resX = NDArray<Float64>.create([n], DType.float64);
        _dcopy(n, pXCurr, resX.pointer.cast<ffi.Double>());
        resX.detachToParentScope();
      }

      final NDArray<Float64> resJac;
      if (outJac != null) {
        _copyPointerToArray(n, pGCurr, outJac);
        resJac = outJac;
      } else {
        resJac = NDArray<Float64>.create([n], DType.float64);
        _dcopy(n, pGCurr, resJac.pointer.cast<ffi.Double>());
        resJac.detachToParentScope();
      }

      return (
        x: resX,
        fun: fCurr,
        success: success,
        nit: nit,
        nfev: nfev,
        message: msg,
        jac: resJac,
      );
    } finally {
      ScratchArena.reset(arenaMarker);
    }
  });
}

/// Unified entry point for multivariate scalar function minimization.
///
/// Dispatches to optimization algorithms defined in [MinimizeMethod]:
/// - [MinimizeMethod.nelderMead]: Derivative-free simplex method via [nelder_mead].
/// - [MinimizeMethod.lbfgs]: Quasi-Newton gradient method via [lbfgs].
///
/// It is an error if [x0], [out], or [outJac] is disposed or if [x0] is not a 1-dimensional vector.
///
/// ### References & Further Reading
/// - [SciPy minimize Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
///
/// {@example /example/optimize_example.dart lang=dart}
OptimizeResult minimize(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
  MinimizeMethod method = MinimizeMethod.nelderMead,
  NDArray<Float64> Function(NDArray<Float64>)? jac,
  void Function(NDArray<Float64> x, NDArray<Float64> out)? jacInto,
  (double, NDArray<Float64>) Function(NDArray<Float64>)? funAndGrad,
  double? tol,
  int? maxiter,
  NDArray<Float64>? out,
  NDArray<Float64>? outJac,
}) {
  switch (method) {
    case MinimizeMethod.nelderMead:
      return nelder_mead(
        fun,
        x0,
        fatol: tol ?? 1e-4,
        xatol: tol ?? 1e-4,
        maxiter: maxiter,
        out: out,
      );
    case MinimizeMethod.lbfgs:
      return lbfgs(
        fun,
        x0,
        jac: jac,
        jacInto: jacInto,
        funAndGrad: funAndGrad,
        gtol: tol ?? 1e-5,
        maxiter: maxiter ?? 15000,
        out: out,
        outJac: outJac,
      );
  }
}
