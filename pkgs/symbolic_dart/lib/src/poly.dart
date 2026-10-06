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

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'ffi/flint_bindings.dart' as fl;
import 'ffi/symengine_bindings.dart' as se;
import 'package:resource_scope/resource_scope.dart';
import 'expr.dart';

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'flint_free',
  assetId: 'package:symbolic_dart/flint',
)
external void _flintFree(ffi.Pointer<ffi.Void> ptr);

BigInt _fmpzToBigInt(ffi.Pointer<fl.fmpz> zPtr) {
  final cStr = fl.fmpz_get_str(ffi.nullptr, 10, zPtr);
  if (cStr == ffi.nullptr) {
    throw StateError('Failed to convert fmpz to string.');
  }
  try {
    return BigInt.parse(cStr.cast<Utf8>().toDartString());
  } finally {
    _flintFree(cStr.cast());
  }
}

void _bigIntToFmpz(ffi.Pointer<fl.fmpz> zPtr, BigInt val) {
  using((arena) {
    final cStr = val.toString().toNativeUtf8(allocator: arena);
    fl.fmpz_set_str(zPtr, cStr.cast(), 10);
  });
}

void _clearFmpz(ffi.Pointer<fl.fmpz> zPtr) {
  fl.fmpz_set_si(zPtr, 0);
  calloc.free(zPtr);
}

void _clearFmpq(ffi.Pointer<fl.fmpq> qPtr) {
  fl.fmpq_set_si(qPtr, 0, 1);
  calloc.free(qPtr);
}

BigInt _toBigInt(Object val, String name) {
  if (val is BigInt) return val;
  if (val is int) return BigInt.from(val);
  throw ArgumentError('$name must be int or BigInt, got ${val.runtimeType}');
}

final _fmpqPolyFinalizer = ffi.NativeFinalizer(
  ffi.Native.addressOf<
        ffi.NativeFunction<ffi.Void Function(ffi.Pointer<fl.fmpq_poly_struct>)>
      >(fl.fmpq_poly_clear)
      .cast<ffi.NativeFinalizerFunction>(),
);

/// Represents an exact univariate polynomial over the rational numbers Q[x]
/// backed by FLINT's `fmpq_poly` exact arithmetic engine.
///
/// Supports exact GCD, quotient/remainder division, derivative, integral, and
/// exact polynomial factorization over Q[x].
final class FlintRationalPoly implements ffi.Finalizable, ScopedResource {
  final ffi.Pointer<fl.fmpq_poly_struct> _ptr;
  bool _disposed = false;

  FlintRationalPoly._(this._ptr) {
    if (_ptr == ffi.nullptr) {
      throw StateError('Cannot wrap a nullptr fmpq_poly_struct');
    }
    _fmpqPolyFinalizer.attach(this, _ptr.cast(), detach: this);
    ResourceScope.track(this);
  }

  /// Allocates a new empty FLINT rational polynomial.
  static ffi.Pointer<fl.fmpq_poly_struct> _alloc() {
    final p = calloc<fl.fmpq_poly_struct>();
    fl.fmpq_poly_init(p);
    return p;
  }

  @override
  bool get isDisposed => _disposed;

  /// Explicitly releases the underlying native C++/C memory immediately.
  @override
  void dispose() {
    if (_disposed) return;
    _fmpqPolyFinalizer.detach(this);
    fl.fmpq_poly_clear(_ptr);
    calloc.free(_ptr);
    _disposed = true;
    ResourceScope.untrack(this);
  }

  @override
  FlintRationalPoly detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  FlintRationalPoly detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw StateError('This FlintRationalPoly has already been disposed.');
    }
  }

  ffi.Pointer<fl.fmpq_poly_struct> get pointer {
    _checkDisposed();
    return _ptr;
  }

  /// Creates a polynomial from integer coefficients in ascending order of degree:
  /// `coeffs[0] + coeffs[1]*x + ... + coeffs[d]*x^d`.
  factory FlintRationalPoly.fromIntCoefficients(List<int> coeffs) {
    final ptr = _alloc();
    for (var i = 0; i < coeffs.length; i++) {
      fl.fmpq_poly_set_coeff_si(ptr, i, coeffs[i]);
    }
    return FlintRationalPoly._(ptr);
  }

  /// Creates a polynomial from arbitrary-precision integer coefficients in ascending
  /// order of degree: `coeffs[0] + coeffs[1]*x + ... + coeffs[d]*x^d`.
  factory FlintRationalPoly.fromBigIntCoefficients(List<BigInt> coeffs) {
    final ptr = _alloc();
    final z = calloc<fl.fmpz>();
    try {
      for (var i = 0; i < coeffs.length; i++) {
        _bigIntToFmpz(z, coeffs[i]);
        fl.fmpq_poly_set_coeff_fmpz(ptr, i, z);
      }
    } finally {
      _clearFmpz(z);
    }
    return FlintRationalPoly._(ptr);
  }

  /// Creates a polynomial from rational coefficients `({numerator, denominator})`
  /// in ascending order of degree.
  ///
  /// The [numerator] and [denominator] can be either an [int] or a [BigInt].
  factory FlintRationalPoly.fromRationalCoefficients(
    List<({Object numerator, Object denominator})> coeffs,
  ) {
    final ptr = _alloc();
    final q = calloc<fl.fmpq>();
    final numPtr = calloc<fl.fmpz>();
    final denPtr = calloc<fl.fmpz>();
    try {
      for (var i = 0; i < coeffs.length; i++) {
        final c = coeffs[i];
        final n = _toBigInt(c.numerator, 'Numerator');
        final d = _toBigInt(c.denominator, 'Denominator');
        if (d == BigInt.zero) {
          throw ArgumentError('Polynomial coefficient denominator cannot be 0');
        }
        _bigIntToFmpz(numPtr, n);
        _bigIntToFmpz(denPtr, d);
        fl.fmpq_set_fmpz_frac(q, numPtr, denPtr);
        fl.fmpq_poly_set_coeff_fmpq(ptr, i, q);
      }
    } finally {
      _clearFmpz(numPtr);
      _clearFmpz(denPtr);
      _clearFmpq(q);
    }
    return FlintRationalPoly._(ptr);
  }

  /// Converts a univariate polynomial [expr] in [variable] into a
  /// [FlintRationalPoly] over Q[x].
  ///
  /// Throws [ArgumentError] if [expr] is not a univariate rational polynomial in [variable].
  factory FlintRationalPoly.fromExpr(Expr expr, Object variable) {
    final v = Expr.fromObject(variable);
    if (se.is_a_Symbol(v.pointer) == 0) {
      throw ArgumentError('Variable must be a Symbol, got $v');
    }
    return _exprNodeToPoly(expr, v);
  }

  static FlintRationalPoly _exprNodeToPoly(Expr e, Expr v) {
    final typeId = se.basic_get_type(e.pointer);
    switch (typeId) {
      case se.TypeID.SYMENGINE_INTEGER:
        final val = BigInt.parse(e.toString());
        return FlintRationalPoly.fromBigIntCoefficients([val]);
      case se.TypeID.SYMENGINE_RATIONAL:
        final nd = e.asNumerDenom();
        try {
          final numBig = BigInt.parse(nd.numerator.toString());
          final denBig = BigInt.parse(nd.denominator.toString());
          return FlintRationalPoly.fromRationalCoefficients([
            (numerator: numBig, denominator: denBig),
          ]);
        } finally {
          nd.numerator.dispose();
          nd.denominator.dispose();
        }
      case se.TypeID.SYMENGINE_SYMBOL:
        if (e == v) {
          return FlintRationalPoly.monomial(1);
        }
        throw ArgumentError(
          'Expression contains symbol $e other than target variable $v',
        );
      case se.TypeID.SYMENGINE_ADD:
        final children = e.args;
        var acc = FlintRationalPoly.zero();
        try {
          for (final child in children) {
            final termPoly = _exprNodeToPoly(child, v);
            final nextAcc = acc + termPoly;
            acc.dispose();
            termPoly.dispose();
            acc = nextAcc;
          }
          return acc;
        } finally {
          for (final child in children) {
            child.dispose();
          }
        }
      case se.TypeID.SYMENGINE_MUL:
        final children = e.args;
        var acc = FlintRationalPoly.one();
        try {
          for (final child in children) {
            final factorPoly = _exprNodeToPoly(child, v);
            final nextAcc = acc * factorPoly;
            acc.dispose();
            factorPoly.dispose();
            acc = nextAcc;
          }
          return acc;
        } finally {
          for (final child in children) {
            child.dispose();
          }
        }
      case se.TypeID.SYMENGINE_POW:
        final children = e.args;
        try {
          if (children.length != 2) {
            throw ArgumentError('Invalid Pow expression: $e');
          }
          final baseExpr = children[0];
          final expExpr = children[1];
          if (se.basic_get_type(expExpr.pointer) !=
              se.TypeID.SYMENGINE_INTEGER) {
            throw ArgumentError(
              'Polynomial exponent must be a non-negative integer, got $expExpr in $e',
            );
          }
          final expVal = int.tryParse(expExpr.toString());
          if (expVal == null || expVal < 0) {
            throw ArgumentError(
              'Polynomial exponent must be a non-negative integer, got $expExpr in $e',
            );
          }
          final basePoly = _exprNodeToPoly(baseExpr, v);
          try {
            return basePoly.pow(expVal);
          } finally {
            basePoly.dispose();
          }
        } finally {
          for (final child in children) {
            child.dispose();
          }
        }
      default:
        throw ArgumentError(
          'Expression is not a univariate rational polynomial in $v: $e',
        );
    }
  }

  /// The zero polynomial `0`.
  factory FlintRationalPoly.zero() =>
      FlintRationalPoly.fromIntCoefficients([0]);

  /// The constant polynomial `1`.
  factory FlintRationalPoly.one() => FlintRationalPoly.fromIntCoefficients([1]);

  /// The monomial polynomial `a * x^d`.
  factory FlintRationalPoly.monomial(int degree, {int coefficient = 1}) {
    if (degree < 0) {
      throw ArgumentError('Monomial degree must be non-negative.');
    }
    final ptr = _alloc();
    fl.fmpq_poly_set_coeff_si(ptr, degree, coefficient);
    return FlintRationalPoly._(ptr);
  }

  /// Returns the degree of this polynomial (`-1` for the zero polynomial).
  int get degree => pointer.ref.length - 1;

  /// Returns the number of terms (degree + 1, or 0 if zero polynomial).
  int get length => pointer.ref.length;

  /// Returns the numeric value of the coefficient at [degreeIndex] as a double.
  double getCoefficientAsDouble(int degreeIndex) {
    if (degreeIndex < 0 || degreeIndex >= length) return 0.0;
    final q = calloc<fl.fmpq>();
    try {
      fl.fmpq_poly_get_coeff_fmpq(q, pointer, degreeIndex);
      return fl.fmpq_get_d(q);
    } finally {
      _clearFmpq(q);
    }
  }

  /// Exact polynomial addition.
  FlintRationalPoly operator +(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_add(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial subtraction.
  FlintRationalPoly operator -(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_sub(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial multiplication.
  FlintRationalPoly operator *(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_mul(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial quotient (`this // other`).
  FlintRationalPoly operator /(FlintRationalPoly other) {
    if (other.length == 0) {
      throw StateError('Polynomial division by zero.');
    }
    final q = _alloc();
    final r = _alloc();
    fl.fmpq_poly_divrem(q, r, pointer, other.pointer);
    fl.fmpq_poly_clear(r);
    calloc.free(r);
    return FlintRationalPoly._(q);
  }

  /// Exact polynomial remainder (`this % other`).
  FlintRationalPoly operator %(FlintRationalPoly other) {
    if (other.length == 0) {
      throw StateError('Polynomial division by zero.');
    }
    final q = _alloc();
    final r = _alloc();
    fl.fmpq_poly_divrem(q, r, pointer, other.pointer);
    fl.fmpq_poly_clear(q);
    calloc.free(q);
    return FlintRationalPoly._(r);
  }

  /// Returns both quotient and remainder `(quotient, remainder)`.
  ({FlintRationalPoly quotient, FlintRationalPoly remainder}) divmod(
    FlintRationalPoly other,
  ) {
    if (other.length == 0) {
      throw StateError('Polynomial division by zero.');
    }
    final q = _alloc();
    final r = _alloc();
    fl.fmpq_poly_divrem(q, r, pointer, other.pointer);
    return (
      quotient: FlintRationalPoly._(q),
      remainder: FlintRationalPoly._(r),
    );
  }

  /// Computes the exact polynomial derivative `dP(x)/dx`.
  FlintRationalPoly derivative() {
    final res = _alloc();
    fl.fmpq_poly_derivative(res, pointer);
    return FlintRationalPoly._(res);
  }

  /// Computes the exact polynomial integral `\int P(x) dx` with zero constant.
  FlintRationalPoly integral() {
    final res = _alloc();
    fl.fmpq_poly_integral(res, pointer);
    return FlintRationalPoly._(res);
  }

  /// Computes the exact monic Greatest Common Divisor (GCD) over Q[x].
  FlintRationalPoly gcd(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_gcd(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial exponentiation `this^exponent` over Q[x].
  FlintRationalPoly pow(int exponent) {
    if (exponent < 0) {
      throw ArgumentError(
        'Polynomial exponent must be non-negative, got $exponent',
      );
    }
    final res = _alloc();
    fl.fmpq_poly_pow(res, pointer, exponent);
    return FlintRationalPoly._(res);
  }

  /// Shorthand operator for exact polynomial exponentiation `this^exponent`.
  FlintRationalPoly operator ^(int exponent) => pow(exponent);

  /// Exact polynomial composition `P(Q(x))` over Q[x].
  FlintRationalPoly compose(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_compose(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Evaluates this polynomial numerically at a real number [x].
  double evaluate(double x) {
    final len = length;
    if (len == 0) return 0.0;
    var res = 0.0;
    for (var i = len - 1; i >= 0; i--) {
      res = res * x + getCoefficientAsDouble(i);
    }
    return res;
  }

  /// Evaluates this polynomial exactly at a rational number `num / den`,
  /// returning the exact rational result `(numerator, denominator)`.
  ///
  /// [num] and [den] can be either an [int] or a [BigInt].
  ({BigInt numerator, BigInt denominator}) evaluateRational(
    Object num, [
    Object den = 1,
  ]) {
    final nVal = _toBigInt(num, 'Numerator');
    final dVal = _toBigInt(den, 'Denominator');
    if (dVal == BigInt.zero) throw ArgumentError('Denominator cannot be zero.');
    final qVal = calloc<fl.fmpq>();
    final qRes = calloc<fl.fmpq>();
    final nPtr = calloc<fl.fmpz>();
    final dPtr = calloc<fl.fmpz>();
    try {
      _bigIntToFmpz(nPtr, nVal);
      _bigIntToFmpz(dPtr, dVal);
      fl.fmpq_set_fmpz_frac(qVal, nPtr, dPtr);
      fl.fmpq_poly_evaluate_fmpq(qRes, pointer, qVal);
      fl.fmpq_numerator(nPtr, qRes);
      fl.fmpq_denominator(dPtr, qRes);
      return (numerator: _fmpzToBigInt(nPtr), denominator: _fmpzToBigInt(dPtr));
    } finally {
      _clearFmpz(nPtr);
      _clearFmpz(dPtr);
      _clearFmpq(qVal);
      _clearFmpq(qRes);
    }
  }

  /// Constructs the [n]-th Legendre polynomial `P_n(x)`.
  factory FlintRationalPoly.legendre(int n) {
    if (n < 0) throw ArgumentError('Legendre degree must be non-negative.');
    final ptr = _alloc();
    fl.fmpq_poly_legendre_p(ptr, n);
    return FlintRationalPoly._(ptr);
  }

  /// Constructs the [n]-th Laguerre polynomial `L_n(x)`.
  factory FlintRationalPoly.laguerre(int n) {
    if (n < 0) throw ArgumentError('Laguerre degree must be non-negative.');
    final ptr = _alloc();
    fl.fmpq_poly_laguerre_l(ptr, n);
    return FlintRationalPoly._(ptr);
  }

  /// Whether this polynomial is monic (leading coefficient is 1).
  bool get isMonic => fl.fmpq_poly_is_monic(pointer) != 0;

  /// Whether this polynomial is square-free (has no repeated roots).
  bool get isSquareFree => fl.fmpq_poly_is_squarefree(pointer) != 0;

  /// Returns a monic normalized copy of this polynomial `P(x) / a_n`.
  FlintRationalPoly makeMonic() {
    final res = _alloc();
    fl.fmpq_poly_make_monic(res, pointer);
    return FlintRationalPoly._(res);
  }

  /// Computes the exact resultant `Res(P, Q)` of this polynomial and [other].
  ({BigInt numerator, BigInt denominator}) resultant(FlintRationalPoly other) {
    final qRes = calloc<fl.fmpq>();
    final nPtr = calloc<fl.fmpz>();
    final dPtr = calloc<fl.fmpz>();
    try {
      fl.fmpq_poly_resultant(qRes, pointer, other.pointer);
      fl.fmpq_numerator(nPtr, qRes);
      fl.fmpq_denominator(dPtr, qRes);
      return (numerator: _fmpzToBigInt(nPtr), denominator: _fmpzToBigInt(dPtr));
    } finally {
      _clearFmpz(nPtr);
      _clearFmpz(dPtr);
      _clearFmpq(qRes);
    }
  }

  /// Truncated formal exponential power series `exp(P(x)) mod x^order`.
  FlintRationalPoly expSeries(int order) {
    final res = _alloc();
    fl.fmpq_poly_exp_series(res, pointer, order);
    return FlintRationalPoly._(res);
  }

  /// Truncated formal logarithmic power series `log(P(x)) mod x^order`.
  FlintRationalPoly logSeries(int order) {
    final res = _alloc();
    fl.fmpq_poly_log_series(res, pointer, order);
    return FlintRationalPoly._(res);
  }

  /// Truncated formal sine power series `sin(P(x)) mod x^order`.
  FlintRationalPoly sinSeries(int order) {
    final res = _alloc();
    fl.fmpq_poly_sin_series(res, pointer, order);
    return FlintRationalPoly._(res);
  }

  /// Truncated formal cosine power series `cos(P(x)) mod x^order`.
  FlintRationalPoly cosSeries(int order) {
    final res = _alloc();
    fl.fmpq_poly_cos_series(res, pointer, order);
    return FlintRationalPoly._(res);
  }

  /// Converts this FLINT exact polynomial into a SymEngine [Expr] using
  /// the symbolic variable [x].
  Expr toExpr(Expr x) {
    var sum = Expr.zero;
    final q = calloc<fl.fmpq>();
    final numPtr = calloc<fl.fmpz>();
    final denPtr = calloc<fl.fmpz>();
    try {
      final len = length;
      for (var i = 0; i < len; i++) {
        fl.fmpq_poly_get_coeff_fmpq(q, pointer, i);
        fl.fmpq_numerator(numPtr, q);
        fl.fmpq_denominator(denPtr, q);
        final n = _fmpzToBigInt(numPtr);
        final d = _fmpzToBigInt(denPtr);
        if (n != BigInt.zero) {
          final termCoeff = d == BigInt.one
              ? Expr.bigInt(n)
              : Expr.bigInt(n) / Expr.bigInt(d);
          final Expr term;
          if (i == 0) {
            term = termCoeff;
          } else if (i == 1) {
            term = termCoeff * x;
            termCoeff.dispose();
          } else {
            final xPow = x ^ i;
            term = termCoeff * xPow;
            xPow.dispose();
            termCoeff.dispose();
          }
          final prevSum = sum;
          sum = prevSum + term;
          prevSum.dispose();
          term.dispose();
        }
      }
      return sum;
    } finally {
      _clearFmpz(numPtr);
      _clearFmpz(denPtr);
      _clearFmpq(q);
    }
  }

  /// Performs exact polynomial factorization over Q[x] into monic/primitive
  /// irreducible factors and rational content:
  /// `P(x) = content * \prod_i factor_i^{exponent_i}`.
  ({
    ({BigInt numerator, BigInt denominator}) content,
    List<({FlintRationalPoly factor, int exponent})> factors,
  })
  factor() {
    var D = BigInt.one;
    final q = calloc<fl.fmpq>();
    final numPtr = calloc<fl.fmpz>();
    final denPtr = calloc<fl.fmpz>();
    try {
      final len = length;
      for (var i = 0; i < len; i++) {
        fl.fmpq_poly_get_coeff_fmpq(q, pointer, i);
        fl.fmpq_denominator(denPtr, q);
        final dVal = _fmpzToBigInt(denPtr);
        if (dVal != BigInt.zero) {
          D = (D * dVal) ~/ D.gcd(dVal);
        }
      }

      final zPoly = calloc<fl.fmpz_poly_struct>();
      fl.fmpz_poly_init(zPoly);
      try {
        for (var i = 0; i < len; i++) {
          fl.fmpq_poly_get_coeff_fmpq(q, pointer, i);
          fl.fmpq_numerator(numPtr, q);
          fl.fmpq_denominator(denPtr, q);
          final nVal = _fmpzToBigInt(numPtr);
          final dVal = _fmpzToBigInt(denPtr);
          final scaledCoeff = (nVal * D) ~/ dVal;
          _bigIntToFmpz(numPtr, scaledCoeff);
          fl.fmpz_poly_set_coeff_fmpz(zPoly, i, numPtr);
        }

        final fac = calloc<fl.fmpz_poly_factor_struct>();
        fl.fmpz_poly_factor_init(fac);
        try {
          fl.fmpz_poly_factor(fac, zPoly);

          fl.fmpz_poly_factor_get_fmpz(numPtr, fac);
          final intContent = _fmpzToBigInt(numPtr);

          final cGcd = intContent.abs().gcd(D);
          final contentNumerator = intContent ~/ cGcd;
          final contentDenominator = D ~/ cGcd;

          final numFactors = fac.ref.num;
          final factorList = <({FlintRationalPoly factor, int exponent})>[];

          final factorZPoly = calloc<fl.fmpz_poly_struct>();
          fl.fmpz_poly_init(factorZPoly);
          try {
            for (var i = 0; i < numFactors; i++) {
              fl.fmpz_poly_factor_get_fmpz_poly(factorZPoly, fac, i);
              final exp = fac.ref.exp[i];

              final factorQPoly = FlintRationalPoly._(
                FlintRationalPoly._alloc(),
              );
              final facLen = factorZPoly.ref.length;
              for (var j = 0; j < facLen; j++) {
                fl.fmpz_poly_get_coeff_fmpz(numPtr, factorZPoly, j);
                fl.fmpq_poly_set_coeff_fmpz(factorQPoly.pointer, j, numPtr);
              }
              factorList.add((factor: factorQPoly, exponent: exp));
            }
          } finally {
            fl.fmpz_poly_clear(factorZPoly);
            calloc.free(factorZPoly);
          }

          return (
            content: (
              numerator: contentNumerator,
              denominator: contentDenominator,
            ),
            factors: factorList,
          );
        } finally {
          fl.fmpz_poly_factor_clear(fac);
          calloc.free(fac);
        }
      } finally {
        fl.fmpz_poly_clear(zPoly);
        calloc.free(zPoly);
      }
    } finally {
      _clearFmpz(numPtr);
      _clearFmpz(denPtr);
      _clearFmpq(q);
    }
  }

  /// Returns the exact rational coefficient at [degreeIndex] as `(numerator, denominator)`.
  ({BigInt numerator, BigInt denominator}) getCoefficientRational(
    int degreeIndex,
  ) {
    if (degreeIndex < 0 || degreeIndex >= length) {
      return (numerator: BigInt.zero, denominator: BigInt.one);
    }
    final q = calloc<fl.fmpq>();
    final numPtr = calloc<fl.fmpz>();
    final denPtr = calloc<fl.fmpz>();
    try {
      fl.fmpq_poly_get_coeff_fmpq(q, pointer, degreeIndex);
      fl.fmpq_numerator(numPtr, q);
      fl.fmpq_denominator(denPtr, q);
      final n = _fmpzToBigInt(numPtr);
      final d = _fmpzToBigInt(denPtr);
      return (numerator: n, denominator: d == BigInt.zero ? BigInt.one : d);
    } finally {
      _clearFmpz(numPtr);
      _clearFmpz(denPtr);
      _clearFmpq(q);
    }
  }

  /// Formats this polynomial as a LaTeX mathematical string in descending order of powers.
  String toLatex([String variable = 'x']) {
    _checkDisposed();
    final d = degree;
    if (d < 0) return '0';
    final terms = <String>[];
    for (var k = d; k >= 0; k--) {
      final c = getCoefficientRational(k);
      if (c.numerator == BigInt.zero) continue;
      final isNeg = c.numerator < BigInt.zero;
      final absNum = c.numerator.abs();
      final den = c.denominator;

      String coeffStr;
      if (den == BigInt.one) {
        if (absNum == BigInt.one && k > 0) {
          coeffStr = '';
        } else {
          coeffStr = '$absNum';
        }
      } else {
        coeffStr =
            r'\frac{'
            '$absNum}{$den}';
      }

      String term;
      if (k == 0) {
        term = coeffStr.isEmpty ? '1' : coeffStr;
      } else if (k == 1) {
        term = '$coeffStr$variable';
      } else {
        term = '$coeffStr$variable^{$k}';
      }

      if (terms.isEmpty) {
        terms.add(isNeg ? '-$term' : term);
      } else {
        terms.add(isNeg ? ' - $term' : ' + $term');
      }
    }
    return terms.isEmpty ? '0' : terms.join();
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! FlintRationalPoly) return false;
    if (_disposed || other._disposed) return false;
    return fl.fmpq_poly_equal(pointer, other.pointer) != 0;
  }

  @override
  int get hashCode {
    if (_disposed) return identityHashCode(this);
    final d = degree;
    if (d < 0) return 0;
    var hash = d.hashCode;
    for (var i = 0; i <= d; i++) {
      final coeff = getCoefficientRational(i);
      hash = Object.hash(hash, coeff.numerator, coeff.denominator);
    }
    return hash;
  }

  @override
  String toString() {
    _checkDisposed();
    final cStr = fl.fmpq_poly_get_str(pointer);
    if (cStr == ffi.nullptr) return '0';
    try {
      return cStr.cast<Utf8>().toDartString();
    } finally {
      _flintFree(cStr.cast());
    }
  }
}

/// Extension adding LaTeX rendering to [FlintRationalPoly.factor] factorization records.
extension PolyFactorizationLatex
    on
        ({
          ({BigInt numerator, BigInt denominator}) content,
          List<({FlintRationalPoly factor, int exponent})> factors,
        }) {
  /// Formats this factorization record as a LaTeX product: `c * (f1)^e1 * (f2)^e2`.
  String toLatex([String variable = 'x']) {
    final sb = StringBuffer();
    final c = content;
    if (c.numerator != BigInt.one ||
        c.denominator != BigInt.one ||
        factors.isEmpty) {
      if (c.denominator == BigInt.one) {
        if (c.numerator == -BigInt.one && factors.isNotEmpty) {
          sb.write('-');
        } else {
          sb.write('${c.numerator}');
        }
      } else {
        sb.write(
          r'\frac{'
          '${c.numerator}}{${c.denominator}}',
        );
      }
    }
    for (final item in factors) {
      sb.write(r'\left(');
      sb.write(item.factor.toLatex(variable));
      sb.write(r'\right)');
      if (item.exponent > 1) {
        sb.write('^{${item.exponent}}');
      }
    }
    return sb.toString();
  }
}
