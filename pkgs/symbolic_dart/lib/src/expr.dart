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
import 'package:meta/meta.dart';
import 'ffi/symengine_bindings.dart' as se;
import 'ffi/flint_bindings.dart' as fl;
import 'package:resource_scope/resource_scope.dart';
import 'expr.dart' as top;
import 'matrix.dart';

/// The NativeFinalizer that frees C heap-allocated `basic_struct` instances
/// when Dart garbage collects an `Expr` object.
final _basicFinalizer = ffi.NativeFinalizer(
  ffi.Native.addressOf<
        ffi.NativeFunction<ffi.Void Function(ffi.Pointer<se.basic_struct>)>
      >(se.basic_free_heap)
      .cast<ffi.NativeFinalizerFunction>(),
);

final bool _flintLoaded = _initFlint();
bool _initFlint() {
  try {
    final p = calloc<fl.fmpq_poly_struct>();
    fl.fmpq_poly_init(p);
    fl.fmpq_poly_clear(p);
    calloc.free(p);
  } catch (_) {}
  return true;
}

/// A symbolic expression backed by the SymEngine native C++ CAS engine.
///
/// Use [Expr] to build algebraic formulas, perform symbolic differentiation,
/// expand polynomial/trigonometric expressions, substitute values, and compile
/// into vectorized numerical evaluators.
final class Expr implements ffi.Finalizable, ScopedResource {
  final ffi.Pointer<se.basic_struct> _ptr;
  bool _disposed = false;

  /// Internal constructor wrapping an allocated [se.basic_struct] pointer.
  /// Automatically registers a [NativeFinalizer] to release C memory.
  Expr._(this._ptr) {
    if (_ptr == ffi.nullptr) {
      throw StateError('Cannot wrap a nullptr in Expr');
    }
    _flintLoaded;
    _basicFinalizer.attach(this, _ptr.cast(), detach: this);
    ResourceScope.track(this);
  }

  /// Package-internal helper to wrap an allocated native [se.basic_struct] pointer.
  @internal
  factory Expr.fromPointer(ffi.Pointer<se.basic_struct> ptr) => Expr._(ptr);

  @override
  bool get isDisposed => _disposed;

  /// Explicitly releases the underlying native C++ memory immediately.
  ///
  /// After calling [dispose], using this expression will throw a [StateError].
  @override
  void dispose() {
    if (_disposed) return;
    _basicFinalizer.detach(this);
    se.basic_free_heap(_ptr);
    _disposed = true;
    ResourceScope.untrack(this);
  }

  @override
  Expr detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  Expr detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw StateError('This Expr has already been disposed.');
    }
  }

  /// Returns the underlying native FFI pointer (`basic_struct*`).
  ffi.Pointer<se.basic_struct> get pointer {
    _checkDisposed();
    return _ptr;
  }

  // ===========================================================================
  // CONSTANTS & FACTORIES
  // ===========================================================================

  /// Creates a symbolic integer with value [val].
  factory Expr.integer(int val) {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.integer_set_si(ptr, val);
    return Expr._(ptr);
  }

  /// Creates a symbolic arbitrary-precision integer from [val].
  factory Expr.bigInt(BigInt val) {
    return Expr.parse(val.toString());
  }

  /// Creates a symbolic floating-point double with value [val].
  factory Expr.real(double val) {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.real_double_set_d(ptr, val);
    return Expr._(ptr);
  }

  /// Creates a symbolic rational number [numerator]/[denominator].
  ///
  /// It is an error if [denominator] is zero.
  factory Expr.rational(int numerator, int denominator) {
    if (denominator == 0) {
      throw ArgumentError('Rational denominator cannot be zero.');
    }
    return Expr.integer(numerator) / Expr.integer(denominator);
  }

  /// Creates a symbolic variable (symbol) with name [name].
  factory Expr.symbol(String name) {
    if (name.isEmpty) {
      throw ArgumentError('Symbol name cannot be empty.');
    }
    _flintLoaded;
    final ptr = se.basic_new_heap();
    using((arena) {
      final cName = name.toNativeUtf8(allocator: arena);
      se.symbol_set(ptr, cName.cast());
    });
    return Expr._(ptr);
  }

  /// Parses a mathematical formula string (e.g., `"sin(x) + x^2"`) into an [Expr].
  factory Expr.parse(String formula) {
    final ptr = se.basic_new_heap();
    using((arena) {
      final cStr = formula.toNativeUtf8(allocator: arena);
      final res = se.basic_parse2(
        ptr,
        cStr.cast(),
        1,
      ); // convert_xor = 1 (^ is pow)
      if (res != se.symengine_exceptions_t.SYMENGINE_NO_EXCEPTION) {
        se.basic_free_heap(ptr);
        throw FormatException('Failed to parse formula: "$formula"');
      }
    });
    return Expr._(ptr);
  }

  /// Converts an object ([Expr], [num], [int], [double], [BigInt], or [String])
  /// into an [Expr].
  factory Expr.fromObject(Object obj) {
    if (obj is Expr) return obj;
    if (obj is int) return Expr.integer(obj);
    if (obj is double) return Expr.real(obj);
    if (obj is BigInt) return Expr.bigInt(obj);
    if (obj is String) return Expr.symbol(obj);
    throw ArgumentError(
      'Unsupported type for symbolic Expr: ${obj.runtimeType}',
    );
  }

  /// The symbolic constant `0`.
  static Expr get zero {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_zero(ptr);
    return Expr._(ptr);
  }

  /// The symbolic constant `1`.
  static Expr get one {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_one(ptr);
    return Expr._(ptr);
  }

  /// The symbolic constant `-1`.
  static Expr get minusOne {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_minus_one(ptr);
    return Expr._(ptr);
  }

  /// The imaginary unit `I = sqrt(-1)`.
  static Expr get i {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_I(ptr);
    return Expr._(ptr);
  }

  /// Archimedes' constant `pi = 3.14159...`.
  static Expr get pi {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_pi(ptr);
    return Expr._(ptr);
  }

  /// Euler's number `e = 2.71828...`.
  static Expr get e {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_E(ptr);
    return Expr._(ptr);
  }

  /// Euler-Mascheroni constant `gamma = 0.577215...`.
  static Expr get eulerGamma {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_EulerGamma(ptr);
    return Expr._(ptr);
  }

  /// Catalan's constant `G = 0.915965...`.
  static Expr get catalan {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_Catalan(ptr);
    return Expr._(ptr);
  }

  /// The Golden Ratio `phi = (1 + sqrt(5)) / 2 = 1.618033...`.
  static Expr get goldenRatio {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_GoldenRatio(ptr);
    return Expr._(ptr);
  }

  /// Positive infinity `+oo`.
  static Expr get infinity {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_infinity(ptr);
    return Expr._(ptr);
  }

  /// Negative infinity `-oo`.
  static Expr get negInfinity {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_neginfinity(ptr);
    return Expr._(ptr);
  }

  /// Complex infinity `zoo`.
  static Expr get complexInfinity {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_complex_infinity(ptr);
    return Expr._(ptr);
  }

  /// Symbolic Not-a-Number `NaN`.
  static Expr get nan {
    _flintLoaded;
    final ptr = se.basic_new_heap();
    se.basic_const_nan(ptr);
    return Expr._(ptr);
  }

  /// Computes the modular multiplicative inverse `a^-1 mod m`.
  static Expr modInverse(Object a, Object m) {
    final ea = Expr.fromObject(a);
    final em = Expr.fromObject(m);
    final res = se.basic_new_heap();
    se.ntheory_mod_inverse(res, ea.pointer, em.pointer);
    return Expr._(res);
  }

  // ===========================================================================
  // ARITHMETIC OPERATORS
  // ===========================================================================

  /// Adds [other] to this expression.
  Expr operator +(Object other) {
    final b = Expr.fromObject(other);
    final res = se.basic_new_heap();
    se.basic_add(res, pointer, b.pointer);
    return Expr._(res);
  }

  /// Subtracts [other] from this expression.
  Expr operator -(Object other) {
    final b = Expr.fromObject(other);
    final res = se.basic_new_heap();
    se.basic_sub(res, pointer, b.pointer);
    return Expr._(res);
  }

  /// Multiplies this expression by [other].
  Expr operator *(Object other) {
    final b = Expr.fromObject(other);
    final res = se.basic_new_heap();
    se.basic_mul(res, pointer, b.pointer);
    return Expr._(res);
  }

  /// Divides this expression by [other].
  Expr operator /(Object other) {
    final b = Expr.fromObject(other);
    final res = se.basic_new_heap();
    se.basic_div(res, pointer, b.pointer);
    return Expr._(res);
  }

  /// Negates this expression (`-this`).
  Expr operator -() {
    final res = se.basic_new_heap();
    se.basic_neg(res, pointer);
    return Expr._(res);
  }

  /// Raises this expression to the power of [exponent].
  ///
  /// Note: The `^` operator in Dart syntax is overloaded for symbolic exponentiation.
  Expr operator ^(Object exponent) => pow(exponent);

  /// Raises this expression to the power of [exponent].
  Expr pow(Object exponent) {
    final b = Expr.fromObject(exponent);
    final res = se.basic_new_heap();
    se.basic_pow(res, pointer, b.pointer);
    return Expr._(res);
  }

  // ===========================================================================
  // CALCULUS & SYMBOLIC MANIPULATION
  // ===========================================================================

  /// Computes the symbolic derivative of this expression with respect to [variable].
  ///
  /// [order] specifies the order of differentiation (defaults to 1).
  ///
  /// Throws [ArgumentError] if [order] is negative or if [variable] is not a symbol.
  Expr diff(Object variable, [int order = 1]) {
    if (order < 0) {
      throw ArgumentError(
        'Differentiation order must be non-negative, got $order.',
      );
    }
    if (order == 0) return this;

    final sym = Expr.fromObject(variable);
    if (se.is_a_Symbol(sym.pointer) == 0) {
      throw ArgumentError('Differentiation variable must be a Symbol.');
    }

    var current = this;
    for (var i = 0; i < order; i++) {
      final res = se.basic_new_heap();
      final err = se.basic_diff(res, current.pointer, sym.pointer);
      if (err != se.symengine_exceptions_t.SYMENGINE_NO_EXCEPTION) {
        se.basic_free_heap(res);
        if (current != this) {
          current.dispose();
        }
        throw StateError('Symbolic differentiation failed.');
      }
      final next = Expr._(res);
      if (current != this) {
        current.dispose();
      }
      current = next;
    }
    return current;
  }

  /// Computes the truncated Taylor series expansion of this expression
  /// with respect to [variable] around the point [at] up to degree [order] - 1.
  ///
  /// Evaluates $\sum_{k=0}^{\text{order}-1} \frac{f^{(k)}(a)}{k!} (x - a)^k$.
  Expr series(Object variable, {Object at = 0, int order = 6}) {
    if (order < 0) {
      throw ArgumentError('Series order must be non-negative, got $order.');
    }
    if (order == 0) return Expr.zero;

    final x = Expr.fromObject(variable);
    if (se.is_a_Symbol(x.pointer) == 0) {
      throw ArgumentError('Series expansion variable must be a Symbol.');
    }
    final a = Expr.fromObject(at);
    final dx = a.isZero ? x : (x - a);

    var result = Expr.zero;
    var currentDeriv = this;
    var factorial = BigInt.one;

    for (var k = 0; k < order; k++) {
      if (k > 0) {
        factorial *= BigInt.from(k);
        final nextDeriv = currentDeriv.diff(x);
        if (currentDeriv != this) {
          currentDeriv.dispose();
        }
        currentDeriv = nextDeriv;
      }

      final coeffAtA = currentDeriv.subs({x: a});
      if (!coeffAtA.isZero) {
        Expr term;
        if (k == 0) {
          term = coeffAtA;
        } else {
          final coeffDiv = (factorial == BigInt.one)
              ? coeffAtA
              : (coeffAtA / Expr.bigInt(factorial));
          term = (k == 1) ? (coeffDiv * dx) : (coeffDiv * dx.pow(k));
        }
        result = (result.isZero) ? term : (result + term);
      }
    }

    if (currentDeriv != this) {
      currentDeriv.dispose();
    }

    return result;
  }

  /// Computes the gradient of this scalar expression with respect to [variables].
  ///
  /// Returns an `n x 1` column vector [SymbolicMatrix] of partial derivatives
  /// `[diff(v_1), ..., diff(v_n)]`.
  SymbolicMatrix gradient(List<Object> variables) {
    final n = variables.length;
    final mat = SymbolicMatrix.zeros(n, 1);
    for (var i = 0; i < n; i++) {
      final d = diff(variables[i]);
      try {
        mat.setCell(i, 0, d);
      } finally {
        d.dispose();
      }
    }
    return mat;
  }

  /// Computes the Hessian matrix of second-order partial derivatives of this
  /// expression with respect to [variables].
  ///
  /// Returns an `n x n` [SymbolicMatrix] where entry `(i, j)` is
  /// `diff(variables[i]).diff(variables[j])`.
  SymbolicMatrix hessian(List<Object> variables) {
    final n = variables.length;
    final mat = SymbolicMatrix.zeros(n, n);
    for (var i = 0; i < n; i++) {
      final d1 = diff(variables[i]);
      try {
        for (var j = 0; j < n; j++) {
          final d2 = d1.diff(variables[j]);
          try {
            mat.setCell(i, j, d2);
          } finally {
            d2.dispose();
          }
        }
      } finally {
        d1.dispose();
      }
    }
    return mat;
  }

  /// Performs algebraic expansion (e.g., `(x + 1)^2` becomes `x^2 + 2*x + 1`).
  Expr expand() {
    final res = se.basic_new_heap();
    se.basic_expand(res, pointer);
    return Expr._(res);
  }

  /// Substitutes symbols or sub-expressions according to [substitutions].
  ///
  /// Keys and values in [substitutions] can be [Symbol], [Expr], [num], or [String].
  Expr subs(Map<Object, Object> substitutions) {
    if (substitutions.isEmpty) return this;

    final map = se.mapbasicbasic_new();
    try {
      for (final entry in substitutions.entries) {
        final key = Expr.fromObject(entry.key);
        final val = Expr.fromObject(entry.value);
        se.mapbasicbasic_insert(map, key.pointer, val.pointer);
      }
      final res = se.basic_new_heap();
      final err = se.basic_subs(res, pointer, map);
      if (err != se.symengine_exceptions_t.SYMENGINE_NO_EXCEPTION) {
        se.basic_free_heap(res);
        throw StateError('Symbolic substitution failed.');
      }
      return Expr._(res);
    } finally {
      se.mapbasicbasic_free(map);
    }
  }

  /// Separates this fraction or expression into numerator and denominator.
  ({Expr numerator, Expr denominator}) asNumerDenom() {
    _checkDisposed();
    final numPtr = se.basic_new_heap();
    final denPtr = se.basic_new_heap();
    se.basic_as_numer_denom(numPtr, denPtr, pointer);
    return (numerator: Expr._(numPtr), denominator: Expr._(denPtr));
  }

  /// Finds the finite exact roots of a polynomial [equation] in [variable].
  ///
  /// Returns a set of exact roots (e.g. integer, rational, or algebraic expressions).
  static Set<Expr> solvePoly(Object equation, Object variable) {
    final eqExpr = Expr.fromObject(equation);
    final varExpr = Expr.fromObject(variable);
    final resSet = se.setbasic_new();
    try {
      se.basic_solve_poly(resSet, eqExpr.pointer, varExpr.pointer);
      return _fromSetBasic(resSet);
    } finally {
      se.setbasic_free(resSet);
    }
  }

  /// Solves a system of exact linear equations $\vec{f}(\vec{x}) = \vec{0}$
  /// for the specified symbolic [variables].
  static List<Expr> solveLinearSystem(
    List<Object> equations,
    List<Object> variables,
  ) {
    final sysVec = se.vecbasic_new();
    final symVec = se.vecbasic_new();
    final solVec = se.vecbasic_new();
    try {
      for (final eq in equations) {
        se.vecbasic_push_back(sysVec, Expr.fromObject(eq).pointer);
      }
      for (final v in variables) {
        se.vecbasic_push_back(symVec, Expr.fromObject(v).pointer);
      }
      se.vecbasic_linsolve(solVec, sysVec, symVec);
      return _fromVecBasic(solVec);
    } finally {
      se.vecbasic_free(sysVec);
      se.vecbasic_free(symVec);
      se.vecbasic_free(solVec);
    }
  }

  static Set<Expr> _fromSetBasic(ffi.Pointer<se.CSetBasic> setPtr) {
    final len = se.setbasic_size(setPtr);
    final set = <Expr>{};
    for (var i = 0; i < len; i++) {
      final ptr = se.basic_new_heap();
      se.setbasic_get(setPtr, i, ptr);
      set.add(Expr._(ptr));
    }
    return set;
  }

  static List<Expr> _fromVecBasic(ffi.Pointer<se.CVecBasic> vecPtr) {
    final len = se.vecbasic_size(vecPtr);
    final list = <Expr>[];
    for (var i = 0; i < len; i++) {
      final ptr = se.basic_new_heap();
      se.vecbasic_get(vecPtr, i, ptr);
      list.add(Expr._(ptr));
    }
    return list;
  }

  // ===========================================================================
  // FLUID METHOD CHAINING FOR MATHEMATICAL FUNCTIONS
  // ===========================================================================

  Expr sin() => top.sin(this);
  Expr cos() => top.cos(this);
  Expr tan() => top.tan(this);
  Expr exp() => top.exp(this);
  Expr asin() => top.asin(this);
  Expr acos() => top.acos(this);
  Expr atan() => top.atan(this);
  Expr atan2(Object other) => top.atan2(this, other);
  Expr csc() => top.csc(this);
  Expr sec() => top.sec(this);
  Expr cot() => top.cot(this);

  Expr sinh() => top.sinh(this);
  Expr cosh() => top.cosh(this);
  Expr tanh() => top.tanh(this);
  Expr asinh() => top.asinh(this);
  Expr acosh() => top.acosh(this);
  Expr atanh() => top.atanh(this);

  Expr abs() => top.abs(this);
  Expr sqrt() => top.sqrt(this);
  Expr cbrt() => top.cbrt(this);
  Expr floor() => top.floor(this);
  Expr ceil() => top.ceil(this);
  Expr log() => top.log(this);
  Expr erf() => top.erf(this);
  Expr erfc() => top.erfc(this);
  Expr gamma() => top.gamma(this);
  Expr lambertw() => top.lambertw(this);
  Expr zeta() => top.zeta(this);

  Expr kroneckerDelta(Object other) => top.kroneckerDelta(this, other);
  Expr gcd(Object other) => top.gcd(this, other);
  Expr lcm(Object other) => top.lcm(this, other);

  /// Returns the immediate child operands (arguments) of this expression.
  List<Expr> get args {
    final vec = se.vecbasic_new();
    try {
      se.basic_get_args(pointer, vec);
      final len = se.vecbasic_size(vec);
      final list = <Expr>[];
      for (var i = 0; i < len; i++) {
        final item = se.basic_new_heap();
        se.vecbasic_get(vec, i, item);
        list.add(Expr._(item));
      }
      return list;
    } finally {
      se.vecbasic_free(vec);
    }
  }

  /// Returns all free symbols referenced inside this expression.
  Set<Expr> get freeSymbols {
    final set = se.setbasic_new();
    try {
      se.basic_free_symbols(pointer, set);
      final len = se.setbasic_size(set);
      final symbols = <Expr>{};
      for (var i = 0; i < len; i++) {
        final item = se.basic_new_heap();
        se.setbasic_get(set, i, item);
        symbols.add(Expr._(item));
      }
      return symbols;
    } finally {
      se.setbasic_free(set);
    }
  }

  // ===========================================================================
  // PREPARING FOR NDARRAY / NUMERICAL EVALUATION
  // ===========================================================================

  /// Whether this expression evaluates to exactly zero.
  bool get isZero =>
      se.is_a_Number(pointer) != 0 && se.number_is_zero(pointer) != 0;

  /// Whether this expression evaluates to a positive real number.
  bool get isPositive =>
      se.is_a_Number(pointer) != 0 && se.number_is_positive(pointer) != 0;

  /// Whether this expression evaluates to a negative real number.
  bool get isNegative =>
      se.is_a_Number(pointer) != 0 && se.number_is_negative(pointer) != 0;

  /// Whether this expression contains any complex numbers.
  bool get isComplex =>
      se.is_a_Number(pointer) != 0 && se.number_is_complex(pointer) != 0;

  /// Whether this expression contains the specific symbol [sym].
  bool hasSymbol(Object sym) {
    final s = Expr.fromObject(sym);
    return se.basic_has_symbol(pointer, s.pointer) != 0;
  }

  /// Evaluates a numeric expression as a double.
  double get asDouble {
    _checkDisposed();
    if (se.is_a_RealDouble(pointer) != 0) {
      return se.real_double_get_d(pointer);
    }
    if (se.is_a_Integer(pointer) != 0) {
      return se.integer_get_si(pointer).toDouble();
    }
    final res = se.basic_new_heap();
    try {
      se.basic_evalf(res, pointer, 53, 1);
      return se.real_double_get_d(res);
    } finally {
      se.basic_free_heap(res);
    }
  }

  // ===========================================================================
  // STRING & FORMAT PRINTERS
  // ===========================================================================

  /// Returns standard human-readable infix representation.
  @override
  String toString() {
    _checkDisposed();
    final cStr = se.basic_str(pointer);
    if (cStr == ffi.nullptr) return '<null Expr>';
    try {
      return cStr.cast<Utf8>().toDartString();
    } finally {
      se.basic_str_free(cStr);
    }
  }

  /// Returns LaTeX formatting for mathematical rendering.
  String toLatex() {
    _checkDisposed();
    final cStr = se.basic_str_latex(pointer);
    if (cStr == ffi.nullptr) return '';
    try {
      return cStr.cast<Utf8>().toDartString();
    } finally {
      se.basic_str_free(cStr);
    }
  }

  /// Returns C source code for compiling into native kernels.
  String toCCode() {
    _checkDisposed();
    final cStr = se.basic_str_ccode(pointer);
    if (cStr == ffi.nullptr) return '';
    try {
      return cStr.cast<Utf8>().toDartString();
    } finally {
      se.basic_str_free(cStr);
    }
  }

  /// Returns JavaScript source code for compiling into JS/Web kernels.
  String toJSCode() {
    _checkDisposed();
    final cStr = se.basic_str_jscode(pointer);
    if (cStr == ffi.nullptr) return '';
    try {
      return cStr.cast<Utf8>().toDartString();
    } finally {
      se.basic_str_free(cStr);
    }
  }

  /// Returns MathML XML markup representation.
  String toMathML() {
    _checkDisposed();
    final cStr = se.basic_str_mathml(pointer);
    if (cStr == ffi.nullptr) return '';
    try {
      return cStr.cast<Utf8>().toDartString();
    } finally {
      se.basic_str_free(cStr);
    }
  }

  // ===========================================================================
  // EQUALITY & HASH
  // ===========================================================================

  /// Checks structural equality between two expressions.
  bool eq(Expr other) => se.basic_eq(pointer, other.pointer) != 0;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! Expr) return false;
    return eq(other);
  }

  @override
  int get hashCode => se.basic_hash(pointer);
}

// =============================================================================
// SHORT SYMBOL FACTORY & ELEMENTARY FUNCTIONS
// =============================================================================

/// A symbolic variable (e.g. `x`, `y`).
///
/// Can be instantiated via `Symbol('x')` and used as a distinct type implementing [Expr].
extension type const Symbol._(Expr _expr) implements Expr {
  /// Creates a symbolic variable with the given [name].
  factory Symbol(String name) => Symbol._(Expr.symbol(name));

  /// Wraps an existing [expr] as a [Symbol].
  ///
  /// Throws [ArgumentError] if [expr] is not a symbol.
  factory Symbol.fromExpr(Expr expr) {
    if (se.is_a_Symbol(expr.pointer) == 0) {
      throw ArgumentError('The provided expression is not a Symbol: $expr');
    }
    return Symbol._(expr);
  }
}

/// Shorthand helper to create a symbolic integer.
// ignore: non_constant_identifier_names
Expr Integer(int val) => Expr.integer(val);

/// Shorthand helper to create a symbolic floating-point real.
// ignore: non_constant_identifier_names
Expr Real(double val) => Expr.real(val);

/// Shorthand helper to create a symbolic rational number.
// ignore: non_constant_identifier_names
Expr Rational(int numerator, int denominator) =>
    Expr.rational(numerator, denominator);

/// Sine of symbolic expression [x].
Expr sin(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_sin(res, e.pointer);
  return Expr._(res);
}

/// Cosine of symbolic expression [x].
Expr cos(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_cos(res, e.pointer);
  return Expr._(res);
}

/// Tangent of symbolic expression [x].
Expr tan(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_tan(res, e.pointer);
  return Expr._(res);
}

/// Arcsine of symbolic expression [x].
Expr asin(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_asin(res, e.pointer);
  return Expr._(res);
}

/// Arccosine of symbolic expression [x].
Expr acos(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_acos(res, e.pointer);
  return Expr._(res);
}

/// Arctangent of symbolic expression [x].
Expr atan(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_atan(res, e.pointer);
  return Expr._(res);
}

/// Exponential `exp(x)` of symbolic expression [x].
Expr exp(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_exp(res, e.pointer);
  return Expr._(res);
}

/// Natural logarithm `log(x)` of symbolic expression [x].
Expr log(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_log(res, e.pointer);
  return Expr._(res);
}

/// Square root `sqrt(x)` of symbolic expression [x].
Expr sqrt(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_sqrt(res, e.pointer);
  return Expr._(res);
}

/// Cube root `cbrt(x)` of symbolic expression [x].
Expr cbrt(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_cbrt(res, e.pointer);
  return Expr._(res);
}

/// Absolute value `abs(x)` of symbolic expression [x].
Expr abs(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_abs(res, e.pointer);
  return Expr._(res);
}

/// Arctangent of two arguments `atan2(y, x)`.
Expr atan2(Object y, Object x) {
  final ey = Expr.fromObject(y);
  final ex = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_atan2(res, ey.pointer, ex.pointer);
  return Expr._(res);
}

/// Cosecant `csc(x)` of symbolic expression [x].
Expr csc(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_csc(res, e.pointer);
  return Expr._(res);
}

/// Secant `sec(x)` of symbolic expression [x].
Expr sec(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_sec(res, e.pointer);
  return Expr._(res);
}

/// Cotangent `cot(x)` of symbolic expression [x].
Expr cot(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_cot(res, e.pointer);
  return Expr._(res);
}

/// Hyperbolic sine `sinh(x)` of symbolic expression [x].
Expr sinh(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_sinh(res, e.pointer);
  return Expr._(res);
}

/// Hyperbolic cosine `cosh(x)` of symbolic expression [x].
Expr cosh(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_cosh(res, e.pointer);
  return Expr._(res);
}

/// Hyperbolic tangent `tanh(x)` of symbolic expression [x].
Expr tanh(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_tanh(res, e.pointer);
  return Expr._(res);
}

/// Inverse hyperbolic sine `asinh(x)` of symbolic expression [x].
Expr asinh(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_asinh(res, e.pointer);
  return Expr._(res);
}

/// Inverse hyperbolic cosine `acosh(x)` of symbolic expression [x].
Expr acosh(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_acosh(res, e.pointer);
  return Expr._(res);
}

/// Inverse hyperbolic tangent `atanh(x)` of symbolic expression [x].
Expr atanh(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_atanh(res, e.pointer);
  return Expr._(res);
}

/// Floor integer rounding of symbolic expression [x].
Expr floor(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_floor(res, e.pointer);
  return Expr._(res);
}

/// Ceiling integer rounding of symbolic expression [x].
Expr ceil(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_ceiling(res, e.pointer);
  return Expr._(res);
}

/// Error function `erf(x)` of symbolic expression [x].
Expr erf(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_erf(res, e.pointer);
  return Expr._(res);
}

/// Complementary error function `erfc(x)` of symbolic expression [x].
Expr erfc(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_erfc(res, e.pointer);
  return Expr._(res);
}

/// Euler gamma function `gamma(x)` of symbolic expression [x].
Expr gamma(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_gamma(res, e.pointer);
  return Expr._(res);
}

/// Lambert W function `lambertw(x)` of symbolic expression [x].
Expr lambertw(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_lambertw(res, e.pointer);
  return Expr._(res);
}

/// Riemann zeta function `zeta(x)` of symbolic expression [x].
Expr zeta(Object x) {
  final e = Expr.fromObject(x);
  final res = se.basic_new_heap();
  se.basic_zeta(res, e.pointer);
  return Expr._(res);
}

/// Kronecker delta `delta(i, j)`, equal to 1 if i == j, 0 otherwise.
Expr kroneckerDelta(Object i, Object j) {
  final ei = Expr.fromObject(i);
  final ej = Expr.fromObject(j);
  final res = se.basic_new_heap();
  se.basic_kronecker_delta(res, ei.pointer, ej.pointer);
  return Expr._(res);
}

/// Exact Greatest Common Divisor `gcd(a, b)` over algebraic expressions or integers.
Expr gcd(Object a, Object b) {
  final ea = Expr.fromObject(a);
  final eb = Expr.fromObject(b);
  final res = se.basic_new_heap();
  se.ntheory_gcd(res, ea.pointer, eb.pointer);
  return Expr._(res);
}

/// Exact Least Common Multiple `lcm(a, b)` over algebraic expressions or integers.
Expr lcm(Object a, Object b) {
  final ea = Expr.fromObject(a);
  final eb = Expr.fromObject(b);
  final res = se.basic_new_heap();
  se.ntheory_lcm(res, ea.pointer, eb.pointer);
  return Expr._(res);
}

/// Convenient arithmetic operations allowing numeric literals on the LHS
/// with symbolic expressions and matrices on the RHS.
extension SymbolicNumExtension on num {
  /// Converts this numeric value (`int` or `double`) into a symbolic [Expr].
  Expr get toExpr => Expr.fromObject(this);

  /// Adds this number to symbolic expression [other].
  Expr operator +(Expr other) => Expr.fromObject(this) + other;

  /// Subtracts symbolic expression [other] from this number.
  Expr operator -(Expr other) => Expr.fromObject(this) - other;

  /// Multiplies this number by symbolic expression [other].
  Expr operator *(Expr other) => Expr.fromObject(this) * other;

  /// Divides this number by symbolic expression [other].
  Expr operator /(Expr other) => Expr.fromObject(this) / other;

  /// Scales a [SymbolicMatrix] by this number (`other * this`).
  SymbolicMatrix scaleMatrix(SymbolicMatrix other) => other * this;
}
