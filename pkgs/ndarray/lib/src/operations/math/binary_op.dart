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

library;

/// Representation of universal binary operations (ufuncs) supported across arrays.
///
/// Universal binary operations take two array inputs (or an array and a scalar)
/// element-wise and support generalized operations such as `reduce`, `accumulate`,
/// `reduceat`, `outer`, and `at`.
enum BinaryOp {
  /// Element-wise addition.
  add,

  /// Element-wise multiplication.
  multiply,

  /// Element-wise minimum.
  minimum,

  /// Element-wise maximum.
  maximum,

  /// Element-wise minimum ignoring NaNs.
  fmin,

  /// Element-wise maximum ignoring NaNs.
  fmax,

  /// Logarithm of the sum of exponentiated inputs `log(exp(x1) + exp(x2))`.
  logaddexp,

  /// Logarithm in base 2 of the sum of exponentiated inputs `log2(2^x1 + 2^x2)`.
  logaddexp2,

  /// Greatest common divisor.
  gcd,

  /// Least common multiple.
  lcm,

  /// Bitwise AND operation.
  bitwiseAnd,

  /// Bitwise OR operation.
  bitwiseOr,

  /// Bitwise XOR operation.
  bitwiseXor,

  /// Element-wise logical AND operation.
  logicalAnd,

  /// Element-wise logical OR operation.
  logicalOr,

  /// Element-wise logical XOR operation.
  logicalXor,

  /// Element-wise subtraction.
  subtract,

  /// Element-wise true division.
  divide,

  /// Element-wise floor division.
  floorDivide,

  /// Element-wise remainder of division.
  remainder,

  /// Element-wise C-style modulo / remainder (`fmod`).
  fmod,

  /// Element-wise exponentiation `x1 ^ x2`.
  power,

  /// Element-wise float exponentiation `x1 ^ x2` promoting integer to float.
  floatPower,

  /// Element-wise arctangent of `x1 / x2` choosing quadrant correctly.
  arctan2,

  /// Element-wise hypotenuse `sqrt(x1^2 + x2^2)`.
  hypot,

  /// Element-wise copy sign from `x2` to `x1`.
  copysign,

  /// Bitwise left shift.
  leftShift,

  /// Bitwise right shift.
  rightShift,

  /// Heaviside step function.
  heaviside,

  /// Element-wise equality comparison (`==`).
  equal,

  /// Element-wise inequality comparison (`!=`).
  notEqual,

  /// Element-wise greater-than comparison (`>`).
  greater,

  /// Element-wise greater-than-or-equal comparison (`>=`).
  greaterEqual,

  /// Element-wise less-than comparison (`<`).
  less,

  /// Element-wise less-than-or-equal comparison (`<=`).
  lessEqual;

  /// Whether this binary operation supports reduction along array axes.
  ///
  /// Reducible operations are associative and commutative (e.g. [add], [multiply], [minimum], [maximum], etc.).
  bool get isReducible {
    switch (this) {
      case BinaryOp.add:
      case BinaryOp.multiply:
      case BinaryOp.minimum:
      case BinaryOp.maximum:
      case BinaryOp.fmin:
      case BinaryOp.fmax:
      case BinaryOp.logaddexp:
      case BinaryOp.logaddexp2:
      case BinaryOp.gcd:
      case BinaryOp.lcm:
      case BinaryOp.bitwiseAnd:
      case BinaryOp.bitwiseOr:
      case BinaryOp.bitwiseXor:
      case BinaryOp.logicalAnd:
      case BinaryOp.logicalOr:
      case BinaryOp.logicalXor:
        return true;
      case BinaryOp.subtract:
      case BinaryOp.divide:
      case BinaryOp.floorDivide:
      case BinaryOp.remainder:
      case BinaryOp.fmod:
      case BinaryOp.power:
      case BinaryOp.floatPower:
      case BinaryOp.arctan2:
      case BinaryOp.hypot:
      case BinaryOp.copysign:
      case BinaryOp.leftShift:
      case BinaryOp.rightShift:
      case BinaryOp.heaviside:
      case BinaryOp.equal:
      case BinaryOp.notEqual:
      case BinaryOp.greater:
      case BinaryOp.greaterEqual:
      case BinaryOp.less:
      case BinaryOp.lessEqual:
        return false;
    }
  }
}

/// Representation of universal unary operations (ufuncs) supported across arrays.
enum UnaryOp {
  /// Element-wise numerical negation (`-x`).
  negative,

  /// Element-wise numerical positive (`+x`).
  positive,

  /// Element-wise absolute value (`|x|`).
  absolute,

  /// Alias for [absolute].
  abs,

  /// Element-wise floating-point absolute value.
  fabs,

  /// Element-wise round to nearest integer.
  rint,

  /// Element-wise sign indication (`-1, 0, +1`).
  sign,

  /// Element-wise complex conjugate.
  conj,

  /// Alias for [conj].
  conjugate,

  /// Element-wise exponential (`e^x`).
  exp,

  /// Element-wise base-2 exponential (`2^x`).
  exp2,

  /// Element-wise natural logarithm (`ln(x)`).
  log,

  /// Element-wise base-2 logarithm (`log2(x)`).
  log2,

  /// Element-wise base-10 logarithm (`log10(x)`).
  log10,

  /// Element-wise `exp(x) - 1`.
  expm1,

  /// Element-wise `log(1 + x)`.
  log1p,

  /// Element-wise principal square root.
  sqrt,

  /// Element-wise square (`x^2`).
  square,

  /// Element-wise cube root.
  cbrt,

  /// Element-wise reciprocal (`1 / x`).
  reciprocal,

  /// Element-wise trigonometric sine.
  sin,

  /// Element-wise trigonometric cosine.
  cos,

  /// Element-wise trigonometric tangent.
  tan,

  /// Element-wise inverse sine.
  arcsin,

  /// Element-wise inverse cosine.
  arccos,

  /// Element-wise inverse tangent.
  arctan,

  /// Element-wise hyperbolic sine.
  sinh,

  /// Element-wise hyperbolic cosine.
  cosh,

  /// Element-wise hyperbolic tangent.
  tanh,

  /// Element-wise inverse hyperbolic sine.
  arcsinh,

  /// Element-wise inverse hyperbolic cosine.
  arccosh,

  /// Element-wise inverse hyperbolic tangent.
  arctanh,

  /// Element-wise conversion from radians to degrees.
  degrees,

  /// Element-wise conversion from degrees to radians.
  radians,

  /// Alias for [radians].
  deg2rad,

  /// Alias for [degrees].
  rad2deg,

  /// Element-wise bitwise inversion (`~x`).
  invert,

  /// Alias for [invert].
  bitwiseNot,

  /// Element-wise logical NOT (`!x`).
  logicalNot,

  /// Element-wise NaN check.
  isnan,

  /// Element-wise infinity check.
  isinf,

  /// Element-wise finiteness check.
  isfinite,

  /// Element-wise signbit check.
  signbit,

  /// Element-wise floor.
  floor,

  /// Element-wise ceiling.
  ceil,

  /// Element-wise truncation toward zero.
  trunc,

  /// Element-wise ULP spacing.
  spacing,
}
