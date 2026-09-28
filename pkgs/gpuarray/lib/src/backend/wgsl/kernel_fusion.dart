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

import 'dart:convert';
import 'dart:typed_data';
import 'wgsl_types.dart';
import 'wgsl_templates.dart';
import '../../dtype.dart';
import '../../gpu_array.dart';
import '../../serialization/webgpu_pipeline.dart';

/// Base class for all expression nodes in a fused kernel computation graph.
typedef GpuExpr = Expr;
typedef GpuVarExpr = VarExpr;
typedef GpuConstExpr = ConstExpr;
typedef GpuScalarParamExpr = ScalarParamExpr;
typedef GpuLoopExpr = LoopExpr;
typedef GpuCoordExpr = CoordExpr;
typedef GpuIndexExpr = IndexExpr;
typedef GpuLetExpr = LetExpr;
typedef GpuLocalVarExpr = LocalVarExpr;
typedef GpuOffsetVarExpr = OffsetVarExpr;
typedef GpuBoundaryMode = BoundaryMode;

abstract class Expr {
  const Expr();

  /// Creates a variable reference representing an input tensor buffer.
  static VarExpr variable(
    String name, {
    int bindingIndex = 0,
    WgslDType dtype = WgslDType.float32,
  }) => VarExpr(name, bindingIndex: bindingIndex, dtype: dtype);

  /// Creates a literal numerical constant.
  static ConstExpr constant(double value) => ConstExpr(value);

  /// Creates a runtime scalar uniform parameter.
  static ScalarParamExpr scalar(String name, {double defaultValue = 0.0}) =>
      ScalarParamExpr(name, defaultValue: defaultValue);

  /// Creates an intrinsic coordinate expression for [axis] (e.g. 0=row/y, 1=col/x).
  ///
  /// When [normalized] is true, coordinates are normalized to $[0, 1]$.
  static CoordExpr coord(
    int axis, {
    List<int>? shape,
    bool normalized = false,
  }) => CoordExpr(axis, shape: shape, normalized: normalized);

  /// Creates an expression representing the global flat element index (`idx`).
  static IndexExpr index() => const IndexExpr();

  static int _letCounter = 0;

  /// Binds [value] to a named local variable evaluated once, passing it to [builder].
  static Expr let(
    Object value,
    Expr Function(Expr local) builder, {
    String? name,
  }) {
    final valExpr = Expr.from(value);
    final varName = name ?? '_let_${_letCounter++}';
    final local = LocalVarExpr(varName);
    final body = builder(local);
    return LetExpr(varName, valExpr, body);
  }

  /// Creates a functional loop AST node with state variables, dynamic condition, and step function.
  static LoopExpr loop({
    required List<Object> initialValues,
    required Object maxIterations,
    required Expr Function(List<Expr> state, Expr iter) condition,
    required List<Expr> Function(List<Expr> state, Expr iter) step,
    Expr Function(List<Expr> state)? result,
    String? name,
  }) => LoopExpr(
    initialValues: initialValues,
    maxIterations: maxIterations,
    condition: condition,
    step: step,
    result: result,
    name: name,
  );

  /// Converts any num or Expr object into an [Expr].
  static Expr from(Object value) {
    if (value is Expr) return value;
    if (value is num) return ConstExpr(value.toDouble());
    throw ArgumentError(
      'Cannot convert $value of type ${value.runtimeType} to Expr',
    );
  }

  /// Evaluates to the WGSL code snippet for this subexpression.
  String toWgsl();

  /// Collects all input variable references used in this expression tree.
  Set<VarExpr> get variables;

  /// Collects all scalar uniform parameters used in this expression tree.
  Set<ScalarParamExpr> get scalarParams;

  /// Computes the depth of the expression tree.
  int get depth;

  /// Total count of AST nodes in this expression subtree.
  int get nodeCount;

  /// Generates a normalized structural fingerprint string for caching.
  String toFingerprint();

  /// Computes the exact symbolic analytical derivative with respect to [wrt].
  Expr grad(VarExpr wrt);

  // Operator overloads for building expression trees fluently with algebraic simplifications
  Expr operator +(Object other) {
    final o = Expr.from(other);
    if (this is ConstExpr && (this as ConstExpr).value == 0.0) return o;
    if (o is ConstExpr && o.value == 0.0) return this;
    return BinaryOpExpr('add', this, o);
  }

  Expr operator -(Object other) {
    final o = Expr.from(other);
    if (o is ConstExpr && o.value == 0.0) return this;
    return BinaryOpExpr('sub', this, o);
  }

  Expr operator *(Object other) {
    final o = Expr.from(other);
    if (this is ConstExpr && (this as ConstExpr).value == 0.0) {
      return ConstExpr(0.0);
    }
    if (o is ConstExpr && o.value == 0.0) return ConstExpr(0.0);
    if (this is ConstExpr && (this as ConstExpr).value == 1.0) return o;
    if (o is ConstExpr && o.value == 1.0) return this;
    return BinaryOpExpr('mul', this, o);
  }

  Expr operator /(Object other) {
    final o = Expr.from(other);
    if (this is ConstExpr && (this as ConstExpr).value == 0.0) {
      return ConstExpr(0.0);
    }
    if (o is ConstExpr && o.value == 1.0) return this;
    return BinaryOpExpr('div', this, o);
  }

  Expr operator -() => UnaryOpExpr('negate', this);

  Expr pow(Object exponent) => BinaryOpExpr('pow', this, Expr.from(exponent));

  Expr max(Object other) => BinaryOpExpr('max', this, Expr.from(other));

  Expr min(Object other) => BinaryOpExpr('min', this, Expr.from(other));

  Expr equal(Object other) => BinaryOpExpr('eq', this, Expr.from(other));

  Expr notEqual(Object other) => BinaryOpExpr('neq', this, Expr.from(other));

  Expr greaterThan(Object other) => BinaryOpExpr('gt', this, Expr.from(other));

  Expr lessThan(Object other) => BinaryOpExpr('lt', this, Expr.from(other));

  Expr greaterEqual(Object other) =>
      BinaryOpExpr('gte', this, Expr.from(other));

  Expr lessEqual(Object other) => BinaryOpExpr('lte', this, Expr.from(other));

  // Logical operators
  Expr operator &(Object other) => BinaryOpExpr('and', this, Expr.from(other));

  Expr operator |(Object other) => BinaryOpExpr('or', this, Expr.from(other));

  Expr operator ~() => UnaryOpExpr('not', this);

  Expr and(Object other) => this & other;

  Expr or(Object other) => this | other;

  Expr not() => ~this;

  // Shader math intrinsics
  Expr mix(Object other, Object t) =>
      TernaryOpExpr('mix', this, Expr.from(other), Expr.from(t));

  Expr smoothstep(Object edge0, Object edge1) =>
      TernaryOpExpr('smoothstep', Expr.from(edge0), Expr.from(edge1), this);

  Expr step(Object edge) => BinaryOpExpr('step', Expr.from(edge), this);

  Expr mod(Object other) => BinaryOpExpr('mod', this, Expr.from(other));

  Expr operator %(Object other) => mod(other);

  Expr fract() => UnaryOpExpr('fract', this);

  Expr atan2(Object x) => BinaryOpExpr('atan2', this, Expr.from(x));

  Expr hypot(Object other) => BinaryOpExpr('hypot', this, Expr.from(other));

  Expr sign() => UnaryOpExpr('sign', this);

  // Common Unary activations and math functions
  Expr relu() => UnaryOpExpr('relu', this);
  Expr silu() => UnaryOpExpr('silu', this);
  Expr gelu() => UnaryOpExpr('gelu', this);
  Expr sigmoid() => UnaryOpExpr('sigmoid', this);
  Expr tanh() => UnaryOpExpr('tanh', this);
  Expr exp() => UnaryOpExpr('exp', this);
  Expr log() => UnaryOpExpr('log', this);
  Expr sqrt() => UnaryOpExpr('sqrt', this);
  Expr rsqrt() => UnaryOpExpr('rsqrt', this);
  Expr abs() => UnaryOpExpr('abs', this);
  Expr sin() => UnaryOpExpr('sin', this);
  Expr cos() => UnaryOpExpr('cos', this);
  Expr tan() => UnaryOpExpr('tan', this);
  Expr sinh() => UnaryOpExpr('sinh', this);
  Expr cosh() => UnaryOpExpr('cosh', this);
  Expr floor() => UnaryOpExpr('floor', this);
  Expr ceil() => UnaryOpExpr('ceil', this);
  Expr round() => UnaryOpExpr('round', this);
  Expr reciprocal() => UnaryOpExpr('reciprocal', this);
  Expr hardswish() => UnaryOpExpr('hardswish', this);
  Expr softplus() => UnaryOpExpr('softplus', this);
  Expr mish() => UnaryOpExpr('mish', this);

  /// Clamps expression between [minVal] and [maxVal].
  Expr clamp(Object minVal, Object maxVal) =>
      TernaryOpExpr('clamp', this, Expr.from(minVal), Expr.from(maxVal));

  /// Selects [thenExpr] if this expression evaluates to > 0.0, else [elseExpr].
  Expr where(Object thenExpr, Object elseExpr) =>
      TernaryOpExpr('where', this, Expr.from(thenExpr), Expr.from(elseExpr));

  /// Optimizes the expression tree by performing Common Subexpression Elimination (CSE).
  ///
  /// Repeated non-trivial subtrees appearing at least [minOccurrences] times with at least
  /// [minNodeCount] nodes are hoisted into [LetExpr] bindings.
  Expr eliminateCommonSubexpressions({
    int minNodeCount = 2,
    int minOccurrences = 2,
  }) {
    final counts = <String, int>{};
    final candidates = <String, Expr>{};

    void countSubtrees(Expr e) {
      if (e is ConstExpr ||
          e is VarExpr ||
          e is LocalVarExpr ||
          e is ScalarParamExpr ||
          e is CoordExpr ||
          e is IndexExpr ||
          e is LetExpr ||
          e is LoopExpr) {
        return;
      }
      final fp = e.toFingerprint();
      counts[fp] = (counts[fp] ?? 0) + 1;
      candidates[fp] = e;

      if (e is UnaryOpExpr) {
        countSubtrees(e.child);
      } else if (e is BinaryOpExpr) {
        countSubtrees(e.left);
        countSubtrees(e.right);
      } else if (e is TernaryOpExpr) {
        countSubtrees(e.first);
        countSubtrees(e.second);
        countSubtrees(e.third);
      }
    }

    countSubtrees(this);

    final cseList = <Expr>[];
    for (final entry in counts.entries) {
      if (entry.value >= minOccurrences) {
        final node = candidates[entry.key]!;
        if (node.nodeCount >= minNodeCount) {
          cseList.add(node);
        }
      }
    }

    if (cseList.isEmpty) return this;

    // Hoist shallower / smaller subexpressions first
    cseList.sort((a, b) => a.depth.compareTo(b.depth));

    var current = this;
    var cseIdx = 0;
    for (final cseNode in cseList) {
      final targetFp = cseNode.toFingerprint();
      final varName = '_cse_${cseIdx++}';

      Expr substitute(Expr e) {
        if (e.toFingerprint() == targetFp) {
          return LocalVarExpr(varName);
        }
        if (e is UnaryOpExpr) {
          return UnaryOpExpr(e.op, substitute(e.child));
        } else if (e is BinaryOpExpr) {
          return BinaryOpExpr(e.op, substitute(e.left), substitute(e.right));
        } else if (e is TernaryOpExpr) {
          return TernaryOpExpr(
            e.op,
            substitute(e.first),
            substitute(e.second),
            substitute(e.third),
          );
        } else if (e is LetExpr) {
          return LetExpr(e.name, substitute(e.value), substitute(e.body));
        }
        return e;
      }

      final substitutedBody = substitute(current);
      current = LetExpr(varName, cseNode, substitutedBody);
    }

    return current;
  }

  /// Fluent alias for [eliminateCommonSubexpressions].
  Expr cse() => eliminateCommonSubexpressions();
}

/// Represents an input tensor variable buffer.
final class VarExpr extends Expr {
  final String name;
  final int bindingIndex;
  final WgslDType dtype;

  const VarExpr(
    this.name, {
    this.bindingIndex = 0,
    this.dtype = WgslDType.float32,
  });

  @override
  String toWgsl() => '${name}_val';

  @override
  Set<VarExpr> get variables => {this};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'var($name:$bindingIndex:${dtype.wgslType})';

  @override
  Expr grad(VarExpr wrt) =>
      this == wrt ? const ConstExpr(1.0) : const ConstExpr(0.0);

  /// Accesses neighbor cells offset by [offsets] (e.g. `[-1, 0]` for top neighbor in 2D).
  Expr offset(
    List<int> offsets, {
    List<int> shape = const [],
    BoundaryMode boundary = BoundaryMode.clamp,
  }) => OffsetVarExpr(this, offsets, shape: shape, boundary: boundary);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is VarExpr &&
          runtimeType == other.runtimeType &&
          name == other.name &&
          bindingIndex == other.bindingIndex &&
          dtype == other.dtype;

  @override
  int get hashCode => Object.hash(name, bindingIndex, dtype);

  @override
  String toString() => 'VarExpr($name, binding: $bindingIndex)';
}

/// Represents a constant floating-point literal.
final class ConstExpr extends Expr {
  final double value;

  const ConstExpr(this.value);

  @override
  String toWgsl() {
    final s = value.toString();
    return (s.contains('.') || s.contains('e') || s.contains('E'))
        ? '${s}f'
        : '$s.0f';
  }

  @override
  Set<VarExpr> get variables => {};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'const($value)';

  @override
  Expr grad(VarExpr wrt) => const ConstExpr(0.0);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ConstExpr &&
          runtimeType == other.runtimeType &&
          value == other.value;

  @override
  int get hashCode => value.hashCode;

  @override
  String toString() => 'ConstExpr($value)';
}

/// Represents a dynamic runtime scalar uniform parameter.
final class ScalarParamExpr extends Expr {
  final String name;
  final double defaultValue;

  const ScalarParamExpr(this.name, {this.defaultValue = 0.0});

  @override
  String toWgsl() => 'uniforms.$name';

  @override
  Set<VarExpr> get variables => {};

  @override
  Set<ScalarParamExpr> get scalarParams => {this};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'scalar($name)';

  @override
  Expr grad(VarExpr wrt) => const ConstExpr(0.0);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ScalarParamExpr &&
          runtimeType == other.runtimeType &&
          name == other.name;

  @override
  int get hashCode => name.hashCode;

  @override
  String toString() => 'ScalarParamExpr($name)';
}

/// Represents an intrinsic dimensional coordinate (e.g. 0=row/y, 1=col/x).
final class CoordExpr extends Expr {
  final int axis;
  final List<int>? shape;
  final bool normalized;

  const CoordExpr(this.axis, {this.shape, this.normalized = false});

  @override
  String toWgsl() {
    final s = shape;
    if (s != null && s.isNotEmpty) {
      if (s.length == 1) {
        return normalized ? '(f32(idx) / ${s[0].toDouble()}f)' : 'f32(idx)';
      }
      if (s.length == 2) {
        final H = s[0];
        final W = s[1];
        if (axis == 1 || axis == -1) {
          return normalized
              ? '(f32(idx % ${W}u) / ${W.toDouble()}f)'
              : 'f32(idx % ${W}u)';
        } else {
          return normalized
              ? '(f32((idx / ${W}u) % ${H}u) / ${H.toDouble()}f)'
              : 'f32((idx / ${W}u) % ${H}u)';
        }
      }
      var stride = 1;
      for (var i = axis + 1; i < s.length; i++) {
        stride *= s[i];
      }
      final dim = s[axis];
      final coordStr = '(idx / ${stride}u) % ${dim}u';
      return normalized
          ? '(f32($coordStr) / ${dim.toDouble()}f)'
          : 'f32($coordStr)';
    }
    return normalized
        ? '(f32(idx) / f32(uniforms.total_elements))'
        : 'f32(idx)';
  }

  @override
  Set<VarExpr> get variables => {};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'coord($axis;${shape ?? []};$normalized)';

  @override
  Expr grad(VarExpr wrt) => const ConstExpr(0.0);

  @override
  String toString() =>
      'CoordExpr(axis: $axis, shape: $shape, norm: $normalized)';
}

/// Represents the global flat element index (`idx`).
final class IndexExpr extends Expr {
  const IndexExpr();

  @override
  String toWgsl() => 'f32(idx)';

  @override
  Set<VarExpr> get variables => {};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'index()';

  @override
  Expr grad(VarExpr wrt) => const ConstExpr(0.0);

  @override
  String toString() => 'IndexExpr()';
}

/// Represents a reference to a local variable bound in a [LetExpr].
final class LocalVarExpr extends Expr {
  final String name;

  const LocalVarExpr(this.name);

  @override
  String toWgsl() => name;

  @override
  Set<VarExpr> get variables => {};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'local($name)';

  @override
  Expr grad(VarExpr wrt) => const ConstExpr(0.0);

  @override
  String toString() => 'LocalVarExpr($name)';
}

/// Represents a local variable binding (`let <name> = <value>; in <body>`).
final class LetExpr extends Expr {
  final String name;
  final Expr value;
  final Expr body;

  const LetExpr(this.name, this.value, this.body);

  @override
  String toWgsl() => body.toWgsl();

  @override
  Set<VarExpr> get variables => {...value.variables, ...body.variables};

  @override
  Set<ScalarParamExpr> get scalarParams => {
    ...value.scalarParams,
    ...body.scalarParams,
  };

  @override
  int get depth => 1 + (value.depth > body.depth ? value.depth : body.depth);

  @override
  int get nodeCount => 1 + value.nodeCount + body.nodeCount;

  @override
  String toFingerprint() =>
      'let($name=${value.toFingerprint()};${body.toFingerprint()})';

  @override
  Expr grad(VarExpr wrt) => LetExpr(name, value, body.grad(wrt));

  @override
  String toString() => 'LetExpr(let $name = $value in $body)';
}

/// Boundary modes for stencil / neighbor sampling.
enum BoundaryMode {
  /// Clamps out-of-bound indices to the edge cells.
  clamp,

  /// Toroidal wrapping / periodic boundary conditions.
  wrap,

  /// Returns 0.0 for any out-of-bound accesses.
  zero,
}

/// Represents a neighbor / stencil sampling access into an input tensor.
final class OffsetVarExpr extends Expr {
  final VarExpr tensor;
  final List<int> offsets;
  final List<int> shape;
  final BoundaryMode boundary;

  const OffsetVarExpr(
    this.tensor,
    this.offsets, {
    this.shape = const [],
    this.boundary = BoundaryMode.clamp,
  });

  String get functionName {
    final offStr = offsets.map((o) => o < 0 ? 'm${-o}' : 'p$o').join('_');
    return 'stencil_${tensor.name}_${offStr}_${boundary.name}';
  }

  @override
  String toWgsl() => '$functionName(idx, uniforms.total_elements)';

  @override
  Set<VarExpr> get variables => {tensor};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() =>
      'offset(${tensor.toFingerprint()};[${offsets.join(",")}];${boundary.name})';

  @override
  Expr grad(VarExpr wrt) => (tensor == wrt && offsets.every((o) => o == 0))
      ? const ConstExpr(1.0)
      : const ConstExpr(0.0);

  /// Generates the standalone WGSL helper function for sampling this stencil offset.
  String generateWgslFunction() {
    final fnName = functionName;
    final tName = tensor.name;

    if (offsets.length == 2) {
      final dr = offsets[0];
      final dc = offsets[1];
      final String wStr;
      final String hStr;
      if (shape.length >= 2) {
        hStr = '${shape[0]}u';
        wStr = '${shape[1]}u';
      } else {
        wStr = 'u32(sqrt(f32(total_elements)))';
        hStr = wStr;
      }

      String sampleLogic;
      switch (boundary) {
        case BoundaryMode.clamp:
          sampleLogic =
              '''
  let W = $wStr;
  let H = $hStr;
  let r = i32(idx / W);
  let c = i32(idx % W);
  let nr = clamp(r + ($dr), 0, i32(H - 1u));
  let nc = clamp(c + ($dc), 0, i32(W - 1u));
  return $tName[u32(nr) * W + u32(nc)];
''';
          break;
        case BoundaryMode.wrap:
          sampleLogic =
              '''
  let W = $wStr;
  let H = $hStr;
  let r = i32(idx / W);
  let c = i32(idx % W);
  let nr = ((r + ($dr)) % i32(H) + i32(H)) % i32(H);
  let nc = ((c + ($dc)) % i32(W) + i32(W)) % i32(W);
  return $tName[u32(nr) * W + u32(nc)];
''';
          break;
        case BoundaryMode.zero:
          sampleLogic =
              '''
  let W = $wStr;
  let H = $hStr;
  let r = i32(idx / W);
  let c = i32(idx % W);
  let nr = r + ($dr);
  let nc = c + ($dc);
  if (nr < 0 || nr >= i32(H) || nc < 0 || nc >= i32(W)) {
    return 0.0f;
  }
  return $tName[u32(nr) * W + u32(nc)];
''';
          break;
      }

      return '''
fn $fnName(idx: u32, total_elements: u32) -> f32 {
$sampleLogic}
''';
    } else {
      final d = offsets.first;
      String sampleLogic;
      switch (boundary) {
        case BoundaryMode.clamp:
          sampleLogic =
              '''
  let target = clamp(i32(idx) + ($d), 0, i32(total_elements - 1u));
  return $tName[u32(target)];
''';
          break;
        case BoundaryMode.wrap:
          sampleLogic =
              '''
  let N = i32(total_elements);
  let target = ((i32(idx) + ($d)) % N + N) % N;
  return $tName[u32(target)];
''';
          break;
        case BoundaryMode.zero:
          sampleLogic =
              '''
  let target = i32(idx) + ($d);
  if (target < 0 || target >= i32(total_elements)) {
    return 0.0f;
  }
  return $tName[u32(target)];
''';
          break;
      }
      return '''
fn $fnName(idx: u32, total_elements: u32) -> f32 {
$sampleLogic}
''';
    }
  }

  @override
  String toString() =>
      'OffsetVarExpr($tensor, offsets: $offsets, boundary: $boundary)';
}

/// Represents a unary operation applied to an expression.
final class UnaryOpExpr extends Expr {
  final String op;
  final Expr child;

  const UnaryOpExpr(this.op, this.child);

  @override
  String toWgsl() =>
      WgslTemplates.getWgslUnaryExpression(op, '(${child.toWgsl()})');

  @override
  Set<VarExpr> get variables => child.variables;

  @override
  Set<ScalarParamExpr> get scalarParams => child.scalarParams;

  @override
  int get depth => 1 + child.depth;

  @override
  int get nodeCount => 1 + child.nodeCount;

  @override
  String toFingerprint() => '$op(${child.toFingerprint()})';

  @override
  Expr grad(VarExpr wrt) {
    final du = child.grad(wrt);
    switch (op.toLowerCase()) {
      case 'negate':
      case 'neg':
      case '-':
        return -du;
      case 'sin':
        return child.cos() * du;
      case 'cos':
        return -child.sin() * du;
      case 'tan':
        return (Expr.constant(1.0) + this * this) * du;
      case 'exp':
        return this * du;
      case 'log':
        return du / child;
      case 'sqrt':
        return du / (child.sqrt() * 2.0);
      case 'rsqrt':
      case 'inversesqrt':
        return (const ConstExpr(-0.5) * du) / (child * child.sqrt());
      case 'abs':
        return child.sign() * du;
      case 'tanh':
        return (Expr.constant(1.0) - this * this) * du;
      case 'sigmoid':
        return this * (Expr.constant(1.0) - this) * du;
      case 'silu':
      case 'swish':
        final s = child.sigmoid();
        return (s + child * s * (Expr.constant(1.0) - s)) * du;
      case 'relu':
        return (child.greaterThan(0.0)).where(du, Expr.constant(0.0));
      case 'asin':
        return du / (Expr.constant(1.0) - child * child).sqrt();
      case 'acos':
        return -du / (Expr.constant(1.0) - child * child).sqrt();
      case 'atan':
        return du / (Expr.constant(1.0) + child * child);
      case 'sinh':
        return child.cosh() * du;
      case 'cosh':
        return child.sinh() * du;
      case 'reciprocal':
        return -du / (child * child);
      case 'not':
      case '~':
      case '!':
      case 'sign':
      case 'floor':
      case 'ceil':
      case 'round':
      case 'fract':
        return const ConstExpr(0.0);
      default:
        throw UnsupportedError('Gradient not implemented for unary op: $op');
    }
  }

  @override
  String toString() => 'UnaryOpExpr($op, $child)';
}

/// Represents a binary operation between two expressions.
final class BinaryOpExpr extends Expr {
  final String op;
  final Expr left;
  final Expr right;

  const BinaryOpExpr(this.op, this.left, this.right);

  @override
  String toWgsl() =>
      '(${WgslTemplates.getWgslOpExpression(op, left.toWgsl(), right.toWgsl())})';

  @override
  Set<VarExpr> get variables => {...left.variables, ...right.variables};

  @override
  Set<ScalarParamExpr> get scalarParams => {
    ...left.scalarParams,
    ...right.scalarParams,
  };

  @override
  int get depth => 1 + (left.depth > right.depth ? left.depth : right.depth);

  @override
  int get nodeCount => 1 + left.nodeCount + right.nodeCount;

  @override
  String toFingerprint() =>
      '$op(${left.toFingerprint()}, ${right.toFingerprint()})';

  @override
  Expr grad(VarExpr wrt) {
    final du = left.grad(wrt);
    final dv = right.grad(wrt);
    switch (op.toLowerCase()) {
      case 'add':
      case '+':
        return du + dv;
      case 'sub':
      case '-':
        return du - dv;
      case 'mul':
      case '*':
        return du * right + left * dv;
      case 'div':
      case '/':
        return (du * right - left * dv) / (right * right);
      case 'pow':
      case 'power':
        if (right is ConstExpr) {
          final n = (right as ConstExpr).value;
          return Expr.constant(n) * (left.pow(n - 1.0)) * du;
        }
        return this * (dv * left.log() + right * du / left);
      case 'max':
      case 'maximum':
        return (left.greaterThan(right)).where(du, dv);
      case 'min':
      case 'minimum':
        return (left.lessThan(right)).where(du, dv);
      case 'atan2':
        return (du * right - left * dv) / (left * left + right * right);
      case 'hypot':
        return (left * du + right * dv) / this;
      case 'step':
      case 'eq':
      case 'equal':
      case '==':
      case 'neq':
      case 'notequal':
      case '!=':
      case 'gt':
      case 'greater':
      case '>':
      case 'lt':
      case 'less':
      case '<':
      case 'gte':
      case 'greaterequal':
      case '>=':
      case 'lte':
      case 'lessequal':
      case '<=':
      case 'and':
      case '&':
      case 'or':
      case '|':
      case 'mod':
      case 'rem':
      case 'remainder':
      case '%':
        return const ConstExpr(0.0);
      default:
        throw UnsupportedError('Gradient not implemented for binary op: $op');
    }
  }

  @override
  String toString() => 'BinaryOpExpr($op, $left, $right)';
}

/// Represents a ternary operation (e.g. clamp or select/where).
final class TernaryOpExpr extends Expr {
  final String op;
  final Expr first;
  final Expr second;
  final Expr third;

  const TernaryOpExpr(this.op, this.first, this.second, this.third);

  @override
  String toWgsl() {
    switch (op) {
      case 'clamp':
        return 'clamp(${first.toWgsl()}, ${second.toWgsl()}, ${third.toWgsl()})';
      case 'where':
        return 'select(${third.toWgsl()}, ${second.toWgsl()}, (${first.toWgsl()}) > 0.0)';
      case 'mix':
        return 'mix(${first.toWgsl()}, ${second.toWgsl()}, ${third.toWgsl()})';
      case 'smoothstep':
        return 'smoothstep(${first.toWgsl()}, ${second.toWgsl()}, ${third.toWgsl()})';
      default:
        throw ArgumentError('Unsupported ternary op: $op');
    }
  }

  @override
  Set<VarExpr> get variables => {
    ...first.variables,
    ...second.variables,
    ...third.variables,
  };

  @override
  Set<ScalarParamExpr> get scalarParams => {
    ...first.scalarParams,
    ...second.scalarParams,
    ...third.scalarParams,
  };

  @override
  int get depth =>
      1 +
      [first.depth, second.depth, third.depth].reduce((a, b) => a > b ? a : b);

  @override
  int get nodeCount => 1 + first.nodeCount + second.nodeCount + third.nodeCount;

  @override
  String toFingerprint() =>
      '$op(${first.toFingerprint()}, ${second.toFingerprint()}, ${third.toFingerprint()})';

  @override
  Expr grad(VarExpr wrt) {
    switch (op) {
      case 'where':
        return first.where(second.grad(wrt), third.grad(wrt));
      case 'clamp':
        final du = first.grad(wrt);
        return (first.lessThan(second) | first.greaterThan(third)).where(
          const ConstExpr(0.0),
          du,
        );
      case 'mix':
        final da = first.grad(wrt);
        final db = second.grad(wrt);
        final dt = third.grad(wrt);
        return (Expr.constant(1.0) - third) * da +
            third * db +
            (second - first) * dt;
      case 'smoothstep':
        final dx = third.grad(wrt);
        final span = second - first;
        final t = ((third - first) / span).clamp(0.0, 1.0);
        return (Expr.constant(6.0) * t * (Expr.constant(1.0) - t) / span) * dx;
      default:
        throw UnsupportedError('Gradient not implemented for ternary op: $op');
    }
  }

  @override
  String toString() => 'TernaryOpExpr($op, $first, $second, $third)';
}

/// Represents a functional bounded loop node in the expression graph.
final class LoopExpr extends Expr {
  final List<Expr> initialValues;
  final Expr maxIterations;
  final Expr conditionExpr;
  final List<Expr> stepExprs;
  final Expr resultExpr;
  final List<VarExpr> stateVars;
  final VarExpr iterVar;
  final String functionName;

  LoopExpr._({
    required this.initialValues,
    required this.maxIterations,
    required this.conditionExpr,
    required this.stepExprs,
    required this.resultExpr,
    required this.stateVars,
    required this.iterVar,
    required this.functionName,
  });

  static int _loopCounter = 0;

  factory LoopExpr({
    required List<Object> initialValues,
    required Object maxIterations,
    required Expr Function(List<Expr> state, Expr iter) condition,
    required List<Expr> Function(List<Expr> state, Expr iter) step,
    Expr Function(List<Expr> state)? result,
    String? name,
  }) {
    final parsedInit = initialValues.map(Expr.from).toList();
    final parsedMax = Expr.from(maxIterations);
    final fnName = name ?? 'fused_loop_${_loopCounter++}';
    final stateVars = List.generate(
      parsedInit.length,
      (i) => VarExpr('${fnName}_s$i'),
    );
    final iterVar = VarExpr('${fnName}_iter');

    final cond = condition(stateVars, iterVar);
    final nextSteps = step(stateVars, iterVar);
    if (nextSteps.length != parsedInit.length) {
      throw ArgumentError(
        'step() returned ${nextSteps.length} values, expected ${parsedInit.length} to match initialValues.',
      );
    }
    final res = result != null ? result(stateVars) : stateVars.first;

    return LoopExpr._(
      initialValues: parsedInit,
      maxIterations: parsedMax,
      conditionExpr: cond,
      stepExprs: nextSteps,
      resultExpr: res,
      stateVars: stateVars,
      iterVar: iterVar,
      functionName: fnName,
    );
  }

  @override
  String toWgsl() {
    final args = variables.toList();
    args.sort((a, b) => a.bindingIndex.compareTo(b.bindingIndex));
    final argList = args.map((v) => '${v.name}_val').join(', ');
    return '$functionName($argList)';
  }

  @override
  Expr grad(VarExpr wrt) => throw UnsupportedError(
    'Analytical differentiation through dynamic LoopExpr is not supported. Use autodiff VJP backprop instead.',
  );

  /// Generates the standalone WGSL helper function for this loop.
  String generateWgslFunction() {
    final args = variables.toList();
    args.sort((a, b) => a.bindingIndex.compareTo(b.bindingIndex));
    final params = args.map((v) => '${v.name}_val: f32').join(', ');

    final initStatements = StringBuffer();
    for (var i = 0; i < initialValues.length; i++) {
      initStatements.writeln(
        '  var ${stateVars[i].name}_val: f32 = ${initialValues[i].toWgsl()};',
      );
    }

    final stepStatements = StringBuffer();
    for (var i = 0; i < stepExprs.length; i++) {
      stepStatements.writeln(
        '    let _next_${stateVars[i].name} = ${stepExprs[i].toWgsl()};',
      );
    }
    for (var i = 0; i < stepExprs.length; i++) {
      stepStatements.writeln(
        '    ${stateVars[i].name}_val = _next_${stateVars[i].name};',
      );
    }

    return '''
fn $functionName($params) -> f32 {
$initStatements  var ${iterVar.name}_val: f32 = 0.0;
  let _max_iter: f32 = ${maxIterations.toWgsl()};
  while (${iterVar.name}_val < _max_iter && ((${conditionExpr.toWgsl()}) > 0.0)) {
$stepStatements    ${iterVar.name}_val += 1.0;
  }
  return ${resultExpr.toWgsl()};
}
''';
  }

  @override
  Set<VarExpr> get variables {
    final localNames = {...stateVars.map((v) => v.name), iterVar.name};
    final allVars = {
      for (final v in initialValues) ...v.variables,
      ...maxIterations.variables,
      ...conditionExpr.variables,
      for (final s in stepExprs) ...s.variables,
      ...resultExpr.variables,
    };
    return allVars.where((v) => !localNames.contains(v.name)).toSet();
  }

  @override
  Set<ScalarParamExpr> get scalarParams => {
    for (final v in initialValues) ...v.scalarParams,
    ...maxIterations.scalarParams,
    ...conditionExpr.scalarParams,
    for (final s in stepExprs) ...s.scalarParams,
    ...resultExpr.scalarParams,
  };

  @override
  int get depth =>
      1 +
      [
        ...initialValues.map((e) => e.depth),
        maxIterations.depth,
        conditionExpr.depth,
        ...stepExprs.map((e) => e.depth),
        resultExpr.depth,
      ].reduce((a, b) => a > b ? a : b);

  @override
  int get nodeCount =>
      1 +
      initialValues.fold<int>(0, (sum, e) => sum + e.nodeCount) +
      maxIterations.nodeCount +
      conditionExpr.nodeCount +
      stepExprs.fold<int>(0, (sum, e) => sum + e.nodeCount) +
      resultExpr.nodeCount;

  @override
  String toFingerprint() =>
      'loop(${initialValues.map((e) => e.toFingerprint()).join(",")};max:${maxIterations.toFingerprint()};cond:${conditionExpr.toFingerprint()};step:${stepExprs.map((e) => e.toFingerprint()).join(",")};res:${resultExpr.toFingerprint()})';

  @override
  String toString() =>
      'LoopExpr($functionName, inits: $initialValues, max: $maxIterations)';
}

/// Descriptor that holds the full configuration for compiling a fused kernel.
final class FusedKernelDescriptor {
  final String name;
  final Expr expression;
  final List<VarExpr> inputs;
  final List<ScalarParamExpr> scalarParams;
  final WgslDType outputDType;
  final bool isStrided;

  FusedKernelDescriptor({
    required this.name,
    Expr? expression,
    Expr? outputExpr,
    List<VarExpr>? inputs,
    List<ScalarParamExpr>? scalarParams,
    this.outputDType = WgslDType.float32,
    this.isStrided = false,
  }) : expression =
           expression ??
           outputExpr ??
           (throw ArgumentError(
             'Either expression or outputExpr must be provided.',
           )),
       inputs = inputs ?? _sortVariables((expression ?? outputExpr!).variables),
       scalarParams =
           scalarParams ?? (expression ?? outputExpr!).scalarParams.toList();

  static List<VarExpr> _sortVariables(Set<VarExpr> vars) {
    final list = vars.toList();
    list.sort((a, b) => a.bindingIndex.compareTo(b.bindingIndex));
    return list;
  }

  static Set<LoopExpr> _collectLoops(Expr expr) {
    final loops = <LoopExpr>{};
    void walk(Expr e) {
      if (e is LoopExpr) {
        loops.add(e);
        for (final init in e.initialValues) {
          walk(init);
        }
        walk(e.maxIterations);
        walk(e.conditionExpr);
        for (final s in e.stepExprs) {
          walk(s);
        }
        walk(e.resultExpr);
      } else if (e is UnaryOpExpr) {
        walk(e.child);
      } else if (e is BinaryOpExpr) {
        walk(e.left);
        walk(e.right);
      } else if (e is TernaryOpExpr) {
        walk(e.first);
        walk(e.second);
        walk(e.third);
      } else if (e is LetExpr) {
        walk(e.value);
        walk(e.body);
      }
    }

    walk(expr);
    return loops;
  }

  static Set<OffsetVarExpr> _collectStencils(Expr expr) {
    final stencils = <OffsetVarExpr>{};
    void walk(Expr e) {
      if (e is OffsetVarExpr) {
        stencils.add(e);
      } else if (e is UnaryOpExpr) {
        walk(e.child);
      } else if (e is BinaryOpExpr) {
        walk(e.left);
        walk(e.right);
      } else if (e is TernaryOpExpr) {
        walk(e.first);
        walk(e.second);
        walk(e.third);
      } else if (e is LetExpr) {
        walk(e.value);
        walk(e.body);
      } else if (e is LoopExpr) {
        for (final init in e.initialValues) {
          walk(init);
        }
        walk(e.maxIterations);
        walk(e.conditionExpr);
        for (final s in e.stepExprs) {
          walk(s);
        }
        walk(e.resultExpr);
      }
    }

    walk(expr);
    return stencils;
  }

  static List<LetExpr> _collectLets(Expr expr) {
    final lets = <LetExpr>[];
    void walk(Expr e) {
      if (e is LetExpr) {
        walk(e.value);
        lets.add(e);
        walk(e.body);
      } else if (e is UnaryOpExpr) {
        walk(e.child);
      } else if (e is BinaryOpExpr) {
        walk(e.left);
        walk(e.right);
      } else if (e is TernaryOpExpr) {
        walk(e.first);
        walk(e.second);
        walk(e.third);
      } else if (e is LoopExpr) {
        for (final init in e.initialValues) {
          walk(init);
        }
        walk(e.maxIterations);
        walk(e.conditionExpr);
        for (final s in e.stepExprs) {
          walk(s);
        }
        walk(e.resultExpr);
      }
    }

    walk(expr);
    return lets;
  }

  /// Produces a deterministic unique cache key for this kernel.
  String generateCacheKey() =>
      'fused_${expression.toFingerprint()}_${outputDType.wgslType}_strided:$isStrided';

  /// Generates the resource binding descriptors.
  List<WgslBinding> createBindings() {
    final bindings = <WgslBinding>[];
    var bindIdx = 0;

    for (final v in inputs) {
      bindings.add(
        WgslBinding(
          group: 0,
          binding: bindIdx++,
          name: v.name,
          dtype: v.dtype,
          access: WgslBufferAccess.read,
        ),
      );
    }

    // Output destination buffer
    bindings.add(
      WgslBinding(
        group: 0,
        binding: bindIdx++,
        name: 'dst',
        dtype: outputDType,
        access: WgslBufferAccess.readWrite,
      ),
    );

    // Uniforms struct
    bindings.add(
      WgslBinding(
        group: 0,
        binding: bindIdx++,
        name: isStrided ? 'meta' : 'uniforms',
        isUniform: true,
        customTypeName: isStrided ? 'StridedMetadata' : 'FusedUniforms',
      ),
    );

    return bindings;
  }

  /// Generates the full WGSL compute shader source code for this fused kernel.
  String generateWgslSource({int workgroupSize = 256, bool enableCse = true}) {
    final bindings = createBindings();
    final wgSize = WgslWorkgroupSize(workgroupSize, 1, 1);
    final bufferDeclarations = bindings
        .map((b) => b.toWgslDeclaration())
        .join('\n');
    final effectiveExpr = enableCse
        ? expression.eliminateCommonSubexpressions()
        : expression;
    final loopFunctions = _collectLoops(
      effectiveExpr,
    ).map((l) => l.generateWgslFunction()).join('\n');
    final stencilFunctions = _collectStencils(
      effectiveExpr,
    ).map((s) => s.generateWgslFunction()).join('\n');
    final letStatements = _collectLets(
      effectiveExpr,
    ).map((l) => '    let ${l.name} = ${l.value.toWgsl()};').join('\n');

    if (!isStrided) {
      // Contiguous 1D fast path
      final loadStatements = inputs
          .map((v) => '  let ${v.name}_val = ${v.name}[idx];')
          .join('\n');

      final uniformFields = StringBuffer('  total_elements: u32,\n');
      for (final sp in scalarParams) {
        uniformFields.writeln('  ${sp.name}: f32,');
      }
      final fieldCount = 1 + scalarParams.length;
      final padNeeded = (4 - (fieldCount % 4)) % 4;
      for (var i = 0; i < padNeeded; i++) {
        uniformFields.writeln('  pad$i: u32,');
      }

      return '''
// WGSL JIT Fused Compute Shader: $name (Contiguous)
// Expression: ${expression.toFingerprint()}
struct FusedUniforms {
$uniformFields}

$bufferDeclarations

${WgslTemplates.mathHelpers}

$stencilFunctions

$loopFunctions

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  var idx = global_id.x;
  let stride = num_workgroups.x * ${workgroupSize}u;
  while (idx < uniforms.total_elements) {
$loadStatements
$letStatements
    let result = ${effectiveExpr.toWgsl()};
    dst[idx] = result;
    idx += stride;
  }
}
''';
    } else {
      // Strided multidimensional path
      final loadStatements = inputs
          .asMap()
          .entries
          .map((entry) {
            final offsetVar = entry.key == 0
                ? 'off_a'
                : (entry.key == 1 ? 'off_b' : 'off_a');
            return '  let ${entry.value.name}_val = ${entry.value.name}[$offsetVar];';
          })
          .join('\n');

      return '''
// WGSL JIT Fused Compute Shader: $name (Strided)
// Expression: ${expression.toFingerprint()}
${WgslTemplates.stridedHeader}

$bufferDeclarations

${WgslTemplates.mathHelpers}

$stencilFunctions

$loopFunctions

@compute ${wgSize.toAttribute()}
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= meta.total_elements) {
    return;
  }

  var off_a: u32 = 0u;
  var off_b: u32 = 0u;
  var off_dst: u32 = 0u;
  flat_to_strided_offsets(idx, meta, &off_a, &off_b, &off_dst);

$loadStatements
$letStatements
  let result = ${effectiveExpr.toWgsl()};
  dst[off_dst] = result;
}
''';
    }
  }

  /// Packages this fused kernel and its input tensors into an interactive [WebGpuWidget].
  WebGpuWidget createBrowserWidget({
    List<dynamic> inputArrays = const [],
    required List<dynamic> outputShape,
    String? title,
    List<dynamic> sliders = const [],
    bool renderToCanvas = false,
    int canvasWidth = 512,
    int canvasHeight = 512,
    String colorMap = 'viridis',
  }) {
    final parsedInputs = inputArrays.map((e) => e as GpuArray).toList();
    final parsedShape = outputShape.map((e) => (e as num).toInt()).toList();
    final parsedSliders = sliders.map((e) => e as WebGpuSlider).toList();

    final wgslSource = generateWgslSource();
    final inputPayloads = <GpuBufferPayload>[];

    for (var i = 0; i < parsedInputs.length; i++) {
      final arr = parsedInputs[i];
      final rawND = arr.toNDArray();
      final rawList = rawND.toList();
      final f32List = Float32List.fromList(
        rawList.map((e) => (e as num).toDouble()).toList(),
      );
      final base64Payload = base64Encode(f32List.buffer.asUint8List());
      rawND.dispose();

      inputPayloads.add(
        GpuBufferPayload(
          bindingIndex: i,
          name: inputs.length > i ? inputs[i].name : 'input_$i',
          dtype: arr.dtype,
          shape: arr.shape,
          base64Data: base64Payload,
          sizeInBytes: arr.buffer.sizeInBytes,
        ),
      );
    }

    final totalOut = parsedShape.reduce((a, b) => a * b);
    final outBytes = totalOut * outputDType.byteSize;

    final outputPayload = GpuBufferPayload(
      bindingIndex: parsedInputs.length,
      name: 'dst',
      dtype: DType.values.byName(
        outputDType.wgslType == 'f32' ? 'float32' : 'float16',
      ),
      shape: parsedShape,
      isOutput: true,
      sizeInBytes: outBytes,
    );

    final scalarList = scalarParams.toList();
    final uniformWords = <int>[totalOut];
    final byteData = ByteData(4);
    for (final sp in scalarList) {
      final matchingSlider = parsedSliders.cast<WebGpuSlider?>().firstWhere(
        (s) => s?.name == sp.name,
        orElse: () => null,
      );
      final initialVal = matchingSlider?.initialValue ?? sp.defaultValue;
      byteData.setFloat32(0, initialVal, Endian.little);
      uniformWords.add(byteData.getUint32(0, Endian.little));
    }
    while (uniformWords.length % 4 != 0) {
      uniformWords.add(0);
    }

    final resolvedSliders = parsedSliders.map((slider) {
      final paramIdx = scalarList.indexWhere((sp) => sp.name == slider.name);
      return WebGpuSlider(
        name: slider.name,
        label: slider.label,
        min: slider.min,
        max: slider.max,
        initialValue: slider.initialValue,
        step: slider.step,
        // In FusedKernelDescriptor, all scalar parameters in FusedUniforms are f32 in WGSL.
        isInteger: false,
        uniformWordIndex: paramIdx != -1
            ? paramIdx + 1
            : slider.uniformWordIndex,
      );
    }).toList();

    final pkg = GpuComputePipelinePackage(
      name: name,
      wgslCode: wgslSource,
      inputs: inputPayloads,
      output: outputPayload,
      uniforms: uniformWords,
      sliders: resolvedSliders,
      renderToCanvas: renderToCanvas,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight,
      colorMap: colorMap,
    );

    return WebGpuWidget(pkg, title: title ?? name);
  }
}
