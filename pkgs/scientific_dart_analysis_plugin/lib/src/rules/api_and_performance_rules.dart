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

import 'package:analyzer/analysis_rule/analysis_rule.dart';
import 'package:analyzer/analysis_rule/rule_context.dart';
import 'package:analyzer/analysis_rule/rule_visitor_registry.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/token.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/error/error.dart';

import '../type_utils.dart';

// =============================================================================
// 1. ndarray_equality_operator
// =============================================================================

/// Flags `a == b` and `a != b` when both operands are `NDArray`s.
final class EqualityOperatorRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_equality_operator',
    'NDArray.operator == checks reference identity (identical(a, b)), not '
        'structural or element-wise equality.',
    correctionMessage:
        'Use a.equals(b), arrayEqual(a, b), or allClose(a, b) for structural '
        'equality, equal(a, b) for element-wise NDArray<Boolean>, or '
        'identical(a, b) if reference identity is intended.',
    severity: DiagnosticSeverity.WARNING,
  );

  EqualityOperatorRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Avoid using == or != on NDArray instances; use .equals(), '
            'arrayEqual(), allClose(), equal(), or identical().',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _EqualityOperatorVisitor(this);
    registry.addBinaryExpression(this, visitor);
  }
}

final class _EqualityOperatorVisitor extends SimpleAstVisitor<void> {
  final EqualityOperatorRule rule;

  _EqualityOperatorVisitor(this.rule);

  @override
  void visitBinaryExpression(BinaryExpression node) {
    final op = node.operator.type;
    if (op != TokenType.EQ_EQ && op != TokenType.BANG_EQ) return;
    if (node.leftOperand is NullLiteral || node.rightOperand is NullLiteral) {
      return;
    }
    if (isNDArrayType(node.leftOperand.staticType) &&
        isNDArrayType(node.rightOperand.staticType)) {
      rule.reportAtNode(node);
    }
  }
}

// =============================================================================
// 2. ndarray_uint64_signed_comparison
// =============================================================================

/// Flags relational comparisons (`<`, `<=`, `>`, `>=`, `.compareTo`) on scalar
/// elements extracted from an `NDArray<Uint64>`.
final class Uint64SignedComparisonRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_uint64_signed_comparison',
    'Dart int is signed 64-bit; relational operators (<, <=, >, >=) on '
        'NDArray<Uint64> elements treat values >= 2^63 as negative.',
    correctionMessage:
        'Use uint64Compare(a, b) (from package:ndarray/ndarray.dart) for '
        'unsigned 64-bit comparisons.',
    severity: DiagnosticSeverity.WARNING,
  );

  Uint64SignedComparisonRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Use uint64Compare() instead of signed relational operators on '
            'NDArray<Uint64> elements.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _Uint64SignedComparisonVisitor(this);
    registry.addBinaryExpression(this, visitor);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _Uint64SignedComparisonVisitor extends SimpleAstVisitor<void> {
  final Uint64SignedComparisonRule rule;

  _Uint64SignedComparisonVisitor(this.rule);

  @override
  void visitBinaryExpression(BinaryExpression node) {
    final op = node.operator.type;
    if (op != TokenType.LT &&
        op != TokenType.LT_EQ &&
        op != TokenType.GT &&
        op != TokenType.GT_EQ) {
      return;
    }
    final decls = _collectEnclosingDeclarations(node);
    if (_isUint64ElementExpression(node.leftOperand, decls) ||
        _isUint64ElementExpression(node.rightOperand, decls)) {
      rule.reportAtNode(node);
    }
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name != 'compareTo') return;
    final target = node.realTarget;
    if (target == null) return;
    final decls = _collectEnclosingDeclarations(node);
    final args = node.argumentList.arguments;
    if (_isUint64ElementExpression(target, decls) ||
        (args.isNotEmpty &&
            _isUint64ElementExpression(args.first.argumentExpression, decls))) {
      rule.reportAtNode(node);
    }
  }

  SubtreeDeclarations _collectEnclosingDeclarations(AstNode node) {
    final body = node.thisOrAncestorOfType<FunctionBody>();
    return body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
  }

  static const Set<String> _uint64CellGetters = {
    'getCell',
    'getCell1D',
    'getCell2D',
    'getCell3D',
    'getCellFlat',
    'getCellRaw',
  };

  bool _isUint64ElementExpression(
    Expression expr,
    SubtreeDeclarations decls, {
    int depth = 0,
  }) {
    if (depth > 4) return false;
    var unwrapped = unwrapParenthesized(expr);
    while (unwrapped is AsExpression) {
      unwrapped = unwrapParenthesized(unwrapped.expression);
    }
    if (unwrapped is IndexExpression) {
      final target = unwrapParenthesized(unwrapped.realTarget);
      if (isNDArrayUint64Type(target.staticType)) {
        return true;
      }
      if (target is PropertyAccess &&
          (target.propertyName.name == 'data' ||
              target.propertyName.name == 'dataRaw') &&
          isNDArrayUint64Type(target.realTarget.staticType)) {
        return true;
      }
      if (target is PrefixedIdentifier &&
          (target.identifier.name == 'data' ||
              target.identifier.name == 'dataRaw') &&
          isNDArrayUint64Type(target.prefix.staticType)) {
        return true;
      }
    } else if (unwrapped is MethodInvocation) {
      if (_uint64CellGetters.contains(unwrapped.methodName.name) &&
          isNDArrayUint64Type(unwrapped.realTarget?.staticType)) {
        return true;
      }
    } else if (unwrapped is PropertyAccess) {
      if (unwrapped.propertyName.name == 'scalar' &&
          isNDArrayUint64Type(unwrapped.realTarget.staticType)) {
        return true;
      }
    } else if (unwrapped is PrefixedIdentifier) {
      if (unwrapped.identifier.name == 'scalar' &&
          isNDArrayUint64Type(unwrapped.prefix.staticType)) {
        return true;
      }
    } else if (unwrapped is SimpleIdentifier) {
      final element = unwrapped.element;
      if (element != null) {
        final decl = decls.variableDeclarations[element];
        final init = decl?.initializer;
        if (init != null) {
          return _isUint64ElementExpression(init, decls, depth: depth + 1);
        }
      }
    }
    return false;
  }
}

// =============================================================================
// 3. ndarray_broadcast_view_as_out
// =============================================================================

/// Flags passing a `broadcastTo` view as an `out:` argument.
///
/// Broadcast views have `isWriteable == false`, so operations reject them as
/// `out:` at runtime.
final class BroadcastViewAsOutRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_broadcast_view_as_out',
    'broadcastTo() returns a read-only view (isWriteable is false); passing '
        'it as out: throws at runtime.',
    correctionMessage:
        'Allocate a writable array (e.g., zeros(...) or empty(...)) or call '
        '.copy() on the broadcast view before passing it as out:.',
    severity: DiagnosticSeverity.WARNING,
  );

  BroadcastViewAsOutRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not pass read-only broadcast views (stride == 0) as out: '
            'arguments.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _BroadcastViewAsOutVisitor(this);
    registry.addNamedArgument(this, visitor);
  }
}

final class _BroadcastViewAsOutVisitor extends SimpleAstVisitor<void> {
  final BroadcastViewAsOutRule rule;

  _BroadcastViewAsOutVisitor(this.rule);

  @override
  void visitNamedArgument(NamedArgument node) {
    if (node.name.lexeme != 'out') return;
    final body = node.thisOrAncestorOfType<FunctionBody>();
    final decls = body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
    if (isBroadcastViewExpression(node.argumentExpression, decls)) {
      rule.reportAtNode(node.argumentExpression);
    }
  }
}

// =============================================================================
// 4. ndarray_hot_loop_element_indexing
// =============================================================================

/// Flags scalar reads `arr[[i, j]]` inside nested loops.
///
/// Each read allocates an index list, crosses into typed-data accessors, and
/// boxes the result; vectorized operations or `NDIter` are usually much
/// faster.
final class HotLoopElementIndexingRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_hot_loop_element_indexing',
    'Element-by-element NDArray indexing inside nested loops allocates an '
        'index list and boxes a scalar on every access.',
    correctionMessage:
        'Prefer vectorized NDArray operations, slicing views, or NDIter for '
        'multi-dimensional traversal.',
    severity: DiagnosticSeverity.INFO,
  );

  HotLoopElementIndexingRule()
    : super(
        name: code.lowerCaseName,
        description: 'Avoid per-element NDArray[[i, j]] reads in nested loops.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _HotLoopElementIndexingVisitor(this);
    registry.addIndexExpression(this, visitor);
  }
}

final class _HotLoopElementIndexingVisitor extends SimpleAstVisitor<void> {
  final HotLoopElementIndexingRule rule;

  _HotLoopElementIndexingVisitor(this.rule);

  @override
  void visitIndexExpression(IndexExpression node) {
    if (!isNDArrayType(node.realTarget.staticType)) return;
    // Writes (`a[[i, j]] = v`) are often unavoidable when filling results.
    if (node.inSetterContext()) return;
    if (_enclosingLoopDepth(node) >= 2) {
      rule.reportAtNode(node);
    }
  }

  int _enclosingLoopDepth(AstNode node) {
    var depth = 0;
    AstNode? current = node.parent;
    while (current != null && current is! FunctionBody) {
      if (current is ForStatement ||
          current is WhileStatement ||
          current is DoStatement) {
        depth++;
      }
      current = current.parent;
    }
    return depth;
  }
}
