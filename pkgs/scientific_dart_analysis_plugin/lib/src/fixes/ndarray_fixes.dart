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

import 'package:analysis_server_plugin/edit/dart/correction_producer.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/token.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/source/source_range.dart';
import 'package:analyzer_plugin/utilities/change_builder/change_builder_core.dart';
import 'package:analyzer_plugin/utilities/fixes/fixes.dart';

import '../type_utils.dart';

/// Quick fix that replaces `a == b` with `a.equals(b)` (or `!a.equals(b)` for
/// `a != b`).
final class ReplaceWithEqualsFix extends ResolvedCorrectionProducer {
  static const FixKind _replaceWithEqualsKind = FixKind(
    'scientific_dart_analysis_plugin.fix.replaceWithEquals',
    50,
    'Replace with .equals() (structural equality)',
  );

  ReplaceWithEqualsFix({required super.context});

  @override
  CorrectionApplicability get applicability =>
      CorrectionApplicability.singleLocation;

  @override
  FixKind get fixKind => _replaceWithEqualsKind;

  @override
  Future<void> compute(ChangeBuilder builder) async {
    final targetNode = node.thisOrAncestorOfType<BinaryExpression>();
    if (targetNode == null) return;
    final leftOperand = targetNode.leftOperand;
    final leftSrc = _needsReceiverParentheses(leftOperand)
        ? '(${leftOperand.toSource()})'
        : leftOperand.toSource();
    final rightSrc = targetNode.rightOperand.toSource();
    final isNegated = targetNode.operator.type == TokenType.BANG_EQ;
    final replacement = isNegated
        ? '!$leftSrc.equals($rightSrc)'
        : '$leftSrc.equals($rightSrc)';

    await builder.addDartFileEdit(file, (builder) {
      builder.addSimpleReplacement(
        SourceRange(targetNode.offset, targetNode.length),
        replacement,
      );
    });
  }
}

bool _needsReceiverParentheses(Expression expr) =>
    expr is BinaryExpression ||
    expr is PrefixExpression ||
    expr is ConditionalExpression ||
    expr is AsExpression ||
    expr is IsExpression ||
    expr is CascadeExpression ||
    expr is AwaitExpression ||
    expr is AssignmentExpression;

/// Quick fix that replaces `a == b` with `identical(a, b)` (or `!identical(a, b)`
/// for `a != b`).
final class ReplaceWithIdenticalFix extends ResolvedCorrectionProducer {
  static const FixKind _replaceWithIdenticalKind = FixKind(
    'scientific_dart_analysis_plugin.fix.replaceWithIdentical',
    49,
    'Replace with identical(a, b) (explicit reference identity)',
  );

  ReplaceWithIdenticalFix({required super.context});

  @override
  CorrectionApplicability get applicability =>
      CorrectionApplicability.singleLocation;

  @override
  FixKind get fixKind => _replaceWithIdenticalKind;

  @override
  Future<void> compute(ChangeBuilder builder) async {
    final targetNode = node.thisOrAncestorOfType<BinaryExpression>();
    if (targetNode == null) return;
    final leftSrc = targetNode.leftOperand.toSource();
    final rightSrc = targetNode.rightOperand.toSource();
    final isNegated = targetNode.operator.type == TokenType.BANG_EQ;
    final replacement = isNegated
        ? '!identical($leftSrc, $rightSrc)'
        : 'identical($leftSrc, $rightSrc)';

    await builder.addDartFileEdit(file, (builder) {
      builder.addSimpleReplacement(
        SourceRange(targetNode.offset, targetNode.length),
        replacement,
      );
    });
  }
}

/// Quick fix that appends `.detachToParentScope()` to an unescaped `NDArray`
/// returned from `NDArray.scope`, or `.copy().detachToParentScope()` when the
/// returned value is a view (views cannot be detached).
final class AddDetachToParentScopeFix extends ResolvedCorrectionProducer {
  static const FixKind _addDetachKind = FixKind(
    'scientific_dart_analysis_plugin.fix.addDetachToParentScope',
    50,
    'Add .detachToParentScope()',
  );

  AddDetachToParentScopeFix({required super.context});

  @override
  CorrectionApplicability get applicability =>
      CorrectionApplicability.singleLocation;

  @override
  FixKind get fixKind => _addDetachKind;

  @override
  Future<void> compute(ChangeBuilder builder) async {
    final expr = node is Expression
        ? node as Expression
        : node.thisOrAncestorOfType<Expression>();
    if (expr == null) return;
    final unwrapped = unwrapParenthesized(expr);
    if (unwrapped is RecordLiteral || unwrapped is ListLiteral) return;

    final body = expr.thisOrAncestorOfType<FunctionBody>();
    final decls = body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
    final isView =
        traceRootArrayAndView(expr, decls).throughView ||
        isViewProducingExpression(expr, decls);
    final insertion = isView
        ? '.copy().detachToParentScope()'
        : '.detachToParentScope()';

    await builder.addDartFileEdit(file, (builder) {
      if (_needsReceiverParentheses(expr)) {
        builder.addSimpleReplacement(
          SourceRange(expr.offset, expr.length),
          '(${expr.toSource()})$insertion',
        );
      } else {
        builder.addSimpleInsertion(expr.end, insertion);
      }
    });
  }
}

/// Quick fix that inserts `.copy()` on a view returned from `NDArray.returning`
/// or before `.detachToParentScope()` / `.detachFromScope()`.
final class AddCopyBeforeViewLifecycleFix extends ResolvedCorrectionProducer {
  static const FixKind _addCopyKind = FixKind(
    'scientific_dart_analysis_plugin.fix.addCopyBeforeViewLifecycle',
    50,
    'Materialize an owning copy with .copy()',
  );

  AddCopyBeforeViewLifecycleFix({required super.context});

  @override
  CorrectionApplicability get applicability =>
      CorrectionApplicability.singleLocation;

  @override
  FixKind get fixKind => _addCopyKind;

  @override
  Future<void> compute(ChangeBuilder builder) async {
    final returningOrScope = _findEnclosingInvocationInSameFunction(
      node,
      (inv) => isReturningInvocation(inv) || isScopeInvocation(inv),
    );
    if (returningOrScope != null) {
      final returnedView = _findReturnedViewInScopeCall(returningOrScope);
      if (returnedView != null) {
        await _appendCopyToExpression(builder, returnedView);
        return;
      }
    }

    final detachInvocation = _findEnclosingInvocationInSameFunction(
      node,
      (inv) =>
          inv.methodName.name == 'detachToParentScope' ||
          inv.methodName.name == 'detachFromScope',
    );
    if (detachInvocation != null) {
      final receiver = detachInvocation.realTarget;
      if (receiver != null) {
        await _appendCopyToExpression(builder, receiver);
        return;
      }
    }

    final returnStmt = _findEnclosingReturnInSameFunction(node);
    final target =
        returnStmt?.expression ??
        (node is Expression
            ? node as Expression
            : node.thisOrAncestorOfType<Expression>());
    if (target != null) {
      await _appendCopyToExpression(builder, target);
    }
  }

  Future<void> _appendCopyToExpression(
    ChangeBuilder builder,
    Expression target,
  ) async {
    await builder.addDartFileEdit(file, (builder) {
      if (_needsReceiverParentheses(target)) {
        builder.addSimpleReplacement(
          SourceRange(target.offset, target.length),
          '(${target.toSource()}).copy()',
        );
      } else {
        builder.addSimpleInsertion(target.end, '.copy()');
      }
    });
  }

  MethodInvocation? _findEnclosingInvocationInSameFunction(
    AstNode start,
    bool Function(MethodInvocation) predicate,
  ) {
    AstNode? current = start;
    while (current != null) {
      if (current is FunctionExpression ||
          current is FunctionDeclaration ||
          current is MethodDeclaration) {
        return null;
      }
      if (current is MethodInvocation && predicate(current)) {
        return current;
      }
      current = current.parent;
    }
    return null;
  }

  ReturnStatement? _findEnclosingReturnInSameFunction(AstNode start) {
    AstNode? current = start;
    while (current != null) {
      if (current is FunctionExpression ||
          current is FunctionDeclaration ||
          current is MethodDeclaration) {
        return null;
      }
      if (current is ReturnStatement) {
        return current;
      }
      current = current.parent;
    }
    return null;
  }

  Expression? _findReturnedViewInScopeCall(MethodInvocation invocation) {
    final args = invocation.argumentList.arguments;
    if (args.isEmpty) return null;
    final callback = unwrapParenthesized(args.first.argumentExpression);
    if (callback is! FunctionExpression) return null;
    final decls = SubtreeDeclarations.collect(callback);
    final body = callback.body;
    if (body is ExpressionFunctionBody) {
      return body.expression;
    }
    if (body is BlockFunctionBody) {
      final finder = _ReturnedViewExpressionFinder(callback, decls);
      body.block.accept(finder);
      return finder.foundView ?? finder.firstReturned;
    }
    return null;
  }
}

final class _ReturnedViewExpressionFinder extends RecursiveAstVisitor<void> {
  final FunctionExpression callback;
  final SubtreeDeclarations decls;
  Expression? foundView;
  Expression? firstReturned;

  _ReturnedViewExpressionFinder(this.callback, this.decls);

  @override
  void visitFunctionExpression(FunctionExpression node) {
    if (!identical(node, callback)) return;
    super.visitFunctionExpression(node);
  }

  @override
  void visitFunctionDeclaration(FunctionDeclaration node) {}

  @override
  void visitReturnStatement(ReturnStatement node) {
    final expr = node.expression;
    if (expr != null) {
      firstReturned ??= expr;
      if (foundView == null && isViewProducingExpression(expr, decls)) {
        foundView = expr;
      }
    }
    super.visitReturnStatement(node);
  }
}

/// Quick fix that rewrites `a < b` on `NDArray<Uint64>` elements to
/// `uint64Compare(a, b) < 0`, or `a.compareTo(b)` to `uint64Compare(a, b)`.
final class ReplaceWithUint64CompareFix extends ResolvedCorrectionProducer {
  static const FixKind _uint64CompareKind = FixKind(
    'scientific_dart_analysis_plugin.fix.replaceWithUint64Compare',
    50,
    'Replace with uint64Compare(a, b)',
  );

  ReplaceWithUint64CompareFix({required super.context});

  @override
  CorrectionApplicability get applicability =>
      CorrectionApplicability.singleLocation;

  @override
  FixKind get fixKind => _uint64CompareKind;

  @override
  Future<void> compute(ChangeBuilder builder) async {
    final compareToNode = node.thisOrAncestorMatching(
      (n) =>
          n is MethodInvocation &&
          n.methodName.name == 'compareTo' &&
          n.target != null &&
          n.argumentList.arguments.length == 1,
    );
    if (compareToNode is MethodInvocation) {
      final leftSrc = compareToNode.realTarget!.toSource();
      final rightSrc = compareToNode
          .argumentList
          .arguments
          .first
          .argumentExpression
          .toSource();
      await builder.addDartFileEdit(file, (builder) {
        builder.addSimpleReplacement(
          SourceRange(compareToNode.offset, compareToNode.length),
          'uint64Compare($leftSrc, $rightSrc)',
        );
      });
      return;
    }

    final binary = node.thisOrAncestorOfType<BinaryExpression>();
    if (binary == null) return;
    final leftSrc = binary.leftOperand.toSource();
    final rightSrc = binary.rightOperand.toSource();
    final opLexeme = binary.operator.lexeme;
    final replacement = 'uint64Compare($leftSrc, $rightSrc) $opLexeme 0';

    await builder.addDartFileEdit(file, (builder) {
      builder.addSimpleReplacement(
        SourceRange(binary.offset, binary.length),
        replacement,
      );
    });
  }
}
