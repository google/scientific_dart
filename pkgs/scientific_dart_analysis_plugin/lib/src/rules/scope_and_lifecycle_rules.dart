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
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/error/error.dart';

import '../type_utils.dart';

// =============================================================================
// 1. ndarray_unescaped_scope_return
// =============================================================================

/// Flags `NDArray` / `ScopedResource` instances allocated inside
/// `NDArray.scope` or `ResourceScope.scope` (or views of them) that escape via
/// `return` or outer assignment without calling `.detachToParentScope()` or
/// `.detachFromScope()`.
///
/// Only values that look allocated in the scope callback are flagged:
/// constructors, operators, and non-view calls without `out:`. Parameters,
/// outer variables, and results of `reshape`/`ravel` (which may or may not be
/// views) are not flagged.
final class UnescapedScopeReturnRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_unescaped_scope_return',
    'This array (or the array it is a view of) was allocated inside '
        'NDArray.scope and is disposed when the scope exits, so it escapes '
        'as freed memory.',
    correctionMessage:
        'Call .detachToParentScope() on the returned array (use '
        '.copy().detachToParentScope() for views), or use NDArray.returning.',
    severity: DiagnosticSeverity.WARNING,
  );

  UnescapedScopeReturnRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not return or escape a ScopedResource allocated inside '
            'NDArray.scope without detaching it from the scope.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _UnescapedScopeReturnVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _UnescapedScopeReturnVisitor extends SimpleAstVisitor<void> {
  final UnescapedScopeReturnRule rule;

  _UnescapedScopeReturnVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (!isScopeInvocation(node)) return;
    final args = node.argumentList.arguments;
    if (args.isEmpty) return;
    final callback = unwrapParenthesized(args.first.argumentExpression);
    if (callback is! FunctionExpression) return;

    final decls = SubtreeDeclarations.collect(callback);
    final body = callback.body;

    if (body is ExpressionFunctionBody) {
      _checkReturnedExpression(body.expression, callback, decls);
    } else if (body is BlockFunctionBody) {
      body.block.accept(_ScopeBodyEscapeVisitor(rule, callback, decls));
    }
  }

  void _checkReturnedExpression(
    Expression expr,
    FunctionExpression scopeCallback,
    SubtreeDeclarations decls,
  ) {
    if (_isUnescapedLocalResource(expr, scopeCallback, decls)) {
      rule.reportAtNode(expr);
    }
  }
}

bool _isUnescapedLocalResource(
  Expression expr,
  FunctionExpression scopeCallback,
  SubtreeDeclarations decls,
) {
  final unwrapped = unwrapParenthesized(expr);
  if (!containsScopedResourceType(unwrapped.staticType)) return false;
  if (isDirectlyDetachedExpression(unwrapped)) return false;

  if (unwrapped is RecordLiteral) {
    for (final field in unwrapped.fields) {
      final valueExpr = field.fieldExpression;
      if (_isUnescapedLocalResource(valueExpr, scopeCallback, decls)) {
        return true;
      }
    }
    return false;
  }

  if (unwrapped is ListLiteral) {
    for (final elem in unwrapped.elements) {
      if (elem is Expression &&
          _isUnescapedLocalResource(elem, scopeCallback, decls)) {
        return true;
      }
    }
    return false;
  }

  // A view of an array allocated inside the scope dangles once the scope
  // disposes the parent, even though the view itself is not tracked.
  final trace = traceRootArrayAndView(unwrapped, decls);
  if (trace.throughView) {
    final root = trace.rootElement;
    if (root == null || !decls.elements.contains(root)) return false;
    return _isUndetachedFreshLocal(root, scopeCallback, decls, expr.offset);
  }

  if (unwrapped is SimpleIdentifier) {
    final element = unwrapped.element;
    if (element == null) return false;
    return _isUndetachedFreshLocal(element, scopeCallback, decls, expr.offset);
  }

  // Direct allocations: constructors, operators, and non-view calls without
  // `out:`.
  return isFreshScopedResourceAllocation(unwrapped);
}

/// Whether [element] is a local of [scopeCallback] initialized with a fresh
/// allocation that has not been detached before [offset].
bool _isUndetachedFreshLocal(
  Element element,
  FunctionExpression scopeCallback,
  SubtreeDeclarations decls,
  int offset,
) {
  // Declared outside the scope callback (e.g. an `out` parameter or outer
  // variable): the inner scope does not dispose it.
  if (!decls.elements.contains(element)) return false;
  final init = decls.variableDeclarations[element]?.initializer;
  if (init == null) return false;
  if (isDirectlyDetachedExpression(init)) return false;
  if (!isFreshScopedResourceAllocation(init)) return false;
  return !wasElementDetachedBefore(element, scopeCallback.body, offset);
}

final class _ScopeBodyEscapeVisitor extends RecursiveAstVisitor<void> {
  final UnescapedScopeReturnRule rule;
  final FunctionExpression scopeCallback;
  final SubtreeDeclarations decls;

  _ScopeBodyEscapeVisitor(this.rule, this.scopeCallback, this.decls);

  @override
  void visitFunctionExpression(FunctionExpression node) {
    // Do not descend into nested closures (including nested NDArray.scope),
    // which have their own return target.
    if (!identical(node, scopeCallback)) return;
    super.visitFunctionExpression(node);
  }

  @override
  void visitFunctionDeclaration(FunctionDeclaration node) {
    // Nested local function has its own return target.
  }

  @override
  void visitReturnStatement(ReturnStatement node) {
    final expr = node.expression;
    if (expr != null && _isUnescapedLocalResource(expr, scopeCallback, decls)) {
      rule.reportAtNode(expr);
    }
    super.visitReturnStatement(node);
  }

  @override
  void visitAssignmentExpression(AssignmentExpression node) {
    final lhs = node.leftHandSide;
    if (lhs is SimpleIdentifier) {
      final lhsElement = lhs.element;
      if (lhsElement != null &&
          !decls.elements.contains(lhsElement) &&
          _isUnescapedLocalResource(node.rightHandSide, scopeCallback, decls)) {
        rule.reportAtNode(node.rightHandSide);
      }
    }
    super.visitAssignmentExpression(node);
  }
}

// =============================================================================
// 2. ndarray_view_lifecycle_misuse
// =============================================================================

/// Flags calling `.detachToParentScope()` or `.detachFromScope()` on a view
/// (`isView == true`), or returning a view from `NDArray.returning`. All of
/// these throw a `StateError` at runtime.
///
/// `dispose()` on a view is a documented no-op and is not flagged.
final class ViewLifecycleMisuseRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_view_lifecycle_misuse',
    'Views do not own native memory; detaching a view or returning one from '
        'NDArray.returning throws a StateError at runtime.',
    correctionMessage:
        'Materialize an owning copy with .copy() before detaching/returning, '
        'or manage the lifetime of the owning parent array instead.',
    severity: DiagnosticSeverity.WARNING,
  );

  ViewLifecycleMisuseRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not call detachToParentScope() or detachFromScope() on an '
            'NDArray view, or return a view from NDArray.returning.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _ViewLifecycleMisuseVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _ViewLifecycleMisuseVisitor extends SimpleAstVisitor<void> {
  final ViewLifecycleMisuseRule rule;

  _ViewLifecycleMisuseVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final name = node.methodName.name;

    if (name == 'detachToParentScope' || name == 'detachFromScope') {
      final enclosingBody = node.thisOrAncestorOfType<FunctionBody>();
      final decls = enclosingBody != null
          ? SubtreeDeclarations.collect(enclosingBody)
          : SubtreeDeclarations();
      final target = node.realTarget;
      if (target != null && isViewProducingExpression(target, decls)) {
        rule.reportAtNode(node);
        return;
      }
    }

    if (isReturningInvocation(node)) {
      final args = node.argumentList.arguments;
      if (args.isEmpty) return;
      final callback = unwrapParenthesized(args.first.argumentExpression);
      if (callback is! FunctionExpression) return;
      final cbDecls = SubtreeDeclarations.collect(callback);
      final body = callback.body;
      if (body is ExpressionFunctionBody) {
        if (isViewProducingExpression(body.expression, cbDecls)) {
          rule.reportAtNode(body.expression);
        }
      } else if (body is BlockFunctionBody) {
        body.block.accept(_ReturningViewVisitor(rule, callback, cbDecls));
      }
    }
  }
}

final class _ReturningViewVisitor extends RecursiveAstVisitor<void> {
  final ViewLifecycleMisuseRule rule;
  final FunctionExpression callback;
  final SubtreeDeclarations decls;

  _ReturningViewVisitor(this.rule, this.callback, this.decls);

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
    if (expr != null && isViewProducingExpression(expr, decls)) {
      rule.reportAtNode(expr);
    }
    super.visitReturnStatement(node);
  }
}

// =============================================================================
// 3. ndarray_loop_reassignment_leak
// =============================================================================

/// Flags `curr = op(curr)` inside a `for`/`while`/`do` loop without `out: curr`,
/// `.dispose()`, or a per-iteration `NDArray.scope`.
///
/// The previous buffers are not leaked permanently: each is released when the
/// enclosing scope ends, or eventually by its `NativeFinalizer`. But the GC
/// does not see native memory pressure, so a long loop can accumulate a large
/// amount of native memory before anything is reclaimed.
final class LoopReassignmentLeakRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_loop_reassignment_leak',
    'Reassigning an outer NDArray variable inside a loop accumulates native '
        'memory: every previous buffer stays alive until the enclosing scope '
        'ends or the GC finalizes it.',
    correctionMessage:
        'Pass out: to reuse the existing buffer in-place, dispose the previous '
        'array before reassigning, or wrap the loop body in NDArray.scope.',
    severity: DiagnosticSeverity.WARNING,
  );

  LoopReassignmentLeakRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Avoid reassigning outer NDArray variables inside loops without '
            'in-place out: reuse or per-iteration disposal.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _LoopReassignmentVisitor(this);
    registry.addAssignmentExpression(this, visitor);
  }
}

final class _LoopReassignmentVisitor extends SimpleAstVisitor<void> {
  final LoopReassignmentLeakRule rule;

  _LoopReassignmentVisitor(this.rule);

  @override
  void visitAssignmentExpression(AssignmentExpression node) {
    final lhs = node.leftHandSide;
    if (lhs is! SimpleIdentifier) return;
    final lhsElement = lhs.element;
    if (lhsElement == null) return;
    final lhsType =
        node.writeType ??
        lhs.staticType ??
        (lhsElement is VariableElement ? lhsElement.type : null);
    if (!isScopedResourceType(lhsType)) return;

    final loopNode = _findEnclosingLoopWithoutInnerScope(node);
    if (loopNode == null) return;

    // If `lhs` was declared inside the loop body, it's not an outer variable
    // reassignment across iterations.
    final loopDecls = SubtreeDeclarations.collect(loopNode);
    if (loopDecls.elements.contains(lhsElement)) return;

    final rhs = unwrapParenthesized(node.rightHandSide);
    if (rhs is SimpleIdentifier && rhs.element == lhsElement) return;

    // Check if rhs passes `out: lhs`.
    if (rhs is MethodInvocation) {
      for (final arg in rhs.argumentList.arguments) {
        if (arg is NamedArgument && arg.name.lexeme == 'out') {
          final outVal = unwrapParenthesized(arg.argumentExpression);
          if (outVal is SimpleIdentifier && outVal.element == lhsElement) {
            return;
          }
        }
      }
    }

    // Check if `lhs.dispose()` or `old.dispose()` (where `final old = lhs;`)
    // appears anywhere inside the loop body.
    if (_loopDisposesVariable(loopNode, lhsElement)) return;

    rule.reportAtNode(node);
  }

  AstNode? _findEnclosingLoopWithoutInnerScope(AstNode node) {
    AstNode? current = node.parent;
    while (current != null && current is! FunctionBody) {
      if (current is MethodInvocation &&
          (isScopeInvocation(current) || isReturningInvocation(current))) {
        // Enclosed in a per-iteration scope before reaching the loop!
        return null;
      }
      if (current is ForStatement ||
          current is WhileStatement ||
          current is DoStatement) {
        return current;
      }
      current = current.parent;
    }
    return null;
  }

  bool _loopDisposesVariable(AstNode loopNode, Element targetElement) {
    final checker = _LoopDisposeChecker(targetElement);
    loopNode.accept(checker);
    return checker.disposed;
  }
}

final class _LoopDisposeChecker extends RecursiveAstVisitor<void> {
  final Element targetElement;
  final Set<Element> aliasElements = {};
  bool disposed = false;

  _LoopDisposeChecker(this.targetElement) {
    aliasElements.add(targetElement);
  }

  @override
  void visitVariableDeclaration(VariableDeclaration node) {
    final init = node.initializer;
    final declaredElem = node.declaredFragment?.element;
    if (declaredElem != null &&
        init is SimpleIdentifier &&
        aliasElements.contains(init.element)) {
      aliasElements.add(declaredElem);
    }
    super.visitVariableDeclaration(node);
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name == 'dispose') {
      final target = node.realTarget;
      if (target is SimpleIdentifier &&
          aliasElements.contains(target.element)) {
        disposed = true;
        return;
      }
    }
    super.visitMethodInvocation(node);
  }
}

// =============================================================================
// 4. ndarray_identity_cast_dispose
// =============================================================================

/// Flags calling `.dispose()`, `.detachToParentScope()`, or `.detachFromScope()`
/// on a variable initialized from an identity-returning cast (`astype(copy: false)`,
/// `castNDArray`, `promoteToDouble`, `promoteToComplex`, `toNDArray`) without
/// guarding `if (!identical(v, src))`.
final class IdentityCastDisposeRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_identity_cast_dispose',
    'Disposing or detaching the result of an identity-returning conversion '
        '(such as astype(..., copy: false) or castNDArray) without checking '
        '!identical(result, source) will dispose the original input array '
        'when no conversion was needed.',
    correctionMessage:
        'Guard the call with `if (!identical(v, src))` or pass `copy: true`.',
    severity: DiagnosticSeverity.WARNING,
  );

  IdentityCastDisposeRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Guard .dispose() on identity-returning casts (astype(copy: false), '
            'castNDArray, promoteToDouble) with !identical(result, source).',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _IdentityCastDisposeVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _IdentityCastDisposeVisitor extends SimpleAstVisitor<void> {
  final IdentityCastDisposeRule rule;

  _IdentityCastDisposeVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final name = node.methodName.name;
    if (name != 'dispose' &&
        name != 'detachToParentScope' &&
        name != 'detachFromScope') {
      return;
    }
    final target = node.realTarget;
    if (target == null) return;

    // Direct invocation `a.astype(dt, copy: false).dispose()`
    if (identityCastSourceExpression(target) != null) {
      rule.reportAtNode(node);
      return;
    }

    if (target is SimpleIdentifier) {
      final element = target.element;
      if (element == null) return;
      final body = node.thisOrAncestorOfType<FunctionBody>();
      if (body == null) return;
      final decls = SubtreeDeclarations.collect(body);
      final decl = decls.variableDeclarations[element];
      final init = decl?.initializer;
      if (init == null) return;
      final srcExpr = identityCastSourceExpression(init);
      if (srcExpr == null) return;

      final srcElem = srcExpr is SimpleIdentifier ? srcExpr.element : null;
      if (_isGuardedByNotIdentical(node, element, srcElem)) return;

      rule.reportAtNode(node);
    }
  }

  bool _isGuardedByNotIdentical(
    AstNode node,
    Element castElement,
    Element? srcElement,
  ) {
    AstNode? current = node.parent;
    while (current != null && current is! FunctionBody) {
      if (current is IfStatement) {
        if (_conditionChecksIdentityInequality(
          current.expression,
          castElement,
          srcElement,
        )) {
          return true;
        }
      }
      current = current.parent;
    }
    return false;
  }

  bool _conditionChecksIdentityInequality(
    Expression condition,
    Element castElement,
    Element? srcElement,
  ) {
    final unwrapped = unwrapParenthesized(condition);
    if (unwrapped is PrefixExpression &&
        unwrapped.operator.type == TokenType.BANG) {
      final operand = unwrapParenthesized(unwrapped.operand);
      if (operand is MethodInvocation &&
          operand.methodName.name == 'identical') {
        final args = operand.argumentList.arguments;
        if (args.length == 2) {
          final a0 = unwrapParenthesized(args[0].argumentExpression);
          final a1 = unwrapParenthesized(args[1].argumentExpression);
          final e0 = a0 is SimpleIdentifier ? a0.element : null;
          final e1 = a1 is SimpleIdentifier ? a1.element : null;
          if (e0 == castElement || e1 == castElement) {
            if (srcElement == null || e0 == srcElement || e1 == srcElement) {
              return true;
            }
          }
        }
      }
    }
    if (unwrapped is BinaryExpression &&
        unwrapped.operator.type == TokenType.BANG_EQ) {
      final l = unwrapParenthesized(unwrapped.leftOperand);
      final r = unwrapParenthesized(unwrapped.rightOperand);
      final le = l is SimpleIdentifier ? l.element : null;
      final re = r is SimpleIdentifier ? r.element : null;
      if (le == castElement || re == castElement) {
        return true;
      }
    }
    if (unwrapped is BinaryExpression &&
        unwrapped.operator.type == TokenType.AMPERSAND_AMPERSAND) {
      return _conditionChecksIdentityInequality(
            unwrapped.leftOperand,
            castElement,
            srcElement,
          ) ||
          _conditionChecksIdentityInequality(
            unwrapped.rightOperand,
            castElement,
            srcElement,
          );
    }
    return false;
  }
}

// =============================================================================
// 5. ndarray_sendable_borrow_outlives_scope
// =============================================================================

/// Flags `.toSendableBorrow()` inside an `NDArray.scope` whose callback is
/// synchronous or never awaits, so the scope can dispose the borrowed buffer
/// while a worker isolate is still reading it.
final class SendableBorrowOutlivesScopeRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_sendable_borrow_outlives_scope',
    'toSendableBorrow() does not copy; if the enclosing NDArray.scope exits '
        'before the worker isolate is done, the borrowed buffer is freed '
        'while still in use.',
    correctionMessage:
        'Make the scope callback async and await the isolate work, or use '
        '.toSendable() to transfer an owned copy.',
    severity: DiagnosticSeverity.WARNING,
  );

  SendableBorrowOutlivesScopeRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Only call toSendableBorrow() inside an async NDArray.scope that '
            'awaits the isolate work.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _SendableBorrowVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _SendableBorrowVisitor extends SimpleAstVisitor<void> {
  final SendableBorrowOutlivesScopeRule rule;

  _SendableBorrowVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name != 'toSendableBorrow') return;
    if (!isNDArrayType(node.realTarget?.staticType)) return;
    final scopeCallback = _findEnclosingScopeCallback(node);
    if (scopeCallback == null) return;
    if (!scopeCallback.body.isAsynchronous ||
        !_hasAwaitExpression(scopeCallback.body)) {
      rule.reportAtNode(node);
    }
  }

  FunctionExpression? _findEnclosingScopeCallback(AstNode node) {
    AstNode? current = node.parent;
    while (current != null) {
      if (current is FunctionExpression) {
        final parent = current.parent;
        if (parent is ArgumentList) {
          final grandParent = parent.parent;
          if (grandParent is MethodInvocation &&
              (isScopeInvocation(grandParent) ||
                  isReturningInvocation(grandParent))) {
            return current;
          }
        }
      }
      current = current.parent;
    }
    return null;
  }

  bool _hasAwaitExpression(AstNode body) {
    final finder = _AwaitFinder();
    body.accept(finder);
    return finder.found;
  }
}

final class _AwaitFinder extends RecursiveAstVisitor<void> {
  bool found = false;

  @override
  void visitAwaitExpression(AwaitExpression node) {
    found = true;
  }
}
