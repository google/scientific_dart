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
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/error/error.dart';

import '../type_utils.dart';

// =============================================================================
// 11. ndarray_0d_reduction_indexing
// =============================================================================

/// Flags non-empty coordinate indexing (`r[[0]]`) on the result of an
/// axis-less reduction such as `sum(a)`, which is 0-dimensional and throws a
/// `RangeError` for any non-empty index.
final class ZeroDimReductionIndexingRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_0d_reduction_indexing',
    'Axis-less reductions return a 0-dimensional (shape []) NDArray; indexing '
        'it with non-empty coordinates throws a RangeError.',
    correctionMessage: 'Read the value with .scalar (or index with []).',
    severity: DiagnosticSeverity.WARNING,
  );

  ZeroDimReductionIndexingRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not index the 0-dimensional result of an axis-less reduction '
            'with non-empty coordinates.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _ZeroDimReductionVisitor(this);
    registry.addIndexExpression(this, visitor);
  }
}

final class _ZeroDimReductionVisitor extends SimpleAstVisitor<void> {
  final ZeroDimReductionIndexingRule rule;

  _ZeroDimReductionVisitor(this.rule);

  @override
  void visitIndexExpression(IndexExpression node) {
    final body = node.thisOrAncestorOfType<FunctionBody>();
    final decls = body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
    if (!isAxislessReductionExpression(node.realTarget, decls)) return;

    final indexExpr = unwrapParenthesized(node.index);
    if (indexExpr is ListLiteral && indexExpr.elements.isEmpty) return;
    rule.reportAtNode(node);
  }
}

// =============================================================================
// 13. nditer_coords_aliasing_or_mutation
// =============================================================================

/// Flags storing `NDIter.coords` into a collection without copying or mutating
/// `NDIter.coords` in-place (`iter.coords[0] = ...`).
final class NDIterCoordsAliasingOrMutationRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'nditer_coords_aliasing_or_mutation',
    'NDIter.coords returns an internal List<int> that moveNext() mutates '
        'in-place; storing or mutating it directly aliases or corrupts '
        'iterator coordinates.',
    correctionMessage:
        'Copy the coordinates with List.of(iter.coords) or [...iter.coords] '
        'before storing, and never mutate iter.coords in-place.',
    severity: DiagnosticSeverity.WARNING,
  );

  NDIterCoordsAliasingOrMutationRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not store or mutate NDIter.coords without copying via '
            'List.of(iter.coords).',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _NDIterCoordsVisitor(this);
    registry.addPropertyAccess(this, visitor);
    registry.addPrefixedIdentifier(this, visitor);
  }
}

final class _NDIterCoordsVisitor extends SimpleAstVisitor<void> {
  final NDIterCoordsAliasingOrMutationRule rule;

  _NDIterCoordsVisitor(this.rule);

  @override
  void visitPropertyAccess(PropertyAccess node) {
    if (node.propertyName.name == 'coords' &&
        isNDIterType(node.realTarget.staticType)) {
      _checkCoordsUsage(node);
    }
  }

  @override
  void visitPrefixedIdentifier(PrefixedIdentifier node) {
    if (node.identifier.name == 'coords' &&
        isNDIterType(node.prefix.staticType)) {
      _checkCoordsUsage(node);
    }
  }

  void _checkCoordsUsage(Expression coordsNode) {
    final parent = coordsNode.parent;

    // 1. In-place index mutation: `iter.coords[0] = ...`
    if (parent is IndexExpression && parent.target == coordsNode) {
      final grandParent = parent.parent;
      if (grandParent is AssignmentExpression &&
          grandParent.leftHandSide == parent) {
        rule.reportAtNode(grandParent);
        return;
      }
    }

    // 2. Mutating List method on `iter.coords`: `.clear()`, `.add()`, `.sort()`
    if (parent is MethodInvocation && parent.realTarget == coordsNode) {
      const mutatingMethods = {
        'add',
        'addAll',
        'clear',
        'fillRange',
        'insert',
        'insertAll',
        'remove',
        'removeAt',
        'removeLast',
        'removeRange',
        'removeWhere',
        'replaceRange',
        'retainWhere',
        'setAll',
        'setRange',
        'shuffle',
        'sort',
      };
      if (mutatingMethods.contains(parent.methodName.name)) {
        rule.reportAtNode(parent);
        return;
      }
    }

    // 3. Passed directly into collection `.add(iter.coords)` or `map[...]`
    // (as opposed to `List.of(iter.coords)` or `List.from(iter.coords)`).
    if (parent is ArgumentList) {
      final invocation = parent.parent;
      if (invocation is MethodInvocation) {
        final fnName = invocation.methodName.name;
        if (fnName == 'add' || fnName == 'insert' || fnName == 'putIfAbsent') {
          rule.reportAtNode(coordsNode);
        }
      }
    }
  }
}

// =============================================================================
// 15. ndarray_lost_mutation_on_copy
// =============================================================================

/// Flags mutating an inline copy-producing method (`a.flatten().fill(0)`,
/// `a.copy()[0] = 1`, `a.astype(dt).setSlice(...)`) in an expression statement
/// where the copy is immediately discarded.
final class LostMutationOnCopyRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_lost_mutation_on_copy',
    'Mutating the result of .flatten(), .copy(), or .astype() in a statement '
        'modifies and leaks a temporary copy while leaving the original '
        'NDArray unchanged.',
    correctionMessage:
        'Use a view method such as .ravel() or .slice() to mutate the '
        'original array in-place, or assign the copy to a variable.',
    severity: DiagnosticSeverity.WARNING,
  );

  LostMutationOnCopyRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not call in-place mutating operations (.fill, .setSlice, []=) '
            'on discarded copies (.flatten(), .copy(), .astype()).',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _LostMutationOnCopyVisitor(this);
    registry.addExpressionStatement(this, visitor);
  }
}

final class _LostMutationOnCopyVisitor extends SimpleAstVisitor<void> {
  final LostMutationOnCopyRule rule;

  _LostMutationOnCopyVisitor(this.rule);

  @override
  void visitExpressionStatement(ExpressionStatement node) {
    final expr = unwrapParenthesized(node.expression);

    // Case 1: `a.flatten().fill(0)` or `a.copy().setSlice(...)`
    if (expr is MethodInvocation) {
      final methodName = expr.methodName.name;
      if (methodName == 'fill' || methodName == 'setSlice') {
        final target = expr.realTarget;
        if (target != null && _isInlineCopyInvocation(target)) {
          rule.reportAtNode(expr);
        }
      }
    }

    // Case 2: `a.flatten()[[0]] = 42`
    if (expr is AssignmentExpression) {
      final lhs = expr.leftHandSide;
      if (lhs is IndexExpression && _isInlineCopyInvocation(lhs.realTarget)) {
        rule.reportAtNode(expr);
      }
    }
  }

  bool _isInlineCopyInvocation(Expression expr) {
    final unwrapped = unwrapParenthesized(expr);
    if (unwrapped is! MethodInvocation) return false;
    final name = unwrapped.methodName.name;
    if (name == 'flatten' || name == 'copy') {
      return isNDArrayType(unwrapped.staticType);
    }
    if (name == 'astype' && isNDArrayType(unwrapped.staticType)) {
      return identityCastSourceExpression(unwrapped) == null;
    }
    return false;
  }
}

// =============================================================================
// 16. ndarray_from_pointer_dangling_arena
// =============================================================================

/// Flags returning an `NDArray.fromPointer(ptr, ...)` from a `try`/`finally`
/// block that frees `ptr` (`calloc.free(ptr)`, `malloc.free(ptr)`, or
/// `ScratchArena.reset(...)`) or from a `using((arena) => ...)` callback.
final class FromPointerDanglingArenaRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_from_pointer_dangling_arena',
    'NDArray.fromPointer wraps raw C memory without copying; returning it '
        'from a block that frees the pointer or resets ScratchArena creates '
        'an immediate use-after-free.',
    correctionMessage:
        'Return .copy() (detached if inside a scope) before the pointer is '
        'freed, or transfer ownership of the pointer by passing '
        'nativeFinalizer: to NDArray.fromPointer instead of freeing it.',
    severity: DiagnosticSeverity.WARNING,
  );

  FromPointerDanglingArenaRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not return an NDArray.fromPointer view over memory freed in a '
            'finally block or arena callback.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _FromPointerDanglingArenaVisitor(this);
    registry.addTryStatement(this, visitor);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _FromPointerDanglingArenaVisitor extends SimpleAstVisitor<void> {
  final FromPointerDanglingArenaRule rule;

  _FromPointerDanglingArenaVisitor(this.rule);

  @override
  void visitTryStatement(TryStatement node) {
    final finallyBlock = node.finallyBlock;
    if (finallyBlock == null) return;
    final finallySrc = finallyBlock.toSource();
    if (!finallySrc.contains('.free(') &&
        !finallySrc.contains('ScratchArena.reset(')) {
      return;
    }

    final decls = SubtreeDeclarations.collect(node.body);
    node.body.accept(_TryReturnFromPointerFinder(rule, decls));
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name != 'using') return;
    final args = node.argumentList.arguments;
    if (args.isEmpty) return;
    final callback = unwrapParenthesized(args.first.argumentExpression);
    if (callback is! FunctionExpression) return;
    final decls = SubtreeDeclarations.collect(callback);
    callback.body.accept(_TryReturnFromPointerFinder(rule, decls));
  }
}

final class _TryReturnFromPointerFinder extends RecursiveAstVisitor<void> {
  final FromPointerDanglingArenaRule rule;
  final SubtreeDeclarations decls;

  _TryReturnFromPointerFinder(this.rule, this.decls);

  @override
  void visitReturnStatement(ReturnStatement node) {
    final expr = node.expression;
    if (expr != null && _isFromPointerWithoutCopy(expr, decls)) {
      rule.reportAtNode(expr);
    }
  }

  @override
  void visitExpressionFunctionBody(ExpressionFunctionBody node) {
    final expr = node.expression;
    if (_isFromPointerWithoutCopy(expr, decls)) {
      rule.reportAtNode(expr);
    }
  }

  bool _isFromPointerWithoutCopy(
    Expression expr,
    SubtreeDeclarations decls, {
    int depth = 0,
  }) {
    if (depth > 4) return false;
    final unwrapped = unwrapParenthesized(expr);
    if (unwrapped is MethodInvocation) {
      if (unwrapped.methodName.name == 'copy') return false;
      if (unwrapped.methodName.name == 'fromPointer') return true;
      if (unwrapped.methodName.name == 'detachToParentScope' ||
          unwrapped.methodName.name == 'detachFromScope') {
        final target = unwrapped.realTarget;
        if (target != null) {
          return _isFromPointerWithoutCopy(target, decls, depth: depth + 1);
        }
      }
    } else if (unwrapped is InstanceCreationExpression) {
      if (unwrapped.constructorName.name?.name == 'fromPointer') {
        return true;
      }
    } else if (unwrapped is SimpleIdentifier) {
      final element = unwrapped.element;
      if (element != null) {
        final init = decls.variableDeclarations[element]?.initializer;
        if (init != null) {
          return _isFromPointerWithoutCopy(init, decls, depth: depth + 1);
        }
      }
    }
    return false;
  }
}

// =============================================================================
// 17. scoped_resource_unawaited_in_scope
// =============================================================================

/// Flags unawaited `Future`-returning calls inside `NDArray.scope` /
/// `ResourceScope.scope` that reference a local `ScopedResource`, because
/// `scope.dispose()` runs before the unawaited `Future` completes.
final class UnawaitedAsyncInScopeRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'scoped_resource_unawaited_in_scope',
    'Invoking an unawaited Future-returning operation on a local '
        'ScopedResource inside NDArray.scope allows the scope to exit and '
        'dispose the resource while the async operation is still running.',
    correctionMessage:
        'Make the NDArray.scope callback async and await (or return) the '
        'Future.',
    severity: DiagnosticSeverity.WARNING,
  );

  UnawaitedAsyncInScopeRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Await or return Futures that access local ScopedResources inside '
            'NDArray.scope.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _UnawaitedAsyncInScopeVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _UnawaitedAsyncInScopeVisitor extends SimpleAstVisitor<void> {
  final UnawaitedAsyncInScopeRule rule;

  _UnawaitedAsyncInScopeVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (!isScopeInvocation(node)) return;
    final args = node.argumentList.arguments;
    if (args.isEmpty) return;
    final callback = unwrapParenthesized(args.first.argumentExpression);
    if (callback is! FunctionExpression) return;

    final decls = SubtreeDeclarations.collect(callback);
    callback.body.accept(
      _UnawaitedFutureStatementFinder(rule, callback, decls),
    );
  }
}

final class _UnawaitedFutureStatementFinder extends RecursiveAstVisitor<void> {
  final UnawaitedAsyncInScopeRule rule;
  final FunctionExpression scopeCallback;
  final SubtreeDeclarations decls;

  _UnawaitedFutureStatementFinder(this.rule, this.scopeCallback, this.decls);

  @override
  void visitFunctionExpression(FunctionExpression node) {
    if (!identical(node, scopeCallback)) return;
    super.visitFunctionExpression(node);
  }

  @override
  void visitExpressionStatement(ExpressionStatement node) {
    final expr = unwrapParenthesized(node.expression);
    if (expr is AwaitExpression) return;
    if (expr is MethodInvocation && expr.methodName.name == 'unawaited') {
      if (_referencesLocalScopedResource(expr)) {
        rule.reportAtNode(expr);
      }
      return;
    }
    final type = expr.staticType;
    if (type != null && (type.isDartAsyncFuture || type.isDartAsyncFutureOr)) {
      if (_referencesLocalScopedResource(expr)) {
        rule.reportAtNode(expr);
      }
    }
  }

  bool _referencesLocalScopedResource(AstNode expr) {
    final finder = _LocalScopedResourceRefFinder(decls.elements);
    expr.accept(finder);
    return finder.found;
  }
}

final class _LocalScopedResourceRefFinder extends RecursiveAstVisitor<void> {
  final Set<Element> localElements;
  bool found = false;

  _LocalScopedResourceRefFinder(this.localElements);

  @override
  void visitSimpleIdentifier(SimpleIdentifier node) {
    final elem = node.element;
    if (elem != null &&
        localElements.contains(elem) &&
        isScopedResourceType(node.staticType)) {
      found = true;
    }
  }
}

// =============================================================================
// 18. symbolic_lambdify_in_loop
// =============================================================================

/// Flags `expr.lambdify(...)` inside a `for`/`while`/`do` loop when `expr` is
/// a variable declared outside the loop and not reassigned inside it, so the
/// same expression is compiled again on every iteration.
final class SymbolicLambdifyInLoopRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'symbolic_lambdify_in_loop',
    'Calling lambdify() on a loop-invariant symbolic expression inside a loop '
        'recompiles it on every iteration.',
    correctionMessage:
        'Hoist `final fn = expr.lambdify(vars);` outside the loop and '
        'evaluate `fn.callScalar(...)` or `fn.callArray(...)`.',
    severity: DiagnosticSeverity.INFO,
  );

  SymbolicLambdifyInLoopRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Hoist loop-invariant symbolic lambdify() compilation outside '
            'loops.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _SymbolicLambdifyInLoopVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _SymbolicLambdifyInLoopVisitor extends SimpleAstVisitor<void> {
  final SymbolicLambdifyInLoopRule rule;

  _SymbolicLambdifyInLoopVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name != 'lambdify') return;
    final receiver = node.realTarget;
    if (receiver == null) return;
    final target = unwrapParenthesized(receiver);
    if (target is! SimpleIdentifier) return;
    final element = target.element;
    if (element == null) return;

    final loopNode = _findEnclosingLoop(node);
    if (loopNode == null) return;
    if (SubtreeDeclarations.collect(loopNode).elements.contains(element)) {
      return;
    }
    final assignments = _AssignmentFinder(element);
    loopNode.accept(assignments);
    if (assignments.found) return;

    rule.reportAtNode(node);
  }

  AstNode? _findEnclosingLoop(AstNode node) {
    AstNode? current = node.parent;
    while (current != null && current is! FunctionBody) {
      if (current is ForStatement ||
          current is WhileStatement ||
          current is DoStatement) {
        return current;
      }
      current = current.parent;
    }
    return null;
  }
}

/// Finds assignments to [target] (`target = ...`, `target += ...`).
final class _AssignmentFinder extends RecursiveAstVisitor<void> {
  final Element target;
  bool found = false;

  _AssignmentFinder(this.target);

  @override
  void visitAssignmentExpression(AssignmentExpression node) {
    final lhs = node.leftHandSide;
    if (lhs is SimpleIdentifier && lhs.element == target) {
      found = true;
      return;
    }
    super.visitAssignmentExpression(node);
  }
}
