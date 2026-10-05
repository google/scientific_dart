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

import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/dart/element/type.dart';

/// Names of methods, getters, and top-level functions in `package:ndarray` that
/// always return a zero-copy view ([NDArray.isView] is `true`).
const Set<String> kAlwaysViewNames = {
  'slice',
  'transpose',
  'T',
  'transposed',
  'squeeze',
  'expandDims',
  'expand_dims',
  'swapaxes',
  'moveaxis',
  'broadcastTo',
  'broadcast_to',
  'flip',
  'fliplr',
  'flipud',
  'rot90',
};

/// Names of operations that return a view for contiguous input but a fresh
/// copy otherwise (e.g. `reshape` and `ravel` of a non-contiguous array).
///
/// Whether the result is a view cannot be decided statically, so rules must
/// neither assume it is a view nor assume it is a fresh allocation.
const Set<String> kMaybeViewNames = {
  'reshape',
  'ravel',
  'diagonal',
  'materializeView',
};

/// Subset of [kAlwaysViewNames] that produce read-only broadcast views
/// (`stride == 0` along broadcasted axes).
const Set<String> kBroadcastViewNames = {'broadcastTo', 'broadcast_to'};

/// Reduction functions in `package:ndarray` that return a 0-dimensional
/// (`shape: []`, `rank: 0`) `NDArray` when called without an `axis:` argument.
const Set<String> kReductionNames = {
  'sum',
  'mean',
  'min',
  'max',
  'prod',
  'std',
  'var_',
  'variance',
  'nansum',
  'nanmean',
  'nanmin',
  'nanmax',
  'nanstd',
  'nanvar',
  'any',
  'all',
  'ptp',
  'median',
  'nanmedian',
  'quantile',
  'nanquantile',
  'percentile',
  'nanpercentile',
  'argmax',
  'argmin',
  'nanargmax',
  'nanargmin',
  'count_nonzero',
  'countNonzero',
};

/// Returns whether [type] is `NDArray`.
bool isNDArrayType(DartType? type) {
  if (type is! InterfaceType) return false;
  if (_isNDArrayElement(type.element)) return true;
  for (final supertype in type.allSupertypes) {
    if (_isNDArrayElement(supertype.element)) return true;
  }
  return false;
}

bool _isNDArrayElement(InterfaceElement element) {
  return element.name == 'NDArray';
}

/// Returns whether [type] is `NDIter`.
bool isNDIterType(DartType? type) {
  if (type is! InterfaceType) return false;
  return type.element.name == 'NDIter';
}

/// Returns whether [type] implements `ScopedResource` (including `NDArray`,
/// `Expr`, `GpuArray`, etc.).
bool isScopedResourceType(DartType? type) {
  if (type is! InterfaceType) return false;
  if (_isScopedResourceElement(type.element)) return true;
  for (final supertype in type.allSupertypes) {
    if (_isScopedResourceElement(supertype.element)) return true;
  }
  return false;
}

bool _isScopedResourceElement(InterfaceElement element) {
  final name = element.name;
  return name == 'ScopedResource' || name == 'NDArray';
}

/// Returns whether [type] is `ScopedResource` or an aggregate (`RecordType`,
/// `Iterable`, `List`, `Set`, `Map`) containing a `ScopedResource`.
bool containsScopedResourceType(DartType? type) {
  if (type == null) return false;
  if (isScopedResourceType(type)) return true;
  if (type is RecordType) {
    for (final field in type.positionalFields) {
      if (containsScopedResourceType(field.type)) return true;
    }
    for (final field in type.namedFields) {
      if (containsScopedResourceType(field.type)) return true;
    }
  }
  if (type is InterfaceType) {
    final name = type.element.name;
    if (name == 'List' ||
        name == 'Iterable' ||
        name == 'Set' ||
        name == 'Map') {
      for (final typeArg in type.typeArguments) {
        if (containsScopedResourceType(typeArg)) return true;
      }
    }
  }
  return false;
}

/// Returns whether [type] is `NDArray<Uint64>`.
bool isNDArrayUint64Type(DartType? type) {
  if (type is! InterfaceType) return false;
  if (type.element.name == 'NDArray' && type.typeArguments.length == 1) {
    final arg = type.typeArguments.first;
    if (arg is InterfaceType && arg.element.name == 'Uint64') {
      return true;
    }
  }
  for (final supertype in type.allSupertypes) {
    if (supertype.element.name == 'NDArray' &&
        supertype.typeArguments.length == 1) {
      final arg = supertype.typeArguments.first;
      if (arg is InterfaceType && arg.element.name == 'Uint64') {
        return true;
      }
    }
  }
  return false;
}

/// Unwraps surrounding parentheses from [expr].
Expression unwrapParenthesized(Expression expr) {
  var current = expr;
  while (current is ParenthesizedExpression) {
    current = current.expression;
  }
  return current;
}

/// Returns whether [invocation] is `NDArray.scope(...)` or
/// `ResourceScope.scope(...)`.
bool isScopeInvocation(MethodInvocation invocation) {
  if (invocation.methodName.name != 'scope') return false;
  final target = invocation.target;
  if (target is SimpleIdentifier) {
    return target.name == 'NDArray' || target.name == 'ResourceScope';
  }
  if (target is PrefixedIdentifier) {
    return target.identifier.name == 'NDArray' ||
        target.identifier.name == 'ResourceScope';
  }
  return false;
}

/// Returns whether [invocation] is `NDArray.returning(...)` or
/// `ResourceScope.returning(...)`.
bool isReturningInvocation(MethodInvocation invocation) {
  if (invocation.methodName.name != 'returning') return false;
  final target = invocation.target;
  if (target is SimpleIdentifier) {
    return target.name == 'NDArray' || target.name == 'ResourceScope';
  }
  if (target is PrefixedIdentifier) {
    return target.identifier.name == 'NDArray' ||
        target.identifier.name == 'ResourceScope';
  }
  return false;
}

/// Returns whether [node] is lexically inside an `NDArray.scope`,
/// `ResourceScope.scope`, `NDArray.returning`, or `ResourceScope.returning`
/// callback.
bool isInsideActiveScope(AstNode node) {
  AstNode? current = node.parent;
  while (current != null) {
    if (current is MethodInvocation &&
        (isScopeInvocation(current) || isReturningInvocation(current))) {
      return true;
    }
    if (current is FunctionDeclaration || current is MethodDeclaration) {
      return false;
    }
    current = current.parent;
  }
  return false;
}

/// Collects all [Element]s (variables, parameters, pattern bindings, local
/// functions) declared within [subtree], plus a map from each local
/// `VariableElement` to its `VariableDeclaration` AST node.
final class SubtreeDeclarations {
  final Set<Element> elements = {};
  final Map<Element, VariableDeclaration> variableDeclarations = {};

  static SubtreeDeclarations collect(AstNode subtree) {
    final collector = _DeclarationCollector();
    subtree.accept(collector);
    return collector.result;
  }
}

final class _DeclarationCollector extends RecursiveAstVisitor<void> {
  final SubtreeDeclarations result = SubtreeDeclarations();

  @override
  void visitVariableDeclaration(VariableDeclaration node) {
    final element = node.declaredFragment?.element;
    if (element != null) {
      result.elements.add(element);
      result.variableDeclarations[element] = node;
    }
    super.visitVariableDeclaration(node);
  }

  @override
  void visitFormalParameterList(FormalParameterList node) {
    for (final param in node.parameters) {
      final element = param.declaredFragment?.element;
      if (element != null) {
        result.elements.add(element);
      }
    }
    super.visitFormalParameterList(node);
  }

  @override
  void visitDeclaredIdentifier(DeclaredIdentifier node) {
    final element = node.declaredFragment?.element;
    if (element != null) {
      result.elements.add(element);
    }
    super.visitDeclaredIdentifier(node);
  }

  @override
  void visitDeclaredVariablePattern(DeclaredVariablePattern node) {
    final element = node.declaredFragment?.element;
    if (element != null) {
      result.elements.add(element);
    }
    super.visitDeclaredVariablePattern(node);
  }

  @override
  void visitFunctionDeclaration(FunctionDeclaration node) {
    final element = node.declaredFragment?.element;
    if (element != null) {
      result.elements.add(element);
    }
    super.visitFunctionDeclaration(node);
  }
}

/// Returns whether [expr] directly detaches a `ScopedResource` (e.g.
/// `x.detachToParentScope()`, `x.detachFromScope()`, `x..detachToParentScope()`,
/// `NDArray.returning(...)`, `NDArray.unmanaged(...)`).
bool isDirectlyDetachedExpression(Expression expr) {
  final unwrapped = unwrapParenthesized(expr);
  if (unwrapped is MethodInvocation) {
    final methodName = unwrapped.methodName.name;
    if (methodName == 'detachToParentScope' ||
        methodName == 'detachFromScope') {
      return true;
    }
    if (isReturningInvocation(unwrapped)) return true;
    if (methodName == 'unmanaged') return true;
  }
  if (unwrapped is CascadeExpression) {
    for (final section in unwrapped.cascadeSections) {
      if (section is MethodInvocation) {
        final name = section.methodName.name;
        if (name == 'detachToParentScope' || name == 'detachFromScope') {
          return true;
        }
      }
    }
  }
  return false;
}

/// Returns whether [element] has `.detachToParentScope()` or
/// `.detachFromScope()` invoked on it inside [scopeBody] prior to [beforeOffset].
bool wasElementDetachedBefore(
  Element element,
  AstNode scopeBody,
  int beforeOffset,
) {
  final finder = _DetachCallFinder(element, beforeOffset);
  scopeBody.accept(finder);
  return finder.found;
}

final class _DetachCallFinder extends RecursiveAstVisitor<void> {
  final Element targetElement;
  final int beforeOffset;
  bool found = false;

  _DetachCallFinder(this.targetElement, this.beforeOffset);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.offset >= beforeOffset) return;
    final name = node.methodName.name;
    if (name == 'detachToParentScope' || name == 'detachFromScope') {
      final target = node.realTarget;
      if (target is SimpleIdentifier && target.element == targetElement) {
        found = true;
        return;
      }
    }
    super.visitMethodInvocation(node);
  }
}

/// Returns whether [expr] (or the initializer of a local variable referenced by
/// [expr] inside [decls]) produces a zero-copy view (`isView == true`).
bool isViewProducingExpression(
  Expression expr,
  SubtreeDeclarations decls, {
  int depth = 0,
}) {
  if (depth > 4) return false;
  final unwrapped = unwrapParenthesized(expr);

  if (unwrapped is MethodInvocation) {
    final name = unwrapped.methodName.name;
    if (name == 'copy') return false;
    if (name == 'detachToParentScope' || name == 'detachFromScope') {
      final target = unwrapped.realTarget;
      if (target != null) {
        return isViewProducingExpression(target, decls, depth: depth + 1);
      }
    }
    if (kAlwaysViewNames.contains(name)) {
      final targetType = unwrapped.realTarget?.staticType;
      if (targetType == null ||
          isNDArrayType(targetType) ||
          (targetType is InterfaceType &&
              targetType.element.name == 'SendableNDArray') ||
          isNDArrayType(unwrapped.staticType)) {
        return true;
      }
    }
  } else if (unwrapped is PropertyAccess) {
    final propName = unwrapped.propertyName.name;
    if ((propName == 'T' || propName == 'transposed') &&
        isNDArrayType(unwrapped.realTarget.staticType)) {
      return true;
    }
  } else if (unwrapped is PrefixedIdentifier) {
    final propName = unwrapped.identifier.name;
    if ((propName == 'T' || propName == 'transposed') &&
        isNDArrayType(unwrapped.prefix.staticType)) {
      return true;
    }
  } else if (unwrapped is CascadeExpression) {
    return isViewProducingExpression(unwrapped.target, decls, depth: depth + 1);
  } else if (unwrapped is SimpleIdentifier) {
    final element = unwrapped.element;
    if (element != null) {
      final decl = decls.variableDeclarations[element];
      final init = decl?.initializer;
      if (init != null) {
        return isViewProducingExpression(init, decls, depth: depth + 1);
      }
    }
  }
  return false;
}

/// Returns whether [expr] (or its local variable initializer in [decls]) is a
/// `broadcastTo` / `broadcast_to` call.
bool isBroadcastViewExpression(
  Expression expr,
  SubtreeDeclarations decls, {
  int depth = 0,
}) {
  if (depth > 4) return false;
  final unwrapped = unwrapParenthesized(expr);
  if (unwrapped is MethodInvocation) {
    final name = unwrapped.methodName.name;
    if (name == 'copy') return false;
    if (kBroadcastViewNames.contains(name)) return true;
  } else if (unwrapped is SimpleIdentifier) {
    final element = unwrapped.element;
    if (element != null) {
      final decl = decls.variableDeclarations[element];
      final init = decl?.initializer;
      if (init != null) {
        return isBroadcastViewExpression(init, decls, depth: depth + 1);
      }
    }
  }
  return false;
}

/// Returns whether [expr] (or its local variable initializer in [decls]) is a
/// reduction call (`sum`, `mean`, `min`, `max`, `prod`, etc.) with no `axis:`
/// argument (or `axis: null`) and no `keepdims: true`, producing a
/// 0-dimensional scalar `NDArray`.
bool isAxislessReductionExpression(
  Expression expr,
  SubtreeDeclarations decls, {
  int depth = 0,
}) {
  if (depth > 4) return false;
  final unwrapped = unwrapParenthesized(expr);
  if (unwrapped is MethodInvocation) {
    final name = unwrapped.methodName.name;
    if (kReductionNames.contains(name) && isNDArrayType(unwrapped.staticType)) {
      if (name == 'quantile' ||
          name == 'nanquantile' ||
          name == 'percentile' ||
          name == 'nanpercentile') {
        final positionalArgs = unwrapped.argumentList.arguments
            .where((a) => a is! NamedArgument)
            .toList();
        final qExpr = unwrapped.realTarget != null
            ? (positionalArgs.isNotEmpty ? positionalArgs.first : null)
            : (positionalArgs.length >= 2 ? positionalArgs[1] : null);
        if (qExpr != null) {
          final qUnwrapped = unwrapParenthesized(qExpr.argumentExpression);
          if (qUnwrapped is ListLiteral ||
              isNDArrayType(qUnwrapped.staticType)) {
            return false;
          }
        }
      }
      var hasAxisOrKeepdims = false;
      for (final arg in unwrapped.argumentList.arguments) {
        if (arg is NamedArgument) {
          final argName = arg.name.lexeme;
          final argVal = unwrapParenthesized(arg.argumentExpression);
          if (argName == 'axis') {
            if (argVal is! NullLiteral) {
              hasAxisOrKeepdims = true;
            }
          } else if (argName == 'keepdims') {
            if (argVal is! BooleanLiteral || argVal.value) {
              hasAxisOrKeepdims = true;
            }
          }
        }
      }
      return !hasAxisOrKeepdims;
    }
  } else if (unwrapped is SimpleIdentifier) {
    final element = unwrapped.element;
    if (element != null) {
      final decl = decls.variableDeclarations[element];
      final init = decl?.initializer;
      if (init != null) {
        return isAxislessReductionExpression(init, decls, depth: depth + 1);
      }
    }
  }
  return false;
}

/// Returns whether [expr] allocates a fresh owning `ScopedResource` (such as a
/// ufunc/arithmetic call, constructor, or binary operator on `NDArray`/`Expr`),
/// excluding views (`slice`, `transpose`, etc.) and scope helpers.
bool isFreshScopedResourceAllocation(Expression expr) {
  final unwrapped = unwrapParenthesized(expr);
  if (!isScopedResourceType(unwrapped.staticType)) return false;

  if (unwrapped is BinaryExpression || unwrapped is PrefixExpression) {
    return true;
  }
  if (unwrapped is InstanceCreationExpression) {
    return true;
  }
  if (unwrapped is MethodInvocation) {
    final name = unwrapped.methodName.name;
    if (kAlwaysViewNames.contains(name) || kMaybeViewNames.contains(name)) {
      return false;
    }
    if (name == 'detachToParentScope' ||
        name == 'detachFromScope' ||
        name == 'scope' ||
        name == 'returning' ||
        name == 'unmanaged') {
      return false;
    }
    // If `out:` is explicitly supplied, no fresh anonymous buffer is allocated.
    final hasOut = unwrapped.argumentList.arguments.any(
      (arg) => arg is NamedArgument && arg.name.lexeme == 'out',
    );
    if (hasOut) return false;
    return true;
  }
  return false;
}

/// Traces [expr] through view operations and local variable initializers to find
/// the underlying root `Element` (`VariableElement` or `FormalParameterElement`)
/// and whether any view transformation occurred along the path.
({Element? rootElement, bool throughView}) traceRootArrayAndView(
  Expression expr,
  SubtreeDeclarations decls, {
  int depth = 0,
}) {
  if (depth > 5) return (rootElement: null, throughView: false);
  final unwrapped = unwrapParenthesized(expr);

  if (unwrapped is SimpleIdentifier) {
    final element = unwrapped.element;
    if (element != null) {
      final decl = decls.variableDeclarations[element];
      final init = decl?.initializer;
      if (init != null) {
        final sub = traceRootArrayAndView(init, decls, depth: depth + 1);
        if (sub.rootElement != null) {
          return sub;
        }
      }
      return (rootElement: element, throughView: false);
    }
  } else if (unwrapped is MethodInvocation) {
    final name = unwrapped.methodName.name;
    if (kAlwaysViewNames.contains(name) || kMaybeViewNames.contains(name)) {
      final target =
          unwrapped.realTarget ??
          (unwrapped.argumentList.arguments.isNotEmpty
              ? unwrapped.argumentList.arguments.first.argumentExpression
              : null);
      if (target != null) {
        final sub = traceRootArrayAndView(target, decls, depth: depth + 1);
        return (rootElement: sub.rootElement, throughView: true);
      }
    }
  } else if (unwrapped is PropertyAccess &&
      (unwrapped.propertyName.name == 'T' ||
          unwrapped.propertyName.name == 'transposed')) {
    final sub = traceRootArrayAndView(
      unwrapped.realTarget,
      decls,
      depth: depth + 1,
    );
    return (rootElement: sub.rootElement, throughView: true);
  } else if (unwrapped is PrefixedIdentifier &&
      (unwrapped.identifier.name == 'T' ||
          unwrapped.identifier.name == 'transposed')) {
    final sub = traceRootArrayAndView(
      unwrapped.prefix,
      decls,
      depth: depth + 1,
    );
    return (rootElement: sub.rootElement, throughView: true);
  }
  return (rootElement: null, throughView: false);
}

/// If [expr] is an operation that may return its source `NDArray` unchanged
/// (`a.astype(..., copy: false)`, `castNDArray(a, ...)`, `promoteToDouble(a)`,
/// `promoteToComplex(a)`, `toNDArray(a)`), returns the source `Expression`.
Expression? identityCastSourceExpression(Expression expr) {
  final unwrapped = unwrapParenthesized(expr);
  if (unwrapped is! MethodInvocation) return null;
  final name = unwrapped.methodName.name;

  if (name == 'astype') {
    final target = unwrapped.realTarget;
    if (target == null || !isNDArrayType(target.staticType)) return null;
    for (final arg in unwrapped.argumentList.arguments) {
      if (arg is NamedArgument && arg.name.lexeme == 'copy') {
        final copyVal = unwrapParenthesized(arg.argumentExpression);
        if (copyVal is BooleanLiteral && !copyVal.value) {
          return target;
        }
      }
    }
    return null;
  }

  if (name == 'castNDArray' ||
      name == 'promoteToDouble' ||
      name == 'promoteToComplex' ||
      name == 'toNDArray') {
    final args = unwrapped.argumentList.arguments;
    if (args.isNotEmpty && args.first is! NamedArgument) {
      final firstArg = args.first.argumentExpression;
      if (isNDArrayType(firstArg.staticType)) {
        return firstArg;
      }
    }
  }
  return null;
}
