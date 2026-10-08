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

/// Static dtype reasoning for `package:ndarray`.
///
/// `package:ndarray` encodes the element type of an array as a type argument
/// (`NDArray<Float64>`, `NDArray<Int32>`, ...). Because class type parameters
/// are covariant in Dart, a same-dtype generic operation such as
/// `add<T extends DTypeTag>(NDArray<T> a, NDArray<T> b)` still compiles when
/// `a` and `b` have different concrete tags (the shared type parameter is
/// simply inferred as `DTypeTag`), and the mismatch only surfaces as an
/// `ArgumentError` at runtime. The helpers in this library recover the
/// concrete dtype of expressions and the same-dtype operand groups of
/// operation sites so rules and quick fixes can reason about them statically.
library;

import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/token.dart';
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/dart/element/type.dart';

/// The 15 concrete `DTypeTag` class names of `package:ndarray`, mapped to the
/// name of the corresponding `DType` enum constant (`Float64` -> `float64`).
const Map<String, String> kConcreteDTypeTagToEnumName = {
  'Float64': 'float64',
  'Float32': 'float32',
  'Float16': 'float16',
  'BFloat16': 'bfloat16',
  'Int64': 'int64',
  'Int32': 'int32',
  'Int16': 'int16',
  'Int8': 'int8',
  'Uint64': 'uint64',
  'Uint32': 'uint32',
  'Uint16': 'uint16',
  'Uint8': 'uint8',
  'Complex128': 'complex128',
  'Complex64': 'complex64',
  'Boolean': 'boolean',
};

/// Concrete tag names of the floating-point dtypes.
const Set<String> kFloatingDTypeTags = {
  'Float64',
  'Float32',
  'Float16',
  'BFloat16',
};

/// Concrete tag names of the complex dtypes.
const Set<String> kComplexDTypeTags = {'Complex128', 'Complex64'};

/// The binary operators of `NDArray` that require an array operand to have the
/// receiver's dtype (`NDArray<T>.operator +(Object? other)` etc.).
///
/// `==` and `!=` are deliberately absent: they are reference identity checks
/// (see `EqualityOperatorRule`).
const Set<TokenType> kSameDTypeBinaryOperators = {
  TokenType.PLUS,
  TokenType.MINUS,
  TokenType.STAR,
  TokenType.SLASH,
  TokenType.TILDE_SLASH,
  TokenType.PERCENT,
  TokenType.AMPERSAND,
  TokenType.BAR,
  TokenType.CARET,
  TokenType.LT_LT,
  TokenType.GT_GT,
  TokenType.LT,
  TokenType.LT_EQ,
  TokenType.GT,
  TokenType.GT_EQ,
};

/// The compound assignment operators desugaring to [kSameDTypeBinaryOperators].
const Set<TokenType> kSameDTypeCompoundAssignmentOperators = {
  TokenType.PLUS_EQ,
  TokenType.MINUS_EQ,
  TokenType.STAR_EQ,
  TokenType.SLASH_EQ,
  TokenType.TILDE_SLASH_EQ,
  TokenType.PERCENT_EQ,
  TokenType.AMPERSAND_EQ,
  TokenType.BAR_EQ,
  TokenType.CARET_EQ,
  TokenType.LT_LT_EQ,
  TokenType.GT_GT_EQ,
};

/// The `NDArray` methods taking an `Object?` operand that must have the
/// receiver's dtype when it is an array.
const Set<String> kSameDTypeMethodNames = {'eq', 'ne'};

/// Returns whether [element] is declared in a library of `package:ndarray`.
bool isDeclaredInNDArrayPackage(Element? element) {
  final uri = element?.library?.uri;
  if (uri == null || !uri.isScheme('package')) return false;
  final segments = uri.pathSegments;
  return segments.isNotEmpty && segments.first == 'ndarray';
}

/// Returns the `NDArray` instantiation of [type] (`NDArray<X>` itself, or the
/// `NDArray` supertype of a subclass), ignoring nullability; `null` if [type]
/// is not an `NDArray`.
InterfaceType? asNDArrayInstance(DartType? type) {
  if (type is! InterfaceType) return null;
  if (_isNDArrayElement(type.element)) return type;
  for (final supertype in type.allSupertypes) {
    if (_isNDArrayElement(supertype.element)) return supertype;
  }
  return null;
}

bool _isNDArrayElement(InterfaceElement element) =>
    element.name == 'NDArray' && isDeclaredInNDArrayPackage(element);

/// Returns the dtype type argument `X` of an `NDArray<X>` [type], looking
/// through one level of `List<NDArray<X>>` / `Iterable<NDArray<X>>`.
DartType? ndarrayDTypeArgument(DartType? type) {
  if (type is! InterfaceType) return null;
  if (_isCoreCollection(type) && type.typeArguments.length == 1) {
    return ndarrayDTypeArgument(type.typeArguments.single);
  }
  final ndarray = asNDArrayInstance(type);
  if (ndarray == null || ndarray.typeArguments.length != 1) return null;
  return ndarray.typeArguments.single;
}

bool _isCoreCollection(InterfaceType type) {
  final name = type.element.name;
  return (name == 'List' || name == 'Iterable') &&
      type.element.library.isDartCore;
}

/// Returns the concrete dtype tag name (`'Float64'`, `'Int32'`, ...) of an
/// `NDArray<Tag>` (or `List<NDArray<Tag>>`) [type].
///
/// Returns `null` when the dtype is not statically known to be one of the 15
/// concrete tags: `NDArray<DTypeTag>`, `NDArray<AnySpec>`, a type variable,
/// `dynamic`, or a non-`NDArray` type.
String? concreteDTypeTagOf(DartType? type) {
  final tag = ndarrayDTypeArgument(type);
  if (tag is! InterfaceType) return null;
  final name = tag.element.name;
  if (name == null || !kConcreteDTypeTagToEnumName.containsKey(name)) {
    return null;
  }
  return isDeclaredInNDArrayPackage(tag.element) ? name : null;
}

/// An operand of a same-dtype operation site.
final class SameDTypeOperand {
  /// The operand expression (an `NDArray`, or a `List<NDArray>` argument that
  /// is not a list literal).
  final Expression expression;

  /// Whether the operand can be wrapped in `.astype(...)` to fix a dtype
  /// mismatch: `false` for output buffers (`out:`-style named arguments) and
  /// for whole-list arguments.
  final bool isCastable;

  const SameDTypeOperand(this.expression, {required this.isCastable});

  /// The concrete dtype tag of [expression], or `null` if not statically known.
  String? get concreteTag => concreteDTypeTagOf(expression.staticType);
}

/// Returns the groups of operands that the operation at [node] requires to
/// share one runtime dtype, in source order; empty if [node] is not a
/// same-dtype operation site.
///
/// Recognized sites are:
/// - [BinaryExpression]s using one of [kSameDTypeBinaryOperators] whose
///   operator resolves to `package:ndarray` and whose operands are both
///   `NDArray`s;
/// - compound [AssignmentExpression]s using one of
///   [kSameDTypeCompoundAssignmentOperators] under the same conditions;
/// - [MethodInvocation]s of [kSameDTypeMethodNames] on an `NDArray` receiver
///   with an `NDArray` argument;
/// - [MethodInvocation]s of top-level functions declared in `package:ndarray`
///   without explicit type arguments, grouped per
///   [sameDTypeParameterGroups].
List<List<SameDTypeOperand>> sameDTypeOperandGroups(AstNode node) {
  return switch (node) {
    BinaryExpression() => _binaryOperandGroups(node),
    AssignmentExpression() => _compoundAssignmentOperandGroups(node),
    MethodInvocation() => _invocationOperandGroups(node),
    _ => const [],
  };
}

List<List<SameDTypeOperand>> _binaryOperandGroups(BinaryExpression node) {
  if (!kSameDTypeBinaryOperators.contains(node.operator.type)) return const [];
  if (!isDeclaredInNDArrayPackage(node.element)) return const [];
  return _pairGroup(node.leftOperand, node.rightOperand);
}

List<List<SameDTypeOperand>> _compoundAssignmentOperandGroups(
  AssignmentExpression node,
) {
  if (!kSameDTypeCompoundAssignmentOperators.contains(node.operator.type)) {
    return const [];
  }
  if (!isDeclaredInNDArrayPackage(node.element)) return const [];
  return _pairGroup(node.leftHandSide, node.rightHandSide);
}

List<List<SameDTypeOperand>> _pairGroup(Expression left, Expression right) {
  if (asNDArrayInstance(left.staticType) == null ||
      asNDArrayInstance(right.staticType) == null) {
    return const [];
  }
  return [
    [
      SameDTypeOperand(left, isCastable: true),
      SameDTypeOperand(right, isCastable: true),
    ],
  ];
}

List<List<SameDTypeOperand>> _invocationOperandGroups(MethodInvocation node) {
  final element = node.methodName.element;
  if (element is MethodElement) {
    if (!kSameDTypeMethodNames.contains(element.name) ||
        !isDeclaredInNDArrayPackage(element)) {
      return const [];
    }
    final target = node.realTarget;
    final args = node.argumentList.arguments;
    if (target == null || args.length != 1 || args.single is NamedArgument) {
      return const [];
    }
    return _pairGroup(target, args.single.argumentExpression);
  }
  if (element is! TopLevelFunctionElement ||
      !isDeclaredInNDArrayPackage(element) ||
      node.typeArguments != null) {
    return const [];
  }
  final parameterGroups = sameDTypeParameterGroups(element);
  if (parameterGroups.isEmpty) return const [];

  final groupOfParameter = <FormalParameterElement, int>{};
  for (var i = 0; i < parameterGroups.length; i++) {
    for (final parameter in parameterGroups[i].parameters) {
      groupOfParameter[parameter] = i;
    }
  }
  final groups = List.generate(
    parameterGroups.length,
    (_) => <SameDTypeOperand>[],
  );
  final positionalParameters = element.formalParameters
      .where((p) => p.isPositional)
      .toList();
  var positionalIndex = 0;
  for (final argument in node.argumentList.arguments) {
    final FormalParameterElement? parameter;
    final bool isOutput;
    if (argument is NamedArgument) {
      final name = argument.name.lexeme;
      parameter = element.formalParameters
          .where((p) => p.isNamed && p.name == name)
          .firstOrNull;
      isOutput = name.startsWith('out');
    } else {
      parameter = positionalIndex < positionalParameters.length
          ? positionalParameters[positionalIndex]
          : null;
      positionalIndex++;
      isOutput = false;
    }
    if (parameter == null) continue;
    final groupIndex = groupOfParameter[parameter];
    if (groupIndex == null) continue;
    final expression = argument.argumentExpression;
    if (parameterGroups[groupIndex].collectionParameters.contains(parameter)) {
      if (expression is ListLiteral) {
        for (final listElement in expression.elements) {
          if (listElement is Expression) {
            groups[groupIndex].add(
              SameDTypeOperand(listElement, isCastable: true),
            );
          }
        }
      } else {
        groups[groupIndex].add(SameDTypeOperand(expression, isCastable: false));
      }
    } else {
      groups[groupIndex].add(
        SameDTypeOperand(expression, isCastable: !isOutput),
      );
    }
  }
  return groups.where((g) => g.length >= 2).toList();
}

/// Formal parameters of a `package:ndarray` function that must receive arrays
/// of one shared dtype.
final class SameDTypeParameterGroup {
  /// The parameters of the group, in declaration order.
  final List<FormalParameterElement> parameters;

  /// The subset of [parameters] typed `List<NDArray<T>>` /
  /// `Iterable<NDArray<T>>`, whose elements must all share the dtype.
  final Set<FormalParameterElement> collectionParameters;

  const SameDTypeParameterGroup(this.parameters, this.collectionParameters);
}

/// Derives the same-dtype parameter groups of [function] from its signature.
///
/// Two formal parameters belong to the same group when their `NDArray` type
/// arguments are identical and *dtype-identifying*: either a bare type
/// parameter of [function] (`NDArray<T>`), or a `DTypeSpec<...>` instantiation
/// whose `RealTag` slot is a type parameter of [function]
/// (`NDArray<DTypeSpec<T, Object?, DTypeTag, ..., R, ...>>`, as used by
/// `atan2`, `hypot`, `logaddexp`). `RealTag` is the only `DTypeSpec` slot that
/// identifies a real dtype; the projection slots (`InexactTag`,
/// `AccumulatorTag`, ...) are shared by several dtypes, so a type parameter
/// bound only there (as in `chebval` or `cov`) does not tie operands to one
/// dtype and is not grouped.
///
/// A group is reported when it has at least two parameters, or when it contains
/// a collection parameter (whose elements must agree among themselves, as in
/// `concatenate(List<NDArray<T>> arrays)`). Functions with distinct type
/// parameters per operand (the mixed-dtype `*As` variants) therefore yield no
/// groups.
List<SameDTypeParameterGroup> sameDTypeParameterGroups(
  TopLevelFunctionElement function,
) {
  final parametersByKey = <String, List<FormalParameterElement>>{};
  final collectionParameters = <FormalParameterElement>{};
  for (final parameter in function.formalParameters) {
    final key = _dtypeIdentifyingKey(parameter.type, function);
    if (key == null) continue;
    parametersByKey.putIfAbsent(key.key, () => []).add(parameter);
    if (key.isCollection) collectionParameters.add(parameter);
  }
  return [
    for (final parameters in parametersByKey.values)
      if (parameters.length >= 2 ||
          parameters.any(collectionParameters.contains))
        SameDTypeParameterGroup(
          parameters,
          parameters.where(collectionParameters.contains).toSet(),
        ),
  ];
}

({String key, bool isCollection})? _dtypeIdentifyingKey(
  DartType type,
  TopLevelFunctionElement function,
) {
  if (type is! InterfaceType) return null;
  if (_isCoreCollection(type) && type.typeArguments.length == 1) {
    final inner = _dtypeIdentifyingKey(type.typeArguments.single, function);
    if (inner == null || inner.isCollection) return null;
    return (key: inner.key, isCollection: true);
  }
  final tag = ndarrayDTypeArgument(type);
  if (tag == null) return null;
  if (_isTypeParameterOf(tag, function)) {
    return (key: tag.getDisplayString(), isCollection: false);
  }
  if (tag is InterfaceType &&
      tag.element.name == 'DTypeSpec' &&
      isDeclaredInNDArrayPackage(tag.element)) {
    final realTagIndex = tag.element.typeParameters.indexWhere(
      (p) => p.name == 'RealTag',
    );
    if (realTagIndex >= 0 &&
        realTagIndex < tag.typeArguments.length &&
        _isTypeParameterOf(tag.typeArguments[realTagIndex], function)) {
      return (key: tag.getDisplayString(), isCollection: false);
    }
  }
  return null;
}

bool _isTypeParameterOf(DartType type, TopLevelFunctionElement function) =>
    type is TypeParameterType && function.typeParameters.contains(type.element);

/// Finds the innermost same-dtype operation site enclosing (or equal to)
/// [start] that has a castable operand covering [offset] whose concrete dtype
/// differs from the group's reference dtype (the first operand with a concrete
/// dtype).
///
/// Returns the operand and the reference dtype tag name, or `null` if there is
/// no such mismatch.
({SameDTypeOperand operand, String referenceTag})? findMismatchedOperandAt(
  AstNode start,
  int offset,
) {
  for (AstNode? current = start; current != null; current = current.parent) {
    for (final group in sameDTypeOperandGroups(current)) {
      final referenceTag = group
          .map((o) => o.concreteTag)
          .whereType<String>()
          .firstOrNull;
      if (referenceTag == null) continue;
      for (final operand in group) {
        final tag = operand.concreteTag;
        if (tag == null || tag == referenceTag || !operand.isCastable) continue;
        final expression = operand.expression;
        if (offset >= expression.offset && offset <= expression.end) {
          return (operand: operand, referenceTag: referenceTag);
        }
      }
    }
  }
  return null;
}
