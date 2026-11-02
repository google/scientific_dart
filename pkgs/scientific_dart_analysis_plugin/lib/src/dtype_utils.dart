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

/// Static dtype reasoning for `package:ndarray` and `package:ndarray_ma`
/// (`0.3.0`).
///
/// `package:ndarray` `0.3.0` encodes the element type of an array as a type
/// argument (`NDArray<Float64>`, `MaskedArray<Int32>`, ...) and uses
/// self-referential `SelfOf<R>` bounds on same-dtype multi-array functions,
/// capability markers (`BitwiseDType`, `IntegerDType`), single-slot `*Of<R>`
/// projections, and dependent `Out extends R` bounds on `out:` buffers.
///
/// Because operators (`+`, `-`, `*`, `/`, `<`, ...) and `Object?`-taking
/// instance methods still accept arbitrary array operands at compile time, and
/// because users writing mixed-dtype or runtime-typed (`NDArray<DTypeTag>`,
/// `NDArray<IntegerDType>`, `NDArray<BitwiseDType>`, `DType<DTypeTag>`) calls
/// in an IDE benefit from targeted diagnostics and `0.3.0` quick fixes
/// (`.astype(...)`, `*As(..., DType.x)`, `.asAnySpec`, `.asBitwiseDType`,
/// `.asIntegerDType`), the helpers in this library recover concrete and marker
/// dtypes, resolve `*Of<R>` projections, and group same-dtype / `Out`-bounded
/// parameters from element signatures.
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

/// Non-concrete `DTypeTag` supertypes/markers that may require explicit
/// widening (`.asAnySpec`, `.asBitwiseDType`, `.asIntegerDType`) when passed to
/// `SelfOf`-, `AnySpec`-, or capability-bounded `0.3.0` APIs.
const Set<String> kUnwidenedMarkerDTypeTags = {
  'DTypeTag',
  'IntegerDType',
  'BitwiseDType',
};

/// The 8 single-slot dtype projection interface names of `package:ndarray`.
const Set<String> kProjectionInterfaceNames = {
  'RealOf',
  'ElementOf',
  'RealFloatOf',
  'ComplexOf',
  'InexactOf',
  'AccumulatorOf',
  'DoublePrecisionOf',
  'DivideOf',
};

/// The binary operators of `NDArray` and `MaskedArray` that require an array
/// operand to have the receiver's dtype (`operator +(Object? other)` etc.).
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

/// The `NDArray` and `MaskedArray` instance methods taking an `Object?`
/// operand that must have the receiver's dtype when it is an array.
const Set<String> kSameDTypeMethodNames = {
  'eq',
  'ne',
  'lt',
  'le',
  'gt',
  'ge',
  'add',
  'subtract',
  'multiply',
  'divide',
  'floorDivide',
  'remainder',
  'minimum',
  'maximum',
  'power',
  'atan2',
  'hypot',
  'bitwiseAnd',
  'bitwiseOr',
  'bitwiseXor',
  'leftShift',
  'rightShift',
};

/// Mapping from binary `NDArray` operators to their mixed-dtype `*As`
/// top-level function counterparts in `package:ndarray`.
const Map<TokenType, String> kBinaryOperatorToMixedAsFunction = {
  TokenType.PLUS: 'addAs',
  TokenType.MINUS: 'subtractAs',
  TokenType.STAR: 'multiplyAs',
  TokenType.SLASH: 'divideAs',
  TokenType.TILDE_SLASH: 'floorDivideAs',
  TokenType.PERCENT: 'remainderAs',
  TokenType.AMPERSAND: 'bitwiseAndAs',
  TokenType.BAR: 'bitwiseOrAs',
  TokenType.CARET: 'bitwiseXorAs',
  TokenType.LT_LT: 'leftShiftAs',
  TokenType.GT_GT: 'rightShiftAs',
};

/// Mapping from same-dtype binary top-level functions in `package:ndarray` to
/// their mixed-dtype `*As` counterparts.
const Map<String, String> kFunctionToMixedAsFunction = {
  'add': 'addAs',
  'subtract': 'subtractAs',
  'multiply': 'multiplyAs',
  'divide': 'divideAs',
  'trueDivide': 'trueDivideAs',
  'floorDivide': 'floorDivideAs',
  'remainder': 'remainderAs',
  'mod': 'modAs',
  'fmod': 'fmodAs',
  'power': 'powerAs',
  'floatPower': 'floatPowerAs',
  'minimum': 'minimumAs',
  'maximum': 'maximumAs',
  'fmin': 'fminAs',
  'fmax': 'fmaxAs',
  'atan2': 'atan2As',
  'hypot': 'hypotAs',
  'copysign': 'copysignAs',
  'nextafter': 'nextafterAs',
  'logaddexp': 'logaddexpAs',
  'logaddexp2': 'logaddexp2As',
  'heaviside': 'heavisideAs',
  'gcd': 'gcdAs',
  'lcm': 'lcmAs',
  'bitwiseAnd': 'bitwiseAndAs',
  'bitwiseOr': 'bitwiseOrAs',
  'bitwiseXor': 'bitwiseXorAs',
  'leftShift': 'leftShiftAs',
  'rightShift': 'rightShiftAs',
  'matmul': 'matmulAs',
  'dot': 'dotAs',
  'vdot': 'vdotAs',
  'inner': 'innerAs',
  'outer': 'outerAs',
  'kron': 'kronAs',
  'tensordot': 'tensordotAs',
};

/// Projection applied to the reference operand when choosing the default target
/// `DType` for a mixed-dtype `*As` call.
const Map<String, String> kMixedAsTargetProjection = {
  'divideAs': 'DivideOf',
  'trueDivideAs': 'DivideOf',
  'atan2As': 'InexactOf',
  'hypotAs': 'InexactOf',
  'logaddexpAs': 'InexactOf',
  'logaddexp2As': 'InexactOf',
  'heavisideAs': 'InexactOf',
  'floatPowerAs': 'DoublePrecisionOf',
};

/// Returns whether [element] is declared in a library of `package:ndarray` or
/// `package:ndarray_ma`.
bool isDeclaredInNDArrayPackage(Element? element) {
  final uri = element?.library?.uri;
  if (uri == null || !uri.isScheme('package')) return false;
  final segments = uri.pathSegments;
  return segments.isNotEmpty &&
      (segments.first == 'ndarray' || segments.first == 'ndarray_ma');
}

/// Returns whether [element] is declared specifically in `package:ndarray`.
bool isDeclaredInCoreNDArrayPackage(Element? element) {
  final uri = element?.library?.uri;
  if (uri == null || !uri.isScheme('package')) return false;
  final segments = uri.pathSegments;
  return segments.isNotEmpty && segments.first == 'ndarray';
}

/// Returns the `NDArray` or `MaskedArray` instantiation of [type], ignoring
/// nullability; `null` if [type] is neither `NDArray` nor `MaskedArray`.
InterfaceType? asNDArrayInstance(DartType? type) {
  if (type is! InterfaceType) return null;
  if (_isTypedArrayElement(type.element)) return type;
  for (final supertype in type.allSupertypes) {
    if (_isTypedArrayElement(supertype.element)) return supertype;
  }
  return null;
}

bool _isTypedArrayElement(InterfaceElement element) =>
    (element.name == 'NDArray' || element.name == 'MaskedArray') &&
    isDeclaredInNDArrayPackage(element);

/// Returns the `DType<X>` instantiation of [type], or `null` if [type] is not
/// `DType`.
InterfaceType? asDTypeInstance(DartType? type) {
  if (type is! InterfaceType) return null;
  if (_isDTypeElement(type.element)) return type;
  for (final supertype in type.allSupertypes) {
    if (_isDTypeElement(supertype.element)) return supertype;
  }
  return null;
}

bool _isDTypeElement(InterfaceElement element) =>
    element.name == 'DType' && isDeclaredInNDArrayPackage(element);

/// Returns the dtype type argument `X` of an `NDArray<X>`, `MaskedArray<X>`,
/// or `DType<X>` [type], looking through one level of `List<...>` /
/// `Iterable<...>`.
DartType? ndarrayDTypeArgument(DartType? type) {
  if (type is! InterfaceType) return null;
  if (_isCoreCollection(type) && type.typeArguments.length == 1) {
    return ndarrayDTypeArgument(type.typeArguments.single);
  }
  final ndarray = asNDArrayInstance(type) ?? asDTypeInstance(type);
  if (ndarray == null || ndarray.typeArguments.length != 1) return null;
  return ndarray.typeArguments.single;
}

bool _isCoreCollection(InterfaceType type) {
  final name = type.element.name;
  return (name == 'List' || name == 'Iterable') &&
      type.element.library.isDartCore;
}

/// Returns the concrete dtype tag name (`'Float64'`, `'Int32'`, ...) of an
/// `NDArray<Tag>`, `MaskedArray<Tag>`, or `List<...>` [type].
///
/// Returns `null` when the dtype is not statically known to be one of the 15
/// concrete tags: `NDArray<DTypeTag>`, `NDArray<AnySpec>`, a type variable,
/// `dynamic`, or a non-array type.
String? concreteDTypeTagOf(DartType? type) {
  final tag = ndarrayDTypeArgument(type);
  if (tag is! InterfaceType) return null;
  final name = tag.element.name;
  if (name == null || !kConcreteDTypeTagToEnumName.containsKey(name)) {
    return null;
  }
  return isDeclaredInNDArrayPackage(tag.element) ? name : null;
}

/// Returns the non-concrete marker tag name (`'DTypeTag'`, `'IntegerDType'`,
/// or `'BitwiseDType'`) of an `NDArray<Tag>`, `MaskedArray<Tag>`, or
/// `DType<Tag>` [type], or `null` if [type] has a concrete or already-widened
/// `*Spec` tag.
String? unwidenedMarkerTagOf(DartType? type) {
  final tag = ndarrayDTypeArgument(type);
  if (tag is! InterfaceType) return null;
  final name = tag.element.name;
  if (name == null || !kUnwidenedMarkerDTypeTags.contains(name)) {
    return null;
  }
  return isDeclaredInNDArrayPackage(tag.element) ? name : null;
}

/// Resolves [tag] (a `DTypeTag` type—such as a concrete tag, `IntegerDType`,
/// or `BitwiseDType`—or an `NDArray<Tag>` / `MaskedArray<Tag>` type) through
/// the single-slot projection interface named [projection] (one of
/// [kProjectionInterfaceNames]).
///
/// Resolution inspects the `*Of<R>` supertype instantiation via
/// [InterfaceType.asInstanceOf] rather than `DTypeSpec` positional slots, so
/// capability markers (`IntegerDType`, `BitwiseDType`) and bounded type
/// parameters resolve any projection they pin even when they do not implement
/// `DTypeSpec`. Returns `null` if [tag] does not implement [projection].
DartType? projectDTypeTag(DartType? tag, String projection) {
  if (!kProjectionInterfaceNames.contains(projection)) return null;
  var resolved = ndarrayDTypeArgument(tag) ?? tag;
  while (resolved is TypeParameterType) {
    resolved = resolved.bound;
  }
  if (resolved is! InterfaceType) return null;
  InterfaceElement? projectionElement;
  if (resolved.element.name == projection &&
      isDeclaredInNDArrayPackage(resolved.element)) {
    projectionElement = resolved.element;
  } else {
    for (final supertype in resolved.allSupertypes) {
      if (supertype.element.name == projection &&
          isDeclaredInNDArrayPackage(supertype.element)) {
        projectionElement = supertype.element;
        break;
      }
    }
  }
  if (projectionElement == null) return null;
  final instance = resolved.asInstanceOf(projectionElement);
  if (instance == null || instance.typeArguments.length != 1) return null;
  return instance.typeArguments.single;
}

/// Resolves [tagType] through the single-slot `*Of<R>` interface named
/// [projectionInterfaceName] (one of [kProjectionInterfaceNames]).
///
/// Alias for [projectDTypeTag].
DartType? resolveDTypeProjection(
  DartType? tagType,
  String projectionInterfaceName,
) => projectDTypeTag(tagType, projectionInterfaceName);

/// Returns the concrete dtype tag name (`'Float64'`, `'Int64'`, ...) produced
/// by resolving [type] (a `DTypeTag`, `NDArray<Tag>`, or `MaskedArray<Tag>`)
/// through the single-slot projection interface [projection].
///
/// Returns `null` when [projection] does not resolve to one of the 15 concrete
/// `DTypeTag` classes (for example when [projection] is `'ElementOf'`, or when
/// [type] is `NDArray<DTypeTag>`, `NDArray<AnySpec>`, or `IntegerDType` under
/// `'AccumulatorOf'`).
String? projectedConcreteDTypeTagOf(DartType? type, String projection) {
  final projected = projectDTypeTag(type, projection);
  if (projected is! InterfaceType) return null;
  final name = projected.element.name;
  if (name == null || !kConcreteDTypeTagToEnumName.containsKey(name)) {
    return null;
  }
  return isDeclaredInNDArrayPackage(projected.element) ? name : null;
}

/// An operand of a same-dtype operation site.
final class SameDTypeOperand {
  /// The operand expression (an `NDArray`, `MaskedArray`, or a `List<...>`
  /// argument that is not a list literal).
  final Expression expression;

  /// Whether the operand can be wrapped in `.astype(...)` to fix a dtype
  /// mismatch: `false` for output buffers (`out:`-style named arguments) and
  /// for whole-list arguments.
  final bool isCastable;

  /// Optional `*Of<R>` projection interface name mapping this operand's
  /// concrete dtype to the group's canonical dtype (used when comparing an
  /// input operand against a projected `Out extends R` `out:` buffer).
  final String? projection;

  const SameDTypeOperand(
    this.expression, {
    required this.isCastable,
    this.projection,
  });

  /// The raw concrete dtype tag of [expression], or `null` if not statically
  /// known.
  String? get rawConcreteTag => concreteDTypeTagOf(expression.staticType);

  /// The concrete dtype tag of [expression] in the group's canonical dtype
  /// space (applying [projection] if non-null), or `null` if not statically
  /// known.
  String? get concreteTag {
    final proj = projection;
    if (proj != null) {
      return projectedConcreteDTypeTagOf(expression.staticType, proj);
    }
    return concreteDTypeTagOf(expression.staticType);
  }
}

/// Returns the groups of operands that the operation at [node] requires to
/// share one runtime dtype, in source order; empty if [node] is not a
/// same-dtype operation site.
///
/// Recognized sites are:
/// - [BinaryExpression]s using one of [kSameDTypeBinaryOperators] whose
///   operator resolves to `package:ndarray` or `package:ndarray_ma` and whose
///   operands are both `NDArray` / `MaskedArray`s;
/// - compound [AssignmentExpression]s using one of
///   [kSameDTypeCompoundAssignmentOperators] under the same conditions;
/// - [MethodInvocation]s of [kSameDTypeMethodNames] on an `NDArray` or
///   `MaskedArray` receiver with an array argument;
/// - [MethodInvocation]s of top-level functions declared in `package:ndarray`
///   or `package:ndarray_ma` without explicit type arguments, grouped per
///   [sameDTypeParameterGroups] (including `Out extends T` `out:` buffers).
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
    if (target == null) return const [];
    final positionalArgs = node.argumentList.arguments
        .where((a) => a is! NamedArgument)
        .toList();
    if (positionalArgs.length != 1) return const [];
    return _pairGroup(target, positionalArgs.single.argumentExpression);
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

/// Formal parameters of a `package:ndarray` or `package:ndarray_ma` function
/// that must receive arrays of one shared dtype.
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
/// Two formal parameters belong to the same group when their `NDArray` or
/// `MaskedArray` type arguments are identical and *dtype-identifying*:
/// - a bare type parameter `T` of [function] (`NDArray<T>`, including when `T`
///   is bounded by `SelfOf<...>` or one of the `*Of<R>` projection interfaces
///   such as `T extends SelfOf<InexactOf<R>>` in `atan2`, `hypot`, or
///   `logaddexp`);
/// - a dependent `Out extends T` type parameter where `T` itself is a direct
///   array type parameter of [function] (as in `sort<T, Out extends T>`,
///   `clip<T, Out extends T>`, `min<T, Out extends T>`);
/// - a direct single-slot projection instantiation (`NDArray<InexactOf<R>>`,
///   etc., from [kProjectionInterfaceNames]) whose type argument is a type
///   parameter of [function].
///
/// A group is reported when it has at least two parameters, or when it contains
/// a collection parameter (whose elements must agree among themselves, as in
/// `concatenate(List<NDArray<T>> arrays)`). Functions with distinct type
/// parameters per operand (the mixed-dtype `*As` variants) therefore yield no
/// groups.
List<SameDTypeParameterGroup> sameDTypeParameterGroups(
  TopLevelFunctionElement function,
) {
  // Collect which function type parameters appear directly as `NDArray<T>` on
  // a non-`out` parameter so dependent `Out extends T` parameters can join `T`'s
  // group without conflating projected `Out extends R` parameters (like `sum`
  // where `T extends AccumulatorOf<R>, Out extends R`).
  final directInputTypeParams = <TypeParameterElement>{};
  for (final parameter in function.formalParameters) {
    if (parameter.isNamed && (parameter.name ?? '').startsWith('out')) continue;
    final tag = ndarrayDTypeArgument(parameter.type);
    if (tag is TypeParameterType &&
        function.typeParameters.contains(tag.element)) {
      directInputTypeParams.add(tag.element);
    }
  }

  final parametersByKey = <String, List<FormalParameterElement>>{};
  final collectionParameters = <FormalParameterElement>{};
  for (final parameter in function.formalParameters) {
    final key = _dtypeIdentifyingKey(
      parameter.type,
      function,
      directInputTypeParams,
    );
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
  Set<TypeParameterElement> directInputTypeParams,
) {
  if (type is! InterfaceType) return null;
  if (_isCoreCollection(type) && type.typeArguments.length == 1) {
    final inner = _dtypeIdentifyingKey(
      type.typeArguments.single,
      function,
      directInputTypeParams,
    );
    if (inner == null || inner.isCollection) return null;
    return (key: inner.key, isCollection: true);
  }
  final tag = ndarrayDTypeArgument(type);
  if (tag == null) return null;
  if (_isTypeParameterOf(tag, function)) {
    final bound = (tag as TypeParameterType).bound;
    if (bound is TypeParameterType &&
        directInputTypeParams.contains(bound.element)) {
      return (key: bound.getDisplayString(), isCollection: false);
    }
    return (key: tag.getDisplayString(), isCollection: false);
  }
  if (tag is InterfaceType &&
      kProjectionInterfaceNames.contains(tag.element.name) &&
      isDeclaredInNDArrayPackage(tag.element) &&
      tag.typeArguments.length == 1 &&
      _isTypeParameterOf(tag.typeArguments.single, function)) {
    return (key: tag.getDisplayString(), isCollection: false);
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
/// Returns the enclosing operation node, the mismatched operand, and the
/// reference dtype tag name, or `null` if there is no such mismatch.
({AstNode site, SameDTypeOperand operand, String referenceTag})?
findMismatchedOperandAt(AstNode start, int offset) {
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
          return (site: current, operand: operand, referenceTag: referenceTag);
        }
      }
    }
  }
  return null;
}

/// Describes a mixed-dtype `*As` call replacement for a mismatched binary
/// operator or function call site.
final class MixedDTypeAsReplacement {
  /// The AST node to replace (`BinaryExpression` or `MethodInvocation`).
  final AstNode site;

  /// The name of the `*As` function (`addAs`, `matmulAs`, ...).
  final String asFunctionName;

  /// The first positional operand expression.
  final Expression firstOperand;

  /// The second positional operand expression.
  final Expression secondOperand;

  /// Additional named arguments to preserve after the `DType.<target>`
  /// argument.
  final List<NamedArgument> namedArguments;

  /// The `DType` enum constant name (`float64`, `int64`, ...) for the third
  /// positional argument.
  final String targetEnumName;

  const MixedDTypeAsReplacement({
    required this.site,
    required this.asFunctionName,
    required this.firstOperand,
    required this.secondOperand,
    required this.namedArguments,
    required this.targetEnumName,
  });
}

/// Finds a mixed-dtype `*As` replacement for the mismatched `NDArray`
/// operation enclosing [start] at [offset], or `null` if the site has no `*As`
/// counterpart (for example, `MaskedArray` operations or `where`).
MixedDTypeAsReplacement? findMixedDTypeAsReplacementAt(
  AstNode start,
  int offset,
) {
  final mismatch = findMismatchedOperandAt(start, offset);
  if (mismatch == null) return null;
  final site = mismatch.site;

  if (site is BinaryExpression) {
    final asFn = kBinaryOperatorToMixedAsFunction[site.operator.type];
    if (asFn == null) return null;
    if (!isDeclaredInCoreNDArrayPackage(site.element)) return null;
    final leftType = asNDArrayInstance(site.leftOperand.staticType);
    final rightType = asNDArrayInstance(site.rightOperand.staticType);
    if (leftType?.element.name != 'NDArray' ||
        rightType?.element.name != 'NDArray') {
      return null;
    }
    final targetTag = _resolveMixedAsTargetTag(
      asFn,
      site.leftOperand.staticType,
      mismatch.referenceTag,
    );
    final enumName = kConcreteDTypeTagToEnumName[targetTag];
    if (enumName == null) return null;
    return MixedDTypeAsReplacement(
      site: site,
      asFunctionName: asFn,
      firstOperand: site.leftOperand,
      secondOperand: site.rightOperand,
      namedArguments: const [],
      targetEnumName: enumName,
    );
  }

  if (site is MethodInvocation) {
    final element = site.methodName.element;
    if (element is! TopLevelFunctionElement ||
        !isDeclaredInCoreNDArrayPackage(element)) {
      return null;
    }
    final asFn = kFunctionToMixedAsFunction[element.name];
    if (asFn == null) return null;
    final positional = site.argumentList.arguments
        .where((a) => a is! NamedArgument)
        .toList();
    if (positional.length != 2) return null;
    final first = positional[0].argumentExpression;
    final second = positional[1].argumentExpression;
    final named = site.argumentList.arguments
        .whereType<NamedArgument>()
        .toList();
    final targetTag = _resolveMixedAsTargetTag(
      asFn,
      first.staticType,
      mismatch.referenceTag,
    );
    final enumName = kConcreteDTypeTagToEnumName[targetTag];
    if (enumName == null) return null;
    return MixedDTypeAsReplacement(
      site: site,
      asFunctionName: asFn,
      firstOperand: first,
      secondOperand: second,
      namedArguments: named,
      targetEnumName: enumName,
    );
  }

  return null;
}

String _resolveMixedAsTargetTag(
  String asFunctionName,
  DartType? firstOperandType,
  String fallbackReferenceTag,
) {
  final projection = kMixedAsTargetProjection[asFunctionName];
  if (projection != null && firstOperandType != null) {
    final projected = projectedConcreteDTypeTagOf(firstOperandType, projection);
    if (projected != null) return projected;
  }
  return fallbackReferenceTag;
}

/// Inspects whether [expression] at an invocation or operator site is an
/// unwidened runtime-typed array (`NDArray<DTypeTag>`, `NDArray<IntegerDType>`,
/// `NDArray<BitwiseDType>`, `MaskedArray<...>`) or `DType<DTypeTag>` that
/// requires `.asAnySpec`, `.asBitwiseDType`, or `.asIntegerDType` to satisfy
/// `0.3.0` static bounds.
///
/// Returns the required getter name (`'asAnySpec'`, `'asBitwiseDType'`, or
/// `'asIntegerDType'`) and the non-concrete marker tag name, or `null` if no
/// widening is needed.
({String getterName, String markerTag, String operationName})?
requiredSpecWideningForParameter({
  required Expression argument,
  required DartType parameterType,
  required List<TypeParameterElement> typeParameters,
  required String operationName,
}) {
  final markerTag = unwidenedMarkerTagOf(argument.staticType);
  if (markerTag == null) return null;

  final paramTag = ndarrayDTypeArgument(parameterType);
  if (paramTag == null) return null;

  final DartType constraintType;
  if (paramTag is TypeParameterType &&
      typeParameters.contains(paramTag.element)) {
    constraintType = paramTag.bound;
  } else {
    constraintType = paramTag;
  }

  final constraintDisplay = constraintType.getDisplayString();
  if (constraintDisplay == 'DTypeTag' || constraintDisplay == 'dynamic') {
    return null;
  }

  if (constraintDisplay.contains('IntegerDType') ||
      constraintDisplay.contains('AnyIntegerSpec')) {
    if (markerTag != 'IntegerDType' || constraintDisplay.contains('SelfOf')) {
      return (
        getterName: 'asIntegerDType',
        markerTag: markerTag,
        operationName: operationName,
      );
    }
    return null;
  }

  if (constraintDisplay.contains('BitwiseDType') ||
      constraintDisplay.contains('AnyBitwiseSpec')) {
    if (markerTag == 'DTypeTag' || constraintDisplay.contains('SelfOf')) {
      return (
        getterName: 'asBitwiseDType',
        markerTag: markerTag,
        operationName: operationName,
      );
    }
    return null;
  }

  if (constraintDisplay.contains('SelfOf') ||
      constraintDisplay.contains('AnySpec')) {
    return (
      getterName: 'asAnySpec',
      markerTag: markerTag,
      operationName: operationName,
    );
  }

  for (final proj in kProjectionInterfaceNames) {
    if (constraintDisplay.contains(proj)) {
      if (projectDTypeTag(argument.staticType, proj) == null) {
        return (
          getterName: 'asAnySpec',
          markerTag: markerTag,
          operationName: operationName,
        );
      }
      return null;
    }
  }

  return null;
}

/// Finds all unwidened runtime-typed arguments (`NDArray<DTypeTag>`,
/// `NDArray<IntegerDType>`, `NDArray<BitwiseDType>`, `MaskedArray<...>`,
/// `DType<DTypeTag>`) inside [node] that require `.asAnySpec`,
/// `.asBitwiseDType`, or `.asIntegerDType`.
List<
  ({
    Expression expression,
    String getterName,
    String markerTag,
    String operationName,
  })
>
unwidenedDynamicDTypeSites(AstNode node) {
  final results =
      <
        ({
          Expression expression,
          String getterName,
          String markerTag,
          String operationName,
        })
      >[];

  if (node is BinaryExpression) {
    final op = node.operator.type;
    final leftMarker = unwidenedMarkerTagOf(node.leftOperand.staticType);
    if (leftMarker != null &&
        asNDArrayInstance(node.leftOperand.staticType) != null) {
      if ((op == TokenType.AMPERSAND ||
              op == TokenType.BAR ||
              op == TokenType.CARET) &&
          leftMarker == 'DTypeTag') {
        results.add((
          expression: node.leftOperand,
          getterName: 'asBitwiseDType',
          markerTag: leftMarker,
          operationName: 'operator ${node.operator.lexeme}',
        ));
      } else if ((op == TokenType.LT_LT || op == TokenType.GT_GT) &&
          (leftMarker == 'DTypeTag' || leftMarker == 'BitwiseDType')) {
        results.add((
          expression: node.leftOperand,
          getterName: 'asIntegerDType',
          markerTag: leftMarker,
          operationName: 'operator ${node.operator.lexeme}',
        ));
      }
    }
    return results;
  }

  if (node is PrefixExpression && node.operator.type == TokenType.TILDE) {
    final marker = unwidenedMarkerTagOf(node.operand.staticType);
    if (marker == 'DTypeTag' &&
        asNDArrayInstance(node.operand.staticType) != null) {
      results.add((
        expression: node.operand,
        getterName: 'asBitwiseDType',
        markerTag: marker!,
        operationName: 'operator ~',
      ));
    }
    return results;
  }

  if (node is MethodInvocation) {
    final element = node.methodName.element;
    if (element is! ExecutableElement || !isDeclaredInNDArrayPackage(element)) {
      return results;
    }
    if (node.typeArguments != null) return results;
    final typeParams = <TypeParameterElement>[
      ...element.typeParameters,
      if (element.enclosingElement is InterfaceElement)
        ...(element.enclosingElement as InterfaceElement).typeParameters,
    ];
    _collectUnwidenedArguments(
      argumentList: node.argumentList,
      formalParameters: element.formalParameters,
      typeParameters: typeParams,
      operationName: element.name ?? node.methodName.name,
      results: results,
    );
  } else if (node is InstanceCreationExpression) {
    final element = node.constructorName.element;
    if (element == null || !isDeclaredInNDArrayPackage(element)) {
      return results;
    }
    if (node.constructorName.type.typeArguments != null) return results;
    final typeParams = <TypeParameterElement>[
      ...element.typeParameters,
      ...element.enclosingElement.typeParameters,
    ];
    final opName = element.name == null || element.name == 'new'
        ? element.enclosingElement.name ?? 'NDArray'
        : '${element.enclosingElement.name}.${element.name}';
    _collectUnwidenedArguments(
      argumentList: node.argumentList,
      formalParameters: element.formalParameters,
      typeParameters: typeParams,
      operationName: opName,
      results: results,
    );
  }

  return results;
}

void _collectUnwidenedArguments({
  required ArgumentList argumentList,
  required List<FormalParameterElement> formalParameters,
  required List<TypeParameterElement> typeParameters,
  required String operationName,
  required List<
    ({
      Expression expression,
      String getterName,
      String markerTag,
      String operationName,
    })
  >
  results,
}) {
  final positionalParameters = formalParameters
      .where((p) => p.isPositional)
      .toList();
  var positionalIndex = 0;
  for (final argument in argumentList.arguments) {
    final FormalParameterElement? parameter;
    if (argument is NamedArgument) {
      final name = argument.name.lexeme;
      parameter = formalParameters
          .where((p) => p.isNamed && p.name == name)
          .firstOrNull;
    } else {
      parameter = positionalIndex < positionalParameters.length
          ? positionalParameters[positionalIndex]
          : null;
      positionalIndex++;
    }
    if (parameter == null) continue;
    final expr = argument.argumentExpression;
    if (expr is ListLiteral &&
        parameter.type is InterfaceType &&
        _isCoreCollection(parameter.type as InterfaceType)) {
      for (final elem in expr.elements) {
        if (elem is Expression) {
          final req = requiredSpecWideningForParameter(
            argument: elem,
            parameterType: parameter.type,
            typeParameters: typeParameters,
            operationName: operationName,
          );
          if (req != null) {
            results.add((
              expression: elem,
              getterName: req.getterName,
              markerTag: req.markerTag,
              operationName: req.operationName,
            ));
          }
        }
      }
      continue;
    }
    final req = requiredSpecWideningForParameter(
      argument: expr,
      parameterType: parameter.type,
      typeParameters: typeParameters,
      operationName: operationName,
    );
    if (req != null) {
      results.add((
        expression: expr,
        getterName: req.getterName,
        markerTag: req.markerTag,
        operationName: req.operationName,
      ));
    }
  }
}

/// Finds the innermost unwidened dynamic dtype expression covering [offset]
/// starting from [start].
({Expression expression, String getterName})? findUnwidenedDynamicDTypeAt(
  AstNode start,
  int offset,
) {
  for (AstNode? current = start; current != null; current = current.parent) {
    for (final site in unwidenedDynamicDTypeSites(current)) {
      final expr = site.expression;
      if (offset >= expr.offset && offset <= expr.end) {
        return (expression: expr, getterName: site.getterName);
      }
    }
  }
  return null;
}
