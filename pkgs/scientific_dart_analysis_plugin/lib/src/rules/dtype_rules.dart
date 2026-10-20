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

import '../dtype_utils.dart';

// =============================================================================
// 16. ndarray_mismatched_dtype_operands
// =============================================================================

/// Flags same-dtype `NDArray` and `MaskedArray` operations whose operands have
/// different concrete static dtypes, such as `f64 + f32`, `f64.eq(f32)`,
/// `sort(f64, out: f32)`, or `add(f64, i32)`.
///
/// In `package:ndarray` `0.3.0`:
/// - Operators (`+ - * / ~/ % & | ^ << >> < <= > >=`), compound assignments,
///   and `Object?`-taking instance methods (`eq`, `ne`, and `MaskedArray`
///   binary methods) accept any array operand at compile time and throw an
///   `ArgumentError` at runtime when dtypes differ.
/// - Same-dtype multi-array functions (`add`, `matmul`, `atan2`, `where`,
///   `concatenate`, ...) and `Out extends T`-bounded `out:` buffers also
///   participate so the IDE surfaces `0.3.0` quick fixes (`.astype(DType.x)`
///   and the mixed-dtype `*As` functions such as `addAs(a, b, DType.x)`).
///
/// Operands whose dtype is not statically concrete (`NDArray<DTypeTag>`,
/// `NDArray<AnySpec>`, type variables, `dynamic`) never participate, and
/// invocations with explicit type arguments are skipped.
final class MismatchedDTypeOperandsRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_mismatched_dtype_operands',
    "Operand of type '{3}<{1}>' in a same-dtype operation whose reference "
        "operand is '{3}<{0}>'; mixed dtypes throw an ArgumentError at "
        'runtime (or fail SelfOf / Out inference).',
    correctionMessage:
        "Cast explicitly with '.astype(DType.{2})' (or cast the other "
        "operand), or use the mixed-dtype '*As' variant (addAs, multiplyAs, "
        'matmulAs, ...) with an explicit result DType.',
    severity: DiagnosticSeverity.WARNING,
  );

  MismatchedDTypeOperandsRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not combine NDArrays or MaskedArrays of different concrete '
            'dtypes in same-dtype operations; cast with .astype() or use the '
            '*As variants.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _MismatchedDTypeOperandsVisitor(this);
    registry.addBinaryExpression(this, visitor);
    registry.addAssignmentExpression(this, visitor);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _MismatchedDTypeOperandsVisitor extends SimpleAstVisitor<void> {
  final MismatchedDTypeOperandsRule rule;

  _MismatchedDTypeOperandsVisitor(this.rule);

  @override
  void visitBinaryExpression(BinaryExpression node) => _check(node);

  @override
  void visitAssignmentExpression(AssignmentExpression node) => _check(node);

  @override
  void visitMethodInvocation(MethodInvocation node) => _check(node);

  void _check(AstNode node) {
    for (final group in sameDTypeOperandGroups(node)) {
      String? referenceTag;
      for (final operand in group) {
        final tag = operand.concreteTag;
        if (tag == null) continue;
        if (referenceTag == null) {
          referenceTag = tag;
        } else if (tag != referenceTag) {
          final arrayClassName =
              asNDArrayInstance(operand.expression.staticType)?.element.name ??
              'NDArray';
          rule.reportAtNode(
            operand.expression,
            arguments: [
              referenceTag,
              tag,
              kConcreteDTypeTagToEnumName[referenceTag]!,
              arrayClassName,
            ],
          );
        }
      }
    }
  }
}

// =============================================================================
// 17. ndarray_unsupported_dtype_operation
// =============================================================================

/// Flags `package:ndarray` and `package:ndarray_ma` operations applied to
/// arrays whose concrete static dtype the operation rejects at runtime.
///
/// While `0.3.0`'s `BitwiseDType` and `IntegerDType` bounds reject floating and
/// complex operands on same-dtype bitwise (`&`, `|`, `^`, `~`, `bitwiseAnd`,
/// `bitwiseOr`, `bitwiseXor`, `invert`) and integer (`<<`, `>>`, `leftShift`,
/// `rightShift`, `gcd`, `lcm`) operations at compile time, several operation
/// families still accept all `DTypeTag`s (or `SelfOf<DTypeTag>`) statically and
/// throw at runtime for unsupported concrete dtypes:
///
/// | Operation family                                                | Rejected dtypes                  |
/// |-----------------------------------------------------------------|----------------------------------|
/// | `bitwiseAndAs`, `bitwiseOrAs`, `bitwiseXorAs` (+ `bitwise*`)    | floating-point, complex          |
/// | `leftShiftAs`, `rightShiftAs`, `gcdAs`, `lcmAs` (+ `leftShift`) | floating-point, complex, boolean |
/// | `greater`, `greaterEqual`, `less`, `lessEqual`, `<`, `<=`, `>`, `>=`, `sort`, `argsort`, `partition`, `argpartition`, `searchsorted`, `ptp`, `clip` | complex |
/// | `remainder`, `mod`, `floorDivide`, `fmod`, `divmod`, `atan2`, `hypot`, `~/`, `%` | complex |
final class UnsupportedDTypeOperationRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_unsupported_dtype_operation',
    "'{0}' does not support {3}<{1}> operands ({2}) and throws at runtime.",
    correctionMessage:
        "Cast to a supported dtype with '.astype(...)' first; for ordering "
        'complex arrays compare real()/imag() parts or abs() magnitudes '
        'instead.',
    severity: DiagnosticSeverity.WARNING,
  );

  UnsupportedDTypeOperationRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not apply bitwise, shift, modulo, or ordering operations to '
            'array dtypes that reject them at runtime.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _UnsupportedDTypeOperationVisitor(this);
    registry.addMethodInvocation(this, visitor);
    registry.addBinaryExpression(this, visitor);
    registry.addAssignmentExpression(this, visitor);
  }
}

/// A family of operations sharing one set of rejected dtypes.
enum _UnsupportedDTypeFamily {
  bitwise(
    {'bitwiseAnd', 'bitwiseOr', 'bitwiseXor', 'invert'},
    {'bitwiseAndAs', 'bitwiseOrAs', 'bitwiseXorAs'},
    {...kFloatingDTypeTags, ...kComplexDTypeTags},
    'bitwise operations are only defined for integer and boolean dtypes',
  ),
  shift(
    {'leftShift', 'rightShift', 'gcd', 'lcm'},
    {'leftShiftAs', 'rightShiftAs', 'gcdAs', 'lcmAs'},
    {...kFloatingDTypeTags, ...kComplexDTypeTags, 'Boolean'},
    'shift, gcd, and lcm operations are only defined for integer dtypes',
  ),
  ordering(
    {
      'greater',
      'greaterEqual',
      'less',
      'lessEqual',
      'lt',
      'le',
      'gt',
      'ge',
      'sort',
      'argsort',
      'partition',
      'argpartition',
      'searchsorted',
      'ptp',
      'clip',
      'clipArray',
    },
    {},
    kComplexDTypeTags,
    'ordering comparisons are undefined for complex dtypes',
  ),
  realOnly(
    {'remainder', 'mod', 'floorDivide', 'divmod', 'atan2', 'hypot', 'fmod'},
    {'remainderAs', 'modAs', 'floorDivideAs', 'fmodAs', 'atan2As', 'hypotAs'},
    kComplexDTypeTags,
    'operation is undefined for complex dtypes',
  );

  /// Same-dtype function/method names of the family.
  final Set<String> functionNames;

  /// Mixed-dtype `*As` function names of the family, whose third positional
  /// argument is the explicit target `DType`.
  final Set<String> asFunctionNames;

  /// Concrete dtype tag names rejected at runtime.
  final Set<String> rejectedTags;

  /// Human-readable reason used in the diagnostic message.
  final String reason;

  const _UnsupportedDTypeFamily(
    this.functionNames,
    this.asFunctionNames,
    this.rejectedTags,
    this.reason,
  );

  static _UnsupportedDTypeFamily? forFunctionName(String name) {
    for (final family in values) {
      if (family.functionNames.contains(name) ||
          family.asFunctionNames.contains(name)) {
        return family;
      }
    }
    return null;
  }

  /// The `DType` enum constant names (`float64`, ...) of [rejectedTags].
  Set<String> get rejectedEnumNames =>
      rejectedTags.map((tag) => kConcreteDTypeTagToEnumName[tag]!).toSet();

  /// The tag name of [enumName], or `null` if it is not a `DType` constant.
  static String? tagOfEnumName(String enumName) {
    for (final entry in kConcreteDTypeTagToEnumName.entries) {
      if (entry.value == enumName) return entry.key;
    }
    return null;
  }
}

final class _UnsupportedDTypeOperationVisitor extends SimpleAstVisitor<void> {
  static const Set<TokenType> _orderingOperators = {
    TokenType.LT,
    TokenType.LT_EQ,
    TokenType.GT,
    TokenType.GT_EQ,
  };

  static const Set<TokenType> _realOnlyOperators = {
    TokenType.TILDE_SLASH,
    TokenType.PERCENT,
  };

  static const Set<TokenType> _realOnlyAssignmentOperators = {
    TokenType.TILDE_SLASH_EQ,
    TokenType.PERCENT_EQ,
  };

  final UnsupportedDTypeOperationRule rule;

  _UnsupportedDTypeOperationVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final element = node.methodName.element;
    if (element is! TopLevelFunctionElement && element is! MethodElement) {
      return;
    }
    if (!isDeclaredInNDArrayPackage(element)) return;
    final name = element?.name;
    if (name == null) return;
    final family = _UnsupportedDTypeFamily.forFunctionName(name);
    if (family == null) return;

    final receiver = node.realTarget;
    if (receiver != null) {
      final receiverTag = concreteDTypeTagOf(receiver.staticType);
      if (receiverTag != null && family.rejectedTags.contains(receiverTag)) {
        final cls =
            asNDArrayInstance(receiver.staticType)?.element.name ?? 'NDArray';
        rule.reportAtNode(
          receiver,
          arguments: [name, receiverTag, family.reason, cls],
        );
        return;
      }
    }

    final positionalArguments = node.argumentList.arguments
        .where((argument) => argument is! NamedArgument)
        .map((argument) => argument.argumentExpression)
        .toList();
    for (final argument in positionalArguments) {
      final tag = concreteDTypeTagOf(argument.staticType);
      if (tag != null && family.rejectedTags.contains(tag)) {
        final cls =
            asNDArrayInstance(argument.staticType)?.element.name ?? 'NDArray';
        rule.reportAtNode(argument, arguments: [name, tag, family.reason, cls]);
        break;
      }
    }
    if (family.asFunctionNames.contains(name) &&
        positionalArguments.length >= 3) {
      final targetDType = positionalArguments[2];
      final enumName = switch (targetDType) {
        PrefixedIdentifier(prefix: SimpleIdentifier(name: 'DType')) =>
          targetDType.identifier.name,
        PropertyAccess(target: SimpleIdentifier(name: 'DType')) =>
          targetDType.propertyName.name,
        _ => null,
      };
      if (enumName != null && family.rejectedEnumNames.contains(enumName)) {
        rule.reportAtNode(
          targetDType,
          arguments: [
            name,
            _UnsupportedDTypeFamily.tagOfEnumName(enumName)!,
            family.reason,
            'NDArray',
          ],
        );
      }
    }
  }

  @override
  void visitBinaryExpression(BinaryExpression node) {
    final _UnsupportedDTypeFamily? family;
    if (_orderingOperators.contains(node.operator.type)) {
      family = _UnsupportedDTypeFamily.ordering;
    } else if (_realOnlyOperators.contains(node.operator.type)) {
      family = _UnsupportedDTypeFamily.realOnly;
    } else {
      return;
    }
    if (!isDeclaredInNDArrayPackage(node.element)) return;
    for (final operand in [node.leftOperand, node.rightOperand]) {
      final tag = concreteDTypeTagOf(operand.staticType);
      if (tag != null && family.rejectedTags.contains(tag)) {
        final cls =
            asNDArrayInstance(operand.staticType)?.element.name ?? 'NDArray';
        rule.reportAtNode(
          operand,
          arguments: [
            'operator ${node.operator.lexeme}',
            tag,
            family.reason,
            cls,
          ],
        );
        break;
      }
    }
  }

  @override
  void visitAssignmentExpression(AssignmentExpression node) {
    if (!_realOnlyAssignmentOperators.contains(node.operator.type)) return;
    if (!isDeclaredInNDArrayPackage(node.element)) return;
    const family = _UnsupportedDTypeFamily.realOnly;
    for (final operand in [node.leftHandSide, node.rightHandSide]) {
      final tag = concreteDTypeTagOf(operand.staticType);
      if (tag != null && family.rejectedTags.contains(tag)) {
        final cls =
            asNDArrayInstance(operand.staticType)?.element.name ?? 'NDArray';
        rule.reportAtNode(
          operand,
          arguments: [
            'operator ${node.operator.lexeme}',
            tag,
            family.reason,
            cls,
          ],
        );
        break;
      }
    }
  }
}

// =============================================================================
// 18. ndarray_unwidened_dynamic_dtype
// =============================================================================

/// Flags runtime-typed `NDArray` / `MaskedArray` / `DType` expressions
/// (`NDArray<DTypeTag>`, `NDArray<IntegerDType>`, `NDArray<BitwiseDType>`,
/// `DType<DTypeTag>`) passed to `0.3.0` functions, constructors, or operators
/// whose `SelfOf<...>`, `*Of<...>`, `AnySpec`, `BitwiseDType`, or
/// `IntegerDType` bounds require explicit widening with `.asAnySpec`,
/// `.asBitwiseDType`, or `.asIntegerDType`.
final class UnwidenedDynamicDTypeRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_unwidened_dynamic_dtype',
    "Runtime-typed '{0}' does not satisfy the static dtype bound of '{1}' in "
        '0.3.0; widen explicitly with .{2}.',
    correctionMessage:
        "Append '.{2}' to opt into runtime dtype validation for this call.",
    severity: DiagnosticSeverity.WARNING,
  );

  UnwidenedDynamicDTypeRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Widen runtime-typed NDArray/MaskedArray/DType<DTypeTag> operands '
            'with .asAnySpec, .asBitwiseDType, or .asIntegerDType when '
            'calling SelfOf- or projection-bounded 0.3.0 APIs.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _UnwidenedDynamicDTypeVisitor(this);
    registry.addMethodInvocation(this, visitor);
    registry.addInstanceCreationExpression(this, visitor);
    registry.addBinaryExpression(this, visitor);
    registry.addPrefixExpression(this, visitor);
  }
}

final class _UnwidenedDynamicDTypeVisitor extends SimpleAstVisitor<void> {
  final UnwidenedDynamicDTypeRule rule;

  _UnwidenedDynamicDTypeVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) => _check(node);

  @override
  void visitInstanceCreationExpression(InstanceCreationExpression node) =>
      _check(node);

  @override
  void visitBinaryExpression(BinaryExpression node) => _check(node);

  @override
  void visitPrefixExpression(PrefixExpression node) => _check(node);

  void _check(AstNode node) {
    for (final site in unwidenedDynamicDTypeSites(node)) {
      final typeDisplay =
          site.expression.staticType?.getDisplayString() ?? site.markerTag;
      rule.reportAtNode(
        site.expression,
        arguments: [typeDisplay, site.operationName, site.getterName],
      );
    }
  }
}
