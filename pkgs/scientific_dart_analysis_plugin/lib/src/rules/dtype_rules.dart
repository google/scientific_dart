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

/// Flags same-dtype `NDArray` operations whose operands have different
/// concrete static dtypes, such as `f64 + f32` or `add(f64, i32)`.
///
/// Same-dtype operations are declared with one shared type parameter
/// (`add<T extends DTypeTag>(NDArray<T> a, NDArray<T> b)`) or take an
/// `Object?` operand (`NDArray<T>.operator +`). Covariant class type
/// parameters let such calls compile for mixed dtypes (the shared `T` is
/// inferred as `DTypeTag`), but they throw an `ArgumentError` at runtime.
///
/// Covered sites (see `sameDTypeOperandGroups`):
/// - the operators `+ - * / ~/ % & | ^ << >> < <= > >=` (and their compound
///   assignment forms) with `NDArray` operands on both sides;
/// - the `eq` / `ne` methods with an `NDArray` argument;
/// - top-level functions declared in `package:ndarray` whose signature ties
///   two or more array parameters to one dtype, derived from the element
///   model rather than a fixed list (`add`, `maximum`, `atan2`, `dot`, `where`,
///   `concatenate(List<NDArray<T>>)`, ...). The mixed-dtype `*As` variants
///   declare distinct type parameters per operand and are therefore never
///   flagged.
///
/// Operands whose dtype is not statically concrete (`NDArray<DTypeTag>`,
/// `NDArray<AnySpec>`, type variables, `dynamic`) never participate, and
/// invocations with explicit type arguments are skipped.
final class MismatchedDTypeOperandsRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_mismatched_dtype_operands',
    "Operand of type 'NDArray<{1}>' in a same-dtype operation whose reference "
        "operand is 'NDArray<{0}>'; mixed dtypes compile (the shared type "
        'parameter is inferred as DTypeTag) but throw an ArgumentError at '
        'runtime.',
    correctionMessage:
        "Cast explicitly with '.astype(DType.{2})' (or cast the other "
        "operand), or use the mixed-dtype '*As' variant (addAs, multiplyAs, "
        '...) with an explicit result DType.',
    severity: DiagnosticSeverity.WARNING,
  );

  MismatchedDTypeOperandsRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not combine NDArrays of different concrete dtypes in '
            'same-dtype operations; cast with .astype() or use the *As '
            'variants.',
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
          rule.reportAtNode(
            operand.expression,
            arguments: [
              referenceTag,
              tag,
              kConcreteDTypeTagToEnumName[referenceTag]!,
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

/// Flags `package:ndarray` operations applied to arrays whose concrete static
/// dtype the operation rejects at runtime.
///
/// The runtime matrix encoded here (verified against
/// `operations/math/bitwise.dart` and `operations/math/logical.dart`):
///
/// | Operation                                        | Rejected dtypes          |
/// |--------------------------------------------------|--------------------------|
/// | `bitwiseAnd`, `bitwiseOr`, `bitwiseXor`, `invert` | floating-point, complex |
/// | `bitwiseAndAs`, `bitwiseOrAs`, `bitwiseXorAs`     | floating-point, complex |
/// | `leftShift`, `rightShift` (+ `*As`)               | floating-point, complex, boolean |
/// | `greater`, `greaterEqual`, `less`, `lessEqual`    | complex                  |
/// | operators `<`, `<=`, `>`, `>=`                    | complex                  |
///
/// Boolean is treated as supported by the bitwise (non-shift) operations, in
/// line with the `BitwiseDType` marker. The bitwise/shift *operators* are not
/// covered: their statically constrained extension signatures already reject
/// unsupported dtypes at compile time.
final class UnsupportedDTypeOperationRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_unsupported_dtype_operation',
    "'{0}' does not support NDArray<{1}> operands ({2}) and throws at "
        'runtime.',
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
            'Do not apply bitwise, shift, or ordering operations to NDArray '
            'dtypes that reject them at runtime.',
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
  shift({'leftShift', 'rightShift'}, {'leftShiftAs', 'rightShiftAs'}, {
    ...kFloatingDTypeTags,
    ...kComplexDTypeTags,
    'Boolean',
  }, 'shift operations are only defined for integer dtypes'),
  ordering(
    {'greater', 'greaterEqual', 'less', 'lessEqual'},
    {},
    kComplexDTypeTags,
    'ordering comparisons are undefined for complex dtypes',
  ),
  realOnly(
    {'remainder', 'floorDivide', 'atan2', 'hypot', 'fmod'},
    {},
    kComplexDTypeTags,
    'operation is undefined for complex dtypes',
  );

  /// Same-dtype function names of the family.
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

  final UnsupportedDTypeOperationRule rule;

  _UnsupportedDTypeOperationVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final element = node.methodName.element;
    if (element is! TopLevelFunctionElement ||
        !isDeclaredInNDArrayPackage(element)) {
      return;
    }
    final name = element.name;
    if (name == null) return;
    final family = _UnsupportedDTypeFamily.forFunctionName(name);
    if (family == null) return;

    final positionalArguments = node.argumentList.arguments
        .where((argument) => argument is! NamedArgument)
        .map((argument) => argument.argumentExpression)
        .toList();
    for (final argument in positionalArguments) {
      final tag = concreteDTypeTagOf(argument.staticType);
      if (tag != null && family.rejectedTags.contains(tag)) {
        rule.reportAtNode(argument, arguments: [name, tag, family.reason]);
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
          ],
        );
      }
    }
  }

  @override
  void visitBinaryExpression(BinaryExpression node) {
    if (!_orderingOperators.contains(node.operator.type)) return;
    if (!isDeclaredInNDArrayPackage(node.element)) return;
    const family = _UnsupportedDTypeFamily.ordering;
    for (final operand in [node.leftOperand, node.rightOperand]) {
      final tag = concreteDTypeTagOf(operand.staticType);
      if (tag != null && family.rejectedTags.contains(tag)) {
        rule.reportAtNode(
          operand,
          arguments: ["operator ${node.operator.lexeme}", tag, family.reason],
        );
        break;
      }
    }
  }
}
