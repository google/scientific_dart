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
    final leftSrc = targetNode.leftOperand.toSource();
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

    final body = expr.thisOrAncestorOfType<FunctionBody>();
    final decls = body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
    final isView = traceRootArrayAndView(expr, decls).throughView;
    final insertion = isView
        ? '.copy().detachToParentScope()'
        : '.detachToParentScope()';

    await builder.addDartFileEdit(file, (builder) {
      builder.addSimpleInsertion(expr.end, insertion);
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
    final target = node;
    if (target is MethodInvocation &&
        (target.methodName.name == 'detachToParentScope' ||
            target.methodName.name == 'detachFromScope')) {
      final receiver = target.realTarget;
      if (receiver != null) {
        await builder.addDartFileEdit(file, (builder) {
          builder.addSimpleInsertion(receiver.end, '.copy()');
        });
        return;
      }
    }
    if (target is Expression) {
      await builder.addDartFileEdit(file, (builder) {
        builder.addSimpleInsertion(target.end, '.copy()');
      });
    }
  }
}

/// Quick fix that rewrites `a < b` on `NDArray<Uint64>` elements to
/// `uint64Compare(a, b) < 0`.
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
