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

import 'package:analysis_server_plugin/plugin.dart';
import 'package:analysis_server_plugin/registry.dart';
import 'package:analyzer/analysis_rule/rule_context.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/diagnostic/diagnostic.dart';
import 'package:analyzer/error/listener.dart';
// ignore: implementation_imports
import 'package:analyzer/src/analysis_rule/rule_context.dart';
// ignore: implementation_imports
import 'package:analyzer/src/lint/linter_visitor.dart';

import 'src/fixes/ndarray_fixes.dart';
import 'src/rules/advanced_memory_and_view_rules.dart';
import 'src/rules/api_and_performance_rules.dart';
import 'src/rules/scope_and_lifecycle_rules.dart';

export 'src/fixes/ndarray_fixes.dart';
export 'src/rules/advanced_memory_and_view_rules.dart';
export 'src/rules/api_and_performance_rules.dart';
export 'src/rules/scope_and_lifecycle_rules.dart';

/// Scope, view-lifecycle, disposal, and cross-isolate lifetime analysis rules.
List<AnalysisRule> createScopeAndLifecycleRules() => [
  UnescapedScopeReturnRule(),
  ViewLifecycleMisuseRule(),
  LoopReassignmentLeakRule(),
  IdentityCastDisposeRule(),
  SendableBorrowOutlivesScopeRule(),
];

/// API contract, DType safety, and performance analysis rules.
List<AnalysisRule> createApiAndPerformanceRules() => [
  EqualityOperatorRule(),
  Uint64SignedComparisonRule(),
  BroadcastViewAsOutRule(),
  HotLoopElementIndexingRule(),
];

/// Advanced memory-lifetime, view-aliasing, iterator, and symbolic rules.
List<AnalysisRule> createAdvancedMemoryAndViewRules() => [
  ZeroDimReductionIndexingRule(),
  NDIterCoordsAliasingOrMutationRule(),
  LostMutationOnCopyRule(),
  FromPointerDanglingArenaRule(),
  UnawaitedAsyncInScopeRule(),
  SymbolicLambdifyInLoopRule(),
];

/// All `scientific_dart_analysis_plugin` analysis rules.
List<AnalysisRule> createAllScientificDartRules() => [
  ...createScopeAndLifecycleRules(),
  ...createApiAndPerformanceRules(),
  ...createAdvancedMemoryAndViewRules(),
];

/// Analyzer server plugin registering rules and quick fixes for
/// `scientific_dart` (`package:ndarray`, `package:resource_scope`, etc.)
/// consumers.
final class ScientificDartAnalysisPlugin extends Plugin {
  @override
  String get name => 'scientific_dart_analysis_plugin';

  @override
  void register(PluginRegistry registry) {
    for (final rule in createAllScientificDartRules()) {
      registry.registerWarningRule(rule);
    }

    // Register IDE Quick Fixes.
    registry.registerFixForRule(
      EqualityOperatorRule.code,
      ReplaceWithEqualsFix.new,
    );
    registry.registerFixForRule(
      EqualityOperatorRule.code,
      ReplaceWithIdenticalFix.new,
    );
    registry.registerFixForRule(
      UnescapedScopeReturnRule.code,
      AddDetachToParentScopeFix.new,
    );
    registry.registerFixForRule(
      ViewLifecycleMisuseRule.code,
      AddCopyBeforeViewLifecycleFix.new,
    );
    registry.registerFixForRule(
      Uint64SignedComparisonRule.code,
      ReplaceWithUint64CompareFix.new,
    );
  }
}

/// Runs [rules] (defaulting to [createAllScientificDartRules]) against a
/// resolved compilation unit [unitResult] and returns all reported
/// [Diagnostic]s.
List<Diagnostic> runScientificDartLintsOnUnit(
  ResolvedUnitResult unitResult, {
  List<AnalysisRule>? rules,
}) {
  final activeRules = rules ?? createAllScientificDartRules();
  final recordingListener = RecordingDiagnosticListener();
  final reporter = DiagnosticReporter(
    recordingListener,
    unitResult.libraryElement.firstFragment.source,
  );

  final contextUnit = RuleContextUnit(
    file: unitResult.file,
    content: unitResult.content,
    diagnosticReporter: reporter,
    unit: unitResult.unit,
  );
  final ruleContext = RuleContextWithResolvedResults(
    [contextUnit],
    contextUnit,
    unitResult.typeProvider,
    unitResult.typeSystem,
    null,
  )..currentUnit = contextUnit;

  final visitorRegistry = RuleVisitorRegistryImpl(enableTiming: false);
  for (final rule in activeRules) {
    rule.reporter = reporter;
    rule.registerNodeProcessors(visitorRegistry, ruleContext);
  }

  final astVisitor = AnalysisRuleVisitor(
    visitorRegistry,
    shouldPropagateExceptions: true,
  );
  unitResult.unit.accept(astVisitor);
  astVisitor.afterLibrary();

  return recordingListener.diagnostics;
}
