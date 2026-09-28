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

import 'dart:io';

import 'package:analysis_server_plugin/registry.dart';
import 'package:analysis_server_plugin/src/correction/fix_generators.dart';
import 'package:analyzer/analysis_rule/analysis_rule.dart';
import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/diagnostic/diagnostic.dart';
import 'package:analyzer/error/error.dart';
import 'package:analyzer/src/lint/config.dart';
import 'package:scientific_dart_analysis_plugin/scientific_dart_analysis_plugin.dart';
import 'package:test/test.dart';

Directory _findWorkspaceRoot() {
  var dir = Directory.current;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/pkgs/ndarray').existsSync()) {
      return dir;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError(
        'Could not locate scientific_dart_workspace root directory.',
      );
    }
    dir = parent;
  }
}

final class _FakePluginRegistry implements PluginRegistry {
  final List<AbstractAnalysisRule> warningRules = [];
  final List<AbstractAnalysisRule> lintRules = [];
  final Map<DiagnosticCode, List<ProducerGenerator>> fixes = {};
  final List<ProducerGenerator> assists = [];

  @override
  Iterable<AbstractAnalysisRule> enabled(Map<String, RuleConfig> ruleConfigs) =>
      [...warningRules, ...lintRules];

  @override
  void registerAssist(ProducerGenerator generator) {
    assists.add(generator);
  }

  @override
  void registerFixForRule(DiagnosticCode code, ProducerGenerator generator) {
    fixes.putIfAbsent(code, () => []).add(generator);
  }

  @override
  void registerLintRule(AbstractAnalysisRule rule) {
    lintRules.add(rule);
  }

  @override
  void registerWarningRule(AbstractAnalysisRule rule) {
    warningRules.add(rule);
  }
}

void main() {
  late Directory workspaceRoot;
  late Directory scratchDir;
  late AnalysisContextCollection collection;

  setUpAll(() {
    workspaceRoot = _findWorkspaceRoot();
    scratchDir = Directory(
      '${workspaceRoot.path}/pkgs/scientific_dart_analysis_plugin/test/_fixtures',
    )..createSync(recursive: true);
    collection = AnalysisContextCollection(includedPaths: [scratchDir.path]);
  });

  tearDownAll(() {
    if (scratchDir.existsSync()) {
      scratchDir.deleteSync(recursive: true);
    }
  });

  var counter = 0;
  Future<List<Diagnostic>> analyzeCode(
    String source, {
    List<AnalysisRule>? rules,
  }) async {
    final file = File('${scratchDir.path}/case_${counter++}.dart');
    file.writeAsStringSync(source);
    final context = collection.contextFor(file.path);
    context.changeFile(file.path);
    await context.applyPendingFileChanges();
    final result = await context.currentSession.getResolvedUnit(file.path);
    if (result is! ResolvedUnitResult) {
      fail('Expected ResolvedUnitResult, got $result');
    }
    final compileErrors = result.diagnostics.where(
      (d) => d.diagnosticCode.type == DiagnosticType.COMPILE_TIME_ERROR,
    );
    expect(
      compileErrors,
      isEmpty,
      reason: 'Fixture had compile errors:\n${compileErrors.join('\n')}',
    );
    return runScientificDartLintsOnUnit(result, rules: rules);
  }

  group('Plugin Registration', () {
    test(
      'ScientificDartAnalysisPlugin registers all rules and quick fixes',
      () {
        final registry = _FakePluginRegistry();
        final plugin = ScientificDartAnalysisPlugin();
        plugin.register(registry);

        final registeredNames = registry.warningRules
            .map((r) => r.name)
            .toSet();
        expect(
          registeredNames,
          equals({
            'ndarray_unescaped_scope_return',
            'ndarray_view_lifecycle_misuse',
            'ndarray_loop_reassignment_leak',
            'ndarray_identity_cast_dispose',
            'ndarray_sendable_borrow_outlives_scope',
            'ndarray_equality_operator',
            'ndarray_uint64_signed_comparison',
            'ndarray_broadcast_view_as_out',
            'ndarray_hot_loop_element_indexing',
            'ndarray_0d_reduction_indexing',
            'nditer_coords_aliasing_or_mutation',
            'ndarray_lost_mutation_on_copy',
            'ndarray_from_pointer_dangling_arena',
            'scoped_resource_unawaited_in_scope',
            'symbolic_lambdify_in_loop',
          }),
        );
        expect(registry.fixes, isNotEmpty);
      },
    );
  });

  void expectOnly(List<Diagnostic> diagnostics, String code, int count) {
    expect(
      diagnostics.map((d) => d.diagnosticCode.lowerCaseName),
      everyElement(code),
    );
    expect(
      diagnostics,
      hasLength(count),
      reason: diagnostics.map((d) => '${d.offset}: ${d.message}').join('\n'),
    );
  }

  group('Scope & Lifecycle Rules', () {
    test(
      'ndarray_unescaped_scope_return catches owned returns and local views, '
      'allows detached returns and views of outer arrays',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badReturn() {
  return NDArray.scope(() {
    final a = NDArray.zeros([4], DType.float64);
    return sin(a); // VIOLATION 1: unescaped fresh array
  });
}

NDArray<Float64> badLocalReturn() {
  return NDArray.scope(() {
    final a = NDArray.ones([4], DType.float64);
    return a; // VIOLATION 2: unescaped local variable
  });
}

NDArray<Float64> badLocalView() {
  return NDArray.scope(() {
    final a = NDArray.ones([4], DType.float64);
    return a.slice([Slice(start: 0, stop: 2)]); // VIOLATION 3: view of scope-owned array
  });
}

NDArray<Float64> goodDetachedReturn() {
  return NDArray.scope(() {
    final a = NDArray.zeros([4], DType.float64);
    return sin(a).detachToParentScope(); // OK
  });
}

NDArray<Float64> goodReturningHelper() {
  return NDArray.returning(() {
    final a = NDArray.zeros([4], DType.float64);
    return sin(a); // OK: NDArray.returning detaches automatically
  });
}

NDArray<Float64> goodOuterParamReturn(NDArray<Float64> out) {
  return NDArray.scope(() {
    final temp = NDArray.ones([4], DType.float64);
    add(temp, temp, out: out);
    return out; // OK: `out` was declared outside the scope
  });
}

NDArray<Float64> goodOuterView(NDArray<Float64> outer) {
  return NDArray.scope(() {
    return outer.slice([Slice(start: 1)]); // OK: views are untracked
  });
}

NDArray<Float64> goodOuterViewViaLocal(NDArray<Float64> outer) {
  return NDArray.scope(() {
    final v = outer.transpose();
    return v; // OK: root buffer belongs to the caller
  });
}
''',
          rules: [UnescapedScopeReturnRule()],
        );

        expectOnly(diagnostics, 'ndarray_unescaped_scope_return', 3);
      },
    );

    test(
      'ndarray_view_lifecycle_misuse catches detaching views and returning '
      'views from NDArray.returning; allows dispose and maybe-view ops',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badReturningSlice() {
  return NDArray.returning(() {
    final a = NDArray.zeros([4, 4], DType.float64);
    return a.slice([Slice(start: 0, stop: 2)]); // VIOLATION 1
  });
}

NDArray<Float64> badDetachTranspose() {
  return NDArray.scope(() {
    final a = NDArray.ones([3, 3], DType.float64);
    final t = a.transpose();
    return t.detachToParentScope(); // VIOLATION 2
  });
}

void goodDisposeReshape(NDArray<Float64> a) {
  // OK: reshape copies for non-contiguous input, so dispose may be needed.
  a.reshape([2, 3]).dispose();
}

NDArray<Float64> goodCopyBeforeReturn() {
  return NDArray.returning(() {
    final a = NDArray.zeros([4, 4], DType.float64);
    return a.slice([Slice(start: 0, stop: 2)]).copy(); // OK
  });
}
''',
          rules: [ViewLifecycleMisuseRule()],
        );

        expectOnly(diagnostics, 'ndarray_view_lifecycle_misuse', 2);
      },
    );

    test(
      'ndarray_loop_reassignment_leak flags loop reassignment without out: or dispose()',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void leakLoop(NDArray<Float64> delta) {
  var x = NDArray.zeros([10], DType.float64);
  for (var i = 0; i < 10; i++) {
    x = add(x, delta); // VIOLATION: previous `x` accumulates every iteration
  }
  x.dispose();
}

void safeInPlaceLoop(NDArray<Float64> delta) {
  final x = NDArray.zeros([10], DType.float64);
  for (var i = 0; i < 10; i++) {
    add(x, delta, out: x); // OK
  }
  x.dispose();
}

void safeDisposedLoop(NDArray<Float64> delta) {
  var x = NDArray.zeros([10], DType.float64);
  for (var i = 0; i < 10; i++) {
    final prev = x;
    x = add(prev, delta);
    prev.dispose(); // OK: previous buffer explicitly disposed
  }
  x.dispose();
}
''',
          rules: [LoopReassignmentLeakRule()],
        );

        expectOnly(diagnostics, 'ndarray_loop_reassignment_leak', 1);
      },
    );

    test(
      'ndarray_identity_cast_dispose flags unguarded .dispose() on astype(copy: false)',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void badCastDispose(NDArray<Float64> a) {
  final casted = a.astype(DType.float64, copy: false);
  casted.dispose(); // VIOLATION: disposes `a` when dtype already matches!
}

void goodGuardedCastDispose(NDArray<Float64> a) {
  final casted = a.astype(DType.float64, copy: false);
  if (!identical(casted, a)) {
    casted.dispose(); // OK: guarded by !identical
  }
}
''',
          rules: [IdentityCastDisposeRule()],
        );

        expectOnly(diagnostics, 'ndarray_identity_cast_dispose', 1);
      },
    );

    test(
      'ndarray_sendable_borrow_outlives_scope flags toSendableBorrow in scopes '
      'that do not await',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'dart:isolate';
import 'package:ndarray/ndarray.dart';

void badSyncScopeBorrow() {
  NDArray.scope(() {
    final a = NDArray.ones([10], DType.float64);
    final borrowed = a.toSendableBorrow(); // VIOLATION: scope exits immediately
    Isolate.run(() => borrowed.materializeView().size);
  });
}

Future<int> goodAsyncAwaitedBorrow() async {
  return await NDArray.scope(() async {
    final a = NDArray.ones([10], DType.float64);
    final borrowed = a.toSendableBorrow(); // OK: scope awaits Isolate.run
    return await Isolate.run(() => borrowed.materializeView().size);
  });
}
''',
          rules: [SendableBorrowOutlivesScopeRule()],
        );

        expectOnly(diagnostics, 'ndarray_sendable_borrow_outlives_scope', 1);
      },
    );
  });

  group('API & Performance Rules', () {
    test(
      'ndarray_equality_operator flags == and != between NDArrays, allows null and .equals()',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

bool checkArrays(NDArray<Float64> a, NDArray<Float64> b, NDArray<Float64>? maybeNull) {
  final badEq = a == b; // VIOLATION 1
  final badNeq = a != b; // VIOLATION 2
  final okNull = maybeNull == null; // OK
  final okEquals = a.equals(b); // OK
  final okIdentical = identical(a, b); // OK
  return badEq && badNeq && okNull && okEquals && okIdentical;
}
''',
          rules: [EqualityOperatorRule()],
        );

        expectOnly(diagnostics, 'ndarray_equality_operator', 2);
      },
    );

    test(
      'ndarray_uint64_signed_comparison flags <, <=, >, >= on NDArray<Uint64> elements',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

bool checkUint64(NDArray<Uint64> u64, NDArray<Int64> i64) {
  final badIndexCmp = (u64[[0]] as int) > 0; // VIOLATION 1
  final elem = u64.scalar;
  final badScalarCmp = elem < 100; // VIOLATION 2
  final okUint64Compare = uint64Compare(u64.scalar, 0) > 0; // OK
  final okInt64Cmp = i64.scalar > 0; // OK: Int64 is signed
  return badIndexCmp && badScalarCmp && okUint64Compare && okInt64Cmp;
}
''',
          rules: [Uint64SignedComparisonRule()],
        );

        expectOnly(diagnostics, 'ndarray_uint64_signed_comparison', 2);
      },
    );

    test(
      'ndarray_broadcast_view_as_out flags broadcastTo passed as out:',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void testBroadcastOut(NDArray<Float64> a) {
  final bcast = broadcastTo(NDArray.zeros([1], DType.float64), [4]);
  sin(a, out: bcast); // VIOLATION: read-only broadcast view as out
}
''',
          rules: [BroadcastViewAsOutRule()],
        );

        expectOnly(diagnostics, 'ndarray_broadcast_view_as_out', 1);
      },
    );

    test(
      'ndarray_hot_loop_element_indexing flags nested-loop reads only',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

double sumSlow(NDArray<Float64> m) {
  final dump = m.toList(); // OK: not flagged any more
  var total = 0.0;
  for (var i = 0; i < 3; i++) {
    for (var j = 0; j < 3; j++) {
      total += m[[i, j]] as double; // VIOLATION: nested loop element read
      m[[i, j]] = 0.0; // OK: writes are not flagged
    }
  }
  return total + dump.length;
}
''',
          rules: [HotLoopElementIndexingRule()],
        );

        expectOnly(diagnostics, 'ndarray_hot_loop_element_indexing', 1);
      },
    );
  });

  group('Memory, Copy/View, Iterator & Symbolic Rules', () {
    test(
      'ndarray_0d_reduction_indexing flags non-empty indexing of axis-less reductions',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

double bad0d(NDArray<Float64> a) {
  final r = sum(a);
  final x = r[[0]] as double; // VIOLATION: rank-0 array
  final okEmpty = r[[]] as double; // OK
  final okScalar = max(a).scalar as double; // OK: no longer flagged
  final k = sum(a, keepdims: true);
  final okKeepdims = k[[0]] as double; // OK: keepdims preserves rank
  final s = sum(a, axis: 0);
  final okAxis = s[[0]] as double; // OK: axis given
  return x + okEmpty + okScalar + okKeepdims + okAxis;
}
''',
          rules: [ZeroDimReductionIndexingRule()],
        );

        expectOnly(diagnostics, 'ndarray_0d_reduction_indexing', 1);
      },
    );

    test(
      'nditer_coords_aliasing_or_mutation catches storing or mutating NDIter.coords directly',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void badIterCoords(NDArray<Float64> a) {
  final iter = NDIter(a);
  final saved = <List<int>>[];
  while (iter.moveNext()) {
    saved.add(iter.coords); // VIOLATION 1: aliasing mutable coords list
    iter.coords[0] = 99; // VIOLATION 2: mutating iterator coords in-place
    saved.add(List.of(iter.coords)); // OK: defensive copy
  }
}
''',
          rules: [NDIterCoordsAliasingOrMutationRule()],
        );

        expectOnly(diagnostics, 'nditer_coords_aliasing_or_mutation', 2);
      },
    );

    test(
      'ndarray_lost_mutation_on_copy catches mutating temporary copies',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void badCopyMutation(NDArray<Float64> a) {
  a.astype(DType.float32)[[0]] = 1.0; // VIOLATION 1: index assignment on astype copy
  a.flatten().fill(0.0); // VIOLATION 2: .fill() on flatten() copy
  a.slice([Slice(start: 0, stop: 2)]).fill(0.0); // OK: slice() returns a view
}
''',
          rules: [LostMutationOnCopyRule()],
        );

        expectOnly(diagnostics, 'ndarray_lost_mutation_on_copy', 2);
      },
    );

    test(
      'ndarray_from_pointer_dangling_arena catches NDArray.fromPointer escaping ScratchArena',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'dart:ffi' as ffi;
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badArenaReturn() {
  final marker = ScratchArena.marker;
  try {
    final ptr = ScratchArena.allocate<ffi.Double>(32);
    return NDArray.fromPointer(ptr.cast(), [4], DType.float64); // VIOLATION
  } finally {
    ScratchArena.reset(marker);
  }
}

NDArray<Float64> goodArenaCopyReturn() {
  final marker = ScratchArena.marker;
  try {
    final ptr = ScratchArena.allocate<ffi.Double>(32);
    return NDArray.fromPointer(ptr.cast(), [4], DType.float64).copy(); // OK
  } finally {
    ScratchArena.reset(marker);
  }
}
''',
          rules: [FromPointerDanglingArenaRule()],
        );

        expectOnly(diagnostics, 'ndarray_from_pointer_dangling_arena', 1);
      },
    );

    test(
      'scoped_resource_unawaited_in_scope catches unawaited Future statements in NDArray.scope',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'dart:async';
import 'package:ndarray/ndarray.dart';

Future<void> helperAsync(NDArray<Float64> a) async {}

void badScopeAsync() {
  NDArray.scope(() {
    final a = NDArray.zeros([4], DType.float64);
    helperAsync(a); // VIOLATION 1
    unawaited(helperAsync(a)); // VIOLATION 2
  });
}

Future<void> goodScopeAsync() async {
  await NDArray.scope(() async {
    final a = NDArray.zeros([4], DType.float64);
    await helperAsync(a); // OK
  });
}
''',
          rules: [UnawaitedAsyncInScopeRule()],
        );

        expectOnly(diagnostics, 'scoped_resource_unawaited_in_scope', 2);
      },
    );

    test(
      'symbolic_lambdify_in_loop flags loop-invariant lambdify only',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:symbolic_dart/symbolic_dart.dart';

double badSymbolicLoop(Expr expr, Expr x) {
  var sum = 0.0;
  for (var i = 0; i < 10; i++) {
    final fn = expr.lambdify([x]); // VIOLATION: recompiled every iteration
    sum += fn.callScalar([i.toDouble()]);
    final subbed = expr.subs({x: Expr.real(i.toDouble())}); // OK: subs varies
    subbed.dispose();
  }
  return sum;
}

double goodReassignedReceiver(Expr expr, Expr x) {
  var current = expr;
  var sum = 0.0;
  for (var i = 0; i < 3; i++) {
    current = current.subs({x: Expr.real(i.toDouble())});
    sum += current.lambdify([x]).callScalar([0.0]); // OK: not loop-invariant
  }
  return sum;
}

double goodSymbolicHoisted(Expr expr, Expr x) {
  final fn = expr.lambdify([x]); // OK: hoisted outside loop
  var sum = 0.0;
  for (var i = 0; i < 10; i++) {
    sum += fn.callScalar([i.toDouble()]);
  }
  return sum;
}
''',
          rules: [SymbolicLambdifyInLoopRule()],
        );

        expectOnly(diagnostics, 'symbolic_lambdify_in_loop', 1);
      },
    );
  });
}
