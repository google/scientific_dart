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

/// Static-type probe tests for `package:ndarray` (R3b).
///
/// Verifies both:
/// 1. Exact inferred static types via `package:analyzer` (`getDisplayString()`
///    string equality) on `test/meta/fixtures/static_type_probes_fixture.dart`,
///    and via the CFE runtime `staticTypeOf<T>(T Function() f) => T` witness.
/// 2. Expected analyzer error diagnostic codes on negative probes loaded via
///    `OverlayResourceProvider` from `test/meta/fixtures/negative_probes.dart.txt`.
library;

import 'dart:io';

import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/diagnostic/diagnostic.dart';
import 'package:analyzer/file_system/overlay_file_system.dart';
import 'package:analyzer/file_system/physical_file_system.dart';
import 'package:analyzer/source/line_info.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

import 'fixtures/static_type_probes_fixture.dart' as fixture;

/// Captures the exact static return type `T` inferred by the Dart compiler for
/// a closure `f` without executing `f`.
Type staticTypeOf<T>(T Function() f) => T;

/// Returns the runtime representation of the static type `T`.
Type typeOf<T>() => T;

Directory _findPackageRoot() {
  var dir = Directory.current;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/hook').existsSync()) {
      return dir;
    }
    final sub = Directory('${dir.path}/pkgs/ndarray');
    if (sub.existsSync()) {
      return sub;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError('Could not locate pkgs/ndarray root directory.');
    }
    dir = parent;
  }
}

String _native(String path) => Uri.file(path).toFilePath();

void main() {
  final pkgRoot = _findPackageRoot();
  final resolvedPkgRoot = _native(pkgRoot.resolveSymbolicLinksSync());

  group('static type probes — positive (analyzer + CFE staticTypeOf)', () {
    test(
      'positive fixture resolves with zero errors and exact expected types',
      () async {
        final fixturePath = _native(
          '$resolvedPkgRoot/test/meta/fixtures/static_type_probes_fixture.dart',
        );
        final sourceLines = File(fixturePath).readAsStringSync().split('\n');
        final expectedByLine = <int, String>{};
        final expectRegex = RegExp(r'//\s*expect:\s*(.+)$');
        for (var i = 0; i < sourceLines.length; i++) {
          final match = expectRegex.firstMatch(sourceLines[i]);
          if (match != null) {
            expectedByLine[i + 1] = match.group(1)!.trim();
          }
        }

        final collection = AnalysisContextCollection(
          includedPaths: [fixturePath],
          resourceProvider: PhysicalResourceProvider.INSTANCE,
        );
        final session = collection.contextFor(fixturePath).currentSession;
        final result = await session.getResolvedUnit(fixturePath);
        expect(result, isA<ResolvedUnitResult>());
        final unitResult = result as ResolvedUnitResult;

        final errors = unitResult.diagnostics
            .where((d) => d.severity == Severity.error)
            .map(
              (d) =>
                  'L${unitResult.lineInfo.getLocation(d.offset).lineNumber} '
                  '${d.diagnosticCode.lowerCaseName}: ${d.message}',
            )
            .toList();
        expect(errors, isEmpty, reason: errors.join('\n'));

        final visitor = _ProbeTypeVisitor(unitResult.lineInfo, expectedByLine);
        unitResult.unit.accept(visitor);

        expect(visitor.missingExpectations, isEmpty);
        expect(
          visitor.mismatches,
          isEmpty,
          reason: visitor.mismatches.join('\n'),
        );
        expect(visitor.checkedCount, greaterThanOrEqualTo(100));
      },
    );

    test(
      'CFE runtime staticTypeOf matches exact static types for acceptance probes',
      () {
        // Transcendental math (InexactOf<R>)
        expect(
          staticTypeOf(() => sin(fixture.i32)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => sin(fixture.f16)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => sin(fixture.c64)),
          equals(typeOf<NDArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => sin(fixture.b)),
          equals(typeOf<NDArray<Float64>>()),
        );

        // True division (DivideOf<R>)
        expect(
          staticTypeOf(() => fixture.f16 / fixture.f16),
          equals(typeOf<NDArray<Float16>>()),
        );
        expect(
          staticTypeOf(() => fixture.i32 / fixture.i32),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => divide(fixture.b, fixture.b)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => fixture.c64 / fixture.c64),
          equals(typeOf<NDArray<Complex64>>()),
        );

        // Accumulating reductions (AccumulatorOf<R>)
        expect(
          staticTypeOf(() => sum(fixture.u8)),
          equals(typeOf<NDArray<Uint64>>()),
        );
        expect(
          staticTypeOf(() => sum(fixture.b)),
          equals(typeOf<NDArray<Int64>>()),
        );
        expect(
          staticTypeOf(() => sum(fixture.f16)),
          equals(typeOf<NDArray<Float16>>()),
        );

        // Real / magnitude projections (RealOf<R>)
        expect(
          staticTypeOf(() => abs(fixture.c128)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => real(fixture.c64)),
          equals(typeOf<NDArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => imag(fixture.c64)),
          equals(typeOf<NDArray<Float32>>()),
        );

        // Element extraction (ElementOf<E>)
        expect(
          staticTypeOf(() => fixture.f16.toList()),
          equals(typeOf<List<double>>()),
        );
        expect(
          staticTypeOf(() => fixture.i32.toList()),
          equals(typeOf<List<int>>()),
        );
        expect(
          staticTypeOf(() => fixture.b.toList()),
          equals(typeOf<List<bool>>()),
        );
        expect(
          staticTypeOf(() => fixture.c64.toList()),
          equals(typeOf<List<Complex>>()),
        );

        // LUB over integer / bitwise tags
        expect(
          staticTypeOf(() => [fixture.i64, fixture.i32]),
          equals(typeOf<List<NDArray<IntegerDType>>>()),
        );
        expect(
          staticTypeOf(() => [fixture.i64, fixture.i32].first.toList()),
          equals(typeOf<List<int>>()),
        );
        expect(
          staticTypeOf(() => sin([fixture.i64, fixture.i32].first)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => [fixture.i32, fixture.b]),
          equals(typeOf<List<NDArray<BitwiseDType>>>()),
        );

        // AnySpec escape hatch
        expect(
          staticTypeOf(() => sin(fixture.anySpec)),
          equals(typeOf<NDArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => fixture.anySpec / fixture.anySpec),
          equals(typeOf<NDArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => sum(fixture.anySpec)),
          equals(typeOf<NDArray<DTypeTag>>()),
        );

        // Same-dtype & capability-bounded operations
        expect(
          staticTypeOf(() => add(fixture.f16, fixture.f16)),
          equals(typeOf<NDArray<Float16>>()),
        );
        expect(
          staticTypeOf(() => bitwiseAnd(fixture.i32, fixture.i32)),
          equals(typeOf<NDArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => fixture.i32 << fixture.i32),
          equals(typeOf<NDArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => fixture.b & fixture.b),
          equals(typeOf<NDArray<Boolean>>()),
        );

        // Multi-projection & additional projections (mean, rint, fft, eigh, slogdet, svd)
        expect(
          staticTypeOf(() => mean(fixture.i32)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => mean(fixture.c64)),
          equals(typeOf<NDArray<Complex128>>()),
        );
        expect(
          staticTypeOf(() => rint(fixture.c64)),
          equals(typeOf<NDArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => fft(fixture.f32)),
          equals(typeOf<NDArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => fft(fixture.i32)),
          equals(typeOf<NDArray<Complex128>>()),
        );
        expect(
          staticTypeOf(() => eigh(fixture.c64)),
          equals(
            typeOf<
              ({NDArray<Float32> eigenvalues, NDArray<Complex64> eigenvectors})
            >(),
          ),
        );
        expect(
          staticTypeOf(() => slogdet(fixture.c64)),
          equals(
            typeOf<({NDArray<Float32> logabsdet, NDArray<Complex64> sign})>(),
          ),
        );
        expect(
          staticTypeOf(() => svd(fixture.f32)),
          equals(
            typeOf<
              ({NDArray<Float32> u, NDArray<Float32> s, NDArray<Float32> vh})
            >(),
          ),
        );

        // Generic helpers bounded by IntegerDType / BitwiseDType
        expect(
          staticTypeOf(() => fixture.genericIntsOf(fixture.i32)),
          equals(typeOf<List<int>>()),
        );
        expect(
          staticTypeOf(() => fixture.genericSinOfInt(fixture.i32)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => fixture.genericSinOfBitwise(fixture.b)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => fixture.genericDivOfInt(fixture.i32)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => fixture.genericDivOfBitwise(fixture.b)),
          equals(typeOf<NDArray<Float64>>()),
        );

        // Same-dtype binary & multi-array operations (SelfOf bounds)
        expect(
          staticTypeOf(() => matmul(fixture.f32, fixture.f32)),
          equals(typeOf<NDArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => atan2(fixture.i32, fixture.i32)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => floatPower(fixture.f32, fixture.f32)),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => gcd(fixture.i64, fixture.i64)),
          equals(typeOf<NDArray<Int64>>()),
        );
        expect(
          staticTypeOf(() => concatenate([fixture.f64, fixture.f64])),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => add(fixture.anySpec, fixture.f32)),
          equals(typeOf<NDArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(
            () => fixture.genericAddSelfOf(fixture.i32, fixture.i32),
          ),
          equals(typeOf<NDArray<Int32>>()),
        );
        expect(
          staticTypeOf(
            () => fixture.genericAtan2SelfOf(fixture.i16, fixture.i16),
          ),
          equals(typeOf<NDArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => [fixture.f32, fixture.i32]),
          equals(typeOf<List<NDArray<DoublePrecisionOf<Float64>>>>()),
        );
      },
    );
  });

  group('static type probes — negative (OverlayResourceProvider)', () {
    test(
      'negative fixture produces the exact expected ERROR diagnostic codes per line',
      () async {
        final txtPath = _native(
          '$resolvedPkgRoot/test/meta/fixtures/negative_probes.dart.txt',
        );
        final virtualPath = _native(
          '$resolvedPkgRoot/test/meta/fixtures/_negative_probes_overlay.dart',
        );
        final content = File(txtPath).readAsStringSync();
        final lines = content.split('\n');

        final expectedCodesByLine = <int, Set<String>>{};
        final errorCommentRegex = RegExp(r'//\s*error:\s*(.+)$');
        for (var i = 0; i < lines.length; i++) {
          final match = errorCommentRegex.firstMatch(lines[i]);
          if (match != null) {
            final codes = match
                .group(1)!
                .split(',')
                .map((s) => s.trim().toLowerCase())
                .where((s) => s.isNotEmpty)
                .toSet();
            expectedCodesByLine[i + 1] = codes;
          }
        }
        expect(expectedCodesByLine.length, greaterThanOrEqualTo(35));

        final overlay = OverlayResourceProvider(
          PhysicalResourceProvider.INSTANCE,
        );
        overlay.setOverlay(virtualPath, content: content, modificationStamp: 0);

        final collection = AnalysisContextCollection(
          includedPaths: [virtualPath],
          resourceProvider: overlay,
        );
        final session = collection.contextFor(virtualPath).currentSession;
        final result = await session.getResolvedUnit(virtualPath);
        expect(result, isA<ResolvedUnitResult>());
        final unitResult = result as ResolvedUnitResult;

        final actualCodesByLine = <int, Set<String>>{};
        for (final d in unitResult.diagnostics) {
          if (d.severity != Severity.error) continue;
          final line = unitResult.lineInfo.getLocation(d.offset).lineNumber;
          actualCodesByLine
              .putIfAbsent(line, () => <String>{})
              .add(d.diagnosticCode.lowerCaseName);
        }

        final mismatches = <String>[];
        for (final entry in expectedCodesByLine.entries) {
          final line = entry.key;
          final expected = entry.value;
          final actual = actualCodesByLine[line] ?? const <String>{};
          if (!actual.containsAll(expected) || !expected.containsAll(actual)) {
            mismatches.add(
              'L$line (${lines[line - 1].trim()}): '
              'expected errors $expected, got $actual',
            );
          }
        }
        for (final entry in actualCodesByLine.entries) {
          if (!expectedCodesByLine.containsKey(entry.key)) {
            mismatches.add(
              'L${entry.key} (${lines[entry.key - 1].trim()}): '
              'unexpected errors ${entry.value}',
            );
          }
        }

        expect(mismatches, isEmpty, reason: mismatches.join('\n'));
      },
    );
  });
}

final class _ProbeTypeVisitor extends RecursiveAstVisitor<void> {
  _ProbeTypeVisitor(this.lineInfo, this.expectedByLine);

  final LineInfo lineInfo;
  final Map<int, String> expectedByLine;
  final List<String> mismatches = [];
  final List<String> missingExpectations = [];
  int checkedCount = 0;

  @override
  void visitVariableDeclaration(VariableDeclaration node) {
    final name = node.name.lexeme;
    if (name.startsWith('t_')) {
      final declLine = lineInfo.getLocation(node.offset).lineNumber;
      final expected = expectedByLine[declLine] ?? expectedByLine[declLine - 1];
      if (expected == null) {
        missingExpectations.add('L$declLine $name: missing // expect: comment');
      } else {
        checkedCount++;
        final actual = node.declaredFragment?.element.type.getDisplayString();
        if (actual != expected) {
          mismatches.add(
            'L$declLine $name: expected `$expected`, got `$actual`',
          );
        }
      }
    }
    super.visitVariableDeclaration(node);
  }
}
