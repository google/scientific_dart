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

import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/results.dart';
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

void main() {
  test(
    'Scan workspace consumer packages (guitar_tuner, ndarray_ma, symbolic_dart) with scientific_dart_analysis_plugin',
    () async {
      final root = _findWorkspaceRoot();
      final targetDirs = [
        Directory('${root.path}/pkgs/guitar_tuner/lib'),
        Directory('${root.path}/pkgs/ndarray_ma/lib'),
        Directory('${root.path}/pkgs/symbolic_dart/lib'),
      ].where((d) => d.existsSync()).toList();

      final collection = AnalysisContextCollection(
        includedPaths: targetDirs.map((d) => d.path).toList(),
      );

      final findingsByRule = <String, List<String>>{};
      for (final dir in targetDirs) {
        final dartFiles =
            dir
                .listSync(recursive: true)
                .whereType<File>()
                .where((f) => f.path.endsWith('.dart'))
                .toList()
              ..sort((a, b) => a.path.compareTo(b.path));

        for (final file in dartFiles) {
          final context = collection.contextFor(file.path);
          final result = await context.currentSession.getResolvedUnit(
            file.path,
          );
          if (result is! ResolvedUnitResult) continue;
          final diagnostics = runScientificDartLintsOnUnit(result);
          for (final d in diagnostics) {
            final loc = result.lineInfo.getLocation(d.offset);
            final relPath = file.path.substring(root.path.length + 1);
            findingsByRule
                .putIfAbsent(d.diagnosticCode.lowerCaseName, () => [])
                .add(
                  '$relPath:${loc.lineNumber}:${loc.columnNumber} — ${d.message}',
                );
          }
        }
      }

      for (final entry in findingsByRule.entries) {
        // Print summary of consumer findings during test runs for visibility.
        print('Rule ${entry.key} (${entry.value.length} findings):');
        for (final item in entry.value.take(5)) {
          print('  $item');
        }
      }

      // Verify that none of the workspace consumer packages have unescaped
      // scope returns, view lifecycle StateErrors, or equality operator bugs.
      expect(
        findingsByRule['ndarray_unescaped_scope_return'] ?? const [],
        isEmpty,
      );
      expect(
        findingsByRule['ndarray_view_lifecycle_misuse'] ?? const [],
        isEmpty,
      );
      expect(findingsByRule['ndarray_equality_operator'] ?? const [], isEmpty);
      expect(
        findingsByRule['ndarray_broadcast_view_as_out'] ?? const [],
        isEmpty,
      );
      expect(
        findingsByRule['ndarray_loop_reassignment_leak'] ?? const [],
        isEmpty,
      );
      expect(findingsByRule['symbolic_lambdify_in_loop'] ?? const [], isEmpty);
    },
  );
}
