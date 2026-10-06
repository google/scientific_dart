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

import 'dart:convert';
import 'dart:io';

/// Per-file branch and line coverage statistics parsed from an LCOV report.
final class FileCoverageStats {
  /// Relative path from `pkgs/ndarray/` (e.g. `lib/src/ndarray.dart`).
  final String relativePath;

  /// Total instrumentable branches (`BRF`).
  final int branchesFound;

  /// Total executed branches (`BRH`).
  final int branchesHit;

  /// Total instrumentable lines (`LF`).
  final int linesFound;

  /// Total executed lines (`LH`).
  final int linesHit;

  /// Creates coverage statistics for [relativePath].
  const FileCoverageStats({
    required this.relativePath,
    required this.branchesFound,
    required this.branchesHit,
    required this.linesFound,
    required this.linesHit,
  });

  /// Number of uncovered branches (`branchesFound - branchesHit`).
  int get missedBranches => branchesFound - branchesHit;

  /// Number of uncovered lines (`linesFound - linesHit`).
  int get missedLines => linesFound - linesHit;

  /// Branch coverage percentage in `[0.0, 100.0]`.
  double get branchCoveragePercent =>
      branchesFound == 0 ? 100.0 : (branchesHit * 100.0) / branchesFound;

  /// Line coverage percentage in `[0.0, 100.0]`.
  double get lineCoveragePercent =>
      linesFound == 0 ? 100.0 : (linesHit * 100.0) / linesFound;

  /// Serializes this record for `branch_coverage_baseline.json`.
  Map<String, Object> toJson() => {
    'missedBranches': missedBranches,
    'branchesFound': branchesFound,
    'branchesHit': branchesHit,
    'branchCoveragePercent': double.parse(
      branchCoveragePercent.toStringAsFixed(2),
    ),
    'missedLines': missedLines,
    'linesFound': linesFound,
    'linesHit': linesHit,
  };
}

/// Runs `dart test --branch-coverage`, formats LCOV coverage via `package:coverage`,
/// and enforces that no library file's missed branch count increases relative to
/// `tool/branch_coverage_baseline.json`.
///
/// Why ratcheting `missedBranches` (`BRF - BRH`) is used instead of global `%`:
/// - Deleting dead or duplicate tested code decreases both `BRH` and `BRF`, which
///   lowers global `%` even though zero untested branches were introduced.
/// - Ratcheting `missedBranches` per file never penalizes dead-code deletion or
///   simplification, while immediately catching any newly added untested branch.
///
/// Usage:
/// ```bash
/// dart run tool/check_branch_coverage.dart [--update-baseline] [--native] [test_files...]
/// ```
Future<void> main(List<String> args) async {
  var updateBaseline = false;
  var includeNative = false;
  String? lcovInputPath;
  final testTargets = <String>[];

  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (arg == '--update-baseline') {
      updateBaseline = true;
    } else if (arg == '--native') {
      includeNative = true;
    } else if (arg.startsWith('--lcov=')) {
      lcovInputPath = arg.substring('--lcov='.length);
    } else if (arg == '--help' || arg == '-h') {
      stdout.writeln(
        'Usage: dart run tool/check_branch_coverage.dart '
        '[--update-baseline] [--native] [--lcov=path/to/lcov.info] [test_paths...]',
      );
      return;
    } else if (arg.startsWith('-')) {
      stderr.writeln('Unknown flag: $arg');
      exitCode = 2;
      return;
    } else {
      testTargets.add(arg);
    }
  }

  final pkgRoot = _findPackageRoot();
  final baselineFile = File(
    '${pkgRoot.path}/tool/branch_coverage_baseline.json',
  );

  final String lcovContent;
  Directory? tempDir;
  try {
    if (lcovInputPath != null) {
      final f = File(lcovInputPath);
      if (!f.existsSync()) {
        stderr.writeln('LCOV file not found: $lcovInputPath');
        exitCode = 2;
        return;
      }
      lcovContent = f.readAsStringSync();
    } else {
      tempDir = Directory.systemTemp.createTempSync('ndarray_branch_cov_');
      final rawCovDir = Directory('${tempDir.path}/raw')..createSync();
      final lcovFile = File('${tempDir.path}/lcov.info');

      stdout.writeln(
        'Running `dart test --branch-coverage` in ${pkgRoot.path}...',
      );
      final env = <String, String>{
        ...Platform.environment,
        if (includeNative) 'NDARRAY_COVERAGE': '1',
      };
      final testRes = await Process.run(
        Platform.resolvedExecutable,
        [
          'test',
          '--coverage=${rawCovDir.path}',
          '--branch-coverage',
          ...testTargets,
        ],
        workingDirectory: pkgRoot.path,
        environment: env,
      );
      if (testRes.exitCode != 0) {
        stderr.writeln('`dart test` failed (exit ${testRes.exitCode}):');
        stderr.writeln(testRes.stdout);
        stderr.writeln(testRes.stderr);
        exitCode = 1;
        return;
      }

      stdout.writeln('Formatting LCOV branch coverage report...');
      final formatRes = await Process.run(Platform.resolvedExecutable, [
        'run',
        'coverage:format_coverage',
        '--lcov',
        '--check-ignore',
        '--workers=${Platform.numberOfProcessors}',
        '--in=${rawCovDir.path}',
        '--out=${lcovFile.path}',
        '--report-on=lib',
      ], workingDirectory: pkgRoot.path);
      if (formatRes.exitCode != 0) {
        stderr.writeln(
          '`coverage:format_coverage` failed (exit ${formatRes.exitCode}):\n'
          '${formatRes.stderr}',
        );
        exitCode = 1;
        return;
      }
      lcovContent = lcovFile.readAsStringSync();
    }
  } finally {
    if (tempDir != null && tempDir.existsSync()) {
      tempDir.deleteSync(recursive: true);
    }
  }

  final statsByFile = parseLcov(lcovContent, pkgRoot.path);
  var totalBranchesFound = 0;
  var totalBranchesHit = 0;
  var totalLinesFound = 0;
  var totalLinesHit = 0;
  for (final s in statsByFile.values) {
    totalBranchesFound += s.branchesFound;
    totalBranchesHit += s.branchesHit;
    totalLinesFound += s.linesFound;
    totalLinesHit += s.linesHit;
  }
  final totalMissedBranches = totalBranchesFound - totalBranchesHit;
  final globalBranchPct = totalBranchesFound == 0
      ? 100.0
      : (totalBranchesHit * 100.0) / totalBranchesFound;
  final globalLinePct = totalLinesFound == 0
      ? 100.0
      : (totalLinesHit * 100.0) / totalLinesFound;

  stdout.writeln(
    'Branch coverage: ${globalBranchPct.toStringAsFixed(2)}% '
    '($totalBranchesHit / $totalBranchesFound branches hit, '
    '$totalMissedBranches missed)',
  );
  stdout.writeln(
    'Line coverage:   ${globalLinePct.toStringAsFixed(2)}% '
    '($totalLinesHit / $totalLinesFound lines hit)',
  );

  if (updateBaseline || !baselineFile.existsSync()) {
    final sortedKeys = statsByFile.keys.toList()..sort();
    final filesJson = <String, Object>{
      for (final k in sortedKeys) k: statsByFile[k]!.toJson(),
    };
    final payload = <String, Object>{
      'summary': {
        'totalMissedBranches': totalMissedBranches,
        'totalBranchesFound': totalBranchesFound,
        'totalBranchesHit': totalBranchesHit,
        'branchCoveragePercent': double.parse(
          globalBranchPct.toStringAsFixed(2),
        ),
        'lineCoveragePercent': double.parse(globalLinePct.toStringAsFixed(2)),
      },
      'files': filesJson,
    };
    baselineFile.writeAsStringSync(
      '${const JsonEncoder.withIndent('  ').convert(payload)}\n',
    );
    stdout.writeln('Updated baseline at ${baselineFile.path}.');
    return;
  }

  final baselineJson =
      jsonDecode(baselineFile.readAsStringSync()) as Map<String, Object?>;
  final baselineFiles =
      (baselineJson['files'] as Map<String, Object?>?) ?? const {};

  final regressions = <String>[];
  final improvements = <String>[];

  for (final entry in statsByFile.entries) {
    final path = entry.key;
    final current = entry.value;
    final baseEntry = baselineFiles[path] as Map<String, Object?>?;
    if (baseEntry == null) {
      if (current.missedBranches > 0) {
        regressions.add(
          '$path (new file): has ${current.missedBranches} missed branches '
          '(${current.branchCoveragePercent.toStringAsFixed(2)}% branch coverage).',
        );
      }
      continue;
    }
    final baseMissed = (baseEntry['missedBranches'] as num).toInt();
    if (current.missedBranches > baseMissed) {
      regressions.add(
        '$path: missed branches increased from $baseMissed to ${current.missedBranches} '
        '(${current.branchCoveragePercent.toStringAsFixed(2)}% branch coverage).',
      );
    } else if (current.missedBranches < baseMissed) {
      improvements.add(
        '$path: missed branches decreased from $baseMissed to ${current.missedBranches} '
        '(+${baseMissed - current.missedBranches} newly covered branches).',
      );
    }
  }

  if (improvements.isNotEmpty) {
    stdout.writeln('\nCoverage improvements detected:');
    for (final imp in improvements) {
      stdout.writeln('  ✓ $imp');
    }
    stdout.writeln(
      'Run `dart run tool/check_branch_coverage.dart --update-baseline` to ratchet the baseline.',
    );
  }

  if (regressions.isNotEmpty) {
    stderr.writeln('\nBranch coverage regressions detected:');
    for (final reg in regressions) {
      stderr.writeln('  ✗ $reg');
    }
    exitCode = 1;
  } else {
    stdout.writeln('\n✓ Zero branch coverage regressions across all files.');
  }
}

/// Parses an LCOV content string into per-file [FileCoverageStats], excluding
/// generated FFI binding files (`ndarray_bindings.dart`, `ndarray_extensions_bindings.dart`).
Map<String, FileCoverageStats> parseLcov(String lcovContent, String pkgRoot) {
  final result = <String, FileCoverageStats>{};
  String? currentFile;
  var brFound = 0;
  var brHit = 0;
  var lnFound = 0;
  var lnHit = 0;

  void flushRecord() {
    final file = currentFile;
    if (file != null &&
        !file.endsWith('ndarray_bindings.dart') &&
        !file.endsWith('ndarray_extensions_bindings.dart')) {
      result[file] = FileCoverageStats(
        relativePath: file,
        branchesFound: brFound,
        branchesHit: brHit,
        linesFound: lnFound,
        linesHit: lnHit,
      );
    }
    currentFile = null;
    brFound = 0;
    brHit = 0;
    lnFound = 0;
    lnHit = 0;
  }

  for (final rawLine in lcovContent.split('\n')) {
    final line = rawLine.trim();
    if (line.startsWith('SF:')) {
      var sfPath = line.substring(3).replaceAll('\\', '/');
      final normalizedRoot = pkgRoot.replaceAll('\\', '/');
      if (sfPath.startsWith('$normalizedRoot/')) {
        sfPath = sfPath.substring(normalizedRoot.length + 1);
      } else {
        final libIdx = sfPath.indexOf('lib/');
        if (libIdx >= 0) {
          sfPath = sfPath.substring(libIdx);
        }
      }
      currentFile = sfPath;
    } else if (line.startsWith('BRDA:')) {
      // BRDA:<line>,<block>,<branch>,<taken>
      final parts = line.substring(5).split(',');
      if (parts.length >= 4) {
        brFound++;
        final taken = parts[3];
        if (taken != '-' && taken != '0') {
          brHit++;
        }
      }
    } else if (line.startsWith('DA:')) {
      // DA:<line>,<execution_count>
      final parts = line.substring(3).split(',');
      if (parts.length >= 2) {
        lnFound++;
        final count = int.tryParse(parts[1]) ?? 0;
        if (count > 0) {
          lnHit++;
        }
      }
    } else if (line == 'end_of_record') {
      flushRecord();
    }
  }
  return result;
}

Directory _findPackageRoot() {
  var dir = Directory.current;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/hook').existsSync()) {
      return dir;
    }
    final sub = Directory('${dir.path}/pkgs/ndarray');
    if (sub.existsSync()) return sub;
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError('Could not locate pkgs/ndarray directory.');
    }
    dir = parent;
  }
}
