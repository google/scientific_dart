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

void main() async {
  print(
    '============================================================================',
  );
  print('         ndarray AUTOMATED TEST COVERAGE INFRASTRUCTURE TOOL');
  print(
    '============================================================================',
  );

  final rawCoverageDir = Directory('coverage/raw');
  if (rawCoverageDir.existsSync()) {
    print('Clearing old raw coverage trace data maps...');
    rawCoverageDir.deleteSync(recursive: true);
  }

  print(
    '\nStep 1: Executing full unit test suite and collecting raw JSON V8 traces...',
  );
  final testProcess = await Process.run(Platform.executable, [
    'test',
    '--coverage=coverage/raw',
    '--branch-coverage',
  ]);

  if (testProcess.exitCode != 0) {
    stderr.writeln(
      'Error: dart test coverage execution failed!\n${testProcess.stdout}\n${testProcess.stderr}',
    );
    exit(testProcess.exitCode);
  }
  print('Unit tests suite executed perfectly with exit code 0.');

  print(
    '\nStep 2: Formatting raw JSON coverage maps into standardized LCOV format...',
  );
  final formatProcess = await Process.run(Platform.executable, [
    'run',
    'coverage:format_coverage',
    '--in=coverage/raw',
    '--out=coverage/lcov.info',
    '--report-on=lib',
    '--lcov',
  ]);

  if (formatProcess.exitCode != 0) {
    stderr.writeln(
      'Error: format_coverage execution failed!\n${formatProcess.stderr}',
    );
    exit(formatProcess.exitCode);
  }
  print(
    'Formatted LCOV report file successfully written to: coverage/lcov.info',
  );

  print(
    '\nStep 3: Parsing LCOV report stream data and building coverage dashboard...',
  );
  final lcovFile = File('coverage/lcov.info');
  if (!lcovFile.existsSync()) {
    stderr.writeln(
      'Error: coverage/lcov.info file not found after formatting pass!',
    );
    exit(1);
  }

  final lines = await lcovFile.readAsLines();

  var totalLinesFound = 0;
  var totalLinesHit = 0;
  var totalBranchesFound = 0;
  var totalBranchesHit = 0;

  String? currentFile;
  var currentFileLinesFound = 0;
  var currentFileLinesHit = 0;
  var currentFileBranchesFound = 0;
  var currentFileBranchesHit = 0;

  print(
    '\n----------------------------------------------------------------------------',
  );
  print(' FILE-BY-FILE COVERAGE METRICS');
  print(
    '----------------------------------------------------------------------------',
  );
  print(
    '${"FILE NAME".padRight(35)} | ${"LINE COV %".padRight(12)} | ${"BRANCH COV %".padRight(12)}',
  );
  print(
    '----------------------------------------------------------------------------',
  );

  for (final line in lines) {
    if (line.startsWith('SF:')) {
      final fullPath = line.substring(3);
      currentFile = fullPath.contains('lib/src/')
          ? fullPath.substring(fullPath.indexOf('lib/src/'))
          : fullPath.substring(fullPath.lastIndexOf('/') + 1);
      currentFileLinesFound = 0;
      currentFileLinesHit = 0;
      currentFileBranchesFound = 0;
      currentFileBranchesHit = 0;
    } else if (line.startsWith('LF:')) {
      currentFileLinesFound = int.parse(line.substring(3));
      totalLinesFound += currentFileLinesFound;
    } else if (line.startsWith('LH:')) {
      currentFileLinesHit = int.parse(line.substring(3));
      totalLinesHit += currentFileLinesHit;
    } else if (line.startsWith('BRDA:')) {
      currentFileBranchesFound++;
      totalBranchesFound++;
      final parts = line.split(',');
      final taken = int.parse(parts.last);
      if (taken > 0) {
        currentFileBranchesHit++;
        totalBranchesHit++;
      }
    } else if (line == 'end_of_record') {
      if (currentFile != null && currentFileLinesFound > 0) {
        final linePct = (currentFileLinesHit / currentFileLinesFound) * 100.0;
        final linePctStr = '${linePct.toStringAsFixed(1)}%';

        final branchPct = currentFileBranchesFound > 0
            ? (currentFileBranchesHit / currentFileBranchesFound) * 100.0
            : 0.0;
        final branchPctStr = currentFileBranchesFound > 0
            ? '${branchPct.toStringAsFixed(1)}%'
            : 'N/A';

        print(
          '${currentFile.padRight(35)} | ${linePctStr.padRight(12)} | ${branchPctStr.padRight(12)} | Line: $currentFileLinesHit/$currentFileLinesFound | Branch: $currentFileBranchesHit/$currentFileBranchesFound',
        );
      }
    }
  }

  print(
    '----------------------------------------------------------------------------',
  );
  if (totalLinesFound == 0) {
    print(
      'Error: No executable code lines discovered in lib/ directory reports!',
    );
  } else {
    final globalLinePct = (totalLinesHit / totalLinesFound) * 100.0;
    final globalBranchPct = totalBranchesFound > 0
        ? (totalBranchesHit / totalBranchesFound) * 100.0
        : 0.0;
    print('🏆 GLOBAL WORKSPACE COVERAGE SUMMARY:');
    print(
      '   GLOBAL LINE COVERAGE   : ${globalLinePct.toStringAsFixed(2)}% ($totalLinesHit / $totalLinesFound)',
    );
    print(
      '   GLOBAL BRANCH COVERAGE : ${globalBranchPct.toStringAsFixed(2)}% ($totalBranchesHit / $totalBranchesFound)',
    );
  }
  print(
    '============================================================================\n',
  );
}
