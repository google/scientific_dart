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

// Checks whether the pinned prebuilt binary hashes in
// `pkgs/<pkg>/lib/src/hook_helpers/hashes.dart` are in sync with the current
// `hook/` native source files.
//
// Emits GitHub Actions `::warning::` annotations when run in CI so PRs and
// commits that modify `hook/` C/C++ files immediately alert the author that
// prebuilt release binaries need to be rebuilt before publishing.
// Pass `--strict` to exit with code 1 if any package's prebuilt binaries are stale.

import 'dart:io';

import 'package:ndarray/src/hook_helpers/hashes.dart' as ndarray_hashes;
import 'package:openblas/src/hook_helpers/hashes.dart' as openblas_hashes;
import 'package:pocketfft/src/hook_helpers/hashes.dart' as pocketfft_hashes;

void main(List<String> args) {
  final strict = args.contains('--strict');
  final isGitHubActions = Platform.environment['GITHUB_ACTIONS'] == 'true';

  final checks =
      <
        ({
          String packageName,
          String version,
          String pinnedSourceHash,
          String currentSourceHash,
          Iterable<String> binaryHashes,
        })
      >[
        (
          packageName: 'pocketfft',
          version: pocketfft_hashes.version,
          pinnedSourceHash: pocketfft_hashes.nativeSourceHash,
          currentSourceHash: pocketfft_hashes.computeNativeSourceHash(
            Directory('pkgs/pocketfft').uri,
          ),
          binaryHashes: pocketfft_hashes.fileHashes.values,
        ),
        (
          packageName: 'openblas',
          version: openblas_hashes.version,
          pinnedSourceHash: openblas_hashes.nativeSourceHash,
          currentSourceHash: openblas_hashes.computeNativeSourceHash(
            Directory('pkgs/openblas').uri,
          ),
          binaryHashes: openblas_hashes.fileHashes.values,
        ),
        (
          packageName: 'ndarray',
          version: ndarray_hashes.version,
          pinnedSourceHash: ndarray_hashes.nativeSourceHash,
          currentSourceHash: ndarray_hashes.computeNativeSourceHash(
            Directory('pkgs/ndarray').uri,
          ),
          binaryHashes: ndarray_hashes.fileHashes.values,
        ),
      ];

  var hasFailure = false;
  var hasStale = false;

  for (final check in checks) {
    final hasPlaceholders = check.binaryHashes.any(
      (h) => h.length != 64 || h.startsWith('00000000'),
    );
    if (hasPlaceholders) {
      hasFailure = true;
      final msg =
          'package:${check.packageName} has unpinned placeholder binary hashes in '
          'pkgs/${check.packageName}/lib/src/hook_helpers/hashes.dart.';
      stderr.writeln('[ERROR] $msg');
      if (isGitHubActions) {
        stdout.writeln(
          '::error file=pkgs/${check.packageName}/lib/src/hook_helpers/hashes.dart,title=Unpinned Binary Hashes::$msg',
        );
      }
      continue;
    }

    if (check.currentSourceHash != check.pinnedSourceHash) {
      hasStale = true;
      final msg =
          'package:${check.packageName} native sources in pkgs/${check.packageName}/hook/ '
          '(${check.currentSourceHash.substring(0, 12)}) have changed since '
          'prebuilt release ${check.version} (${check.pinnedSourceHash.substring(0, 12)}). '
          'Before publishing to pub.dev, run `.github/workflows/artifacts.yml` and '
          '`dart tool/regenerate_hashes.dart <new-release-tag>`.';
      stderr.writeln('[STALE] $msg');
      if (isGitHubActions) {
        stdout.writeln(
          '::warning file=pkgs/${check.packageName}/lib/src/hook_helpers/hashes.dart,title=Prebuilt Artifacts Out of Date::$msg',
        );
      }
    } else {
      stdout.writeln(
        '[OK]    package:${check.packageName} (${check.version}): '
        'native sources match prebuilt artifacts (${check.pinnedSourceHash.substring(0, 12)})',
      );
    }
  }

  if (hasFailure || (strict && hasStale)) {
    exit(1);
  }
}
