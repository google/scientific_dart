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
import 'dart:typed_data';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:openblas/src/hook_helpers/build_options.dart';
import 'package:openblas/src/hook_helpers/hashes.dart';
import 'package:test/test.dart';

BuildInput _makeInput({
  required Uri baseUri,
  Map<String, Object?> defines = const {},
}) {
  final builder = BuildInputBuilder()
    ..setupShared(
      packageRoot: baseUri,
      packageName: 'openblas',
      outputDirectoryShared: Directory.systemTemp.uri,
      outputFile: Directory.systemTemp.uri.resolve('out.json'),
      userDefines: PackageUserDefines(
        workspacePubspec: PackageUserDefinesSource(
          defines: defines,
          basePath: baseUri,
        ),
      ),
    )
    ..config.setupBuild(linkingEnabled: false);
  return BuildInput(builder.json);
}

void main() {
  group('openblas BuildOptions & hashes', () {
    test(
      'defaults to BuildModeEnum.fetch (implicit) when user_defines is empty',
      () {
        final input = _makeInput(baseUri: Directory.current.uri);
        final options = BuildOptions.fromDefines(
          input.userDefines,
          environment: const {},
        );
        expect(options.buildMode, equals(BuildModeEnum.fetch));
        expect(options.isExplicit, isFalse);
        expect(options.sanitize, isNull);
        expect(options.coverage, isFalse);
      },
    );

    test(
      'switches implicit fetch to source when source-only options are set',
      () {
        final input = _makeInput(
          baseUri: Directory.current.uri,
          defines: const {'sanitize': 'address,undefined', 'coverage': true},
        );
        final options = BuildOptions.fromDefines(
          input.userDefines,
          environment: const {},
        );
        expect(options.buildMode, equals(BuildModeEnum.source));
        expect(options.isExplicit, isFalse);
        expect(options.sanitize, equals('address,undefined'));
        expect(options.coverage, isTrue);
      },
    );

    test(
      'parses buildMode: local and resolves localPath and localExtensionsPath',
      () {
        final baseUri = Directory.current.uri;
        final input = _makeInput(
          baseUri: baseUri,
          defines: const {
            'buildMode': 'local',
            'localPath': 'dist/openblas-linux-x64.so',
            'localExtensionsPath': 'dist/openblas_extensions-linux-x64.so',
          },
        );
        final options = BuildOptions.fromDefines(
          input.userDefines,
          environment: const {},
        );
        expect(options.buildMode, equals(BuildModeEnum.local));
        expect(options.isExplicit, isTrue);
        expect(
          options.localPath,
          equals(baseUri.resolve('dist/openblas-linux-x64.so')),
        );
        expect(
          options.localExtensionsPath,
          equals(baseUri.resolve('dist/openblas_extensions-linux-x64.so')),
        );
      },
    );

    test('throws ArgumentError on conflicting options', () {
      final baseUri = Directory.current.uri;

      // explicit fetch with checkoutPath
      expect(
        () => BuildOptions.fromDefines(
          _makeInput(
            baseUri: baseUri,
            defines: const {
              'buildMode': 'fetch',
              'checkoutPath': '/tmp/openblas',
            },
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );

      // explicit fetch with sanitize
      expect(
        () => BuildOptions.fromDefines(
          _makeInput(
            baseUri: baseUri,
            defines: const {'buildMode': 'fetch', 'sanitize': 'undefined'},
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );

      // source mode with localPath
      expect(
        () => BuildOptions.fromDefines(
          _makeInput(
            baseUri: baseUri,
            defines: const {
              'buildMode': 'source',
              'localPath': 'dist/openblas-linux-x64.so',
            },
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );
    });

    test('artifact names and embedded source hash verification', () {
      expect(
        openblasArtifactName(OS.linux, Architecture.x64, 'openblas'),
        equals('openblas-linux-x64.so'),
      );
      expect(
        openblasArtifactName(OS.linux, Architecture.x64, 'openblas_extensions'),
        equals('openblas_extensions-linux-x64.so'),
      );

      final currentHash = computeNativeSourceHash(Directory.current.uri);
      expect(currentHash.length, equals(64));

      final payload = Uint8List.fromList([
        0x7f,
        0x45,
        0x4c,
        0x46,
        0x00,
        ...ascii.encode('$sourceHashMarkerPrefix$nativeSourceHash'),
        0x00,
      ]);
      expect(extractEmbeddedSourceHash(payload), equals(nativeSourceHash));
      verifyArtifactSourceHash(payload, currentSourceHash: nativeSourceHash);

      expect(
        () => verifyArtifactSourceHash(
          Uint8List.fromList([1, 2, 3, 4]),
          currentSourceHash: nativeSourceHash,
          releaseVersion: 'artifacts-v0.0.3',
        ),
        throwsStateError,
      );
    });

    test(
      'nativeSourceHash matches the current hook/ sources only if they are unchanged since the release tag',
      () {
        final pkgRoot = Directory.current;
        final tagCheck = Process.runSync('git', [
          'rev-parse',
          '--verify',
          '--quiet',
          'refs/tags/$version',
        ], workingDirectory: pkgRoot.path);
        if (tagCheck.exitCode != 0) {
          markTestSkipped(
            'Release tag $version is not available locally; run `git fetch --tags`.',
          );
          return;
        }
        final diff = Process.runSync('git', [
          'diff',
          '--quiet',
          version,
          '--',
          for (final extension in ['dart', 'c', 'cpp', 'h', 'def'])
            ':(glob)hook/*.$extension',
        ], workingDirectory: pkgRoot.path);
        expect(
          diff.exitCode,
          anyOf(0, 1),
          reason: 'git diff failed: ${diff.stderr}',
        );
        final sourcesChangedSinceRelease = diff.exitCode == 1;
        final pinMatchesCurrentSources =
            nativeSourceHash == computeNativeSourceHash(pkgRoot.uri);
        expect(
          pinMatchesCurrentSources,
          !sourcesChangedSinceRelease,
          reason: sourcesChangedSinceRelease
              ? 'hook/ sources changed since release $version, but '
                    'nativeSourceHash claims they match the prebuilt binaries. '
                    'Keep the pin, or cut a new release and run '
                    '`dart tool/regenerate_hashes.dart <tag>`.'
              : 'hook/ sources are unchanged since release $version, but '
                    'nativeSourceHash does not match them.',
        );
      },
    );
  });
}
