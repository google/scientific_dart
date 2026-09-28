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
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:pocketfft/src/hook_helpers/build_options.dart';
import 'package:pocketfft/src/hook_helpers/hashes.dart';
import 'package:test/test.dart';

BuildInput _makeInput({
  required Uri baseUri,
  Map<String, Object?> defines = const {},
}) {
  final builder = BuildInputBuilder()
    ..setupShared(
      packageRoot: baseUri,
      packageName: 'pocketfft',
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
  group('BuildOptions & hashes', () {
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
      'switches implicit fetch to source when sanitize or coverage is set',
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

    test('parses buildMode: local and resolves relative localPath', () {
      final baseUri = Directory.current.uri;
      final input = _makeInput(
        baseUri: baseUri,
        defines: const {
          'buildMode': 'local',
          'localPath': 'dist/libpocketfft.so',
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
        equals(baseUri.resolve('dist/libpocketfft.so')),
      );
    });

    test('throws ArgumentError on conflicting or invalid options', () {
      final baseUri = Directory.current.uri;

      // Invalid buildMode
      expect(
        () => BuildOptions.fromDefines(
          _makeInput(
            baseUri: baseUri,
            defines: const {'buildMode': 'invalid_mode'},
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );

      // Explicit fetch with sanitize
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

      // localPath with source mode
      expect(
        () => BuildOptions.fromDefines(
          _makeInput(
            baseUri: baseUri,
            defines: const {
              'buildMode': 'source',
              'localPath': 'dist/libpocketfft.so',
            },
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );
    });

    test('pocketfftArtifactName and embedded source hash verification', () {
      expect(
        pocketfftArtifactName(OS.linux, Architecture.x64),
        equals('pocketfft-linux-x64.so'),
      );
      expect(
        pocketfftArtifactName(OS.macOS, Architecture.arm64),
        equals('pocketfft-macos-arm64.dylib'),
      );
      expect(
        pocketfftArtifactName(OS.windows, Architecture.x64),
        equals('pocketfft-windows-x64.dll'),
      );
      expect(fileHashes.containsKey((OS.linux, Architecture.x64)), isTrue);
      expect(sha256.convert(const [1, 2, 3]).toString().length, equals(64));

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

      // Missing marker on release > artifacts-v0.0.2 throws StateError
      expect(
        () => verifyArtifactSourceHash(
          Uint8List.fromList([1, 2, 3, 4]),
          currentSourceHash: nativeSourceHash,
          releaseVersion: 'artifacts-v0.0.3',
        ),
        throwsStateError,
      );

      // Mismatched embedded hash throws StateError
      final wrongHash = '0' * 64;
      final stalePayload = Uint8List.fromList(
        ascii.encode('$sourceHashMarkerPrefix$wrongHash'),
      );
      expect(
        () => verifyArtifactSourceHash(
          stalePayload,
          currentSourceHash: nativeSourceHash,
        ),
        throwsStateError,
      );
    });

    test(
      'nativeSourceHash matches the current hook/ sources only if they are unchanged since the release tag',
      () {
        final pkgRoot = Directory('pkgs/pocketfft').existsSync()
            ? Directory('pkgs/pocketfft').absolute
            : Directory.current;
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
