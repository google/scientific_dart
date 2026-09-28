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

import 'package:code_assets/code_assets.dart';
import 'package:gpuarray/src/hook_helpers/build_options.dart';
import 'package:gpuarray/src/hook_helpers/hashes.dart';
import 'package:hooks/hooks.dart';
import 'package:test/test.dart';

void main() {
  final pkgRoot = Directory.current.path.endsWith('pkgs/gpuarray')
      ? Directory.current
      : Directory('pkgs/gpuarray');

  group('Build infrastructure & hook invariants', () {
    test(
      'fileHashes in hashes.dart contains valid 64-hex SHA-256 hashes for all major platforms',
      () {
        expect(wgpuVersion, startsWith('v'));
        expect(
          fileHashes.keys,
          containsAll([
            (OS.linux, Architecture.x64),
            (OS.linux, Architecture.arm64),
            (OS.macOS, Architecture.arm64),
            (OS.macOS, Architecture.x64),
            (OS.windows, Architecture.x64),
          ]),
        );
        final sha256Hex = RegExp(r'^[0-9a-f]{64}$');
        for (final entry in fileHashes.entries) {
          expect(
            sha256Hex.hasMatch(entry.value.sha256),
            isTrue,
            reason:
                'Invalid SHA-256 hash for ${entry.key}: ${entry.value.sha256}',
          );
          expect(entry.value.zipName, endsWith('.zip'));
          expect(entry.value.libName, isNotEmpty);
        }
      },
    );

    BuildInput makeInput(Map<String, Object?> defines) {
      final builder = BuildInputBuilder()
        ..setupShared(
          packageRoot: pkgRoot.absolute.uri,
          packageName: 'gpuarray',
          outputDirectoryShared: Directory.systemTemp.uri,
          outputFile: Directory.systemTemp.uri.resolve('out.json'),
          userDefines: PackageUserDefines(
            workspacePubspec: PackageUserDefinesSource(
              defines: defines,
              basePath: pkgRoot.absolute.uri,
            ),
          ),
        )
        ..config.setupBuild(linkingEnabled: false);
      return BuildInput(builder.json);
    }

    test(
      'BuildOptions.fromDefines parses valid modes and throws FormatException on invalid values',
      () {
        final defaults = BuildOptions.fromDefines(
          makeInput(const {}).userDefines,
        );
        expect(defaults.buildMode, equals(BuildModeEnum.fetch));
        expect(defaults.localPath, isNull);
        expect(defaults.checkoutPath, isNull);

        final custom = BuildOptions.fromDefines(
          makeInput(const {
            'buildMode': 'local',
            'localPath': 'libwgpu_native.so',
          }).userDefines,
        );
        expect(custom.buildMode, equals(BuildModeEnum.local));
        expect(custom.localPath, isNotNull);

        expect(
          () => BuildOptions.fromDefines(
            makeInput(const {'buildMode': 'nonexistent'}).userDefines,
          ),
          throwsFormatException,
        );
      },
    );

    test(
      'hook/build.dart CLI entrypoint exits with code 2 and writes to stderr on invalid user defines',
      () async {
        final tempDir = await Directory.systemTemp.createTemp(
          'gpuarray_hook_test_',
        );
        try {
          final outDir = Directory('${tempDir.path}/out')..createSync();
          final sharedDir = Directory('${tempDir.path}/shared')..createSync();
          final outputFile = File('${tempDir.path}/output.json');
          final inputFile = File('${tempDir.path}/input.json');

          final buildInputJson = {
            'assets': <String, Object?>{},
            'config': {
              'build_asset_types': ['code_assets/code'],
              'linking_enabled': false,
              'extensions': {
                'code_assets': {
                  'c_compiler': null,
                  'link_mode_preference': 'dynamic',
                  'target_architecture': 'x64',
                  'target_os': 'linux',
                },
              },
            },
            'out_dir_shared': sharedDir.path,
            'out_dir': outDir.path,
            'out_file': outputFile.path,
            'package_name': 'gpuarray',
            'package_root': pkgRoot.absolute.path,
            'user_defines': {
              'workspace_pubspec': {
                'base_path': pkgRoot.absolute.path,
                'defines': {'buildMode': 'invalid_mode_value'},
              },
            },
            'version': '1.9.0',
          };
          inputFile.writeAsStringSync(jsonEncode(buildInputJson));

          final result = await Process.run(Platform.resolvedExecutable, [
            '${pkgRoot.absolute.path}/hook/build.dart',
            '--config=${inputFile.path}',
          ], workingDirectory: pkgRoot.absolute.path);

          expect(
            result.exitCode,
            equals(2),
            reason:
                'Expected CLI exitCode 2 on invalid user_defines, got ${result.exitCode}.\n'
                'stdout: ${result.stdout}\nstderr: ${result.stderr}',
          );
          expect(
            result.stderr.toString(),
            contains('Unknown buildMode "invalid_mode_value"'),
          );
        } finally {
          await tempDir.delete(recursive: true);
        }
      },
    );
  });
}
