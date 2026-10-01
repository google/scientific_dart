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
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/cpu_check.dart';
import 'package:ndarray/src/hook_helpers/build_options.dart';
import 'package:ndarray/src/hook_helpers/hashes.dart' as ndarray_hashes;
import 'package:openblas/src/hook_helpers/hashes.dart' as openblas_hashes;
import 'package:pocketfft/src/hook_helpers/hashes.dart' as pocketfft_hashes;
import 'package:test/test.dart';

BuildInput _makeNdarrayInput({
  required Uri baseUri,
  Map<String, Object?> defines = const {},
}) {
  final builder = BuildInputBuilder()
    ..setupShared(
      packageRoot: baseUri,
      packageName: 'ndarray',
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

Directory _findRepoRoot() {
  var dir = Directory.current.absolute;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/pkgs/ndarray').existsSync()) {
      return dir;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError(
        'Could not locate workspace root from ${Directory.current.path}',
      );
    }
    dir = parent;
  }
}

void main() {
  final repoRoot = _findRepoRoot();

  group('B1 & B2: Source Hashes & Embedded Binary Marker Verification', () {
    test(
      'nativeSourceFiles is non-recursive and deterministic across all packages',
      () {
        for (final (pkgName, filesFn, computeFn) in [
          (
            'ndarray',
            ndarray_hashes.nativeSourceFiles,
            ndarray_hashes.computeNativeSourceHash,
          ),
          (
            'openblas',
            openblas_hashes.nativeSourceFiles,
            openblas_hashes.computeNativeSourceHash,
          ),
          (
            'pocketfft',
            pocketfft_hashes.nativeSourceFiles,
            pocketfft_hashes.computeNativeSourceHash,
          ),
        ]) {
          final pkgUri = Directory('${repoRoot.path}/pkgs/$pkgName').uri;
          final files = filesFn(pkgUri);
          expect(files, isNotEmpty, reason: '$pkgName must have hook files');
          for (final file in files) {
            expect(
              file.parent.uri.normalizePath(),
              equals(pkgUri.resolve('hook/').normalizePath()),
              reason:
                  '$pkgName nativeSourceFiles must only list direct files in hook/',
            );
          }
          final hash1 = computeFn(pkgUri);
          final hash2 = computeFn(pkgUri);
          expect(hash1, equals(hash2));
          expect(hash1.length, equals(64));
        }
      },
    );

    test(
      'extractEmbeddedSourceHash and verifyArtifactSourceHash enforce freshness',
      () {
        const pinnedHash = ndarray_hashes.nativeSourceHash;
        final validBytes = Uint8List.fromList([
          0,
          1,
          2,
          ...ascii.encode(
            '${ndarray_hashes.sourceHashMarkerPrefix}$pinnedHash',
          ),
          0,
        ]);
        expect(
          ndarray_hashes.extractEmbeddedSourceHash(validBytes),
          equals(pinnedHash),
        );
        ndarray_hashes.verifyArtifactSourceHash(
          validBytes,
          currentSourceHash: pinnedHash,
        );

        // Missing marker on release > artifacts-v0.0.2 throws StateError
        expect(
          () => ndarray_hashes.verifyArtifactSourceHash(
            Uint8List.fromList([1, 2, 3]),
            currentSourceHash: pinnedHash,
            releaseVersion: 'artifacts-v0.0.3',
          ),
          throwsStateError,
        );

        // Mismatched currentSourceHash vs nativeSourceHash throws StateError
        expect(
          () => ndarray_hashes.verifyArtifactSourceHash(
            validBytes,
            currentSourceHash: 'b' * 64,
          ),
          throwsStateError,
        );

        // Mismatched embedded hash throws StateError
        final staleBytes = Uint8List.fromList(
          ascii.encode('${ndarray_hashes.sourceHashMarkerPrefix}${'a' * 64}'),
        );
        expect(
          () => ndarray_hashes.verifyArtifactSourceHash(
            staleBytes,
            currentSourceHash: pinnedHash,
          ),
          throwsStateError,
        );
      },
    );

    test('built shared libraries embed current <PKG>_SOURCE_HASH marker', () {
      // Ensure native assets are loaded
      NDArray.scope(() {
        final a = NDArray.zeros([2], DType.float64);
        expect(a.size, equals(2));
      });

      final sharedDir = Directory(
        '${repoRoot.path}/.dart_tool/hooks_runner/shared',
      );
      if (!sharedDir.existsSync()) return;

      final ext = Platform.isLinux
          ? '.so'
          : Platform.isMacOS
          ? '.dylib'
          : '.dll';

      final ndarrayHash = ndarray_hashes.computeNativeSourceHash(
        Directory('${repoRoot.path}/pkgs/ndarray').uri,
      );
      final openblasHash = openblas_hashes.computeNativeSourceHash(
        Directory('${repoRoot.path}/pkgs/openblas').uri,
      );
      final pocketfftHash = pocketfft_hashes.computeNativeSourceHash(
        Directory('${repoRoot.path}/pkgs/pocketfft').uri,
      );

      final allLibs = sharedDir
          .listSync(recursive: true)
          .whereType<File>()
          .where((f) => !f.path.contains('-artifacts-'))
          .toList();

      File latestLib(List<File> candidates) {
        candidates.sort(
          (a, b) => b.lastModifiedSync().compareTo(a.lastModifiedSync()),
        );
        return candidates.first;
      }

      final ndarrayLibs = allLibs
          .where(
            (f) =>
                (f.path.endsWith('libndarray$ext') ||
                    f.path.endsWith('ndarray$ext')) &&
                !f.path.contains('cpu_check'),
          )
          .toList();
      expect(ndarrayLibs, isNotEmpty);
      final activeNdarrayLib = latestLib(ndarrayLibs);
      expect(
        ndarray_hashes.extractEmbeddedSourceHash(
          activeNdarrayLib.readAsBytesSync(),
        ),
        equals(ndarrayHash),
        reason:
            '${activeNdarrayLib.path} must embed current NDARRAY_SOURCE_HASH',
      );

      final openblasExtLibs = allLibs
          .where(
            (f) =>
                f.path.contains('openblas_extensions') && f.path.endsWith(ext),
          )
          .toList();
      expect(openblasExtLibs, isNotEmpty);
      final activeOpenblasExtLib = latestLib(openblasExtLibs);
      expect(
        openblas_hashes.extractEmbeddedSourceHash(
          activeOpenblasExtLib.readAsBytesSync(),
        ),
        equals(openblasHash),
        reason:
            '${activeOpenblasExtLib.path} must embed current OPENBLAS_SOURCE_HASH',
      );

      final pocketfftLibs = allLibs
          .where((f) => f.path.contains('pocketfft') && f.path.endsWith(ext))
          .toList();
      expect(pocketfftLibs, isNotEmpty);
      final activePocketfftLib = latestLib(pocketfftLibs);
      expect(
        pocketfft_hashes.extractEmbeddedSourceHash(
          activePocketfftLib.readAsBytesSync(),
        ),
        equals(pocketfftHash),
        reason:
            '${activePocketfftLib.path} must embed current POCKETFFT_SOURCE_HASH',
      );
    });
  });

  group('H6: BuildOptions user_defines, Sanitizer Flags, & Cache Key Isolation', () {
    test('defaults to implicit fetch mode with empty user_defines and env', () {
      final input = _makeNdarrayInput(baseUri: repoRoot.uri);
      final opts = BuildOptions.fromDefines(
        input.userDefines,
        environment: const {},
      );
      expect(opts.buildMode, equals(BuildModeEnum.fetch));
      expect(opts.isExplicit, isFalse);
      expect(opts.sanitize, isNull);
      expect(opts.effectiveSanitize, isNull);
      expect(opts.coverage, isFalse);
      expect(opts.x86Flags, isNull);
    });

    test(
      'automatically appends float-cast-overflow when undefined sanitizer is enabled',
      () {
        final input = _makeNdarrayInput(
          baseUri: repoRoot.uri,
          defines: const {
            'buildMode': 'source',
            'sanitize': 'address,undefined',
          },
        );
        final opts = BuildOptions.fromDefines(
          input.userDefines,
          environment: const {},
        );
        expect(opts.buildMode, equals(BuildModeEnum.source));
        expect(opts.isExplicit, isTrue);
        expect(
          opts.effectiveSanitize,
          equals('address,undefined,float-cast-overflow'),
        );
        expect(
          opts.sanitizeFlags,
          contains('-fsanitize=address,undefined,float-cast-overflow'),
        );
        expect(opts.sanitizeFlags, contains('-fno-omit-frame-pointer'));
      },
    );

    test('does not duplicate float-cast-overflow if already present', () {
      final input = _makeNdarrayInput(
        baseUri: repoRoot.uri,
        defines: const {
          'buildMode': 'source',
          'sanitize': 'undefined,float-cast-overflow',
        },
      );
      final opts = BuildOptions.fromDefines(
        input.userDefines,
        environment: const {},
      );
      expect(opts.effectiveSanitize, equals('undefined,float-cast-overflow'));
    });

    test(
      'implicit fetch upgrades to source when sanitize, coverage, or x86Flags are set',
      () {
        for (final defines in [
          const {'sanitize': 'undefined'},
          const {'coverage': true},
          const {'x86Flags': '-msse4.2'},
        ]) {
          final input = _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: defines,
          );
          final opts = BuildOptions.fromDefines(
            input.userDefines,
            environment: const {},
          );
          expect(opts.buildMode, equals(BuildModeEnum.source));
          expect(opts.isExplicit, isFalse);
        }
      },
    );

    test('rejects conflicting user_defines options with ArgumentError', () {
      // Explicit fetch + sanitize
      expect(
        () => BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {'buildMode': 'fetch', 'sanitize': 'address'},
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );

      // Explicit fetch + x86Flags
      expect(
        () => BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {'buildMode': 'fetch', 'x86Flags': '-mavx2'},
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );

      // Source mode with localPath
      expect(
        () => BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {
              'buildMode': 'source',
              'localPath': 'dist/ndarray-linux-x64.so',
            },
          ).userDefines,
          environment: const {},
        ),
        throwsArgumentError,
      );
    });

    test(
      'validateTarget rejects unsupported OS/compiler/architecture combinations',
      () {
        final sanitizeOpts = BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {'buildMode': 'source', 'sanitize': 'address'},
          ).userDefines,
          environment: const {},
        );
        expect(
          () => sanitizeOpts.validateTarget(
            targetOS: OS.windows,
            targetArch: Architecture.x64,
            isMSVC: true,
          ),
          throwsUnsupportedError,
        );

        final x86Opts = BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {'buildMode': 'source', 'x86Flags': '-msse4.2'},
          ).userDefines,
          environment: const {},
        );
        expect(
          () => x86Opts.validateTarget(
            targetOS: OS.linux,
            targetArch: Architecture.arm64,
            isMSVC: false,
          ),
          throwsUnsupportedError,
        );
      },
    );

    test(
      'cacheKey isolates builds with different compiler/instrumentation flags',
      () {
        final defaultSource = BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {'buildMode': 'source'},
          ).userDefines,
          environment: const {},
        );
        final asanSource = BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {
              'buildMode': 'source',
              'sanitize': 'address,undefined',
            },
          ).userDefines,
          environment: const {},
        );
        final covSource = BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {'buildMode': 'source', 'coverage': true},
          ).userDefines,
          environment: const {},
        );
        final sseSource = BuildOptions.fromDefines(
          _makeNdarrayInput(
            baseUri: repoRoot.uri,
            defines: const {'buildMode': 'source', 'x86Flags': '-msse4.2'},
          ).userDefines,
          environment: const {},
        );

        final keys = {
          defaultSource.cacheKey,
          asanSource.cacheKey,
          covSource.cacheKey,
          sseSource.cacheKey,
        };
        expect(keys.length, equals(4));
      },
    );
  });

  group('M7: Baseline x86_64 CPU Feature Guard (ensureCpuSupported)', () {
    test(
      'ensureCpuSupported succeeds on the current host and is idempotent',
      () {
        ensureCpuSupported();
        ensureCpuSupported();
      },
    );

    test(
      'verifyCpuFeatureMask accepts full x86FeatureAllDefault mask and rejects missing features',
      () {
        verifyCpuFeatureMask(actualFeatures: x86FeatureAllDefault);

        for (var mask = 0; mask < x86FeatureAllDefault; mask++) {
          final missingMatchers = <Matcher>[
            contains('x86Flags: ""'),
            if ((mask & x86FeatureOsxsave) == 0) contains('OSXSAVE'),
            if ((mask & x86FeatureYmmState) == 0)
              contains('OS AVX/YMM state (XGETBV)'),
            if ((mask & x86FeatureAvx) == 0) contains('AVX'),
            if ((mask & x86FeatureAvx2) == 0) contains('AVX2'),
            if ((mask & x86FeatureFma) == 0) contains('FMA'),
            if ((mask & x86FeatureF16c) == 0) contains('F16C'),
          ];
          expect(
            () => verifyCpuFeatureMask(actualFeatures: mask),
            throwsA(
              isA<UnsupportedError>().having(
                (e) => e.message,
                'message',
                allOf(missingMatchers),
              ),
            ),
          );
        }
      },
    );
  });

  group('Packaging & Workspace Consistency', () {
    test(
      'hooks and code_assets version constraints match across workspace pubspecs',
      () {
        final pubspecPaths = [
          'pubspec.yaml',
          'pkgs/ndarray/pubspec.yaml',
          'pkgs/openblas/pubspec.yaml',
          'pkgs/pocketfft/pubspec.yaml',
        ];
        for (final relPath in pubspecPaths) {
          final content = File('${repoRoot.path}/$relPath').readAsStringSync();
          expect(
            content,
            contains("hooks: '>=1.0.3 <3.0.0'"),
            reason: '$relPath must use aligned hooks constraint',
          );
          expect(
            content,
            contains("code_assets: '>=1.0.0 <3.0.0'"),
            reason: '$relPath must use aligned code_assets constraint',
          );
        }
      },
    );

    test(
      'pkgs/ndarray/.pubignore supersets pkgs/ndarray/.gitignore active rules and excludes build artifacts',
      () {
        final pubignoreFile = File('${repoRoot.path}/pkgs/ndarray/.pubignore');
        final gitignoreFile = File('${repoRoot.path}/pkgs/ndarray/.gitignore');
        expect(
          pubignoreFile.existsSync(),
          isTrue,
          reason: '.pubignore must exist',
        );
        expect(
          gitignoreFile.existsSync(),
          isTrue,
          reason: '.gitignore must exist',
        );

        final pubignoreContent = pubignoreFile.readAsStringSync();
        final pubignoreLines = pubignoreContent
            .split('\n')
            .map((l) => l.trim())
            .where((l) => l.isNotEmpty && !l.startsWith('#'))
            .toSet();

        final gitignoreActiveLines = gitignoreFile
            .readAsStringSync()
            .split('\n')
            .map((l) => l.trim())
            .where((l) => l.isNotEmpty && !l.startsWith('#'))
            .toList();

        for (final rule in gitignoreActiveLines) {
          expect(
            pubignoreLines.contains(rule),
            isTrue,
            reason: '.pubignore must contain active .gitignore rule: "$rule"',
          );
        }

        // Specifically assert critical artifacts are ignored
        expect(pubignoreContent, contains('coverage/'));
        expect(pubignoreContent, contains('scratch/'));
        expect(pubignoreContent, contains('pubspec.lock'));
        expect(pubignoreContent, contains('third_party/highway/build/'));
        expect(pubignoreContent, contains('third_party/highway/hwy_build/'));
      },
    );

    test(
      'pkgs/ndarray/.pubignore excludes Highway docs, tests, and build artifacts',
      () {
        final pubignoreFile = File('${repoRoot.path}/pkgs/ndarray/.pubignore');
        expect(pubignoreFile.existsSync(), isTrue);
        final content = pubignoreFile.readAsStringSync();
        expect(content, contains('third_party/highway/g3doc/'));
        expect(content, contains('third_party/highway/docs/'));
        expect(content, contains('third_party/highway/hwy/tests/'));
        expect(content, contains('third_party/highway/hwy_build/'));
      },
    );

    test(
      'C++ macro and OOM flag invariants: NoThrowBuffer sets OOM flag and VECTORIZED_TARGETS is guarded',
      () {
        final indexingFile = File(
          '${repoRoot.path}/pkgs/ndarray/hook/custom_indexing.cpp',
        );
        expect(indexingFile.existsSync(), isTrue);
        final indexingContent = indexingFile.readAsStringSync();

        // NoThrowBuffer calls ndarray_set_oom_flag() on allocation failure
        expect(
          indexingContent,
          contains('ndarray_set_oom_flag()'),
          reason:
              'custom_indexing.cpp NoThrowBuffer must call ndarray_set_oom_flag() on OOM',
        );

        final ufuncsFile = File(
          '${repoRoot.path}/pkgs/ndarray/hook/custom_ufuncs.cpp',
        );
        expect(ufuncsFile.existsSync(), isTrue);
        final ufuncsContent = ufuncsFile.readAsStringSync();

        // #define VECTORIZED_TARGETS guarded by #ifndef VECTORIZED_TARGETS
        final guardedRegex = RegExp(
          r'#ifndef\s+VECTORIZED_TARGETS\s+[\s\S]*?#define\s+VECTORIZED_TARGETS',
        );
        expect(
          guardedRegex.hasMatch(ufuncsContent),
          isTrue,
          reason:
              'custom_ufuncs.cpp #define VECTORIZED_TARGETS must be guarded by #ifndef VECTORIZED_TARGETS',
        );
      },
    );
  });
}
