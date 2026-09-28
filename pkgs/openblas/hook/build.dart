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
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:openblas/src/hook_helpers/build_options.dart';
import 'package:openblas/src/hook_helpers/hashes.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final BuildOptions buildOptions;
    try {
      buildOptions = BuildOptions.fromDefines(input.userDefines);
    } catch (e) {
      throw ArgumentError(BuildOptions.usageError(e));
    }
    print('openblas build options: $buildOptions');

    final currentSourceHash = computeNativeSourceHash(input.packageRoot);
    if (currentSourceHash != nativeSourceHash &&
        buildOptions.buildMode == BuildModeEnum.source) {
      print(
        'WARNING: Native sources in package:${input.packageName}/hook/ '
        '(${currentSourceHash.substring(0, 12)}) differ from prebuilt release '
        '$version (${nativeSourceHash.substring(0, 12)}). '
        'Remember to build & attest new release artifacts and run '
        '`dart tool/regenerate_hashes.dart <tag>` before publishing.',
      );
    }

    BuildMode buildMode = switch (buildOptions.buildMode) {
      BuildModeEnum.fetch => FetchMode(input),
      BuildModeEnum.local => LocalMode(
        input,
        buildOptions.localPath,
        buildOptions.localExtensionsPath,
      ),
      BuildModeEnum.source => SourceMode(
        input,
        buildOptions.checkoutPath,
        buildOptions,
      ),
    };

    ({Uri openblasUri, Uri extensionsUri}) builtLibraries;
    if (buildOptions.buildMode == BuildModeEnum.fetch &&
        !buildOptions.isExplicit &&
        currentSourceHash != nativeSourceHash) {
      print(
        'Prebuilt openblas binary for release $version differs from local '
        'native sources in hook/; falling back to `buildMode: source`.',
      );
      buildMode = SourceMode(input, buildOptions.checkoutPath, buildOptions);
      builtLibraries = await buildMode.build();
    } else {
      try {
        builtLibraries = await buildMode.build();
      } catch (e) {
        if (buildOptions.buildMode == BuildModeEnum.fetch &&
            !buildOptions.isExplicit) {
          print(
            'Prebuilt openblas binary unavailable ($e); '
            'falling back to `buildMode: source`.',
          );
          buildMode = SourceMode(
            input,
            buildOptions.checkoutPath,
            buildOptions,
          );
          builtLibraries = await buildMode.build();
        } else {
          rethrow;
        }
      }
    }

    final (:openblasUri, :extensionsUri) = builtLibraries;

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'openblas',
        linkMode: DynamicLoadingBundled(),
        file: openblasUri,
      ),
    );
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'openblas_extensions',
        linkMode: DynamicLoadingBundled(),
        file: extensionsUri,
      ),
    );
    output.dependencies.addAll(buildMode.dependencies);
    output.dependencies.add(input.packageRoot.resolve('pubspec.yaml'));
  });
}

String _canonicalOpenblasName(OS os) => os == OS.windows
    ? 'libopenblas.dll'
    : ((os == OS.macOS || os == OS.iOS)
          ? 'libopenblas.dylib'
          : 'libopenblas.so');

String _canonicalExtensionsName(OS os) => os == OS.windows
    ? 'libopenblas_extensions.dll'
    : ((os == OS.macOS || os == OS.iOS)
          ? 'libopenblas_extensions.dylib'
          : 'libopenblas_extensions.so');

sealed class BuildMode {
  final BuildInput input;

  const BuildMode(this.input);

  List<Uri> get dependencies;

  Future<({Uri openblasUri, Uri extensionsUri})> build();
}

final class FetchMode extends BuildMode {
  FetchMode(super.input);

  @override
  Future<({Uri openblasUri, Uri extensionsUri})> build() async {
    final currentSourceHash = computeNativeSourceHash(input.packageRoot);
    if (currentSourceHash != nativeSourceHash) {
      throw StateError(
        'Prebuilt openblas binary for release $version is out of date with native sources in hook/!\n'
        'Pinned nativeSourceHash: $nativeSourceHash\n'
        'Current hook/ hash:      $currentSourceHash\n'
        'If you are the package author, build and attest new release artifacts (.github/workflows/artifacts.yml) and run:\n'
        '  dart tool/regenerate_hashes.dart <new-release-tag>\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` while developing native code.')}',
      );
    }

    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;

    final openblasArtifact = openblasArtifactName(os, arch, 'openblas');
    final extArtifact = openblasArtifactName(os, arch, 'openblas_extensions');
    final expectedOpenblasHash = fileHashes[(os, arch, 'openblas')];
    final expectedExtHash = fileHashes[(os, arch, 'openblas_extensions')];

    if (expectedOpenblasHash == null ||
        expectedOpenblasHash.startsWith('00000000') ||
        expectedExtHash == null ||
        expectedExtHash.startsWith('00000000')) {
      throw StateError(
        'No prebuilt openblas binary hashes are pinned for ($os, $arch) in release $version.\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` or `buildMode: local`.')}',
      );
    }

    final sharedDir = input.outputDirectoryShared.resolve(
      'openblas-$version/${os.name}-${arch.name}/',
    );
    final cachedOpenblas = File.fromUri(
      sharedDir.resolve(_canonicalOpenblasName(os)),
    );
    final cachedExt = File.fromUri(
      sharedDir.resolve(_canonicalExtensionsName(os)),
    );

    final openblasUri = await _fetchOrUseCached(
      cachedFile: cachedOpenblas,
      artifactName: openblasArtifact,
      expectedHash: expectedOpenblasHash,
      currentSourceHash: currentSourceHash,
      verifySourceStamp: false,
    );
    final extensionsUri = await _fetchOrUseCached(
      cachedFile: cachedExt,
      artifactName: extArtifact,
      expectedHash: expectedExtHash,
      currentSourceHash: currentSourceHash,
      verifySourceStamp: true,
    );

    return (openblasUri: openblasUri, extensionsUri: extensionsUri);
  }

  Future<Uri> _fetchOrUseCached({
    required File cachedFile,
    required String artifactName,
    required String expectedHash,
    required String currentSourceHash,
    required bool verifySourceStamp,
  }) async {
    if (await cachedFile.exists()) {
      final cachedBytes = await cachedFile.readAsBytes();
      final cachedHash = sha256.convert(cachedBytes).toString();
      if (cachedHash == expectedHash) {
        if (verifySourceStamp) {
          verifyArtifactSourceHash(
            cachedBytes,
            currentSourceHash: currentSourceHash,
          );
        }
        print('Using cached openblas artifact from ${cachedFile.path}.');
        return cachedFile.uri;
      }
    }

    final remoteUri = Uri.parse(
      'https://github.com/$repository/releases/download/$version/$artifactName',
    );
    print('Fetching prebuilt openblas artifact from $remoteUri...');
    final bytes = await _downloadBytesWithRedirects(remoteUri);
    final actualHash = sha256.convert(bytes).toString();
    if (actualHash != expectedHash) {
      throw StateError(
        'SHA-256 mismatch for prebuilt openblas artifact at $remoteUri:\n'
        'Expected: $expectedHash\n'
        'Actual:   $actualHash',
      );
    }
    if (verifySourceStamp) {
      verifyArtifactSourceHash(bytes, currentSourceHash: currentSourceHash);
    }

    await cachedFile.parent.create(recursive: true);
    await cachedFile.writeAsBytes(bytes, flush: true);
    return cachedFile.uri;
  }

  @override
  List<Uri> get dependencies => [
    for (final file in nativeSourceFiles(input.packageRoot)) file.uri,
  ];
}

final class LocalMode extends BuildMode {
  final Uri? localPath;
  final Uri? localExtensionsPath;

  LocalMode(super.input, this.localPath, this.localExtensionsPath);

  (File, File) _resolveLocalFiles() {
    if (localPath == null) {
      throw ArgumentError(
        '`localPath` is not set in `hooks.user_defines.openblas` '
        '(or `LOCAL_OPENBLAS_BINARY` environment variable).',
      );
    }
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final entityPath = localPath!.toFilePath(windows: Platform.isWindows);

    if (FileSystemEntity.isDirectorySync(entityPath)) {
      final dirUri = Directory(entityPath).uri;
      var openblasFile = File.fromUri(
        dirUri.resolve(_canonicalOpenblasName(os)),
      );
      if (!openblasFile.existsSync()) {
        openblasFile = File.fromUri(
          dirUri.resolve(openblasArtifactName(os, arch, 'openblas')),
        );
      }
      var extFile = File.fromUri(dirUri.resolve(_canonicalExtensionsName(os)));
      if (!extFile.existsSync()) {
        extFile = File.fromUri(
          dirUri.resolve(openblasArtifactName(os, arch, 'openblas_extensions')),
        );
      }
      if (!openblasFile.existsSync() || !extFile.existsSync()) {
        throw FileSystemException(
          'Could not find both ${_canonicalOpenblasName(os)} and '
          '${_canonicalExtensionsName(os)} in localPath directory.',
          entityPath,
        );
      }
      return (openblasFile, extFile);
    }

    final openblasFile = File(entityPath);
    if (!openblasFile.existsSync()) {
      throw FileSystemException(
        'Could not find local openblas binary.',
        entityPath,
      );
    }
    final File extFile;
    if (localExtensionsPath != null) {
      extFile = File(
        localExtensionsPath!.toFilePath(windows: Platform.isWindows),
      );
    } else {
      extFile = File.fromUri(
        openblasFile.parent.uri.resolve(_canonicalExtensionsName(os)),
      );
    }
    if (!extFile.existsSync()) {
      throw FileSystemException(
        'Could not find local openblas_extensions binary.',
        extFile.path,
      );
    }
    return (openblasFile, extFile);
  }

  @override
  Future<({Uri openblasUri, Uri extensionsUri})> build() async {
    final (srcOpenblas, srcExt) = _resolveLocalFiles();
    final os = input.config.code.targetOS;
    final dstOpenblas = File.fromUri(
      input.outputDirectory.resolve(_canonicalOpenblasName(os)),
    );
    final dstExt = File.fromUri(
      input.outputDirectory.resolve(_canonicalExtensionsName(os)),
    );
    await dstOpenblas.parent.create(recursive: true);
    await srcOpenblas.copy(dstOpenblas.path);
    await srcExt.copy(dstExt.path);
    return (openblasUri: dstOpenblas.uri, extensionsUri: dstExt.uri);
  }

  @override
  List<Uri> get dependencies {
    final (srcOpenblas, srcExt) = _resolveLocalFiles();
    return [srcOpenblas.uri, srcExt.uri];
  }
}

final class SourceMode extends BuildMode {
  final Uri? checkoutPath;
  final BuildOptions? buildOptions;

  SourceMode(super.input, this.checkoutPath, [this.buildOptions]);

  Uri get _root => checkoutPath ?? input.packageRoot;

  @override
  Future<({Uri openblasUri, Uri extensionsUri})> build() async {
    final openblas = OpenBlasBinary.forBuild(input);
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final cCompiler = input.config.code.cCompiler;
    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final customExtensionsPath = _root
        .resolve('hook/custom_extensions.c')
        .toFilePath();

    final currentSourceHash = computeNativeSourceHash(_root);
    final stampFile = File.fromUri(
      outputDir.uri.resolve('source_hash_stamp.c'),
    );
    await stampFile.writeAsString('''
#if defined(_WIN32)
#define STAMP_EXPORT __declspec(dllexport)
#define STAMP_USED
#else
#define STAMP_EXPORT __attribute__((visibility("default"), used))
#define STAMP_USED __attribute__((used))
#endif

STAMP_USED static const char _openblas_source_hash_marker[] =
    "$sourceHashMarkerPrefix$currentSourceHash";

STAMP_EXPORT const char* openblas_embedded_source_hash(void) {
  return _openblas_source_hash_marker;
}
''');

    switch (openblas) {
      case MacosAccelerateBinary():
        final compilerPath = cCompiler?.compiler.toFilePath() ?? 'cc';
        final stubFile = File.fromUri(
          outputDir.uri.resolve('accelerate_stub.c'),
        );
        await stubFile.writeAsString('''
static int _accelerate_num_threads = 1;
int openblas_get_num_threads(void) { return _accelerate_num_threads; }
void openblas_set_num_threads(int num_threads) {
  if (num_threads > 0) _accelerate_num_threads = num_threads;
}
const char* openblas_get_config(void) { return "MacOS Accelerate Framework"; }
''');

        final libFile = File.fromUri(
          outputDir.uri.resolve('libopenblas.dylib'),
        );
        final extLibFile = File.fromUri(
          outputDir.uri.resolve('libopenblas_extensions.dylib'),
        );

        final stubCompileArgs = [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            '-Wl,-install_name,@rpath/libopenblas.dylib',
            '-Wl,-headerpad_max_install_names',
          ],
          '-dynamiclib',
          '-O3',
          stubFile.path,
          customExtensionsPath,
          stampFile.path,
          '-o',
          libFile.path,
          '-framework',
          'Accelerate',
          '-Wl,-reexport_framework,Accelerate',
        ];
        final stubRes = await Process.run(compilerPath, stubCompileArgs);
        if (stubRes.exitCode != 0) {
          throw StateError(
            'Failed to compile Accelerate stub (exit ${stubRes.exitCode}):\n'
            'stdout: ${stubRes.stdout}\n'
            'stderr: ${stubRes.stderr}',
          );
        }

        final extCompileArgs = [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            '-Wl,-install_name,@rpath/libopenblas_extensions.dylib',
            '-Wl,-headerpad_max_install_names',
          ],
          '-dynamiclib',
          '-O3',
          customExtensionsPath,
          stampFile.path,
          '-o',
          extLibFile.path,
          '-framework',
          'Accelerate',
        ];
        final extRes = await Process.run(compilerPath, extCompileArgs);
        if (extRes.exitCode != 0) {
          throw StateError(
            'Failed to compile custom extensions (exit ${extRes.exitCode}):\n'
            'stdout: ${extRes.stdout}\n'
            'stderr: ${extRes.stderr}',
          );
        }

        return (openblasUri: libFile.uri, extensionsUri: extLibFile.uri);

      case PrecompiledBinary():
        final compilerPath =
            cCompiler?.compiler.toFilePath() ??
            (os == OS.windows ? 'cl' : 'cc');
        final compilerLower = compilerPath.toLowerCase();
        final isGNU =
            compilerLower.contains('gcc') ||
            compilerLower.contains('clang') ||
            compilerLower.contains('g++');
        final isMSVC = os == OS.windows && !isGNU;

        final zipUrl = Uri.parse(
          'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/OpenBLAS-0.3.33-x64.zip',
        );
        final extractDir = outputDir.uri.resolve('OpenBLAS-precompiled/');
        final extractDirFile = Directory.fromUri(extractDir);

        if (!extractDirFile.existsSync()) {
          print('Downloading precompiled OpenBLAS zip...');
          final zipBytes = await _downloadBytesWithRedirects(zipUrl);

          final actualZipHash = sha256.convert(zipBytes).toString();
          const expectedZipHash =
              '7ad797ef0c9a5c42e28903bf726eaaaade307dafe187ff0e923d90cd4002780c';
          if (actualZipHash != expectedZipHash) {
            throw StateError(
              'SHA-256 mismatch for OpenBLAS zip: expected $expectedZipHash, got $actualZipHash',
            );
          }

          final archive = ZipDecoder().decodeBytes(zipBytes);
          final extractDirPath = Directory.fromUri(extractDir).path;
          final safeExtractPrefix =
              extractDirPath.endsWith(Platform.pathSeparator)
              ? extractDirPath
              : '$extractDirPath${Platform.pathSeparator}';
          for (final file in archive) {
            final outPath = extractDir.resolve(file.name).toFilePath();
            if (!outPath.startsWith(safeExtractPrefix)) {
              throw FormatException(
                'Path traversal attempt in OpenBLAS zip: ${file.name}',
              );
            }
            if (file.isFile) {
              final outFile = File(outPath);
              outFile.createSync(recursive: true);
              outFile.writeAsBytesSync(file.content as List<int>, flush: true);
            } else {
              Directory(outPath).createSync(recursive: true);
            }
          }
        }

        final dllFile = File.fromUri(extractDir.resolve('bin/libopenblas.dll'));
        final headersDir = extractDir.resolve('include/');
        final libDir = extractDir.resolve('lib/');

        if (isMSVC) {
          final lapackHeader = File(
            headersDir.resolve('lapack.h').toFilePath(),
          );
          if (lapackHeader.existsSync()) {
            var content = await lapackHeader.readAsString();
            content = content.replaceAll(
              RegExp(r'typedef\s+[^;]+int32_t\s*;'),
              '/* patched typedef int32_t */',
            );
            content = content.replaceAll(
              RegExp(r'typedef\s+[^;]+uint32_t\s*;'),
              '/* patched typedef uint32_t */',
            );
            await lapackHeader.writeAsString(content);
          }
        }

        final openblasLibName = isMSVC
            ? 'libopenblas.lib'
            : 'libopenblas.dll.a';
        final openblasLibFile = File.fromUri(libDir.resolve(openblasLibName));

        final extLibFile = File(
          outputDir.uri.resolve('libopenblas_extensions.dll').toFilePath(),
        );
        final compileArgs = isMSVC
            ? [
                '/LD',
                '/O2',
                '/EHsc',
                '/I${headersDir.toFilePath()}',
                customExtensionsPath,
                stampFile.path,
                '/Fe:${extLibFile.path}',
                openblasLibFile.path,
                '/link',
                '/EXPORT:get_dgetrf_ptr',
                '/EXPORT:get_sgetrf_ptr',
                '/EXPORT:get_zgetrf_ptr',
                '/EXPORT:get_cgetrf_ptr',
                '/EXPORT:openblas_embedded_source_hash',
              ]
            : [
                '-shared',
                '-fPIC',
                '-O3',
                '-I${headersDir.toFilePath()}',
                customExtensionsPath,
                stampFile.path,
                '-o',
                extLibFile.path,
                openblasLibFile.path,
              ];

        final runEnv = <String, String>{...Platform.environment};
        if (isMSVC) {
          final msvcEnv = await getMSVCEnvironment(arch);
          for (final key in ['INCLUDE', 'LIB', 'LIBPATH']) {
            final val = msvcEnv[key] ?? msvcEnv[key.toLowerCase()];
            if (val != null) {
              runEnv[key] = val;
            }
          }
        }

        final extRes = await Process.run(
          compilerPath,
          compileArgs,
          environment: runEnv,
        );
        if (extRes.exitCode != 0) {
          throw StateError(
            'Failed to compile custom extensions (exit ${extRes.exitCode}):\n'
            'stdout: ${extRes.stdout}\n'
            'stderr: ${extRes.stderr}',
          );
        }

        return (openblasUri: dllFile.uri, extensionsUri: extLibFile.uri);

      case CompileOpenBlas(:final sourceUrl):
        String openBlasTarget = 'GENERIC';
        if (arch == Architecture.arm64) {
          openBlasTarget = 'ARMV8';
        } else if (arch == Architecture.arm) {
          openBlasTarget = 'ARMV7';
        } else if (arch == Architecture.ia32) {
          openBlasTarget = 'ATOM';
        }

        final legacyExtractDir = Directory.fromUri(
          outputDir.uri.resolve('OpenBLAS-0.3.33/'),
        );
        final sharedOpenblasBase = legacyExtractDir.existsSync()
            ? outputDir
            : Directory.fromUri(
                input.outputDirectoryShared.resolve(
                  'openblas-src-${os.name}-${arch.name}/',
                ),
              );
        if (!sharedOpenblasBase.existsSync()) {
          sharedOpenblasBase.createSync(recursive: true);
        }
        final extractDir = sharedOpenblasBase.uri
            .resolve('OpenBLAS-0.3.33/')
            .toFilePath();
        final libName = _canonicalOpenblasName(os);
        final libFile = File(
          sharedOpenblasBase.uri
              .resolve('OpenBLAS-0.3.33/$libName')
              .toFilePath(),
        );

        if (!libFile.existsSync()) {
          print('Downloading OpenBLAS release...');
          final tarGzBytes = await _downloadBytesWithRedirects(
            Uri.parse(sourceUrl),
          );

          final actualTarHash = sha256.convert(tarGzBytes).toString();
          const expectedTarHash =
              '6761af1d9f5d353ab4f0b7497be2643313b36c8f31caec0144bfef198e71e6ab';
          if (actualTarHash != expectedTarHash) {
            throw StateError(
              'SHA-256 mismatch for OpenBLAS archive: expected $expectedTarHash, got $actualTarHash',
            );
          }

          final unzippedBytes = GZipDecoder().decodeBytes(tarGzBytes);
          final archive = TarDecoder().decodeBytes(unzippedBytes);

          final safeOutputPrefix =
              sharedOpenblasBase.path.endsWith(Platform.pathSeparator)
              ? sharedOpenblasBase.path
              : '${sharedOpenblasBase.path}${Platform.pathSeparator}';
          for (final file in archive) {
            final outPath = sharedOpenblasBase.uri
                .resolve(file.name)
                .toFilePath();
            if (!outPath.startsWith(safeOutputPrefix)) {
              throw FormatException(
                'Path traversal attempt in OpenBLAS archive: ${file.name}',
              );
            }
            if (file.isFile) {
              final outFile = File(outPath);
              outFile.createSync(recursive: true);
              outFile.writeAsBytesSync(file.content as List<int>, flush: true);
            } else {
              Directory(outPath).createSync(recursive: true);
            }
          }

          print('Building OpenBLAS with target $openBlasTarget...');
          final makeArgs = <String>[
            'shared',
            '-j${Platform.numberOfProcessors}',
            'TARGET=$openBlasTarget',
            if (arch == Architecture.x64) ...[
              'DYNAMIC_ARCH=1',
              'DYNAMIC_LIST=NEHALEM SANDYBRIDGE HASWELL SKYLAKEX ZEN',
            ],
            'USE_THREAD=1',
            'FIXED_LIBNAME=1',
            if (os != OS.current || arch != Architecture.current)
              OS.current == OS.macOS ? 'HOSTCC=clang' : 'HOSTCC=gcc',
          ];

          if (cCompiler != null) {
            makeArgs.add('CC=${cCompiler.compiler.toFilePath()}');
            makeArgs.add('AR=${cCompiler.archiver.toFilePath()}');
          }

          await Process.run('chmod', [
            '-R',
            '+x',
            '.',
          ], workingDirectory: extractDir);

          final buildResult = await Process.run(
            'make',
            makeArgs,
            workingDirectory: extractDir,
          );
          if (buildResult.exitCode != 0) {
            throw StateError(
              'Failed to build OpenBLAS (exit ${buildResult.exitCode}):\n'
              '${buildResult.stderr}',
            );
          }
        }

        final extLibFile = File(
          outputDir.uri.resolve(_canonicalExtensionsName(os)).toFilePath(),
        );
        final compilerPath =
            cCompiler?.compiler.toFilePath() ??
            (os == OS.windows ? 'cl' : 'cc');
        final compilerLower = compilerPath.toLowerCase();
        final isMSVC =
            os == OS.windows &&
            (compilerLower.endsWith('cl.exe') || compilerLower == 'cl') &&
            !compilerLower.contains('clang');

        final compileArgs = isMSVC
            ? [
                '/LD',
                '/O2',
                '/EHsc',
                '/I${extractDir}lapack-netlib/LAPACKE/include',
                customExtensionsPath,
                stampFile.path,
                '/Fe:${extLibFile.path}',
                '/link',
                '/LIBPATH:$extractDir',
                'libopenblas.lib',
                '/EXPORT:openblas_embedded_source_hash',
              ]
            : [
                '-shared',
                '-fPIC',
                '-O3',
                if (os == OS.android) '-Wl,-z,max-page-size=16384',
                '-I${extractDir}lapack-netlib/LAPACKE/include',
                customExtensionsPath,
                stampFile.path,
                '-o',
                extLibFile.path,
                '-L$extractDir',
                '-Wl,-rpath,\$ORIGIN',
                '-lopenblas',
                '-lm',
              ];

        final extRes = await Process.run(compilerPath, compileArgs);
        if (extRes.exitCode != 0) {
          throw StateError(
            'Failed to compile custom extensions: ${extRes.stderr}',
          );
        }

        return (openblasUri: libFile.uri, extensionsUri: extLibFile.uri);
    }
  }

  @override
  List<Uri> get dependencies => [
    for (final file in nativeSourceFiles(_root)) file.uri,
  ];
}

sealed class OpenBlasBinary {
  OpenBlasBinary._();

  factory OpenBlasBinary.forBuild(BuildInput input) {
    if (input.config.code.targetOS == OS.macOS ||
        input.config.code.targetOS == OS.iOS) {
      return MacosAccelerateBinary();
    }
    if (input.config.code.targetOS == OS.windows) {
      return PrecompiledBinary();
    }
    return CompileOpenBlas(
      'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/OpenBLAS-0.3.33.tar.gz',
    );
  }
}

final class MacosAccelerateBinary extends OpenBlasBinary {
  MacosAccelerateBinary() : super._();
}

final class PrecompiledBinary extends OpenBlasBinary {
  PrecompiledBinary() : super._();
}

final class CompileOpenBlas extends OpenBlasBinary {
  final String sourceUrl;
  CompileOpenBlas(this.sourceUrl) : super._();
}

Future<Uint8List> _downloadBytesWithRedirects(Uri url) async {
  final client = HttpClient();
  try {
    var currentUrl = url;
    for (var redirectCount = 0; redirectCount < 5; redirectCount++) {
      final request = await client.getUrl(currentUrl);
      final response = await request.close();
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.value(HttpHeaders.locationHeader) != null) {
        currentUrl = currentUrl.resolve(
          response.headers.value(HttpHeaders.locationHeader)!,
        );
        continue;
      }
      if (response.statusCode != 200) {
        throw HttpException(
          'Failed to download $currentUrl (HTTP ${response.statusCode})',
        );
      }
      final builder = BytesBuilder(copy: false);
      await for (final chunk in response) {
        builder.add(chunk);
      }
      return builder.takeBytes();
    }
    throw HttpException('Too many redirects while downloading $url');
  } finally {
    client.close();
  }
}

Future<Map<String, String>> getMSVCEnvironment(Architecture targetArch) async {
  if (!Platform.isWindows) return {};

  String vswherePath = 'vswhere.exe';
  final programFilesX86 =
      Platform.environment['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
  final defaultVswhere =
      '$programFilesX86\\Microsoft Visual Studio\\Installer\\vswhere.exe';
  if (await File(defaultVswhere).exists()) {
    vswherePath = defaultVswhere;
  }

  try {
    final vswhereRes = await Process.run(vswherePath, [
      '-latest',
      '-property',
      'installationPath',
    ]);
    if (vswhereRes.exitCode != 0) return {};

    final vsPath = vswhereRes.stdout.toString().trim();
    if (vsPath.isEmpty) return {};

    final vcvarsPath = '$vsPath\\VC\\Auxiliary\\Build\\vcvarsall.bat';
    if (!await File(vcvarsPath).exists()) return {};

    final vcvarsArch = targetArch == Architecture.arm64
        ? 'arm64'
        : (targetArch == Architecture.ia32 ? 'x86' : 'amd64');

    final tempDir = Directory.systemTemp;
    final tempFile = File(
      '${tempDir.path}\\get_msvc_env_${DateTime.now().millisecondsSinceEpoch}.bat',
    );
    await tempFile.writeAsString(
      '@echo off\ncall "$vcvarsPath" $vcvarsArch\nset\n',
    );
    final envRes = await Process.run('cmd.exe', ['/c', tempFile.path]);
    try {
      await tempFile.delete();
    } catch (_) {}

    if (envRes.exitCode != 0) return {};

    final envMap = <String, String>{};
    for (final line in envRes.stdout.toString().split('\n')) {
      final parts = line.split('=');
      if (parts.length >= 2) {
        final key = parts[0].trim();
        final value = parts.sublist(1).join('=').trim();
        if (key.isNotEmpty) {
          envMap[key] = value;
        }
      }
    }
    return envMap;
  } catch (_) {
    return {};
  }
}
