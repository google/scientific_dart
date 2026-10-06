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

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:ndarray/src/hook_helpers/build_options.dart';
import 'package:ndarray/src/hook_helpers/hashes.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final BuildOptions buildOptions;
    try {
      buildOptions = BuildOptions.fromDefines(input.userDefines);
    } catch (e) {
      throw ArgumentError.value(
        input.userDefines,
        'userDefines',
        BuildOptions.usageError(e),
      );
    }
    print('ndarray build options: $buildOptions');

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
      BuildModeEnum.local => LocalMode(input, buildOptions.localPath),
      BuildModeEnum.source => SourceMode(
        input,
        buildOptions.checkoutPath,
        buildOptions,
      ),
    };

    final requirePrebuilt =
        Platform.environment['NDARRAY_REQUIRE_PREBUILT'] == '1';

    Uri builtLibrary;
    if (buildOptions.buildMode == BuildModeEnum.fetch &&
        !buildOptions.isExplicit &&
        !requirePrebuilt &&
        currentSourceHash != nativeSourceHash) {
      print(
        'Prebuilt ndarray binary for release $version differs from local '
        'native sources in hook/; falling back to `buildMode: source`.',
      );
      buildMode = SourceMode(input, buildOptions.checkoutPath, buildOptions);
      builtLibrary = await buildMode.build();
    } else {
      try {
        builtLibrary = await buildMode.build();
      } catch (e) {
        if (buildOptions.buildMode == BuildModeEnum.fetch &&
            !buildOptions.isExplicit &&
            !requirePrebuilt) {
          print(
            'Prebuilt ndarray binary unavailable ($e); '
            'falling back to `buildMode: source`.',
          );
          buildMode = SourceMode(
            input,
            buildOptions.checkoutPath,
            buildOptions,
          );
          builtLibrary = await buildMode.build();
        } else {
          rethrow;
        }
      }
    }

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'ndarray',
        linkMode: DynamicLoadingBundled(),
        file: builtLibrary,
      ),
    );

    final cpuCheckUri = await _buildCpuCheckLibrary(
      input,
      buildOptions,
      isSourceBuild: buildMode is SourceMode,
    );
    if (cpuCheckUri != null) {
      output.assets.code.add(
        CodeAsset(
          package: input.packageName,
          name: 'ndarray_cpu_check',
          linkMode: DynamicLoadingBundled(),
          file: cpuCheckUri,
        ),
      );
    }

    output.dependencies.addAll(buildMode.dependencies);
    output.dependencies.add(input.packageRoot.resolve('pubspec.yaml'));
  });
}

String _canonicalLibName(OS os) => os == OS.windows
    ? 'libndarray.dll'
    : ((os == OS.macOS || os == OS.iOS) ? 'libndarray.dylib' : 'libndarray.so');

String _canonicalCpuCheckLibName(OS os) => os == OS.windows
    ? 'libndarray_cpu_check.dll'
    : ((os == OS.macOS || os == OS.iOS)
          ? 'libndarray_cpu_check.dylib'
          : 'libndarray_cpu_check.so');

int _requiredX86FeatureMask(
  Architecture arch,
  BuildOptions options, {
  required bool isMSVC,
  required bool isSourceBuild,
}) {
  if (arch != Architecture.x64) return 0;
  if (!isSourceBuild) return 0x3F;
  final flags = options.effectiveX86Flags(isMSVC: isMSVC);
  if (flags.isEmpty) return 0;
  var mask = 0;
  for (final flag in flags) {
    final lower = flag.toLowerCase();
    if (lower == '/arch:avx2' || lower == '-arch:avx2') {
      mask |= 0x3F;
    } else if (lower == '-mavx2') {
      mask |= 0x01 | 0x02 | 0x04 | 0x20;
    } else if (lower == '-mavx' ||
        lower == '/arch:avx' ||
        lower == '-arch:avx') {
      mask |= 0x01 | 0x02 | 0x04;
    } else if (lower == '-mfma') {
      mask |= 0x01 | 0x02 | 0x08;
    } else if (lower == '-mf16c') {
      mask |= 0x01 | 0x02 | 0x10;
    }
  }
  return mask;
}

Future<T> _withSharedLock<T>(
  Directory sharedDir,
  Future<T> Function() action,
) async {
  if (!sharedDir.existsSync()) {
    sharedDir.createSync(recursive: true);
  }
  final lockFile = File.fromUri(sharedDir.uri.resolve('.build.lock'));
  final raf = lockFile.openSync(mode: FileMode.write);
  try {
    raf.lockSync(FileLock.blockingExclusive);
    return await action();
  } finally {
    try {
      raf.unlockSync();
    } catch (_) {}
    try {
      raf.closeSync();
    } catch (_) {}
  }
}

Future<Uri?> _buildCpuCheckLibrary(
  BuildInput input,
  BuildOptions buildOptions, {
  required bool isSourceBuild,
}) async {
  final os = input.config.code.targetOS;
  final arch = input.config.code.targetArchitecture;
  final cCompiler = input.config.code.cCompiler;

  final compilerPath =
      cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');
  final compilerLower = compilerPath.toLowerCase();
  final isClangCl = compilerLower.contains('clang-cl');
  final isGNU =
      !isClangCl &&
      (compilerLower.contains('gcc') ||
          compilerLower.contains('clang') ||
          compilerLower.contains('g++'));
  final isMSVC =
      isClangCl ||
      (os == OS.windows &&
          (!isGNU ||
              compilerLower.endsWith('cl.exe') ||
              compilerLower == 'cl' ||
              compilerLower.contains('msvc')));

  final requiredMask = _requiredX86FeatureMask(
    arch,
    buildOptions,
    isMSVC: isMSVC,
    isSourceBuild: isSourceBuild,
  );

  final sharedDir = Directory.fromUri(
    input.outputDirectoryShared.resolve(
      'ndarray-cpu-check-${os.name}-${arch.name}-m$requiredMask/',
    ),
  );
  final libName = _canonicalCpuCheckLibName(os);
  final sharedLib = File.fromUri(sharedDir.uri.resolve(libName));
  final outputDir = Directory.fromUri(input.outputDirectory);
  if (!outputDir.existsSync()) {
    outputDir.createSync(recursive: true);
  }
  final outLib = File.fromUri(outputDir.uri.resolve(libName));

  final builtOk = await _withSharedLock(sharedDir, () async {
    if (sharedLib.existsSync()) return true;
    final srcFile = File.fromUri(sharedDir.uri.resolve('cpu_check.c'));
    await srcFile.writeAsString('''
#include <stdint.h>

#if defined(_WIN32)
#define CPU_CHECK_EXPORT __declspec(dllexport)
#else
#define CPU_CHECK_EXPORT __attribute__((visibility("default"), used))
#endif

#ifndef NDARRAY_REQUIRED_X86_FEATURES
#define NDARRAY_REQUIRED_X86_FEATURES $requiredMask
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#include <immintrin.h>
#endif

CPU_CHECK_EXPORT int32_t ndarray_x86_cpu_features(void) {
#if defined(__x86_64__) || defined(_M_X64)
  uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER) && !defined(__clang__)
  int cpu_info[4] = {0, 0, 0, 0};
  __cpuid(cpu_info, 0);
  uint32_t max_leaf = (uint32_t)cpu_info[0];
  if (max_leaf < 1) return 0;
  __cpuidex(cpu_info, 1, 0);
  eax = (uint32_t)cpu_info[0];
  ebx = (uint32_t)cpu_info[1];
  ecx = (uint32_t)cpu_info[2];
  edx = (uint32_t)cpu_info[3];
#else
  uint32_t max_leaf = 0;
  __asm__ volatile("cpuid"
                   : "=a"(max_leaf), "=b"(ebx), "=c"(ecx), "=d"(edx)
                   : "a"(0), "c"(0));
  if (max_leaf < 1) return 0;
  __asm__ volatile("cpuid"
                   : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                   : "a"(1), "c"(0));
#endif
  int32_t mask = 0;
  int has_osxsave = ((ecx >> 27) & 1u) != 0;
  int has_avx = ((ecx >> 28) & 1u) != 0;
  int has_fma = ((ecx >> 12) & 1u) != 0;
  int has_f16c = ((ecx >> 29) & 1u) != 0;
  if (has_osxsave) mask |= 0x01;
  if (has_avx) mask |= 0x04;
  if (has_fma) mask |= 0x08;
  if (has_f16c) mask |= 0x10;

  if (has_osxsave) {
    uint32_t xcr0_lo = 0;
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned __int64 xcr0 = _xgetbv(0);
    xcr0_lo = (uint32_t)xcr0;
#else
    uint32_t xcr0_hi = 0;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
                     : "=a"(xcr0_lo), "=d"(xcr0_hi)
                     : "c"(0));
#endif
    if ((xcr0_lo & 0x6u) == 0x6u) {
      mask |= 0x02;
    }
  }

  if (max_leaf >= 7) {
#if defined(_MSC_VER) && !defined(__clang__)
    __cpuidex(cpu_info, 7, 0);
    ebx = (uint32_t)cpu_info[1];
#else
    __asm__ volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(7), "c"(0));
#endif
    if (((ebx >> 5) & 1u) != 0) {
      mask |= 0x20;
    }
  }
  return mask;
#else
  return 0x3F;
#endif
}

CPU_CHECK_EXPORT int32_t ndarray_x86_required_features(void) {
  return (int32_t)(NDARRAY_REQUIRED_X86_FEATURES);
}
''');

    final tempLib = File(
      '${sharedLib.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    final runEnv = <String, String>{
      ...Platform.environment,
      if (isMSVC) ...await getMSVCEnvironment(arch),
    };
    final compileArgs = isMSVC
        ? <String>[
            '/LD',
            '/MD',
            '/O2',
            '/DNDARRAY_REQUIRED_X86_FEATURES=$requiredMask',
            srcFile.path,
            '/Fe:${tempLib.path}',
            '/link',
            '/EXPORT:ndarray_x86_cpu_features',
            '/EXPORT:ndarray_x86_required_features',
          ]
        : <String>[
            if (os == OS.macOS || os == OS.iOS) ...[
              '-arch',
              arch == Architecture.arm64 ? 'arm64' : 'x86_64',
              '-Wl,-install_name,@rpath/$libName',
              '-Wl,-headerpad_max_install_names',
            ],
            '-shared',
            '-fPIC',
            '-O2',
            if (arch == Architecture.x64) ...[
              '-mno-avx',
              '-mno-avx2',
              '-mno-fma',
              '-mno-f16c',
            ],
            '-DNDARRAY_REQUIRED_X86_FEATURES=$requiredMask',
            if (os == OS.android) '-Wl,-z,max-page-size=16384',
            srcFile.path,
            '-o',
            tempLib.path,
          ];

    try {
      final res = await Process.run(
        compilerPath,
        compileArgs,
        environment: runEnv,
      );
      if (res.exitCode != 0) {
        if (tempLib.existsSync()) {
          try {
            tempLib.deleteSync();
          } catch (_) {}
        }
        if (isSourceBuild) {
          throw StateError(
            'Failed to compile ndarray_cpu_check helper (exit ${res.exitCode}):\n'
            'stdout: ${res.stdout}\nstderr: ${res.stderr}',
          );
        }
        return false;
      }
      await tempLib.rename(sharedLib.path);
      return true;
    } catch (_) {
      if (tempLib.existsSync()) {
        try {
          tempLib.deleteSync();
        } catch (_) {}
      }
      if (isSourceBuild) rethrow;
      return false;
    }
  });

  if (!builtOk) return null;
  final tempOutLib = File(
    '${outLib.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
  );
  await sharedLib.copy(tempOutLib.path);
  await tempOutLib.rename(outLib.path);
  return outLib.uri;
}

sealed class BuildMode {
  final BuildInput input;

  const BuildMode(this.input);

  List<Uri> get dependencies;

  Future<Uri> build();
}

final class FetchMode extends BuildMode {
  FetchMode(super.input);

  @override
  Future<Uri> build() async {
    final currentSourceHash = computeNativeSourceHash(input.packageRoot);
    if (currentSourceHash != nativeSourceHash) {
      throw StateError(
        'Prebuilt ndarray binary for release $version is out of date with native sources in hook/!\n'
        'Pinned nativeSourceHash: $nativeSourceHash\n'
        'Current hook/ hash:      $currentSourceHash\n'
        'If you are the package author, build and attest new release artifacts (.github/workflows/artifacts.yml) and run:\n'
        '  dart tool/regenerate_hashes.dart <new-release-tag>\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` while developing native code.')}',
      );
    }

    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final artifactName = ndarrayArtifactName(os, arch);
    final expectedHash = fileHashes[(os, arch)];

    if (expectedHash == null || expectedHash.startsWith('00000000')) {
      throw StateError(
        'No prebuilt ndarray binary hash is pinned for ($os, $arch) in release $version.\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` or `buildMode: local`.')}',
      );
    }

    final libName = _canonicalLibName(os);
    final sharedCacheDir = Directory.fromUri(
      input.outputDirectoryShared.resolve(
        'ndarray-$version/${os.name}-${arch.name}/',
      ),
    );
    final cachedLibrary = File.fromUri(sharedCacheDir.uri.resolve(libName));

    final cachedUri = await _withSharedLock(sharedCacheDir, () async {
      if (await cachedLibrary.exists()) {
        final cachedBytes = await cachedLibrary.readAsBytes();
        final cachedHash = sha256.convert(cachedBytes).toString();
        if (cachedHash == expectedHash) {
          verifyArtifactSourceHash(
            cachedBytes,
            currentSourceHash: currentSourceHash,
          );
          print('Using cached ndarray binary from ${cachedLibrary.path}.');
          return cachedLibrary.uri;
        }
      }

      final remoteUri = Uri.parse(
        'https://github.com/$repository/releases/download/$version/$artifactName',
      );
      print('Fetching prebuilt ndarray binary from $remoteUri...');
      final bytes = await _downloadBytesWithRedirects(remoteUri);
      final actualHash = sha256.convert(bytes).toString();
      if (actualHash != expectedHash) {
        throw StateError(
          'SHA-256 mismatch for prebuilt ndarray binary at $remoteUri:\n'
          'Expected: $expectedHash\n'
          'Actual:   $actualHash',
        );
      }
      verifyArtifactSourceHash(bytes, currentSourceHash: currentSourceHash);

      final tempFile = File(
        '${cachedLibrary.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
      );
      await tempFile.writeAsBytes(bytes, flush: true);
      await tempFile.rename(cachedLibrary.path);
      return cachedLibrary.uri;
    });

    final targetFile = File.fromUri(input.outputDirectory.resolve(libName));
    await targetFile.parent.create(recursive: true);
    final tempOut = File(
      '${targetFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    await File.fromUri(cachedUri).copy(tempOut.path);
    await tempOut.rename(targetFile.path);
    return targetFile.uri;
  }

  @override
  List<Uri> get dependencies => [
    for (final file in nativeSourceFiles(input.packageRoot)) file.uri,
  ];
}

final class LocalMode extends BuildMode {
  final Uri? localPath;

  LocalMode(super.input, this.localPath);

  File _resolveLocalFile() {
    if (localPath == null) {
      throw ArgumentError.value(
        localPath,
        'localPath',
        'Must be set in `hooks.user_defines.ndarray` '
            '(or `LOCAL_NDARRAY_BINARY` environment variable).',
      );
    }
    final os = input.config.code.targetOS;
    final entityPath = localPath!.toFilePath(windows: Platform.isWindows);
    if (FileSystemEntity.isDirectorySync(entityPath)) {
      final candidate = File.fromUri(
        Directory(entityPath).uri.resolve(_canonicalLibName(os)),
      );
      if (candidate.existsSync()) return candidate;
      final artifactCandidate = File.fromUri(
        Directory(entityPath).uri.resolve(
          ndarrayArtifactName(os, input.config.code.targetArchitecture),
        ),
      );
      if (artifactCandidate.existsSync()) return artifactCandidate;
      throw FileSystemException(
        'Could not find ${_canonicalLibName(os)} in localPath directory.',
        entityPath,
      );
    }
    final file = File(entityPath);
    if (!file.existsSync()) {
      throw FileSystemException(
        'Could not find local ndarray binary.',
        entityPath,
      );
    }
    return file;
  }

  @override
  Future<Uri> build() async {
    final sourceFile = _resolveLocalFile();
    final targetUri = input.outputDirectory.resolve(
      _canonicalLibName(input.config.code.targetOS),
    );
    final targetFile = File.fromUri(targetUri);
    await targetFile.parent.create(recursive: true);
    final tempFile = File(
      '${targetFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    await sourceFile.copy(tempFile.path);
    await tempFile.rename(targetFile.path);
    return targetFile.uri;
  }

  @override
  List<Uri> get dependencies => [_resolveLocalFile().uri];
}

final class SourceMode extends BuildMode {
  final Uri? checkoutPath;
  final BuildOptions? buildOptions;

  SourceMode(super.input, this.checkoutPath, [this.buildOptions]);

  Uri get _root => checkoutPath ?? input.packageRoot;

  @override
  Future<Uri> build() async {
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final cCompiler = input.config.code.cCompiler;
    final options =
        buildOptions ?? const BuildOptions(buildMode: BuildModeEnum.source);

    final libName = _canonicalLibName(os);
    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    final compilerPath =
        cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');
    final compilerLower = compilerPath.toLowerCase();
    final isClangCl = compilerLower.contains('clang-cl');
    final isGNU =
        !isClangCl &&
        (compilerLower.contains('gcc') ||
            compilerLower.contains('clang') ||
            compilerLower.contains('g++'));
    final isMSVC =
        isClangCl ||
        (os == OS.windows &&
            (!isGNU ||
                compilerLower.endsWith('cl.exe') ||
                compilerLower == 'cl' ||
                compilerLower.contains('msvc')));

    options.validateTarget(targetOS: os, targetArch: arch, isMSVC: isMSVC);

    final msvcEnv = <String, String>{
      ...Platform.environment,
      if (isMSVC) ...await getMSVCEnvironment(arch),
    };

    var cppCompilerPath = compilerPath;
    if (isMSVC && !isClangCl) {
      for (final candidate in const [
        r'C:\Program Files\LLVM\bin\clang-cl.exe',
        'clang-cl.exe',
      ]) {
        try {
          final check = await Process.run(candidate, [
            '--version',
          ], environment: msvcEnv);
          if (check.exitCode == 0) {
            cppCompilerPath = candidate;
            break;
          }
        } catch (_) {}
      }
    } else if (!isMSVC && compilerPath.endsWith('gcc')) {
      cppCompilerPath =
          '${compilerPath.substring(0, compilerPath.length - 3)}g++';
    } else if (compilerPath.endsWith('clang')) {
      cppCompilerPath = '$compilerPath++';
    } else if (compilerPath.endsWith('cc')) {
      cppCompilerPath = 'c++';
    } else if (compilerPath.contains('gcc-')) {
      cppCompilerPath = compilerPath.replaceAll('gcc-', 'g++-');
    } else if (compilerPath.contains('clang-')) {
      cppCompilerPath = compilerPath.replaceAll('clang-', 'clang++-');
    }

    final x86Flags = options.effectiveX86Flags(isMSVC: isMSVC);
    final cacheKey = options.cacheKey;
    final hwyCacheKey = options.x86Flags == null
        ? ''
        : '-${sha256.convert(options.x86Flags!.codeUnits).toString().substring(0, 12)}';
    final currentSourceHash = computeNativeSourceHash(_root);

    final highwayDir = _root.resolve('third_party/highway/');
    final legacyHwyDir = Directory.fromUri(outputDir.uri.resolve('hwy_build'));
    final highwayBuildDir =
        (!isMSVC && hwyCacheKey.isEmpty && legacyHwyDir.existsSync())
        ? legacyHwyDir
        : Directory.fromUri(
            input.outputDirectoryShared.resolve(
              isMSVC
                  ? 'hwy_build_v2-${os.name}-${arch.name}$hwyCacheKey/'
                  : 'hwy_build-${os.name}-${arch.name}$hwyCacheKey/',
            ),
          );

    final String hwyLibName = isMSVC ? 'hwy.lib' : 'libhwy.a';
    final String hwyContribLibName = isMSVC
        ? 'hwy_contrib.lib'
        : 'libhwy_contrib.a';

    File resolveHwyLib(String name) {
      final direct = File.fromUri(highwayBuildDir.uri.resolve(name));
      if (direct.existsSync()) return direct;
      final release = File.fromUri(
        highwayBuildDir.uri.resolve('Release/$name'),
      );
      if (release.existsSync()) return release;
      return direct;
    }

    await _withSharedLock(highwayBuildDir, () async {
      if (!resolveHwyLib(hwyLibName).existsSync() ||
          !resolveHwyLib(hwyContribLibName).existsSync()) {
        print('Highway static libraries not found. Compiling highway...');
        if (!highwayBuildDir.existsSync()) {
          highwayBuildDir.createSync(recursive: true);
        }

        final cmakeCCompiler =
            (isMSVC && cppCompilerPath.toLowerCase().contains('clang-cl'))
            ? cppCompilerPath.replaceAll('\\', '/')
            : compilerPath.replaceAll('\\', '/');
        final cmakeCxxCompiler = cppCompilerPath.replaceAll('\\', '/');

        final cmakeRes = await Process.run(
          'cmake',
          [
            if (isMSVC) ...['-G', 'NMake Makefiles'],
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CXX_STANDARD=17',
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
            '-DHWY_ENABLE_TESTS=OFF',
            '-DHWY_ENABLE_EXAMPLES=OFF',
            if (arch == Architecture.x64 && x86Flags.isNotEmpty)
              '-DCMAKE_CXX_FLAGS=${x86Flags.join(' ')}',
            if (cCompiler != null ||
                (isMSVC &&
                    cppCompilerPath.toLowerCase().contains('clang-cl'))) ...[
              '-DCMAKE_C_COMPILER=$cmakeCCompiler',
              '-DCMAKE_CXX_COMPILER=$cmakeCxxCompiler',
            ],
            if (os == OS.macOS || os == OS.iOS)
              '-DCMAKE_OSX_ARCHITECTURES=${arch == Architecture.arm64 ? 'arm64' : 'x86_64'}',
            highwayDir.toFilePath(),
          ],
          workingDirectory: highwayBuildDir.path,
          environment: msvcEnv,
        );

        if (cmakeRes.exitCode != 0) {
          throw StateError(
            'CMake failed for highway (exit ${cmakeRes.exitCode}):\n'
            'stdout: ${cmakeRes.stdout}\n'
            'stderr: ${cmakeRes.stderr}',
          );
        }

        final buildRes = await Process.run(
          'cmake',
          [
            '--build',
            '.',
            '--target',
            'hwy',
            'hwy_contrib',
            if (!isMSVC) '--parallel',
          ],
          workingDirectory: highwayBuildDir.path,
          environment: msvcEnv,
        );

        if (buildRes.exitCode != 0) {
          throw StateError(
            'Build failed for highway (exit ${buildRes.exitCode}):\n'
            'stdout: ${buildRes.stdout}\n'
            'stderr: ${buildRes.stderr}',
          );
        }
      }
    });

    final libhwy = resolveHwyLib(hwyLibName);
    final libhwyContrib = resolveHwyLib(hwyContribLibName);

    String computeInputDigest(String src, List<String> args) {
      final bytes = BytesBuilder(copy: false);
      bytes.add(cacheKey.codeUnits);
      bytes.add(args.join(' ').codeUnits);
      bytes.add(File(src).readAsBytesSync());
      final headerPaths = <String>{
        'hook/custom_indexing.h',
        'hook/custom_sorting.h',
        'hook/custom_ufuncs.h',
        'hook/ndarray_common.h',
        'hook/npz_io.h',
        'third_party/miniz/miniz.h',
      };
      final hookDir = Directory.fromUri(_root.resolve('hook/'));
      if (hookDir.existsSync()) {
        final dynamicHeaders =
            hookDir
                .listSync(followLinks: false)
                .whereType<File>()
                .where((f) => f.path.endsWith('.h'))
                .map((f) => 'hook/${f.uri.pathSegments.last}')
                .toList()
              ..sort();
        headerPaths.addAll(dynamicHeaders);
      }
      final sortedHeaders = headerPaths.toList()..sort();
      for (final header in sortedHeaders) {
        final hF = File(_root.resolve(header).toFilePath());
        if (hF.existsSync()) {
          bytes.add(hF.readAsBytesSync());
        }
      }
      return sha256.convert(bytes.takeBytes()).toString();
    }

    final ufuncsSrc = _root.resolve('hook/custom_ufuncs.cpp').toFilePath();
    final sortingSrc = _root.resolve('hook/custom_sorting.cpp').toFilePath();
    final indexingSrc = _root.resolve('hook/custom_indexing.cpp').toFilePath();
    final minizSrc = _root.resolve('third_party/miniz/miniz.c').toFilePath();
    final npzIoSrc = _root.resolve('hook/npz_io.cpp').toFilePath();

    if (isMSVC) {
      final sharedObjDir = Directory.fromUri(
        input.outputDirectoryShared.resolve(
          'ndarray-objs-${os.name}-${arch.name}$cacheKey/',
        ),
      );
      if (!sharedObjDir.existsSync()) {
        sharedObjDir.createSync(recursive: true);
      }
      final ufuncsObj = sharedObjDir.uri
          .resolve('custom_ufuncs.obj')
          .toFilePath();
      final sortingObj = sharedObjDir.uri
          .resolve('custom_sorting.obj')
          .toFilePath();
      final indexingObj = sharedObjDir.uri
          .resolve('custom_indexing.obj')
          .toFilePath();
      final minizObj = sharedObjDir.uri.resolve('miniz.obj').toFilePath();
      final npzIoObj = sharedObjDir.uri.resolve('npz_io.obj').toFilePath();
      final stampSrc = sharedObjDir.uri
          .resolve('source_hash_stamp.c')
          .toFilePath();
      final stampObj = sharedObjDir.uri
          .resolve('source_hash_stamp.obj')
          .toFilePath();

      Future<bool> runMsvcCompile(
        String label,
        String src,
        String obj,
        String exe,
        List<String> args,
      ) async {
        final objF = File(obj);
        final hashF = File('$obj.sha256');
        final digest = computeInputDigest(src, args);
        if (objF.existsSync() &&
            hashF.existsSync() &&
            hashF.readAsStringSync().trim() == digest) {
          return false;
        }
        final res = await Process.run(exe, args, environment: msvcEnv);
        if (res.exitCode != 0) {
          throw StateError(
            '$label compilation failed:\nstdout: ${res.stdout}\nstderr: ${res.stderr}',
          );
        }
        await hashF.writeAsString(digest);
        return true;
      }

      await _withSharedLock(sharedObjDir, () async {
        await File(stampSrc).writeAsString('''
#define STAMP_EXPORT __declspec(dllexport)
static const char _ndarray_source_hash_marker[] =
    "$sourceHashMarkerPrefix$currentSourceHash";
STAMP_EXPORT const char* ndarray_embedded_source_hash(void) {
  return _ndarray_source_hash_marker;
}
''');

        var anyCompiled = false;
        if (await runMsvcCompile(
          'Ufuncs',
          ufuncsSrc,
          ufuncsObj,
          cppCompilerPath,
          [
            '/c',
            '/std:c++17',
            '/bigobj',
            '/O2',
            '/MD',
            '/EHsc',
            '/Zc:strictStrings',
            if (arch == Architecture.x64) ...x86Flags,
            '/D_USE_MATH_DEFINES',
            '/DNOMINMAX',
            '/DVECTORIZED_TARGETS=',
            '/I${_root.toFilePath()}',
            ufuncsSrc,
            '/Fo:$ufuncsObj',
          ],
        )) {
          anyCompiled = true;
        }
        if (await runMsvcCompile(
          'Sorting',
          sortingSrc,
          sortingObj,
          cppCompilerPath,
          [
            '/c',
            '/std:c++17',
            '/bigobj',
            '/O2',
            '/MD',
            '/EHsc',
            '/Zc:strictStrings',
            if (arch == Architecture.x64) ...x86Flags,
            '/D_USE_MATH_DEFINES',
            '/DNOMINMAX',
            '/I${_root.toFilePath()}',
            '/I${_root.resolve('third_party/highway/').toFilePath()}',
            sortingSrc,
            '/Fo:$sortingObj',
          ],
        )) {
          anyCompiled = true;
        }
        if (await runMsvcCompile(
          'Indexing',
          indexingSrc,
          indexingObj,
          cppCompilerPath,
          [
            '/c',
            '/std:c++17',
            '/bigobj',
            '/O2',
            '/MD',
            '/EHsc',
            '/Zc:strictStrings',
            if (arch == Architecture.x64) ...x86Flags,
            '/D_USE_MATH_DEFINES',
            '/DNOMINMAX',
            '/I${_root.toFilePath()}',
            indexingSrc,
            '/Fo:$indexingObj',
          ],
        )) {
          anyCompiled = true;
        }
        if (await runMsvcCompile('miniz', minizSrc, minizObj, compilerPath, [
          '/c',
          '/O2',
          '/MD',
          '/I${_root.toFilePath()}',
          minizSrc,
          '/Fo:$minizObj',
        ])) {
          anyCompiled = true;
        }
        if (await runMsvcCompile(
          'npz_io',
          npzIoSrc,
          npzIoObj,
          cppCompilerPath,
          [
            '/c',
            '/std:c++17',
            '/bigobj',
            '/O2',
            '/MD',
            '/EHsc',
            '/Zc:strictStrings',
            '/DNOMINMAX',
            '/I${_root.toFilePath()}',
            npzIoSrc,
            '/Fo:$npzIoObj',
          ],
        )) {
          anyCompiled = true;
        }
        if (await runMsvcCompile(
          'source_hash_stamp',
          stampSrc,
          stampObj,
          compilerPath,
          ['/c', '/O2', '/MD', stampSrc, '/Fo:$stampObj'],
        )) {
          anyCompiled = true;
        }

        final libUri = libFile.uri;
        if (anyCompiled || !File.fromUri(libUri).existsSync()) {
          final defFile = await _generateWindowsDefFile(_root, outputDir);
          final tempLibFile = File(
            '${libFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
          );

          final res = await Process.run(cppCompilerPath, [
            '/LD',
            '/MD',
            ufuncsObj,
            sortingObj,
            indexingObj,
            minizObj,
            npzIoObj,
            stampObj,
            libhwyContrib.path,
            libhwy.path,
            '/Fe:${tempLibFile.path}',
            '/link',
            '/def:${defFile.path}',
          ], environment: msvcEnv);
          if (res.exitCode != 0) {
            if (tempLibFile.existsSync()) {
              try {
                tempLibFile.deleteSync();
              } catch (_) {}
            }
            throw StateError(
              'Linking failed:\nstdout: ${res.stdout}\nstderr: ${res.stderr}',
            );
          }
          await tempLibFile.rename(libFile.path);
        }
      });
    } else {
      final sharedObjDir = Directory.fromUri(
        input.outputDirectoryShared.resolve(
          'ndarray-objs-${os.name}-${arch.name}$cacheKey/',
        ),
      );
      if (!sharedObjDir.existsSync()) {
        sharedObjDir.createSync(recursive: true);
      }
      final ufuncsObj = sharedObjDir.uri
          .resolve('custom_ufuncs.o')
          .toFilePath();
      final sortingObj = sharedObjDir.uri
          .resolve('custom_sorting.o')
          .toFilePath();
      final indexingObj = sharedObjDir.uri
          .resolve('custom_indexing.o')
          .toFilePath();
      final minizObj = sharedObjDir.uri.resolve('miniz.o').toFilePath();
      final npzIoObj = sharedObjDir.uri.resolve('npz_io.o').toFilePath();
      final stampSrc = sharedObjDir.uri
          .resolve('source_hash_stamp.c')
          .toFilePath();
      final stampObj = sharedObjDir.uri
          .resolve('source_hash_stamp.o')
          .toFilePath();

      final sanitizeFlags = options.sanitizeFlags;
      final coverageFlags = options.coverageFlags;
      const cppHardeningFlags = <String>[
        '-Wall',
        '-Wextra',
        '-Wundef',
        '-Wformat=2',
        '-fno-strict-aliasing',
        '-Wno-unused-parameter',
        '-Wno-unused-function',
      ];

      Future<bool> compileIfNeeded(
        String label,
        String src,
        String obj,
        String exe,
        List<String> args,
      ) async {
        final objF = File(obj);
        final hashF = File('$obj.sha256');
        final digest = computeInputDigest(src, args);
        if (objF.existsSync() &&
            hashF.existsSync() &&
            hashF.readAsStringSync().trim() == digest) {
          return false;
        }
        final res = await Process.run(exe, args);
        if (res.exitCode != 0) {
          throw StateError('$label compilation failed: ${res.stderr}');
        }
        await hashF.writeAsString(digest);
        return true;
      }

      await _withSharedLock(sharedObjDir, () async {
        await File(stampSrc).writeAsString('''
#define STAMP_EXPORT __attribute__((visibility("default"), used))
#define STAMP_USED __attribute__((used))
STAMP_USED static const char _ndarray_source_hash_marker[] =
    "$sourceHashMarkerPrefix$currentSourceHash";
STAMP_EXPORT const char* ndarray_embedded_source_hash(void) {
  return _ndarray_source_hash_marker;
}
''');

        final compiledAny = await Future.wait([
          compileIfNeeded('Ufuncs', ufuncsSrc, ufuncsObj, cppCompilerPath, [
            if (os == OS.macOS || os == OS.iOS) ...[
              '-arch',
              arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            ],
            '-std=c++17',
            '-c',
            '-fPIC',
            '-O2',
            '-fno-exceptions',
            ...cppHardeningFlags,
            ...sanitizeFlags,
            ...coverageFlags,
            if (arch == Architecture.x64) ...x86Flags,
            '-DVECTORIZED_TARGETS=',
            '-fno-math-errno',
            '-I${_root.toFilePath()}',
            ufuncsSrc,
            '-o',
            ufuncsObj,
          ]),
          compileIfNeeded('Sorting', sortingSrc, sortingObj, cppCompilerPath, [
            if (os == OS.macOS || os == OS.iOS) ...[
              '-arch',
              arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            ],
            '-std=c++17',
            '-c',
            '-fPIC',
            '-O2',
            '-fno-exceptions',
            ...cppHardeningFlags,
            ...sanitizeFlags,
            ...coverageFlags,
            if (arch == Architecture.x64) ...x86Flags,
            '-fno-math-errno',
            '-I${_root.toFilePath()}',
            '-I${_root.resolve('third_party/highway/').toFilePath()}',
            sortingSrc,
            '-o',
            sortingObj,
          ]),
          compileIfNeeded(
            'Indexing',
            indexingSrc,
            indexingObj,
            cppCompilerPath,
            [
              if (os == OS.macOS || os == OS.iOS) ...[
                '-arch',
                arch == Architecture.arm64 ? 'arm64' : 'x86_64',
              ],
              '-std=c++17',
              '-c',
              '-fPIC',
              '-O2',
              '-fno-exceptions',
              ...cppHardeningFlags,
              ...sanitizeFlags,
              ...coverageFlags,
              if (arch == Architecture.x64) ...x86Flags,
              '-fno-math-errno',
              '-I${_root.toFilePath()}',
              indexingSrc,
              '-o',
              indexingObj,
            ],
          ),
          compileIfNeeded('miniz', minizSrc, minizObj, compilerPath, [
            if (os == OS.macOS || os == OS.iOS) ...[
              '-arch',
              arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            ],
            '-c',
            '-fPIC',
            '-O3',
            '-fno-strict-aliasing',
            ...sanitizeFlags,
            if (sanitizeFlags.isNotEmpty) '-fno-sanitize=alignment',
            ...coverageFlags,
            '-I${_root.toFilePath()}',
            minizSrc,
            '-o',
            minizObj,
          ]),
          compileIfNeeded('npz_io', npzIoSrc, npzIoObj, cppCompilerPath, [
            if (os == OS.macOS || os == OS.iOS) ...[
              '-arch',
              arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            ],
            '-std=c++17',
            '-c',
            '-fPIC',
            '-O3',
            '-fno-exceptions',
            ...cppHardeningFlags,
            ...sanitizeFlags,
            ...coverageFlags,
            '-I${_root.toFilePath()}',
            npzIoSrc,
            '-o',
            npzIoObj,
          ]),
          compileIfNeeded(
            'source_hash_stamp',
            stampSrc,
            stampObj,
            compilerPath,
            [
              if (os == OS.macOS || os == OS.iOS) ...[
                '-arch',
                arch == Architecture.arm64 ? 'arm64' : 'x86_64',
              ],
              '-c',
              '-fPIC',
              '-O2',
              stampSrc,
              '-o',
              stampObj,
            ],
          ),
        ]);

        final tempLibFile = File(
          '${libFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
        );
        final linkArgs = <String>[
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            '-Wl,-install_name,@rpath/$libName',
            '-Wl,-headerpad_max_install_names',
          ],
          '-shared',
          '-fPIC',
          ...sanitizeFlags,
          ...coverageFlags,
          if (os == OS.android) '-Wl,-z,max-page-size=16384',
          ufuncsObj,
          sortingObj,
          indexingObj,
          minizObj,
          npzIoObj,
          stampObj,
          libhwyContrib.path,
          libhwy.path,
          '-o',
          tempLibFile.path,
          if (os != OS.windows) '-lm',
        ];

        final linkHashFile = File('${libFile.path}.sha256');
        final linkDigestBuilder = BytesBuilder(copy: false)
          ..add(cacheKey.codeUnits)
          ..add(currentSourceHash.codeUnits)
          ..add(libFile.path.codeUnits)
          ..add(sanitizeFlags.join(' ').codeUnits)
          ..add(coverageFlags.join(' ').codeUnits);
        for (final objPath in [
          ufuncsObj,
          sortingObj,
          indexingObj,
          minizObj,
          npzIoObj,
          stampObj,
        ]) {
          final hFile = File('$objPath.sha256');
          if (hFile.existsSync()) {
            linkDigestBuilder.add(hFile.readAsBytesSync());
          }
        }
        final linkDigest = sha256
            .convert(linkDigestBuilder.takeBytes())
            .toString();

        final needsLink =
            !libFile.existsSync() ||
            compiledAny.any((c) => c) ||
            !linkHashFile.existsSync() ||
            linkHashFile.readAsStringSync().trim() != linkDigest;

        if (needsLink) {
          final res = await Process.run(cppCompilerPath, linkArgs);
          if (res.exitCode != 0) {
            if (tempLibFile.existsSync()) {
              try {
                tempLibFile.deleteSync();
              } catch (_) {}
            }
            throw StateError('Linking failed: ${res.stderr}');
          }
          await tempLibFile.rename(libFile.path);
          await linkHashFile.writeAsString(linkDigest);
        }
      });

      if (options.hasInstrumentation) {
        _verifyInstrumentationSymbols(libFile, options);
      }
    }

    return libFile.uri;
  }

  void _verifyInstrumentationSymbols(File libFile, BuildOptions options) {
    final content = String.fromCharCodes(libFile.readAsBytesSync());
    final effSanitize = options.effectiveSanitize ?? '';
    if (effSanitize.contains('address') && !content.contains('__asan')) {
      throw StateError(
        'Sanitizer "address" was requested for package:ndarray, but the '
        'linked binary (${libFile.path}) contains no `__asan` symbols.',
      );
    }
    if (effSanitize.contains('undefined') && !content.contains('__ubsan')) {
      throw StateError(
        'Sanitizer "undefined" was requested for package:ndarray, but the '
        'linked binary (${libFile.path}) contains no `__ubsan` symbols.',
      );
    }
    if (options.coverage &&
        !content.contains('__gcov') &&
        !content.contains('llvm_gcda')) {
      throw StateError(
        'Native coverage was requested for package:ndarray, but the '
        'linked binary (${libFile.path}) contains no gcov/llvm-cov symbols.',
      );
    }
  }

  @override
  List<Uri> get dependencies => [
    for (final file in nativeSourceFiles(_root)) file.uri,
    _root.resolve('third_party/miniz/miniz.c'),
    _root.resolve('third_party/miniz/miniz.h'),
    ..._highwayDependencies(_root),
  ];
}

List<Uri> _highwayDependencies(Uri root) {
  final result = <Uri>[];
  final cmakeFile = File.fromUri(
    root.resolve('third_party/highway/CMakeLists.txt'),
  );
  if (cmakeFile.existsSync()) {
    result.add(cmakeFile.uri);
  }
  final hwyDir = Directory.fromUri(root.resolve('third_party/highway/hwy/'));
  if (hwyDir.existsSync()) {
    final files =
        hwyDir
            .listSync(recursive: true, followLinks: false)
            .whereType<File>()
            .where((f) {
              final p = f.path;
              if (p.contains('/tests/') ||
                  p.contains(r'\tests\') ||
                  p.endsWith('_test.cc')) {
                return false;
              }
              return p.endsWith('.h') ||
                  p.endsWith('.cc') ||
                  p.endsWith('.cpp') ||
                  p.endsWith('.inc');
            })
            .toList()
          ..sort((a, b) => a.path.compareTo(b.path));
    for (final f in files) {
      result.add(f.uri);
    }
  }
  return result;
}

Future<Uint8List> _downloadBytesWithRedirects(Uri url) async {
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 30);
  try {
    var currentUrl = url;
    for (var redirectCount = 0; redirectCount < 5; redirectCount++) {
      final request = await client
          .getUrl(currentUrl)
          .timeout(const Duration(seconds: 60));
      final response = await request.close().timeout(
        const Duration(seconds: 60),
      );
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.value(HttpHeaders.locationHeader) != null) {
        final location = response.headers.value(HttpHeaders.locationHeader)!;
        await response.drain<void>().timeout(const Duration(seconds: 30));
        final nextUrl = currentUrl.resolve(location);
        if (nextUrl.scheme != 'https') {
          throw HttpException('Refusing redirect to non-HTTPS URL: $nextUrl');
        }
        currentUrl = nextUrl;
        continue;
      }
      if (response.statusCode != 200) {
        await response.drain<void>().timeout(const Duration(seconds: 30));
        throw HttpException(
          'Failed to download $currentUrl (HTTP ${response.statusCode})',
        );
      }
      final builder = BytesBuilder(copy: false);
      await for (final chunk in response.timeout(const Duration(seconds: 60))) {
        builder.add(chunk);
      }
      return builder.takeBytes();
    }
    throw HttpException('Too many redirects while downloading $url');
  } finally {
    client.close();
  }
}

Future<File> _generateWindowsDefFile(Uri root, Directory outputDir) async {
  final allExports = [
    ...extractExportsFromBindings(
      root.resolve('lib/src/ndarray_bindings.dart').toFilePath(),
    ),
    ...extractExportsFromBindings(
      root.resolve('lib/src/ndarray_extensions_bindings.dart').toFilePath(),
    ),
    'ndarray_embedded_source_hash',
  ];
  if (allExports.isEmpty) {
    throw StateError('No exported symbols found for Windows .def file.');
  }

  final defFile = File(outputDir.uri.resolve('libndarray.def').toFilePath());
  await defFile.writeAsString(
    ['LIBRARY libndarray', 'EXPORTS', ...allExports].join('\n'),
  );
  return defFile;
}

List<String> extractExportsFromBindings(String bindingsPath) {
  final file = File(bindingsPath);
  if (!file.existsSync()) {
    throw StateError(
      'Cannot generate Windows .def file: bindings file does not exist at $bindingsPath',
    );
  }

  final content = file.readAsStringSync();
  final regex = RegExp(r'external\s+[\w\d_<>.]+\s+(\w+)\s*\(');

  final exports = <String>[];
  for (final match in regex.allMatches(content)) {
    final name = match.group(1);
    if (name != null && !exports.contains(name)) {
      exports.add(name);
    }
  }
  if (exports.isEmpty) {
    throw StateError(
      'Cannot generate Windows .def file: 0 exported symbols found in $bindingsPath',
    );
  }
  return exports;
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
      '${tempDir.path}\\get_msvc_env_${DateTime.now().millisecondsSinceEpoch}_$pid.bat',
    );
    ProcessResult envRes;
    try {
      await tempFile.writeAsString(
        '@echo off\ncall "$vcvarsPath" $vcvarsArch\nset\n',
      );
      envRes = await Process.run('cmd.exe', ['/c', tempFile.path]);
    } finally {
      try {
        if (await tempFile.exists()) {
          await tempFile.delete();
        }
      } catch (_) {}
    }

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
