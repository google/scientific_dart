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
import 'package:pocketfft/src/hook_helpers/build_options.dart';
import 'package:pocketfft/src/hook_helpers/hashes.dart';

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
    print('pocketfft build options: $buildOptions');

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
        Platform.environment['POCKETFFT_REQUIRE_PREBUILT'] == '1';

    Uri builtLibrary;
    if (buildOptions.buildMode == BuildModeEnum.fetch &&
        !buildOptions.isExplicit &&
        !requirePrebuilt &&
        currentSourceHash != nativeSourceHash) {
      print(
        'Prebuilt pocketfft binary for release $version differs from local '
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
            'Prebuilt pocketfft binary unavailable ($e); '
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
        name: 'pocketfft',
        linkMode: DynamicLoadingBundled(),
        file: builtLibrary,
      ),
    );
    output.dependencies.addAll(buildMode.dependencies);
    output.dependencies.add(input.packageRoot.resolve('pubspec.yaml'));
  });
}

String _canonicalLibName(OS os) => os == OS.windows
    ? 'libpocketfft.dll'
    : ((os == OS.macOS || os == OS.iOS)
          ? 'libpocketfft.dylib'
          : 'libpocketfft.so');

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
        'Prebuilt pocketfft binary for release $version is out of date with native sources in hook/!\n'
        'Pinned nativeSourceHash: $nativeSourceHash\n'
        'Current hook/ hash:      $currentSourceHash\n'
        'If you are the package author, build and attest new release artifacts (.github/workflows/artifacts.yml) and run:\n'
        '  dart tool/regenerate_hashes.dart <new-release-tag>\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` while developing native code.')}',
      );
    }

    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final artifactName = pocketfftArtifactName(os, arch);
    final expectedHash = fileHashes[(os, arch)];

    if (expectedHash == null || expectedHash.startsWith('00000000')) {
      throw StateError(
        'No prebuilt pocketfft binary hash is pinned for ($os, $arch) in release $version.\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` or `buildMode: local`.')}',
      );
    }

    final libName = _canonicalLibName(os);
    final sharedCacheDir = Directory.fromUri(
      input.outputDirectoryShared.resolve(
        'pocketfft-$version/${os.name}-${arch.name}/',
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
          print('Using cached pocketfft binary from ${cachedLibrary.path}.');
          return cachedLibrary.uri;
        }
      }

      final remoteUri = Uri.parse(
        'https://github.com/$repository/releases/download/$version/$artifactName',
      );
      print('Fetching prebuilt pocketfft binary from $remoteUri...');
      final bytes = await _downloadBytesWithRedirects(remoteUri);
      final actualHash = sha256.convert(bytes).toString();
      if (actualHash != expectedHash) {
        throw StateError(
          'SHA-256 mismatch for prebuilt pocketfft binary at $remoteUri:\n'
          'Expected: $expectedHash\n'
          'Actual:   $actualHash',
        );
      }
      verifyArtifactSourceHash(bytes, currentSourceHash: currentSourceHash);

      await cachedLibrary.parent.create(recursive: true);
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
      throw ArgumentError(
        '`localPath` is not set in `hooks.user_defines.pocketfft` '
        '(or `LOCAL_POCKETFFT_BINARY` environment variable).',
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
          pocketfftArtifactName(os, input.config.code.targetArchitecture),
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
        'Could not find local pocketfft binary.',
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

  String _resolveCxxCompiler(String cCompilerPath, OS os, bool isMSVC) {
    if (isMSVC) return cCompilerPath;
    if (cCompilerPath == 'cc') return 'c++';
    if (cCompilerPath.endsWith('/clang') || cCompilerPath.endsWith(r'\clang')) {
      final candidate = '$cCompilerPath++';
      if (File(candidate).existsSync()) return candidate;
    } else if (cCompilerPath == 'clang') {
      return 'clang++';
    } else if (cCompilerPath.endsWith('/gcc') ||
        cCompilerPath.endsWith(r'\gcc')) {
      final candidate =
          '${cCompilerPath.substring(0, cCompilerPath.length - 3)}g++';
      if (File(candidate).existsSync()) return candidate;
    } else if (cCompilerPath == 'gcc') {
      return 'g++';
    } else if (cCompilerPath.endsWith('/cc') ||
        cCompilerPath.endsWith(r'\cc')) {
      final candidate =
          '${cCompilerPath.substring(0, cCompilerPath.length - 2)}c++';
      if (File(candidate).existsSync()) return candidate;
    }
    return cCompilerPath;
  }

  @override
  Future<Uri> build() async {
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final cCompiler = input.config.code.cCompiler;

    final hookDir = Directory.fromUri(_root.resolve('hook/'));
    final wrapperFile = File.fromUri(
      hookDir.uri.resolve('pocketfft_wrapper.cpp'),
    );
    if (!wrapperFile.existsSync()) {
      throw FileSystemException(
        'Missing PocketFFT C++ wrapper source in hook/.',
        wrapperFile.path,
      );
    }

    final libName = _canonicalLibName(os);
    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));
    final tempLibFile = File(
      '${libFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );

    final currentSourceHash = computeNativeSourceHash(_root);
    final stampFile = File.fromUri(
      outputDir.uri.resolve('source_hash_stamp.cpp'),
    );
    await stampFile.writeAsString('''
#if defined(_WIN32)
#define STAMP_EXPORT extern "C" __declspec(dllexport)
#define STAMP_USED
#else
#define STAMP_EXPORT extern "C" __attribute__((visibility("default"), used))
#define STAMP_USED __attribute__((used))
#endif

STAMP_USED static const char _pocketfft_source_hash_marker[] =
    "$sourceHashMarkerPrefix$currentSourceHash";

STAMP_EXPORT const char* pocketfft_embedded_source_hash(void) {
  return _pocketfft_source_hash_marker;
}
''');

    final rawCompilerPath =
        cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'c++');
    final compilerLower = rawCompilerPath.toLowerCase();
    final isClangCl = compilerLower.contains('clang-cl');
    final isGNU =
        !isClangCl &&
        (compilerLower.contains('gcc') ||
            compilerLower.contains('clang') ||
            compilerLower.contains('g++') ||
            compilerLower.contains('c++'));
    final isMSVC =
        isClangCl ||
        (os == OS.windows &&
            (!isGNU ||
                compilerLower.endsWith('cl.exe') ||
                compilerLower == 'cl' ||
                compilerLower.contains('msvc')));
    final compilerPath = _resolveCxxCompiler(rawCompilerPath, os, isMSVC);

    final sanitize = buildOptions?.sanitize;
    final coverage = buildOptions?.coverage ?? false;
    if (isMSVC && (sanitize != null || coverage)) {
      throw UnsupportedError(
        'Native sanitizers and coverage are not supported with MSVC on Windows.',
      );
    }
    final sanitizeFlags = (sanitize != null && sanitize.isNotEmpty)
        ? <String>[
            '-fsanitize=$sanitize',
            if (sanitize.contains('undefined'))
              '-fno-sanitize=float-cast-overflow',
            '-fno-sanitize-recover=all',
            '-fno-omit-frame-pointer',
            '-g',
          ]
        : const <String>[];
    final coverageFlags = coverage
        ? const <String>['--coverage', '-O1', '-g']
        : const <String>[];

    final compileArgs = isMSVC
        ? <String>[
            '/LD',
            '/MD',
            '/O2',
            '/std:c++17',
            '/EHsc',
            '/DPOCKETFFT_NO_MULTITHREADING=1',
            '/DPOCKETFFT_CACHE_SIZE=64',
            '/I',
            hookDir.path,
            wrapperFile.path,
            stampFile.path,
            '/Fe:${tempLibFile.path}',
            '/link',
            '/EXPORT:kiss_fft_alloc',
            '/EXPORT:kiss_fft',
            '/EXPORT:kiss_fft_stride',
            '/EXPORT:kiss_fft_cleanup',
            '/EXPORT:kiss_fft_next_fast_size',
            '/EXPORT:kiss_fftr_alloc',
            '/EXPORT:kiss_fftr',
            '/EXPORT:kiss_fftri',
            '/EXPORT:kiss_fftnd_alloc',
            '/EXPORT:kiss_fftnd',
            '/EXPORT:free',
            '/EXPORT:pocketfft_embedded_source_hash',
          ]
        : <String>[
            if (os == OS.macOS || os == OS.iOS) ...[
              '-arch',
              arch == Architecture.arm64 ? 'arm64' : 'x86_64',
              '-Wl,-install_name,@rpath/$libName',
              '-Wl,-headerpad_max_install_names',
            ],
            '-std=c++17',
            '-shared',
            '-fPIC',
            '-O3',
            ...sanitizeFlags,
            ...coverageFlags,
            if (os == OS.android) '-Wl,-z,max-page-size=16384',
            '-DPOCKETFFT_NO_MULTITHREADING=1',
            '-DPOCKETFFT_CACHE_SIZE=64',
            if (os != OS.windows) '-DPOCKETFFT_USE_POSIX_MEMALIGN=1',
            '-I',
            hookDir.path,
            wrapperFile.path,
            stampFile.path,
            '-o',
            tempLibFile.path,
            if (os != OS.windows) '-lm',
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

    final res = await Process.run(
      compilerPath,
      compileArgs,
      environment: runEnv,
    );
    if (res.exitCode != 0) {
      if (tempLibFile.existsSync()) {
        try {
          tempLibFile.deleteSync();
        } catch (_) {}
      }
      throw StateError(
        'PocketFFT native C++ compilation failed (exit ${res.exitCode}):\n'
        'stdout: ${res.stdout}\n'
        'stderr: ${res.stderr}',
      );
    }

    await tempLibFile.rename(libFile.path);
    return libFile.uri;
  }

  @override
  List<Uri> get dependencies => [
    for (final file in nativeSourceFiles(_root)) file.uri,
  ];
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
