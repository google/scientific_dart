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
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:openblas/src/hook_helpers/build_options.dart';
import 'package:openblas/src/hook_helpers/hashes.dart';

const int _minAndroidNdkApi = 24;
const int _minIosVersion = 13;

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

    final requirePrebuilt =
        Platform.environment['OPENBLAS_REQUIRE_PREBUILT'] == '1';

    ({Uri openblasUri, Uri extensionsUri}) builtLibraries;
    if (buildOptions.buildMode == BuildModeEnum.fetch &&
        !buildOptions.isExplicit &&
        !requirePrebuilt &&
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
            !buildOptions.isExplicit &&
            !requirePrebuilt) {
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
    final iosSdk = _targetIosSdk(input);

    final openblasArtifact = openblasArtifactName(
      os,
      arch,
      'openblas',
      iosSdk: iosSdk,
    );
    final extArtifact = openblasArtifactName(
      os,
      arch,
      'openblas_extensions',
      iosSdk: iosSdk,
    );
    final expectedOpenblasHash = fileHashes[(os, arch, iosSdk, 'openblas')];
    final expectedExtHash =
        fileHashes[(os, arch, iosSdk, 'openblas_extensions')];

    if (expectedOpenblasHash == null ||
        expectedOpenblasHash.startsWith('00000000') ||
        expectedExtHash == null ||
        expectedExtHash.startsWith('00000000')) {
      throw StateError(
        'No prebuilt openblas binary hashes are pinned for ($os, $arch${iosSdk != null ? ', $iosSdk' : ''}) in release $version.\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` or `buildMode: local`.')}',
      );
    }

    final sharedDir = input.outputDirectoryShared.resolve(
      'openblas-$version/${_targetTripleTag(os, arch, iosSdk)}/',
    );
    final sharedCacheDir = Directory.fromUri(sharedDir);
    final cachedOpenblas = File.fromUri(
      sharedDir.resolve(_canonicalOpenblasName(os)),
    );
    final cachedExt = File.fromUri(
      sharedDir.resolve(_canonicalExtensionsName(os)),
    );

    final (cachedOpenblasUri, cachedExtUri) = await _withSharedLock(
      sharedCacheDir,
      () async {
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

        return (openblasUri, extensionsUri);
      },
    );

    final dstOpenblas = File.fromUri(
      input.outputDirectory.resolve(_canonicalOpenblasName(os)),
    );
    final dstExt = File.fromUri(
      input.outputDirectory.resolve(_canonicalExtensionsName(os)),
    );
    await dstOpenblas.parent.create(recursive: true);
    final tmpOpenblas = File(
      '${dstOpenblas.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    await File.fromUri(cachedOpenblasUri).copy(tmpOpenblas.path);
    await tmpOpenblas.rename(dstOpenblas.path);
    final tmpExt = File(
      '${dstExt.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    await File.fromUri(cachedExtUri).copy(tmpExt.path);
    await tmpExt.rename(dstExt.path);

    return (openblasUri: dstOpenblas.uri, extensionsUri: dstExt.uri);
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
    final tempFile = File(
      '${cachedFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    await tempFile.writeAsBytes(bytes, flush: true);
    await tempFile.rename(cachedFile.path);
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
    final iosSdk = _targetIosSdk(input);
    final entityPath = localPath!.toFilePath(windows: Platform.isWindows);

    if (FileSystemEntity.isDirectorySync(entityPath)) {
      final dirUri = Directory(entityPath).uri;
      var openblasFile = File.fromUri(
        dirUri.resolve(_canonicalOpenblasName(os)),
      );
      if (!openblasFile.existsSync()) {
        openblasFile = File.fromUri(
          dirUri.resolve(
            openblasArtifactName(os, arch, 'openblas', iosSdk: iosSdk),
          ),
        );
      }
      var extFile = File.fromUri(dirUri.resolve(_canonicalExtensionsName(os)));
      if (!extFile.existsSync()) {
        extFile = File.fromUri(
          dirUri.resolve(
            openblasArtifactName(
              os,
              arch,
              'openblas_extensions',
              iosSdk: iosSdk,
            ),
          ),
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
    final tmpOpenblas = File(
      '${dstOpenblas.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    await srcOpenblas.copy(tmpOpenblas.path);
    await tmpOpenblas.rename(dstOpenblas.path);
    final tmpExt = File(
      '${dstExt.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
    );
    await srcExt.copy(tmpExt.path);
    await tmpExt.rename(dstExt.path);
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

    final sanitize = buildOptions?.sanitize;
    final coverage = buildOptions?.coverage ?? false;
    final sanitizeFlags = (sanitize != null && sanitize.isNotEmpty)
        ? <String>[
            '-fsanitize=$sanitize',
            '-fno-sanitize-recover=all',
            '-fno-omit-frame-pointer',
            '-g',
          ]
        : const <String>[];
    final coverageFlags = coverage
        ? const <String>['--coverage', '-O1', '-g']
        : const <String>[];

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
        final iosSdk = _targetIosSdk(input);
        final int? iosVersion = os == OS.iOS
            ? math.max(input.config.code.iOS.targetVersion, _minIosVersion)
            : null;
        final String? iosSdkPath = os == OS.iOS
            ? await _resolveAppleSdkPath(iosSdk!.type)
            : null;
        final appleTargetFlags = <String>[
          if (os == OS.macOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
          ],
          if (os == OS.iOS) ...[
            '--target=${_iosClangTarget(arch, iosSdk!, iosVersion!)}',
            '-isysroot',
            iosSdkPath!,
          ],
        ];
        final stubFile = File.fromUri(
          outputDir.uri.resolve('accelerate_stub.c'),
        );
        await stubFile.writeAsString('''
#include <TargetConditionals.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define ACCEL_EXPORT __attribute__((visibility("default"), used))

static int _accelerate_num_threads = 1;
ACCEL_EXPORT int openblas_get_num_threads(void) { return _accelerate_num_threads; }
ACCEL_EXPORT void openblas_set_num_threads(int num_threads) {
  if (num_threads > 0) _accelerate_num_threads = num_threads;
}
ACCEL_EXPORT const char* openblas_get_config(void) {
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
  return "iOS Accelerate Framework";
#else
  return "MacOS Accelerate Framework";
#endif
}

typedef float (*_fn_cblas_sdot_t)(int, const float*, int, const float*, int);
typedef double (*_fn_cblas_ddot_t)(int, const double*, int, const double*, int);
typedef void (*_fn_cblas_saxpy_t)(int, float, const float*, int, float*, int);
typedef void (*_fn_cblas_daxpy_t)(int, double, const double*, int, double*, int);
typedef void (*_fn_cblas_dcopy_t)(int, const double*, int, double*, int);
typedef void (*_fn_cblas_scopy_t)(int, const float*, int, float*, int);
typedef void (*_fn_cblas_dscal_t)(int, double, double*, int);
typedef void (*_fn_cblas_sscal_t)(int, float, float*, int);
typedef double (*_fn_cblas_dnrm2_t)(int, const double*, int);
typedef float (*_fn_cblas_snrm2_t)(int, const float*, int);
typedef void (*_fn_cblas_dgemm_t)(int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
typedef void (*_fn_cblas_sgemm_t)(int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
typedef void (*_fn_cblas_cgemm_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
typedef void (*_fn_cblas_zgemm_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
typedef void (*_fn_cblas_dgemv_t)(int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
typedef void (*_fn_cblas_sgemv_t)(int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
typedef void (*_fn_cblas_cgemv_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
typedef void (*_fn_cblas_zgemv_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);

static _fn_cblas_sdot_t _p_cblas_sdot = NULL;
static _fn_cblas_ddot_t _p_cblas_ddot = NULL;
static _fn_cblas_saxpy_t _p_cblas_saxpy = NULL;
static _fn_cblas_daxpy_t _p_cblas_daxpy = NULL;
static _fn_cblas_dcopy_t _p_cblas_dcopy = NULL;
static _fn_cblas_scopy_t _p_cblas_scopy = NULL;
static _fn_cblas_dscal_t _p_cblas_dscal = NULL;
static _fn_cblas_sscal_t _p_cblas_sscal = NULL;
static _fn_cblas_dnrm2_t _p_cblas_dnrm2 = NULL;
static _fn_cblas_snrm2_t _p_cblas_snrm2 = NULL;
static _fn_cblas_dgemm_t _p_cblas_dgemm = NULL;
static _fn_cblas_sgemm_t _p_cblas_sgemm = NULL;
static _fn_cblas_cgemm_t _p_cblas_cgemm = NULL;
static _fn_cblas_zgemm_t _p_cblas_zgemm = NULL;
static _fn_cblas_dgemv_t _p_cblas_dgemv = NULL;
static _fn_cblas_sgemv_t _p_cblas_sgemv = NULL;
static _fn_cblas_cgemv_t _p_cblas_cgemv = NULL;
static _fn_cblas_zgemv_t _p_cblas_zgemv = NULL;

static pthread_once_t _accel_cblas_once = PTHREAD_ONCE_INIT;

static void* _accel_sym_or_abort(void* handle, const char* name) {
  void* sym = dlsym(handle, name);
  if (sym == NULL) {
    abort();
  }
  return sym;
}

static void _init_accel_cblas(void) {
  static const char* const candidates[] = {
    "/System/Library/Frameworks/Accelerate.framework/Accelerate",
    "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/libBLAS.dylib",
    "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib",
  };
  void* handle = NULL;
  for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
    void* h = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
    if (h != NULL && dlsym(h, "cblas_dgemm") != NULL) {
      handle = h;
      break;
    }
  }
  if (handle == NULL) {
    void* next_sym = dlsym(RTLD_NEXT, "cblas_dgemm");
    if (next_sym != NULL) {
      handle = RTLD_NEXT;
    }
  }
  if (handle == NULL) {
    abort();
  }
#define LOAD_CBLAS_SYM(name) do { void* _sym = _accel_sym_or_abort(handle, #name); memcpy(&_p_##name, &_sym, sizeof(_p_##name)); } while (0)
  LOAD_CBLAS_SYM(cblas_sdot);
  LOAD_CBLAS_SYM(cblas_ddot);
  LOAD_CBLAS_SYM(cblas_saxpy);
  LOAD_CBLAS_SYM(cblas_daxpy);
  LOAD_CBLAS_SYM(cblas_dcopy);
  LOAD_CBLAS_SYM(cblas_scopy);
  LOAD_CBLAS_SYM(cblas_dscal);
  LOAD_CBLAS_SYM(cblas_sscal);
  LOAD_CBLAS_SYM(cblas_dnrm2);
  LOAD_CBLAS_SYM(cblas_snrm2);
  LOAD_CBLAS_SYM(cblas_dgemm);
  LOAD_CBLAS_SYM(cblas_sgemm);
  LOAD_CBLAS_SYM(cblas_cgemm);
  LOAD_CBLAS_SYM(cblas_zgemm);
  LOAD_CBLAS_SYM(cblas_dgemv);
  LOAD_CBLAS_SYM(cblas_sgemv);
  LOAD_CBLAS_SYM(cblas_cgemv);
  LOAD_CBLAS_SYM(cblas_zgemv);
#undef LOAD_CBLAS_SYM
}

static inline void _ensure_accel_cblas(void) {
  pthread_once(&_accel_cblas_once, _init_accel_cblas);
}

ACCEL_EXPORT float cblas_sdot(int n, const float* x, int incx, const float* y, int incy) {
  _ensure_accel_cblas();
  return _p_cblas_sdot(n, x, incx, y, incy);
}
ACCEL_EXPORT double cblas_ddot(int n, const double* x, int incx, const double* y, int incy) {
  _ensure_accel_cblas();
  return _p_cblas_ddot(n, x, incx, y, incy);
}
ACCEL_EXPORT void cblas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_saxpy(n, alpha, x, incx, y, incy);
}
ACCEL_EXPORT void cblas_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_daxpy(n, alpha, x, incx, y, incy);
}
ACCEL_EXPORT void cblas_dcopy(int n, const double* x, int incx, double* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_dcopy(n, x, incx, y, incy);
}
ACCEL_EXPORT void cblas_scopy(int n, const float* x, int incx, float* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_scopy(n, x, incx, y, incy);
}
ACCEL_EXPORT void cblas_dscal(int n, double alpha, double* x, int incx) {
  _ensure_accel_cblas();
  _p_cblas_dscal(n, alpha, x, incx);
}
ACCEL_EXPORT void cblas_sscal(int n, float alpha, float* x, int incx) {
  _ensure_accel_cblas();
  _p_cblas_sscal(n, alpha, x, incx);
}
ACCEL_EXPORT double cblas_dnrm2(int n, const double* x, int incx) {
  _ensure_accel_cblas();
  return _p_cblas_dnrm2(n, x, incx);
}
ACCEL_EXPORT float cblas_snrm2(int n, const float* x, int incx) {
  _ensure_accel_cblas();
  return _p_cblas_snrm2(n, x, incx);
}
ACCEL_EXPORT void cblas_dgemm(int order, int transA, int transB, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) {
  _ensure_accel_cblas();
  _p_cblas_dgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
ACCEL_EXPORT void cblas_sgemm(int order, int transA, int transB, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc) {
  _ensure_accel_cblas();
  _p_cblas_sgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
ACCEL_EXPORT void cblas_cgemm(int order, int transA, int transB, int m, int n, int k, const void* alpha, const void* a, int lda, const void* b, int ldb, const void* beta, void* c, int ldc) {
  _ensure_accel_cblas();
  _p_cblas_cgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
ACCEL_EXPORT void cblas_zgemm(int order, int transA, int transB, int m, int n, int k, const void* alpha, const void* a, int lda, const void* b, int ldb, const void* beta, void* c, int ldc) {
  _ensure_accel_cblas();
  _p_cblas_zgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
ACCEL_EXPORT void cblas_dgemv(int order, int transA, int m, int n, double alpha, const double* a, int lda, const double* x, int incx, double beta, double* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_dgemv(order, transA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
ACCEL_EXPORT void cblas_sgemv(int order, int transA, int m, int n, float alpha, const float* a, int lda, const float* x, int incx, float beta, float* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_sgemv(order, transA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
ACCEL_EXPORT void cblas_cgemv(int order, int transA, int m, int n, const void* alpha, const void* a, int lda, const void* x, int incx, const void* beta, void* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_cgemv(order, transA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
ACCEL_EXPORT void cblas_zgemv(int order, int transA, int m, int n, const void* alpha, const void* a, int lda, const void* x, int incx, const void* beta, void* y, int incy) {
  _ensure_accel_cblas();
  _p_cblas_zgemv(order, transA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
''');

        final libFile = File.fromUri(
          outputDir.uri.resolve('libopenblas.dylib'),
        );
        final tempLibFile = File(
          '${libFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
        );
        final extLibFile = File.fromUri(
          outputDir.uri.resolve('libopenblas_extensions.dylib'),
        );
        final tempExtLibFile = File(
          '${extLibFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
        );

        final stubCompileArgs = [
          ...appleTargetFlags,
          '-Wl,-install_name,@rpath/libopenblas.dylib',
          '-Wl,-headerpad_max_install_names',
          if (os == OS.iOS) '-Wl,-encryptable',
          '-dynamiclib',
          '-O3',
          ...sanitizeFlags,
          ...coverageFlags,
          stubFile.path,
          customExtensionsPath,
          stampFile.path,
          '-o',
          tempLibFile.path,
          '-framework',
          'Accelerate',
        ];
        final stubRes = await Process.run(compilerPath, stubCompileArgs);
        if (stubRes.exitCode != 0) {
          throw StateError(
            'Failed to compile Accelerate stub (exit ${stubRes.exitCode}):\n'
            'stdout: ${stubRes.stdout}\n'
            'stderr: ${stubRes.stderr}',
          );
        }
        await tempLibFile.rename(libFile.path);

        final extCompileArgs = [
          ...appleTargetFlags,
          '-Wl,-install_name,@rpath/libopenblas_extensions.dylib',
          '-Wl,-headerpad_max_install_names',
          if (os == OS.iOS) '-Wl,-encryptable',
          '-dynamiclib',
          '-O3',
          ...sanitizeFlags,
          ...coverageFlags,
          customExtensionsPath,
          stampFile.path,
          '-o',
          tempExtLibFile.path,
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
        await tempExtLibFile.rename(extLibFile.path);

        return (openblasUri: libFile.uri, extensionsUri: extLibFile.uri);

      case PrecompiledBinary():
        final compilerPath =
            cCompiler?.compiler.toFilePath() ??
            (os == OS.windows ? 'cl' : 'cc');
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

        final zipUrl = Uri.parse(
          'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/OpenBLAS-0.3.33-x64.zip',
        );
        final extractDir = input.outputDirectoryShared.resolve(
          'OpenBLAS-precompiled-0.3.33-x64/',
        );
        final extractDirFile = Directory.fromUri(extractDir);
        final completeMarker = File.fromUri(extractDir.resolve('.complete'));

        await _withSharedLock(extractDirFile, () async {
          if (!completeMarker.existsSync()) {
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
            final extractDirPath = extractDirFile.path;
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
                outFile.writeAsBytesSync(
                  file.content as List<int>,
                  flush: true,
                );
              } else {
                Directory(outPath).createSync(recursive: true);
              }
            }

            final lapackHeader = File(
              extractDir.resolve('include/lapack.h').toFilePath(),
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
            await completeMarker.writeAsString('ok', flush: true);
          }
        });

        final dllFile = File.fromUri(extractDir.resolve('bin/libopenblas.dll'));
        final outDllFile = File.fromUri(
          outputDir.uri.resolve(_canonicalOpenblasName(os)),
        );
        final tempOutDll = File(
          '${outDllFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
        );
        await dllFile.copy(tempOutDll.path);
        await tempOutDll.rename(outDllFile.path);

        final headersDir = extractDir.resolve('include/');
        final libDir = extractDir.resolve('lib/');

        final openblasLibName = isMSVC
            ? 'libopenblas.lib'
            : 'libopenblas.dll.a';
        final openblasLibFile = File.fromUri(libDir.resolve(openblasLibName));

        final extLibFile = File(
          outputDir.uri.resolve('libopenblas_extensions.dll').toFilePath(),
        );
        final tempExtLibFile = File(
          '${extLibFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
        );
        final compileArgs = isMSVC
            ? [
                '/LD',
                '/O2',
                '/EHsc',
                '/I${headersDir.toFilePath()}',
                customExtensionsPath,
                stampFile.path,
                '/Fe:${tempExtLibFile.path}',
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
                ...sanitizeFlags,
                ...coverageFlags,
                '-I${headersDir.toFilePath()}',
                customExtensionsPath,
                stampFile.path,
                '-o',
                tempExtLibFile.path,
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
        await tempExtLibFile.rename(extLibFile.path);

        return (openblasUri: outDllFile.uri, extensionsUri: extLibFile.uri);

      case CompileOpenBlas(:final sourceUrl):
        final isAndroid = os == OS.android;
        final int? androidNdkApi = isAndroid
            ? math.max(
                input.config.code.android.targetNdkApi,
                _minAndroidNdkApi,
              )
            : null;
        final androidNdkTools = isAndroid
            ? _detectAndroidNdkTools(cCompiler)
            : null;
        if (isAndroid && androidNdkTools == null) {
          throw StateError(
            'Android NDK toolchain not found for building OpenBLAS. '
            'Pass `cCompiler` or set ANDROID_NDK_LATEST_HOME / ANDROID_NDK_ROOT.',
          );
        }
        final androidSysroot = isAndroid
            ? _androidSysrootFromCompiler(androidNdkTools!.clang)
            : null;

        String openBlasTarget = isAndroid && arch == Architecture.x64
            ? 'NEHALEM'
            : 'GENERIC';
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
        final buildTargetTag = isAndroid
            ? 'android-${arch.name}-api$androidNdkApi'
            : '${os.name}-${arch.name}';
        final sharedOpenblasBase = (!isAndroid && legacyExtractDir.existsSync())
            ? outputDir
            : Directory.fromUri(
                input.outputDirectoryShared.resolve(
                  'openblas-src-$buildTargetTag/',
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

        await _withSharedLock(sharedOpenblasBase, () async {
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
                outFile.writeAsBytesSync(
                  file.content as List<int>,
                  flush: true,
                );
              } else {
                Directory(outPath).createSync(recursive: true);
              }
            }

            print('Building OpenBLAS with target $openBlasTarget...');
            final List<String> makeArgs;
            if (isAndroid) {
              final tripleApi = '${_androidTargetTriple(arch)}$androidNdkApi';
              final wrapperSuffix = Platform.isWindows ? '.cmd' : '';
              final wrapperFile = File.fromUri(
                androidNdkTools!.clang.resolve(
                  '$tripleApi-clang$wrapperSuffix',
                ),
              );
              final useWrapper = wrapperFile.existsSync();
              // Keep `CC` as a single executable path without `--sysroot=...`
              // slashes so OpenBLAS's `c_check` (`dirname "$compiler_name"`)
              // does not corrupt `CROSS_SUFFIX`.
              final androidCc = useWrapper
                  ? wrapperFile.path
                  : androidNdkTools.clang.toFilePath();
              final androidCflags = <String>[
                if (!useWrapper) ...[
                  '--target=$tripleApi',
                  if (androidSysroot != null)
                    '--sysroot=${androidSysroot.toFilePath()}',
                ],
                '-Wno-implicit-function-declaration',
                '-Wno-incompatible-pointer-types',
              ].join(' ');
              final lldExt = Platform.isWindows ? '.exe' : '';
              final lldFile = File.fromUri(
                androidNdkTools.clang.resolve('ld.lld$lldExt'),
              );
              makeArgs = <String>[
                'shared',
                '-j${Platform.numberOfProcessors}',
                'OSNAME=Android',
                'TARGET=$openBlasTarget',
                'BINARY=64',
                'NOFORTRAN=1',
                'C_LAPACK=1',
                'NO_LAPACK=0',
                'NO_LAPACKE=0',
                'USE_THREAD=1',
                'NUM_THREADS=16',
                'FIXED_LIBNAME=1',
                OS.current == OS.macOS ? 'HOSTCC=clang' : 'HOSTCC=gcc',
                'CC=$androidCc',
                'CFLAGS=$androidCflags',
                'AR=${androidNdkTools.ar.toFilePath()}',
                if (androidNdkTools.ranlib != null)
                  'RANLIB=${androidNdkTools.ranlib!.toFilePath()}',
                if (lldFile.existsSync()) 'LD=${lldFile.path}',
                'LDFLAGS=-Wl,-z,max-page-size=16384',
              ];
            } else {
              makeArgs = <String>[
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
                if (cCompiler != null) ...[
                  'CC=${cCompiler.compiler.toFilePath()}',
                  'AR=${cCompiler.archiver.toFilePath()}',
                ],
              ];
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
                'stdout: ${buildResult.stdout}\n'
                'stderr: ${buildResult.stderr}',
              );
            }
          }
        });

        final outOpenblasFile = File.fromUri(outputDir.uri.resolve(libName));
        final tempOutOpenblas = File(
          '${outOpenblasFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
        );
        await libFile.copy(tempOutOpenblas.path);
        await tempOutOpenblas.rename(outOpenblasFile.path);

        final extLibFile = File(
          outputDir.uri.resolve(_canonicalExtensionsName(os)).toFilePath(),
        );
        final tempExtLibFile = File(
          '${extLibFile.path}.tmp.$pid.${DateTime.now().microsecondsSinceEpoch}',
        );
        final compilerPath = isAndroid
            ? androidNdkTools!.clang.toFilePath()
            : (cCompiler?.compiler.toFilePath() ??
                  (os == OS.windows ? 'cl' : 'cc'));
        final compilerLower = compilerPath.toLowerCase();
        final isClangCl = compilerLower.contains('clang-cl');
        final isMSVC =
            os == OS.windows &&
            (isClangCl ||
                (!compilerLower.contains('clang') &&
                    (compilerLower.endsWith('cl.exe') ||
                        compilerLower == 'cl')));

        final compileArgs = isMSVC
            ? [
                '/LD',
                '/O2',
                '/EHsc',
                '/I${extractDir}lapack-netlib/LAPACKE/include',
                customExtensionsPath,
                stampFile.path,
                '/Fe:${tempExtLibFile.path}',
                '/link',
                '/LIBPATH:$extractDir',
                'libopenblas.lib',
                '/EXPORT:get_dgetrf_ptr',
                '/EXPORT:get_sgetrf_ptr',
                '/EXPORT:get_zgetrf_ptr',
                '/EXPORT:get_cgetrf_ptr',
                '/EXPORT:openblas_embedded_source_hash',
              ]
            : [
                if (isAndroid) ...[
                  '--target=${_androidTargetTriple(arch)}$androidNdkApi',
                  if (androidSysroot != null)
                    '--sysroot=${androidSysroot.toFilePath()}',
                ],
                '-shared',
                '-fPIC',
                '-O3',
                ...sanitizeFlags,
                ...coverageFlags,
                if (isAndroid) '-Wl,-z,max-page-size=16384',
                '-I${extractDir}lapack-netlib/LAPACKE/include',
                customExtensionsPath,
                stampFile.path,
                '-o',
                tempExtLibFile.path,
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
        await tempExtLibFile.rename(extLibFile.path);

        return (
          openblasUri: outOpenblasFile.uri,
          extensionsUri: extLibFile.uri,
        );
    }
  }

  @override
  List<Uri> get dependencies => [
    for (final file in nativeSourceFiles(_root)) file.uri,
  ];
}

IOSSdk? _targetIosSdk(BuildInput input) => input.config.code.targetOS == OS.iOS
    ? input.config.code.iOS.targetSdk
    : null;

String _targetTripleTag(OS os, Architecture arch, IOSSdk? iosSdk) =>
    iosSdk != null
    ? '${os.name}-${iosSdk.type}-${arch.name}'
    : '${os.name}-${arch.name}';

String _androidTargetTriple(Architecture arch) => switch (arch) {
  Architecture.arm64 => 'aarch64-linux-android',
  Architecture.x64 => 'x86_64-linux-android',
  Architecture.arm => 'armv7a-linux-androideabi',
  Architecture.ia32 => 'i686-linux-android',
  Architecture.riscv64 => 'riscv64-linux-android',
  _ => throw UnsupportedError('Unsupported Android architecture: $arch'),
};

String _iosClangTarget(Architecture arch, IOSSdk sdk, int version) =>
    switch ((arch, sdk)) {
      (Architecture.arm64, IOSSdk.iPhoneOS) => 'arm64-apple-ios$version.0',
      (Architecture.arm64, IOSSdk.iPhoneSimulator) =>
        'arm64-apple-ios$version.0-simulator',
      (Architecture.x64, IOSSdk.iPhoneSimulator) =>
        'x86_64-apple-ios$version.0-simulator',
      _ => throw UnsupportedError('Unsupported iOS target: ($arch, $sdk)'),
    };

Future<String> _resolveAppleSdkPath(String sdkName) async {
  final res = await Process.run('xcrun', ['--sdk', sdkName, '--show-sdk-path']);
  if (res.exitCode != 0) {
    throw StateError(
      'Failed to locate Apple SDK "$sdkName" via xcrun (exit ${res.exitCode}):\n'
      '${res.stderr}',
    );
  }
  final path = (res.stdout as String).trim();
  if (path.isEmpty) {
    throw StateError('xcrun returned empty SDK path for "$sdkName".');
  }
  return path;
}

Uri? _androidSysrootFromCompiler(Uri compilerUri) {
  final sysroot = compilerUri.resolve('../sysroot/');
  return Directory.fromUri(sysroot).existsSync() ? sysroot : null;
}

List<Directory> _androidNdkCandidateDirs() {
  final ndkCandidates = <Directory>[];
  for (final key in const [
    'ANDROID_NDK_LATEST_HOME',
    'ANDROID_NDK_ROOT',
    'ANDROID_NDK_HOME',
    'ANDROID_NDK',
  ]) {
    final val = Platform.environment[key];
    if (val != null && val.isNotEmpty) {
      final dir = Directory(val);
      if (dir.existsSync()) ndkCandidates.add(dir);
    }
  }
  for (final sdkKey in const ['ANDROID_HOME', 'ANDROID_SDK_ROOT']) {
    final sdkVal = Platform.environment[sdkKey];
    if (sdkVal == null || sdkVal.isEmpty) continue;
    final ndkParent = Directory('$sdkVal/ndk');
    if (ndkParent.existsSync()) {
      final versions = ndkParent.listSync().whereType<Directory>().toList()
        ..sort(_compareVersionDesc);
      ndkCandidates.addAll(versions);
    }
  }
  return ndkCandidates;
}

int _compareVersionDesc(Directory a, Directory b) {
  final aName = a.uri.pathSegments.where((s) => s.isNotEmpty).last;
  final bName = b.uri.pathSegments.where((s) => s.isNotEmpty).last;
  final aParts = aName.split('.').map(int.tryParse).toList();
  final bParts = bName.split('.').map(int.tryParse).toList();
  final len = math.max(aParts.length, bParts.length);
  for (var i = 0; i < len; i++) {
    final av = i < aParts.length ? aParts[i] : 0;
    final bv = i < bParts.length ? bParts[i] : 0;
    if (av == null || bv == null) return bName.compareTo(aName);
    if (av != bv) return bv.compareTo(av);
  }
  return bName.compareTo(aName);
}

({Uri clang, Uri ar, Uri? ranlib})? _detectAndroidNdkTools(
  CCompilerConfig? cCompiler,
) {
  final ext = Platform.isWindows ? '.exe' : '';
  if (cCompiler != null) {
    final ranlibCandidate = File.fromUri(
      cCompiler.compiler.resolve('llvm-ranlib$ext'),
    );
    return (
      clang: cCompiler.compiler,
      ar: cCompiler.archiver,
      ranlib: ranlibCandidate.existsSync() ? ranlibCandidate.uri : null,
    );
  }
  for (final ndkDir in _androidNdkCandidateDirs()) {
    final prebuilt = Directory.fromUri(
      ndkDir.uri.resolve('toolchains/llvm/prebuilt/'),
    );
    if (!prebuilt.existsSync()) continue;
    for (final hostDir in prebuilt.listSync().whereType<Directory>()) {
      final clang = File.fromUri(hostDir.uri.resolve('bin/clang$ext'));
      final ar = File.fromUri(hostDir.uri.resolve('bin/llvm-ar$ext'));
      final ranlib = File.fromUri(hostDir.uri.resolve('bin/llvm-ranlib$ext'));
      if (clang.existsSync() && ar.existsSync()) {
        return (
          clang: clang.uri,
          ar: ar.uri,
          ranlib: ranlib.existsSync() ? ranlib.uri : null,
        );
      }
    }
  }
  return null;
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
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 30);
  try {
    var currentUri = url;
    for (var redirectCount = 0; redirectCount < 5; redirectCount++) {
      final request = await client
          .getUrl(currentUri)
          .timeout(const Duration(seconds: 30));
      final response = await request.close().timeout(
        const Duration(seconds: 30),
      );
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.value(HttpHeaders.locationHeader) != null) {
        final location = response.headers.value(HttpHeaders.locationHeader)!;
        await response.drain<void>();
        final nextUri = currentUri.resolve(location);
        if (currentUri.scheme == 'https' && nextUri.scheme != 'https') {
          throw HttpException(
            'Refusing HTTPS-to-HTTP redirect downgrade: $nextUri',
          );
        }
        currentUri = nextUri;
        continue;
      }
      if (response.statusCode != 200) {
        await response.drain<void>();
        throw HttpException(
          'Failed to download $currentUri (HTTP ${response.statusCode})',
        );
      }
      final builder = BytesBuilder(copy: false);
      await for (final chunk in response.timeout(const Duration(minutes: 5))) {
        builder.add(chunk);
      }
      return builder.takeBytes();
    }
    throw HttpException('Too many redirects while downloading $url');
  } finally {
    client.close(force: true);
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
      '${tempDir.path}\\get_msvc_env_${DateTime.now().microsecondsSinceEpoch}_$pid.bat',
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
